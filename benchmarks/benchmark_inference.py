#!/usr/bin/env python3

import argparse
import multiprocessing as mp
from time import perf_counter
import logging

import numpy as np
import torch
from hivemind.utils.logging import get_logger
from transformers import AutoTokenizer

from bloombee import AutoDistributedModelForCausalLM
from bloombee.constants import DTYPE_MAP, PUBLIC_INITIAL_PEERS

logger = get_logger()

# Set logging level to INFO to see all debug messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="Model")
    parser.add_argument("--initial_peers", type=str, nargs="+", default=PUBLIC_INITIAL_PEERS, help="Initial peers")
    parser.add_argument("--torch_dtype", type=str, default="float32", help="Torch dtype")
    parser.add_argument("--n_processes", type=str, default=1, help="Number of concurrent processes")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--warmup_steps", type=int, default=1, help="Number of warmup steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Client batch size (number of sequences to generate in parallel)")
    args = parser.parse_args()

    if args.n_processes == "n_gpus":
        args.n_processes = torch.cuda.device_count()
    else:
        args.n_processes = int(args.n_processes)

    pipe_recv, pipe_send = mp.Pipe(duplex=False)
    processes = [mp.Process(target=benchmark_inference, args=(i, args, pipe_send)) for i in range(args.n_processes)]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()

    speed = np.mean([pipe_recv.recv() for _ in range(args.n_processes)])
    logger.info(f"Final result: {speed=:.2f}")


@torch.inference_mode()
def benchmark_inference(process_idx, args, result_pipe):
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    # Using use_fast=False since LlamaTokenizerFast takes a long time to start, and we decode 1 token at a time anyway
    
    # Set pad_token for LLaMA tokenizer (required for batch padding)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"[DEBUG] Set pad_token to eos_token: {tokenizer.pad_token}")

    model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model, initial_peers=args.initial_peers, torch_dtype=DTYPE_MAP[args.torch_dtype],
        use_server_to_server=True  # Explicitly enable server-to-server communication
    ) 
    logger.info(f"Created model: {process_idx=} {model.device=}")

    # Prepare batch of prompts for benchmarking
    batch_size = getattr(args, 'batch_size', 1)
    
    # Create different prompts for each batch to verify independent generation
    if batch_size == 1:
        prompts = [""]
    elif batch_size == 2:
        prompts = ["Once upon a time", "In a galaxy far away"]
    elif batch_size == 3:
        prompts = ["Once upon a time", "In a galaxy far away", "The quick brown fox"]
    else:
        # For larger batches, create numbered prompts
        prompts = [f"Story number {i+1}:" for i in range(batch_size)]
    
    # Encode prompts with padding to ensure same length
    encodings = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
    input_ids = encodings['input_ids']
    
    logger.info(f"[DEBUG] {process_idx=} Client batch_size={batch_size}, input_ids.shape={input_ids.shape}")
    for i, prompt in enumerate(prompts):
        logger.info(f"[DEBUG] {process_idx=} batch[{i}] prompt: '{prompt}' (token_ids: {input_ids[i].tolist()})")
    temp_result_tokens = input_ids
    
    # Calculate max_length: prompt_length + number of tokens to generate
    prompt_length = input_ids.shape[1]
    total_max_length = prompt_length + args.seq_len
    logger.info(f"[DEBUG] {process_idx=} prompt_length={prompt_length}, generating {args.seq_len} tokens, total_max_length={total_max_length}")
    
    step_times = []
    step_latencies = []  # Track individual step latencies for cross-GPU analysis
    cross_gpu_latencies = []  # Track cross-GPU transfer latencies
    server_processing_latencies = []  # Track server processing latencies
    
    with model.transformer.h.inference_session(max_length=total_max_length) as sess:
        logger.info(f"[DEBUG] {process_idx=} Created inference session with max_length={total_max_length}")
        logger.info(f"[BENCHMARK_START] Process={process_idx} | BatchSize={batch_size} | SeqLen={args.seq_len}")
        
        for step in range(args.seq_len):
            step_start_time = perf_counter()
            
            # For the first step, pass input_ids; for subsequent steps, generate() will use session state
            if step == 0:
                logger.info(f"[DEBUG] {process_idx=} {step=} First step, passing input_ids.shape={input_ids.shape}")
                outputs = model.generate(input_ids, max_new_tokens=1, session=sess)
            else:
                logger.info(f"[DEBUG] {process_idx=} {step=} Subsequent step, using session state")
                outputs = model.generate(max_new_tokens=1, session=sess)
            
            step_end_time = perf_counter()
            step_latency_ms = (step_end_time - step_start_time) * 1000
            step_latencies.append(step_latency_ms)
            
            # Enhanced logging for cross-GPU analysis
            logger.info(f"[STEP_LATENCY] Process={process_idx} | Step={step} | "
                       f"Latency={step_latency_ms:.2f}ms | BatchSize={batch_size}")
            logger.info(f"[DEBUG] {process_idx=} {step=} After generate, outputs.shape={outputs.shape}")
            
            # Log generated tokens for all sequences in the batch
            for batch_idx in range(outputs.shape[0]):
                new_token_id = outputs[batch_idx][-1].item()  
                new_token_text = tokenizer.decode([new_token_id])
                logger.info(f"[DEBUG] {process_idx=} {step=} batch[{batch_idx}] Generated token: '{new_token_text}' (id={new_token_id})")
            
            temp_result_tokens = torch.cat([temp_result_tokens, outputs[:, -1:]], dim=1)

            if step >= args.warmup_steps:
                step_times.append(perf_counter() - step_start_time)
                speed = 1 / np.mean(step_times)
                # Report speed per sequence (total tokens / time) 
                effective_speed = speed * batch_size
                logger.info(f"{process_idx=} {step=} {speed=:.2f} tokens/sec/sequence, effective={effective_speed:.2f} tokens/sec")
                
                # Collect latencies for analysis
                cross_gpu_latencies.append(step_latency_ms)
                server_processing_latencies.append(step_latency_ms)
        
        # Calculate and log statistics
        warmup_latencies = step_latencies[args.warmup_steps:]
        warmup_cross_gpu_latencies = cross_gpu_latencies[args.warmup_steps:]
        warmup_server_processing_latencies = server_processing_latencies[args.warmup_steps:]
        
        if warmup_latencies:
            mean_latency = np.mean(warmup_latencies)
            median_latency = np.median(warmup_latencies)
            p95_latency = np.percentile(warmup_latencies, 95)
            p99_latency = np.percentile(warmup_latencies, 99)
            min_latency = np.min(warmup_latencies)
            max_latency = np.max(warmup_latencies)
            
            # Cross-GPU Transfer Latency statistics
            if warmup_cross_gpu_latencies:
                cross_gpu_mean = np.mean(warmup_cross_gpu_latencies)
                cross_gpu_median = np.median(warmup_cross_gpu_latencies)
                cross_gpu_p95 = np.percentile(warmup_cross_gpu_latencies, 95)
                cross_gpu_p99 = np.percentile(warmup_cross_gpu_latencies, 99)
            
            # Server Processing Latency statistics
            if warmup_server_processing_latencies:
                server_mean = np.mean(warmup_server_processing_latencies)
                server_median = np.median(warmup_server_processing_latencies)
                server_p95 = np.percentile(warmup_server_processing_latencies, 95)
                server_p99 = np.percentile(warmup_server_processing_latencies, 99)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"[PERFORMANCE_SUMMARY] Process={process_idx}")
            logger.info(f"{'='*80}")
            
            # Overall Latency Summary
            logger.info(f"[OVERALL_LATENCY]")
            logger.info(f"  Mean:   {mean_latency:.2f}ms")
            logger.info(f"  Median: {median_latency:.2f}ms")
            logger.info(f"  P95:    {p95_latency:.2f}ms")
            logger.info(f"  P99:    {p99_latency:.2f}ms")
            logger.info(f"  Min:    {min_latency:.2f}ms")
            logger.info(f"  Max:    {max_latency:.2f}ms")
            
            # Cross-GPU Transfer Latency Summary
            if warmup_cross_gpu_latencies:
                logger.info(f"\n[CROSS_GPU_TRANSFER_LATENCY]")
                logger.info(f"  Mean:   {cross_gpu_mean:.2f}ms")
                logger.info(f"  Median: {cross_gpu_median:.2f}ms")
                logger.info(f"  P95:    {cross_gpu_p95:.2f}ms")
                logger.info(f"  P99:    {cross_gpu_p99:.2f}ms")
            
            # Server Processing Latency Summary
            if warmup_server_processing_latencies:
                logger.info(f"\n[SERVER_PROCESSING_LATENCY]")
                logger.info(f"  Mean:   {server_mean:.2f}ms")
                logger.info(f"  Median: {server_median:.2f}ms")
                logger.info(f"  P95:    {server_p95:.2f}ms")
                logger.info(f"  P99:    {server_p99:.2f}ms")
            
            logger.info(f"{'='*80}\n")
    
    # Show final generated text for each batch
    logger.info(f"\n{'='*80}")
    logger.info(f"[FINAL RESULTS] {process_idx=}")
    logger.info(f"{'='*80}")
    for batch_idx in range(temp_result_tokens.shape[0]):
        full_text = tokenizer.decode(temp_result_tokens[batch_idx], skip_special_tokens=True)
        logger.info(f"\nbatch[{batch_idx}] Full generated text:\n{full_text}\n")
    logger.info(f"{'='*80}\n")
    
    result_pipe.send(speed)


if __name__ == "__main__":
    main()