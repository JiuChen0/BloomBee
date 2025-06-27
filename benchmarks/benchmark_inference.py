#!/usr/bin/env python3

import argparse
import multiprocessing as mp
from time import perf_counter

import numpy as np
import torch
from hivemind.utils.logging import get_logger
from transformers import AutoTokenizer

from bloombee import AutoDistributedModelForCausalLM
from bloombee.constants import DTYPE_MAP, PUBLIC_INITIAL_PEERS

logger = get_logger()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="Model")
    parser.add_argument("--initial_peers", type=str, nargs="+", default=PUBLIC_INITIAL_PEERS, help="Initial peers")
    parser.add_argument("--torch_dtype", type=str, default="float32", help="Torch dtype")
    parser.add_argument("--n_processes", type=str, default=1, help="Number of concurrent processes")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--warmup_steps", type=int, default=1, help="Number of warmup steps")
    parser.add_argument("--prompt_len", type=int, default=1, help="Length of initial prompt")
    
    # 采样参数选项
    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling (higher = more random)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty (>1.0 to avoid repetition)")
    
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

    model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model, initial_peers=args.initial_peers, torch_dtype=DTYPE_MAP[args.torch_dtype]
    )
    logger.info(f"Created model: {process_idx=} {model.device=}")

    # 构建生成参数
    generate_kwargs = {"max_new_tokens": 1}
    
    if args.do_sample:
        generate_kwargs.update({
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty
        })
        logger.info(f"Using sampling: temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, repetition_penalty={args.repetition_penalty}")
    else:
        logger.info("Using greedy decoding (deterministic)")

    #  Create real multi-token input to test prompt_len > 1
    if args.prompt_len > 1:
        # Create an initial prompt containing multiple tokens
        test_prompt = "Simply put, the theory of relativity states that"
        input_ids = tokenizer.encode(test_prompt, return_tensors="pt", add_special_tokens=False)
        # Ensure we have enough tokens, repeat if not enough
        while input_ids.shape[1] < args.prompt_len:
            input_ids = torch.cat([input_ids, input_ids], dim=1)
        # Truncate to specified length
        input_ids = input_ids[:, :args.prompt_len]
        logger.info(f"Using initial prompt with {args.prompt_len} tokens: {input_ids.shape}")
        
        # First process the initial multi-token input
        with model.transformer.h.inference_session(max_length=args.seq_len) as sess:
            start_time = perf_counter()
            
            #  Key: Use multi-token input for the first inference
            outputs = model.generate(input_ids, session=sess, **generate_kwargs)
            result = tokenizer.decode(outputs[0])
            
            step_times = [perf_counter() - start_time]
            logger.info(f"Initial {args.prompt_len}-token input processed successfully!")
            
            # Track generated length for correct decoding
            previous_length = len(outputs[0])
            
            # Continue generating remaining tokens
            for step in range(1, args.seq_len - args.prompt_len):
                start_time = perf_counter()
                outputs = model.generate(session=sess, **generate_kwargs)
                
                # Debug: print outputs info
                logger.info(f"Step {step}: outputs[0].shape = {outputs[0].shape}, previous_length = {previous_length}")
                
                # Only decode the new token(s)
                if len(outputs[0]) > previous_length:
                    new_token_ids = outputs[0][previous_length:]
                    new_token = tokenizer.decode(new_token_ids, skip_special_tokens=True)
                    previous_length = len(outputs[0])
                else:
                    # Fallback: decode last token only
                    new_token = tokenizer.decode(outputs[0][-1:], skip_special_tokens=True)
                
                result += new_token

                if step >= args.warmup_steps:
                    step_times.append(perf_counter() - start_time)
                    speed = 1 / np.mean(step_times)
                    logger.info(f"{process_idx=} {step=} {speed=:.2f} | New token: {repr(new_token)} | Current result: {repr(result)}")
    else:
        # Original single-token logic
        result = ""
        step_times = []
        previous_length = 0  # Track generated sequence length
        
        with model.transformer.h.inference_session(max_length=args.seq_len) as sess:
            for step in range(args.seq_len):
                start_time = perf_counter()

                outputs = model.generate(session=sess, **generate_kwargs)
                
                # Debug: print outputs info
                logger.info(f"Step {step}: outputs[0].shape = {outputs[0].shape}, previous_length = {previous_length}")
                
                # Only decode the new token(s)
                if len(outputs[0]) > previous_length:
                    new_token_ids = outputs[0][previous_length:]
                    new_token = tokenizer.decode(new_token_ids, skip_special_tokens=True)
                    previous_length = len(outputs[0])
                else:
                    # Fallback: decode last token only
                    new_token = tokenizer.decode(outputs[0][-1:], skip_special_tokens=True)
                
                result += new_token

                if step >= args.warmup_steps:
                    step_times.append(perf_counter() - start_time)
                    speed = 1 / np.mean(step_times)
                    logger.info(f"{process_idx=} {step=} {speed=:.2f} | New token: {repr(new_token)} | Current result: {repr(result)}")

    speed = 1 / np.mean(step_times) if step_times else 0.0
    
    # Print the generated text result
    logger.info(f"Generated text (process {process_idx}): {repr(result)}")
    logger.info(f"Generated text length: {len(result)} characters")
    
    result_pipe.send(speed)


if __name__ == "__main__":
    main()
