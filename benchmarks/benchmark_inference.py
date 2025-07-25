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
    parser.add_argument("--test_prompt", type=str, default="none", choices=["short", "long", "none"], help="Choose test prompt length")
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

    step_times = []
    
    logger.info(f"🔍 [Process {process_idx}] BOS token id: {tokenizer.bos_token_id}")
    logger.info(f"🔍 [Process {process_idx}] Starting inference session...")
    
    if args.test_prompt == "short":
        test_prompt = "Hello"
        input_ids = tokenizer.encode(test_prompt, return_tensors="pt", add_special_tokens=True)
        logger.info(f"🔍 [Process {process_idx}] Using {args.test_prompt} prompt: {repr(test_prompt)}")
        logger.info(f"🔍 [Process {process_idx}] Input tokens: {input_ids.shape[1]} tokens")
        initial_text = test_prompt
    elif args.test_prompt == "long":
        test_prompt = "You've made significant progress in incorporating the KV cache into your LLaMA MHA generation function! The main issues in"
        input_ids = tokenizer.encode(test_prompt, return_tensors="pt", add_special_tokens=True)
        logger.info(f"🔍 [Process {process_idx}] Using {args.test_prompt} prompt: {repr(test_prompt)}")
        logger.info(f"🔍 [Process {process_idx}] Input tokens: {input_ids.shape[1]} tokens")
        initial_text = test_prompt
    else:  # args.test_prompt == "none"
        input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long)
        logger.info(f"🔍 [Process {process_idx}] Using no prompt, starting with BOS token only (id: {tokenizer.bos_token_id})")
        logger.info(f"🔍 [Process {process_idx}] Input tokens: 1 token (BOS)")
        initial_text = tokenizer.decode(tokenizer.bos_token_id)
        
    # 【新】在客户端维护一个完整的token序列，作为唯一的事实来源
    full_sequence_ids = input_ids
    result = initial_text
    
    input_length = full_sequence_ids.shape[1]
    required_length = input_length + args.seq_len
    effective_max_length = max(args.seq_len, required_length)
    
    logger.info(f"🔍 [Process {process_idx}] Input length: {input_length}, Required max_length: {effective_max_length}")
    
    with model.transformer.h.inference_session(max_length=effective_max_length) as sess:
        for step in range(args.seq_len):
            start_time = perf_counter()

            logger.info(f"🔍 [Process {process_idx}] Step {step} - Before generation:")
            logger.info(f"🔍 [Process {process_idx}] Current result length: {len(result)}")
            logger.info(f"🔍 [Process {process_idx}] Current result text: {repr(result)}")
            
            # 【修改】生成逻辑
            if step == 0:
                # 第一步（prefill），传入完整的 input_ids
                # generate 会返回 input_ids + new_token
                logger.info(f"🚀 [Process {process_idx}] FIRST STEP: Calling model.generate with input_ids of shape {full_sequence_ids.shape}")
                outputs = model.generate(full_sequence_ids, max_new_tokens=1, session=sess)
                full_sequence_ids = outputs
            else:
                # 后续步骤（decoding），不传入 input_ids
                # generate 只会返回 new_token
                logger.info(f"🚀 [Process {process_idx}] SUBSEQUENT STEP {step}: Calling model.generate without input_ids")
                new_token_only = model.generate(max_new_tokens=1, session=sess)
                # 【关键】手动将新token拼接到我们的完整序列后面
                full_sequence_ids = torch.cat([full_sequence_ids, new_token_only], dim=1)

            logger.info(f"🔍 [Process {process_idx}] Step {step} - After generation:")
            logger.info(f"🔍 [Process {process_idx}] Full sequence shape: {full_sequence_ids.shape}") # 使用 full_sequence_ids
            logger.info(f"🔍 [Process {process_idx}] Full sequence: {full_sequence_ids[0]}") # 使用 full_sequence_ids

            # 【修改】从我们自己维护的完整序列中获取新token
            new_token_id = full_sequence_ids[0, -1].item()
            logger.info(f"🔍 [Process {process_idx}] New token id: {new_token_id}")
            
            new_token_text = tokenizer.decode(new_token_id)
            logger.info(f"🔍 [Process {process_idx}] New token text: {repr(new_token_text)}")
            
            # 【注意】这里的解码和字符串拼接逻辑是正确的，可以保留
            result += new_token_text

            logger.info(f"🔍 [Process {process_idx}] Updated result: {repr(result)}")
            logger.info(f"🔍 [Process {process_idx}] Updated result length: {len(result)}")

            if step >= args.warmup_steps:
                step_times.append(perf_counter() - start_time)
                speed = 1 / np.mean(step_times)
                logger.info(f"{process_idx=} {step=} {speed=:.2f}")
                
            logger.info(f"🔍 [Process {process_idx}] Step {step} completed\n" + "="*50)
            
    logger.info(f"Generated text (process {process_idx}): {repr(result)}")
    logger.info(f"Generated text length: {len(result)} characters")
    
    speed_value = 1 / np.mean(step_times) if step_times else 0
    result_pipe.send(speed_value)


if __name__ == "__main__":
    main()