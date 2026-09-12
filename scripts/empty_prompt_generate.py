#!/usr/bin/env python3
"""Generate a short greedy continuation from BOS (no user prompt)."""
from __future__ import annotations

import argparse
import os
import sys

import torch
from transformers import AutoTokenizer

from bloombee import AutoDistributedModelForCausalLM


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--initial_peers", nargs="+", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--torch_dtype", default="float16")
    parser.add_argument("--client_device", default="cpu")
    parser.add_argument("--dht_prefix", default=None)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    dtype = getattr(torch, args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, revision=args.revision)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    if bos is None:
        raise SystemExit("tokenizer has neither bos_token_id nor eos_token_id")

    model_kwargs = dict(
        initial_peers=args.initial_peers,
        torch_dtype=dtype,
    )
    if args.dht_prefix:
        model_kwargs["dht_prefix"] = args.dht_prefix
    if args.revision:
        model_kwargs["revision"] = args.revision
    model_kwargs["use_safetensors"] = True

    model = AutoDistributedModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.client_device and args.client_device != "cpu":
        model = model.to(args.client_device)

    input_ids = torch.tensor([[int(bos)]], device=model.device if hasattr(model, "device") else "cpu")
    print(f"BOS id={bos!r} token={tokenizer.decode([bos])!r} device={input_ids.device}", flush=True)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    token_ids = outputs[0].tolist()
    text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    new_text = tokenizer.decode(outputs[0][1:], skip_special_tokens=False)
    print("TOKEN_IDS=" + " ".join(str(t) for t in token_ids), flush=True)
    print("FULL_TEXT_BEGIN", flush=True)
    print(text, flush=True)
    print("FULL_TEXT_END", flush=True)
    print("CONTINUATION_BEGIN", flush=True)
    print(new_text, flush=True)
    print("CONTINUATION_END", flush=True)
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        import traceback

        traceback.print_exc()
        rc = 1
    os._exit(rc)
