"""Measure real LoRA optimizer steps against existing workers, without downloads.

PYTHONPATH=src python benchmarks/profile_lora_training.py --help
Workers must have BLOOMBEE_ENABLE_LORA=1. Set BLOOMBEE_LORA_PROFILE_DIR on
workers as well for their block/transfer JSONL traces. This client writes a
separate trace directory; collect worker files separately on multi-host runs.
Synthetic tokens measure execution cost, not task quality. RPC elapsed time
includes server work/queueing; protobuf bytes exclude TCP/libp2p framing.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch

from bloombee import AutoDistributedModelForCausalLM, DistributedLoRA, LoRAConfig
from bloombee.utils.lora import LORA_CACHE_ENV, LORA_CACHED_MODES
from bloombee.utils.lora_profile import memory_snapshot, profile_step, synchronize
from bloombee.utils.training_plan import TrainingPlan


def summarize_stage_profiles(events, config, *, accumulation=1):
    """Keep per-microbatch diagnostics and estimate cost from complete optimizer steps.

    Keep each actual peer/span separate. A deployment operator must still provide
    shared-resource reservations and calibrated rank utility before planning.
    """
    pairs = defaultdict(dict)
    samples = defaultdict(lambda: defaultdict(dict))
    cache_by_stage = {}
    steps = set()
    for event in events:
        if event["kind"] == "rpc" and event.get("warmup") is False:
            if event["phase"] in pairs[event["id"]]:
                raise ValueError("Duplicate profiling RPC phase")
            pairs[event["id"]][event["phase"]] = event
    for pair in pairs.values():
        if set(pair) != {"forward", "backward"}:
            raise ValueError("Incomplete training profile RPC pair")
        f, b = pair["forward"], pair["backward"]
        key = (*[f[k] for k in ("peer", "start", "end")], f.get("execution_mode", "block"))
        if key != (*[b[k] for k in ("peer", "start", "end")], b.get("execution_mode", "block")):
            raise ValueError("Forward/backward profile route differs")
        cache_settings = f.get("cache_settings")
        if (
            cache_settings != b.get("cache_settings")
            or cache_by_stage.setdefault(key, cache_settings) != cache_settings
        ):
            raise ValueError("Worker cache settings changed during profiling")
        step, microbatch = f["step"], f["microbatch"]
        if (step, microbatch) != (b["step"], b["microbatch"]) or not 0 <= microbatch < accumulation:
            raise ValueError("Invalid forward/backward profile step or microbatch")
        if microbatch in samples[key][step]:
            raise ValueError("Duplicate span profile within a microbatch")
        samples[key][step][microbatch] = f["total_s"] + b["total_s"]
        steps.add(step)
    profiles = []
    for (peer, start, end, mode), by_step in sorted(samples.items()):
        values = [seconds for batch in by_step.values() for seconds in batch.values()]
        complete = [sum(batch.values()) for batch in by_step.values() if len(batch) == accumulation]
        # Adaptive routes can measure a span in only part of a step. Such
        # samples are useful diagnostics, but cannot cost a fixed-stage plan.
        step_cost = statistics.median(complete) if len(complete) == len(steps) else None
        cache_settings = cache_by_stage[(peer, start, end, mode)] or {}
        profiles.append(
            dict(
                peer=peer,
                execution_mode=mode,
                start=start,
                end=end,
                samples=len(values),
                median_forward_backward_rpc_s=statistics.median(values),
                optimizer_step_samples=len(complete),
                median_step_rpc_s=step_cost,
                rpc_seconds=step_cost / accumulation if step_cost is not None else None,
                rank_by_layer=[{t: config.rank_for(i, t) for t in config.target_modules} for i in range(start, end)],
                **{key: cache_settings[key] for key in LORA_CACHE_ENV if key in cache_settings},
            )
        )
    return profiles


def run_profile(model, batches, optimizer, *, warmup, steps):
    """Include Adam's first allocation in cold_steps; report steady samples separately."""
    device = batches[0].device
    samples = []
    for step in range(warmup + steps):
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for microbatch, ids in enumerate(batches):
            with profile_step(step, microbatch, warmup=step < warmup):
                loss = model(input_ids=ids, labels=ids, use_cache=False).loss
                (loss / len(batches)).backward()
                losses.append(float(loss.detach()))
        parameters = [p for p in model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        synchronize(device)
        optimizer_start = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        finished = time.perf_counter()
        samples.append(
            dict(
                step=step,
                warmup=step < warmup,
                seconds=finished - start,
                optimizer_seconds=finished - optimizer_start,
                loss=statistics.mean(losses),
                **memory_snapshot(device),
            )
        )
    measured = samples[warmup:]
    elapsed = sum(s["seconds"] for s in measured)
    effective_tokens = steps * sum(x.numel() for x in batches)
    return dict(
        schema_version=1,
        execution_mode="stateless_lora_v1",
        instrumented=True,
        samples=samples,
        steady_median_step_s=statistics.median(s["seconds"] for s in measured),
        effective_tokens_per_second=effective_tokens / elapsed,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        microbatch_size=batches[0].shape[0],
        sequence_length=batches[0].shape[1],
        accumulation=len(batches),
        effective_batch_size=batches[0].shape[0] * len(batches),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--initial-peers", nargs="+", required=True)
    parser.add_argument("--dht-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--lora-config", type=Path, help="LoRAConfig JSON with optional rank/alpha patterns")
    parser.add_argument("--plan", type=Path, help="Static training plan to measure on deployed workers")
    parser.add_argument("--worker-peers", type=Path, help="JSON mapping planned workers to peer IDs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if bool(args.plan) != bool(args.worker_peers) or (args.plan and args.lora_config):
        parser.error("--plan requires --worker-peers and supplies its own adapter config")
    plan = TrainingPlan.load(args.plan) if args.plan else None
    if plan and (args.model, args.batch_size, args.sequence_length, args.accumulation) != (
        plan.workload.model,
        plan.workload.batch_size,
        plan.workload.sequence_length,
        plan.workload.accumulation,
    ):
        parser.error("Profile settings must match the plan workload")
    if min(args.batch_size, args.accumulation, args.steps) < 1 or args.sequence_length < 2 or args.warmup < 1:
        parser.error("batch-size, accumulation, steps, warmup must be positive; sequence-length >=2")
    if args.output.exists() or args.output.with_suffix(".traces").exists():
        raise FileExistsError("Use a new report and trace directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["BLOOMBEE_LORA_PROFILE_DIR"] = str(args.output.with_suffix(".traces").resolve())
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    revision = plan.workload.model_revision if plan else None
    torch_dtype = plan.workload.torch_dtype if plan else torch.float16
    base = AutoDistributedModelForCausalLM.from_pretrained(
        args.model,
        initial_peers=args.initial_peers,
        dht_prefix=args.dht_prefix,
        revision=revision[3:] if revision and revision.startswith("hf:") else None,
        torch_dtype=torch_dtype,
        request_timeout=300,
        connect_timeout=30,
        max_retries=1,
    ).to(args.device)
    config = (
        plan.lora_config
        if plan
        else LoRAConfig(**json.loads(args.lora_config.read_text()))
        if args.lora_config
        else LoRAConfig(args.r, args.alpha, target_modules=tuple(args.target_modules))
    )
    model = (
        DistributedLoRA(
            base, config, plan=plan, worker_peers=json.loads(args.worker_peers.read_text()) if plan else None
        )
        .to(args.device)
        .train()
    )
    try:
        batches = [
            torch.randint(base.config.vocab_size, (args.batch_size, args.sequence_length), device=args.device)
            for _ in range(args.accumulation)
        ]
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
        report = run_profile(model, batches, optimizer, warmup=args.warmup, steps=args.steps)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        report.update(
            execution_mode="stateless_lora_v2"
            if config.rank_pattern or config.alpha_pattern or plan
            else "stateless_lora_v1",
            training_plan=plan.to_dict() if plan else None,
            model=args.model,
            model_revision=model.remote_layers.base_model_revision,
            model_type=base.config.model_type,
            input_dtype=str(torch_dtype),
            model_config=base.config.to_dict(),
            adapter_config=model.lora_config.to_dict(),
            host=socket.gethostname(),
            platform=platform.platform(),
            seed=args.seed,
            data="fixed synthetic unpadded token batches; not a quality evaluation",
            tokens_sha256=hashlib.sha256(b"".join(x.cpu().numpy().tobytes() for x in batches)).hexdigest(),
            source_revision=revision.stdout.strip() if revision.returncode == 0 else None,
            source_dirty=bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
            versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")},
            device=torch.cuda.get_device_name(args.device) if torch.device(args.device).type == "cuda" else args.device,
            measurement_notes=[
                "Synchronized instrumentation affects overlap and adds overhead; not an uninstrumented throughput claim.",
                "Worker CUDA peak is cumulative since its first profiled task; client peak resets per optimizer step.",
                "RSS includes shared mappings; do not sum process RSS as unique host memory.",
                "Block forward time includes weight staging; runtime_transfer counts task tensors only.",
                "RPC time includes serialization, server work, queues and network; not a bandwidth measurement.",
            ],
        )
        events = [
            json.loads(line)
            for path in args.output.with_suffix(".traces").glob("lora-*.jsonl")
            for line in path.read_text().splitlines()
        ]
        report["stage_profiles"] = summarize_stage_profiles(events, config, accumulation=args.accumulation)
        if plan and any(s.execution_mode != "block" for s in plan.stages):
            report["execution_mode"] = "stateless_lora_span_v3"
        if plan and any(s.execution_mode in LORA_CACHED_MODES for s in plan.stages):
            report["execution_mode"] = "cached_lora_span_v4"
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {k: report[k] for k in ("steady_median_step_s", "effective_tokens_per_second", "trainable_parameters")}
            )
        )
    finally:
        model.remote_layers.sequence_manager.shutdown()


if __name__ == "__main__":
    main()
