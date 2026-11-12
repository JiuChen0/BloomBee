"""Audit collected client/worker JSONL traces against the stateless LoRA contract.

This checks successful matched forward/backward RPC pairs, excluding eval-only
calls. Use --require-worker-events after collecting all remote worker traces.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from bloombee.utils.lora import LORA_CACHED_MODES
from bloombee.utils.lora_profile import expected_stateless_cost


def audit_events(events, require_workers=False):
    groups = defaultdict(list)
    for event in events:
        if "id" in event:
            groups[event["id"]].append(event)
    pairs, worker_pairs, errors = 0, 0, []
    totals = defaultdict(int)
    durations = defaultdict(float)
    for request_id, records in groups.items():
        rpc = [r for r in records if r["kind"] == "rpc"]
        forward = [r for r in rpc if r["phase"] == "forward"]
        backward = [r for r in rpc if r["phase"] == "backward"]
        if not backward:
            continue
        if len(forward) != 1 or len(backward) != 1:
            errors.append(f"{request_id}: duplicate or unmatched RPC")
            continue
        f, b = forward[0], backward[0]
        pairs += 1
        durations["client_rpc_s"] += f["total_s"] + b["total_s"]
        boundary = f["response_tensor_bytes"]
        cached = f.get("execution_mode") in LORA_CACHED_MODES
        adapter = f["adapter_tensor_bytes"] if cached else f["request_tensor_bytes"] - boundary
        expected = expected_stateless_cost(f["end"] - f["start"], boundary, adapter)
        mode = f.get("execution_mode", "block")
        if mode != b.get("execution_mode", "block"):
            errors.append(f"{request_id}: forward/backward execution mode differs")
        if mode == "span_recompute":
            n = f["end"] - f["start"]
            expected["forward_calls"] = 2 * n + n * (n - 1) // 2
        if cached:
            expected["request_tensor_bytes"] = 3 * boundary + adapter * (
                int(f["parameters_sent"]) + int(b["parameters_sent"])
            )
            if b["checkpoint_hit"]:
                expected["forward_calls"] = 2 * (f["end"] - f["start"])
        for direction in ("request_tensor_bytes", "response_tensor_bytes"):
            measured = f[direction] + b[direction]
            totals[direction] += measured
            if measured != expected[direction]:
                errors.append(f"{request_id}: {direction} {measured} != {expected[direction]}")
        blocks = [r for r in records if r["kind"] == "block" and r.get("status") == "ok"]
        if blocks:
            worker_pairs += 1
            nf = sum(r["phase"] in ("forward", "recompute") for r in blocks)
            nb = sum(r["phase"] == "backward" for r in blocks)
            if nf != expected["forward_calls"] or nb != expected["backward_calls"]:
                errors.append(f"{request_id}: block calls F={nf}, B={nb} disagree with span size")
            transfers = [r for r in records if r["kind"] == "runtime_transfer"]
            retries = [r for r in records if r["kind"] == "cache_retry"]
            if len(transfers) != (expected["forward_calls"] if mode == "block" else 2 + len(retries)):
                errors.append(f"{request_id}: incomplete runtime transfer records")
            durations["worker_block_s_including_weight_staging"] += sum(r["seconds"] for r in blocks)
            durations["worker_task_transfer_s"] += sum(
                r["input_transfer_s"] + r["output_transfer_s"] for r in transfers
            )
            durations["worker_queue_s"] += sum(r["queue_wait_s"] for r in transfers)
            durations["worker_task_transfer_s"] += sum(r["seconds"] for r in records if r["kind"] == "span_transfer")
            for record in records:
                if record["kind"] == "span_execution" and record["checkpoint_bytes"] > record["checkpoint_limit_bytes"]:
                    errors.append(f"{request_id}: span checkpoint reservation exceeded its limit")
                if record["kind"] == "lora_cache" and (
                    record["adapter_bytes"] > record["adapter_limit_bytes"]
                    or record["activation_bytes"] > record["activation_limit_bytes"]
                ):
                    errors.append(f"{request_id}: cache exceeded byte budget")
            totals["retry_request_tensor_bytes"] += sum(r["request_tensor_bytes"] for r in retries)
        elif require_workers:
            errors.append(f"{request_id}: missing worker events")
    if not pairs:
        errors.append("No complete training forward/backward RPC pairs")
    if worker_pairs == pairs:
        durations["unattributed_s_including_protocol_serialization_and_instrumentation"] = (
            durations["client_rpc_s"]
            - durations["worker_block_s_including_weight_staging"]
            - durations["worker_task_transfer_s"]
            - durations["worker_queue_s"]
        )
    return dict(pairs=pairs, worker_pairs=worker_pairs, totals=dict(totals), durations=dict(durations), errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--require-worker-events", action="store_true")
    args = parser.parse_args()
    events = [json.loads(line) for p in args.directory.rglob("lora-*.jsonl") for line in p.read_text().splitlines()]
    report = audit_events(events, args.require_worker_events)
    print(json.dumps(report, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
