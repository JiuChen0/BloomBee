#!/usr/bin/env python3
import argparse
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


NETWORK_SUMMARY_RE = re.compile(
    r"\[NETWORK_TX\] SUMMARY \| step_id=(?P<step_id>[a-f0-9\-]+) \| "
    r"send=(?P<send_kb>[0-9.]+)KB \| recv=(?P<recv_kb>[0-9.]+)KB \| "
    r"serialize=(?P<serialize_ms>[0-9.]+)ms \| network=(?P<network_ms>[0-9.]+)ms \| "
    r"deserialize=(?P<deserialize_ms>[0-9.]+)ms \| total=(?P<total_ms>[0-9.]+)ms"
)
CLIENT_SERVER_END_RE = re.compile(
    r"\[CLIENT_SERVER_END\] ServerIdx=(?P<server_idx>\d+) \| Blocks=(?P<blocks>\d+:\d+) \| Duration=(?P<duration_ms>[0-9.]+)ms"
)
STEP_LATENCY_RE = re.compile(
    r"\[STEP_LATENCY\] Process=(?P<process>\d+) \| Step=(?P<step>\d+) \| Latency=(?P<latency_ms>[0-9.]+)ms"
)
MBPIPE_SUMMARY_RE = re.compile(
    r"\[MBPIPE_SUMMARY\] step=(?P<step>\d+) mb=(?P<mb>\d+) "
    r"compute=(?P<compute_ms>[0-9.]+)ms elapsed=(?P<elapsed_ms>[0-9.]+)ms "
    r"wait=(?P<wait_ms>[0-9.]+)ms\((?P<wait_pct>[0-9.]+)%\) "
    r"launch=(?P<launch_ms>[0-9.]+)ms\((?P<launch_pct>[0-9.]+)%\) "
    r"efficiency=(?P<efficiency_pct>[0-9.]+)%"
)
CROSS_STAGE_SUMMARY_RE = re.compile(
    r"\[CROSS_STAGE_OVERLAP_SUMMARY\] step=(?P<step_id>[a-f0-9\-]+) "
    r"overlap=(?P<overlap_ms>[0-9.]+)ms, Stage2_compute=(?P<stage2_compute_ms>[0-9.]+)ms, "
    r"efficiency=(?P<efficiency_pct>[0-9.]+)%, strict_efficiency=(?P<strict_efficiency_pct>[0-9.]+)%, "
    r"comparable_compute=(?P<comparable_compute_ms>[0-9.]+)ms"
)
LOSSLESS_PROFILE_RE = re.compile(
    r"\[LOSSLESS_PROFILE\] side=(?P<side>\S+) dir=(?P<dir>\S+) "
    r"stage=(?P<stage>\S+) step_id=(?P<step_id>[a-f0-9\-]+) batch=(?P<batch>\d+) (?P<body>.+)$"
)
STEP_TIMING_BREAKDOWN_RE = re.compile(
    r"\[STEP_TIMING_BREAKDOWN\] step_id=(?P<step_id>\S+) mode=(?P<mode>\S+) "
    r"queue_wait=(?P<queue_wait_ms>-?[0-9.]+)ms queue_source=(?P<queue_source>\S+) "
    r"deserialize=(?P<deserialize_ms>-?[0-9.]+)ms compute=(?P<compute_ms>-?[0-9.]+)ms "
    r"serialize=(?P<serialize_ms>-?[0-9.]+)ms residual=(?P<residual_ms>-?[0-9.]+)ms "
    r"step_total=(?P<step_total_ms>-?[0-9.]+)ms total_with_queue=(?P<total_with_queue_ms>-?[0-9.]+)ms "
    r"cross_gpu_window=(?P<cross_gpu_window_ms>-?[0-9.]+)ms "
    r"batch=(?P<batch>\d+) seq_inc=(?P<seq_inc>-?\d+)"
    r"(?: raw_seq=(?P<raw_seq>-?\d+))? is_spec_dec=(?P<is_spec_dec>\d+)"
)
STEP_TIMING_BREAKDOWN_MB_RE = re.compile(
    r"\[STEP_TIMING_BREAKDOWN_MB\] step_id=(?P<step_id>\S+) mode=micro_batch "
    r"expected_mb=(?P<expected_mb>\d+) recv_mb=(?P<recv_mb>\d+) "
    r"queue_wait_sum=(?P<queue_wait_sum_ms>-?[0-9.]+)ms "
    r"deserialize_sum=(?P<deserialize_sum_ms>-?[0-9.]+)ms "
    r"compute_sum=(?P<compute_sum_ms>-?[0-9.]+)ms "
    r"serialize=(?P<serialize_ms>-?[0-9.]+)ms "
    r"residual=(?P<residual_ms>-?[0-9.]+)ms "
    r"total=(?P<total_ms>-?[0-9.]+)ms "
    r"queue_sources=(?P<queue_sources>\S+)"
    r"(?: queue_wait_pre=(?P<queue_wait_pre_ms>-?[0-9.]+)ms "
    r"queue_wait_inter=(?P<queue_wait_inter_ms>-?[0-9.]+)ms "
    r"total_with_pre_wait=(?P<total_with_pre_wait_ms>-?[0-9.]+)ms)?"
)
HANDLER_STEP_TIMING_RE = re.compile(
    r"\[HANDLER_STEP_TIMING\] step_id=(?P<step_id>\S+) "
    r"queue_wait=(?P<queue_wait_ms>-?[0-9.]+)ms queue_source=(?P<queue_source>\S+) "
    r"push_schedule=(?P<push_schedule_ms>-?[0-9.]+)ms "
    r"response_emit=(?P<response_emit_ms>-?[0-9.]+)ms "
    r"handler_total=(?P<handler_total_ms>-?[0-9.]+)ms "
    r"can_push=(?P<can_push>[01])"
)


def _extract_named_values(text: str, expected_keys):
    """
    Parse key=value tokens with optional unit suffixes (ms/KB).
    Returns {key: float} for keys listed in expected_keys.
    """
    expected = set(expected_keys)
    values = {}
    for m in re.finditer(r"(?P<key>[A-Za-z0-9_]+)=(?P<val>-?[0-9.]+)(?P<unit>ms|KB)?", text):
        key = m.group("key")
        if key in expected:
            values[key] = float(m.group("val"))
    return values


def _safe_mean(xs):
    return statistics.mean(xs) if xs else 0.0


def _safe_median(xs):
    return statistics.median(xs) if xs else 0.0


def _fmt(v):
    return f"{v:.2f}"


def _pearson_corr(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = _safe_mean(xs)
    mean_y = _safe_mean(ys)
    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        dy = y - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy
    if var_x <= 0 or var_y <= 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def _parse_stage_start(stage: str):
    try:
        head = str(stage).split(":", 1)[0]
        return int(head)
    except Exception:
        return None


def parse_client_log(path: Path):
    step_network = defaultdict(list)
    server_duration = defaultdict(list)
    step_latency = []
    lossless_rows = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = NETWORK_SUMMARY_RE.search(line)
            if m:
                step_network[m.group("step_id")].append(
                    {
                        "send_kb": float(m.group("send_kb")),
                        "recv_kb": float(m.group("recv_kb")),
                        "serialize_ms": float(m.group("serialize_ms")),
                        "network_ms": float(m.group("network_ms")),
                        "deserialize_ms": float(m.group("deserialize_ms")),
                        "total_ms": float(m.group("total_ms")),
                    }
                )
                continue

            m = CLIENT_SERVER_END_RE.search(line)
            if m:
                idx = int(m.group("server_idx"))
                server_duration[idx].append(float(m.group("duration_ms")))
                continue

            m = STEP_LATENCY_RE.search(line)
            if m:
                step_latency.append(float(m.group("latency_ms")))
                continue

            m = LOSSLESS_PROFILE_RE.search(line)
            if m:
                body_values = _extract_named_values(
                    m.group("body"),
                    expected_keys={
                        "serialize_base",
                        "wrap",
                        "compress",
                        "raw",
                        "wire",
                        "ratio",
                        "act_nnz_ratio",
                        "serialize_calls",
                        "compress_ok",
                        "deserialize_base",
                        "unwrap",
                        "decompress",
                        "deserialize_calls",
                        "decompress_calls",
                    },
                )
                row = {
                    "side": m.group("side"),
                    "dir": m.group("dir"),
                    "stage": m.group("stage"),
                    "step_id": m.group("step_id"),
                    "batch": int(m.group("batch")),
                }
                row.update(body_values)
                lossless_rows.append(row)

    return step_network, server_duration, step_latency, lossless_rows


def parse_server1_log(path: Path):
    vals = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = MBPIPE_SUMMARY_RE.search(line)
            if m:
                vals.append(
                    {
                        "compute_ms": float(m.group("compute_ms")),
                        "elapsed_ms": float(m.group("elapsed_ms")),
                        "wait_ms": float(m.group("wait_ms")),
                        "launch_ms": float(m.group("launch_ms")),
                        "efficiency_pct": float(m.group("efficiency_pct")),
                    }
                )
    return vals


def parse_server2_log(path: Path):
    vals = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = CROSS_STAGE_SUMMARY_RE.search(line)
            if m:
                vals.append(
                    {
                        "overlap_ms": float(m.group("overlap_ms")),
                        "stage2_compute_ms": float(m.group("stage2_compute_ms")),
                        "efficiency_pct": float(m.group("efficiency_pct")),
                        "strict_efficiency_pct": float(m.group("strict_efficiency_pct")),
                        "comparable_compute_ms": float(m.group("comparable_compute_ms")),
                    }
                )
    return vals


def parse_detailed_server_logs(logs):
    step_rows = []
    mb_rows = []
    handler_rows = []

    for source_name, path in logs:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = STEP_TIMING_BREAKDOWN_RE.search(line)
                if m:
                    lossless_vals = _extract_named_values(
                        line,
                        expected_keys={
                            "lossless_rx_base",
                            "lossless_rx_decompress",
                            "lossless_rx_wire",
                            "lossless_rx_raw",
                            "lossless_rx_ratio",
                            "lossless_tx_base",
                            "lossless_tx_compress",
                            "lossless_tx_raw",
                            "lossless_tx_wire",
                            "lossless_tx_ratio",
                        },
                    )
                    step_rows.append(
                        {
                            "source": source_name,
                            "step_id": m.group("step_id"),
                            "mode": m.group("mode"),
                            "queue_source": m.group("queue_source"),
                            "queue_wait_ms": float(m.group("queue_wait_ms")),
                            "deserialize_ms": float(m.group("deserialize_ms")),
                            "compute_ms": float(m.group("compute_ms")),
                            "serialize_ms": float(m.group("serialize_ms")),
                            "residual_ms": float(m.group("residual_ms")),
                            "step_total_ms": float(m.group("step_total_ms")),
                            "total_with_queue_ms": float(m.group("total_with_queue_ms")),
                            "cross_gpu_window_ms": float(m.group("cross_gpu_window_ms")),
                            "lossless_rx_base_ms": lossless_vals.get("lossless_rx_base"),
                            "lossless_rx_decompress_ms": lossless_vals.get("lossless_rx_decompress"),
                            "lossless_rx_wire_kb": lossless_vals.get("lossless_rx_wire"),
                            "lossless_rx_raw_kb": lossless_vals.get("lossless_rx_raw"),
                            "lossless_rx_ratio": lossless_vals.get("lossless_rx_ratio"),
                            "lossless_tx_base_ms": lossless_vals.get("lossless_tx_base"),
                            "lossless_tx_compress_ms": lossless_vals.get("lossless_tx_compress"),
                            "lossless_tx_raw_kb": lossless_vals.get("lossless_tx_raw"),
                            "lossless_tx_wire_kb": lossless_vals.get("lossless_tx_wire"),
                            "lossless_tx_ratio": lossless_vals.get("lossless_tx_ratio"),
                        }
                    )
                    continue

                m = STEP_TIMING_BREAKDOWN_MB_RE.search(line)
                if m:
                    lossless_vals = _extract_named_values(
                        line,
                        expected_keys={
                            "lossless_rx_base_sum",
                            "lossless_rx_decompress_sum",
                            "lossless_rx_wire_sum",
                            "lossless_rx_raw_sum",
                            "lossless_rx_ratio",
                            "lossless_tx_base",
                            "lossless_tx_compress",
                            "lossless_tx_raw",
                            "lossless_tx_wire",
                            "lossless_tx_ratio",
                        },
                    )
                    queue_wait_pre_ms = m.group("queue_wait_pre_ms")
                    queue_wait_inter_ms = m.group("queue_wait_inter_ms")
                    total_with_pre_wait_ms = m.group("total_with_pre_wait_ms")
                    mb_rows.append(
                        {
                            "source": source_name,
                            "step_id": m.group("step_id"),
                            "expected_mb": int(m.group("expected_mb")),
                            "recv_mb": int(m.group("recv_mb")),
                            "queue_wait_sum_ms": float(m.group("queue_wait_sum_ms")),
                            "deserialize_sum_ms": float(m.group("deserialize_sum_ms")),
                            "compute_sum_ms": float(m.group("compute_sum_ms")),
                            "serialize_ms": float(m.group("serialize_ms")),
                            "residual_ms": float(m.group("residual_ms")),
                            "total_ms": float(m.group("total_ms")),
                            "queue_sources": m.group("queue_sources"),
                            "queue_wait_pre_ms": float(queue_wait_pre_ms) if queue_wait_pre_ms is not None else None,
                            "queue_wait_inter_ms": float(queue_wait_inter_ms) if queue_wait_inter_ms is not None else None,
                            "total_with_pre_wait_ms": float(total_with_pre_wait_ms)
                            if total_with_pre_wait_ms is not None
                            else None,
                            "lossless_rx_base_sum_ms": lossless_vals.get("lossless_rx_base_sum"),
                            "lossless_rx_decompress_sum_ms": lossless_vals.get("lossless_rx_decompress_sum"),
                            "lossless_rx_wire_sum_kb": lossless_vals.get("lossless_rx_wire_sum"),
                            "lossless_rx_raw_sum_kb": lossless_vals.get("lossless_rx_raw_sum"),
                            "lossless_rx_ratio": lossless_vals.get("lossless_rx_ratio"),
                            "lossless_tx_base_ms": lossless_vals.get("lossless_tx_base"),
                            "lossless_tx_compress_ms": lossless_vals.get("lossless_tx_compress"),
                            "lossless_tx_raw_kb": lossless_vals.get("lossless_tx_raw"),
                            "lossless_tx_wire_kb": lossless_vals.get("lossless_tx_wire"),
                            "lossless_tx_ratio": lossless_vals.get("lossless_tx_ratio"),
                        }
                    )
                    continue

                m = HANDLER_STEP_TIMING_RE.search(line)
                if m:
                    handler_rows.append(
                        {
                            "source": source_name,
                            "step_id": m.group("step_id"),
                            "queue_source": m.group("queue_source"),
                            "queue_wait_ms": float(m.group("queue_wait_ms")),
                            "push_schedule_ms": float(m.group("push_schedule_ms")),
                            "response_emit_ms": float(m.group("response_emit_ms")),
                            "handler_total_ms": float(m.group("handler_total_ms")),
                            "can_push": int(m.group("can_push")),
                        }
                    )
                    continue

    return step_rows, mb_rows, handler_rows


def summarize(client_log: Path, server1_log: Path, server2_log: Path):
    step_network, server_duration, step_latency, client_lossless_rows = parse_client_log(client_log)
    s1_vals = parse_server1_log(server1_log)
    s2_vals = parse_server2_log(server2_log)
    step_breakdown_rows, mb_breakdown_rows, handler_step_rows = parse_detailed_server_logs(
        [("server1", server1_log), ("server2", server2_log)]
    )

    step_records = []
    for step_id, entries in step_network.items():
        entries = sorted(entries, key=lambda x: x["network_ms"], reverse=True)
        if len(entries) >= 2:
            e0, e1 = entries[0], entries[1]
        elif len(entries) == 1:
            e0, e1 = entries[0], None
        else:
            continue

        rec = {
            "step_id": step_id,
            "send_total_kb": e0["send_kb"] + (e1["send_kb"] if e1 else 0.0),
            "recv_total_kb": e0["recv_kb"] + (e1["recv_kb"] if e1 else 0.0),
            "serialize_total_ms": e0["serialize_ms"] + (e1["serialize_ms"] if e1 else 0.0),
            "network_total_ms": e0["network_ms"] + (e1["network_ms"] if e1 else 0.0),
            "deserialize_total_ms": e0["deserialize_ms"] + (e1["deserialize_ms"] if e1 else 0.0),
            "network_stack_total_ms": e0["total_ms"] + (e1["total_ms"] if e1 else 0.0),
            "stage0_network_ms": e0["network_ms"],
            "stage1_network_ms": e1["network_ms"] if e1 else 0.0,
        }
        step_records.append(rec)

    client_lossless_by_step = {}
    for row in client_lossless_rows:
        step_id = row.get("step_id")
        if not step_id:
            continue
        rec = client_lossless_by_step.setdefault(
            step_id,
            {
                "tx_serialize_base_ms": 0.0,
                "tx_wrap_ms": 0.0,
                "tx_compress_ms": 0.0,
                "tx_raw_kb": 0.0,
                "tx_wire_kb": 0.0,
                "tx_serialize_calls": 0.0,
                "tx_compress_ok": 0.0,
                "tx_act_nnz_sum": 0.0,
                "tx_act_nnz_count": 0.0,
                "rx_deserialize_base_ms": 0.0,
                "rx_unwrap_ms": 0.0,
                "rx_decompress_ms": 0.0,
                "rx_wire_kb": 0.0,
                "rx_raw_kb": 0.0,
                "rx_deserialize_calls": 0.0,
                "rx_decompress_calls": 0.0,
                "rx_act_nnz_sum": 0.0,
                "rx_act_nnz_count": 0.0,
            },
        )
        if row.get("dir") == "tx":
            rec["tx_serialize_base_ms"] += float(row.get("serialize_base", 0.0))
            rec["tx_wrap_ms"] += float(row.get("wrap", 0.0))
            rec["tx_compress_ms"] += float(row.get("compress", 0.0))
            rec["tx_raw_kb"] += float(row.get("raw", 0.0))
            rec["tx_wire_kb"] += float(row.get("wire", 0.0))
            rec["tx_serialize_calls"] += float(row.get("serialize_calls", 0.0))
            rec["tx_compress_ok"] += float(row.get("compress_ok", 0.0))
            rec["tx_act_nnz_sum"] += float(row.get("act_nnz_ratio", 0.0))
            rec["tx_act_nnz_count"] += 1.0
        elif row.get("dir") == "rx":
            rec["rx_deserialize_base_ms"] += float(row.get("deserialize_base", 0.0))
            rec["rx_unwrap_ms"] += float(row.get("unwrap", 0.0))
            rec["rx_decompress_ms"] += float(row.get("decompress", 0.0))
            rec["rx_wire_kb"] += float(row.get("wire", 0.0))
            rec["rx_raw_kb"] += float(row.get("raw", 0.0))
            rec["rx_deserialize_calls"] += float(row.get("deserialize_calls", 0.0))
            rec["rx_decompress_calls"] += float(row.get("decompress_calls", 0.0))
            rec["rx_act_nnz_sum"] += float(row.get("act_nnz_ratio", 0.0))
            rec["rx_act_nnz_count"] += 1.0

    client_lossless_records = []
    for rec in client_lossless_by_step.values():
        tx_ratio = rec["tx_wire_kb"] / rec["tx_raw_kb"] if rec["tx_raw_kb"] > 0 else 1.0
        rx_ratio = rec["rx_wire_kb"] / rec["rx_raw_kb"] if rec["rx_raw_kb"] > 0 else 1.0
        tx_nnz_ratio = rec["tx_act_nnz_sum"] / rec["tx_act_nnz_count"] if rec["tx_act_nnz_count"] > 0 else 0.0
        rx_nnz_ratio = rec["rx_act_nnz_sum"] / rec["rx_act_nnz_count"] if rec["rx_act_nnz_count"] > 0 else 0.0
        merged = dict(rec)
        merged["tx_ratio"] = tx_ratio
        merged["rx_ratio"] = rx_ratio
        merged["tx_act_nnz_ratio"] = tx_nnz_ratio
        merged["rx_act_nnz_ratio"] = rx_nnz_ratio
        client_lossless_records.append(merged)

    client_lossless_rows_enriched = []
    stage_starts = sorted(
        {
            s
            for s in (_parse_stage_start(r.get("stage", "")) for r in client_lossless_rows)
            if s is not None
        }
    )
    split_point = (len(stage_starts) + 1) // 2
    front_starts = set(stage_starts[:split_point])
    for row in client_lossless_rows:
        raw_kb = float(row.get("raw", 0.0))
        wire_kb = float(row.get("wire", 0.0))
        ratio = (wire_kb / raw_kb) if raw_kb > 0 else float(row.get("ratio", 1.0))
        stage_start = _parse_stage_start(row.get("stage", ""))
        if stage_start is None:
            layer_bucket = "unknown"
        elif stage_start in front_starts:
            layer_bucket = "front"
        else:
            layer_bucket = "back"
        enriched = dict(row)
        enriched["raw_kb"] = raw_kb
        enriched["wire_kb"] = wire_kb
        enriched["ratio_wire_over_raw"] = ratio
        enriched["act_nnz_ratio"] = float(row.get("act_nnz_ratio", 0.0))
        enriched["layer_bucket"] = layer_bucket
        client_lossless_rows_enriched.append(enriched)

    print("=" * 92)
    print("Latency/Volume Breakdown (from existing logs)")
    print("=" * 92)
    print(f"client_log  : {client_log}")
    print(f"server1_log : {server1_log}")
    print(f"server2_log : {server2_log}")
    print()
    print("Client-side (per generation step):")
    print("-" * 92)
    print(
        f"{'metric':38} {'mean':>12} {'median':>12} {'count':>10}"
    )
    print("-" * 92)
    if step_records:
        for key, label in [
            ("send_total_kb", "send_total_kb"),
            ("recv_total_kb", "recv_total_kb"),
            ("serialize_total_ms", "serialize_total_ms"),
            ("network_total_ms", "network_total_ms"),
            ("deserialize_total_ms", "deserialize_total_ms"),
            ("network_stack_total_ms", "network_stack_total_ms"),
            ("stage0_network_ms", "stage0_network_ms"),
            ("stage1_network_ms", "stage1_network_ms"),
        ]:
            vals = [r[key] for r in step_records]
            print(f"{label:38} {_fmt(_safe_mean(vals)):>12} {_fmt(_safe_median(vals)):>12} {len(vals):>10}")
    else:
        print("No [NETWORK_TX] SUMMARY entries found.")
    print("-" * 92)

    if step_latency:
        print(f"{'step_latency_ms':38} {_fmt(_safe_mean(step_latency)):>12} {_fmt(_safe_median(step_latency)):>12} {len(step_latency):>10}")
        print("-" * 92)

    print()
    print("Client lossless wrapper profiling (from [LOSSLESS_PROFILE]):")
    print("-" * 92)
    print(f"{'metric':38} {'mean':>12} {'median':>12} {'count':>10}")
    print("-" * 92)
    if client_lossless_records:
        for key, label in [
            ("tx_serialize_base_ms", "tx_serialize_base_ms"),
            ("tx_compress_ms", "tx_compress_ms"),
            ("tx_wrap_ms", "tx_wrap_ms"),
            ("tx_raw_kb", "tx_raw_kb"),
            ("tx_wire_kb", "tx_wire_kb"),
            ("tx_ratio", "tx_wire_over_raw_ratio"),
            ("tx_act_nnz_ratio", "tx_act_nnz_ratio"),
            ("rx_deserialize_base_ms", "rx_deserialize_base_ms"),
            ("rx_decompress_ms", "rx_decompress_ms"),
            ("rx_unwrap_ms", "rx_unwrap_ms"),
            ("rx_wire_kb", "rx_wire_kb"),
            ("rx_raw_kb", "rx_raw_kb"),
            ("rx_ratio", "rx_wire_over_raw_ratio"),
            ("rx_act_nnz_ratio", "rx_act_nnz_ratio"),
        ]:
            vals = [r[key] for r in client_lossless_records]
            print(f"{label:38} {_fmt(_safe_mean(vals)):>12} {_fmt(_safe_median(vals)):>12} {len(vals):>10}")
    else:
        print("No [LOSSLESS_PROFILE] entries found in client log.")
    print("-" * 92)

    print()
    print("Client compression ratio by stage/layer (from [LOSSLESS_PROFILE]):")
    print("-" * 92)
    stage_groups = defaultdict(list)
    for row in client_lossless_rows_enriched:
        key = (row.get("dir", "?"), row.get("stage", "?"), row.get("layer_bucket", "unknown"))
        stage_groups[key].append(row)
    if stage_groups:
        print(f"{'dir':>5} {'stage':>12} {'bucket':>8} {'ratio_mean':>12} {'nnz_mean':>10} {'raw_kb':>10} {'wire_kb':>10} {'count':>8}")
        for (dir_name, stage_name, bucket), rows in sorted(stage_groups.items()):
            ratios = [r["ratio_wire_over_raw"] for r in rows]
            nnzs = [r["act_nnz_ratio"] for r in rows]
            raws = [r["raw_kb"] for r in rows]
            wires = [r["wire_kb"] for r in rows]
            print(
                f"{dir_name:>5} {str(stage_name):>12} {bucket:>8} {_fmt(_safe_mean(ratios)):>12} "
                f"{_fmt(_safe_mean(nnzs)):>10} {_fmt(_safe_mean(raws)):>10} {_fmt(_safe_mean(wires)):>10} {len(rows):>8}"
            )
    else:
        print("No per-stage compression rows found.")
    print("-" * 92)

    print()
    print("Client compression ratio by layer bucket (front/back):")
    print("-" * 92)
    bucket_groups = defaultdict(list)
    for row in client_lossless_rows_enriched:
        key = (row.get("dir", "?"), row.get("layer_bucket", "unknown"))
        bucket_groups[key].append(row)
    if bucket_groups:
        print(f"{'dir':>5} {'bucket':>10} {'ratio_mean':>12} {'nnz_mean':>10} {'raw_kb':>10} {'wire_kb':>10} {'count':>8}")
        for (dir_name, bucket), rows in sorted(bucket_groups.items()):
            ratios = [r["ratio_wire_over_raw"] for r in rows]
            nnzs = [r["act_nnz_ratio"] for r in rows]
            raws = [r["raw_kb"] for r in rows]
            wires = [r["wire_kb"] for r in rows]
            print(
                f"{dir_name:>5} {bucket:>10} {_fmt(_safe_mean(ratios)):>12} {_fmt(_safe_mean(nnzs)):>10} "
                f"{_fmt(_safe_mean(raws)):>10} {_fmt(_safe_mean(wires)):>10} {len(rows):>8}"
            )
    else:
        print("No bucket-level compression rows found.")
    print("-" * 92)

    print()
    print("Client compression ratio by batch size (from [LOSSLESS_PROFILE]):")
    print("-" * 92)
    batch_groups = defaultdict(list)
    for row in client_lossless_rows_enriched:
        key = (row.get("dir", "?"), int(row.get("batch", 0)))
        batch_groups[key].append(row)
    if batch_groups:
        print(f"{'dir':>5} {'batch':>8} {'ratio_mean':>12} {'nnz_mean':>10} {'raw_kb':>10} {'wire_kb':>10} {'count':>8}")
        for (dir_name, batch), rows in sorted(batch_groups.items()):
            ratios = [r["ratio_wire_over_raw"] for r in rows]
            nnzs = [r["act_nnz_ratio"] for r in rows]
            raws = [r["raw_kb"] for r in rows]
            wires = [r["wire_kb"] for r in rows]
            print(
                f"{dir_name:>5} {batch:>8} {_fmt(_safe_mean(ratios)):>12} {_fmt(_safe_mean(nnzs)):>10} "
                f"{_fmt(_safe_mean(raws)):>10} {_fmt(_safe_mean(wires)):>10} {len(rows):>8}"
            )
    else:
        print("No batch-level compression rows found.")
    print("-" * 92)

    print()
    print("Compression ratio vs activation non-zero ratio (from [LOSSLESS_PROFILE]):")
    print("-" * 92)
    print(f"{'dir':>5} {'pearson(ratio,nnz)':>22} {'mean_ratio':>12} {'mean_nnz':>10} {'count':>8}")
    printed_corr = False
    for dir_name in ("tx", "rx"):
        rows = [r for r in client_lossless_rows_enriched if r.get("dir") == dir_name]
        ratios = [r["ratio_wire_over_raw"] for r in rows]
        nnzs = [r["act_nnz_ratio"] for r in rows]
        if len(ratios) >= 2:
            corr = _pearson_corr(ratios, nnzs)
            printed_corr = True
            print(
                f"{dir_name:>5} {_fmt(corr):>22} {_fmt(_safe_mean(ratios)):>12} "
                f"{_fmt(_safe_mean(nnzs)):>10} {len(rows):>8}"
            )
    if not printed_corr:
        print("Not enough rows to compute correlation.")
    print("-" * 92)

    print()
    print("Client server-span duration (from [CLIENT_SERVER_END]):")
    print("-" * 92)
    print(f"{'server_idx':>10} {'mean_ms':>12} {'median_ms':>12} {'count':>10}")
    print("-" * 92)
    if server_duration:
        for idx in sorted(server_duration):
            vals = server_duration[idx]
            print(f"{idx:>10} {_fmt(_safe_mean(vals)):>12} {_fmt(_safe_median(vals)):>12} {len(vals):>10}")
    else:
        print("No [CLIENT_SERVER_END] entries found.")
    print("-" * 92)

    print()
    print("Server1 micro-batch compute/offload (from [MBPIPE_SUMMARY]):")
    print("-" * 92)
    if s1_vals:
        print(
            f"compute_ms_mean={_fmt(_safe_mean([x['compute_ms'] for x in s1_vals]))}, "
            f"elapsed_ms_mean={_fmt(_safe_mean([x['elapsed_ms'] for x in s1_vals]))}, "
            f"wait_ms_mean={_fmt(_safe_mean([x['wait_ms'] for x in s1_vals]))}, "
            f"launch_ms_mean={_fmt(_safe_mean([x['launch_ms'] for x in s1_vals]))}, "
            f"efficiency_pct_mean={_fmt(_safe_mean([x['efficiency_pct'] for x in s1_vals]))}, "
            f"samples={len(s1_vals)}"
        )
    else:
        print("No [MBPIPE_SUMMARY] entries found in server1 log.")
    print("-" * 92)

    print()
    print("Server2 cross-stage overlap (from [CROSS_STAGE_OVERLAP_SUMMARY]):")
    print("-" * 92)
    if s2_vals:
        print(
            f"stage2_compute_ms_mean={_fmt(_safe_mean([x['stage2_compute_ms'] for x in s2_vals]))}, "
            f"overlap_ms_mean={_fmt(_safe_mean([x['overlap_ms'] for x in s2_vals]))}, "
            f"efficiency_pct_mean={_fmt(_safe_mean([x['efficiency_pct'] for x in s2_vals]))}, "
            f"strict_efficiency_pct_mean={_fmt(_safe_mean([x['strict_efficiency_pct'] for x in s2_vals]))}, "
            f"samples={len(s2_vals)}"
        )
    else:
        print("No [CROSS_STAGE_OVERLAP_SUMMARY] entries found in server2 log.")
    print("-" * 92)

    print()
    print("Server detailed step breakdown (from [STEP_TIMING_BREAKDOWN]):")
    print("-" * 92)
    if step_breakdown_rows:
        mode_counts = Counter(x["mode"] for x in step_breakdown_rows)
        source_counts = Counter(x["source"] for x in step_breakdown_rows)
        queue_source_counts = Counter(x["queue_source"] for x in step_breakdown_rows)
        print(
            "modes="
            + ",".join(f"{k}:{v}" for k, v in sorted(mode_counts.items()))
            + " | sources="
            + ",".join(f"{k}:{v}" for k, v in sorted(source_counts.items()))
            + " | queue_sources="
            + ",".join(f"{k}:{v}" for k, v in sorted(queue_source_counts.items()))
        )
        print(
            f"{'metric':38} {'mean':>12} {'median':>12} {'count':>10}"
        )
        for key, label in [
            ("queue_wait_ms", "queue_wait_ms"),
            ("deserialize_ms", "deserialize_ms"),
            ("compute_ms", "compute_ms"),
            ("serialize_ms", "serialize_ms"),
            ("residual_ms", "residual_ms"),
            ("step_total_ms", "step_total_ms"),
            ("total_with_queue_ms", "total_with_queue_ms"),
            ("cross_gpu_window_ms", "cross_gpu_window_ms"),
        ]:
            vals = [r[key] for r in step_breakdown_rows]
            print(f"{label:38} {_fmt(_safe_mean(vals)):>12} {_fmt(_safe_median(vals)):>12} {len(vals):>10}")
    else:
        print("No [STEP_TIMING_BREAKDOWN] entries found.")
    print("-" * 92)

    print()
    print("Server lossless profiling (from [STEP_TIMING_BREAKDOWN]):")
    print("-" * 92)
    print(f"{'metric':38} {'mean':>12} {'median':>12} {'count':>10}")
    print("-" * 92)
    step_lossless_metrics = [
        ("lossless_rx_base_ms", "lossless_rx_base_ms"),
        ("lossless_rx_decompress_ms", "lossless_rx_decompress_ms"),
        ("lossless_rx_wire_kb", "lossless_rx_wire_kb"),
        ("lossless_rx_raw_kb", "lossless_rx_raw_kb"),
        ("lossless_rx_ratio", "lossless_rx_wire_over_raw_ratio"),
        ("lossless_tx_base_ms", "lossless_tx_base_ms"),
        ("lossless_tx_compress_ms", "lossless_tx_compress_ms"),
        ("lossless_tx_raw_kb", "lossless_tx_raw_kb"),
        ("lossless_tx_wire_kb", "lossless_tx_wire_kb"),
        ("lossless_tx_ratio", "lossless_tx_wire_over_raw_ratio"),
    ]
    printed_step_lossless = False
    if step_breakdown_rows:
        for key, label in step_lossless_metrics:
            vals = [r[key] for r in step_breakdown_rows if r.get(key) is not None]
            if vals:
                printed_step_lossless = True
                print(f"{label:38} {_fmt(_safe_mean(vals)):>12} {_fmt(_safe_median(vals)):>12} {len(vals):>10}")
    if not printed_step_lossless:
        print("No lossless fields found in [STEP_TIMING_BREAKDOWN] entries.")
    print("-" * 92)

    print()
    print("Server merged micro-batch step breakdown (from [STEP_TIMING_BREAKDOWN_MB]):")
    print("-" * 92)
    if mb_breakdown_rows:
        source_counts = Counter(x["source"] for x in mb_breakdown_rows)
        print("sources=" + ",".join(f"{k}:{v}" for k, v in sorted(source_counts.items())))
        print(
            f"{'metric':38} {'mean':>12} {'median':>12} {'count':>10}"
        )
        for key, label in [
            ("queue_wait_sum_ms", "queue_wait_sum_ms"),
            ("deserialize_sum_ms", "deserialize_sum_ms"),
            ("compute_sum_ms", "compute_sum_ms"),
            ("serialize_ms", "serialize_ms"),
            ("residual_ms", "residual_ms"),
            ("total_ms", "total_ms"),
        ]:
            vals = [r[key] for r in mb_breakdown_rows]
            print(f"{label:38} {_fmt(_safe_mean(vals)):>12} {_fmt(_safe_median(vals)):>12} {len(vals):>10}")
        pre_vals = [r["queue_wait_pre_ms"] for r in mb_breakdown_rows if r["queue_wait_pre_ms"] is not None]
        inter_vals = [r["queue_wait_inter_ms"] for r in mb_breakdown_rows if r["queue_wait_inter_ms"] is not None]
        twp_vals = [r["total_with_pre_wait_ms"] for r in mb_breakdown_rows if r["total_with_pre_wait_ms"] is not None]
        if pre_vals:
            print(f"{'queue_wait_pre_ms':38} {_fmt(_safe_mean(pre_vals)):>12} {_fmt(_safe_median(pre_vals)):>12} {len(pre_vals):>10}")
        if inter_vals:
            print(f"{'queue_wait_inter_ms':38} {_fmt(_safe_mean(inter_vals)):>12} {_fmt(_safe_median(inter_vals)):>12} {len(inter_vals):>10}")
        if twp_vals:
            print(f"{'total_with_pre_wait_ms':38} {_fmt(_safe_mean(twp_vals)):>12} {_fmt(_safe_median(twp_vals)):>12} {len(twp_vals):>10}")
        print("Note: queue_wait_sum may not be additive with total across micro-batches.")
    else:
        print("No [STEP_TIMING_BREAKDOWN_MB] entries found.")
    print("-" * 92)

    print()
    print("Server micro-batch lossless profiling (from [STEP_TIMING_BREAKDOWN_MB]):")
    print("-" * 92)
    print(f"{'metric':38} {'mean':>12} {'median':>12} {'count':>10}")
    print("-" * 92)
    mb_lossless_metrics = [
        ("lossless_rx_base_sum_ms", "lossless_rx_base_sum_ms"),
        ("lossless_rx_decompress_sum_ms", "lossless_rx_decompress_sum_ms"),
        ("lossless_rx_wire_sum_kb", "lossless_rx_wire_sum_kb"),
        ("lossless_rx_raw_sum_kb", "lossless_rx_raw_sum_kb"),
        ("lossless_rx_ratio", "lossless_rx_wire_over_raw_ratio"),
        ("lossless_tx_base_ms", "lossless_tx_base_ms"),
        ("lossless_tx_compress_ms", "lossless_tx_compress_ms"),
        ("lossless_tx_raw_kb", "lossless_tx_raw_kb"),
        ("lossless_tx_wire_kb", "lossless_tx_wire_kb"),
        ("lossless_tx_ratio", "lossless_tx_wire_over_raw_ratio"),
    ]
    printed_mb_lossless = False
    if mb_breakdown_rows:
        for key, label in mb_lossless_metrics:
            vals = [r[key] for r in mb_breakdown_rows if r.get(key) is not None]
            if vals:
                printed_mb_lossless = True
                print(f"{label:38} {_fmt(_safe_mean(vals)):>12} {_fmt(_safe_median(vals)):>12} {len(vals):>10}")
    if not printed_mb_lossless:
        print("No lossless fields found in [STEP_TIMING_BREAKDOWN_MB] entries.")
    print("-" * 92)

    print()
    print("Handler-side step overhead (from [HANDLER_STEP_TIMING]):")
    print("-" * 92)
    if handler_step_rows:
        source_counts = Counter(x["source"] for x in handler_step_rows)
        queue_source_counts = Counter(x["queue_source"] for x in handler_step_rows)
        print(
            "sources="
            + ",".join(f"{k}:{v}" for k, v in sorted(source_counts.items()))
            + " | queue_sources="
            + ",".join(f"{k}:{v}" for k, v in sorted(queue_source_counts.items()))
        )
        print(
            f"{'metric':38} {'mean':>12} {'median':>12} {'count':>10}"
        )
        for key, label in [
            ("queue_wait_ms", "queue_wait_ms"),
            ("push_schedule_ms", "push_schedule_ms"),
            ("response_emit_ms", "response_emit_ms"),
            ("handler_total_ms", "handler_total_ms"),
        ]:
            vals = [r[key] for r in handler_step_rows]
            print(f"{label:38} {_fmt(_safe_mean(vals)):>12} {_fmt(_safe_median(vals)):>12} {len(vals):>10}")
    else:
        print("No [HANDLER_STEP_TIMING] entries found.")
    print("-" * 92)

    if step_records and step_latency and s1_vals and s2_vals:
        mean_step = _safe_mean(step_latency)
        mean_client_net = _safe_mean([r["network_total_ms"] for r in step_records])
        mean_client_ser = _safe_mean([r["serialize_total_ms"] for r in step_records])
        mean_client_deser = _safe_mean([r["deserialize_total_ms"] for r in step_records])
        mean_s1_elapsed = _safe_mean([x["elapsed_ms"] for x in s1_vals])
        mean_s2_compute = _safe_mean([x["stage2_compute_ms"] for x in s2_vals])
        mean_overlap = _safe_mean([x["overlap_ms"] for x in s2_vals])

        implied_other = mean_step - (mean_client_ser + mean_client_deser + mean_client_net)
        overlap_adjusted_compute = mean_s1_elapsed + mean_s2_compute - mean_overlap

        print()
        print("Derived proxies (approximate, for trend comparison only):")
        print("-" * 92)
        print(f"mean_step_latency_ms              : {_fmt(mean_step)}")
        print(f"mean_client_network_stack_ms     : {_fmt(mean_client_ser + mean_client_net + mean_client_deser)}")
        print(f"implied_non_network_ms           : {_fmt(implied_other)}")
        print(f"overlap_adjusted_s1s2_compute_ms : {_fmt(overlap_adjusted_compute)}")
        print("-" * 92)
        print("Note: exact queue/scheduling split needs additional server-side step-level timestamps.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze BloomBee communication/computation/scheduling breakdown from logs."
    )
    parser.add_argument("--client-log", type=Path, default=Path("/home/cc/inference_batch.log"))
    parser.add_argument("--server1-log", type=Path, default=Path("/home/cc/server1.log"))
    parser.add_argument("--server2-log", type=Path, default=Path("/home/cc/server2.log"))
    args = parser.parse_args()
    summarize(args.client_log, args.server1_log, args.server2_log)


if __name__ == "__main__":
    main()
