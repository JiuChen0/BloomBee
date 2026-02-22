#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from analyze_latency_breakdown import (
    parse_client_log,
    parse_detailed_server_logs,
    parse_server1_log,
    parse_server2_log,
)


def _safe_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _fmt(v):
    return f"{v:.2f}"


def _pct_delta(v, base):
    if base == 0:
        return 0.0
    return (v - base) / base * 100.0


def _parse_case_spec(spec: str):
    """
    Parse one experiment case.
    Format:
      name=bs84_comp1_off1,client=/path/infer.log,server1=/path/s1.log,server2=/path/s2.log,compression=1,offload=1,batch=84,seq=512
    """
    parsed = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid --case token: {token!r}, expected key=value")
        key, value = token.split("=", 1)
        parsed[key.strip()] = value.strip()

    required = ("name", "client", "server1", "server2")
    missing = [k for k in required if k not in parsed]
    if missing:
        raise ValueError(f"--case missing required keys: {missing}")
    return parsed


def _build_step_records(step_network):
    step_records = []
    for step_id, entries in step_network.items():
        ordered = sorted(entries, key=lambda x: x["network_ms"], reverse=True)
        if len(ordered) >= 2:
            e0, e1 = ordered[0], ordered[1]
        elif len(ordered) == 1:
            e0, e1 = ordered[0], None
        else:
            continue
        step_records.append(
            {
                "step_id": step_id,
                "send_total_kb": e0["send_kb"] + (e1["send_kb"] if e1 else 0.0),
                "recv_total_kb": e0["recv_kb"] + (e1["recv_kb"] if e1 else 0.0),
                "network_total_ms": e0["network_ms"] + (e1["network_ms"] if e1 else 0.0),
                "serialize_total_ms": e0["serialize_ms"] + (e1["serialize_ms"] if e1 else 0.0),
                "deserialize_total_ms": e0["deserialize_ms"] + (e1["deserialize_ms"] if e1 else 0.0),
            }
        )
    return step_records


def _collect_case_metrics(case):
    client_log = Path(case["client"])
    server1_log = Path(case["server1"])
    server2_log = Path(case["server2"])

    step_network, _, step_latency, lossless_rows = parse_client_log(client_log)
    s1_vals = parse_server1_log(server1_log)
    s2_vals = parse_server2_log(server2_log)
    step_breakdown_rows, mb_rows, _ = parse_detailed_server_logs([("server1", server1_log), ("server2", server2_log)])

    step_records = _build_step_records(step_network)

    tx_rows = [r for r in lossless_rows if r.get("dir") == "tx"]
    rx_rows = [r for r in lossless_rows if r.get("dir") == "rx"]

    tx_ratios = []
    for r in tx_rows:
        raw = float(r.get("raw", 0.0))
        wire = float(r.get("wire", 0.0))
        if raw > 0:
            tx_ratios.append(wire / raw)
    rx_ratios = []
    for r in rx_rows:
        raw = float(r.get("raw", 0.0))
        wire = float(r.get("wire", 0.0))
        if raw > 0:
            rx_ratios.append(wire / raw)

    mean_step_latency_ms = _safe_mean(step_latency)
    mean_send_kb = _safe_mean([r["send_total_kb"] for r in step_records])
    mean_recv_kb = _safe_mean([r["recv_total_kb"] for r in step_records])
    mean_network_ms = _safe_mean([r["network_total_ms"] for r in step_records])
    mean_queue_wait_ms = _safe_mean([r["queue_wait_ms"] for r in step_breakdown_rows])
    mean_compute_ms = _safe_mean([r["compute_ms"] for r in step_breakdown_rows])
    mean_step_total_ms = _safe_mean([r["step_total_ms"] for r in step_breakdown_rows])
    mean_mb_compute_ms = _safe_mean([r["compute_sum_ms"] for r in mb_rows])
    mean_s1_elapsed_ms = _safe_mean([x["elapsed_ms"] for x in s1_vals])
    mean_s2_compute_ms = _safe_mean([x["stage2_compute_ms"] for x in s2_vals])
    mean_overlap_ms = _safe_mean([x["overlap_ms"] for x in s2_vals])

    overlap_adjusted_compute = mean_s1_elapsed_ms + mean_s2_compute_ms - mean_overlap_ms
    mean_tx_compress_ms = _safe_mean([float(r.get("compress", 0.0)) for r in tx_rows])
    mean_rx_decompress_ms = _safe_mean([float(r.get("decompress", 0.0)) for r in rx_rows])
    mean_tx_nnz = _safe_mean([float(r.get("act_nnz_ratio", 0.0)) for r in tx_rows])
    mean_rx_nnz = _safe_mean([float(r.get("act_nnz_ratio", 0.0)) for r in rx_rows])

    comm_share_pct = (mean_network_ms / mean_step_latency_ms * 100.0) if mean_step_latency_ms > 0 else 0.0
    queue_share_pct = (mean_queue_wait_ms / mean_step_latency_ms * 100.0) if mean_step_latency_ms > 0 else 0.0

    return {
        "name": case.get("name", "case"),
        "compression": int(case.get("compression", "0")),
        "offload": int(case.get("offload", "0")),
        "batch": int(case.get("batch", case.get("batch_size", "0"))),
        "seq": int(case.get("seq", case.get("seq_len", "0"))),
        "step_latency_ms": mean_step_latency_ms,
        "network_total_ms": mean_network_ms,
        "queue_wait_ms": mean_queue_wait_ms,
        "compute_ms": mean_compute_ms,
        "step_total_ms": mean_step_total_ms,
        "send_kb": mean_send_kb,
        "recv_kb": mean_recv_kb,
        "tx_wire_over_raw": _safe_mean(tx_ratios) if tx_ratios else 1.0,
        "rx_wire_over_raw": _safe_mean(rx_ratios) if rx_ratios else 1.0,
        "tx_compress_ms": mean_tx_compress_ms,
        "rx_decompress_ms": mean_rx_decompress_ms,
        "tx_nnz_ratio": mean_tx_nnz,
        "rx_nnz_ratio": mean_rx_nnz,
        "mb_compute_ms": mean_mb_compute_ms,
        "s1_elapsed_ms": mean_s1_elapsed_ms,
        "s2_compute_ms": mean_s2_compute_ms,
        "overlap_ms": mean_overlap_ms,
        "overlap_adjusted_compute_ms": overlap_adjusted_compute,
        "comm_share_pct": comm_share_pct,
        "queue_share_pct": queue_share_pct,
        "client_log": str(client_log),
        "server1_log": str(server1_log),
        "server2_log": str(server2_log),
    }


def _print_case_table(results):
    print("=" * 140)
    print("Compression/Offload Trade-off Matrix")
    print("=" * 140)
    print(
        f"{'name':24} {'cmp':>4} {'off':>4} {'batch':>6} {'seq':>6} "
        f"{'step_ms':>10} {'net_ms':>10} {'queue_ms':>10} {'send_kb':>10} {'recv_kb':>10} "
        f"{'tx_ratio':>9} {'rx_ratio':>9} {'tx_cmp':>9} {'rx_dcmp':>9} {'comm%':>8} {'queue%':>8}"
    )
    print("-" * 140)
    for r in results:
        print(
            f"{r['name'][:24]:24} {r['compression']:>4} {r['offload']:>4} {r['batch']:>6} {r['seq']:>6} "
            f"{_fmt(r['step_latency_ms']):>10} {_fmt(r['network_total_ms']):>10} {_fmt(r['queue_wait_ms']):>10} "
            f"{_fmt(r['send_kb']):>10} {_fmt(r['recv_kb']):>10} {_fmt(r['tx_wire_over_raw']):>9} {_fmt(r['rx_wire_over_raw']):>9} "
            f"{_fmt(r['tx_compress_ms']):>9} {_fmt(r['rx_decompress_ms']):>9} {_fmt(r['comm_share_pct']):>8} {_fmt(r['queue_share_pct']):>8}"
        )
    print("-" * 140)


def _print_delta_table(results, baseline_name):
    baseline = None
    for r in results:
        if r["name"] == baseline_name:
            baseline = r
            break
    if baseline is None:
        baseline = results[0]
        baseline_name = baseline["name"]

    print()
    print(f"Relative Delta vs baseline={baseline_name}")
    print("-" * 140)
    print(
        f"{'name':24} {'d_step%':>10} {'d_net%':>10} {'d_queue%':>10} "
        f"{'d_send%':>10} {'d_recv%':>10} {'d_tx_ratio%':>12} {'d_rx_ratio%':>12}"
    )
    print("-" * 140)
    for r in results:
        print(
            f"{r['name'][:24]:24} "
            f"{_fmt(_pct_delta(r['step_latency_ms'], baseline['step_latency_ms'])):>10} "
            f"{_fmt(_pct_delta(r['network_total_ms'], baseline['network_total_ms'])):>10} "
            f"{_fmt(_pct_delta(r['queue_wait_ms'], baseline['queue_wait_ms'])):>10} "
            f"{_fmt(_pct_delta(r['send_kb'], baseline['send_kb'])):>10} "
            f"{_fmt(_pct_delta(r['recv_kb'], baseline['recv_kb'])):>10} "
            f"{_fmt(_pct_delta(r['tx_wire_over_raw'], baseline['tx_wire_over_raw'])):>12} "
            f"{_fmt(_pct_delta(r['rx_wire_over_raw'], baseline['rx_wire_over_raw'])):>12}"
        )
    print("-" * 140)
    print("Interpretation: negative d_step%/d_net%/d_queue% is better (latency reduced).")


def _write_csv(results, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "compression",
        "offload",
        "batch",
        "seq",
        "step_latency_ms",
        "network_total_ms",
        "queue_wait_ms",
        "compute_ms",
        "step_total_ms",
        "send_kb",
        "recv_kb",
        "tx_wire_over_raw",
        "rx_wire_over_raw",
        "tx_compress_ms",
        "rx_decompress_ms",
        "tx_nnz_ratio",
        "rx_nnz_ratio",
        "mb_compute_ms",
        "s1_elapsed_ms",
        "s2_compute_ms",
        "overlap_ms",
        "overlap_adjusted_compute_ms",
        "comm_share_pct",
        "queue_share_pct",
        "client_log",
        "server1_log",
        "server2_log",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple BloomBee experiments and quantify compression/offload trade-offs."
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help=(
            "Experiment case spec, repeatable. Example: "
            "name=bs84_comp1_off1,client=/home/cc/inference.log,server1=/home/cc/server1.log,"
            "server2=/home/cc/server2.log,compression=1,offload=1,batch=84,seq=512"
        ),
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="",
        help="Case name used as baseline in relative-delta table (default: first case).",
    )
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    cases = [_parse_case_spec(spec) for spec in args.case]
    results = [_collect_case_metrics(case) for case in cases]

    _print_case_table(results)
    _print_delta_table(results, args.baseline)
    if args.output_csv is not None:
        _write_csv(results, args.output_csv)
        print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
