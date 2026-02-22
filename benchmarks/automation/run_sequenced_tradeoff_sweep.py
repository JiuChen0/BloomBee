#!/usr/bin/env python3
"""
Sequenced multi-host sweep runner for BloomBee trade-off studies.

Flow per case (default):
  1) start DHT
  2) wait 30s
  3) start server1
  4) wait 30s
  5) start server2
  6) wait 30s
  7) run benchmark inference (foreground)
  8) stop server2/server1/dht
  9) collect logs to local machine
 10) generate per-case summary + update sweep matrix

Designed for running from Windows Git Bash (or Linux/macOS) with SSH access.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


NETWORK_SUMMARY_RE = re.compile(
    r"\[NETWORK_TX\] SUMMARY \| step_id=(?P<step_id>[a-f0-9\-]+) \| "
    r"send=(?P<send_kb>[0-9.]+)KB \| recv=(?P<recv_kb>[0-9.]+)KB \| "
    r"serialize=(?P<serialize_ms>[0-9.]+)ms \| network=(?P<network_ms>[0-9.]+)ms \| "
    r"deserialize=(?P<deserialize_ms>[0-9.]+)ms \| total=(?P<total_ms>[0-9.]+)ms"
)
STEP_LATENCY_RE = re.compile(
    r"\[STEP_LATENCY\] Process=(?P<process>\d+) \| Step=(?P<step>\d+) \| Latency=(?P<latency_ms>[0-9.]+)ms"
)
STEP_TIMING_BREAKDOWN_RE = re.compile(
    r"\[STEP_TIMING_BREAKDOWN\] step_id=(?P<step_id>\S+) mode=(?P<mode>\S+) "
    r"queue_wait=(?P<queue_wait_ms>-?[0-9.]+)ms queue_source=(?P<queue_source>\S+)"
)
LOSSLESS_PROFILE_RE = re.compile(
    r"\[LOSSLESS_PROFILE\] side=(?P<side>\S+) dir=(?P<dir>\S+) "
    r"stage=(?P<stage>\S+) step_id=(?P<step_id>[a-f0-9\-]+) batch=(?P<batch>\d+) (?P<body>.+)$"
)


def _utc_now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_name(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in s)


def _safe_mean(xs: List[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def _fmt(v: float) -> str:
    return f"{v:.2f}"


def _pct_delta(v: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (v - base) / base * 100.0


def _render(value: str, context: Mapping[str, Any]) -> str:
    try:
        return str(value).format(**context)
    except KeyError as exc:
        missing = str(exc).strip("'")
        raise ValueError(f"Missing template variable '{missing}' in value: {value!r}") from exc


def _merge_dict(*dicts: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = v
    return out


def _extract_named_values(text: str, expected_keys):
    expected = set(expected_keys)
    values = {}
    for m in re.finditer(r"(?P<key>[A-Za-z0-9_]+)=(?P<val>-?[0-9.]+)(?P<unit>ms|KB)?", text):
        key = m.group("key")
        if key in expected:
            values[key] = float(m.group("val"))
    return values


@dataclass(frozen=True)
class HostSpec:
    role: str
    host: str
    user: str = ""
    port: Optional[int] = None
    repo_dir: str = "/root/BloomBee"
    activate: str = ""
    ssh_options: Optional[List[str]] = None

    @property
    def address(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host


@dataclass
class BackgroundProc:
    role: str
    name: str
    pid_path: str


class SequencedSweepRunner:
    def __init__(self, cfg: Dict[str, Any], run_tag: str, local_root: Path, dry_run: bool = False):
        self.cfg = cfg
        self.run_tag = run_tag
        self.local_root = local_root
        self.dry_run = dry_run
        self._stop_requested = False
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    def _on_signal(self, signum, _frame):
        self._stop_requested = True
        print(f"[sweep] received signal {signum}, stopping after current cleanup", flush=True)

    def _sleep(self, sec: int):
        if sec <= 0:
            return
        if self.dry_run:
            print(f"[sweep] dry-run skip sleep {sec}s", flush=True)
            return
        time.sleep(sec)

    def _build_hosts(self) -> Dict[str, HostSpec]:
        global_defaults = self.cfg.get("host_defaults", {})
        hosts_cfg = self.cfg.get("hosts", {})
        required = ("dht", "server1", "server2", "client")
        missing = [x for x in required if x not in hosts_cfg]
        if missing:
            raise ValueError(f"Config missing required host roles: {missing}")

        hosts: Dict[str, HostSpec] = {}
        for role, raw in hosts_cfg.items():
            merged = _merge_dict(global_defaults, raw)
            port_val = merged.get("port")
            port = None if port_val in (None, "") else int(port_val)
            hosts[role] = HostSpec(
                role=role,
                host=str(merged["host"]),
                user=str(merged.get("user", "")),
                port=port,
                repo_dir=str(merged.get("repo_dir", "/root/BloomBee")),
                activate=str(merged.get("activate", "")),
                ssh_options=list(merged.get("ssh_options", [])) if merged.get("ssh_options") else None,
            )
        return hosts

    def _ssh_cmd(self, host: HostSpec, remote_cmd: str) -> List[str]:
        cmd = ["ssh"]
        if host.port is not None:
            cmd.extend(["-p", str(host.port)])
        if host.ssh_options:
            for opt in host.ssh_options:
                cmd.extend(["-o", opt])
        cmd.extend([host.address, "bash", "-lc", remote_cmd])
        return cmd

    def _run_cmd(self, cmd: List[str], *, check: bool = True, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        pretty = " ".join(shlex.quote(x) for x in cmd)
        print(f"[sweep] $ {pretty}", flush=True)
        if self.dry_run:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        result = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=timeout)
        if check and result.returncode != 0:
            raise RuntimeError(
                f"Command failed ({result.returncode}): {pretty}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result

    def _compose_role_cmd(self, host: HostSpec, role_env: Mapping[str, str], role_cmd: str) -> str:
        parts = []
        if host.activate:
            parts.append(host.activate)
        parts.append(f"cd {shlex.quote(host.repo_dir)}")
        if role_env:
            env_tokens = [f"{k}={shlex.quote(str(v))}" for k, v in role_env.items() if k]
            if env_tokens:
                parts.append("export " + " ".join(env_tokens))
        parts.append(role_cmd)
        return " && ".join(parts)

    def _start_background(
        self,
        host: HostSpec,
        role: str,
        name: str,
        role_env: Mapping[str, str],
        role_cmd: str,
        remote_log: str,
    ) -> BackgroundProc:
        pid_path = f"{remote_log}.pid"
        run_cmd = self._compose_role_cmd(host, role_env, role_cmd)
        remote = (
            "set -euo pipefail; "
            f"mkdir -p {shlex.quote(str(Path(remote_log).parent))}; "
            f"if [ -f {shlex.quote(pid_path)} ] && kill -0 \"$(cat {shlex.quote(pid_path)})\" 2>/dev/null; then "
            f"  kill \"$(cat {shlex.quote(pid_path)})\" 2>/dev/null || true; sleep 1; "
            "fi; "
            f"nohup setsid bash -lc {shlex.quote(run_cmd)} > {shlex.quote(remote_log)} 2>&1 < /dev/null & "
            f"echo $! > {shlex.quote(pid_path)}; "
            f"echo STARTED {shlex.quote(name)} PID=$(cat {shlex.quote(pid_path)})"
        )
        result = self._run_cmd(self._ssh_cmd(host, remote), check=True)
        if result.stdout.strip():
            print(result.stdout.strip(), flush=True)
        return BackgroundProc(role=role, name=name, pid_path=pid_path)

    def _stop_background(self, host: HostSpec, proc: BackgroundProc):
        remote = (
            "set -euo pipefail; "
            f"if [ ! -f {shlex.quote(proc.pid_path)} ]; then echo NO_PIDFILE {shlex.quote(proc.name)}; exit 0; fi; "
            f"PID=$(cat {shlex.quote(proc.pid_path)}); "
            "if kill -0 \"$PID\" 2>/dev/null; then kill \"$PID\" 2>/dev/null || true; sleep 2; "
            "if kill -0 \"$PID\" 2>/dev/null; then kill -9 \"$PID\" 2>/dev/null || true; fi; fi; "
            f"rm -f {shlex.quote(proc.pid_path)}; echo STOPPED {shlex.quote(proc.name)} PID=$PID"
        )
        result = self._run_cmd(self._ssh_cmd(host, remote), check=False)
        if result.stdout.strip():
            print(result.stdout.strip(), flush=True)
        if result.returncode != 0:
            print(
                f"[sweep] WARN stop failed role={proc.role} name={proc.name}: "
                f"{result.stderr.strip()}",
                flush=True,
            )

    def _run_client_foreground(
        self,
        host: HostSpec,
        role_env: Mapping[str, str],
        role_cmd: str,
        remote_log: str,
        timeout_sec: int,
    ):
        run_cmd = self._compose_role_cmd(host, role_env, role_cmd)
        remote = (
            "set -euo pipefail; "
            f"mkdir -p {shlex.quote(str(Path(remote_log).parent))}; "
            f"bash -lc {shlex.quote(run_cmd)} > {shlex.quote(remote_log)} 2>&1; "
            "echo CLIENT_DONE"
        )
        result = self._run_cmd(self._ssh_cmd(host, remote), check=True, timeout=timeout_sec)
        if result.stdout.strip():
            print(result.stdout.strip(), flush=True)

    def _run_role_setup(
        self,
        host: HostSpec,
        role_env: Mapping[str, str],
        setup_cmd: str,
        timeout_sec: int = 1800,
    ):
        run_cmd = self._compose_role_cmd(host, role_env, setup_cmd)
        remote = f"set -euo pipefail; bash -lc {shlex.quote(run_cmd)}"
        result = self._run_cmd(self._ssh_cmd(host, remote), check=True, timeout=timeout_sec)
        if result.stdout.strip():
            print(result.stdout.strip(), flush=True)

    def _scp_one(self, host: HostSpec, remote_file: str, local_file: Path) -> bool:
        local_file.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["scp"]
        if host.port is not None:
            cmd.extend(["-P", str(host.port)])
        if host.ssh_options:
            for opt in host.ssh_options:
                cmd.extend(["-o", opt])
        cmd.extend([f"{host.address}:{remote_file}", str(local_file)])
        result = self._run_cmd(cmd, check=False)
        if result.returncode != 0:
            print(
                f"[sweep] WARN failed to collect {remote_file} from {host.address}: {result.stderr.strip()}",
                flush=True,
            )
            return False
        return True

    @staticmethod
    def _build_env(default_env: Mapping[str, Any], case_env: Mapping[str, Any], role: str, context: Mapping[str, Any]) -> Dict[str, str]:
        def _render_section(blob: Mapping[str, Any], key: str) -> Dict[str, str]:
            sec = blob.get(key, {})
            if not isinstance(sec, Mapping):
                return {}
            out: Dict[str, str] = {}
            for k, v in sec.items():
                out[str(k)] = _render(str(v), context)
            return out

        return _merge_dict(
            _render_section(default_env, "all"),
            _render_section(default_env, role),
            _render_section(case_env, "all"),
            _render_section(case_env, role),
        )

    @staticmethod
    def _analyze_case_logs(case_dir: Path) -> Dict[str, float]:
        client_log = case_dir / "inference.log"
        server1_log = case_dir / "server1.log"
        server2_log = case_dir / "server2.log"

        step_network: Dict[str, List[Dict[str, float]]] = {}
        step_latencies: List[float] = []
        tx_ratios: List[float] = []
        rx_ratios: List[float] = []
        queue_waits: List[float] = []

        if client_log.exists():
            with client_log.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = NETWORK_SUMMARY_RE.search(line)
                    if m:
                        step_id = m.group("step_id")
                        step_network.setdefault(step_id, []).append(
                            {
                                "send_kb": float(m.group("send_kb")),
                                "recv_kb": float(m.group("recv_kb")),
                                "network_ms": float(m.group("network_ms")),
                            }
                        )
                        continue
                    m = STEP_LATENCY_RE.search(line)
                    if m:
                        step_latencies.append(float(m.group("latency_ms")))
                        continue
                    m = LOSSLESS_PROFILE_RE.search(line)
                    if m:
                        body_vals = _extract_named_values(m.group("body"), {"raw", "wire", "ratio"})
                        raw = body_vals.get("raw", 0.0)
                        wire = body_vals.get("wire", 0.0)
                        ratio = (wire / raw) if raw > 0 else body_vals.get("ratio", 1.0)
                        if m.group("dir") == "tx":
                            tx_ratios.append(ratio)
                        elif m.group("dir") == "rx":
                            rx_ratios.append(ratio)

        for log_path in (server1_log, server2_log):
            if not log_path.exists():
                continue
            with log_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = STEP_TIMING_BREAKDOWN_RE.search(line)
                    if m:
                        queue_waits.append(float(m.group("queue_wait_ms")))

        step_records: List[Dict[str, float]] = []
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
                    "send_total_kb": e0["send_kb"] + (e1["send_kb"] if e1 else 0.0),
                    "recv_total_kb": e0["recv_kb"] + (e1["recv_kb"] if e1 else 0.0),
                    "network_total_ms": e0["network_ms"] + (e1["network_ms"] if e1 else 0.0),
                }
            )

        return {
            "step_count": float(len(step_latencies)),
            "mean_step_latency_ms": _safe_mean(step_latencies),
            "mean_network_total_ms": _safe_mean([x["network_total_ms"] for x in step_records]),
            "mean_send_total_kb": _safe_mean([x["send_total_kb"] for x in step_records]),
            "mean_recv_total_kb": _safe_mean([x["recv_total_kb"] for x in step_records]),
            "mean_queue_wait_ms": _safe_mean(queue_waits),
            "mean_tx_wire_over_raw_ratio": _safe_mean(tx_ratios) if tx_ratios else 1.0,
            "mean_rx_wire_over_raw_ratio": _safe_mean(rx_ratios) if rx_ratios else 1.0,
        }

    @staticmethod
    def _write_case_summary(case_dir: Path, row: Mapping[str, Any]):
        lines = [
            f"case_name={row.get('case_name', '')}",
            f"case_tag={row.get('case_tag', '')}",
            f"batch_size={row.get('batch_size', '')}",
            f"micro_batch_size={row.get('micro_batch_size', '')}",
            f"compression={row.get('compression', '')}",
            f"offload={row.get('offload', '')}",
            f"policy_w_gpu_percent={row.get('policy_w_gpu_percent', '')}",
            f"policy_w_cpu_percent={row.get('policy_w_cpu_percent', '')}",
            f"policy_cache_gpu_percent={row.get('policy_cache_gpu_percent', '')}",
            f"policy_cache_cpu_percent={row.get('policy_cache_cpu_percent', '')}",
            f"policy_act_gpu_percent={row.get('policy_act_gpu_percent', '')}",
            f"policy_act_cpu_percent={row.get('policy_act_cpu_percent', '')}",
            f"eval_tokens={row.get('eval_tokens', '')}",
            "",
            f"mean_step_latency_ms={_fmt(float(row.get('mean_step_latency_ms', 0.0)))}",
            f"mean_network_total_ms={_fmt(float(row.get('mean_network_total_ms', 0.0)))}",
            f"mean_send_total_kb={_fmt(float(row.get('mean_send_total_kb', 0.0)))}",
            f"mean_recv_total_kb={_fmt(float(row.get('mean_recv_total_kb', 0.0)))}",
            f"mean_queue_wait_ms={_fmt(float(row.get('mean_queue_wait_ms', 0.0)))}",
            f"mean_tx_wire_over_raw_ratio={_fmt(float(row.get('mean_tx_wire_over_raw_ratio', 1.0)))}",
            f"mean_rx_wire_over_raw_ratio={_fmt(float(row.get('mean_rx_wire_over_raw_ratio', 1.0)))}",
            f"step_count={int(float(row.get('step_count', 0.0)))}",
        ]
        (case_dir / "analysis_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _write_matrix(local_root: Path, run_tag: str, rows: List[Dict[str, Any]]):
        out_dir = local_root / run_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "sweep_matrix.csv"
        txt_path = out_dir / "sweep_matrix.txt"

        fields = [
            "case_name",
            "case_tag",
            "batch_size",
            "micro_batch_size",
            "compression",
            "offload",
            "policy_w_gpu_percent",
            "policy_w_cpu_percent",
            "policy_cache_gpu_percent",
            "policy_cache_cpu_percent",
            "policy_act_gpu_percent",
            "policy_act_cpu_percent",
            "eval_tokens",
            "mean_step_latency_ms",
            "mean_network_total_ms",
            "mean_send_total_kb",
            "mean_recv_total_kb",
            "mean_queue_wait_ms",
            "mean_tx_wire_over_raw_ratio",
            "mean_rx_wire_over_raw_ratio",
            "step_count",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fields})

        lines: List[str] = []
        lines.append("=" * 120)
        lines.append(f"Sweep Matrix (run_tag={run_tag})")
        lines.append("=" * 120)
        lines.append(
            f"{'case_name':24} {'off':>4} {'cmp':>4} {'bs':>6} {'mb':>6} "
            f"{'c_gpu':>6} {'c_cpu':>6} {'eval':>6} "
            f"{'step_ms':>10} {'net_ms':>10} {'queue_ms':>10} {'send_kb':>10} {'recv_kb':>10}"
        )
        lines.append("-" * 120)
        for row in rows:
            lines.append(
                f"{str(row.get('case_name', ''))[:24]:24} "
                f"{int(row.get('offload', 0)):>4} {int(row.get('compression', 0)):>4} "
                f"{int(row.get('batch_size', 0)):>6} {int(row.get('micro_batch_size', 0)):>6} "
                f"{int(row.get('policy_cache_gpu_percent', 0)):>6} {int(row.get('policy_cache_cpu_percent', 0)):>6} "
                f"{int(row.get('eval_tokens', 0)):>6} "
                f"{_fmt(float(row.get('mean_step_latency_ms', 0.0))):>10} "
                f"{_fmt(float(row.get('mean_network_total_ms', 0.0))):>10} "
                f"{_fmt(float(row.get('mean_queue_wait_ms', 0.0))):>10} "
                f"{_fmt(float(row.get('mean_send_total_kb', 0.0))):>10} "
                f"{_fmt(float(row.get('mean_recv_total_kb', 0.0))):>10}"
            )
        lines.append("-" * 120)

        if rows:
            base = rows[0]
            lines.append(f"Baseline: {base.get('case_name', '')}")
            lines.append(
                f"{'case_name':24} {'d_step%':>10} {'d_net%':>10} {'d_queue%':>10} {'d_send%':>10} {'d_recv%':>10}"
            )
            for row in rows:
                lines.append(
                    f"{str(row.get('case_name', ''))[:24]:24} "
                    f"{_fmt(_pct_delta(float(row.get('mean_step_latency_ms', 0.0)), float(base.get('mean_step_latency_ms', 0.0)))):>10} "
                    f"{_fmt(_pct_delta(float(row.get('mean_network_total_ms', 0.0)), float(base.get('mean_network_total_ms', 0.0)))):>10} "
                    f"{_fmt(_pct_delta(float(row.get('mean_queue_wait_ms', 0.0)), float(base.get('mean_queue_wait_ms', 0.0)))):>10} "
                    f"{_fmt(_pct_delta(float(row.get('mean_send_total_kb', 0.0)), float(base.get('mean_send_total_kb', 0.0)))):>10} "
                    f"{_fmt(_pct_delta(float(row.get('mean_recv_total_kb', 0.0)), float(base.get('mean_recv_total_kb', 0.0)))):>10}"
                )
            lines.append("Interpretation: negative d_step%/d_net%/d_queue% is better.")

        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[sweep] matrix written: {csv_path}", flush=True)
        print(f"[sweep] matrix written: {txt_path}", flush=True)

    def run(self):
        hosts = self._build_hosts()
        commands = self.cfg.get("commands", {})
        for role in ("dht", "server1", "server2", "client"):
            if role not in commands:
                raise ValueError(f"Config missing commands.{role}")

        schedule = self.cfg.get("schedule", {})
        wait_after_dht_sec = int(schedule.get("wait_after_dht_sec", 30))
        wait_after_server1_sec = int(schedule.get("wait_after_server1_sec", 30))
        wait_after_server2_sec = int(schedule.get("wait_after_server2_sec", 30))
        inter_case_sec = int(schedule.get("inter_case_sec", 5))
        teardown_wait_sec = int(schedule.get("teardown_wait_sec", 3))
        client_timeout_sec = int(schedule.get("client_timeout_sec", 4 * 3600))
        results_root = str(schedule.get("results_root", "/tmp/bb_sensitivity_runs"))
        reinstall_each_case = bool(schedule.get("reinstall_editable_each_case", False))
        reinstall_roles = list(schedule.get("reinstall_roles", ["server1", "server2", "client"]))
        reinstall_cmd = str(schedule.get("reinstall_cmd", "pip install -e ."))
        reinstall_timeout_sec = int(schedule.get("reinstall_timeout_sec", 3600))

        shared_vars = dict(self.cfg.get("shared_vars", {}))
        default_env = dict(self.cfg.get("default_env", {}))
        cases = list(self.cfg.get("cases", []))
        if not cases:
            raise ValueError("Config has no cases")

        self.local_root.mkdir(parents=True, exist_ok=True)
        run_dir = self.local_root / self.run_tag
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config_used.json").write_text(json.dumps(self.cfg, indent=2), encoding="utf-8")

        print(
            f"[sweep] run_tag={self.run_tag} cases={len(cases)} "
            f"stage_waits={wait_after_dht_sec}/{wait_after_server1_sec}/{wait_after_server2_sec}s "
            f"results_root={results_root} local_root={self.local_root} dry_run={self.dry_run}",
            flush=True,
        )

        rows: List[Dict[str, Any]] = []
        for idx, case in enumerate(cases):
            if self._stop_requested:
                break

            case_name = str(case.get("name", f"case_{idx:03d}"))
            case_tag = f"{self.run_tag}_{idx:03d}_{_safe_name(case_name)}"
            remote_case_root = f"{results_root}/{case_tag}"
            local_case_dir = self.local_root / case_tag
            local_case_dir.mkdir(parents=True, exist_ok=True)

            case_vars = _merge_dict(shared_vars, dict(case.get("vars", {})))
            case_vars["case_index"] = idx
            case_vars["case_name"] = case_name
            case_vars["case_tag"] = case_tag
            case_vars["run_tag"] = self.run_tag
            case_vars["eval_tokens"] = int(case_vars.get("eval_tokens", 200))
            case_vars["seq_len"] = int(case_vars.get("seq_len", 512))
            case_vars["batch_size"] = int(case_vars.get("batch_size", 1))
            case_vars["micro_batch_size"] = int(case_vars.get("micro_batch_size", 0))
            case_vars["lossless_wrapper"] = int(case_vars.get("lossless_wrapper", 0))
            case_vars["offload"] = int(case_vars.get("offload", 1))
            case_vars["policy_w_gpu_percent"] = int(case_vars.get("policy_w_gpu_percent", 100))
            case_vars["policy_w_cpu_percent"] = int(case_vars.get("policy_w_cpu_percent", 0))
            case_vars["policy_cache_gpu_percent"] = int(case_vars.get("policy_cache_gpu_percent", 100))
            case_vars["policy_cache_cpu_percent"] = int(case_vars.get("policy_cache_cpu_percent", 0))
            case_vars["policy_act_gpu_percent"] = int(case_vars.get("policy_act_gpu_percent", 100))
            case_vars["policy_act_cpu_percent"] = int(case_vars.get("policy_act_cpu_percent", 0))

            case_env = dict(case.get("env", {}))
            dht_env = self._build_env(default_env, case_env, "dht", case_vars)
            s1_env = self._build_env(default_env, case_env, "server1", case_vars)
            s2_env = self._build_env(default_env, case_env, "server2", case_vars)
            cli_env = self._build_env(default_env, case_env, "client", case_vars)

            dht_cmd = _render(commands["dht"], case_vars)
            s1_cmd = _render(commands["server1"], case_vars)
            s2_cmd = _render(commands["server2"], case_vars)
            cli_cmd = _render(commands["client"], case_vars)

            active: List[BackgroundProc] = []
            case_error: Optional[str] = None

            print("=" * 96, flush=True)
            print(
                f"[sweep] CASE {idx + 1}/{len(cases)} name={case_name} "
                f"bs={case_vars['batch_size']} mb={case_vars['micro_batch_size']} "
                f"cmp={case_vars['lossless_wrapper']} offload={case_vars['offload']} eval={case_vars['eval_tokens']}",
                flush=True,
            )
            print("=" * 96, flush=True)

            try:
                if reinstall_each_case:
                    seen_targets = set()
                    for role in reinstall_roles:
                        if role not in hosts:
                            continue
                        host = hosts[role]
                        target_key = (host.address, host.port, host.repo_dir)
                        if target_key in seen_targets:
                            continue
                        seen_targets.add(target_key)
                        if role == "dht":
                            role_env = dht_env
                        elif role == "server1":
                            role_env = s1_env
                        elif role == "server2":
                            role_env = s2_env
                        elif role == "client":
                            role_env = cli_env
                        else:
                            role_env = {}
                        print(f"[sweep] reinstall editable package on role={role} host={host.address}", flush=True)
                        self._run_role_setup(
                            host=host,
                            role_env=role_env,
                            setup_cmd=reinstall_cmd,
                            timeout_sec=reinstall_timeout_sec,
                        )

                active.append(
                    self._start_background(
                        hosts["dht"], "dht", f"{case_tag}_dht", dht_env, dht_cmd, f"{remote_case_root}/dht.log"
                    )
                )
                self._sleep(wait_after_dht_sec)

                active.append(
                    self._start_background(
                        hosts["server1"],
                        "server1",
                        f"{case_tag}_server1",
                        s1_env,
                        s1_cmd,
                        f"{remote_case_root}/server1.log",
                    )
                )
                self._sleep(wait_after_server1_sec)

                active.append(
                    self._start_background(
                        hosts["server2"],
                        "server2",
                        f"{case_tag}_server2",
                        s2_env,
                        s2_cmd,
                        f"{remote_case_root}/server2.log",
                    )
                )
                self._sleep(wait_after_server2_sec)

                self._run_client_foreground(
                    hosts["client"],
                    cli_env,
                    cli_cmd,
                    f"{remote_case_root}/inference.log",
                    timeout_sec=client_timeout_sec,
                )

            except Exception as exc:
                case_error = str(exc)
                print(f"[sweep] ERROR case={case_name}: {exc}", flush=True)
            finally:
                # Kill server2/server1/dht in reverse startup order.
                for proc in reversed(active):
                    self._stop_background(hosts[proc.role], proc)
                self._sleep(teardown_wait_sec)

            # Collect logs to local (analysis input and provenance).
            dht_ok = self._scp_one(hosts["dht"], f"{remote_case_root}/dht.log", local_case_dir / "dht.log")
            s1_ok = self._scp_one(hosts["server1"], f"{remote_case_root}/server1.log", local_case_dir / "server1.log")
            s2_ok = self._scp_one(hosts["server2"], f"{remote_case_root}/server2.log", local_case_dir / "server2.log")
            cli_ok = self._scp_one(hosts["client"], f"{remote_case_root}/inference.log", local_case_dir / "inference.log")

            metrics = {}
            if s1_ok and s2_ok and cli_ok and not self.dry_run:
                metrics = self._analyze_case_logs(local_case_dir)
            else:
                metrics = {
                    "step_count": 0.0,
                    "mean_step_latency_ms": 0.0,
                    "mean_network_total_ms": 0.0,
                    "mean_send_total_kb": 0.0,
                    "mean_recv_total_kb": 0.0,
                    "mean_queue_wait_ms": 0.0,
                    "mean_tx_wire_over_raw_ratio": 1.0,
                    "mean_rx_wire_over_raw_ratio": 1.0,
                }

            row = {
                "case_name": case_name,
                "case_tag": case_tag,
                "batch_size": case_vars["batch_size"],
                "micro_batch_size": case_vars["micro_batch_size"],
                "compression": case_vars["lossless_wrapper"],
                "offload": case_vars["offload"],
                "policy_w_gpu_percent": case_vars["policy_w_gpu_percent"],
                "policy_w_cpu_percent": case_vars["policy_w_cpu_percent"],
                "policy_cache_gpu_percent": case_vars["policy_cache_gpu_percent"],
                "policy_cache_cpu_percent": case_vars["policy_cache_cpu_percent"],
                "policy_act_gpu_percent": case_vars["policy_act_gpu_percent"],
                "policy_act_cpu_percent": case_vars["policy_act_cpu_percent"],
                "eval_tokens": case_vars["eval_tokens"],
                "error": case_error or "",
                "collected_dht_log": int(dht_ok),
                "collected_server1_log": int(s1_ok),
                "collected_server2_log": int(s2_ok),
                "collected_client_log": int(cli_ok),
            }
            row.update(metrics)
            rows.append(row)

            self._write_case_summary(local_case_dir, row)
            print(
                f"[sweep] case done: {case_name} | step={_fmt(float(row['mean_step_latency_ms']))}ms "
                f"net={_fmt(float(row['mean_network_total_ms']))}ms "
                f"queue={_fmt(float(row['mean_queue_wait_ms']))}ms",
                flush=True,
            )

            if idx < len(cases) - 1 and not self._stop_requested:
                self._sleep(inter_case_sec)

        self._write_matrix(self.local_root, self.run_tag, rows)
        print("[sweep] all done.", flush=True)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run sequenced BloomBee trade-off sweep with auto collect + per-case analysis."
    )
    parser.add_argument("--config", type=Path, required=True, help="JSON config path")
    parser.add_argument("--run-tag", type=str, default="", help="Run tag; default from config.schedule.run_tag or UTC")
    parser.add_argument("--local-root", type=Path, default=Path("./collected_runs"), help="Where to store pulled logs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_tag = args.run_tag or str(cfg.get("schedule", {}).get("run_tag", "")).strip() or _utc_now_tag()
    runner = SequencedSweepRunner(cfg=cfg, run_tag=run_tag, local_root=args.local_root, dry_run=args.dry_run)
    runner.run()


if __name__ == "__main__":
    main()
