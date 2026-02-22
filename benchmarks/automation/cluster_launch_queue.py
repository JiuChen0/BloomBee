#!/usr/bin/env python3
"""
Queue and launch BloomBee experiments across 3 independent servers via SSH.

Typical topology:
  - host role "dht"     : runs run_dht
  - host role "server1" : runs stage-1 run_server
  - host role "server2" : runs stage-2 run_server
  - host role "client"  : runs benchmark_inference (can be same machine as server2)

One controller machine executes this script and coordinates all remote processes.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _utc_now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_name(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in s)


def _render(value: str, context: Mapping[str, Any]) -> str:
    try:
        return str(value).format(**context)
    except KeyError as exc:
        missing = str(exc).strip("'")
        raise ValueError(f"Missing template variable '{missing}' in value: {value!r}") from exc


def _merge_dict(*dicts: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for d in dicts:
        for k, v in d.items():
            merged[k] = v
    return merged


def _env_export_cmd(env_map: Mapping[str, str]) -> str:
    if not env_map:
        return ""
    parts = []
    for k, v in env_map.items():
        if not k:
            continue
        parts.append(f"{k}={shlex.quote(str(v))}")
    if not parts:
        return ""
    return "export " + " ".join(parts)


@dataclass(frozen=True)
class HostSpec:
    role: str
    host: str
    user: str = ""
    port: Optional[int] = None
    repo_dir: str = "/home/cc/BloomBee"
    activate: str = ""
    ssh_options: Optional[List[str]] = None

    @property
    def address(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host


class ClusterLauncher:
    def __init__(self, cfg: Dict[str, Any], dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        self.active_bg: List[Dict[str, str]] = []
        self._stop_requested = False

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, _frame):
        self._stop_requested = True
        print(f"[cluster-launch] Received signal {signum}, stopping and cleaning up ...", flush=True)

    def _sleep(self, seconds: int):
        if seconds <= 0:
            return
        if self.dry_run:
            print(f"[cluster-launch] dry-run skip sleep {seconds}s", flush=True)
            return
        time.sleep(seconds)

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
            port: Optional[int]
            if port_val in (None, ""):
                port = None
            else:
                port = int(port_val)
            hosts[role] = HostSpec(
                role=role,
                host=str(merged["host"]),
                user=str(merged.get("user", "")),
                port=port,
                repo_dir=str(merged.get("repo_dir", "/home/cc/BloomBee")),
                activate=str(merged.get("activate", "")),
                ssh_options=list(merged.get("ssh_options", [])) if merged.get("ssh_options") else None,
            )
        return hosts

    def _ssh_run(self, host: HostSpec, remote_cmd: str, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        cmd = ["ssh"]
        if host.port is not None:
            cmd.extend(["-p", str(host.port)])
        if host.ssh_options:
            for opt in host.ssh_options:
                cmd.extend(["-o", opt])
        cmd.extend([host.address, "bash", "-lc", remote_cmd])

        print(f"[cluster-launch] {host.role}: {remote_cmd}", flush=True)
        if self.dry_run:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)

    def _compose_job_cmd(self, host: HostSpec, role_env: Mapping[str, str], role_cmd: str) -> str:
        steps = []
        if host.activate:
            steps.append(host.activate)
        steps.append(f"cd {shlex.quote(host.repo_dir)}")
        env_export = _env_export_cmd(role_env)
        if env_export:
            steps.append(env_export)
        steps.append(role_cmd)
        return " && ".join(steps)

    def _start_background(self, host: HostSpec, name: str, role_env: Mapping[str, str], role_cmd: str, log_path: str):
        pid_path = f"{log_path}.pid"
        run_cmd = self._compose_job_cmd(host, role_env, role_cmd)
        remote = (
            "set -euo pipefail; "
            f"mkdir -p {shlex.quote(str(Path(log_path).parent))}; "
            f"if [ -f {shlex.quote(pid_path)} ] && kill -0 \"$(cat {shlex.quote(pid_path)})\" 2>/dev/null; then "
            f"  kill \"$(cat {shlex.quote(pid_path)})\" 2>/dev/null || true; "
            "  sleep 1; "
            "fi; "
            f"nohup setsid bash -lc {shlex.quote(run_cmd)} > {shlex.quote(log_path)} 2>&1 < /dev/null & "
            f"echo $! > {shlex.quote(pid_path)}; "
            f"echo STARTED {shlex.quote(name)} PID=$(cat {shlex.quote(pid_path)}) LOG={shlex.quote(log_path)}"
        )
        result = self._ssh_run(host, remote)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start background job {name} on {host.role}:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        if result.stdout.strip():
            print(result.stdout.strip(), flush=True)
        self.active_bg.append(
            {
                "host_role": host.role,
                "name": name,
                "log_path": log_path,
                "pid_path": pid_path,
            }
        )

    def _stop_background(self, hosts: Dict[str, HostSpec], host_role: str, name: str, pid_path: str):
        host = hosts[host_role]
        remote = (
            "set -euo pipefail; "
            f"if [ ! -f {shlex.quote(pid_path)} ]; then "
            f"  echo NO_PIDFILE {shlex.quote(name)}; "
            "  exit 0; "
            "fi; "
            f"PID=$(cat {shlex.quote(pid_path)}); "
            "if kill -0 \"$PID\" 2>/dev/null; then "
            "  kill \"$PID\" 2>/dev/null || true; "
            "  sleep 2; "
            "  if kill -0 \"$PID\" 2>/dev/null; then kill -9 \"$PID\" 2>/dev/null || true; fi; "
            "fi; "
            f"rm -f {shlex.quote(pid_path)}; "
            f"echo STOPPED {shlex.quote(name)} PID=$PID"
        )
        result = self._ssh_run(host, remote)
        if result.returncode == 0 and result.stdout.strip():
            print(result.stdout.strip(), flush=True)
        elif result.returncode != 0:
            print(
                f"[cluster-launch] WARN failed to stop {name} on {host_role}: "
                f"stdout={result.stdout.strip()} stderr={result.stderr.strip()}",
                flush=True,
            )

    def _stop_all_background(self, hosts: Dict[str, HostSpec]):
        # Stop in reverse launch order.
        for rec in reversed(self.active_bg):
            self._stop_background(
                hosts=hosts,
                host_role=rec["host_role"],
                name=rec["name"],
                pid_path=rec["pid_path"],
            )
        self.active_bg = []

    def _run_client(self, host: HostSpec, role_env: Mapping[str, str], role_cmd: str, log_path: str, timeout_sec: int):
        run_cmd = self._compose_job_cmd(host, role_env, role_cmd)
        remote = (
            "set -euo pipefail; "
            f"mkdir -p {shlex.quote(str(Path(log_path).parent))}; "
            f"bash -lc {shlex.quote(run_cmd)} > {shlex.quote(log_path)} 2>&1; "
            f"echo CLIENT_DONE LOG={shlex.quote(log_path)}"
        )
        result = self._ssh_run(host, remote, timeout=timeout_sec)
        if result.returncode != 0:
            raise RuntimeError(
                f"Client benchmark failed on {host.role}:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        if result.stdout.strip():
            print(result.stdout.strip(), flush=True)

    def run(self):
        hosts = self._build_hosts()
        schedule = self.cfg.get("schedule", {})
        commands = self.cfg.get("commands", {})
        cases = self.cfg.get("cases", [])
        if not cases:
            raise ValueError("Config has no cases.")
        for role in ("dht", "server1", "server2", "client"):
            if role not in commands:
                raise ValueError(f"Config missing commands.{role}")

        interval_sec = int(schedule.get("interval_sec", 1800))
        startup_wait_sec = int(schedule.get("startup_wait_sec", 60))
        teardown_wait_sec = int(schedule.get("teardown_wait_sec", 3))
        client_timeout_sec = int(schedule.get("client_timeout_sec", 4 * 3600))
        reuse_dht = bool(schedule.get("reuse_dht", True))
        run_tag = str(schedule.get("run_tag", _utc_now_tag()))
        results_root = str(schedule.get("results_root", "/home/cc/bb_sweep_runs"))

        shared_vars = dict(self.cfg.get("shared_vars", {}))
        default_env = dict(self.cfg.get("default_env", {}))

        print(
            f"[cluster-launch] run_tag={run_tag} cases={len(cases)} interval_sec={interval_sec} "
            f"startup_wait_sec={startup_wait_sec} reuse_dht={reuse_dht} dry_run={self.dry_run}",
            flush=True,
        )

        try:
            if reuse_dht:
                dht_vars = dict(shared_vars)
                dht_cmd = _render(commands["dht"], dht_vars)
                dht_env = self._build_role_env(default_env, {}, "dht", dht_vars)
                dht_log = f"{results_root}/{run_tag}/shared_dht.log"
                self._start_background(hosts["dht"], f"{run_tag}_dht_shared", dht_env, dht_cmd, dht_log)
                print("[cluster-launch] Shared DHT started.", flush=True)
                self._sleep(startup_wait_sec)

            for idx, case in enumerate(cases):
                if self._stop_requested:
                    break

                case_name = str(case.get("name", f"case_{idx:03d}"))
                case_tag = f"{run_tag}_{idx:03d}_{_safe_name(case_name)}"
                case_root = f"{results_root}/{case_tag}"

                case_vars = _merge_dict(shared_vars, dict(case.get("vars", {})))
                case_vars["case_index"] = idx
                case_vars["case_name"] = case_name
                case_vars["case_tag"] = case_tag
                case_vars["case_root"] = case_root

                print("=" * 88, flush=True)
                print(
                    f"[cluster-launch] START case {idx + 1}/{len(cases)}: {case_name} "
                    f"(tag={case_tag})",
                    flush=True,
                )
                print("=" * 88, flush=True)

                if not reuse_dht:
                    dht_cmd = _render(commands["dht"], case_vars)
                    dht_env = self._build_role_env(default_env, dict(case.get("env", {})), "dht", case_vars)
                    dht_log = f"{case_root}/dht.log"
                    self._start_background(hosts["dht"], f"{case_tag}_dht", dht_env, dht_cmd, dht_log)
                    self._sleep(startup_wait_sec)

                s1_cmd = _render(commands["server1"], case_vars)
                s2_cmd = _render(commands["server2"], case_vars)
                cli_cmd = _render(commands["client"], case_vars)

                case_env = dict(case.get("env", {}))
                s1_env = self._build_role_env(default_env, case_env, "server1", case_vars)
                s2_env = self._build_role_env(default_env, case_env, "server2", case_vars)
                cli_env = self._build_role_env(default_env, case_env, "client", case_vars)

                self._start_background(hosts["server1"], f"{case_tag}_server1", s1_env, s1_cmd, f"{case_root}/server1.log")
                self._start_background(hosts["server2"], f"{case_tag}_server2", s2_env, s2_cmd, f"{case_root}/server2.log")

                if startup_wait_sec > 0:
                    print(f"[cluster-launch] waiting {startup_wait_sec}s for servers to warm up ...", flush=True)
                self._sleep(startup_wait_sec)

                self._run_client(
                    hosts["client"],
                    cli_env,
                    cli_cmd,
                    f"{case_root}/inference.log",
                    timeout_sec=client_timeout_sec,
                )

                print(f"[cluster-launch] case {case_name} finished; stopping servers ...", flush=True)

                # Stop non-shared jobs created for this case only.
                current_bg = list(self.active_bg)
                self.active_bg = []
                for rec in reversed(current_bg):
                    keep_shared_dht = reuse_dht and rec["name"].endswith("_dht_shared")
                    if keep_shared_dht:
                        self.active_bg.append(rec)
                        continue
                    self._stop_background(
                        hosts=hosts,
                        host_role=rec["host_role"],
                        name=rec["name"],
                        pid_path=rec["pid_path"],
                    )

                self._sleep(teardown_wait_sec)

                if idx < len(cases) - 1 and interval_sec > 0 and not self._stop_requested:
                    print(f"[cluster-launch] sleeping {interval_sec}s before next case ...", flush=True)
                    self._sleep(interval_sec)

        finally:
            self._stop_all_background(hosts)
            print("[cluster-launch] Cleanup completed.", flush=True)

    @staticmethod
    def _build_role_env(
        default_env: Mapping[str, Any],
        case_env: Mapping[str, Any],
        role: str,
        context: Mapping[str, Any],
    ) -> Dict[str, str]:
        def _get_map(blob: Mapping[str, Any], key: str) -> Dict[str, str]:
            val = blob.get(key, {})
            if not isinstance(val, Mapping):
                return {}
            out: Dict[str, str] = {}
            for k, v in val.items():
                out[str(k)] = _render(str(v), context)
            return out

        merged = _merge_dict(
            _get_map(default_env, "all"),
            _get_map(default_env, role),
            _get_map(case_env, "all"),
            _get_map(case_env, role),
        )
        return merged


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Launch queued BloomBee experiments across multiple servers.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON config. See benchmarks/automation/cluster_launch_config.example.json",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print remote commands without executing them.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    launcher = ClusterLauncher(cfg=cfg, dry_run=args.dry_run)
    launcher.run()


if __name__ == "__main__":
    main()
