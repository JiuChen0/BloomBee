#!/usr/bin/env python3
"""
Collect logs from multi-host BloomBee runs and run local analysis automatically.

Typical usage (Windows or Linux controller):
  python benchmarks/automation/collect_and_analyze.py \
    --config benchmarks/automation/cluster_launch_config.windows_aliases.example.json \
    --run-tag weekly_sensitivity
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def _safe_name(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in s)


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


def _parse_int_like(v: Any, default: int) -> int:
    try:
        return int(str(v))
    except Exception:
        return default


@dataclass(frozen=True)
class HostSpec:
    role: str
    host: str
    user: str = ""
    port: Optional[int] = None
    ssh_options: Optional[List[str]] = None

    @property
    def address(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host


class Collector:
    def __init__(
        self,
        cfg: Dict[str, Any],
        run_tag: str,
        local_root: Path,
        dry_run: bool = False,
        continue_on_missing: bool = True,
        skip_analysis: bool = False,
        analyze_root: Optional[Path] = None,
    ):
        self.cfg = cfg
        self.run_tag = run_tag
        self.local_root = local_root
        self.dry_run = dry_run
        self.continue_on_missing = continue_on_missing
        self.skip_analysis = skip_analysis
        self.analyze_root = analyze_root

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
            if port_val in (None, ""):
                port = None
            else:
                port = int(port_val)
            hosts[role] = HostSpec(
                role=role,
                host=str(merged["host"]),
                user=str(merged.get("user", "")),
                port=port,
                ssh_options=list(merged.get("ssh_options", [])) if merged.get("ssh_options") else None,
            )
        return hosts

    def _run(self, cmd: List[str], *, check: bool = True) -> subprocess.CompletedProcess:
        pretty = " ".join(shlex.quote(x) for x in cmd)
        print(f"[collect] $ {pretty}", flush=True)
        if self.dry_run:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if check and result.returncode != 0:
            raise RuntimeError(
                f"Command failed ({result.returncode}): {pretty}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result

    def _scp_copy(self, host: HostSpec, remote_path: str, local_path: Path) -> bool:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["scp"]
        if host.port is not None:
            cmd.extend(["-P", str(host.port)])
        if host.ssh_options:
            for opt in host.ssh_options:
                cmd.extend(["-o", opt])
        cmd.extend([f"{host.address}:{remote_path}", str(local_path)])
        result = self._run(cmd, check=False)
        ok = result.returncode == 0
        if not ok:
            msg = (
                f"[collect] WARN missing/unreadable remote file: {host.address}:{remote_path}\n"
                f"stdout: {result.stdout.strip()}\n"
                f"stderr: {result.stderr.strip()}"
            )
            if self.continue_on_missing:
                print(msg, flush=True)
            else:
                raise RuntimeError(msg)
        return ok

    def _run_python(self, args: List[str], output_file: Path) -> bool:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable] + args
        result = self._run(cmd, check=False)
        if result.returncode != 0:
            output_file.write_text(
                f"[FAILED] returncode={result.returncode}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n",
                encoding="utf-8",
            )
            return False
        output_file.write_text(result.stdout, encoding="utf-8")
        return True

    def _build_case_plan(self, results_root: str) -> List[Dict[str, Any]]:
        cases = self.cfg.get("cases", [])
        shared_vars = dict(self.cfg.get("shared_vars", {}))
        default_env = dict(self.cfg.get("default_env", {}))
        plan: List[Dict[str, Any]] = []
        for idx, case in enumerate(cases):
            case_name = str(case.get("name", f"case_{idx:03d}"))
            case_tag = f"{self.run_tag}_{idx:03d}_{_safe_name(case_name)}"
            case_vars = _merge_dict(shared_vars, dict(case.get("vars", {})))
            case_vars["case_index"] = idx
            case_vars["case_name"] = case_name
            case_vars["case_tag"] = case_tag
            case_vars["run_tag"] = self.run_tag
            remote_case_root = f"{results_root}/{case_tag}"

            # Derive comparison metadata for analyze_tradeoff_matrix.py
            compression = self._resolve_compression_flag(default_env, case, case_vars)
            offload = _parse_int_like(case_vars.get("offload", 1), 1)
            batch = _parse_int_like(case_vars.get("batch_size", 0), 0)
            seq = _parse_int_like(case_vars.get("seq_len", 0), 0)

            plan.append(
                {
                    "index": idx,
                    "name": case_name,
                    "tag": case_tag,
                    "case_vars": case_vars,
                    "remote_case_root": remote_case_root,
                    "compression": compression,
                    "offload": offload,
                    "batch": batch,
                    "seq": seq,
                }
            )
        return plan

    @staticmethod
    def _resolve_compression_flag(default_env: Mapping[str, Any], case: Mapping[str, Any], context: Mapping[str, Any]) -> int:
        case_env = case.get("env", {})
        def _env_lookup(blob: Mapping[str, Any], section: str, key: str) -> Optional[str]:
            sec = blob.get(section, {})
            if not isinstance(sec, Mapping):
                return None
            val = sec.get(key)
            if val is None:
                return None
            return _render(str(val), context)

        for blob in (case_env, default_env):
            for sec in ("all",):
                v = _env_lookup(blob, sec, "BLOOMBEE_LOSSLESS_WRAPPER")
                if v is not None:
                    return _parse_int_like(v, 0)
        if "lossless_wrapper" in context:
            return _parse_int_like(context.get("lossless_wrapper"), 0)
        return 0

    def run(self):
        hosts = self._build_hosts()
        schedule = self.cfg.get("schedule", {})
        results_root = str(schedule.get("results_root", "/tmp/bb_sensitivity_runs"))
        reuse_dht = bool(schedule.get("reuse_dht", True))
        case_plan = self._build_case_plan(results_root)
        self.local_root.mkdir(parents=True, exist_ok=True)

        print(
            f"[collect] run_tag={self.run_tag} cases={len(case_plan)} results_root={results_root} "
            f"local_root={self.local_root}",
            flush=True,
        )

        # Pull shared/per-case dht logs (optional).
        if reuse_dht:
            remote = f"{results_root}/{self.run_tag}/shared_dht.log"
            local = self.local_root / self.run_tag / "shared_dht.log"
            self._scp_copy(hosts["dht"], remote, local)
        else:
            for item in case_plan:
                remote = f"{item['remote_case_root']}/dht.log"
                local = self.local_root / item["tag"] / "dht.log"
                self._scp_copy(hosts["dht"], remote, local)

        collected_cases: List[Dict[str, Any]] = []
        for item in case_plan:
            local_case_dir = self.local_root / item["tag"]
            local_case_dir.mkdir(parents=True, exist_ok=True)

            ok_s1 = self._scp_copy(
                hosts["server1"],
                f"{item['remote_case_root']}/server1.log",
                local_case_dir / "server1.log",
            )
            ok_s2 = self._scp_copy(
                hosts["server2"],
                f"{item['remote_case_root']}/server2.log",
                local_case_dir / "server2.log",
            )
            ok_cli = self._scp_copy(
                hosts["client"],
                f"{item['remote_case_root']}/inference.log",
                local_case_dir / "inference.log",
            )

            if ok_s1 and ok_s2 and ok_cli:
                collected_cases.append(item)

        # Optional local analysis (requires analyze scripts to exist on controller machine).
        if self.analyze_root is not None:
            script_root = self.analyze_root.resolve()
        else:
            script_root = Path(__file__).resolve().parents[2]
        analyze_breakdown = script_root / "benchmarks" / "analyze_latency_breakdown.py"
        analyze_tradeoff = script_root / "benchmarks" / "analyze_tradeoff_matrix.py"
        can_analyze = (
            (not self.skip_analysis)
            and analyze_breakdown.exists()
            and analyze_tradeoff.exists()
        )
        if not can_analyze:
            reason_parts = []
            if self.skip_analysis:
                reason_parts.append("skip-analysis enabled")
            if not analyze_breakdown.exists():
                reason_parts.append(f"missing {analyze_breakdown}")
            if not analyze_tradeoff.exists():
                reason_parts.append(f"missing {analyze_tradeoff}")
            reason = ", ".join(reason_parts) if reason_parts else "unknown reason"
            print(f"[collect] analysis skipped: {reason}", flush=True)
            print(f"[collect] logs collected under: {self.local_root}", flush=True)
            print("[collect] done.", flush=True)
            return

        for item in collected_cases:
            case_dir = self.local_root / item["tag"]
            ok = self._run_python(
                [
                    str(analyze_breakdown),
                    "--client-log",
                    str(case_dir / "inference.log"),
                    "--server1-log",
                    str(case_dir / "server1.log"),
                    "--server2-log",
                    str(case_dir / "server2.log"),
                ],
                output_file=case_dir / "latency_breakdown.txt",
            )
            if not ok:
                print(f"[collect] WARN analysis failed for case={item['name']}", flush=True)

        # Run global trade-off matrix if >=1 case is collected.
        if collected_cases:
            args = [str(analyze_tradeoff)]
            for item in collected_cases:
                case_dir = self.local_root / item["tag"]
                args.extend(
                    [
                        "--case",
                        (
                            f"name={item['name']},"
                            f"client={case_dir / 'inference.log'},"
                            f"server1={case_dir / 'server1.log'},"
                            f"server2={case_dir / 'server2.log'},"
                            f"compression={item['compression']},"
                            f"offload={item['offload']},"
                            f"batch={item['batch']},"
                            f"seq={item['seq']}"
                        ),
                    ]
                )
            args.extend(
                [
                    "--baseline",
                    collected_cases[0]["name"],
                    "--output-csv",
                    str(self.local_root / self.run_tag / "tradeoff_matrix.csv"),
                ]
            )
            self._run_python(args, output_file=self.local_root / self.run_tag / "tradeoff_matrix.txt")
            print(f"[collect] trade-off outputs: {self.local_root / self.run_tag}", flush=True)
        else:
            print("[collect] No complete cases collected; skip trade-off matrix.", flush=True)

        print("[collect] done.", flush=True)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Collect multi-host benchmark logs and run local analysis.")
    parser.add_argument("--config", type=Path, required=True, help="Cluster config JSON used for launch.")
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Run tag to collect, defaults to schedule.run_tag in config.",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path("./collected_runs"),
        help="Local directory for downloaded logs and generated reports.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Only collect logs; do not run analyze_latency_breakdown.py / analyze_tradeoff_matrix.py.",
    )
    parser.add_argument(
        "--analyze-root",
        type=Path,
        default=None,
        help="Repo root containing benchmarks/analyze_latency_breakdown.py and analyze_tradeoff_matrix.py.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately if any remote log is missing (default: continue with warnings).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_tag = args.run_tag or str(cfg.get("schedule", {}).get("run_tag", "")).strip()
    if not run_tag:
        raise ValueError("run_tag is empty; pass --run-tag or set schedule.run_tag in config")

    collector = Collector(
        cfg=cfg,
        run_tag=run_tag,
        local_root=args.local_root,
        dry_run=args.dry_run,
        continue_on_missing=not args.strict,
        skip_analysis=args.skip_analysis,
        analyze_root=args.analyze_root,
    )
    collector.run()


if __name__ == "__main__":
    main()
