# Cluster Launch Queue

This directory contains an SSH-based scheduler for 3-machine BloomBee experiments.

## Files

- `cluster_launch_queue.py`: queue launcher that coordinates `dht`, `server1`, `server2`, and `client`.
- `cluster_launch_config.example.json`: template config based on your 3-server topology.
- `cluster_launch_config.windows_aliases.example.json`: template for Windows controller using SSH host aliases.
- `collect_and_analyze.py`: one-click log collection + analysis (`latency_breakdown` + `tradeoff_matrix`).
- `run_sequenced_tradeoff_sweep.py`: sequenced runner for per-case `dht -> server1 -> server2 -> inference` with auto stop/collect/analyze.
- `sequenced_tradeoff_windows_aliases.example.json`: config for the sequenced trade-off sweep.

## How It Works

1. Controller machine runs `cluster_launch_queue.py`.
2. For each case in `cases`:
   - starts remote `server1` and `server2` in background;
   - runs remote `client` benchmark in foreground;
   - collects logs in `results_root/<run_tag>_<case_idx>_<case_name>/`;
   - waits `interval_sec` before next case.
3. `dht` can be reused across all cases (`reuse_dht=true`) or restarted per case.

## Usage

```bash
cd /home/cc/BloomBee
python benchmarks/automation/cluster_launch_queue.py \
  --config benchmarks/automation/cluster_launch_config.example.json \
  --dry-run
```

After checking rendered commands, remove `--dry-run`:

```bash
python benchmarks/automation/cluster_launch_queue.py \
  --config benchmarks/automation/cluster_launch_config.example.json
```

## Windows Controller

If you run the controller from Windows and already have SSH host aliases in
`C:\Users\<you>\.ssh\config`, use:

- `cluster_launch_config.windows_aliases.example.json`

This config uses host aliases (`merced`, `Canada4090`, `NewJersey4090`) and
does not hardcode key paths/ports in JSON.

PowerShell example:

```powershell
cd C:\path\to\BloomBee
python .\benchmarks\automation\cluster_launch_queue.py `
  --config .\benchmarks\automation\cluster_launch_config.windows_aliases.example.json `
  --dry-run
```

Then run without `--dry-run`.

You do **not** need the full BloomBee repo on Windows for queue launch.
At minimum, copy these files:

- `benchmarks/automation/cluster_launch_queue.py`
- `benchmarks/automation/collect_and_analyze.py`
- one JSON config file (e.g. `cluster_launch_config.windows_aliases.example.json`)
- optional: `run_sequenced_tradeoff_sweep.py` + `sequenced_tradeoff_windows_aliases.example.json`

## One-Click Postprocess

After queued runs finish, collect logs from all hosts and analyze locally:

```powershell
python .\benchmarks\automation\collect_and_analyze.py `
  --config .\benchmarks\automation\cluster_launch_config.windows_aliases.example.json `
  --run-tag weekly_sensitivity `
  --local-root .\collected_runs
```

If Windows does not have analysis scripts (`benchmarks/analyze_latency_breakdown.py` and
`benchmarks/analyze_tradeoff_matrix.py`), collect only:

```powershell
python .\benchmarks\automation\collect_and_analyze.py `
  --config .\benchmarks\automation\cluster_launch_config.windows_aliases.example.json `
  --run-tag weekly_sensitivity `
  --local-root .\collected_runs `
  --skip-analysis
```

Outputs:

- Per-case logs + `latency_breakdown.txt`:
  - `collected_runs/<run_tag>_<idx>_<case_name>/...`
- Global matrix summary:
  - `collected_runs/<run_tag>/tradeoff_matrix.txt`
  - `collected_runs/<run_tag>/tradeoff_matrix.csv`

## Sequenced Trade-off Sweep

If you need strict startup ordering and per-case shutdown:

- start DHT
- wait 30s
- start server1
- wait 30s
- start server2
- wait 30s
- run benchmark (`eval_tokens=200`)
- kill dht/server1/server2
- collect logs + generate per-case summary
- continue next case

Use:

```powershell
python .\benchmarks\automation\run_sequenced_tradeoff_sweep.py `
  --config .\benchmarks\automation\sequenced_tradeoff_windows_aliases.example.json `
  --local-root .\collected_runs `
  --dry-run
```

Then run without `--dry-run`.

Main outputs:

- per case: `collected_runs/<run_tag>_<idx>_<case>/analysis_summary.txt`
- sweep matrix:
  - `collected_runs/<run_tag>/sweep_matrix.csv`
  - `collected_runs/<run_tag>/sweep_matrix.txt`

This runner supports per-case policy tuning via env variables:

- `BLOOMBEE_POLICY_W_GPU_PERCENT`, `BLOOMBEE_POLICY_W_CPU_PERCENT`
- `BLOOMBEE_POLICY_CACHE_GPU_PERCENT`, `BLOOMBEE_POLICY_CACHE_CPU_PERCENT`
- `BLOOMBEE_POLICY_ACT_GPU_PERCENT`, `BLOOMBEE_POLICY_ACT_CPU_PERCENT`
- `BLOOMBEE_ENABLE_KV_OFFLOAD` (`0`/`1`)

So you can sweep cache/offload/compression without editing source files each round.
If you still want to rebuild each case, set:

- `schedule.reinstall_editable_each_case=true`
- `schedule.reinstall_cmd="pip install -e ."`

## Notes

- The scheduler uses SSH and remote background jobs (`nohup setsid ...`).
- You can use SSH config host aliases (e.g. `Canada4090`, `NewJersey4090`, `merced`).
  If `port` is omitted in JSON, SSH will use the port from your `~/.ssh/config`.
- Each host should have:
  - the same BloomBee repo path (or override per host in config),
  - conda env `bb` (or customize `host_defaults.activate`),
  - passwordless SSH from controller host.
- `default_env` + per-case `env` control micro-batching/compression toggles without editing code.
