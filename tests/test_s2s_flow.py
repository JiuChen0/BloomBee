import importlib.util
import sys
import types
from pathlib import Path


def test_s2s_link_telemetry_constructs_and_records_without_torch():
    microbatch_config = types.ModuleType("bloombee.utils.microbatch_config")
    microbatch_config.MBPIPE_LOG_PREFIX = "[MBPIPE]"
    sys.modules.setdefault("bloombee", types.ModuleType("bloombee"))
    sys.modules.setdefault("bloombee.utils", types.ModuleType("bloombee.utils"))
    sys.modules["bloombee.utils.microbatch_config"] = microbatch_config

    module_path = Path(__file__).resolve().parents[1] / "src" / "bloombee" / "server" / "s2s_flow.py"
    spec = importlib.util.spec_from_file_location("bloombee.server.s2s_flow", module_path)
    assert spec is not None
    assert spec.loader is not None
    s2s_flow = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = s2s_flow
    spec.loader.exec_module(s2s_flow)

    telemetry = s2s_flow.S2SLinkTelemetry(label="decode:0-1->2-3", window_size=2)

    assert telemetry.record(
        latency_ms=10.0,
        raw_latency_ms=12.0,
        bandwidth_mbps=100.0,
        total_bytes=1024,
        clock_sync_ok=True,
    ) == 0.0
    assert telemetry.record(
        latency_ms=13.5,
        raw_latency_ms=15.0,
        bandwidth_mbps=80.0,
        total_bytes=2048,
        clock_sync_ok=False,
    ) == 3.5

    assert telemetry.samples == 2
    assert telemetry.total_bytes == 3072
    assert telemetry.clock_sync_samples == 1
    assert list(telemetry.latency_ms_window) == [10.0, 13.5]
    assert list(telemetry.bandwidth_mbps_window) == [100.0, 80.0]

