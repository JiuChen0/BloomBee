import importlib.util
import sys
import types
from pathlib import Path


def _load_s2s_flow_module():
    sys.modules.setdefault("bloombee", types.ModuleType("bloombee"))
    sys.modules.setdefault("bloombee.utils", types.ModuleType("bloombee.utils"))
    microbatch_config = types.ModuleType("bloombee.utils.microbatch_config")
    microbatch_config.MBPIPE_LOG_PREFIX = "[MBPIPE]"
    sys.modules["bloombee.utils.microbatch_config"] = microbatch_config

    module_path = Path(__file__).resolve().parents[1] / "src" / "bloombee" / "server" / "s2s_flow.py"
    spec = importlib.util.spec_from_file_location("bloombee.server.s2s_flow", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_s2s_link_telemetry_initializes_and_records_windowed_samples():
    S2SLinkTelemetry = _load_s2s_flow_module().S2SLinkTelemetry
    telemetry = S2SLinkTelemetry(label="rpc_push_microbatch:0:1->1:2", window_size=2)

    first_jitter = telemetry.record(
        latency_ms=10.0,
        raw_latency_ms=12.0,
        bandwidth_mbps=100.0,
        total_bytes=1024,
        clock_sync_ok=True,
    )
    second_jitter = telemetry.record(
        latency_ms=15.5,
        raw_latency_ms=17.5,
        bandwidth_mbps=90.0,
        total_bytes=2048,
        clock_sync_ok=False,
    )
    telemetry.record(
        latency_ms=14.0,
        raw_latency_ms=16.0,
        bandwidth_mbps=95.0,
        total_bytes=512,
        clock_sync_ok=True,
    )

    assert first_jitter == 0.0
    assert second_jitter == 5.5
    assert telemetry.samples == 3
    assert telemetry.total_bytes == 3584
    assert telemetry.clock_sync_samples == 2
    assert list(telemetry.latency_ms_window) == [15.5, 14.0]
    assert list(telemetry.raw_latency_ms_window) == [17.5, 16.0]
