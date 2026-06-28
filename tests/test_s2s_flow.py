import importlib.util
import sys
import types
from pathlib import Path


def _load_s2s_flow_module():
    """Load the module without importing bloombee.__init__ and its heavy deps."""
    microbatch_config = types.ModuleType("bloombee.utils.microbatch_config")
    microbatch_config.MBPIPE_LOG_PREFIX = "[mbpipe]"

    stubs = {
        "bloombee": types.ModuleType("bloombee"),
        "bloombee.utils": types.ModuleType("bloombee.utils"),
        "bloombee.utils.microbatch_config": microbatch_config,
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        path = Path(__file__).resolve().parents[1] / "src" / "bloombee" / "server" / "s2s_flow.py"
        spec = importlib.util.spec_from_file_location("s2s_flow_under_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous_module in previous.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


def test_s2s_link_telemetry_constructs_and_records_samples():
    S2SLinkTelemetry = _load_s2s_flow_module().S2SLinkTelemetry
    telemetry = S2SLinkTelemetry(label="micro_batch:0:1->1:2", window_size=2)

    jitter_ms = telemetry.record(
        latency_ms=10.0,
        raw_latency_ms=12.0,
        bandwidth_mbps=4.0,
        total_bytes=512,
        clock_sync_ok=True,
    )
    telemetry.record(
        latency_ms=15.0,
        raw_latency_ms=16.0,
        bandwidth_mbps=8.0,
        total_bytes=256,
        clock_sync_ok=False,
    )
    telemetry.record(
        latency_ms=12.0,
        raw_latency_ms=13.0,
        bandwidth_mbps=6.0,
        total_bytes=128,
        clock_sync_ok=True,
    )

    assert jitter_ms == 0.0
    assert telemetry.samples == 3
    assert telemetry.total_bytes == 896
    assert telemetry.clock_sync_samples == 2
    assert list(telemetry.latency_ms_window) == [15.0, 12.0]
    assert list(telemetry.bandwidth_mbps_window) == [8.0, 6.0]
    assert list(telemetry.jitter_ms_window) == [5.0, 3.0]
    assert list(telemetry.raw_latency_ms_window) == [16.0, 13.0]
