import importlib.util
import sys
import types
from pathlib import Path


def _load_s2s_flow(monkeypatch):
    microbatch_config = types.ModuleType("bloombee.utils.microbatch_config")
    microbatch_config.MBPIPE_LOG_PREFIX = "[MBPIPE]"

    monkeypatch.setitem(sys.modules, "bloombee", types.ModuleType("bloombee"))
    monkeypatch.setitem(sys.modules, "bloombee.utils", types.ModuleType("bloombee.utils"))
    monkeypatch.setitem(sys.modules, "bloombee.utils.microbatch_config", microbatch_config)

    module_path = Path(__file__).resolve().parents[1] / "src" / "bloombee" / "server" / "s2s_flow.py"
    spec = importlib.util.spec_from_file_location("_s2s_flow_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_s2s_link_telemetry_constructs_and_tracks_rolling_stats(monkeypatch):
    s2s_flow = _load_s2s_flow(monkeypatch)

    telemetry = s2s_flow.S2SLinkTelemetry(label="micro_batch:0:1->1:2", window_size=2)

    assert telemetry.label == "micro_batch:0:1->1:2"
    assert telemetry.window_size == 2
    assert telemetry.record(
        latency_ms=10.0,
        raw_latency_ms=12.0,
        bandwidth_mbps=100.0,
        total_bytes=1024,
        clock_sync_ok=True,
    ) == 0.0
    assert telemetry.record(
        latency_ms=15.0,
        raw_latency_ms=16.0,
        bandwidth_mbps=80.0,
        total_bytes=2048,
        clock_sync_ok=False,
    ) == 5.0
    assert telemetry.record(
        latency_ms=11.0,
        raw_latency_ms=13.0,
        bandwidth_mbps=120.0,
        total_bytes=4096,
        clock_sync_ok=True,
    ) == 4.0

    assert telemetry.samples == 3
    assert telemetry.total_bytes == 7168
    assert telemetry.clock_sync_samples == 2
    assert list(telemetry.latency_ms_window) == [15.0, 11.0]
    assert list(telemetry.raw_latency_ms_window) == [16.0, 13.0]
    assert list(telemetry.bandwidth_mbps_window) == [80.0, 120.0]
    assert list(telemetry.jitter_ms_window) == [5.0, 4.0]
