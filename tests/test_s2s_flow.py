from bloombee.server.s2s_flow import S2SLinkTelemetry


def test_s2s_link_telemetry_constructs_and_records_samples():
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
