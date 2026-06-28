from bloombee.server.s2s_flow import S2SLinkTelemetry


def test_s2s_link_telemetry_initializes_and_records_samples():
    telemetry = S2SLinkTelemetry(label="decode:0:8->8:16", window_size=2)

    assert telemetry.label == "decode:0:8->8:16"
    assert telemetry.latency_ms_window.maxlen == 2

    assert (
        telemetry.record(
            latency_ms=10.0,
            raw_latency_ms=12.0,
            bandwidth_mbps=100.0,
            total_bytes=1024,
            clock_sync_ok=True,
        )
        == 0.0
    )
    assert (
        telemetry.record(
            latency_ms=13.5,
            raw_latency_ms=15.0,
            bandwidth_mbps=80.0,
            total_bytes=2048,
            clock_sync_ok=False,
        )
        == 3.5
    )

    assert telemetry.samples == 2
    assert telemetry.total_bytes == 3072
    assert telemetry.clock_sync_samples == 1
    assert list(telemetry.latency_ms_window) == [10.0, 13.5]
