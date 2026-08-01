from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def test_s2s_link_telemetry_is_dataclass():
    source = read_source("src/bloombee/server/s2s_flow.py")

    assert "@dataclass\nclass S2SLinkTelemetry" in source


def test_backend_forward_none_fails_instead_of_identity_fallback():
    source = read_source("src/bloombee/server/backend.py")

    assert "raise RuntimeError(f\"{self.name} module.forward returned None during inference\")" in source
    assert "return (hidden_states, None)" not in source


def test_push_limiter_slot_released_when_setup_fails_before_task_handoff():
    source = read_source("src/bloombee/server/handler.py")

    assert "release_slot_handed_off = False" in source
    assert "release_slot_handed_off = acquired_slot" in source
    assert "if acquired_slot and not release_slot_handed_off" in source
    assert "await self._push_limiter.release(send_time_ms=0.0, success=False)" in source


def test_lossless_receive_rejects_unsafe_wrapper_metadata():
    source = read_source("src/bloombee/utils/lossless_transport.py")

    assert "def _validate_lossless_original_size" in source
    assert "BLOOMBEE_LOSSLESS_MAX_ORIGINAL_BYTES" in source
    assert "algo_id == _ALGO_ZIPNN and not _lossless_zipnn_transport_enabled()" in source
    assert "def _validate_byte_split_elem_size" in source
    assert "if int(elem_size) not in (2, 4)" in source
