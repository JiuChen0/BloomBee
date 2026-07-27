from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_s2s_link_telemetry_is_constructible_dataclass():
    source = read_source("src/bloombee/server/s2s_flow.py")

    assert "@dataclass\nclass S2SLinkTelemetry:" in source


def test_backend_none_forward_fails_instead_of_identity_success():
    source = read_source("src/bloombee/server/backend.py")

    assert 'raise RuntimeError("module.forward returned None")' in source
    assert "return (hidden_states, None)" not in source


def test_speculative_reorder_failures_propagate():
    source = read_source("src/bloombee/server/memory_cache_manager.py")

    assert 'logging.exception("Async cache reorder failed")' in source
    assert "logging.error(f\"Async cache reorder failed: {e}\")" not in source


def test_gemma4_external_masks_are_merged_with_sliding_mask():
    source = read_source("src/bloombee/models/gemma4/block.py")

    assert "def _normalize_external_attention_mask" in source
    assert "attention_mask = torch.minimum(attention_mask, layer_type_mask)" in source
    assert "attention_mask = attention_mask.unsqueeze(1)" in source


def test_mixtral_preserves_backend_attention_masks():
    source = read_source("src/bloombee/models/mixtral/block.py")

    assert "def _build_causal_attention_mask" in source
    assert "def _normalize_attention_mask" in source
    assert "attention_mask = _normalize_attention_mask(attention_mask, hidden_states.dtype)" in source
    assert "attention_mask = None\n\n        position_ids" not in source


def test_microbatch_merge_layout_errors_are_fatal():
    source = read_source("src/bloombee/server/block_functions.py")

    assert "Micro-batch merge layout check failed" in source
    assert "logger.error(message)" in source
    assert "_drop_mb_step_state(mb_accum_key, accum=True)" in source
    assert "_drop_mb_step_state(overlap_tracking_key, overlap=True)" in source
    assert "raise ValueError(message)" in source


def test_push_limiter_slot_released_when_task_setup_fails():
    source = read_source("src/bloombee/server/handler.py")

    assert "# From here the background task owns releasing the limiter slot." in source
    assert "if locals().get(\"acquired_slot\", False):" in source
    assert "await self._push_limiter.release(send_time_ms=0.0, success=False)" in source
