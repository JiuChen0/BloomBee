from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text()


def test_s2s_link_telemetry_is_constructible_dataclass():
    source = _source("src/bloombee/server/s2s_flow.py")

    assert "from dataclasses import dataclass, field" in source
    assert "@dataclass\nclass S2SLinkTelemetry:" in source


def test_backend_forward_none_is_fatal():
    source = _source("src/bloombee/server/backend.py")

    assert "raise RuntimeError(f\"{self.name}: module.forward returned None\")" in source
    assert "return (hidden_states, None)" not in source


def test_full_batch_gqa_expanded_kv_layout_is_not_split_by_kv_heads():
    source = _source("src/bloombee/server/memory_cache_manager.py")

    attention_head_guard = "if source_bh % attention_heads == 0:\n            return attention_heads"
    kv_head_lookup = 'kv_heads = getattr(self.block_config, "num_key_value_heads", None)'
    assert attention_head_guard in source
    assert source.index(attention_head_guard) < source.index(kv_head_lookup)


def test_microbatch_merge_layout_validation_is_fatal_and_cleans_state():
    source = _source("src/bloombee/server/block_functions.py")

    layout_block = source[
        source.index("if layout_issues:") : source.index("elif log_mb_detail:", source.index("if layout_issues:"))
    ]
    assert "_drop_mb_step_state(overlap_tracking_key, overlap=True)" in layout_block
    assert "_drop_mb_step_state(mb_accum_key, accum=True)" in layout_block
    assert "raise ValueError(" in layout_block


def test_push_limiter_slot_is_released_if_setup_fails_after_acquire():
    source = _source("src/bloombee/server/handler.py")

    method = source[source.index("async def _push_microbatch(") : source.index("async def _do_rpc_push_async(")]
    assert "acquired_slot = False" in method
    assert "release_slot_handed_off = False" in method
    assert "release_slot_handed_off = acquired_slot" in method
    assert (
        "if acquired_slot and not release_slot_handed_off and hasattr(self, \"_push_limiter\"):\n"
        "                await self._push_limiter.release(send_time_ms=0.0, success=False)"
    ) in method
