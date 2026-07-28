from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def test_s2s_link_telemetry_is_dataclass():
    source = read_source("src/bloombee/server/s2s_flow.py")
    class_idx = source.index("class S2SLinkTelemetry:")
    assert "@dataclass" in source[:class_idx].splitlines()[-3:]


def test_backend_forward_none_is_fatal():
    source = read_source("src/bloombee/server/backend.py")
    start = source.index("    def _run_block_forward(")
    end = source.index("    def _finalize_cache_update(", start)
    block = source[start:end]
    assert "if forward_result is None:" in block
    assert "raise RuntimeError" in block
    assert "return None" not in block

    step_start = source.index("                        step_result = self._run_block_forward(")
    step_end = source.index("                        output_hidden_states_chunk, new_kvs = step_result", step_start)
    step_block = source[step_start:step_end]
    assert "if step_result is None:" in step_block
    assert "raise RuntimeError" in step_block
    assert "return (hidden_states, None)" not in source


def test_microbatch_merge_layout_validation_is_fatal_and_cleans_state():
    source = read_source("src/bloombee/server/block_functions.py")
    start = source.index("                sorted_indices = sorted(accum['results'].keys())")
    end = source.index("                merged_hidden_list = []", start)
    block = source[start:end]
    assert "merge_issues" in block
    assert "logger.error" in block
    assert "_drop_mb_step_state(mb_accum_key, accum=True)" in block
    assert "_drop_mb_step_state(overlap_tracking_key, overlap=True)" in block
    assert "raise ValueError" in block
    assert "logger.warning" not in block


def test_full_batch_source_head_inference_prefers_attention_layout():
    source = read_source("src/bloombee/server/memory_cache_manager.py")
    start = source.index("    def _source_heads_per_batch(")
    end = source.index("    def _get_slot_state_key_for_mb(", start)
    block = source[start:end]
    attention_layout = "if source_bh > 0 and source_bh % attention_heads == 0:"
    kv_heads = "kv_heads = getattr(self.block_config, \"num_key_value_heads\", None)"
    assert attention_layout in block
    assert block.index(attention_layout) < block.index(kv_heads)


def test_zipnn_transport_is_explicit_opt_in():
    source = read_source("src/bloombee/utils/lossless_transport.py")
    assert "_LOSSLESS_ZIPNN_TRANSPORT_ENV = \"BLOOMBEE_LOSSLESS_ZIPNN_TRANSPORT\"" in source
    assert "def _zipnn_transport_enabled() -> bool:" in source

    supports_start = source.index("def _supports_zipnn_transport(")
    supports_end = source.index("def _zipnn_skip_reason(", supports_start)
    supports_block = source[supports_start:supports_end]
    assert "return _zipnn_transport_enabled() and _supports_zipnn_compare" in supports_block

    decompress_start = source.index("    elif algo_id == _ALGO_ZIPNN:")
    decompress_end = source.index("        decompressor = _get_zipnn_decompressor()", decompress_start)
    decompress_block = source[decompress_start:decompress_end]
    assert "if not _zipnn_transport_enabled():" in decompress_block
    assert "raise ValueError" in decompress_block

    max_start = source.index("def _max_decoded_bytes() -> int:")
    max_end = source.index("def _parse_wrapper(", max_start)
    max_block = source[max_start:max_end]
    assert "_LOSSLESS_MAX_ORIGINAL_BYTES_ENV" in max_block
    assert "1 << 28" in max_block


def test_push_limiter_releases_on_pre_handoff_failure():
    source = read_source("src/bloombee/server/handler.py")
    start = source.index("    async def _push_microbatch(")
    end = source.index("    async def _do_rpc_push_async(", start)
    block = source[start:end]
    assert "acquired_slot = False" in block
    assert "release_slot_handed_off = False" in block
    assert "release_slot_handed_off = acquired_slot" in block
    assert "if acquired_slot and not release_slot_handed_off" in block
    assert "await self._push_limiter.release(send_time_ms=0.0, success=False)" in block
