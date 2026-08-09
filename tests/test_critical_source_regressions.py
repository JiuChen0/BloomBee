from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text()


def test_microbatch_merge_layout_validation_is_fatal():
    source = _read("src/bloombee/server/block_functions.py")
    layout_pos = source.index("if layout_issues:")
    merge_pos = source.index("merged_hidden_states = _merge_inference_microbatch_hidden_states")
    layout_block = source[layout_pos:merge_pos]

    assert "_drop_mb_step_state(mb_accum_key, accum=True)" in layout_block
    assert "_drop_mb_step_state(overlap_tracking_key, overlap=True)" in layout_block
    assert "raise ValueError(" in layout_block
    assert "Refusing to merge partial or misordered micro-batches" in layout_block


def test_push_limiter_slot_released_on_setup_failure():
    source = _read("src/bloombee/server/handler.py")
    acquire_pos = source.index("acquired_slot = False")
    except_pos = source.index("Failed to push micro-batch")
    setup_block = source[acquire_pos:except_pos]

    assert "release_slot_handed_off = False" in setup_block
    assert "release_slot_handed_off = acquired_slot" in setup_block
    assert "acquired_slot and not release_slot_handed_off" in setup_block
    assert "await self._push_limiter.release(send_time_ms=0.0, success=False)" in setup_block


def test_gemma4_external_masks_merge_with_layer_type_mask():
    source = _read("src/bloombee/models/gemma4/block.py")
    assert "def _to_additive_4d_mask" in source
    assert "layer_type_mask = _build_layer_type_mask(" in source
    assert "attention_mask = torch.minimum(attention_mask, layer_type_mask.to(dtype=attention_mask.dtype))" in source


def test_mixtral_preserves_backend_attention_masks():
    source = _read("src/bloombee/models/mixtral/block.py")
    forward_start = source.index("    def forward(")
    forward_source = source[forward_start:]

    assert "def _to_additive_4d_mask" in source
    assert "if attention_mask is None:" in forward_source
    assert "attention_mask = _to_additive_4d_mask(" in forward_source
    assert "attention_mask = None" not in forward_source


def test_mixed_device_gpu_multiplexing_fails_fast():
    source = _read("src/bloombee/server/memory_cache_manager.py")
    message = "MixedDevice cache + GPU micro-batch multiplexing is unsupported"
    assert source.count(message) >= 2
    assert "raise RuntimeError(" in source[source.index(message) - 100: source.index(message) + 200]


def test_active_row_retry_history_uses_hypo_ids_gather():
    source = _read("src/bloombee/client/inference_session.py")
    start = source.index("row_selector = None")
    end = source.index("self.history = torch.cat", start)
    block = source[start:end]

    assert "hypo_ids" in block
    assert "row_selector = hypo_ids.to" in block
    assert "self.history = self.history.index_select(0, row_selector)" in block
    assert "Active-row compaction hypo_ids are out of bounds" in block


def test_s2s_int8_hidden_requires_quant_metadata():
    source = _read("src/bloombee/utils/s2s_activation_quant.py")
    function_start = source.index("def dequantize_s2s_hidden_from_transport(")
    function_source = source[function_start:]

    assert "if hidden_states.dtype == torch.int8:" in function_source
    assert function_source.count("Received int8 S2S hidden states without quantization metadata") >= 2
    assert "Unsupported S2S activation quantization scheme" in function_source
