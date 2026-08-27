from pathlib import Path


def _method_source(source: str, method_name: str) -> str:
    marker = f"    def {method_name}("
    async_marker = f"    async def {method_name}("
    try:
        start = source.index(marker)
    except ValueError:
        start = source.index(async_marker)
    next_method = source.find("\n    def ", start + len(marker))
    next_async_method = source.find("\n    async def ", start + len(marker))
    candidates = [idx for idx in (next_method, next_async_method) if idx != -1]
    if not candidates:
        return source[start:]
    return source[start:min(candidates)]


def _class_method_source(source: str, class_name: str, method_name: str) -> str:
    class_start = source.index(f"class {class_name}")
    return _method_source(source[class_start:], method_name)


def test_spec_cache_valid_mask_metadata_mismatch_fails_closed():
    source = Path("src/bloombee/server/backend.py").read_text()
    method = _method_source(source, "_spec_cache_valid_mask")
    mismatch_block = method[method.index("if ids.ndim < 2 or ids.shape[0] != batch_size:"):]

    assert "raise RuntimeError(" in mismatch_block
    assert "falling back to all-prefix-valid" not in mismatch_block


def test_mixtral_preserves_backend_attention_masks():
    source = Path("src/bloombee/models/mixtral/block.py").read_text()
    method = _class_method_source(source, "WrappedMixtralBlock", "forward")

    assert "attention_mask = None" not in method
    assert "if attention_mask is None:" in method
    assert "torch.triu(causal" in method
    assert "attention_mask = attention_mask.unsqueeze(1)" in method


def test_gemma4_external_masks_keep_sliding_window_restriction():
    source = Path("src/bloombee/models/gemma4/block.py").read_text()
    method = _class_method_source(source, "WrappedGemma4Block", "forward")

    assert "if self.is_sliding:" in method
    assert "_build_layer_type_mask(" in method
    assert "attention_mask = torch.minimum(attention_mask, sliding_mask)" in method


def test_microbatch_merge_layout_validation_is_fatal():
    source = Path("src/bloombee/server/block_functions.py").read_text()
    layout_block = source[source.index("if layout_issues:"):]

    assert "_drop_mb_step_state(mb_accum_key, accum=True)" in layout_block
    assert "raise ValueError(" in layout_block
    assert "Refusing to merge partial or misordered rows" in layout_block


def test_cross_stage_push_releases_limiter_slot_before_handoff():
    source = Path("src/bloombee/server/handler.py").read_text()
    method = _method_source(source, "_push_microbatch")

    assert "release_slot_handed_off = False" in method
    assert "release_slot_handed_off = acquired_slot" in method
    assert "if acquired_slot and not release_slot_handed_off" in method
    assert "await self._push_limiter.release(send_time_ms=measured_send_ms, success=False)" in method
