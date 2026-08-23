from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def _function_body(source: str, name: str) -> str:
    start = source.index(f"def {name}(")
    next_def = source.find("\n    def ", start + 1)
    return source[start:] if next_def == -1 else source[start:next_def]


def test_speculative_reorder_update_is_synchronous():
    source = _read("src/bloombee/server/memory_cache_manager.py")
    body = _function_body(source, "update_cache_and_async_reorder")

    assert "self.wait_for_pending_reorder()" in body
    assert "self._do_reorder_task(" in body
    assert "self._reorder_executor.submit" not in body
    assert "self._pending_reorder = future" not in body


def test_speculative_mask_metadata_mismatch_is_fatal():
    source = _read("src/bloombee/server/backend.py")
    body = _function_body(source, "_spec_cache_valid_mask")
    mismatch_branch = body[body.index("if ids.ndim < 2 or ids.shape[0] != batch_size:"):]

    assert "missing kv_cache_position_ids" in body
    assert "raise RuntimeError" in mismatch_branch
    assert "return torch.ones" not in body
    assert "falling back to all-prefix-valid cache mask" not in mismatch_branch


def test_block_forward_none_is_fatal_at_helper_boundary():
    source = _read("src/bloombee/server/backend.py")
    body = _function_body(source, "_run_block_forward")

    assert "if forward_result is None:" in body
    assert "raise RuntimeError" in body
    assert "return None" not in body


def test_mixtral_preserves_backend_attention_masks():
    source = _read("src/bloombee/models/mixtral/block.py")

    assert "attention_mask = None" not in source
    assert "_prepare_bloombee_attention_mask(" in source
    assert "attention_mask=attention_mask" in source
    assert "masked_fill(~mask" in source
