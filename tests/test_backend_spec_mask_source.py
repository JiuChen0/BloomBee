from pathlib import Path


def _function_body(source: str, name: str) -> str:
    marker = f"    def {name}("
    start = source.index(marker)
    next_def = source.index("\n    def ", start + len(marker))
    return source[start:next_def]


def test_spec_cache_valid_mask_fails_closed_on_batch_mismatch():
    source = Path("src/bloombee/server/backend.py").read_text()
    body = _function_body(source, "_spec_cache_valid_mask")
    mismatch_marker = "if ids.ndim < 2 or ids.shape[0] != batch_size:"
    start = body.index(mismatch_marker)
    end = body.index("valid_mask = ids >= 0", start)
    mismatch_branch = body[start:end]

    assert "falling back to empty cache mask" in mismatch_branch
    assert "torch.zeros(batch_size, cache_len" in mismatch_branch
    assert "torch.ones(batch_size, cache_len" not in mismatch_branch


def test_run_block_forward_raises_on_none_forward_result():
    source = Path("src/bloombee/server/backend.py").read_text()
    body = _function_body(source, "_run_block_forward")
    none_branch = body[body.index("if forward_result is None:") :]

    assert 'raise RuntimeError("module.forward returned None")' in none_branch
    assert "return None" not in none_branch
