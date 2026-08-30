from pathlib import Path


def test_spec_cache_valid_mask_rejects_batch_metadata_mismatch():
    source = Path("src/bloombee/server/backend.py").read_text()
    method_start = source.index("    def _spec_cache_valid_mask(")
    method_end = source.index("    def _expand_local_tree_attention_mask(", method_start)
    method_source = source[method_start:method_end]

    assert "kv_cache_position_ids is None or is_dummy(kv_cache_position_ids)" in method_source
    assert "ids.ndim < 2 or ids.shape[0] != batch_size" in method_source
    assert "raise RuntimeError(" in method_source
    assert "return torch.ones" not in method_source
    assert "all-prefix-valid" not in method_source

