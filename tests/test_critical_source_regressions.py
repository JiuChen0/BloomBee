from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def _function_body(source: str, name: str, next_marker: str) -> str:
    start = source.index(f"    def {name}")
    end = source.index(next_marker, start)
    return source[start:end]


def test_spec_cache_reorder_runs_synchronously():
    source = _source("src/bloombee/server/memory_cache_manager.py")
    body = _function_body(source, "update_cache_and_async_reorder", "\n    @staticmethod\n    def _plain_tensor_or_none")

    assert "self.wait_for_pending_reorder()" in body
    assert "self._do_reorder_task(" in body
    assert "_reorder_executor.submit" not in body
    assert "_pending_reorder = future" not in body


def test_spec_cache_mask_metadata_mismatch_fails_closed():
    source = _source("src/bloombee/server/backend.py")
    body = _function_body(source, "_spec_cache_valid_mask", "\n    def _expand_local_tree_attention_mask")

    assert "missing kv_cache_position_ids" in body
    assert "treating cache prefix as invalid" in body
    assert "falling back to all-prefix-valid" not in body
    assert body.count("return torch.zeros(batch_size, cache_len") >= 2


def test_mixtral_preserves_backend_attention_masks():
    source = _source("src/bloombee/models/mixtral/block.py")
    body = _function_body(source, "forward", "\n    def _reorder_cache_from_bloom")

    assert "_build_causal_attention_mask(" in source
    assert "attention_mask = None" not in body
    assert "attention_mask = attention_mask.unsqueeze(1)" in body
    assert "attention_mask.dtype == torch.bool" in body
