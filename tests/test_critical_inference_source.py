from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gemma4_backend_mask_path_merges_layer_type_mask():
    source = (ROOT / "src/bloombee/models/gemma4/block.py").read_text()

    assert "def _merge_with_layer_type_mask" in source
    assert "torch.minimum(attention_mask, layer_type_mask)" in source
    assert "attention_mask = _merge_with_layer_type_mask(" in source
    assert "attention_mask = layer_type_mask" in source


def test_speculative_reorder_failures_are_not_swallowed():
    source = (ROOT / "src/bloombee/server/memory_cache_manager.py").read_text()

    assert 'logger.exception("Cache reorder failed during speculative cache update")' in source
    assert 'logger.exception("Cache reorder failed during speculative cache update")\n            raise' in source
