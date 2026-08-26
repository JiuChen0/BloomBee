from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text()


def _function_source(source: str, name: str) -> str:
    marker = f"    def {name}("
    start = source.index(marker)
    next_method = source.find("\n    def ", start + len(marker))
    return source[start:] if next_method == -1 else source[start:next_method]


def test_speculative_cache_update_runs_reorder_synchronously():
    source = _source("src/bloombee/server/memory_cache_manager.py")
    body = _function_source(source, "update_cache_and_async_reorder")

    assert "wait_for_pending_reorder()" in body
    assert "self._do_reorder_task(" in body
    assert "self._reorder_executor.submit" not in body
    assert "self._pending_reorder = future" not in body


def test_mixtral_preserves_backend_attention_masks():
    source = _source("src/bloombee/models/mixtral/block.py")
    forward_body = _function_source(source, "forward")

    assert "attention_mask = _prepare_mixtral_attention_mask(" in forward_body
    assert "attention_mask = None" not in forward_body

    assert "def _prepare_mixtral_attention_mask(" in source
    assert "attention_mask = attention_mask.unsqueeze(1)" in source
    assert "def _build_causal_attention_mask(" in source
