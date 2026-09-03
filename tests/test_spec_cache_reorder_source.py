from pathlib import Path


def _function_body(source: str, name: str) -> str:
    marker = f"    def {name}("
    start = source.index(marker)
    next_def = source.index("\n    def ", start + len(marker))
    return source[start:next_def]


def test_spec_cache_update_runs_reorder_synchronously():
    source = Path("src/bloombee/server/memory_cache_manager.py").read_text()
    body = _function_body(source, "update_cache_and_async_reorder")

    assert "self.wait_for_pending_reorder()" in body
    assert "self._do_reorder_task(" in body
    assert "_reorder_executor.submit" not in body
    assert "_pending_reorder = future" not in body
