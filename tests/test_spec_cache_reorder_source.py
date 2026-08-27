from pathlib import Path


def _method_source(class_source: str, method_name: str) -> str:
    marker = f"    def {method_name}("
    start = class_source.index(marker)
    next_method = class_source.find("\n    def ", start + len(marker))
    if next_method == -1:
        return class_source[start:]
    return class_source[start:next_method]


def test_speculative_cache_update_runs_reorder_synchronously():
    source = Path("src/bloombee/server/memory_cache_manager.py").read_text()
    method = _method_source(source, "update_cache_and_async_reorder")

    assert "wait_for_pending_reorder()" in method
    assert "self._do_reorder_task(" in method
    assert "_reorder_executor.submit" not in method
    assert "self._pending_reorder = future" not in method
