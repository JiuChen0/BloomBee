import ast
from pathlib import Path


def _load_update_method() -> ast.FunctionDef:
    source = Path("src/bloombee/server/memory_cache_manager.py").read_text()
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "KVCacheManager":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "update_cache_and_async_reorder":
                    return item
    raise AssertionError("KVCacheManager.update_cache_and_async_reorder not found")


def test_spec_cache_reorder_update_runs_reorder_synchronously():
    update_method = _load_update_method()

    calls = [
        node
        for node in ast.walk(update_method)
        if isinstance(node, ast.Call)
    ]

    assert any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "_do_reorder_task"
        for call in calls
    )
    assert not any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "submit"
        for call in calls
    )
