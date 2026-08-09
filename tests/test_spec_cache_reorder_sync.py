import ast
from pathlib import Path


def _update_cache_and_async_reorder_method():
    source_path = Path(__file__).resolve().parents[1] / "src/bloombee/server/memory_cache_manager.py"
    source = source_path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "KVCacheManager":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "update_cache_and_async_reorder":
                    return item
    raise AssertionError("KVCacheManager.update_cache_and_async_reorder not found")


def test_spec_cache_reorder_update_runs_reorder_synchronously():
    method = _update_cache_and_async_reorder_method()
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]

    assert any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "wait_for_pending_reorder"
        for call in calls
    )
    assert any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "_do_reorder_task"
        for call in calls
    )
    assert not any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "submit"
        for call in calls
    ), "speculative KV reorder must not be scheduled in the background"

    wait_call = next(
        call for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "wait_for_pending_reorder"
    )
    reorder_call = next(
        call for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "_do_reorder_task"
    )
    assert wait_call.lineno < reorder_call.lineno
