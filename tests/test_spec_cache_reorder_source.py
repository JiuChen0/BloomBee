import ast
from pathlib import Path


def _update_cache_and_async_reorder_ast() -> ast.FunctionDef:
    source = Path("src/bloombee/server/memory_cache_manager.py").read_text()
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "KVCacheManager":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "update_cache_and_async_reorder":
                    return item
    raise AssertionError("KVCacheManager.update_cache_and_async_reorder not found")


def test_spec_cache_reorder_update_runs_reorder_synchronously():
    method = _update_cache_and_async_reorder_ast()

    submitted_to_reorder_executor = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "submit"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "_reorder_executor"
    ]
    direct_reorder_calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_do_reorder_task"
    ]

    assert not submitted_to_reorder_executor
    assert direct_reorder_calls
