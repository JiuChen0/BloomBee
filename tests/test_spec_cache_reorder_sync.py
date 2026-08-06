import ast
from pathlib import Path


def test_spec_cache_reorder_update_blocks_until_reorder_finishes():
    source_path = Path(__file__).resolve().parents[1] / "src/bloombee/server/memory_cache_manager.py"
    tree = ast.parse(source_path.read_text())

    method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "KVCacheManager":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "update_cache_and_async_reorder":
                    method = item
                    break
    assert method is not None

    calls_submit = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "submit"
    ]
    assert calls_submit == []

    direct_reorder_calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_do_reorder_task"
    ]
    assert direct_reorder_calls
