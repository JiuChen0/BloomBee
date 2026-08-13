import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _function_node(path: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((ROOT / path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def test_spec_cache_update_runs_reorder_synchronously():
    node = _function_node("src/bloombee/server/memory_cache_manager.py", "update_cache_and_async_reorder")

    calls = [
        call.func
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    ]
    assert any(
        isinstance(func, ast.Attribute) and func.attr == "_do_reorder_task"
        for func in calls
    )
    assert not any(
        isinstance(func, ast.Attribute) and func.attr == "submit"
        for func in calls
    )


def test_spec_cache_valid_mask_batch_mismatch_is_fatal():
    node = _function_node("src/bloombee/server/backend.py", "_spec_cache_valid_mask")
    source = ast.get_source_segment(
        (ROOT / "src/bloombee/server/backend.py").read_text(),
        node,
    )

    assert "kv_cache_position_ids batch mismatch" in source
    assert "raise RuntimeError" in source
    assert "falling back to all-prefix-valid cache mask" not in source
