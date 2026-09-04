import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text()


def _function_node(source: str, name: str) -> ast.FunctionDef:
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def test_speculative_cache_update_runs_reorder_synchronously():
    source = _source("src/bloombee/server/memory_cache_manager.py")
    fn = _function_node(source, "update_cache_and_async_reorder")

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_do_reorder_task"
        for node in ast.walk(fn)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "submit"
        for node in ast.walk(fn)
    )


def test_speculative_cache_mask_mismatch_fails_closed():
    source = _source("src/bloombee/server/backend.py")
    fn = _function_node(source, "_spec_cache_valid_mask")
    snippet = ast.get_source_segment(source, fn)

    assert "treating cache prefix as invalid" in snippet
    assert "return torch.zeros(batch_size, cache_len" in snippet
    assert "falling back to all-prefix-valid" not in snippet


def test_mixtral_preserves_backend_attention_masks():
    source = _source("src/bloombee/models/mixtral/block.py")
    fn = _function_node(source, "forward")
    snippet = ast.get_source_segment(source, fn)

    assert "attention_mask = None" not in snippet
    assert "attention_mask = attention_mask.unsqueeze(1)" in snippet
    assert "_build_causal_mask(" in snippet
