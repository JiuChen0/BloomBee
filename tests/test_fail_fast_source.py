import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_module(relative_path: str) -> tuple[str, ast.Module]:
    source = (ROOT / relative_path).read_text()
    return source, ast.parse(source)


def _class_method(module: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} not found")


def _returns_hidden_states_none(node: ast.AST) -> bool:
    if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Tuple):
        return False
    if len(node.value.elts) != 2:
        return False
    first, second = node.value.elts
    return (
        isinstance(first, ast.Name)
        and first.id == "hidden_states"
        and isinstance(second, ast.Constant)
        and second.value is None
    )


def test_backend_inference_failures_are_not_identity_fallbacks():
    _, module = _read_module("src/bloombee/server/backend.py")
    inference_step = _class_method(module, "TransformerBackend", "inference_step")
    run_block_forward = _class_method(module, "TransformerBackend", "_run_block_forward")

    assert not any(_returns_hidden_states_none(node) for node in ast.walk(inference_step))

    none_forward_checks = [
        node
        for node in ast.walk(run_block_forward)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "forward_result"
    ]
    assert none_forward_checks, "_run_block_forward must explicitly handle None forward results"
    assert any(any(isinstance(child, ast.Raise) for child in ast.walk(check)) for check in none_forward_checks)


def test_speculative_reorder_failures_are_not_swallowed():
    source, module = _read_module("src/bloombee/server/memory_cache_manager.py")
    reorder_task = _class_method(module, "KVCacheManager", "_do_reorder_task")

    assert "select_cache returned no KV tensors during speculative cache reorder" in source
    exception_handlers = [node for node in ast.walk(reorder_task) if isinstance(node, ast.ExceptHandler)]
    assert exception_handlers, "_do_reorder_task should log and re-raise unexpected failures"
    assert all(any(isinstance(child, ast.Raise) for child in ast.walk(handler)) for handler in exception_handlers)
