import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _class_method(source_path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(source_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} not found in {source_path}")


def _returns_hidden_states_none(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Return) or not isinstance(child.value, ast.Tuple):
            continue
        elements = child.value.elts
        if (
            len(elements) == 2
            and isinstance(elements[0], ast.Name)
            and elements[0].id == "hidden_states"
            and isinstance(elements[1], ast.Constant)
            and elements[1].value is None
        ):
            return True
    return False


def _except_handlers_raise(node: ast.AST) -> bool:
    handlers = [child for child in ast.walk(node) if isinstance(child, ast.ExceptHandler)]
    return bool(handlers) and all(any(isinstance(item, ast.Raise) for item in handler.body) for handler in handlers)


def test_block_forward_none_is_not_treated_as_success():
    method = _class_method(
        ROOT / "src" / "bloombee" / "server" / "backend.py",
        "TransformerBackend",
        "_run_block_forward",
    )

    none_checks = []
    for node in ast.walk(method):
        if isinstance(node, ast.If) and "forward_result" in ast.unparse(node.test) and "None" in ast.unparse(node.test):
            none_checks.append(node)

    assert none_checks, "_run_block_forward must explicitly handle module.forward returning None"
    assert any(any(isinstance(item, ast.Raise) for item in check.body) for check in none_checks)
    assert not any(isinstance(node, ast.Return) and node.value is None for node in ast.walk(method))


def test_inference_step_propagates_failures_instead_of_identity_fallback():
    method = _class_method(
        ROOT / "src" / "bloombee" / "server" / "backend.py",
        "TransformerBackend",
        "inference_step",
    )

    assert not _returns_hidden_states_none(method)
    assert _except_handlers_raise(method)


def test_speculative_reorder_failures_propagate_to_caller():
    method = _class_method(
        ROOT / "src" / "bloombee" / "server" / "memory_cache_manager.py",
        "KVCacheManager",
        "_do_reorder_task",
    )

    assert _except_handlers_raise(method)
