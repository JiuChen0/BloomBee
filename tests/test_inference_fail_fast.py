import ast
from pathlib import Path


BACKEND_PATH = Path(__file__).resolve().parents[1] / "src" / "bloombee" / "server" / "backend.py"


def _get_method(class_node: ast.ClassDef, method_name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"{method_name} not found")


def _get_transformer_backend(tree: ast.AST) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "TransformerBackend":
            return node
    raise AssertionError("TransformerBackend not found")


def _returns_hidden_states_none(node: ast.Return) -> bool:
    value = node.value
    if not isinstance(value, ast.Tuple) or len(value.elts) != 2:
        return False
    first, second = value.elts
    return (
        isinstance(first, ast.Name)
        and first.id == "hidden_states"
        and isinstance(second, ast.Constant)
        and second.value is None
    )


def test_inference_step_does_not_hide_block_failures_as_identity_success():
    tree = ast.parse(BACKEND_PATH.read_text(encoding="utf-8"))
    backend = _get_transformer_backend(tree)
    inference_step = _get_method(backend, "inference_step")

    forbidden_returns = [
        node
        for node in ast.walk(inference_step)
        if isinstance(node, ast.Return) and _returns_hidden_states_none(node)
    ]

    assert not forbidden_returns, "inference_step must propagate failures instead of returning hidden_states unchanged"


def test_run_block_forward_rejects_none_forward_result():
    tree = ast.parse(BACKEND_PATH.read_text(encoding="utf-8"))
    backend = _get_transformer_backend(tree)
    run_block_forward = _get_method(backend, "_run_block_forward")

    forward_none_branches = []
    for node in ast.walk(run_block_forward):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "forward_result"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Is)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        ):
            forward_none_branches.append(node)

    assert forward_none_branches, "_run_block_forward must explicitly handle module.forward returning None"
    for branch in forward_none_branches:
        assert any(isinstance(stmt, ast.Raise) for stmt in branch.body), (
            "_run_block_forward must raise when module.forward returns None"
        )
