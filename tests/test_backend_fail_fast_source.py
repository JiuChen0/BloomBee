import ast
from pathlib import Path


BACKEND_SOURCE = Path(__file__).resolve().parents[1] / "src" / "bloombee" / "server" / "backend.py"


def _function_def(source: str, name: str) -> ast.FunctionDef:
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def test_block_forward_none_raises_instead_of_returning_identity_output():
    source = BACKEND_SOURCE.read_text(encoding="utf-8")
    run_block_forward = _function_def(source, "_run_block_forward")

    none_guard = None
    for node in ast.walk(run_block_forward):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            left = node.test.left
            comparators = node.test.comparators
            if (
                isinstance(left, ast.Name)
                and left.id == "forward_result"
                and any(isinstance(comp, ast.Constant) and comp.value is None for comp in comparators)
            ):
                none_guard = node
                break

    assert none_guard is not None
    assert any(isinstance(stmt, ast.Raise) for stmt in none_guard.body)
    assert not any(isinstance(stmt, ast.Return) for stmt in none_guard.body)


def test_inference_step_has_no_hidden_state_success_fallback():
    source = BACKEND_SOURCE.read_text(encoding="utf-8")
    inference_step = _function_def(source, "inference_step")

    for node in ast.walk(inference_step):
        if not isinstance(node, ast.Return):
            continue
        value = node.value
        assert not (
            isinstance(value, ast.Tuple)
            and len(value.elts) == 2
            and isinstance(value.elts[0], ast.Name)
            and value.elts[0].id == "hidden_states"
            and isinstance(value.elts[1], ast.Constant)
            and value.elts[1].value is None
        )
