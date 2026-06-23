import ast
from pathlib import Path


BACKEND_SOURCE = Path(__file__).resolve().parents[1] / "src" / "bloombee" / "server" / "backend.py"


def _method_source(method_name: str) -> str:
    source = BACKEND_SOURCE.read_text()
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == "TransformerBackend":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return ast.get_source_segment(source, item) or ""
    raise AssertionError(f"TransformerBackend.{method_name} not found")


def test_inference_step_does_not_fallback_to_passthrough_hidden_states():
    source = _method_source("inference_step")

    assert "return (hidden_states, None)" not in source
    assert "return hidden_states, None" not in source
    assert "raise" in source


def test_run_block_forward_treats_missing_forward_result_as_failure():
    source = _method_source("_run_block_forward")

    assert "raise RuntimeError" in source
    assert "return None" not in source


def test_spec_rotary_prefill_uses_normalized_lengths_on_hidden_device():
    source = _method_source("inference_step")

    assert "_normalize_spec_prefill_length" in source
    assert "device=hidden_states.device" in source
    assert 'device="cuda"' not in source
