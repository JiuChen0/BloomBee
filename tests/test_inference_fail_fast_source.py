from pathlib import Path


BACKEND_SOURCE = Path("src/bloombee/server/backend.py")


def _backend_source() -> str:
    return BACKEND_SOURCE.read_text()


def test_inference_step_does_not_return_identity_fallback():
    source = _backend_source()

    assert "return (hidden_states, None)" not in source
    assert "return original input" not in source.lower()


def test_run_block_forward_none_result_is_fatal():
    source = _backend_source()

    assert "raise RuntimeError(\"module.forward returned None\")" in source
    assert "module.forward returned None" in source
