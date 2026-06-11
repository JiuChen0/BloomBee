from pathlib import Path


def test_backend_inference_step_does_not_silently_return_input_on_error():
    source = Path("src/bloombee/server/backend.py").read_text()

    assert "return (hidden_states, None)" not in source
    assert "module.forward returned None for block" in source
    assert "raise" in source[source.index("inference_step failed for block"):]
