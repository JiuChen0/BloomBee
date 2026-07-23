from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text()


def test_backend_block_forward_none_fails_fast():
    """A failed block must not be reported as a successful identity pass."""
    source = _read("src/bloombee/server/backend.py")

    assert "module.forward returned None during inference" in source
    assert "raise RuntimeError" in source
    assert "return (hidden_states, None)" not in source


def test_mixtral_preserves_backend_attention_mask():
    """Mixtral must keep backend masks for causal/speculative restrictions."""
    source = _read("src/bloombee/models/mixtral/block.py")

    assert "attention_mask = None" not in source
    assert "attention_mask = causal.unsqueeze(0).unsqueeze(0)" in source
    assert "attention_mask = attention_mask.unsqueeze(1)" in source
