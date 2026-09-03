from pathlib import Path


def _forward_body(source: str) -> str:
    marker = "    def forward("
    start = source.index(marker)
    next_def = source.index("\n    def ", start + len(marker))
    return source[start:next_def]


def test_mixtral_block_preserves_backend_attention_mask():
    source = Path("src/bloombee/models/mixtral/block.py").read_text()
    body = _forward_body(source)

    assert "attention_mask = None" not in body
    assert "if attention_mask is None:" in body
    assert "torch.triu" in body
    assert "elif attention_mask.dim() == 3:" in body
    assert "attention_mask = attention_mask.unsqueeze(1)" in body
