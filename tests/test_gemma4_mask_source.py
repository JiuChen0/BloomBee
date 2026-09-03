from pathlib import Path


def _forward_body(source: str) -> str:
    marker = "    def forward("
    start = source.index(marker)
    next_def = source.index("\n    def ", start + len(marker))
    return source[start:next_def]


def test_gemma4_external_sliding_mask_merges_layer_window():
    source = Path("src/bloombee/models/gemma4/block.py").read_text()
    body = _forward_body(source)

    assert "attention_mask is None or self.is_sliding" in body
    assert "attention_mask.dtype == torch.bool" in body
    assert "attention_mask = attention_mask[:, None, None, :]" in body
    assert "attention_mask = attention_mask.unsqueeze(1)" in body
    assert "torch.minimum(attention_mask, layer_type_mask)" in body
