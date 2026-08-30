from pathlib import Path


def test_mixtral_block_preserves_backend_attention_mask():
    source = Path("src/bloombee/models/mixtral/block.py").read_text()
    forward_start = source.index("    def forward(")
    reorder_start = source.index("    def _reorder_cache_from_bloom(", forward_start)
    forward_source = source[forward_start:reorder_start]

    assert "attention_mask = None" not in forward_source
    assert "attention_mask = causal.unsqueeze(0).unsqueeze(0)" in forward_source
    assert "attention_mask = attention_mask.unsqueeze(1)" in forward_source

