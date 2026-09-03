from pathlib import Path


def _function_body(source: str, name: str) -> str:
    marker = f"    def {name}("
    start = source.index(marker)
    next_def = source.index("\n    def ", start + len(marker))
    return source[start:next_def]


def test_llama_4d_cache_reorder_slices_gqa_kv_heads():
    source = Path("src/bloombee/models/llama/block.py").read_text()
    body = _function_body(source, "_reorder_cache_from_bloom_to_llama")
    branch_start = body.index("if key_states.dim() == 4 and value_states.dim() == 4:")
    branch_end = body.index("if key_states.dim() == 3:", branch_start)
    branch = body[branch_start:branch_end]

    assert "nkv = self._num_key_value_heads()" in branch
    assert "key_states[:, :nkv, :, :]" in branch
    assert "value_states[:, :nkv, :, :]" in branch
    assert "return key_states, value_states" not in branch
