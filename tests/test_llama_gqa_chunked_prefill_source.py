"""FlexGen LLaMA GQA must not slice H-broadcast 4D past down to n_kv.

Default LLaMA serving uses FlexGen attention, which repeat_interleaves GQA KV
onto num_attention_heads. Backend chunked prefill (prompt longer than
max_chunk_length, ~2k tokens for Llama-3 8B B=1 fp16) concatenates that past
as [B, H, S, D] and feeds it back as layer_past.

Slicing to n_kv makes cache_read_buf (S, B*n_kv, D) while mha_gen_llama writes
(tgt, B*H, D). The next chunk then crashes in torch.cat.

This is independent of the full-batch KV write crash (PR #71): chunked prefill
never reaches _write_kvs on the second chunk.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace


_BLOCK_SRC = Path("src/bloombee/models/llama/block.py")


def _load_reorder_fn():
    tree = ast.parse(_BLOCK_SRC.read_text())
    methods = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "WrappedLlamaBlock":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in (
                    "_reorder_cache_from_bloom_to_llama",
                    "_num_key_value_heads",
                ):
                    methods.append(item)
    names = {item.name for item in methods}
    assert "_reorder_cache_from_bloom_to_llama" in names
    assert "_num_key_value_heads" in names
    namespace = {}
    exec(compile(ast.Module(body=methods, type_ignores=[]), "<llama_reorder>", "exec"), namespace)
    return namespace["_reorder_cache_from_bloom_to_llama"]


class _FakeTensor:
    def __init__(self, shape, sliced=False):
        self.shape = tuple(shape)
        self.sliced = sliced

    def dim(self):
        return len(self.shape)

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        new_shape = []
        for dim, sl in zip(self.shape, idx):
            if sl == slice(None):
                new_shape.append(dim)
            elif isinstance(sl, slice) and sl.stop is not None:
                start = 0 if sl.start is None else sl.start
                new_shape.append(sl.stop - start)
            else:
                new_shape.append(dim)
        new_shape.extend(self.shape[len(idx) :])
        return _FakeTensor(new_shape, sliced=True)


def test_4d_gqa_broadcast_past_is_not_sliced_to_nkv():
    reorder = _load_reorder_fn()
    block = SimpleNamespace(
        self_attn=SimpleNamespace(
            num_heads=32,
            num_key_value_heads=8,
            num_key_value_groups=4,
            head_dim=128,
        )
    )
    key = _FakeTensor((1, 32, 16, 128))
    value = _FakeTensor((1, 32, 16, 128))
    key_out, value_out = reorder(block, (key, value), batch_size=1, seq_length=16)
    assert key_out.shape == (1, 32, 16, 128)
    assert value_out.shape == (1, 32, 16, 128)
    assert not key_out.sliced
    assert not value_out.sliced


def test_4d_reorder_source_does_not_slice_nkv():
    tree = ast.parse(_BLOCK_SRC.read_text())
    method = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "WrappedLlamaBlock":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_reorder_cache_from_bloom_to_llama":
                    method = item
                    break
    assert method is not None
    four_d_if = next(stmt for stmt in method.body if isinstance(stmt, ast.If))
    assert any(
        isinstance(n, ast.Constant) and n.value == 4 for n in ast.walk(four_d_if.test)
    )
    # The 4D branch must return the H-broadcast tensors as-is, with no
    # key_states[:, :nkv] slice. Subscripts in the test itself are none.
    assert not [n for n in ast.walk(four_d_if) if isinstance(n, ast.Subscript)]
