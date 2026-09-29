"""Source-level coverage for Falcon packed-GQA 4D cache reorder.

Falcon-40B/180B (`new_decoder_architecture`, H=128, n_kv=8) write packed KV
into the first `num_kv_heads` slots of an attention-head slab. Decode must
slice those packed heads before `_expand_states`. Taking all H rows makes
expand treat the slab as packed n_kv of (H/n_kv) fake batches, so the first
decode step cat()s a 2048-row past against 128-row new keys and crashes.

These tests avoid torch/transformers so they can run in the lightweight
automation environment (`python3 -m pytest --noconftest`).
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_BLOCK_SRC = Path("src/bloombee/models/falcon/block.py")


def _load_packed_kv_heads_helper():
    tree = ast.parse(_BLOCK_SRC.read_text())
    helper = None
    reorder = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_falcon_packed_kv_heads":
            helper = node
        if isinstance(node, ast.ClassDef) and node.name == "WrappedFalconBlock":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_reorder_cache_from_bloom_to_falcon":
                    reorder = item
    assert helper is not None, "_falcon_packed_kv_heads is missing"
    assert reorder is not None, "WrappedFalconBlock._reorder_cache_from_bloom_to_falcon is missing"
    called = [
        n.func.id
        for n in ast.walk(reorder)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert "_falcon_packed_kv_heads" in called
    namespace = {"FalconConfig": object}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "<helper>", "exec"), namespace)
    return namespace["_falcon_packed_kv_heads"]


def _expand_states(state: np.ndarray, *, num_kv_heads: int, num_attention_heads: int, groups: int) -> np.ndarray:
    packed, seq_len, head_dim = state.shape
    batch_size = packed // num_kv_heads
    viewed = state.reshape(batch_size, num_kv_heads, 1, seq_len, head_dim)
    expanded = np.broadcast_to(viewed, (batch_size, num_kv_heads, groups, seq_len, head_dim))
    return np.ascontiguousarray(expanded.reshape(batch_size * num_attention_heads, seq_len, head_dim))


def test_falcon_40b_4d_slab_uses_packed_kv_heads():
    packed_kv_heads = _load_packed_kv_heads_helper()
    config = SimpleNamespace(new_decoder_architecture=True, num_kv_heads=8, multi_query=False)
    assert packed_kv_heads(config, allocated_heads=128) == 8


def test_falcon_7b_mqa_4d_slab_uses_one_kv_head():
    packed_kv_heads = _load_packed_kv_heads_helper()
    config = SimpleNamespace(new_decoder_architecture=False, num_kv_heads=1, multi_query=True)
    assert packed_kv_heads(config, allocated_heads=71) == 1


def test_falcon_mha_4d_slab_keeps_all_heads():
    packed_kv_heads = _load_packed_kv_heads_helper()
    config = SimpleNamespace(new_decoder_architecture=False, num_kv_heads=32, multi_query=False)
    assert packed_kv_heads(config, allocated_heads=32) == 32


def test_falcon_40b_4d_reorder_expand_matches_attention_rows():
    """Prefill writes packed n_kv; decode must expand back to B*H, not B*(H/n_kv)*H."""
    batch_size = 1
    num_heads = 128
    num_kv_heads = 8
    groups = num_heads // num_kv_heads
    seq_len = 4
    head_dim = 2

    slab = np.zeros((batch_size, num_heads, seq_len, head_dim), dtype=np.float32)
    packed = np.arange(batch_size * num_kv_heads * seq_len * head_dim, dtype=np.float32).reshape(
        batch_size, num_kv_heads, seq_len, head_dim
    )
    slab[:, :num_kv_heads, :, :] = packed

    packed_rows = slab[:, :num_kv_heads, :, :].reshape(batch_size * num_kv_heads, seq_len, head_dim)
    expanded = _expand_states(
        packed_rows,
        num_kv_heads=num_kv_heads,
        num_attention_heads=num_heads,
        groups=groups,
    )
    assert expanded.shape == (batch_size * num_heads, seq_len, head_dim)
    np.testing.assert_array_equal(expanded[0], packed[0, 0])
    np.testing.assert_array_equal(expanded[groups - 1], packed[0, 0])
    np.testing.assert_array_equal(expanded[groups], packed[0, 1])

    buggy_rows = slab.reshape(batch_size * num_heads, seq_len, head_dim)
    buggy_expanded = _expand_states(
        buggy_rows,
        num_kv_heads=num_kv_heads,
        num_attention_heads=num_heads,
        groups=groups,
    )
    assert buggy_expanded.shape == (batch_size * (num_heads // num_kv_heads) * num_heads, seq_len, head_dim)
    assert buggy_expanded.shape[0] != batch_size * num_heads


def test_falcon_4d_reorder_source_does_not_treat_gqa_as_full_heads():
    src = _BLOCK_SRC.read_text()
    assert "num_kv_heads = 1 if (not self.config.new_decoder_architecture" not in src
    assert "_falcon_packed_kv_heads(self.config, H)" in src
    assert "_expand_states" in src
