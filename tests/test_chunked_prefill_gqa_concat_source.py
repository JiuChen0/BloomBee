import ast
from pathlib import Path
from typing import Optional, Tuple

import torch


_BACKEND_SRC = Path("src/bloombee/server/backend.py")


def _load_concat_helper():
    tree = ast.parse(_BACKEND_SRC.read_text())
    helper = None
    layout = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "TransformerBackend":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_concat_llama_kvs":
                    helper = item
                if isinstance(item, ast.FunctionDef) and item.name == "_kv_to_llama_layout":
                    layout = item
    assert helper is not None, "_concat_llama_kvs is missing"
    assert layout is not None, "_kv_to_llama_layout is missing"
    helper.decorator_list = []
    layout.decorator_list = []
    future = ast.parse("from __future__ import annotations").body[0]
    namespace = {"torch": torch, "Optional": Optional, "Tuple": Tuple}
    exec(
        compile(ast.Module(body=[future, layout, helper], type_ignores=[]), "<backend-kv>", "exec"),
        namespace,
    )
    return namespace["_concat_llama_kvs"], namespace["_kv_to_llama_layout"]


def test_inference_step_concats_chunk_past_in_llama_layout():
    src = _BACKEND_SRC.read_text()
    assert "chunk_layer_past_llama = self._kv_to_llama_layout(chunk_layer_past, batch_size)" in src
    assert "chunk_layer_past = self._concat_llama_kvs(chunk_layer_past_llama, new_kvs_llama)" in src


def test_concat_aligns_select_cache_heads_with_packed_gqa_new_kvs():
    concat_kvs, to_llama = _load_concat_helper()
    batch, attn_heads, kv_heads, prefix, chunk, dim = 1, 32, 8, 128, 64, 16

    past_key = torch.arange(batch * attn_heads * prefix * dim, dtype=torch.float32).view(
        batch, attn_heads, prefix, dim
    )
    past_value = past_key + 0.5
    new_key_bloom = torch.arange(batch * kv_heads * dim * chunk, dtype=torch.float32).view(
        batch * kv_heads, dim, chunk
    )
    new_value_bloom = torch.arange(batch * kv_heads * chunk * dim, dtype=torch.float32).view(
        batch * kv_heads, chunk, dim
    )

    new_llama = to_llama((new_key_bloom, new_value_bloom), batch)
    merged_key, merged_value = concat_kvs((past_key, past_value), new_llama)

    assert merged_key.shape == (batch, kv_heads, prefix + chunk, dim)
    assert merged_value.shape == (batch, kv_heads, prefix + chunk, dim)
    torch.testing.assert_close(merged_key[:, :, :prefix], past_key[:, :kv_heads])
    torch.testing.assert_close(merged_value[:, :, :prefix], past_value[:, :kv_heads])
    torch.testing.assert_close(
        merged_key[:, :, prefix:],
        new_key_bloom.permute(0, 2, 1).view(batch, kv_heads, chunk, dim),
    )


def test_concat_keeps_matching_flexgen_broadcast_heads():
    concat_kvs, _ = _load_concat_helper()
    batch, heads, prefix, chunk, dim = 2, 32, 8, 4, 8
    left_key = torch.ones(batch, heads, prefix, dim)
    right_key = torch.full((batch, heads, chunk, dim), 2.0)
    merged_key, merged_value = concat_kvs((left_key, left_key), (right_key, right_key))
    assert merged_key.shape == (batch, heads, prefix + chunk, dim)
    assert merged_value.shape == (batch, heads, prefix + chunk, dim)
    torch.testing.assert_close(merged_key[:, :, :prefix], left_key)
    torch.testing.assert_close(merged_key[:, :, prefix:], right_key)


def test_concat_rejects_mixed_ranks_instead_of_dropping_prefix():
    concat_kvs, _ = _load_concat_helper()
    left = (torch.zeros(1, 8, 4, 2), torch.zeros(1, 8, 4, 2))
    right = (torch.zeros(8, 2, 3), torch.zeros(8, 3, 2))
    try:
        concat_kvs(left, right)
    except RuntimeError as exc:
        assert "mixed ranks" in str(exc)
    else:
        raise AssertionError("expected mixed-rank concat to fail closed")
