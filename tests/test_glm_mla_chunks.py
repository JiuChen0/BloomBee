"""Chunked GLM prefill must keep the two MLA latent widths intact."""

import importlib.util
from pathlib import Path

import pytest
import torch


def _load_mla_chunk():
    path = Path(__file__).resolve().parents[1] / "src" / "bloombee" / "server" / "mla_chunk.py"
    spec = importlib.util.spec_from_file_location("mla_chunk_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mla_chunk = _load_mla_chunk()


def _pair(batch, seq, kv_dim, rot_dim, fill):
    kv = torch.full((batch, 1, seq, kv_dim), fill, dtype=torch.float32)
    rot = torch.full((batch, 1, seq, rot_dim), fill + 0.5, dtype=torch.float32)
    return kv, rot


def test_shared_head_view_cannot_represent_glm_flash_latents():
    """The bloom chunk conversion views K and V with K's last dimension."""
    kv_dim, rot_dim = 512, 64  # GLM-4.7-Flash kv_lora_rank, qk_rope_head_dim
    kv, rot = _pair(batch=1, seq=4, kv_dim=kv_dim, rot_dim=rot_dim, fill=1.0)
    bsz, heads, seq, head_dim = kv.shape
    with pytest.raises(RuntimeError):
        rot.reshape(bsz * heads, seq, head_dim)


def test_chunked_prefill_preserves_prefix_and_new_latents():
    # Default server budget is 256MiB. GLM-4.7-Flash has 20 attention heads.
    # A bf16 batch-1 prompt of 4096 tokens costs 20 * 2 * 4096 = 163840 bytes
    # per token of attention scores, so the chunk length is
    # 256MiB // 163840 = 1638 and the prompt is split. This test uses the
    # real latent widths with shorter stand-in chunks.
    kv_dim, rot_dim = 512, 64
    prefix_len, first, second = 5, 4, 3
    prefix = _pair(2, prefix_len, kv_dim, rot_dim, fill=1.0)
    chunk_a = _pair(2, first, kv_dim, rot_dim, fill=2.0)
    chunk_b = _pair(2, second, kv_dim, rot_dim, fill=3.0)

    chunks = []
    layer_past = mla_chunk.record_glm_mla_chunk(prefix, chunk_a, chunks)
    layer_past = mla_chunk.record_glm_mla_chunk(layer_past, chunk_b, chunks)
    written = mla_chunk.concat_mla_chunks(chunks)

    assert layer_past[0].shape == (2, 1, prefix_len + first + second, kv_dim)
    assert layer_past[1].shape == (2, 1, prefix_len + first + second, rot_dim)
    assert written[0].shape == (2, 1, first + second, kv_dim)
    assert written[1].shape == (2, 1, first + second, rot_dim)
    assert torch.equal(layer_past[0][:, :, :prefix_len], prefix[0])
    assert torch.equal(layer_past[1][:, :, prefix_len : prefix_len + first], chunk_a[1])
    assert torch.equal(written[0][:, :, :first], chunk_a[0])
    assert torch.equal(written[0][:, :, first:], chunk_b[0])
    assert torch.equal(written[1][:, :, first:], chunk_b[1])


def test_fresh_prefill_chunk_does_not_require_a_prefix():
    chunk = _pair(1, 3, 512, 64, fill=4.0)
    chunks = []
    layer_past = mla_chunk.record_glm_mla_chunk(None, chunk, chunks)
    written = mla_chunk.concat_mla_chunks(chunks)
    assert torch.equal(layer_past[0], chunk[0])
    assert torch.equal(written[1], chunk[1])


def test_width_mismatch_is_rejected():
    past = _pair(1, 2, 512, 64, fill=1.0)
    new = _pair(1, 2, 256, 64, fill=2.0)
    with pytest.raises(RuntimeError, match="latent widths"):
        mla_chunk.record_glm_mla_chunk(past, new, [])


def test_missing_chunk_cache_is_rejected():
    with pytest.raises(RuntimeError, match="returned no cache"):
        mla_chunk.record_glm_mla_chunk(None, None, [])
