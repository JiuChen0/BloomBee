import os
import threading

import pytest
import torch

from bloombee.flexgen_utils.llama_config import flexgen_np_cache_dirname, flexgen_np_cache_identity
from bloombee.flexgen_utils.pytorch_backend import rms_norm
from bloombee.server.memory_cache import _is_paged_kv_enabled
from bloombee.utils import lossless_transport as lt


def test_flexgen_cache_identity_separates_sources_and_revisions():
    a = flexgen_np_cache_identity("orgA/model")
    b = flexgen_np_cache_identity("orgB/model")
    c = flexgen_np_cache_identity("orgA/model", revision="abc")
    assert a != b
    assert a != c
    assert flexgen_np_cache_dirname("orgA/model", "model") != flexgen_np_cache_dirname("orgB/model", "model")


def test_paged_kv_ignores_debug_group(monkeypatch):
    monkeypatch.delenv("BLOOMBEE_PAGED_KV", raising=False)
    monkeypatch.setenv("BLOOMBEE_DEBUG_KV_CACHE", "1")
    assert _is_paged_kv_enabled() is False
    monkeypatch.setenv("BLOOMBEE_PAGED_KV", "1")
    assert _is_paged_kv_enabled() is True
    monkeypatch.setenv("BLOOMBEE_PAGED_KV", "0")
    assert _is_paged_kv_enabled() is False


def test_zstd_decompress_rejects_declared_size_mismatch():
    if lt._zstd is None:
        pytest.skip("zstandard is not installed")
    raw = b"x" * 64
    compressed = lt._zstd.ZstdCompressor(level=1).compress(raw)
    with pytest.raises(ValueError):
        lt._zstd_stream_decompress(lt._get_zstd_decompressor(), compressed, original_size=16)


def test_zstd_compressor_is_thread_local():
    if lt._zstd is None:
        pytest.skip("zstandard is not installed")
    held = [None, None]
    barrier = threading.Barrier(2)

    def worker(slot):
        held[slot] = lt._get_zstd_compressor(1)
        barrier.wait()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert held[0] is not None and held[1] is not None
    assert held[0] is not held[1]


def test_rms_norm_honors_epsilon():
    hidden = torch.full((1, 4), 0.001, dtype=torch.float32)
    weight = torch.ones(4, dtype=torch.float32)
    default = rms_norm(hidden, weight, variance_epsilon=1e-5)
    tight = rms_norm(hidden, weight, variance_epsilon=1e-6)
    assert not torch.allclose(default, tight)


def test_llama_inv_freq_applies_linear_scaling():
    from types import SimpleNamespace

    from bloombee.models.llama.flex_llama import compute_llama_inv_freq

    head_dim = 8
    base = SimpleNamespace(rope_theta=10000.0, rope_scaling=None, max_position_embeddings=2048)
    scaled = SimpleNamespace(
        rope_theta=10000.0,
        rope_scaling={"type": "linear", "factor": 8.0, "rope_type": "linear"},
        max_position_embeddings=2048,
    )
    freq_base = compute_llama_inv_freq(base, head_dim)
    freq_scaled = compute_llama_inv_freq(scaled, head_dim)
    assert freq_base.shape == freq_scaled.shape == (head_dim // 2,)
    assert not torch.allclose(freq_base, freq_scaled)


def test_decode_mask_buffer_is_bounded():
    from bloombee.server.backend import TransformerBackend

    class _Dummy:
        def _get_decode_mask_scores(self, batch_size, src_len, device):
            return TransformerBackend._get_decode_mask_scores(self, batch_size, src_len, device)

    dummy = _Dummy()
    dummy._decode_mask_zeros = None
    first = dummy._get_decode_mask_scores(2, 8, torch.device("cpu"))
    second = dummy._get_decode_mask_scores(2, 128, torch.device("cpu"))
    third = dummy._get_decode_mask_scores(2, 32, torch.device("cpu"))
    assert first.shape == (2, 1, 8)
    assert second.shape == (2, 1, 128)
    assert third.shape == (2, 1, 32)
    assert dummy._decode_mask_zeros.shape[-1] == 128
    assert dummy._decode_mask_zeros.numel() == 2 * 1 * 128


def test_spec_history_gathers_accepted_kv_not_prefix():
    from bloombee.client.inference_session import _ServerInferenceSession

    holder = type("Holder", (), {})()
    holder.history = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
    holder._position = 5
    _ServerInferenceSession.compact_history_to_accepted_kv(holder, torch.tensor([2, 4]))
    assert holder.history.squeeze().tolist() == [0.0, 1.0, 2.0, 4.0]
    assert holder._position == 4


def test_client_cache_budget_uses_allocator_units():
    from types import SimpleNamespace

    from bloombee.client.routing.sequence_manager import RemoteSequenceManager

    span = SimpleNamespace(length=10, server_info=SimpleNamespace(cache_tokens_left=4096))
    assert RemoteSequenceManager._has_cache_for(span, 1024) is True
    assert RemoteSequenceManager._has_cache_for(span, 8192) is False


def test_verify_path_uses_leaf_bonus_distribution():
    from bloombee.models.llama.spec_decoding_verify import verify_path

    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    draft = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    tokens = torch.tensor([0, 1])
    bonus = torch.tensor([1.0, 0.0])
    gen = torch.Generator().manual_seed(0)
    committed, accepted = verify_path(target, draft, tokens, generator=gen, bonus_probs=bonus)
    assert accepted == 2
    assert committed[-1] == 0


def test_s2s_push_forwards_padding_mask_without_spec_flag():
    from bloombee.server.microbatch import (
        S2S_SPEC_TENSOR_NAMES,
        s2s_extra_tensor_names,
        unpack_s2s_extras,
    )

    mask = torch.tensor([[True, True, False], [True, False, False]])
    names = s2s_extra_tensor_names(False, {"tree_attention_mask": mask})
    assert names == ("tree_attention_mask",)
    hidden = torch.zeros(2, 3, 4)
    keep = torch.arange(3)
    parsed_mask, kv_pos, draft, prefill = unpack_s2s_extras(
        (hidden, keep, mask), {"s2s_padding_mask": True}
    )
    assert torch.equal(parsed_mask, mask)
    assert kv_pos is None and draft is None and prefill is None
    assert s2s_extra_tensor_names(False, {}) == ()
    assert s2s_extra_tensor_names(True, {"tree_attention_mask": mask}) == S2S_SPEC_TENSOR_NAMES


def test_append_sequence_history_grows_without_recopying_prefix():
    from bloombee.client.inference_session import append_sequence_history

    first = torch.arange(8, dtype=torch.float32).view(1, 2, 4)
    history, storage = append_sequence_history(None, first)
    assert history.shape == (1, 2, 4)
    assert storage.shape[1] >= 4
    storage_ptr = storage.data_ptr()
    for i in range(6):
        nxt = torch.full((1, 1, 4), float(i + 2))
        history, storage = append_sequence_history(history, nxt, storage)
    assert history.shape == (1, 8, 4)
    assert torch.equal(history[:, :2], first)
    assert history[0, -1, 0].item() == 7.0
    assert storage.shape[1] > history.shape[1]
    assert storage.data_ptr() == storage_ptr


def test_quantize_module_warns_for_non_none():
    import warnings

    import torch.nn as nn

    from bloombee.utils.convert_block import QuantType, quantize_module

    mod = nn.Linear(2, 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = quantize_module(mod, quant_type=QuantType.INT8)
    assert out is mod
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
