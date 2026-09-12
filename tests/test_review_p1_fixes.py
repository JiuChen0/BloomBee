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
    parsed_mask, kv_pos, draft, prefill, hypo = unpack_s2s_extras(
        (hidden, keep, mask), {"s2s_padding_mask": True}
    )
    assert torch.equal(parsed_mask, mask)
    assert kv_pos is None and draft is None and prefill is None and hypo is None
    assert s2s_extra_tensor_names(False, {}) == ()
    assert s2s_extra_tensor_names(True, {"tree_attention_mask": mask}) == S2S_SPEC_TENSOR_NAMES


def test_s2s_push_forwards_hypo_ids_without_spec_flag():
    from bloombee.server.microbatch import (
        S2S_SPEC_TENSOR_NAMES,
        build_s2s_spec_tensors,
        s2s_extra_tensor_names,
        unpack_s2s_extras,
    )

    mask = torch.tensor([[True, True, False], [True, False, False]])
    hypo = torch.tensor([1, 0], dtype=torch.int64)
    packed = build_s2s_spec_tensors(is_spec_dec=False, tree_attention_mask=mask, hypo_ids=hypo)
    assert packed["hypo_ids"] is hypo
    names = s2s_extra_tensor_names(False, packed)
    assert names == ("tree_attention_mask", "hypo_ids")
    hidden = torch.zeros(2, 3, 4)
    keep = torch.arange(3)
    parsed_mask, kv_pos, draft, prefill, parsed_hypo = unpack_s2s_extras(
        (hidden, keep, mask, hypo),
        {"s2s_padding_mask": True, "s2s_hypo_ids": True},
    )
    assert torch.equal(parsed_mask, mask)
    assert torch.equal(parsed_hypo, hypo)
    assert kv_pos is None and draft is None and prefill is None

    hypo_only = build_s2s_spec_tensors(is_spec_dec=False, hypo_ids=hypo)
    assert s2s_extra_tensor_names(False, hypo_only) == ("hypo_ids",)
    _, _, _, _, parsed_hypo_only = unpack_s2s_extras(
        (hidden, keep, hypo), {"s2s_hypo_ids": True}
    )
    assert torch.equal(parsed_hypo_only, hypo)

    spec_with_hypo = build_s2s_spec_tensors(
        is_spec_dec=True,
        tree_attention_mask=mask,
        kv_cache_position_ids=torch.tensor([0]),
        draft_tokens=torch.tensor([1]),
        prefill_length=torch.tensor([3]),
        hypo_ids=hypo,
    )
    assert s2s_extra_tensor_names(True, spec_with_hypo) == S2S_SPEC_TENSOR_NAMES + ("hypo_ids",)
    tree, kv_pos, draft, prefill, parsed_spec_hypo = unpack_s2s_extras(
        (
            hidden,
            keep,
            spec_with_hypo["tree_attention_mask"],
            spec_with_hypo["kv_cache_position_ids"],
            spec_with_hypo["draft_tokens"],
            spec_with_hypo["prefill_length"],
            hypo,
        ),
        {"is_spec_dec": True, "s2s_hypo_ids": True},
    )
    assert torch.equal(tree, mask)
    assert torch.equal(parsed_spec_hypo, hypo)
    assert kv_pos is not None and draft is not None and prefill is not None


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


def test_shared_misc_helpers_cover_former_copies():
    from bloombee.utils.misc import dtype_name, flag_to_bool, slice_batch_aligned, to_int

    assert flag_to_bool(None) is False
    assert flag_to_bool(0) is False
    assert flag_to_bool(1) is True
    assert flag_to_bool(torch.empty(0)) is False
    assert flag_to_bool(torch.tensor([0, 1])) is True
    assert to_int("12") == 12
    assert to_int("nope", 7) == 7
    assert dtype_name(torch.float16) == "float16"
    assert dtype_name(None) == ""

    full = torch.arange(6).view(3, 2)
    sliced = slice_batch_aligned(full, 1, 3, 3)
    assert torch.equal(sliced, full[1:3])
    assert slice_batch_aligned(torch.tensor(5), 0, 1, 3).item() == 5
    assert slice_batch_aligned(None, 0, 1, 3) is None


def test_get_choice_uses_flexgen_helper():
    from bloombee.flexgen_utils.utils import get_choice

    choices = ["disk", "cpu", "gpu"]
    percents = [0.0, 50.0, 50.0]
    assert get_choice(10, percents, choices) == "cpu"
    assert get_choice(60, percents, choices) == "gpu"


def test_flexgen_raw_path_ignores_baked_local_llama_path():
    from bloombee.flexgen_utils.llama_config import _looks_like_hf_repo_id

    assert _looks_like_hf_repo_id("huggyllama/llama-7b")
    assert _looks_like_hf_repo_id("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    assert _looks_like_hf_repo_id("llama-7b")
    assert not _looks_like_hf_repo_id("/home/sgugger/tmp/llama/llama-7b/")
    assert not _looks_like_hf_repo_id(None)
    assert not _looks_like_hf_repo_id("./weights")


def test_falcon_default_revision_is_not_pinned_to_bin_commits():
    from pathlib import Path

    src = Path("src/bloombee/utils/auto_config.py").read_text()
    assert "4e2d06f0a7c6370ebabbc30c6f59377ae8f73d76" not in src
    assert "f1ba7d328c06aa6fbb4a8afd3c756f46d7e6b232" not in src
    assert "DEFAULT_REVISIONS: dict[str, str] = {}" in src


def _load_mixtral_expert_remap():
    import ast
    import re
    from collections import defaultdict
    from pathlib import Path

    tree = ast.parse(Path("src/bloombee/server/from_pretrained.py").read_text())
    needed = []
    wanted = {
        "_MIXTRAL_LEGACY_EXPERT_KEY_RE",
        "_MIXTRAL_LEGACY_GATE_KEY_RE",
        "_remap_mixtral_expert_state_dict",
    }
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    needed.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            needed.append(node)
    namespace = {"re": re, "defaultdict": defaultdict, "torch": torch}
    exec(compile(ast.Module(body=needed, type_ignores=[]), "<mixtral_remap>", "exec"), namespace)
    return namespace["_remap_mixtral_expert_state_dict"]


def test_mixtral_legacy_experts_remap_to_packed_tensors():
    remap = _load_mixtral_expert_remap()
    hidden, intermediate, num_experts = 4, 8, 2
    state = {
        "block_sparse_moe.gate.weight": torch.arange(num_experts * hidden, dtype=torch.float32).view(
            num_experts, hidden
        ),
        "self_attn.q_proj.weight": torch.ones(hidden, hidden),
    }
    for i in range(num_experts):
        state[f"block_sparse_moe.experts.{i}.w1.weight"] = torch.full(
            (intermediate, hidden), float(i + 1)
        )
        state[f"block_sparse_moe.experts.{i}.w3.weight"] = torch.full(
            (intermediate, hidden), float(i + 10)
        )
        state[f"block_sparse_moe.experts.{i}.w2.weight"] = torch.full(
            (hidden, intermediate), float(i + 20)
        )

    remapped = remap(state)
    assert "mlp.gate.weight" in remapped
    assert remapped["mlp.experts.gate_up_proj"].shape == (num_experts, 2 * intermediate, hidden)
    assert remapped["mlp.experts.down_proj"].shape == (num_experts, hidden, intermediate)
    assert torch.equal(remapped["mlp.experts.gate_up_proj"][0, :intermediate], torch.full((intermediate, hidden), 1.0))
    assert torch.equal(remapped["mlp.experts.gate_up_proj"][0, intermediate:], torch.full((intermediate, hidden), 10.0))
    assert torch.equal(remapped["mlp.experts.down_proj"][1], torch.full((hidden, intermediate), 21.0))
    assert remapped["self_attn.q_proj.weight"].shape == (hidden, hidden)
    assert not any(key.startswith("block_sparse_moe.") for key in remapped)

    packed = {
        "mlp.gate.weight": torch.ones(2, 4),
        "mlp.experts.gate_up_proj": torch.zeros(2, 16, 4),
        "mlp.experts.down_proj": torch.zeros(2, 4, 8),
    }
    unchanged = remap(dict(packed))
    assert set(unchanged) == set(packed)
