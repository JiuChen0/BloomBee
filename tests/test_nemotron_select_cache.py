"""Nemotron-H KV reads must keep packed key/value heads when the batch is larger than 1."""

from types import SimpleNamespace

import pytest
import torch

from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
from bloombee.flexgen_utils.compression import CompressionConfig
from bloombee.flexgen_utils.policy import Policy
from bloombee.flexgen_utils.pytorch_backend import TorchDevice, TorchTensor, fix_recursive_import
from bloombee.server.memory_cache_manager import KVCacheManager


def _cpu_cache_manager(config):
    fix_recursive_import()
    device = TorchDevice("cpu")
    env = ExecutionEnv(gpu=device, cpu=device)
    compression = CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False)
    policy = Policy(
        gpu_batch_size=1,
        num_gpu_batches=1,
        w_gpu_percent=100,
        w_cpu_percent=0,
        cache_gpu_percent=0,
        cache_cpu_percent=100,
        act_gpu_percent=100,
        act_cpu_percent=0,
        overlap=False,
        sep_layer=True,
        pin_weight=False,
        cpu_cache_compute=False,
        attn_sparsity=1.0,
        compress_weight=False,
        comp_weight_config=compression,
        compress_cache=False,
        comp_cache_config=compression,
    )
    return KVCacheManager(1024, None, policy, env, config)


@pytest.mark.parametrize("batch", [1, 2, 16])
def test_nemotron_select_cache_keeps_packed_kv_heads(batch):
    """Batched Nemotron decode must see [B, num_key_value_heads, S, D].

    Query heads are 32 and KV heads are 2, so batch 2 is 4 packed rows.
    Viewing those rows with num_attention_heads yields [1, 4, S, D] (or
    [1, 32, S, D] once the row count reaches 32) and the next token's cache
    update crashes.
    """
    query_heads, kv_heads, seq, dim = 32, 2, 3, 4
    config = SimpleNamespace(
        model_type="nemotron_h",
        num_attention_heads=query_heads,
        num_key_value_heads=kv_heads,
        hidden_size=query_heads * dim,
    )
    manager = _cpu_cache_manager(config)
    rows = batch * kv_heads
    values = torch.arange(seq * rows * dim, dtype=torch.float32).view(seq, rows, dim)
    caches = (
        (
            TorchTensor.create_from_torch(values, manager.env.cpu),
            TorchTensor.create_from_torch(values + 10_000, manager.env.cpu),
        ),
    )
    try:
        k_pkv, v_pkv, _ = manager.select_cache(
            prefix_length=seq,
            cache_tensors=caches,
            requested_rows=batch,
        )
        assert k_pkv.shape == (batch, kv_heads, seq, dim)
        assert v_pkv.shape == (batch, kv_heads, seq, dim)
        expected = values.view(seq, batch, kv_heads, dim).permute(1, 2, 0, 3)
        torch.testing.assert_close(k_pkv, expected)
        torch.testing.assert_close(v_pkv, expected + 10_000)
    finally:
        manager._reorder_executor.shutdown(wait=True)


def test_attention_head_slab_select_cache_unchanged():
    """Non-Nemotron slabs stay strided by query heads."""
    batch, heads, seq, dim = 2, 8, 3, 4
    config = SimpleNamespace(num_attention_heads=heads, num_key_value_heads=2, num_key_value_groups=4, hidden_size=32)
    manager = _cpu_cache_manager(config)
    rows = batch * heads
    values = torch.arange(seq * rows * dim, dtype=torch.float32).view(seq, rows, dim)
    caches = (
        (
            TorchTensor.create_from_torch(values, manager.env.cpu),
            TorchTensor.create_from_torch(values + 5, manager.env.cpu),
        ),
    )
    try:
        k_pkv, v_pkv, _ = manager.select_cache(
            prefix_length=seq,
            cache_tensors=caches,
            requested_rows=batch,
        )
        assert k_pkv.shape == (batch, heads, seq, dim)
        expected = values.view(seq, batch, heads, dim).permute(1, 2, 0, 3)
        torch.testing.assert_close(k_pkv, expected)
        torch.testing.assert_close(v_pkv, expected + 5)
    finally:
        manager._reorder_executor.shutdown(wait=True)
