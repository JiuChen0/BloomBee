"""Immutable versioning, isolation, expiry, bounded storage and recompute fallback."""

import pytest
import torch
from test_lora_span import block_reference, setup_span

from bloombee.server.lora_cache import CachedLoRASpan, TensorLRU
from bloombee.server.lora_execution import LoRASpanExecutor


def setup_cache(mode="span_cached_cpu", **kwargs):
    backends, params, x, g, metadata = setup_span()
    executor = LoRASpanExecutor(backends, checkpoint_bytes=100000)
    cache = CachedLoRASpan(executor, **kwargs)
    metadata.update(
        cache_owner="owner-a",
        cache_job="a" * 32,
        cache_request="b" * 32,
        adapter_version=1,
        execution_mode=mode,
        cache_requires_backward=True,
    )
    return backends, cache, params, x, g, metadata


@pytest.mark.parametrize("mode", ["span_cached_cpu", "span_cached_gpu"])
def test_cached_forward_backward_duplicate_and_fallback_match_block(mode, monkeypatch):
    from bloombee.server import lora

    backends, cache, params, x, g, meta = setup_cache(mode)
    expected, expected_grads = block_reference(backends, params, x, g, meta)
    calls = []
    original = lora.run_lora_block

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(lora, "run_lora_block", counted)

    def unexpected_block_validation(*args):
        pytest.fail("A validated adapter snapshot was rescanned by block execution")

    monkeypatch.setattr(lora, "validate_lora_parameters", unexpected_block_validation)
    out = cache(x, torch.empty(0), {**meta, "backward": False}, params)
    assert out[-1].tolist() == [1, 0]
    torch.testing.assert_close(out[0], expected, rtol=0, atol=0)
    result = cache(x, g, {**meta, "backward": True}, ())
    assert result[-1].tolist() == [1, 1] and not cache.activations.entries
    assert len(calls) == 6
    for a, b in zip(result[:-1], expected_grads):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Repeated backward reconstructs inputs; no worker-side gradient accumulation.
    retry = cache(x, g, {**meta, "backward": True}, ())
    assert retry[-1].tolist() == [1, 0]
    for a, b in zip(retry[:-1], expected_grads):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_version_and_checkpoint_bindings_are_enforced_and_owners_are_isolated():
    _, cache, params, x, g, meta = setup_cache()
    cache(x, torch.empty(0), {**meta, "backward": False}, params)
    modified = [p.clone() for p in params]
    modified[0].add_(1)
    with pytest.raises(ValueError, match="version reused"):
        cache(x, g, {**meta, "backward": True}, modified)
    for changed in ({"seed": meta["seed"] + 1}, {"adapter_version": 2}):
        with pytest.raises(ValueError, match="Checkpoint request"):
            cache(x, g, {**meta, **changed, "backward": True}, modified if "adapter_version" in changed else ())
    with pytest.raises(ValueError, match="Checkpoint request"):
        cache(x + 1, g, {**meta, "backward": True}, ())
    for changed in ({"cache_owner": "owner-b"}, {"cache_job": "c" * 32}):
        result = cache(x, g, {**meta, **changed, "backward": True}, ())
        assert result[0].tolist() == [-1, 0]
    assert cache(x, g, {**meta, "backward": True}, ())[-1].tolist() == [1, 1]


def test_cache_expiration_and_zero_budget_do_not_change_gradients():
    backends, cache, params, x, g, meta = setup_cache(ttl=1.0)
    _, expected = block_reference(backends, params, x, g, meta)
    now = [0.0]
    cache.adapters.clock = cache.activations.clock = lambda: now[0]
    cache(x, torch.empty(0), {**meta, "backward": False}, params)
    now[0] = 2.0
    assert cache(x, g, {**meta, "backward": True}, ())[0].tolist() == [-1, 0]
    for current in (cache, CachedLoRASpan(cache.executor, adapter_bytes=0, activation_bytes=0)):
        result = current(x, g, {**meta, "backward": True}, params)
        for a, b in zip(result[:-1], expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert current.adapters.bytes <= current.adapters.limit_bytes
        assert current.activations.bytes <= current.activations.limit_bytes


def test_lru_byte_and_entry_budgets_and_expiration():
    now = [0.0]
    store = TensorLRU(10, 1.0, max_entries=2, clock=lambda: now[0])
    assert store.put("a", 1, 4) and store.put("b", 2, 4)
    assert store.get("a") == 1
    assert store.put("c", 3, 4) and store.get("b") is None
    assert store.bytes == 8
    assert not store.put("oversized", 9, 11) and store.bytes == 8
    now[0] = 2.0
    assert store.get("a") is None and store.bytes == 0


def test_eval_forward_does_not_retain_activations():
    _, cache, params, x, _, meta = setup_cache()
    cache(x, torch.empty(0), {**meta, "backward": False, "cache_requires_backward": False}, params)
    assert not cache.activations.entries and cache.adapters.bytes > 0


def test_runtime_shutdown_releases_all_retained_tensors():
    _, cache, params, x, _, meta = setup_cache()
    cache.executor.cache = cache
    cache(x, torch.empty(0), {**meta, "backward": False}, params)
    assert cache.adapters.bytes and cache.activations.bytes
    cache.executor.clear_cache()
    assert not hasattr(cache.executor, "cache")
    assert cache.adapters.bytes == cache.activations.bytes == 0
    assert not cache.adapters.entries and not cache.activations.entries
