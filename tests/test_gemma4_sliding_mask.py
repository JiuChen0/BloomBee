"""Test the dual-attention-mask builder in WrappedGemma4Block.

Phase 3: full_attention layers get a plain causal mask; sliding_attention
layers get the causal mask AND-ed with a sliding-window cap. These tests
exercise the pure function so they don't need a checkpoint or GPU.
"""

import pytest
import torch

from bloombee.models.gemma4.block import _build_layer_type_mask, _merge_layer_type_mask


def _is_masked(mask, q, k):
    """Return True if position (q, k) is masked (additive value == -inf)."""
    val = mask[0, 0, q, k].item()
    return val < -1e30  # fp32 -inf after fp16 round-trip stays very negative


NEG_INF_OK = lambda v: v < -1e30


def test_full_layer_gets_plain_causal_mask():
    """Full layers: causal only. Every position in the past is visible;
    only strictly-future keys are masked."""
    mask = _build_layer_type_mask(
        layer_type="full_attention",
        sliding_window=1024,        # irrelevant for full
        query_length=4,
        past_length=10,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert mask.shape == (1, 1, 4, 14)

    # First query sits at absolute position 10. Keys 0..10 should be visible,
    # keys 11..13 masked.
    for k in range(11):
        assert mask[0, 0, 0, k].item() == 0.0, f"causal full: key {k} should be visible"
    for k in range(11, 14):
        assert NEG_INF_OK(mask[0, 0, 0, k].item()), f"causal full: key {k} should be masked"

    # Last query (idx 3, absolute position 13): all 14 keys visible.
    for k in range(14):
        assert mask[0, 0, 3, k].item() == 0.0, f"last query, key {k} should be visible"


def test_sliding_layer_masks_beyond_window():
    """Sliding layers: the causal mask PLUS a cap at sliding_window-1 keys
    of history behind the query."""
    W = 4  # sliding window of 4 tokens total (self + 3 earlier)
    mask = _build_layer_type_mask(
        layer_type="sliding_attention",
        sliding_window=W,
        query_length=2,
        past_length=10,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert mask.shape == (1, 1, 2, 12)

    # First query at absolute position 10; window allows keys 7..10 visible.
    for k in range(7):
        assert NEG_INF_OK(mask[0, 0, 0, k].item()), f"sliding out-of-window k={k} should be masked"
    for k in range(7, 11):
        assert mask[0, 0, 0, k].item() == 0.0, f"sliding in-window k={k} should be visible"
    for k in range(11, 12):
        assert NEG_INF_OK(mask[0, 0, 0, k].item()), f"sliding future k={k} should be masked"

    # Second query at absolute position 11; window allows keys 8..11 visible.
    for k in range(8):
        assert NEG_INF_OK(mask[0, 0, 1, k].item()), f"sliding q=1 out-of-window k={k} should be masked"
    for k in range(8, 12):
        assert mask[0, 0, 1, k].item() == 0.0, f"sliding q=1 in-window k={k} should be visible"


def test_sliding_layer_with_window_covering_full_context_equals_full_mask():
    """If the sliding window is wider than the history, the mask is
    identical to the full_attention mask."""
    full_mask = _build_layer_type_mask(
        layer_type="full_attention",
        sliding_window=None,
        query_length=4,
        past_length=6,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    sliding_mask = _build_layer_type_mask(
        layer_type="sliding_attention",
        sliding_window=1024,  # much wider than past+query=10
        query_length=4,
        past_length=6,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert torch.equal(full_mask, sliding_mask)


def test_empty_sequence_returns_empty_mask():
    """Edge case: query_length + past_length == 0."""
    mask = _build_layer_type_mask(
        layer_type="sliding_attention",
        sliding_window=4,
        query_length=0,
        past_length=0,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert mask.shape == (1, 1, 0, 0)


def test_fp16_mask_uses_finfo_min():
    """In fp16, finfo.min is approximately -65504, not -inf. Make sure we
    still produce the largest negative value (so softmax zeros it)."""
    mask = _build_layer_type_mask(
        layer_type="full_attention",
        sliding_window=None,
        query_length=2,
        past_length=2,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    assert mask.dtype == torch.float16
    # Check a masked position
    masked_val = mask[0, 0, 0, 3].item()
    assert masked_val < -1e3, f"fp16 mask should use finfo(float16).min, got {masked_val}"


def test_sliding_with_none_window_falls_back_to_plain_causal():
    """Defensive: if config somehow has sliding_window=None but layer_type
    is sliding_attention, we shouldn't crash — just act like full."""
    mask_none = _build_layer_type_mask(
        layer_type="sliding_attention",
        sliding_window=None,
        query_length=3,
        past_length=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    mask_full = _build_layer_type_mask(
        layer_type="full_attention",
        sliding_window=None,
        query_length=3,
        past_length=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert torch.equal(mask_none, mask_full)


def test_backend_mask_is_lifted_and_merged_with_sliding_cap():
    """BloomBee passes backend masks as [B, S, K]; sliding layers must still
    apply Gemma4's layer-local window instead of silently using full context.
    """
    backend_mask = torch.zeros((2, 2, 12), dtype=torch.float32)
    backend_mask[1, 0, 9] = torch.finfo(torch.float32).min

    merged = _merge_layer_type_mask(
        backend_mask,
        layer_type="sliding_attention",
        sliding_window=4,
        query_length=2,
        past_length=10,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert merged.shape == (2, 1, 2, 12)
    # First query is absolute position 10; only keys 7..10 are in-window.
    for batch in range(2):
        for key in range(7):
            assert NEG_INF_OK(merged[batch, 0, 0, key].item())
        assert merged[batch, 0, 0, 7].item() == 0.0
        assert merged[batch, 0, 0, 10].item() == 0.0
        assert NEG_INF_OK(merged[batch, 0, 0, 11].item())

    # Existing backend constraints are preserved while adding the sliding cap.
    assert NEG_INF_OK(merged[1, 0, 0, 9].item())
