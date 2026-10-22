"""Nemotron-H full attention must accept BloomBee's batched score mask.

Server inference builds an additive mask of shape [batch, query, key]. HF
eager attention adds that tensor to scores of shape [batch, heads, query, key].
A 3D mask lines batch up with the head dimension and crashes whenever the
batch is larger than 1, including the default server pool that merges
concurrent requests.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig


def _load_block_class():
    """Import the block without executing bloombee/__init__.py.

    Package init pulls the server and hivemind. This block only needs torch
    and transformers, so load it as a package submodule directly. If the real
    package was already imported, reuse that class.
    """
    loaded = sys.modules.get("bloombee.models.nemotron_h.block")
    if loaded is not None and hasattr(loaded, "WrappedNemotronHBlock"):
        return loaded.WrappedNemotronHBlock

    root = Path(__file__).resolve().parents[1] / "src" / "bloombee"
    packages = {
        "bloombee": root,
        "bloombee.utils": root / "utils",
        "bloombee.models": root / "models",
        "bloombee.models.nemotron_h": root / "models" / "nemotron_h",
    }
    for name, path in packages.items():
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            module.__package__ = name
            sys.modules[name] = module

    def load(qualname, path):
        spec = importlib.util.spec_from_file_location(qualname, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualname] = module
        spec.loader.exec_module(module)
        return module

    load("bloombee.utils.cache_compat", root / "utils" / "cache_compat.py")
    return load(
        "bloombee.models.nemotron_h.block",
        root / "models" / "nemotron_h" / "block.py",
    ).WrappedNemotronHBlock


WrappedNemotronHBlock = _load_block_class()


def _block():
    config = NemotronHConfig(
        vocab_size=32,
        hidden_size=32,
        layers_block_type=["full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        mamba_num_heads=4,
        mamba_head_dim=8,
        ssm_state_size=4,
        n_groups=1,
        conv_kernel=2,
        max_position_embeddings=32,
        num_logits_to_keep=1,
    )
    config._attn_implementation = "eager"
    return WrappedNemotronHBlock(config, layer_idx=0).eval()


def _scores(batch, query, key):
    # Same layout as TransformerBackend.convert_mask_to_scores: 0 keep, large
    # negative mask, with a causal lower triangle.
    allowed = torch.ones(batch, query, key, dtype=torch.bool).tril() if query == key else torch.ones(
        batch, query, key, dtype=torch.bool
    )
    scores = torch.full((batch, query, key), -65504.0)
    scores[allowed] = 0.0
    return scores


@pytest.mark.parametrize("batch", [1, 2])
def test_batched_prefill_accepts_bloombee_score_mask(batch):
    torch.manual_seed(0)
    block = _block()
    seq = 4
    hidden = torch.randn(batch, seq, block.config.hidden_size)
    output, present = block(hidden, attention_mask=_scores(batch, seq, seq), use_cache=True)
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    key, value = present
    assert key.shape[0] == batch * block.config.num_key_value_heads
    assert value.shape[0] == batch * block.config.num_key_value_heads


@pytest.mark.parametrize("batch", [1, 2])
def test_batched_decode_accepts_bloombee_score_mask(batch):
    torch.manual_seed(0)
    block = _block()
    past = 5
    heads = block.config.num_key_value_heads
    head_dim = block.config.head_dim
    # BloomBee KV layout: K [B*Hkv, D, S], V [B*Hkv, S, D]
    past_k = torch.randn(batch * heads, head_dim, past)
    past_v = torch.randn(batch * heads, past, head_dim)
    hidden = torch.randn(batch, 1, block.config.hidden_size)
    output, present = block(
        hidden,
        attention_mask=_scores(batch, 1, past + 1),
        layer_past=(past_k, past_v),
        use_cache=True,
    )
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    key, value = present
    assert key.shape == (batch * heads, head_dim, 1)
    assert value.shape == (batch * heads, 1, head_dim)
