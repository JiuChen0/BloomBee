"""GLM-4.7-Flash RoPE frequencies must survive the server dtype cast.

Server loading ends in ``block.to(dtype)``. A bare decoder layer does not
inherit a full-model fp32 exclusion, so that cast used to store ``inv_freq``
in bf16/fp16. ``Glm4MoeLiteRotaryEmbedding`` promotes the buffer with
``.float()`` afterwards, which cannot restore the rounded mantissa. At a few
thousand tokens the resulting angle error is already tens of degrees.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _load_wrapped_block():
    """Import the wrapper without initializing the full BloomBee package.

    The stub is removed before returning so later tests in the same process
    can still import the real package.
    """
    if "bloombee" in sys.modules and hasattr(sys.modules["bloombee"], "__file__"):
        from bloombee.models.glm4_moe_lite.block import WrappedGlm4MoeLiteBlock

        return WrappedGlm4MoeLiteBlock

    root = Path(__file__).resolve().parents[1] / "src"
    package = types.ModuleType("bloombee")
    package.__path__ = [str(root / "bloombee")]
    package.__package__ = "bloombee"
    utils_pkg = types.ModuleType("bloombee.utils")
    utils_pkg.__path__ = [str(root / "bloombee" / "utils")]
    utils_pkg.__package__ = "bloombee.utils"
    cache_compat = types.ModuleType("bloombee.utils.cache_compat")
    cache_compat.make_empty_kv_cache = lambda *args, **kwargs: None
    cache_compat.make_past_kv_cache = lambda *args, **kwargs: None
    cache_compat.read_kv_from_cache = lambda *args, **kwargs: (None, None)

    module_name = "bloombee.models.glm4_moe_lite.block_under_test"
    sys.modules["bloombee"] = package
    sys.modules["bloombee.utils"] = utils_pkg
    sys.modules["bloombee.utils.cache_compat"] = cache_compat
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            root / "bloombee" / "models" / "glm4_moe_lite" / "block.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.WrappedGlm4MoeLiteBlock
    finally:
        for name in (
            "bloombee",
            "bloombee.utils",
            "bloombee.utils.cache_compat",
            module_name,
        ):
            sys.modules.pop(name, None)


def _mini_config():
    from transformers.models.glm4_moe_lite import Glm4MoeLiteConfig

    return Glm4MoeLiteConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=1,
        num_experts_per_tok=1,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        qk_nope_head_dim=16,
        max_position_embeddings=2048,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        mlp_layer_types=["dense"],
    )


def test_glm_rotary_inv_freq_survives_bf16_cast():
    wrapped = _load_wrapped_block()
    block = wrapped(_mini_config(), layer_idx=0).eval()
    original = block._rotary_emb.inv_freq.detach().clone()
    original_copy = block._rotary_emb.original_inv_freq.detach().clone()

    # The cast the server performs is lossy for these frequencies. If the
    # guard merely upcasts afterwards, this comparison fails.
    assert not torch.equal(original.to(torch.bfloat16).float(), original)

    block.to(torch.bfloat16)

    assert block.input_layernorm.weight.dtype == torch.bfloat16
    assert block._rotary_emb.inv_freq.dtype == torch.float32
    assert block._rotary_emb.original_inv_freq.dtype == torch.float32
    torch.testing.assert_close(block._rotary_emb.inv_freq, original, rtol=0, atol=0)
    torch.testing.assert_close(block._rotary_emb.original_inv_freq, original_copy, rtol=0, atol=0)


def test_glm_rotary_inv_freq_survives_fp16_cast():
    wrapped = _load_wrapped_block()
    block = wrapped(_mini_config(), layer_idx=0).eval()
    original = block._rotary_emb.inv_freq.detach().clone()

    assert not torch.equal(original.to(torch.float16).float(), original)

    block.half()

    assert block._rotary_emb.inv_freq.dtype == torch.float32
    assert block._rotary_emb.original_inv_freq.dtype == torch.float32
    torch.testing.assert_close(block._rotary_emb.inv_freq, original, rtol=0, atol=0)
