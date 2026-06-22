import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer

from bloombee.models.mixtral.block import WrappedMixtralBlock


def _make_config():
    cfg = MixtralConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    cfg._attn_implementation = "eager"
    return cfg


def test_mixtral_builds_causal_mask_when_backend_passes_none(monkeypatch):
    captured = {}

    def fake_forward(self, hidden_states, *args, **kwargs):
        captured["attention_mask"] = kwargs["attention_mask"]
        return hidden_states

    monkeypatch.setattr(MixtralDecoderLayer, "forward", fake_forward)

    cfg = _make_config()
    block = WrappedMixtralBlock(cfg, layer_idx=0).eval()
    hidden = torch.randn(1, 4, cfg.hidden_size)

    out, kv = block(hidden, attention_mask=None, use_cache=False)

    assert out.shape == hidden.shape
    assert kv is None
    mask = captured["attention_mask"]
    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0, 0, 1] < 0
    assert mask[0, 0, 3, 0] == 0


def test_mixtral_lifts_backend_3d_mask(monkeypatch):
    captured = {}

    def fake_forward(self, hidden_states, *args, **kwargs):
        captured["attention_mask"] = kwargs["attention_mask"]
        return hidden_states

    monkeypatch.setattr(MixtralDecoderLayer, "forward", fake_forward)

    cfg = _make_config()
    block = WrappedMixtralBlock(cfg, layer_idx=0).eval()
    hidden = torch.randn(2, 3, cfg.hidden_size)
    backend_mask = torch.zeros(2, 3, 3)

    block(hidden, attention_mask=backend_mask, use_cache=False)

    assert captured["attention_mask"].shape == (2, 1, 3, 3)
