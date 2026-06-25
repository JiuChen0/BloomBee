import torch
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer

from bloombee.models.mixtral.block import WrappedMixtralBlock


def _mini_mixtral_config():
    cfg = MixtralConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=32,
    )
    cfg._attn_implementation = "eager"
    return cfg


def test_backend_attention_mask_is_forwarded_to_mixtral(monkeypatch):
    block = WrappedMixtralBlock(_mini_mixtral_config(), layer_idx=0).eval()
    captured = {}

    def fake_forward(self, hidden_states, *args, attention_mask=None, **kwargs):
        captured["attention_mask"] = attention_mask
        return hidden_states

    monkeypatch.setattr(MixtralDecoderLayer, "forward", fake_forward)

    hidden_states = torch.randn(2, 3, 16)
    backend_mask = torch.zeros(2, 3, 3)

    output_hidden, kv = block(hidden_states, attention_mask=backend_mask, use_cache=False)

    assert output_hidden is hidden_states
    assert kv is None
    assert captured["attention_mask"].shape == (2, 1, 3, 3)
    torch.testing.assert_close(captured["attention_mask"].squeeze(1), backend_mask)
