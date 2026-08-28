import torch

from bloombee.models.mixtral.block import WrappedMixtralBlock


def test_mixtral_prepare_decoder_attention_mask_preserves_backend_scores():
    backend_mask = torch.tensor(
        [[[0.0, torch.finfo(torch.float32).min], [0.0, 0.0]]],
        dtype=torch.float32,
    )

    prepared = WrappedMixtralBlock._prepare_decoder_attention_mask(
        backend_mask,
        batch_size=1,
        seq_length=2,
        past_key_values_length=0,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert prepared.shape == (1, 1, 2, 2)
    torch.testing.assert_close(prepared[:, 0], backend_mask)


def test_mixtral_prepare_decoder_attention_mask_builds_causal_fallback():
    prepared = WrappedMixtralBlock._prepare_decoder_attention_mask(
        None,
        batch_size=2,
        seq_length=2,
        past_key_values_length=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert prepared.shape == (2, 1, 2, 5)
    assert prepared[0, 0, 0, :4].eq(0).all()
    assert prepared[0, 0, 0, 4].item() < -1e30
    assert prepared[0, 0, 1].eq(0).all()
