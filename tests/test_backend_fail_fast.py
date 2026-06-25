import inspect

import pytest
import torch

from bloombee.server.backend import TransformerBackend


class _NullForwardModule:
    def forward(self, *args, **kwargs):
        return None


def test_run_block_forward_raises_when_module_returns_none():
    backend = TransformerBackend.__new__(TransformerBackend)
    backend.module = _NullForwardModule()
    backend.name = "test-block"

    with pytest.raises(RuntimeError, match="module.forward returned None"):
        backend._run_block_forward(
            torch.zeros(1, 1, 4),
            layer_past=None,
            attention_mask=None,
            position_ids=None,
            rotary_position_ids=None,
        )


def test_inference_step_has_no_passthrough_failure_fallback():
    source = inspect.getsource(TransformerBackend.inference_step)

    assert "return (hidden_states, None)" not in source
