from types import SimpleNamespace

import pytest
import torch

from bloombee.server import block_functions
from bloombee.server.backend import TransformerBackend
from bloombee.utils.convert_block import QuantType


class _UnusedPrioritizer:
    def prioritize(self, *args, **kwargs):
        raise AssertionError("max-length guard should run before prioritization")


async def _single_microbatch_request(metadata):
    yield SimpleNamespace(tensors=[object(), object()]), metadata


def test_cross_stage_microbatch_respects_max_length(event_loop, monkeypatch):
    hidden_states = torch.zeros((1, 1, 4), dtype=torch.float32)
    keep_indices = torch.arange(1, dtype=torch.int64)
    tensors = iter((hidden_states, keep_indices))
    monkeypatch.setattr(block_functions, "deserialize_torch_tensor", lambda _: next(tensors))

    metadata = {
        "type": "micro_batch",
        "mb_idx": 0,
        "expected_num_mb": 1,
        "offset": 0,
        "size": 1,
        "full_batch_size": 1,
        "step_id": "step-overflow",
        "session_id": "session-overflow",
        "start_from_position": 5,
    }
    backend = SimpleNamespace(dtype=torch.float32)
    iterator = block_functions.iterate_rpc_inference(
        requested_uids=("block.0",),
        requested_backends=(backend,),
        active_adapter=None,
        input_iterator=_single_microbatch_request(metadata),
        cache_handles=((),),
        pruner_manager=None,
        max_length=5,
        prioritizer=_UnusedPrioritizer(),
        points=0,
        quant_type=QuantType.NONE,
    )

    with pytest.raises(ValueError, match=r"Maximum length exceeded"):
        event_loop.run_until_complete(iterator.__anext__())


def test_block_forward_none_is_failure():
    backend = TransformerBackend.__new__(TransformerBackend)
    backend.module = SimpleNamespace(forward=lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="module.forward returned None"):
        backend._run_block_forward(
            torch.zeros((1, 1, 4), dtype=torch.float32),
            layer_past=None,
            attention_mask=None,
            position_ids=None,
            rotary_position_ids=None,
        )
