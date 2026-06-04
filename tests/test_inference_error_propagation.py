from types import MethodType

import pytest
import torch

from bloombee.server.backend import TransformerBackend
from bloombee.server.memory_cache_manager import KVCacheManager


class _RaisesForward:
    def forward(self, *args, **kwargs):
        raise RuntimeError("block failed")


class _NoneForward:
    def forward(self, *args, **kwargs):
        return None


def test_block_forward_exceptions_are_not_silent_passthroughs():
    backend = TransformerBackend.__new__(TransformerBackend)
    backend.module = _RaisesForward()

    with pytest.raises(RuntimeError, match="block failed"):
        backend._run_block_forward(
            torch.zeros(1, 1, 2),
            layer_past=None,
            attention_mask=None,
            position_ids=None,
            rotary_position_ids=None,
        )


def test_block_forward_none_result_is_fatal():
    backend = TransformerBackend.__new__(TransformerBackend)
    backend.module = _NoneForward()

    with pytest.raises(RuntimeError, match="returned None"):
        backend._run_block_forward(
            torch.zeros(1, 1, 2),
            layer_past=None,
            attention_mask=None,
            position_ids=None,
            rotary_position_ids=None,
        )


def test_speculative_cache_reorder_failures_propagate():
    manager = KVCacheManager.__new__(KVCacheManager)

    def fail_write(self, *args, **kwargs):
        raise RuntimeError("cache write failed")

    manager._write_kvs = MethodType(fail_write, manager)

    with pytest.raises(RuntimeError, match="cache write failed"):
        manager._do_reorder_task(
            new_kvs=None,
            kv_cache_position_ids=None,
            cache_tensors=(),
            batch_offset=0,
            full_batch_size=0,
            micro_batch_size=0,
            cache_manager=manager,
        )
