import threading
from concurrent.futures import ThreadPoolExecutor
from types import MethodType

import pytest
import torch

from bloombee.server.memory_cache_manager import KVCacheManager


def test_spec_cache_reorder_update_blocks_until_reorder_finishes():
    manager = KVCacheManager.__new__(KVCacheManager)

    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    def fake_reorder_task(self, *args):
        entered.set()
        assert release.wait(timeout=1.0)
        completed.set()

    manager._do_reorder_task = MethodType(fake_reorder_task, manager)

    # Present only to make this test fail against the old implementation,
    # which submitted the reorder work to a background executor and returned.
    manager._reorder_executor = ThreadPoolExecutor(max_workers=1)
    caller = threading.Thread(
        target=manager.update_cache_and_async_reorder,
        args=(None, None, ()),
    )

    try:
        caller.start()
        assert entered.wait(timeout=1.0)
        assert caller.is_alive()
        assert not completed.is_set()

        release.set()
        caller.join(timeout=1.0)

        assert not caller.is_alive()
        assert completed.is_set()
    finally:
        release.set()
        manager._reorder_executor.shutdown(wait=True)
        caller.join(timeout=1.0)


def test_spec_cache_reorder_failures_propagate_after_partial_write():
    manager = KVCacheManager.__new__(KVCacheManager)
    events = []

    def fake_rollback(self, *args, **kwargs):
        events.append("rollback")

    def fake_write(self, *args, **kwargs):
        events.append("write")

    def fake_track(self, *args, **kwargs):
        events.append("track")

    def fake_select(self, *args, **kwargs):
        events.append("select")
        return torch.empty((1, 1, 5, 1)), torch.empty((1, 1, 5, 1)), 5

    def fake_reorder(self, *args, **kwargs):
        events.append("reorder")
        raise RuntimeError("reorder boom")

    def fake_commit(self, *args, **kwargs):
        events.append("commit")

    manager._rollback_paged_to = MethodType(fake_rollback, manager)
    manager._write_kvs = MethodType(fake_write, manager)
    manager._track_paged_write = MethodType(fake_track, manager)
    manager.select_cache = MethodType(fake_select, manager)
    manager.reorder_and_write_cache = MethodType(fake_reorder, manager)
    manager._commit_paged_to = MethodType(fake_commit, manager)

    new_kvs = (torch.zeros((1, 1, 3)), torch.zeros((1, 3, 1)))
    kv_cache_position_ids = torch.tensor([[0, 1]])

    with pytest.raises(RuntimeError, match="reorder boom"):
        manager._do_reorder_task(
            new_kvs,
            kv_cache_position_ids,
            (),
            batch_offset=0,
            full_batch_size=0,
            micro_batch_size=0,
            cache_manager=manager,
        )

    assert events == ["rollback", "write", "track", "select", "reorder"]
