import threading
from concurrent.futures import ThreadPoolExecutor
from types import MethodType

import pytest

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


def test_spec_cache_reorder_failures_propagate():
    manager = KVCacheManager.__new__(KVCacheManager)

    def fail_write(self, *args, **kwargs):
        raise RuntimeError("write failed")

    manager._write_kvs = MethodType(fail_write, manager)

    with pytest.raises(RuntimeError, match="write failed"):
        manager._do_reorder_task(
            new_kvs=(None, None),
            kv_cache_position_ids=None,
            cache_tensors=(),
            batch_offset=0,
            full_batch_size=0,
            micro_batch_size=0,
            cache_manager=manager,
        )
