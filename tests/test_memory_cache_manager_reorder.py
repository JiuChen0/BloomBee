from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("hivemind")  # KVCacheManager imports hivemind during module import

from bloombee.server.memory_cache_manager import KVCacheManager


def test_spec_dec_reorder_update_does_not_return_before_reorder_finishes():
    manager = object.__new__(KVCacheManager)
    # Keep the old async implementation's executor attribute present so this
    # test fails against the race it is guarding.
    manager._reorder_executor = ThreadPoolExecutor(max_workers=1)

    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    def fake_reorder(*_args, **_kwargs):
        started.set()
        assert release.wait(timeout=1)
        completed.set()

    manager._do_reorder_task = fake_reorder

    update_thread = threading.Thread(
        target=manager.update_cache_and_async_reorder,
        args=(None, None, []),
        kwargs={"batch_offset": 0, "full_batch_size": 0, "micro_batch_size": 0},
    )

    try:
        update_thread.start()
        assert started.wait(timeout=1)
        assert update_thread.is_alive()
        assert not completed.is_set()

        release.set()
        update_thread.join(timeout=1)
        assert completed.is_set()
        assert not update_thread.is_alive()
    finally:
        release.set()
        update_thread.join(timeout=1)
        manager._reorder_executor.shutdown(wait=True)
