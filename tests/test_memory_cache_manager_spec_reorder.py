import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("hivemind")

from bloombee.server.memory_cache_manager import KVCacheManager


def test_spec_reorder_update_waits_for_reorder_completion(monkeypatch):
    manager = KVCacheManager.__new__(KVCacheManager)
    manager._reorder_executor = ThreadPoolExecutor(max_workers=1)

    started = threading.Event()
    finish_allowed = threading.Event()
    finished = threading.Event()

    def fake_reorder_task(*_args, **_kwargs):
        started.set()
        assert finish_allowed.wait(timeout=2.0)
        finished.set()

    monkeypatch.setattr(manager, "_do_reorder_task", fake_reorder_task)

    caller = threading.Thread(
        target=manager.update_cache_and_async_reorder,
        kwargs={
            "new_kvs": None,
            "kv_cache_position_ids": None,
            "cache_tensors": (),
        },
    )
    caller.start()

    try:
        assert started.wait(timeout=1.0)
        assert caller.is_alive(), "spec-dec cache reorder must complete before update returns"

        finish_allowed.set()
        caller.join(timeout=1.0)

        assert not caller.is_alive()
        assert finished.is_set()
    finally:
        finish_allowed.set()
        manager._reorder_executor.shutdown(wait=True)
        caller.join(timeout=1.0)
