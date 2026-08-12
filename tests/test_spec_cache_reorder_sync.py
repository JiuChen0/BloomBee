import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MethodType

try:
    from bloombee.server.memory_cache_manager import KVCacheManager
except ModuleNotFoundError:
    KVCacheManager = None


def _method_source(path: str, name: str, next_marker: str) -> str:
    source = Path(path).read_text()
    start = source.index(f"    def {name}(")
    end = source.index(next_marker, start)
    return source[start:end]


def test_spec_cache_reorder_update_blocks_until_reorder_finishes():
    if KVCacheManager is None:
        method_source = _method_source(
            "src/bloombee/server/memory_cache_manager.py",
            "update_cache_and_async_reorder",
            "    @staticmethod",
        )

        assert "self._reorder_executor.submit" not in method_source
        assert "self._do_reorder_task(" in method_source
        return

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


def test_spec_cache_valid_mask_batch_mismatch_fails_closed():
    method_source = _method_source(
        "src/bloombee/server/backend.py",
        "_spec_cache_valid_mask",
        "    def _expand_local_tree_attention_mask",
    )

    mismatch_block = method_source[
        method_source.index("if ids.ndim < 2 or ids.shape[0] != batch_size:"):
        method_source.index("valid_mask = ids >= 0"),
    ]

    assert "raise RuntimeError" in mismatch_block
    assert "torch.ones" not in mismatch_block
