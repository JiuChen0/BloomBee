from pathlib import Path


def test_spec_cache_position_mismatch_does_not_fail_open():
    source = Path("src/bloombee/server/backend.py").read_text()
    mismatch_block = source[
        source.index("if ids.ndim < 2 or ids.shape[0] != batch_size:") :
        source.index("valid_mask = ids >= 0")
    ]

    assert "cannot build a safe speculative cache mask" in mismatch_block
    assert "raise RuntimeError" in mismatch_block
    assert "return torch.ones" not in mismatch_block
    assert "all-prefix-valid" not in mismatch_block


def test_spec_cache_reorder_update_stays_synchronous():
    source = Path("src/bloombee/server/memory_cache_manager.py").read_text()
    method = source[
        source.index("def update_cache_and_async_reorder(") :
        source.index("    @staticmethod", source.index("def update_cache_and_async_reorder("))
    ]

    assert "self.wait_for_pending_reorder()" in method
    assert "self._do_reorder_task(" in method
    assert "_reorder_executor.submit" not in method
