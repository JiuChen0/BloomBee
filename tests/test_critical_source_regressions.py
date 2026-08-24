from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _function_body(source: str, start_marker: str, end_marker: str) -> str:
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def test_speculative_cache_update_runs_reorder_synchronously():
    source = (ROOT / "src/bloombee/server/memory_cache_manager.py").read_text()
    body = _function_body(
        source,
        "    def update_cache_and_async_reorder(",
        "    @staticmethod\n    def _plain_tensor_or_none",
    )

    assert "self.wait_for_pending_reorder()" in body
    assert "self._do_reorder_task(" in body
    assert "self._reorder_executor.submit" not in body
    assert "self._pending_reorder = future" not in body


def test_speculative_cache_mask_metadata_mismatch_fails_closed():
    source = (ROOT / "src/bloombee/server/backend.py").read_text()
    branch = _function_body(
        source,
        "        if ids.ndim < 2 or ids.shape[0] != batch_size:",
        "        valid_mask = ids >= 0",
    )

    assert "raise RuntimeError(" in branch
    assert "Cannot safely build cache mask" in branch
    assert "return torch.ones" not in branch
    assert "all-prefix-valid" not in branch


def test_mixtral_wrapper_preserves_backend_attention_masks():
    source = (ROOT / "src/bloombee/models/mixtral/block.py").read_text()
    body = _function_body(
        source,
        "    def forward(",
        "    def _reorder_cache_from_bloom(",
    )

    assert "attention_mask = None" not in body
    assert "attention_mask = attention_mask.unsqueeze(1)" in body
    assert "causal = torch.full(" in body
    assert "attention_mask=attention_mask" in body
