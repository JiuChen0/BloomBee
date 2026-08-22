from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def test_speculative_cache_reorder_runs_synchronously():
    source = _source("src/bloombee/server/memory_cache_manager.py")
    method = source.split("def update_cache_and_async_reorder(", 1)[1].split("\n    @staticmethod", 1)[0]

    assert "self.wait_for_pending_reorder()" in method
    assert "self._do_reorder_task(" in method
    assert "_reorder_executor.submit" not in method
    assert "_pending_reorder = future" not in method


def test_speculative_mask_metadata_mismatch_fails_closed():
    source = _source("src/bloombee/server/backend.py")
    method = source.split("def _spec_cache_valid_mask(", 1)[1].split("\n    def _expand_local_tree_attention_mask", 1)[0]
    mismatch_branch = method.split("if ids.ndim < 2 or ids.shape[0] != batch_size:", 1)[1].split(
        "\n\n        valid_mask = ids >= 0",
        1,
    )[0]

    assert "raise RuntimeError" in mismatch_branch
    assert "torch.ones(batch_size, cache_len" not in mismatch_branch
    assert "falling back to all-prefix-valid" not in mismatch_branch


def test_block_forward_none_is_fatal_at_helper_boundary():
    source = _source("src/bloombee/server/backend.py")
    method = source.split("def _run_block_forward(", 1)[1].split("\n    @staticmethod", 1)[0]

    assert "if forward_result is None:" in method
    assert 'raise RuntimeError("module.forward returned None")' in method
    assert "return None" not in method


def test_mixtral_wrapper_preserves_backend_attention_masks():
    source = _source("src/bloombee/models/mixtral/block.py")
    forward = source.split("def forward(", 1)[1].split("\n    def _reorder_cache_from_bloom", 1)[0]

    assert "attention_mask = None" not in forward
    assert "if attention_mask is None:" in forward
    assert "torch.triu(causal, diagonal=past_key_values_length + 1)" in forward
    assert "elif attention_mask.dim() == 3:" in forward
    assert "attention_mask = attention_mask.unsqueeze(1)" in forward
