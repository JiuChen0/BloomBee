from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text()


def test_speculative_cache_update_runs_reorder_synchronously():
    source = _source("src/bloombee/server/memory_cache_manager.py")
    method = source.split("    def update_cache_and_async_reorder(", 1)[1].split(
        "\n    @staticmethod\n    def _plain_tensor_or_none", 1
    )[0]

    assert "self.wait_for_pending_reorder()" in method
    assert "self._do_reorder_task(" in method
    assert "_reorder_executor.submit" not in method
    assert "_pending_reorder = future" not in method


def test_speculative_cache_mask_mismatch_fails_closed():
    source = _source("src/bloombee/server/backend.py")
    method = source.split("    def _spec_cache_valid_mask(", 1)[1].split(
        "\n    def _expand_local_tree_attention_mask", 1
    )[0]

    assert "raise RuntimeError" in method
    assert "refusing to mark all cache columns valid" in method
    assert "falling back to all-prefix-valid cache mask" not in method


def test_mixtral_preserves_backend_attention_mask():
    source = _source("src/bloombee/models/mixtral/block.py")
    forward = source.split("    def forward(", 1)[1].split(
        "\n    @staticmethod\n    def _prepare_decoder_attention_mask", 1
    )[0]

    assert "self._prepare_decoder_attention_mask(" in forward
    assert "attention_mask = None" not in forward
    assert "attention_mask=attention_mask" in forward
