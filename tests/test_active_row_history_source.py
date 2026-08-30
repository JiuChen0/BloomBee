from pathlib import Path


def test_active_row_compaction_gathers_session_history_by_hypo_ids():
    source = Path("src/bloombee/client/inference_session.py").read_text()
    block_start = source.index("            if self.history.shape[0] != inputs.shape[0]:")
    block_end = source.index(
        "            self.history = torch.cat(",
        block_start,
    )
    history_source = source[block_start:block_end]

    assert "hypo_ids" in history_source
    assert "row_selector = hypo_ids.to" in history_source
    assert "self.history = self.history.index_select(0, row_selector)" in history_source
    assert "raise RuntimeError(" in history_source
    assert "self.history = self.history[: inputs.shape[0]]" not in history_source

