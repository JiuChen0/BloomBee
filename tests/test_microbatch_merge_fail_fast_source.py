from pathlib import Path


def test_microbatch_merge_layout_mismatch_is_fatal():
    source = Path("src/bloombee/server/block_functions.py").read_text()
    marker = "                # Sort by mb_idx and merge"
    start = source.index(marker)
    end = source.index("                merged_hidden_list = []", start)
    body = source[start:end]

    assert "layout_issues = []" in body
    assert "non-contiguous micro-batch indices" in body
    assert "logger.error(message)" in body
    assert "_drop_mb_step_state(mb_accum_key, accum=True)" in body
    assert "_drop_mb_step_state(overlap_tracking_key, overlap=True)" in body
    assert "raise ValueError(message)" in body
    assert "logger.warning(" not in body
