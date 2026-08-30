from pathlib import Path


def test_microbatch_merge_layout_validation_is_fatal():
    source = Path("src/bloombee/server/block_functions.py").read_text()
    merge_start = source.index("                sorted_indices = sorted(accum['results'].keys())")
    merge_end = source.index("                merged_hidden_list = []", merge_start)
    merge_source = source[merge_start:merge_end]

    assert "layout_issues = []" in merge_source
    assert "sorted_indices != expected_indices" in merge_source
    assert "_drop_mb_step_state(mb_accum_key, overlap=True, accum=True)" in merge_source
    assert "raise ValueError(" in merge_source
    assert "logger.warning(" not in merge_source

