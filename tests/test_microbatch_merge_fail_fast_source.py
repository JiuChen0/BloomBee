from pathlib import Path


BLOCK_FUNCTIONS_SOURCE = Path(__file__).resolve().parents[1] / "src" / "bloombee" / "server" / "block_functions.py"


def test_microbatch_merge_layout_errors_are_fatal():
    source = BLOCK_FUNCTIONS_SOURCE.read_text()
    check_start = source.index("sorted_indices = sorted(accum['results'].keys())")
    check_end = source.index("merged_hidden_list = []", check_start)
    merge_check = source[check_start:check_end]

    assert "layout_issues.append" in merge_check
    assert "non_contiguous_indices" in merge_check
    assert "raise ValueError(message)" in merge_check
    assert "Micro-batch merge layout check failed" in merge_check
