from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def test_microbatch_merge_layout_issues_are_fatal_and_clear_state():
    source = _source("src/bloombee/server/block_functions.py")
    layout_block = source[source.index("layout_issues = []") : source.index("merged_hidden_list = []")]

    assert "non-contiguous indices" in layout_block
    assert "logger.error(message)" in layout_block
    assert "_drop_mb_step_state(mb_accum_key, accum=True)" in layout_block
    assert "_drop_mb_step_state(overlap_tracking_key, overlap=True)" in layout_block
    assert "raise ValueError(message)" in layout_block
    assert "logger.warning(" not in layout_block


def test_backend_forward_none_fails_instead_of_identity_passthrough():
    source = _source("src/bloombee/server/backend.py")
    forward_block = source[source.index("def _run_block_forward") : source.index("def _finalize_cache_update")]

    assert "if forward_result is None:" in forward_block
    assert "raise RuntimeError" in forward_block
    assert "return None" not in forward_block
    assert "return (hidden_states, None)" not in source


def test_zipnn_transport_is_opt_in_before_send_or_receive():
    source = _source("src/bloombee/utils/lossless_transport.py")
    send_block = source[source.index("def _supports_zipnn_transport") : source.index("def _zipnn_skip_reason")]
    recv_block = source[source.index("elif algo_id == _ALGO_ZIPNN:") : source.index("decompressor = _get_zipnn_decompressor()")]

    assert "_LOSSLESS_ZIPNN_TRANSPORT_ENV" in source
    assert "default=False" in source[source.index("def _zipnn_transport_enabled") : source.index("@lru_cache(maxsize=1)")]
    assert "_zipnn_transport_enabled() and _supports_zipnn_compare" in send_block
    assert "if not _zipnn_transport_enabled():" in recv_block
    assert "raise ValueError" in recv_block
