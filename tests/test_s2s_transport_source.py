from pathlib import Path


def _method_source(source: str, start: str, end: str) -> str:
    start_idx = source.index(start)
    return source[start_idx: source.index(end, start_idx)]


def test_rpc_push_does_not_ack_unknown_sessions():
    source = Path("src/bloombee/server/handler.py").read_text()

    put_method = _method_source(
        source,
        "def _put_into_session_queue(",
        "    async def _get_from_session_queue",
    )
    assert "return False" in put_method
    assert "return True" in put_method

    rpc_push_method = _method_source(
        source,
        "async def rpc_push(",
        "    async def _handle_microbatch_push",
    )
    assert "if not self._put_into_session_queue(session_id, request):" in rpc_push_method
    assert "Cannot push to unknown or closed inference session" in rpc_push_method


def test_microbatch_push_marks_processed_only_after_queue_success():
    source = Path("src/bloombee/server/handler.py").read_text()
    method = _method_source(
        source,
        "async def _handle_microbatch_push(",
        "    async def _push_outputs",
    )

    queue_check = method.index("if not self._put_into_session_queue(session_id, mb_queue_item):")
    processed_add = method.index("self._mb_processed[mb_key].add(mb_idx)")
    received_increment = method.index("self._mb_received[mb_key] = self._mb_received.get(mb_key, 0) + 1")

    assert queue_check < processed_add < received_increment
    assert "Cannot push micro-batch to unknown or closed inference session" in method


def test_microbatch_merge_layout_failure_is_fatal():
    source = Path("src/bloombee/server/block_functions.py").read_text()
    block = source[
        source.index("if layout_issues:") :
        source.index("elif log_mb_detail:", source.index("if layout_issues:"))
    ]

    assert "raise ValueError" in block
    assert "_drop_mb_step_state(mb_accum_key, accum=True)" in block
    assert "Micro-batch merge layout check failed" in block
