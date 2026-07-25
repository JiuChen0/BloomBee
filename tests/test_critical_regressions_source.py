from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def test_s2s_link_telemetry_is_dataclass_initializable():
    source = _read("src/bloombee/server/s2s_flow.py")

    assert "@dataclass\nclass S2SLinkTelemetry" in source
    assert "def __post_init__(self) -> None:" in source


def test_microbatch_merge_layout_validation_is_fatal():
    source = _read("src/bloombee/server/block_functions.py")

    assert "if sorted_indices != expected_indices:" in source
    non_contiguous_block = source.split("if sorted_indices != expected_indices:", 1)[1].split(
        "# Validate merged layout coverage", 1
    )[0]
    assert "raise ValueError(" in non_contiguous_block
    assert "_drop_mb_step_state(mb_accum_key, accum=True)" in non_contiguous_block

    layout_block = source.split("if layout_issues:", 1)[1].split("elif log_mb_detail:", 1)[0]
    assert "raise ValueError(" in layout_block
    assert "_drop_mb_step_state(mb_accum_key, accum=True)" in layout_block


def test_push_limiter_slot_released_before_async_handoff_on_failure():
    source = _read("src/bloombee/server/handler.py")

    acquire_idx = source.index("sem_wait_time = await self._push_limiter.acquire()")
    create_task_idx = source.index("send_task = asyncio.create_task(", acquire_idx)
    handoff_idx = source.index("acquired_slot = False", create_task_idx)
    except_idx = source.index("except Exception as e:", handoff_idx)
    release_idx = source.index("await self._push_limiter.release(send_time_ms=0.0, success=False)", except_idx)

    assert acquire_idx < create_task_idx < handoff_idx < except_idx < release_idx


def test_zipnn_receive_path_rejects_unbounded_decompression():
    source = _read("src/bloombee/utils/lossless_transport.py")

    zipnn_branch = source.split("elif algo_id == _ALGO_ZIPNN:", 1)[1].split("else:", 1)[0]
    assert "raise ValueError(" in zipnn_branch
    assert "bounded-output decompression limit" in zipnn_branch
    assert "decompressor.decompress(payload)" not in zipnn_branch


def test_gqa_full_batch_cache_infers_attention_head_layout_first():
    source = _read("src/bloombee/server/memory_cache_manager.py")

    runtime_batch_idx = source.index("if full_batch_size > 0 and micro_batch_size > 0:")
    attention_layout_idx = source.index("if source_bh % attention_heads == 0:", runtime_batch_idx)
    kv_heads_idx = source.index('kv_heads = getattr(self.block_config, "num_key_value_heads", None)', runtime_batch_idx)

    assert runtime_batch_idx < attention_layout_idx < kv_heads_idx
    attention_layout_block = source[attention_layout_idx:kv_heads_idx]
    assert "return attention_heads" in attention_layout_block
