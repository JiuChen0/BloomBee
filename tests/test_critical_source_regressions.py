from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def test_backend_forward_none_fails_instead_of_returning_identity_success():
    source = _read("src/bloombee/server/backend.py")

    assert "module.forward returned None" in source
    assert "raise RuntimeError(f\"{self.name} module.forward returned None\")" in source
    assert "return (hidden_states, None)" not in source


def test_microbatch_merge_layout_errors_are_fatal():
    source = _read("src/bloombee/server/block_functions.py")
    merge_start = source.index("sorted_indices = sorted(accum['results'].keys())")
    merge_end = source.index("merged_hidden_list = []", merge_start)
    merge_validation = source[merge_start:merge_end]

    assert "sorted_indices != expected_indices" in merge_validation
    assert "layout_issues" in merge_validation
    assert merge_validation.count("_drop_mb_step_state(mb_accum_key, accum=True)") >= 2
    assert merge_validation.count("raise ValueError(") >= 2
    assert "logger.warning(" not in merge_validation


def test_gqa_and_gemma4_kv_head_inference_use_runtime_head_dim():
    source = _read("src/bloombee/server/memory_cache_manager.py")

    assert "source_head_dim: Optional[int] = None" in source
    assert 'getattr(self.block_config, "num_global_key_value_heads", None)' in source
    assert 'getattr(self.block_config, "global_head_dim", None)' in source
    assert "source_head_dim == global_head_dim" in source
    assert "_source_heads_per_batch(H, BH_src, full_batch_size, micro_batch_size, source_head_dim=D_src)" in source


def test_speculative_reorder_errors_propagate():
    source = _read("src/bloombee/server/memory_cache_manager.py")
    except_start = source.index("except Exception as e:", source.index("def _do_reorder_task"))
    except_block = source[except_start : except_start + 180]

    assert "logging.exception" in except_block
    assert "raise" in except_block


def test_llama_4d_cache_read_slices_to_kv_heads():
    source = _read("src/bloombee/models/llama/block.py")
    branch_start = source.index("if key_states.dim() == 4 and value_states.dim() == 4:")
    branch = source[branch_start : source.index("# Otherwise", branch_start)]

    assert "nkv = self._num_key_value_heads()" in branch
    assert "key_states = key_states[:, :nkv, :, :]" in branch
    assert "value_states = value_states[:, :nkv, :, :]" in branch


def test_gemma4_external_masks_are_merged_with_layer_mask():
    source = _read("src/bloombee/models/gemma4/block.py")

    assert "def _merge_layer_type_mask(" in source
    assert "attention_mask.dtype == torch.bool" in source
    assert "torch.minimum(external_mask, layer_mask)" in source
    assert "attention_mask = _merge_layer_type_mask(" in source


def test_mixtral_external_masks_are_preserved():
    source = _read("src/bloombee/models/mixtral/block.py")
    forward_start = source.index("def forward(")
    forward_source = source[forward_start:]

    assert "attention_mask = None" not in forward_source
    assert "attention_mask = attention_mask.unsqueeze(1)" in forward_source
    assert "attention_mask.dtype == torch.bool" in forward_source
    assert "attention_mask = scores.masked_fill(attention_mask, 0.0)" in forward_source


def test_lossless_byte_split_decoders_reject_unsupported_elem_size():
    source = _read("src/bloombee/utils/lossless_transport.py")

    assert "def _validate_byte_split_elem_size(" in source
    assert "elem_size not in (2, 4)" in source
    assert source.count("elem_size = _validate_byte_split_elem_size(elem_size)") >= 3


def test_cross_stage_push_limiter_releases_setup_failure_slot():
    source = _read("src/bloombee/server/handler.py")
    push_start = source.index("async def _push_microbatch(")
    push_end = source.index("async def _do_rpc_push_async(", push_start)
    push_source = source[push_start:push_end]

    assert "release_slot_handed_off = False" in push_source
    assert "release_slot_handed_off = acquired_slot" in push_source
    assert "if acquired_slot and not release_slot_handed_off:" in push_source
    assert "await self._push_limiter.release(send_time_ms=0.0, success=False)" in push_source
