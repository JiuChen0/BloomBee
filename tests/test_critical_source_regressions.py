from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def test_cross_stage_microbatch_enforces_max_length_before_cache_write():
    src = _source("src/bloombee/server/block_functions.py")
    micro_branch = src[src.index('if step_metadata.get("type") == "micro_batch":') :]
    guard = "if prefix_length + token_increment > max_length:"
    assert guard in micro_branch
    assert micro_branch.index(guard) < micro_branch.index("hidden_states = hidden_states.to")
    assert micro_branch.index(guard) < micro_branch.index("build_inference_metadata_batch")


def test_microbatch_merge_layout_validation_is_fatal():
    src = _source("src/bloombee/server/block_functions.py")
    merge_block = src[src.index("# Validate merged layout coverage") : src.index("merged_hidden_list = []")]
    assert "layout_issues.append" in merge_block
    assert "raise ValueError" in merge_block
    assert "_drop_mb_step_state(mb_accum_key, overlap=True, accum=True)" in merge_block
    assert "logger.warning" not in merge_block


def test_mixtral_preserves_backend_attention_mask():
    src = _source("src/bloombee/models/mixtral/block.py")
    mask_block = src[src.index("if attention_mask is None:") : src.index("position_ids = kwargs.pop")]
    assert "_build_causal_mask" in mask_block
    assert "attention_mask = None" not in mask_block
    assert "attention_mask = attention_mask.unsqueeze(1)" in mask_block


def test_gemma4_backend_masks_are_merged_with_layer_mask():
    src = _source("src/bloombee/models/gemma4/block.py")
    assert "def _compose_layer_attention_mask" in src
    helper = src[src.index("def _compose_layer_attention_mask") : src.index("class WrappedGemma4Block")]
    assert "torch.minimum(attention_mask, layer_mask)" in helper
    forward_branch = src[src.index("elif attention_mask.dim() == 3:") : src.index("position_ids = kwargs.pop")]
    assert "_compose_layer_attention_mask" in forward_branch


def test_push_limiter_slot_released_if_setup_fails_before_send_task():
    src = _source("src/bloombee/server/handler.py")
    push_method = src[src.index("async def _push_microbatch") : src.index("async def _do_rpc_push_async")]
    assert "release_slot_handed_off = False" in push_method
    assert "release_slot_handed_off = True" in push_method
    except_block = push_method[push_method.index("except Exception as e:") :]
    assert "if acquired_slot and not release_slot_handed_off:" in except_block
    assert "await self._push_limiter.release" in except_block
