from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def test_session_history_uses_active_row_gather():
    source = _source("src/bloombee/client/inference_session.py")

    assert "self.history = self.history.index_select(0, gather)" in source
    assert "out-of-range hypo_ids" in source


def test_spec_cache_valid_mask_fails_closed_on_batch_mismatch():
    source = _source("src/bloombee/server/backend.py")
    fn = source[source.index("    def _spec_cache_valid_mask("): source.index("    def _expand_local_tree_attention_mask(")]
    mismatch_block = fn[fn.index("        if ids.ndim < 2 or ids.shape[0] != batch_size:"): fn.index("        valid_mask = ids >= 0")]

    assert "refusing to mark all prefix cache slots valid" in mismatch_block
    assert "raise RuntimeError" in mismatch_block
    assert "return torch.ones(batch_size, cache_len" not in mismatch_block


def test_microbatch_merge_layout_validation_is_fatal():
    source = _source("src/bloombee/server/block_functions.py")
    layout_check = source[source.index("                if layout_issues:"): source.index("                elif log_mb_detail:", source.index("                if layout_issues:"))]

    assert "_drop_mb_step_state(mb_accum_key, accum=True)" in layout_check
    assert "raise ValueError" in layout_check
    assert "logger.warning" not in layout_check


def test_push_limiter_releases_slot_on_setup_failure():
    source = _source("src/bloombee/server/handler.py")
    push_fn = source[source.index("    async def _push_microbatch("): source.index("    async def _do_rpc_push_async(")]

    assert "release_slot_handed_off = False" in push_fn
    assert "release_slot_handed_off = acquired_slot" in push_fn
    assert "await self._push_limiter.release(send_time_ms=0.0, success=False)" in push_fn


def test_model_wrappers_preserve_backend_attention_masks():
    gemma = _source("src/bloombee/models/gemma4/block.py")
    mixtral = _source("src/bloombee/models/mixtral/block.py")
    mixtral_forward = mixtral[mixtral.index("    def forward("): mixtral.index("    def _reorder_cache_from_bloom(")]

    assert "torch.minimum(attention_mask, layer_type_mask)" in gemma
    assert "attention_mask = attention_mask.unsqueeze(1)" in gemma
    assert "def _build_causal_mask(" in mixtral
    assert "attention_mask = attention_mask.unsqueeze(1)" in mixtral_forward
    assert "attention_mask = None" not in mixtral_forward


def test_lossless_byte_split_rejects_unsupported_elem_size():
    source = _source("src/bloombee/utils/lossless_transport.py")

    assert "def _validate_byte_split_elem_size(" in source
    assert "if elem_size not in (2, 4):" in source
    assert source.count("_validate_byte_split_elem_size(elem_size, original_size)") >= 3


def test_s2s_int8_hidden_states_require_quant_metadata():
    source = _source("src/bloombee/utils/s2s_activation_quant.py")
    fn = source[source.index("def dequantize_s2s_hidden_from_transport("):]

    assert "Received int8 S2S hidden states without quantization metadata" in fn
    assert "Unsupported S2S activation quantization scheme" in fn


def test_active_row_cache_compaction_validates_permutation():
    source = _source("src/bloombee/server/memory_cache_manager.py")
    fn = source[source.index("    def permute_batch_rows("): source.index("    async def use_cache(")]

    assert "perm contains out-of-range rows" in fn
    assert "perm contains duplicate rows" in fn
    assert "torch.unique(perm).numel() != n" in fn
