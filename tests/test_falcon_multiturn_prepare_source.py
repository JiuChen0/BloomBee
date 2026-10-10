"""Source-level lock for Falcon multi-turn generate input cropping.

RemoteGenerationMixin concatenates prior session.output_ids with the new
prompt, then _SkipTokensMixin keeps only the last cached token plus the new
prompt. Falcon used to apply upstream's full-sequence crop to that suffix,
so documented `.generate(new_prompts, session=...)` forwarded a single token.
"""

import ast
from pathlib import Path


_FALCON_SRC = Path("src/bloombee/models/falcon/model.py")
_REMOTE_GEN_SRC = Path("src/bloombee/client/remote_generation.py")


def _crop_after_skip(full_len, remaining_len, past_length, *, mask_aware):
    """Return how many suffix tokens survive skip + prepare_inputs crop."""
    if mask_aware and full_len > remaining_len:
        return full_len - past_length
    if past_length < remaining_len:
        return remaining_len - past_length
    return remaining_len


def _legacy_falcon_crop_after_skip(remaining_len, past_length):
    if remaining_len > past_length:
        return remaining_len - past_length
    return 1


def test_falcon_prepare_uses_mask_aware_crop():
    src = _FALCON_SRC.read_text()
    tree = ast.parse(src)
    prepare = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_inputs_for_generation":
            prepare = node
            break
    assert prepare is not None, "Falcon prepare_inputs_for_generation is missing"
    text = ast.get_source_segment(src, prepare)
    assert "attention_mask.shape[1] > input_ids.shape[1]" in text
    assert "remove_prefix_length = input_ids.shape[1] - 1" not in text
    assert "-(attention_mask.shape[1] - past_length)" in text


def test_skip_mixin_still_keeps_last_cached_token():
    src = _REMOTE_GEN_SRC.read_text()
    assert "_skipped_tokens.set(max(0, n_prev_tokens - 1))" in src
    assert "inputs = torch.cat([session.output_ids, inputs], dim=1)" in src


def test_multiturn_chat_keeps_last_token_and_new_prompt():
    # First turn: 10-token prompt + 8 new tokens. Last sampled token has not
    # been forwarded, so session.position is one behind output_ids.
    n_prev = 18
    past_length = 17
    new_prompt = 15
    full_len = n_prev + new_prompt
    remaining_len = full_len - (n_prev - 1)  # last cached token + new prompt

    assert remaining_len == 16
    assert _legacy_falcon_crop_after_skip(remaining_len, past_length) == 1
    assert _crop_after_skip(full_len, remaining_len, past_length, mask_aware=True) == 16


def test_resume_generate_none_still_refeeds_last_token():
    n_prev = 18
    past_length = 17
    full_len = n_prev
    remaining_len = full_len - (n_prev - 1)

    assert remaining_len == 1
    assert _legacy_falcon_crop_after_skip(remaining_len, past_length) == 1
    assert _crop_after_skip(full_len, remaining_len, past_length, mask_aware=True) == 1


def test_first_prefill_keeps_full_prompt():
    full_len = remaining_len = 10
    past_length = 0
    assert _crop_after_skip(full_len, remaining_len, past_length, mask_aware=True) == 10
    assert _legacy_falcon_crop_after_skip(remaining_len, past_length) == 10
