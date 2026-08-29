import ast
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_CACHE_MANAGER = REPO_ROOT / "src" / "bloombee" / "server" / "memory_cache_manager.py"
BACKEND = REPO_ROOT / "src" / "bloombee" / "server" / "backend.py"
BLOCK_FUNCTIONS = REPO_ROOT / "src" / "bloombee" / "server" / "block_functions.py"
MIXTRAL_BLOCK = REPO_ROOT / "src" / "bloombee" / "models" / "mixtral" / "block.py"


def _get_function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def test_spec_cache_update_runs_current_reorder_synchronously():
    fn = _get_function(MEMORY_CACHE_MANAGER, "update_cache_and_async_reorder")

    calls = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"submit", "_do_reorder_task", "wait_for_pending_reorder"}
    ]

    assert any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "wait_for_pending_reorder"
        for call in calls
    )
    assert any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "_do_reorder_task"
        for call in calls
    )
    assert not any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "submit"
        for call in calls
    ), "current speculative KV reorder must complete before update returns"


def test_block_forward_none_is_fatal_at_helper_boundary():
    fn = _get_function(BACKEND, "_run_block_forward")
    source = textwrap.dedent(ast.get_source_segment(BACKEND.read_text(), fn))

    none_branch = source.split("if forward_result is None:", 1)[1].split(
        "output_hidden_states_chunk, new_kvs = forward_result", 1
    )[0]
    assert "raise RuntimeError" in none_branch
    assert "return None" not in none_branch


def test_spec_cache_mask_metadata_mismatch_fails_closed():
    fn = _get_function(BACKEND, "_spec_cache_valid_mask")
    source = textwrap.dedent(ast.get_source_segment(BACKEND.read_text(), fn))

    mismatch_branch = source.split("if ids.ndim < 2 or ids.shape[0] != batch_size:", 1)[1].split(
        "valid_mask = ids >= 0", 1
    )[0]
    assert "torch.zeros(batch_size, cache_len" in mismatch_branch
    assert "torch.ones(batch_size, cache_len" not in mismatch_branch


def test_mixtral_preserves_backend_attention_masks():
    fn = _get_function(MIXTRAL_BLOCK, "forward")
    source = textwrap.dedent(ast.get_source_segment(MIXTRAL_BLOCK.read_text(), fn))

    assert "attention_mask = None" not in source
    assert "attention_mask = causal.unsqueeze(0).unsqueeze(0)" in source
    assert "attention_mask = attention_mask.unsqueeze(1)" in source


def test_microbatch_merge_layout_violations_are_fatal():
    source = BLOCK_FUNCTIONS.read_text()
    merge_start = source.index("# Sort by mb_idx and merge")
    merge_end = source.index("merged_hidden_states = _merge_inference_microbatch_hidden_states", merge_start)
    merge_source = source[merge_start:merge_end]

    assert "raise ValueError" in merge_source
    assert "_drop_mb_step_state(mb_accum_key, overlap=True, accum=True)" in merge_source
    assert "logger.warning(\n" not in merge_source
