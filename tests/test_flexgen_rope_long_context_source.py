"""Source tests for FlexGen LLaMA RoPE past the hardcoded 4k table.

Default serving used to set rotary_position_ids=None. FlexGen then built a
2048*2 frequency table and indexed ``freq_cis[src_s - tgt_s:src_s]``. At
cache_len=4096 that slice is empty and apply_rotary_emb crashes; Llama-3's
8k window hits this on the default AR path.
"""
import ast
from pathlib import Path


_BACKEND_SRC = Path("src/bloombee/server/backend.py")
_FLEXGEN_SRC = Path("src/bloombee/flexgen_utils/pytorch_backend.py")
_DEFAULT_TABLE_LEN = 2048 * 2


def _function_source(path: Path, class_name: str, func_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == func_name:
                    return item
    raise AssertionError(f"{class_name}.{func_name} not found in {path}")


def test_legacy_sequential_table_cannot_index_position_4096():
    table_len = _DEFAULT_TABLE_LEN
    src_s = table_len + 1
    tgt_s = 1
    freqs_slice = list(range(table_len))[src_s - tgt_s : src_s]
    assert freqs_slice == []
    # apply_rotary_emb views freqs as (1, tgt_s, 1, half); an empty table
    # has the wrong number of elements and raises.
    half = 4
    try:
        memoryview(bytes(len(freqs_slice) * half)).cast("B")
        reshape_ok = len(freqs_slice) * half == 1 * tgt_s * 1 * half
    except Exception:
        reshape_ok = False
    assert not reshape_ok


def test_non_spec_inference_forwards_position_ids_to_rotary():
    src = _BACKEND_SRC.read_text()
    assert "rotary_position_ids = position_ids" in src
    fn = _function_source(_BACKEND_SRC, "TransformerBackend", "inference_step")
    assigned_none = False
    assigned_position_ids = False
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "rotary_position_ids" for t in node.targets):
            continue
        if isinstance(node.value, ast.Constant) and node.value.value is None:
            assigned_none = True
        if isinstance(node.value, ast.Name) and node.value.id == "position_ids":
            assigned_position_ids = True
    assert assigned_position_ids, "non-spec path must pass position_ids into FlexGen RoPE"
    assert not assigned_none, "non-spec path must not drop rotary_position_ids"


def test_flexgen_rope_table_grows_past_default_cap():
    tree = ast.parse(_FLEXGEN_SRC.read_text())
    helper = None
    default_assign = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_rope_table_len":
            helper = node
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_DEFAULT_ROPE_TABLE_LEN"
        ):
            default_assign = node
    assert helper is not None, "_rope_table_len helper is missing"
    assert default_assign is not None, "_DEFAULT_ROPE_TABLE_LEN is missing"
    namespace = {"max": max, "int": int}
    exec(
        compile(ast.Module(body=[default_assign, helper], type_ignores=[]), "<helper>", "exec"),
        namespace,
    )
    assert namespace["_DEFAULT_ROPE_TABLE_LEN"] == _DEFAULT_TABLE_LEN
    rope_table_len = namespace["_rope_table_len"]
    assert rope_table_len(1) == _DEFAULT_TABLE_LEN
    assert rope_table_len(_DEFAULT_TABLE_LEN) == _DEFAULT_TABLE_LEN
    assert rope_table_len(_DEFAULT_TABLE_LEN + 1) == _DEFAULT_TABLE_LEN + 1
    assert rope_table_len(8192) == 8192

    llama_fns = (
        _function_source(_FLEXGEN_SRC, "TorchDevice", "mha_llama"),
        _function_source(_FLEXGEN_SRC, "TorchDevice", "mha_gen_llama"),
    )
    for fn in llama_fns:
        called = [
            n.func.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        ]
        assert "_rope_table_len" in called, f"{fn.name} must size the RoPE table to the sequence"
        assert "precompute_freqs_cis" in called
