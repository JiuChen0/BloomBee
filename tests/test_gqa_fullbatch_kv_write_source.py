"""Default full-batch GQA KV writes must see the runtime batch size.

FlexGen LLaMA broadcasts GQA KV onto the full attention-head stride, so new_kvs
has BH = B * num_attention_heads. The cache slab is also B * H. Packed GQA
models (Gemma-4) instead emit B * num_key_value_heads.

_source_heads_per_batch uses full_batch_size to distinguish those layouts.
The default merged_fullbatch / separate_fullbatch paths used to omit it, so
Llama-3 8B (H=32, n_kv=8, B=1, BH=32) inferred 8 packed heads and tried to
write 128 BH rows into a 32-row slab.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace


_CACHE_SRC = Path("src/bloombee/server/memory_cache_manager.py")
_BLOCK_FN_SRC = Path("src/bloombee/server/block_functions.py")


def _load_source_heads_fn():
    tree = ast.parse(_CACHE_SRC.read_text())
    method = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "KVCacheManager":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_source_heads_per_batch":
                    method = item
                    break
    assert method is not None, "_source_heads_per_batch is missing"
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<source_heads>", "exec"), namespace)
    return namespace["_source_heads_per_batch"]


def _write_bh_span(source_heads: int, bh_src: int, attention_heads: int) -> int:
    source_batch_size = bh_src // source_heads
    return source_batch_size * attention_heads if source_heads < attention_heads else bh_src


def _apply_fullhead_guard(
    source_heads: int,
    bh_src: int,
    bh_dst: int,
    attention_heads: int,
) -> tuple[int, int]:
    """Mirror the _write_kvs reinterpret of FlexGen's broadcast GQA layout."""
    write_bh_span = _write_bh_span(source_heads, bh_src, attention_heads)
    if (
        source_heads < attention_heads
        and bh_src <= bh_dst
        and attention_heads > 0
        and bh_src % attention_heads == 0
        and write_bh_span > bh_dst
    ):
        source_heads = attention_heads
        write_bh_span = bh_src
    return source_heads, write_bh_span


class _Llama3GqaConfig:
    num_attention_heads = 32
    num_key_value_heads = 8
    num_key_value_groups = 4
    head_dim = 128


def test_llama3_gqa_without_batch_hint_inflates_write_span():
    infer = _load_source_heads_fn()
    manager = SimpleNamespace(block_config=_Llama3GqaConfig())
    attention_heads = 32
    bh_src = 32  # FlexGen broadcast: B=1 * H
    source_heads = infer(manager, attention_heads, bh_src, 0, 0, source_head_dim=128)
    assert source_heads == 8
    assert _write_bh_span(source_heads, bh_src, attention_heads) == 128


def test_llama3_gqa_full_batch_hint_keeps_broadcast_heads():
    infer = _load_source_heads_fn()
    manager = SimpleNamespace(block_config=_Llama3GqaConfig())
    attention_heads = 32
    bh_src = 32
    source_heads = infer(manager, attention_heads, bh_src, 1, 1, source_head_dim=128)
    assert source_heads == 32
    assert _write_bh_span(source_heads, bh_src, attention_heads) == 32


def test_llama3_gqa_write_guard_recovers_broadcast_layout():
    infer = _load_source_heads_fn()
    manager = SimpleNamespace(block_config=_Llama3GqaConfig())
    attention_heads = 32
    bh_src = bh_dst = 32
    source_heads = infer(manager, attention_heads, bh_src, 0, 0, source_head_dim=128)
    fixed_heads, write_span = _apply_fullhead_guard(source_heads, bh_src, bh_dst, attention_heads)
    assert fixed_heads == 32
    assert write_span == 32


def test_packed_gqa_batch4_is_unchanged_by_write_guard():
    """Gemma-style packed rows: BH = B * n_kv. Guard must not fire."""
    infer = _load_source_heads_fn()

    class Packed:
        num_attention_heads = 16
        num_key_value_heads = 4
        num_key_value_groups = 4
        head_dim = 256

    manager = SimpleNamespace(block_config=Packed())
    attention_heads = 16
    batch = 4
    bh_src = batch * 4
    bh_dst = batch * attention_heads
    source_heads = infer(manager, attention_heads, bh_src, batch, batch, source_head_dim=256)
    assert source_heads == 4
    write_span = _write_bh_span(source_heads, bh_src, attention_heads)
    assert write_span == bh_dst
    fixed_heads, fixed_span = _apply_fullhead_guard(source_heads, bh_src, bh_dst, attention_heads)
    assert fixed_heads == 4
    assert fixed_span == bh_dst


def _kwargs_for_fullbatch_call(tree: ast.AST, execution_mode: str, call_name: str) -> set[str]:
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.kwargs = None
            self._saw_mode = False

        def visit_Assign(self, node):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "execution_mode"
                and isinstance(node.value, ast.Constant)
                and node.value.value == execution_mode
            ):
                self._saw_mode = True
            self.generic_visit(node)

        def visit_Call(self, node):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            if self._saw_mode and func_name == call_name and self.kwargs is None:
                self.kwargs = {kw.arg for kw in node.keywords if kw.arg}
                self._saw_mode = False
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(tree)
    assert visitor.kwargs is not None, f"no {call_name} call after {execution_mode}"
    return visitor.kwargs


def test_default_fullbatch_paths_forward_runtime_batch():
    tree = ast.parse(_BLOCK_FN_SRC.read_text())
    required = {"full_batch_size", "micro_batch_size", "batch_offset"}
    merged = _kwargs_for_fullbatch_call(tree, "merged_fullbatch", "build_inference_metadata_batch")
    separate = _kwargs_for_fullbatch_call(tree, "separate_fullbatch", "build_inference_metadata")
    assert required <= merged
    assert required <= separate


def test_write_kvs_contains_fullhead_overflow_guard():
    src = _CACHE_SRC.read_text()
    assert "write_bh_span > BH_dst" in src
    assert "BH_src % H == 0" in src
