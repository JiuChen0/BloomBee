"""Source-level guard: rotary _apply must save inv_freq before Module.to(dtype).

BloomBee loads bare decoder layers and calls ``block.to(dtype=fp16/bf16)``.
HF's full-model ``_keep_in_fp32_modules`` does not run on those layers, so each
wrapper's ``_apply`` has to keep RoPE frequencies in fp32.

Restoring with ``buf.float()`` *after* the dtype cast keeps dtype=fp32 but
stores already-rounded values. That escaped dtype-only tests and shows up as
repeated-token collapse. These AST checks lock in "save originals, then
super()._apply, then write the saved tensors back".
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _apply_method(relative_path: str, class_name: str) -> ast.FunctionDef:
    tree = ast.parse((ROOT / relative_path).read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_apply":
                    return item
    raise AssertionError(f"{class_name}._apply not found in {relative_path}")


def _calls_super_apply(stmt: ast.AST) -> bool:
    for node in ast.walk(stmt):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "_apply" or not isinstance(node.func.value, ast.Call):
            continue
        callee = node.func.value.func
        if isinstance(callee, ast.Name) and callee.id == "super":
            return True
    return False


def _split_around_super_apply(fn: ast.FunctionDef):
    idx = next((i for i, stmt in enumerate(fn.body) if _calls_super_apply(stmt)), None)
    if idx is None:
        raise AssertionError(f"{fn.name} never calls super()._apply")
    return fn.body[:idx], fn.body[idx + 1 :]


def _mentions_name(stmts, name: str) -> bool:
    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and node.id == name:
                return True
    return False


def _calls_method(stmts, method: str) -> bool:
    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == method:
                return True
    return False


def test_qwen3_saves_rotary_before_dtype_apply():
    before, after = _split_around_super_apply(
        _apply_method("src/bloombee/models/qwen3/block.py", "WrappedQwen3Block")
    )
    assert _mentions_name(before, "rotary_buffers")
    assert _mentions_name(after, "rotary_buffers")
    assert not _calls_method(after, "float")


def test_gemma4_saves_rotary_before_dtype_apply():
    before, after = _split_around_super_apply(
        _apply_method("src/bloombee/models/gemma4/block.py", "WrappedGemma4Block")
    )
    assert _mentions_name(before, "rotary_buffers")
    assert _mentions_name(after, "rotary_buffers")
    # The previous restore used register_buffer(..., buf.float()) after the cast.
    assert not _calls_method(after, "float")
    assert not _calls_method(after, "register_buffer")


def test_deepseekv3_saves_rotary_before_dtype_apply():
    before, after = _split_around_super_apply(
        _apply_method("src/bloombee/models/deepseekv3/block.py", "WrappedDeepseekV3Block")
    )
    assert _mentions_name(before, "rotary_buffers")
    assert _mentions_name(after, "rotary_buffers")
    assert not _calls_method(after, "float")


def test_gpt_oss_saves_rotary_before_dtype_apply():
    before, after = _split_around_super_apply(
        _apply_method("src/bloombee/models/gpt_oss/block.py", "WrappedGptOssBlock")
    )
    assert _mentions_name(before, "rotary_buffers")
    assert _mentions_name(after, "rotary_buffers")
    # Layernorms are intentionally forced to fp32 with .float() after the cast;
    # rotary restore must still use the pre-cast tensors, not buf.float().
    restored_from_saved = False
    for stmt in after:
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "to"
            ):
                for child in ast.walk(node.func.value):
                    if isinstance(child, ast.Name) and child.id == "value":
                        restored_from_saved = True
    assert restored_from_saved
