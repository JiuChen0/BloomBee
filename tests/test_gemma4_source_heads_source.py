"""Source-level coverage for Gemma-4 full vs sliding KV head inference.

Importing KVCacheManager pulls hivemind/torch/flexgen, which are not always
present in the minimal automation environment. This test extracts
``_source_heads_per_batch`` from the manager source and drives the Gemma-4
31B layout that previously crashed or silently collapsed batches.
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace
from typing import Optional

MANAGER_PATH = pathlib.Path(__file__).resolve().parents[1] / "src" / "bloombee" / "server" / "memory_cache_manager.py"


def _load_source_heads_fn():
    tree = ast.parse(MANAGER_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "KVCacheManager":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_source_heads_per_batch":
                    module = ast.Module(body=[item], type_ignores=[])
                    ast.fix_missing_locations(module)
                    namespace = {"Optional": Optional}
                    exec(compile(module, str(MANAGER_PATH), "exec"), namespace)
                    return namespace["_source_heads_per_batch"]
    raise AssertionError("_source_heads_per_batch not found")


_source_heads_per_batch = _load_source_heads_fn()


def _gemma4_31b_config():
    return SimpleNamespace(
        num_attention_heads=32,
        num_key_value_heads=16,
        num_global_key_value_heads=4,
        head_dim=256,
        global_head_dim=512,
    )


def _infer(config, source_bh, source_head_dim, full_batch_size=0, micro_batch_size=0):
    return _source_heads_per_batch(
        SimpleNamespace(block_config=config),
        32,
        source_bh,
        full_batch_size,
        micro_batch_size,
        source_head_dim,
    )


def test_gemma4_full_attention_batch1_uses_global_kv_heads():
    # B=1 full-attention write: BH=4, D=512. Sliding kv_heads=16 does not divide 4,
    # so the old path returned H=32 and asserted BH % 32 == 0.
    assert _infer(_gemma4_31b_config(), source_bh=4, source_head_dim=512) == 4


def test_gemma4_full_attention_batch4_does_not_collapse_to_sliding():
    # B=4 full-attention write: BH=16 is divisible by sliding kv_heads=16.
    # Treating that as one sliding row silently corrupts the other three sequences.
    assert _infer(_gemma4_31b_config(), source_bh=16, source_head_dim=512) == 4


def test_gemma4_sliding_attention_keeps_sliding_kv_heads():
    assert _infer(_gemma4_31b_config(), source_bh=16, source_head_dim=256) == 16
    assert _infer(_gemma4_31b_config(), source_bh=64, source_head_dim=256) == 16


def test_uniform_gqa_still_uses_num_key_value_heads():
    llama_like = SimpleNamespace(
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
    )
    assert _infer(llama_like, source_bh=8, source_head_dim=128) == 8
    assert _infer(llama_like, source_bh=32, source_head_dim=128) == 8


def test_microbatch_inference_still_matches_emitted_rows():
    # When the write batch is known, BH / micro_batch_size is the source stride.
    assert _infer(_gemma4_31b_config(), source_bh=16, source_head_dim=512, full_batch_size=16, micro_batch_size=4) == 4
    assert _infer(_gemma4_31b_config(), source_bh=64, source_head_dim=256, full_batch_size=16, micro_batch_size=4) == 16
