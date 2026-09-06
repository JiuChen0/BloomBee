"""Source-level regression tests for Gemma-4 heterogeneous KV head inference.

``KVCacheManager`` is shared across sliding and full-attention layers. The
TF 5.16 ``per_layer_config`` path used to union every layer's
``num_key_value_heads`` and treat ``source_bh in candidates`` as decisive.
On Gemma-4 31B that collides: batch=4 full-attention writes have BH=16,
which also equals the sliding-layer head count, so four sequences were
packed into one sliding row.

These tests extract ``_source_heads_per_batch`` from source so they run
without hivemind/torch.
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MANAGER_PATH = REPO_ROOT / "src" / "bloombee" / "server" / "memory_cache_manager.py"


def _load_source_heads_fn():
    tree = ast.parse(MANAGER_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "KVCacheManager":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_source_heads_per_batch":
                    module = ast.Module(body=[item], type_ignores=[])
                    ast.fix_missing_locations(module)
                    ns = {}
                    exec(compile(module, str(MANAGER_PATH), "exec"), ns)
                    return ns["_source_heads_per_batch"]
    raise AssertionError("_source_heads_per_batch not found")


def _manager(block_config):
    fn = _load_source_heads_fn()
    mgr = SimpleNamespace(block_config=block_config)
    return lambda *args, **kwargs: fn(mgr, *args, **kwargs)


def _layer(num_key_value_heads, head_dim):
    return SimpleNamespace(num_key_value_heads=num_key_value_heads, head_dim=head_dim)


def _gemma4_31b_per_layer_config():
    # 5 sliding + 1 full, repeated, matching the 31B 5:1 pattern.
    sliding = _layer(16, 256)
    full = _layer(4, 512)
    return [sliding] * 5 + [full] + [sliding] * 5 + [full]


def _gemma4_31b_config(*, with_per_layer=True):
    return SimpleNamespace(
        num_attention_heads=32,
        num_key_value_heads=16,
        num_global_key_value_heads=4,
        head_dim=256,
        global_head_dim=512,
        per_layer_config=_gemma4_31b_per_layer_config() if with_per_layer else None,
    )


def test_full_attention_batch4_not_collapsed_onto_sliding_heads():
    """B=4 full-attention write: BH=16 must stay 4 heads, not sliding's 16."""
    infer = _manager(_gemma4_31b_config())
    assert infer(32, 16, source_head_dim=512) == 4


def test_full_attention_batch8_not_max_of_all_layer_heads():
    """B=8 full-attention write: BH=32 is divisible by both 4 and 16."""
    infer = _manager(_gemma4_31b_config())
    assert infer(32, 32, source_head_dim=512) == 4


def test_sliding_and_full_single_batch_still_match():
    infer = _manager(_gemma4_31b_config())
    assert infer(32, 16, source_head_dim=256) == 16
    assert infer(32, 4, source_head_dim=512) == 4


def test_sliding_batched_write_keeps_sliding_heads():
    infer = _manager(_gemma4_31b_config())
    assert infer(32, 64, source_head_dim=256) == 16


def test_legacy_named_fields_use_global_head_dim():
    infer = _manager(_gemma4_31b_config(with_per_layer=False))
    assert infer(32, 16, source_head_dim=512) == 4
    assert infer(32, 16, source_head_dim=256) == 16
    assert infer(32, 32, source_head_dim=512) == 4


def test_runtime_batch_hint_still_wins():
    infer = _manager(_gemma4_31b_config())
    assert infer(32, 16, full_batch_size=4, micro_batch_size=4, source_head_dim=512) == 4
