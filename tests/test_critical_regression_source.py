"""Source-level regression checks for critical inference-path invariants.

These checks avoid importing BloomBee's distributed runtime dependencies while
still locking in the small safety guards fixed here.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text()


def test_mixtral_preserves_or_builds_attention_mask():
    src = _read("src/bloombee/models/mixtral/block.py")

    assert "attention_mask = None" not in src
    assert "if attention_mask is None:" in src
    assert "torch.triu(causal, diagonal=past_key_values_length + 1)" in src
    assert "elif attention_mask.dim() == 3:" in src
    assert "attention_mask = attention_mask.unsqueeze(1)" in src


def test_relay_microbatch_merge_suppresses_duplicate_full_batch_push():
    src = _read("src/bloombee/server/block_functions.py")

    marker = 'step_metadata["cross_stage_pushed"] = True'
    yield_marker = "yield output_tensors, can_push, accum['step_metadata']"
    assert marker in src
    assert src.index(marker) < src.index(yield_marker)


def test_push_limiter_releases_slot_if_setup_fails_before_task_creation():
    src = _read("src/bloombee/server/handler.py")

    assert "release_slot_handed_off = False" in src
    assert "release_slot_handed_off = True" in src
    assert "if acquired_slot and not release_slot_handed_off" in src
    assert "await self._push_limiter.release(" in src


def test_mixed_device_gpu_multiplexing_fails_fast():
    src = _read("src/bloombee/server/memory_cache_manager.py")

    assert "MixedDevice KV cache is incompatible with GPU multiplexing" in src
    assert "raise RuntimeError(" in src


def test_gemma4_global_kv_head_inference_uses_source_head_dim():
    src = _read("src/bloombee/server/memory_cache_manager.py")

    assert "source_head_dim: int = 0" in src
    assert 'getattr(self.block_config, "global_head_dim", None)' in src
    assert 'getattr(self.block_config, "num_global_key_value_heads", None)' in src
    assert "source_head_dim == global_head_dim" in src
    assert "source_head_dim=D_src" in src
