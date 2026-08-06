from pathlib import Path


def test_prefill_tree_position_ids_use_actual_mask_suffix():
    source_path = Path(__file__).resolve().parents[1] / "src/bloombee/server/backend.py"
    source = source_path.read_text()

    assert "prompt_part_len = target_seq_len - tree_len" not in source
    assert "actual_tree_len = max(0, total_len - prompt_part_len)" in source
    assert "local_tree_mask = mask[:, prompt_part_len:total_len, prompt_part_len:total_len]" in source
    assert "tree_offsets = local_tree_mask.to(torch.long).sum(dim=-1).sub_(1).clamp_min_(0)" in source
