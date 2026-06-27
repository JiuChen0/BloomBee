from pathlib import Path


BLOOM_BLOCK_SOURCE = Path("src/bloombee/models/bloom/block.py")


def test_bloom_decode_returns_only_new_kv_span_for_cache_append():
    source = BLOOM_BLOCK_SOURCE.read_text()

    assert "present_key_layer = key_layer" in source
    assert "present_value_layer = value_layer" in source
    assert "Only return the newly computed span" in source
    assert "present = (key_layer_bhd, value_layer_bhd)" not in source
