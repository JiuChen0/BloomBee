from pathlib import Path


FALCON_BLOCK_SOURCE = Path("src/bloombee/models/falcon/block.py")


def test_falcon_new_decoder_reads_only_written_kv_heads_from_cache():
    source = FALCON_BLOCK_SOURCE.read_text()

    assert "if self.config.new_decoder_architecture:" in source
    assert "num_kv_heads = int(self.config.num_kv_heads)" in source
    assert "elif getattr(self.config, 'multi_query', False):" in source
    assert "num_kv_heads = 1 if (not self.config.new_decoder_architecture" not in source
