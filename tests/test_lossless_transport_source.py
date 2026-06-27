from pathlib import Path


LOSSLESS_SOURCE = Path("src/bloombee/utils/lossless_transport.py")


def test_lossless_receive_validates_declared_sizes_before_decompression():
    source = LOSSLESS_SOURCE.read_text()

    assert "_DEFAULT_LOSSLESS_MAX_ORIGINAL_BYTES = 256 * 1024 * 1024" in source
    assert "def _validate_lossless_original_size" in source
    assert "if original_size > max_original_size" in source
    assert "def _decompress_with_algo" in source
    assert "_validate_lossless_original_size(original_size)\n    t0 = time.perf_counter()" in source
    assert "if strict:\n        _validate_lossless_original_size(original_size)" in source


def test_zipnn_transport_receive_is_explicit_opt_in():
    source = LOSSLESS_SOURCE.read_text()

    assert '_LOSSLESS_ZIPNN_TRANSPORT_ENV = "BLOOMBEE_LOSSLESS_ZIPNN_TRANSPORT"' in source
    assert "def _zipnn_transport_enabled" in source
    assert "return _zipnn_transport_enabled() and _supports_zipnn_compare" in source
    assert "if not _zipnn_transport_enabled()" in source
    assert "ZipNN transport receive is disabled" in source
