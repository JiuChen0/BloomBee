from pathlib import Path


def _function_body(source: str, name: str) -> str:
    marker = f"def {name}("
    start = source.index(marker)
    next_def = source.find("\ndef ", start + len(marker))
    if next_def == -1:
        next_def = len(source)
    return source[start:next_def]


def test_byte_split_decoders_validate_elem_size_before_size_arithmetic():
    source = Path("src/bloombee/utils/lossless_transport.py").read_text()

    validator = _function_body(source, "_validate_byte_split_elem_size")
    assert "elem_size not in (2, 4)" in validator

    for name in (
        "_decode_dict_byte_split_with",
        "_decode_zstd_byte_split_payload",
        "_decode_zstd_byte_split_high_only_payload",
    ):
        body = _function_body(source, name)
        unpack_idx = body.index("_BYTE_SPLIT_PAYLOAD_STRUCT.unpack_from")
        validate_idx = body.index("_validate_byte_split_elem_size")
        arithmetic_idx = body.index("original_size // elem_size")
        assert unpack_idx < validate_idx < arithmetic_idx
        assert "original_size % max(1, elem_size)" not in body
