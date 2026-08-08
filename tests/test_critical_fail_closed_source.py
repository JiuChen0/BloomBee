import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _function_source(path: str, class_name: str, function_name: str) -> str:
    source_path = ROOT / path
    source = source_path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == function_name:
                    return ast.get_source_segment(source, item) or ""
    raise AssertionError(f"{class_name}.{function_name} not found in {path}")


def _module_function_source(path: str, function_name: str) -> str:
    source_path = ROOT / path
    source = source_path.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{function_name} not found in {path}")


def test_row_compaction_rejects_invalid_hypo_gathers():
    source = _function_source(
        "src/bloombee/server/memory_cache_manager.py",
        "KVCacheManager",
        "permute_batch_rows",
    )

    assert "assert 0 < n <= B_cache" not in source
    assert "perm < 0" in source
    assert "perm >= B_cache" in source
    assert "perm.unique().numel()" in source
    assert "duplicate row indices" in source


def test_s2s_int8_hidden_requires_quant_metadata():
    source = _module_function_source(
        "src/bloombee/utils/s2s_activation_quant.py",
        "dequantize_s2s_hidden_from_transport",
    )

    assert "Received int8 S2S hidden states without quantization metadata" in source
    assert "Unsupported S2S activation quantization scheme" in source


def test_local_spec_mask_mismatch_fails_closed():
    source = _function_source(
        "src/bloombee/server/backend.py",
        "TransformerBackend",
        "_spec_cache_valid_mask",
    )

    assert "falling back to all-prefix-valid cache mask" not in source
    assert "raise RuntimeError" in source
    assert "refusing to expose all cache columns" in source
