"""Source-level lock for Mixtral generate() injecting RemotePastKeyValues.

Transformers 5 removed `_supports_cache_class` from MixtralPreTrainedModel.
RemoteGenerationMixin.generate used to read the attribute directly, so Mixtral
`.generate()` crashed before the remote KV stub was installed.
"""

import ast
from pathlib import Path


_MIXTRAL_MODEL = Path("src/bloombee/models/mixtral/model.py")
_REMOTE_GENERATION = Path("src/bloombee/client/remote_generation.py")

_CAUSAL_LM_FILES = (
    Path("src/bloombee/models/llama/model.py"),
    Path("src/bloombee/models/bloom/model.py"),
    Path("src/bloombee/models/falcon/model.py"),
    Path("src/bloombee/models/qwen3/model.py"),
    Path("src/bloombee/models/gemma4/model.py"),
    Path("src/bloombee/models/mixtral/model.py"),
    Path("src/bloombee/models/deepseekv3/model.py"),
    Path("src/bloombee/models/gpt_oss/model.py"),
)


def _class_assigns_true(tree: ast.AST, class_name: str, attr: str) -> bool:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (
                    isinstance(item, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == attr for t in item.targets)
                    and isinstance(item.value, ast.Constant)
                    and item.value.value is True
                ):
                    return True
    return False


def test_mixtral_causal_lm_declares_cache_class_support():
    tree = ast.parse(_MIXTRAL_MODEL.read_text())
    assert _class_assigns_true(
        tree, "DistributedMixtralForCausalLM", "_supports_cache_class"
    ), "DistributedMixtralForCausalLM must set _supports_cache_class = True"


def test_all_causal_lms_declare_cache_class_support():
    missing = []
    for path in _CAUSAL_LM_FILES:
        tree = ast.parse(path.read_text())
        found = False
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if "ForCausalLM" not in node.name:
                continue
            if _class_assigns_true(tree, node.name, "_supports_cache_class"):
                found = True
                break
        if not found:
            missing.append(str(path))
    assert not missing, f"Causal LM wrappers missing _supports_cache_class = True: {missing}"


def test_generate_does_not_raw_access_cache_class_flag():
    tree = ast.parse(_REMOTE_GENERATION.read_text())
    generate_fn = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "RemoteGenerationMixin":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "generate":
                    generate_fn = item
    assert generate_fn is not None

    raw_access = False
    uses_getattr = False
    for node in ast.walk(generate_fn):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "_supports_cache_class"
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            raw_access = True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and any(
                isinstance(arg, ast.Constant) and arg.value == "_supports_cache_class"
                for arg in node.args
            )
        ):
            uses_getattr = True
            assert len(node.args) >= 3 and isinstance(node.args[2], ast.Constant) and node.args[2].value is True, (
                "getattr(_supports_cache_class) must default True so TF 5 Mixtral still "
                "injects RemotePastKeyValues"
            )
    assert uses_getattr, "generate() must getattr _supports_cache_class with a True default"
    assert not raw_access, "generate() must not raw-access self._supports_cache_class"
