"""Identify the base weights used by a planned LoRA deployment."""

import hashlib
import json
import re
from pathlib import Path


def valid_model_revision(value):
    return isinstance(value, str) and re.fullmatch(r"(hf:[0-9a-f]{40}|sha256:[0-9a-f]{64})", value) is not None


def model_revision(model_name, config):
    root = Path(model_name).expanduser()
    if not root.is_dir():
        revision = f"hf:{getattr(config, '_commit_hash', None)}"
        if not valid_model_revision(revision):
            raise ValueError(
                "Could not resolve the LoRA base model revision; use a pinned Hub commit or a local checkpoint"
            )
        return revision

    weights = {p.name for pattern in ("*.safetensors", "*.bin") for p in root.glob(pattern)}
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = root / name
        if index.is_file():
            weights.add(name)
            weights.update(json.loads(index.read_text())["weight_map"].values())
    if not weights:
        raise ValueError(f"No base model weight files to identify in {root}")
    digest = hashlib.sha256()
    for name in sorted({"config.json", *weights}):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid checkpoint file path: {name}")
        file_digest = hashlib.sha256()
        with (root / relative).open("rb") as stream:
            for chunk in iter(lambda: stream.read(2**20), b""):
                file_digest.update(chunk)
        digest.update(name.encode() + b"\0" + file_digest.digest())
    return f"sha256:{digest.hexdigest()}"
