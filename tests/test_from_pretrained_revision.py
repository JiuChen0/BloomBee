from types import SimpleNamespace

import torch

from bloombee.server import from_pretrained


def test_load_pretrained_block_forwards_revision_to_config_and_weights(monkeypatch):
    revision = "0123456789abcdef"
    config_calls = []
    weight_calls = []
    config = SimpleNamespace(
        block_class=from_pretrained.WrappedBloomBlock,
        block_prefix="transformer.h",
        model_type="bloom",
        _name_or_path="test/model",
    )
    block = torch.nn.Linear(1, 1)

    def fake_config_from_pretrained(model_name, **kwargs):
        config_calls.append((model_name, kwargs))
        return config

    def fake_load_hf_block_weights(*args, **kwargs):
        weight_calls.append((args, kwargs))
        return block

    monkeypatch.setattr(
        from_pretrained.AutoDistributedConfig,
        "from_pretrained",
        fake_config_from_pretrained,
    )
    monkeypatch.setattr(from_pretrained, "resolve_block_dtype", lambda config, dtype: dtype)
    monkeypatch.setattr(from_pretrained, "get_model_block", lambda *args, **kwargs: block)
    monkeypatch.setattr(from_pretrained, "_load_hf_block_weights", fake_load_hf_block_weights)

    result = from_pretrained.load_pretrained_block(
        "test/model",
        block_index=3,
        env=None,
        policy=None,
        weight_home=None,
        path="unused",
        torch_dtype=torch.float32,
        revision=revision,
        token="token",
        cache_dir="/tmp/cache",
    )

    assert result is block
    assert config_calls == [("test/model", {"use_auth_token": "token", "revision": revision})]
    assert weight_calls[0][1]["revision"] == revision


def test_sharded_index_and_weights_use_requested_revision(tmp_path, monkeypatch):
    revision = "0123456789abcdef"
    block_prefix = "model.layers.3."
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text('{"weight_map": {"model.layers.3.weight": "model-00001-of-00002.safetensors"}}')
    index_calls = []
    file_calls = []
    shard_calls = []

    def fake_find_index_file(model_name, **kwargs):
        index_calls.append((model_name, kwargs))
        return index_path.name

    def fake_get_file_from_repo(model_name, **kwargs):
        file_calls.append((model_name, kwargs))
        return str(index_path)

    def fake_load_state_dict_file(model_name, filename, **kwargs):
        shard_calls.append((model_name, filename, kwargs))
        return {"model.layers.3.weight": torch.ones(1)}

    monkeypatch.setattr(from_pretrained, "_find_index_file", fake_find_index_file)
    monkeypatch.setattr(from_pretrained, "get_file_from_repo", fake_get_file_from_repo)
    monkeypatch.setattr(
        from_pretrained,
        "_load_state_dict_from_repo_file",
        fake_load_state_dict_file,
    )

    state_dict = from_pretrained._load_state_dict_from_repo(
        "test/model",
        block_prefix,
        revision=revision,
        cache_dir=str(tmp_path),
    )

    assert set(state_dict) == {"weight"}
    assert index_calls[0][1]["revision"] == revision
    assert file_calls[0][1]["revision"] == revision
    assert shard_calls[0][2]["revision"] == revision
