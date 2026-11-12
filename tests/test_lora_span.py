"""Span execution parity, bounded activation storage and legacy plan compatibility."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from bloombee.server.lora import run_lora_block
from bloombee.server.lora_execution import LoRASpanExecutor
from bloombee.utils.lora import LoRAConfig, install_lora_hooks, projection_key, projection_shapes
from bloombee.utils.training_plan import StageChoice, TrainingPlan, TrainingWorkload


def setup_span(device="cpu"):
    from bloombee.models.qwen3.block import WrappedQwen3Block
    from bloombee.models.qwen3.config import DistributedQwen3Config
    from bloombee.utils.convert_block import QuantType, convert_block

    torch.manual_seed(741)
    model_config = DistributedQwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=5,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        attention_dropout=0.0,
    )
    model_config._attn_implementation = "eager"
    config = LoRAConfig(r=2, lora_dropout=0.25, rank_pattern={projection_key(4, t): 4 for t in ("q_proj", "v_proj")})
    backends, parameters = {}, []
    shapes = projection_shapes(model_config)
    for i in range(2, 5):
        module = WrappedQwen3Block(model_config, layer_idx=i).eval().requires_grad_(False)
        install_lora_hooks(module)
        wrapper = convert_block(module, i, model_config, [torch.device(device)], torch.device(device), QuantType.NONE)
        name = f"model.{i}"
        backends[name] = SimpleNamespace(
            name=name,
            config=model_config,
            module=wrapper,
            lora_pool=SimpleNamespace(device=device),
            _ensure_model_on_device=lambda: None,
        )
        parameters += [
            tensor
            for t in config.target_modules
            for tensor in (
                torch.randn(config.rank_for(i, t), shapes[t][1]) * 0.03,
                torch.randn(shapes[t][0], config.rank_for(i, t)) * 0.03,
            )
        ]
    x, grad = torch.randn(2, 4, 16), torch.randn(2, 4, 16)
    metadata = dict(config=config.to_dict(), seed=173, training=True, span_uids=list(backends))
    return backends, [p.to(device) for p in parameters], x.to(device), grad.to(device), metadata


def block_reference(backends, parameters, x, grad, metadata):
    inputs = [x]
    for j, backend in enumerate(backends.values()):
        meta = {**metadata, "seed": metadata["seed"] + j, "layer_index": 2 + j}
        inputs.append(run_lora_block(backend, inputs[-1], torch.empty(0), meta, *parameters[j * 4 : j * 4 + 4])[0])
    parameter_grads = [None] * len(backends)
    for j, backend in reversed(list(enumerate(backends.values()))):
        meta = {**metadata, "seed": metadata["seed"] + j, "layer_index": 2 + j}
        result = run_lora_block(backend, inputs[j], grad, meta, *parameters[j * 4 : j * 4 + 4])
        grad, parameter_grads[j] = result[0], result[1:]
    return inputs[-1], (grad, *(g for layer in parameter_grads for g in layer))


@pytest.mark.parametrize("mode", ["span_cpu", "span_gpu", "span_recompute"])
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA span parity"))],
)
def test_span_matches_block_outputs_gradients_dropout_and_base_weights(mode, device, monkeypatch):
    from bloombee.server import lora

    backends, params, x, grad, meta = setup_span(device)
    expected, expected_grads = block_reference(backends, params, x, grad, meta)
    snapshots = [(p, p.detach().clone()) for b in backends.values() for p in b.module.parameters()]
    rng = torch.get_rng_state().clone()
    calls = []
    original = lora.run_lora_block

    def counted(backend, inputs, gradient, metadata, *parameters, **kwargs):
        calls.append(bool(gradient.numel()))
        return original(backend, inputs, gradient, metadata, *parameters, **kwargs)

    monkeypatch.setattr(lora, "run_lora_block", counted)
    executor = LoRASpanExecutor(backends, checkpoint_bytes=2 * x.numel() * x.element_size())
    actual = executor(x, torch.empty(0), {**meta, "backward": False, "execution_mode": mode}, *params)
    actual_grads = executor(x, grad, {**meta, "backward": True, "execution_mode": mode}, *params)
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a.to(b.device), b, rtol=0, atol=0)
    assert len(calls) == (9 if mode == "span_recompute" else 8) and sum(calls) == 3
    assert torch.equal(rng, torch.get_rng_state())
    for parameter, snapshot in snapshots:
        assert parameter.grad is None and torch.equal(parameter, snapshot)


def test_span_budget_failure_happens_before_any_computation_and_recompute_needs_no_checkpoints(monkeypatch):
    from bloombee.server import lora

    backends, params, x, grad, meta = setup_span()
    executor = LoRASpanExecutor(backends, checkpoint_bytes=0)

    def unexpected(*args):
        pytest.fail("Checkpoint budget must be checked before computation")

    with monkeypatch.context() as patch:
        patch.setattr(lora, "run_lora_block", unexpected)
        with pytest.raises(RuntimeError, match="checkpoint budget"):
            executor(x, grad, {**meta, "backward": True, "execution_mode": "span_gpu"}, *params)
    result = executor(x, grad, {**meta, "backward": True, "execution_mode": "span_recompute"}, *params)
    assert len(result) == len(params) + 1 and all(torch.isfinite(t).all() for t in result)


def test_old_plan_serialization_and_new_executor_choice_roundtrip():
    stage = StageChoice("worker", 0, 2, (2, 4), 50, 1.0, {"gpu": 100}, 2, "test")
    plan = TrainingPlan(TrainingWorkload("test", 2, 1, 8, 64), LoRAConfig(), (stage,), {"gpu": 200}, 1, 1024)
    assert "execution_mode" not in plan.to_dict()["stages"][0]
    assert TrainingPlan.from_dict(plan.to_dict()) == plan
    new = replace(plan, stages=(replace(stage, execution_mode="span_cpu"),))
    assert TrainingPlan.from_dict(new.to_dict()) == new
    with pytest.raises(ValueError, match="execution mode"):
        replace(stage, execution_mode="unknown")
