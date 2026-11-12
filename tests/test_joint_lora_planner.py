"""Joint planner constraints, exact search oracle and mixed-rank PEFT parity."""

import itertools
from dataclasses import replace

import pytest
import torch

from bloombee.utils.lora import LoRAConfig, LoRAProjection, lora_context, lora_linear, projection_key
from bloombee.utils.training_plan import StageChoice, TrainingPlan, TrainingWorkload, solve_training_plan


def candidate(worker, start, end, rank, cost, memory, utility, percent=100):
    return StageChoice(
        worker,
        start,
        end,
        (rank,) * (end - start),
        percent,
        cost,
        {"shared_gpu": memory},
        utility,
        "synthetic-correctness-only",
    )


def problem():
    return dict(
        workload=TrainingWorkload("test", 2, 1, 8, 64, accumulation=2, model_revision="hf:" + "a" * 40),
        base_lora=LoRAConfig(r=4, lora_alpha=8),
        candidates=[
            candidate("a", 0, 1, 2, 1, 3, 1),
            candidate("a", 0, 1, 8, 2, 5, 2),
            candidate("b", 1, 2, 2, 1, 3, 1),
            candidate("b", 1, 2, 8, 2, 5, 4),
            candidate("b", 1, 2, 8, 3, 2, 4, percent=0),
            candidate("a", 0, 2, 8, 3, 10, 6),
        ],
        resource_budgets={"shared_gpu": 8},
        baseline_resources={"shared_gpu": 1},
        minimum_utility=5,
        max_parameters=640,
    )


def test_joint_search_chooses_rank_placement_and_offload_together(tmp_path):
    plan = solve_training_plan(**problem())
    assert [(s.worker, s.ranks, s.weight_gpu_percent) for s in plan.stages] == [("a", (2,), 100), ("b", (8,), 0)]
    assert plan.parameters == 640 and plan.utility == 5 and plan.resources == {"shared_gpu": 6}
    assert plan.estimated_rpc_seconds_per_step == 8
    assert plan.lora_config.rank_for(0, "q_proj") == 2
    assert plan.lora_config.rank_for(1, "v_proj") == 8
    assert plan.lora_config.alpha_for(1, "v_proj") == 16
    path = tmp_path / "plan.json"
    plan.save(path)
    assert TrainingPlan.load(path) == plan
    with pytest.raises(FileExistsError):
        plan.save(path)


@pytest.mark.parametrize("budget", [4, 6, 8, 11])
@pytest.mark.parametrize("floor", [0, 2, 5, 6])
def test_solver_matches_independent_exhaustive_oracle(budget, floor):
    args = problem()
    args.update(resource_budgets={"shared_gpu": budget}, minimum_utility=floor, max_parameters=1024)
    feasible = []
    for length in (1, 2):
        for stages in itertools.product(args["candidates"], repeat=length):
            if stages[0].start != 0 or stages[-1].end != 2:
                continue
            if any(a.end != b.start for a, b in zip(stages, stages[1:])):
                continue
            if len({s.worker for s in stages}) != length:
                continue
            if 1 + sum(s.resource_bytes["shared_gpu"] for s in stages) > budget:
                continue
            if sum(s.utility for s in stages) < floor or sum(sum(s.ranks) * 64 for s in stages) > 1024:
                continue
            feasible.append(sum(s.rpc_seconds for s in stages) * 2)
    if feasible:
        assert solve_training_plan(**args).estimated_rpc_seconds_per_step == min(feasible)
    else:
        with pytest.raises(ValueError, match="No feasible"):
            solve_training_plan(**args)


def test_search_budget_is_explicit_and_invalid_plans_are_rejected():
    args = problem()
    with pytest.raises(RuntimeError, match="search budget"):
        solve_training_plan(**args, max_states=1)
    plan = solve_training_plan(**args)
    with pytest.raises(ValueError, match="shared resource"):
        replace(plan, resource_budgets={"shared_gpu": 5})
    with pytest.raises(ValueError, match="ordered layers"):
        replace(plan, stages=plan.stages[::-1])
    with pytest.raises(ValueError, match="each worker"):
        replace(plan, stages=(plan.stages[0], replace(plan.stages[1], worker="a")))
    with pytest.raises(ValueError, match="every model layer"):
        replace(plan, stages=plan.stages[:1])
    with pytest.raises(ValueError, match="parameter budget"):
        replace(plan, max_parameters=639)
    with pytest.raises(ValueError, match="Missing resource"):
        solve_training_plan(**{**args, "resource_budgets": {"other_gpu": 100}})
    with pytest.raises(ValueError, match="integer"):
        replace(plan.stages[0], weight_gpu_percent=50.5)


def test_new_plan_requires_revision_but_can_read_legacy_plan():
    args = problem()
    plan = solve_training_plan(**args)
    old = plan.to_dict()
    del old["workload"]["model_revision"]
    legacy = TrainingPlan.from_dict(old)
    assert legacy.to_dict() == old
    with pytest.raises(ValueError, match="model_revision"):
        solve_training_plan(**{**args, "workload": legacy.workload})


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_non_fp16_planning_requires_a_supported_model_family(dtype):
    args = problem()
    workload = replace(args["workload"], input_dtype=str(dtype))
    with pytest.raises(ValueError, match="model_type"):
        solve_training_plan(**{**args, "workload": workload})
    with pytest.raises(ValueError, match="LLaMA.*FP16"):
        replace(workload, model_type="llama")
    assert (
        solve_training_plan(**{**args, "workload": replace(workload, model_type="qwen3")}).workload.torch_dtype == dtype
    )


def test_planner_excludes_spans_over_checkpoint_or_activation_cache_limits():
    args = problem()
    args.update(
        workload=replace(args["workload"], hidden_size=4),
        minimum_utility=0,
        resource_budgets={"shared_gpu": 20},
        max_parameters=1024,
    )
    stage = replace(args["candidates"][-1], execution_mode="span_cached_gpu")
    with pytest.raises(ValueError, match="measured adapter_cache_limit_bytes"):
        solve_training_plan(**{**args, "candidates": [stage]})
    stage = replace(stage, adapter_cache_limit_bytes=256, cache_ttl_seconds=12)
    # A boundary is 64 bytes. Backward reconstruction needs 64, retention 128.
    backward_over_limit = replace(stage, checkpoint_limit_bytes=63)
    retention_over_limit = replace(stage, activation_cache_limit_bytes=127)
    recompute = replace(args["candidates"][-1], execution_mode="span_recompute", rpc_seconds=10)
    args["candidates"] = [backward_over_limit, retention_over_limit, recompute]
    assert solve_training_plan(**args).stages == (recompute,)
    with pytest.raises(ValueError, match="checkpoint budget exceeded"):
        solve_training_plan(**{**args, "candidates": [backward_over_limit]})
    with pytest.raises(ValueError, match="activation cache budget exceeded"):
        solve_training_plan(**{**args, "candidates": [retention_over_limit]})
    zero = replace(stage, checkpoint_limit_bytes=64, activation_cache_limit_bytes=0)
    assert solve_training_plan(**{**args, "candidates": [zero]}).stages == (zero,)


def test_local_base_revision_matches_copied_files_and_detects_changed_weights(tmp_path):
    from types import SimpleNamespace

    from bloombee.utils.lora_model_revision import model_revision

    first, second = (tmp_path / name for name in ("first", "second"))
    config = SimpleNamespace(_commit_hash="a" * 40)
    for root in (first, second):
        root.mkdir()
        (root / "config.json").write_text("{}")
        (root / "model.safetensors").write_bytes(b"weights-a")
    assert model_revision(first, config) == model_revision(second, config)
    (second / "model.safetensors").write_bytes(b"weights-b")
    assert model_revision(first, config) != model_revision(second, config)
    assert model_revision("org/model", config) == "hf:" + "a" * 40
    with pytest.raises(ValueError, match="pinned Hub commit"):
        model_revision("org/model", SimpleNamespace())


@pytest.mark.parametrize(
    "patterns",
    [
        {"rank_pattern": {"layers.0.self_attn.q_proj": 0}},
        {"rank_pattern": {"layers.0.self_attn.q_proj": True}},
        {"rank_pattern": {"layers.0.mlp.q_proj": 2}},
        {"rank_pattern": {"q_proj": 2}},
        {"rank_pattern": {"layers.*.self_attn.q_proj": 2}},
        {"alpha_pattern": {"layers.0.self_attn.q_proj": float("nan")}},
    ],
)
def test_invalid_patterns_fail_instead_of_silently_falling_back(patterns):
    with pytest.raises(ValueError):
        LoRAConfig(**patterns)


def test_uniform_v1_config_serialization_is_unchanged():
    assert set(LoRAConfig().to_dict()) == {"r", "lora_alpha", "lora_dropout", "target_modules"}
    with pytest.raises(ValueError, match="outside"):
        LoRAConfig(rank_pattern={projection_key(2, "q_proj"): 2}).validate_layers(2)


def test_worker_rejects_deployment_with_extra_unplanned_layers():
    import asyncio
    from types import SimpleNamespace

    from bloombee.server.lora import LoRAService

    async def gather(requests, context):
        return "model.0", [], dict(version=2, config=LoRAConfig().to_dict(), planned_span=[0, 1])

    service = LoRAService(
        SimpleNamespace(
            request_timeout=5, _gather_inputs=gather, module_backends={"model.0": object(), "model.1": object()}
        )
    )

    async def call():
        async for response in service.rpc_lora_stream(None, None):
            pytest.fail("Mismatched placement was accepted")

    with pytest.raises(ValueError, match="layer placement"):
        asyncio.run(call())


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"model_revision": "sha256:" + "b" * 64}, "base model revision"),
        ({"checkpoint_limit_bytes": 64}, "checkpoint limit does not match"),
        ({"version": 4, "execution_mode": "span_cached_gpu", "adapter_cache_limit_bytes": 0}, "adapter_cache_limit"),
        ({"version": 4, "execution_mode": "span_cached_gpu", "cache_ttl_seconds": 1}, "cache_ttl_seconds"),
        ({}, "checkpoint budget exceeded"),
    ],
)
def test_worker_rejects_incompatible_plan_before_forward_runtime_task(overrides, match):
    import asyncio
    from types import SimpleNamespace

    from bloombee.server.lora import LoRAService

    revision = "sha256:" + "a" * 64
    x = torch.zeros(1, 8, 4, dtype=torch.float16)

    async def gather(requests, context):
        return (
            "model.0 model.1",
            [x, *([torch.empty(0)] * 8)],
            dict(
                config=LoRAConfig().to_dict(),
                planned_span=[0, 2],
                backward=False,
                training=True,
                seed=1,
                **{"version": 3, "execution_mode": "span_gpu", "model_revision": revision, **overrides},
            ),
        )

    async def submit_task(*args, **kwargs):
        pytest.fail("An incompatible plan reached Runtime")

    span_pool = SimpleNamespace(
        process_func=SimpleNamespace(
            checkpoint_bytes=63, cache_settings={"adapter_cache_limit_bytes": 256, "cache_ttl_seconds": 300}
        ),
        submit_task=submit_task,
    )
    backend = SimpleNamespace(
        config=SimpleNamespace(_bloombee_lora_model_revision=revision),
        lora_pool=object(),
        lora_span_pool=span_pool,
    )
    service = LoRAService(
        SimpleNamespace(
            request_timeout=5, _gather_inputs=gather, module_backends={"model.0": backend, "model.1": backend}
        )
    )

    async def call():
        async for _ in service.rpc_lora_stream(None, SimpleNamespace(remote_id="test-peer")):
            pytest.fail("An incompatible plan received a response")

    with pytest.raises(ValueError, match=match):
        asyncio.run(call())


def test_planned_route_never_falls_back_when_a_peer_disappears():
    import threading
    from types import MethodType, SimpleNamespace

    from bloombee.client.lora import LoRARemoteSequential
    from bloombee.client.routing.sequence_manager import RemoteSequenceManager
    from bloombee.data_structures import ServerState

    plan = solve_training_plan(**problem())
    info = SimpleNamespace(state=ServerState.ONLINE)
    ready = threading.Event()
    ready.set()
    blocks = [SimpleNamespace(servers={"peer-a": info}), SimpleNamespace(servers={"peer-b": info})]
    manager = SimpleNamespace(
        ready=ready,
        _thread_start_lock=threading.Lock(),
        is_alive=lambda: True,
        lock_changes=threading.Lock(),
        state=SimpleNamespace(sequence_info=SimpleNamespace(block_infos=blocks)),
    )
    manager._ensure_routing_ready = MethodType(RemoteSequenceManager._ensure_routing_ready, manager)
    manager.make_fixed_sequence = MethodType(RemoteSequenceManager.make_fixed_sequence, manager)
    sequence = SimpleNamespace(
        training_plan=plan, sequence_manager=manager, worker_peers={"a": "peer-a", "b": "peer-b"}
    )
    inputs = torch.zeros(1, 8, 4, dtype=torch.float16)
    route = LoRARemoteSequential._planned_route(sequence, inputs)
    assert [(s.peer_id, s.start, s.end) for s in route] == [("peer-a", 0, 1), ("peer-b", 1, 2)]
    blocks[1].servers = {"unplanned-peer": info}
    with pytest.raises(RuntimeError, match="explicit replanning"):
        LoRARemoteSequential._planned_route(sequence, inputs)
    with pytest.raises(ValueError, match="shape/dtype"):
        LoRARemoteSequential._planned_route(sequence, inputs.float())


def test_mixed_rank_alpha_matches_peft_model_and_all_gradients():
    from peft import LoraConfig as PeftConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(71)
    config = LoRAConfig(
        r=4,
        lora_alpha=8,
        target_modules=("q_proj", "v_proj"),
        rank_pattern={projection_key(0, "q_proj"): 2, projection_key(1, "v_proj"): 8},
        alpha_pattern={projection_key(0, "q_proj"): 3, projection_key(1, "v_proj"): 12},
    )
    reference = get_peft_model(
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        ),
        PeftConfig(**config.to_dict()),
    )
    count = 0
    for index, layer in enumerate(reference.base_model.model.model.layers):
        for target in config.target_modules:
            linear = getattr(layer.self_attn, target)
            rank = config.rank_for(index, target)
            assert linear.r["default"] == rank
            assert linear.scaling["default"] == config.alpha_for(index, target) / rank
            adapter = LoRAProjection(16, 16, rank)
            with torch.no_grad():
                adapter.lora_B.normal_()
                linear.lora_A["default"].weight.copy_(adapter.lora_A)
                linear.lora_B["default"].weight.copy_(adapter.lora_B)
            x = torch.randn(2, 3, 16, requires_grad=True)
            other = x.detach().clone().requires_grad_()
            with lora_context(config, (adapter.lora_A, adapter.lora_B) * 2, training=True, seed=2, layer_index=index):
                actual = lora_linear(x, linear.base_layer.weight, target)
            expected = linear(other)
            grad = torch.randn_like(actual)
            actual_grads = torch.autograd.grad(actual, (x, adapter.lora_A, adapter.lora_B), grad)
            expected_grads = torch.autograd.grad(
                expected, (other, linear.lora_A["default"].weight, linear.lora_B["default"].weight), grad
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            for a, b in zip(actual_grads, expected_grads):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            count += sum(p.numel() for p in adapter.parameters())
    assert count == 32 * (2 + 4 + 4 + 8)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_base_fp32_adapter_addition_matches_peft(dtype):
    from peft.tuners.lora.layer import Linear
    from torch import nn

    torch.manual_seed(431)
    base = nn.Linear(16, 24, bias=False).to(dtype).requires_grad_(False)
    reference = Linear(base, "default", r=2, lora_alpha=3, lora_dropout=0.0)
    reference.lora_A["default"].float()
    reference.lora_B["default"].float()
    adapter = LoRAProjection(16, 24, 2)
    with torch.no_grad():
        adapter.lora_B.normal_(0, 0.1)
        reference.lora_A["default"].weight.copy_(adapter.lora_A)
        reference.lora_B["default"].weight.copy_(adapter.lora_B)
    x = torch.randn(2, 8, 16, dtype=dtype, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    with lora_context(
        LoRAConfig(r=2, lora_alpha=3, target_modules=("q_proj",)),
        (adapter.lora_A, adapter.lora_B),
        training=True,
        seed=3,
    ):
        actual = lora_linear(x, base.weight, "q_proj")
    expected = reference(other)
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (x, adapter.lora_A, adapter.lora_B), grad)
    expected_grads = torch.autograd.grad(
        expected, (other, reference.lora_A["default"].weight, reference.lora_B["default"].weight), grad
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_v2_dropout_and_gradients_do_not_depend_on_span_partition(monkeypatch):
    from types import SimpleNamespace

    from torch import nn

    from bloombee.client import lora as client

    class Manager:
        def __len__(self):
            return 2

        def make_sequence(self, **kwargs):
            return self.route

    class Original(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="llama", hidden_size=8, intermediate_size=16, num_attention_heads=2, num_key_value_heads=2
            )
            self.sequence_manager = Manager()

        def __len__(self):
            return 2

    def local_apply(inputs, manager, span, metadata, *parameters):
        for j, layer in enumerate(range(span.start, span.end)):
            with lora_context(
                config, parameters[2 * j : 2 * j + 2], training=True, seed=metadata["seed"] + j, layer_index=layer
            ):
                inputs = lora_linear(inputs, torch.eye(8), "q_proj")
        return inputs

    config = LoRAConfig(
        r=2, lora_dropout=0.3, target_modules=("q_proj",), rank_pattern={projection_key(1, "q_proj"): 4}
    )
    sequence = client.LoRARemoteSequential(Original(), config)
    with torch.no_grad():
        for layer in sequence.adapters:
            layer["q_proj"].lora_B.normal_(0, 0.1)
    monkeypatch.setattr(client, "_RemoteLoRAFunction", SimpleNamespace(apply=local_apply))
    manager = sequence.sequence_manager
    x = torch.randn(1, 8, 8)
    outputs, gradients, rngs = [], [], []
    for route in (
        (SimpleNamespace(start=0, end=2),),
        (SimpleNamespace(start=0, end=1), SimpleNamespace(start=1, end=2)),
    ):
        manager.route = route
        torch.manual_seed(921)
        sequence.zero_grad(set_to_none=True)
        outputs.append(sequence(x))
        outputs[-1].sum().backward()
        gradients.append([p.grad.clone() for p in sequence.parameters()])
        rngs.append(torch.get_rng_state())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    assert torch.equal(rngs[0], rngs[1])
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
