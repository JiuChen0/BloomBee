import json
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from bloombee.server.lora_pool import LoRATaskPool
from bloombee.server.task_pool import Task
from bloombee.utils.lora_profile import expected_stateless_cost, measure, profile_step, request_metadata, tensor_bytes


def test_payload_accounting_matches_explicit_forward_backward_messages():
    x = torch.empty(2, 8, 32, dtype=torch.float16)
    params = (torch.empty(4, 32), torch.empty(32, 4), torch.empty(4, 32), torch.empty(32, 4))
    expected = expected_stateless_cost(2, tensor_bytes((x,)), tensor_bytes(params))
    assert expected["request_tensor_bytes"] == tensor_bytes((x, *params)) + tensor_bytes((x, *params, x))
    assert expected["response_tensor_bytes"] == tensor_bytes((x,)) + tensor_bytes((x, *params))
    assert expected["forward_calls"] == 5 and expected["backward_calls"] == 2


def test_disabled_profile_does_not_synchronize_or_write(monkeypatch, tmp_path):
    monkeypatch.delenv("BLOOMBEE_LORA_PROFILE_DIR", raising=False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: pytest.fail("Unexpected CUDA synchronization"))
    assert request_metadata() == {}
    with measure({}, "forward", "cuda"):
        pass
    assert not list(tmp_path.iterdir())


def test_profile_metadata_context_and_rng_are_preserved(monkeypatch, tmp_path):
    monkeypatch.setenv("BLOOMBEE_LORA_PROFILE_DIR", str(tmp_path))
    rng = torch.get_rng_state().clone()
    with profile_step(4, 2, warmup=True):
        metadata = request_metadata()
        with measure(metadata, "forward", "cpu", layer="test.0"):
            pass
    assert torch.equal(rng, torch.get_rng_state())
    assert "step" not in request_metadata()["profile"]
    event = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert event["step"] == 4 and event["microbatch"] == 2 and event["warmup"]
    assert event["seconds"] >= 0 and event["status"] == "ok"


def test_task_pool_profile_preserves_tensor_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("BLOOMBEE_LORA_PROFILE_DIR", str(tmp_path))
    pool = LoRATaskPool(lambda *a: None, max_batch_size=32, name="test_lora", device="cpu")
    try:
        x = torch.ones(1, 3, 4)
        meta = {**request_metadata(), "backward": False}
        # Exercise the runtime boundary directly. MPFuture's origin and its
        # serialized runtime copy must live in different processes; real RPC
        # integration tests cover that transport separately.
        future = Future()
        future._uid = 1
        pool._ordered_tasks.put(Task(2.0, time.monotonic(), future, (x, torch.empty(0), meta)))
        pool.batch_sender.send(None)
        uid, inputs = pool.load_batch_to_runtime(timeout=5)
        pool.send_outputs_from_runtime(uid, (inputs[0] * 2,))
        result = future.result(timeout=5)
        assert len(result) == 1 and isinstance(result[0], torch.Tensor)
        torch.testing.assert_close(result[0], x * 2)
        event = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
        assert event["input_tensor_bytes"] == 48 and event["output_tensor_bytes"] == 48
        assert event["h2d_tensor_bytes"] == event["d2h_tensor_bytes"] == 0
        assert not pool._profiles
    finally:
        pool.batch_sender.close()
        pool.batch_receiver.close()
        pool.submitted_tasks.close()


def test_profile_run_measures_first_optimizer_allocation(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "profile_lora_training", Path(__file__).parents[1] / "benchmarks/profile_lora_training.py"
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(loss=((input_ids.float() * self.weight) ** 2).mean())

    model = Toy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    report = benchmark.run_profile(model, [torch.ones(1, 3, dtype=torch.long)] * 2, optimizer, warmup=1, steps=2)
    assert len(report["samples"]) == 3 and report["samples"][0]["warmup"]
    assert report["effective_batch_size"] == 2
    assert report["steady_median_step_s"] > 0 and report["effective_tokens_per_second"] > 0
    assert optimizer.state


def test_stage_profile_cost_uses_matched_steady_pairs_and_physical_ranks():
    import importlib.util
    from pathlib import Path

    from bloombee.utils.lora import LoRAConfig

    spec = importlib.util.spec_from_file_location(
        "profile_lora_training", Path(__file__).parents[1] / "benchmarks/profile_lora_training.py"
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    events = [
        dict(
            kind="rpc",
            id=str(i),
            phase=phase,
            warmup=i == 0,
            peer="worker-a",
            start=0,
            end=2,
            total_s=100 if i == 0 else i,
        )
        for i in range(3)
        for phase in ("forward", "backward")
    ]
    config = LoRAConfig(rank_pattern={"layers.1.self_attn.q_proj": 2})
    result = benchmark.summarize_stage_profiles(events, config)
    assert len(result) == 1 and result[0]["samples"] == 2
    assert result[0]["median_forward_backward_rpc_s"] == 3
    assert result[0]["rank_by_layer"] == [{"q_proj": 8, "v_proj": 8}, {"q_proj": 2, "v_proj": 8}]
    with pytest.raises(ValueError, match="Incomplete"):
        benchmark.summarize_stage_profiles(events[:-1], config)
