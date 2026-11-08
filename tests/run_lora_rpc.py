"""Real private DHT + worker processes + full causal-LM LoRA training on CUDA.

Run from the checkout with PYTHONPATH=src. Creates a tiny deterministic random
checkpoint, never downloads weights, checks against independent HuggingFace PEFT.
This is an integration test, not a pretrained-model quality/performance benchmark.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from bloombee import AutoDistributedModelForCausalLM, DistributedLoRA, LoRAConfig as BBLoRAConfig
from bloombee.data_structures import ServerState
from bloombee.utils.dht import get_remote_module_infos
from bloombee.utils.hivemind_compat import DHT
from bloombee.utils.lora import LORA_EXECUTION_MODES, PROJECTIONS, projection_shapes
from bloombee.utils.training_plan import StageChoice, TrainingPlan, TrainingWorkload


def save_tiny_tokenizer(base_dir):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(
        WordLevel({"<unk>": 0, "<bos>": 1, "<eos>": 2, **{f"w{i}": i for i in range(3, 64)}}, unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", bos_token="<bos>", eos_token="<eos>"
    ).save_pretrained(base_dir)


def check_cli_resume(base_dir, peers, prefix, output, plan=None, worker_peers=None):
    from safetensors.torch import load_file

    data = output / "train.jsonl"
    data.write_text(
        "\n".join(json.dumps({"text": " ".join(f"w{i}" for i in range(start, start + 16))}) for start in (3, 19, 35))
        + "\n"
    )
    command = [
        sys.executable,
        "examples/train_lora.py",
        "--model",
        str(base_dir.resolve()),
        "--initial_peers",
        *peers,
        "--dht_prefix",
        prefix,
        "--data",
        str(data),
        "--seq_len",
        "8",
        "--batch_size",
        "1",
        "--gradient_accumulation_steps",
        "2",
        "--r",
        "4",
        "--lora_alpha",
        "8",
        "--lora_dropout",
        ".3",
        "--lr",
        ".01",
    ]
    if plan:
        plan = replace(
            plan,
            workload=replace(plan.workload, batch_size=1, accumulation=2),
            base_lora=replace(plan.base_lora, lora_dropout=0.3),
        )
        plan_path, binding_path = output / "resume-plan.json", output / "worker-peers.json"
        plan.save(plan_path)
        binding_path.write_text(json.dumps(worker_peers))
        command += ["--plan", str(plan_path), "--worker_peers", str(binding_path)]
    for name, steps, resume in (
        ("continuous", 4, None),
        ("interrupted", 2, None),
        ("resumed", 4, output / "interrupted" / "checkpoint-2"),
    ):
        cmd = command + ["--output", str(output / name), "--steps", str(steps), "--checkpoint_every", "1"]
        if resume:
            cmd += ["--resume_checkpoint", str(resume), "--seed", "999"]
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    expected = load_file(str(output / "continuous" / "adapter_model.safetensors"))
    actual = load_file(str(output / "resumed" / "adapter_model.safetensors"))
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    states = [
        torch.load(output / name / "checkpoint-4" / "training_state.pt", weights_only=True)
        for name in ("continuous", "resumed")
    ]

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b

    assert equal(states[0], states[1]), "Optimizer, loss history or RNG differs after resume"
    # Reject changed token data before making any remote training request.
    data.write_text(json.dumps({"text": " ".join(["w9"] * 40)}) + "\n")
    invalid = subprocess.run(
        command
        + [
            "--output",
            str(output / "invalid"),
            "--steps",
            "4",
            "--resume_checkpoint",
            str(output / "interrupted" / "checkpoint-2"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert invalid.returncode != 0 and "same model, tokenized data" in invalid.stderr
    return dict(
        adapter_exact=True,
        optimizer_rng_losses_exact=True,
        dropout=0.3,
        gradient_accumulation_steps=2,
        split_steps=[2, 2],
        changed_data_rejected=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("llama", "qwen3"), default="llama")
    parser.add_argument("--stages", type=int, choices=(1, 2), default=2)
    parser.add_argument("--weight-gpu-percent", type=int, default=100)
    parser.add_argument("--cache-gpu-percent", type=int, default=100)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--check-resume", action="store_true")
    parser.add_argument("--check-merged", action="store_true")
    parser.add_argument("--check-profile", action="store_true")
    parser.add_argument("--check-plan", action="store_true", help="Mixed ranks, fixed peers and offload enforcement")
    parser.add_argument(
        "--check-cache-recovery", action="store_true", help="Expire cached state between forward and backward"
    )
    parser.add_argument("--execution-mode", choices=LORA_EXECUTION_MODES, default="block")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.execution_mode != "block" and not args.check_plan:
        parser.error("Span tests require --check-plan")
    if args.check_cache_recovery and (
        not args.execution_mode.startswith("span_cached_") or os.getenv("BLOOMBEE_LORA_CACHE_TTL") != "1"
    ):
        parser.error("Cache recovery requires a cached mode and BLOOMBEE_LORA_CACHE_TTL=1")
    if not torch.cuda.is_available():
        raise RuntimeError("This integration test requires a CUDA GPU")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    base_dir = args.output / "base"
    torch.manual_seed(173)
    klass, config_class = (LlamaForCausalLM, LlamaConfig) if args.family == "llama" else (Qwen3ForCausalLM, Qwen3Config)
    config = config_class(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    # FP16 on disk: FlexGen's canonical conversion stores FP16 base weights.
    klass(config).half().save_pretrained(base_dir)
    save_tiny_tokenizer(base_dir)
    prefix = "lora_" + uuid.uuid4().hex[:12]
    dht = DHT(start=True, host_maddrs=["/ip4/127.0.0.1/tcp/0"], use_relay=False)
    peers = [str(addr) for addr in dht.get_visible_maddrs()]
    processes, logs, managers = [], [], []
    report = {
        "family": args.family,
        "stages": args.stages,
        "weight_gpu_percent": args.weight_gpu_percent,
        "cache_gpu_percent": args.cache_gpu_percent,
        "dtype": "float16 base / float32 LoRA",
        "device": torch.cuda.get_device_name(),
        "checkpoint": "deterministic random tiny model, 2 layers, hidden=32, GQA 4/2",
        "seed": 173,
        "status": "running",
    }

    def start_workers(model_dir, worker_prefix, *, pipeline=False):
        for stage in range(args.stages):
            start, end = (0, 2) if args.stages == 1 else (stage, stage + 1)
            log = (args.output / f"{'merged-' if pipeline else ''}worker-{stage}.log").open("w")
            logs.append(log)
            env = {
                **os.environ,
                "BLOOMBEE_ENABLE_LORA": "1",
                "BLOOMBEE_ENABLE_SPEC_PRUNER": "0",
                "BLOOMBEE_ENABLE_MICROBATCH_PIPELINE": "0",
                "BLOOMBEE_LOSSLESS_WRAPPER": "0",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "HF_HUB_OFFLINE": "1",
            }
            if pipeline:
                env.update(
                    BLOOMBEE_ENABLE_MICROBATCH_PIPELINE="1",
                    BLOOMBEE_MICRO_BATCH_SIZE="1",
                    BLOOMBEE_DEBUG_MICROBATCH="1",
                    BLOOMBEE_LOSSLESS_WRAPPER="1",
                    BLOOMBEE_LOSSLESS_MIN_BYTES="0",
                    BLOOMBEE_LOSSLESS_ALGO="zstd",
                    BLOOMBEE_ENABLE_LORA="0",
                )
            command = [
                sys.executable,
                "-m",
                "bloombee.cli.run_server",
                str(model_dir.resolve()),
                "--block_indices",
                f"{start}:{end}",
                "--dht_prefix",
                worker_prefix,
                "--num_handlers",
                "1",
                "--batch_size",
                "2",
                "--torch_dtype",
                "float16",
                "--throughput",
                "100",
                "--update_period",
                "1",
                "--expiration",
                "30",
                "--inference_max_length",
                "64",
                "--attn_cache_tokens",
                "128",
                "--max_batch_size",
                "512",
                "--balance_quality",
                "0",
                "--no_auto_relay",
                "--host_maddrs",
                "/ip4/127.0.0.1/tcp/0",
                "--skip_reachability_check",
                "--cache_gpu_percent",
                str(args.cache_gpu_percent),
                "--w_gpu_percent",
                str(args.weight_gpu_percent),
                "--initial_peers",
                *peers,
            ]
            processes.append(
                subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            )
        deadline = time.monotonic() + 180
        uids = [f"{worker_prefix}.{i}" for i in range(2)]
        while time.monotonic() < deadline:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("Worker exited; inspect worker-*.log")
            infos = get_remote_module_infos(dht, uids, latest=True)
            if all(any(info.state == ServerState.ONLINE for info in item.servers.values()) for item in infos):
                break
            time.sleep(0.5)
        else:
            raise TimeoutError("Private swarm startup timed out")
        print("Workers ONLINE", flush=True)

    try:
        start_workers(base_dir, prefix)

        def load_client(model_dir=base_dir, model_prefix=prefix):
            model = AutoDistributedModelForCausalLM.from_pretrained(
                str(model_dir.resolve()),
                dht_prefix=model_prefix,
                initial_peers=peers,
                torch_dtype=torch.float16,
                connect_timeout=30,
                request_timeout=60,
                max_retries=1,
                update_period=1,
            ).cuda()
            seq = next(m for m in model.modules() if m.__class__.__name__ == "RemoteSequential")
            managers.append(seq.sequence_manager)
            return model

        client = load_client()
        ids = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29], [1, 6, 10, 14, 18, 22, 26, 30]], device="cuda")
        with torch.no_grad():
            base_logits = client(ids).logits.clone()
        lora_config = BBLoRAConfig(r=4, lora_alpha=8, target_modules=PROJECTIONS)
        plan, worker_peers = None, None
        if args.check_plan:
            infos = get_remote_module_infos(dht, [f"{prefix}.{i}" for i in range(2)], latest=True)
            stages, worker_peers = [], {}
            for index in range(args.stages):
                start, end = (0, 2) if args.stages == 1 else (index, index + 1)
                worker = f"worker-{index}"
                peer = next(peer for peer, info in infos[start].servers.items() if info.state == ServerState.ONLINE)
                worker_peers[worker] = str(peer)
                # Synthetic reservations/costs exercise execution only, not performance prediction.
                stages.append(
                    StageChoice(
                        worker,
                        start,
                        end,
                        (2, 8)[start:end],
                        args.weight_gpu_percent,
                        1.0,
                        {"shared_gpu": 1},
                        end - start,
                        "synthetic-correctness-only",
                    )
                )
            plan = TrainingPlan(
                TrainingWorkload(
                    str(base_dir.resolve()), 2, 2, 8, sum(sum(s) for s in projection_shapes(config).values())
                ),
                lora_config,
                tuple(stages),
                {"shared_gpu": 2},
                2,
                100000,
            )
            plan = replace(plan, stages=tuple(replace(s, execution_mode=args.execution_mode) for s in plan.stages))
            plan.save(args.output / "plan.json")
            lora_config = plan.lora_config
        model = DistributedLoRA(client, lora_config).cuda()
        model.save_pretrained(args.output / "initial-adapter")
        reference = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float16, attn_implementation="eager").cuda(),
            args.output / "initial-adapter",
            is_trainable=True,
        )
        # PEFT 0.8 follows the base dtype on load. Compare the same FP32 adapter
        # master weights used by BloomBee, retaining the FP16 frozen base.
        for name, parameter in reference.named_parameters():
            if ".lora_" in name:
                parameter.data = parameter.data.float()
        model.train()
        reference.train()
        base_snapshot = {k: p.detach().clone() for k, p in reference.named_parameters() if not p.requires_grad}
        actual = model(ids, labels=ids)
        expected = reference(ids, labels=ids, use_cache=False)
        report["initial_logit_max_error"] = float((actual.logits - expected.logits).abs().max())
        report["initial_loss_error"] = abs(float(actual.loss) - float(expected.loss))
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0.02, atol=1e-3)
        torch.testing.assert_close(actual.loss, expected.loss, rtol=0.001, atol=1e-3)
        actual.loss.backward()
        expected.loss.backward()
        ref_params = dict(reference.named_parameters())
        grad_errors, grad_relative_norm_errors, grad_failures, grad_atols = {}, {}, [], {}
        for key, parameter in model._adapter_state().items():
            ref_key = key.replace(".lora_A.weight", ".lora_A.default.weight").replace(
                ".lora_B.weight", ".lora_B.default.weight"
            )
            target = ref_params[ref_key]
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            grad_errors[key] = float((parameter.grad - target.grad).abs().max())
            grad_relative_norm_errors[key] = float(
                (parameter.grad - target.grad).norm() / target.grad.norm().clamp_min(1e-12)
            )
            # Separate-block FP16 backward can round accumulation differently
            # from a full HF graph. Near cancellation, element-relative error
            # is unstable: bound it by one FP16 epsilon at this tensor's scale,
            # and additionally require <0.1% whole-tensor relative L2 error.
            grad_atols[key] = max(5e-6, torch.finfo(torch.float16).eps * float(target.grad.abs().max()))
            try:
                torch.testing.assert_close(parameter.grad, target.grad, rtol=0.05, atol=grad_atols[key])
                assert grad_relative_norm_errors[key] < 0.001, "Gradient relative L2 error exceeds 0.1%"
            except AssertionError as error:
                grad_failures.append(f"{key}: {error}")
        report["lora_gradient_max_error"] = max(grad_errors.values())
        report["gradient_errors"] = grad_errors
        report["gradient_relative_norm_errors"] = grad_relative_norm_errors
        assert not grad_failures, "\n".join(grad_failures)
        params = [p for p in model.parameters() if p.requires_grad]
        report["trainable_parameters"] = sum(p.numel() for p in params)
        report["route"] = [
            {"peer": str(s.peer_id), "start": s.start, "end": s.end}
            for s in model.remote_layers.sequence_manager.make_sequence(mode="max_throughput")
        ]
        assert len(report["route"]) == args.stages
        # Explicit optimizer update parity against PEFT, independently of the loss trend.
        opt = torch.optim.SGD(params, lr=0.3)
        ref_opt = torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr=0.3)
        opt.step()
        ref_opt.step()
        for key, p in model._adapter_state().items():
            ref_key = key.replace(".lora_A.weight", ".lora_A.default.weight").replace(
                ".lora_B.weight", ".lora_B.default.weight"
            )
            torch.testing.assert_close(p, ref_params[ref_key], rtol=0.01, atol=max(2e-6, 0.3 * grad_atols[key]))
        opt.zero_grad(set_to_none=True)
        ref_opt.zero_grad(set_to_none=True)

        # Compare full-batch gradients with two explicitly accumulated microbatches.
        full_loss = model(ids, labels=ids).loss
        full_loss.backward()
        full_grads = [p.grad.clone() for p in params]
        opt.zero_grad(set_to_none=True)
        for row in ids.split(1):
            (model(row, labels=row).loss / len(ids)).backward()
        report["accumulation_gradient_max_error"] = max(
            float((p.grad - g).abs().max()) for p, g in zip(params, full_grads)
        )
        for p, g in zip(params, full_grads):
            torch.testing.assert_close(p.grad, g, rtol=0.08, atol=1e-5)
        opt.zero_grad(set_to_none=True)
        optimizer = torch.optim.AdamW(params, lr=0.015, weight_decay=0.0)
        if plan:
            if args.execution_mode != "block":
                baseline = model(ids, labels=ids)
                baseline.loss.backward()
                baseline_grads = [p.grad.clone() for p in params]
                optimizer.zero_grad(set_to_none=True)
            model.remote_layers.set_training_plan(plan, worker_peers)
            if args.execution_mode != "block":
                planned = model(ids, labels=ids)
                planned.loss.backward()
                torch.testing.assert_close(planned.logits, baseline.logits, rtol=0, atol=0)
                for parameter, gradient in zip(params, baseline_grads):
                    torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
                optimizer.zero_grad(set_to_none=True)
                report["span_block_gradient_parity_exact"] = True
                if args.check_cache_recovery:
                    delayed = model(ids, labels=ids)
                    time.sleep(1.2)
                    delayed.loss.backward()
                    for parameter, gradient in zip(params, baseline_grads):
                        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
                    optimizer.zero_grad(set_to_none=True)
                    report["cache_expiry_recovery_exact"] = True
            # A planned job must never call the adaptive/random route selector.
            def unexpected_route(*args, **kwargs):
                raise AssertionError("Planned training unexpectedly selected a new route")

            model.remote_layers.sequence_manager.make_sequence = unexpected_route
            with torch.no_grad():
                try:
                    model(ids[:1])
                except ValueError as error:
                    assert "profiled training workload" in str(error)
                else:
                    raise AssertionError("Planned shape mismatch was not rejected")
        losses = []
        for step in range(args.steps):
            output = model(ids, labels=ids)
            losses.append(float(output.loss))
            output.loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step % 5 == 0:
                print(f"step={step} loss={losses[-1]:.6f}", flush=True)
        model.eval()
        with torch.no_grad():
            final = model(ids, labels=ids)
        report["losses"] = losses
        report["initial_loss"] = losses[0]
        report["final_loss"] = float(final.loss)
        assert report["final_loss"] < losses[0] - 0.02
        assert all(p.grad is None for p in model.model.parameters() if not p.requires_grad)
        model.save_pretrained(args.output / "adapter")
        torch.save(optimizer.state_dict(), args.output / "optimizer.pt")
        # Roundtrip through our loader and through an independent PEFT model.
        loaded = DistributedLoRA.from_pretrained(load_client(), args.output / "adapter").cuda().eval()
        with torch.no_grad():
            reloaded = loaded(ids).logits
        torch.testing.assert_close(final.logits, reloaded, rtol=0, atol=0)
        peft_loaded = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float16, attn_implementation="eager").cuda(),
            args.output / "adapter",
        ).eval()
        for name, parameter in peft_loaded.named_parameters():
            if ".lora_" in name:
                parameter.data = parameter.data.float()
        with torch.no_grad():
            peft_logits = peft_loaded(ids).logits
        report["peft_reload_logit_max_error"] = float((final.logits - peft_logits).abs().max())
        torch.testing.assert_close(final.logits, peft_logits, rtol=0.04, atol=0.003)
        # The same workers must still serve the unchanged base to a fresh ordinary client.
        ordinary = load_client().eval()
        with torch.no_grad():
            ordinary_logits = ordinary(ids).logits
        torch.testing.assert_close(base_logits, ordinary_logits, rtol=0, atol=0)
        report["base_output_unchanged"] = True
        report["adapter_roundtrip_exact"] = True
        report["generated_ids"] = loaded.generate(ids[:1, :3], max_new_tokens=3).tolist()
        for k, p in reference.named_parameters():
            if not p.requires_grad:
                assert torch.equal(p, base_snapshot[k])
        if args.check_resume:
            report["cli_resume"] = check_cli_resume(base_dir, peers, prefix, args.output, plan, worker_peers)
        if plan:
            report["joint_plan"] = dict(
                execution_mode=args.execution_mode,
                ranks=[2, 8],
                fixed_route=True,
                shape_mismatch_rejected=True,
                performance_claim=False,
                worker_peers=worker_peers,
            )
            bad = replace(
                plan,
                stages=tuple(replace(s, weight_gpu_percent=(args.weight_gpu_percent + 50) % 101) for s in plan.stages),
            )
            model.remote_layers.set_training_plan(bad, worker_peers)
            try:
                model(ids)
            except Exception as error:
                assert "offload policy" in str(error), str(error)
            else:
                raise AssertionError("Worker accepted a mismatched offload plan")
            model.remote_layers.set_training_plan(plan, worker_peers)
            report["joint_plan"]["offload_mismatch_rejected"] = True
        if args.check_profile:
            profile_path = args.output / "profile.json"
            profile_args = []
            if plan:
                profile_plan = replace(plan, workload=replace(plan.workload, batch_size=1, accumulation=2))
                profile_plan.save(args.output / "profile-plan.json")
                (args.output / "profile-peers.json").write_text(json.dumps(worker_peers))
                profile_args = [
                    "--plan",
                    str(args.output / "profile-plan.json"),
                    "--worker-peers",
                    str(args.output / "profile-peers.json"),
                ]
            with (args.output / "profile-cli.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "benchmarks/profile_lora_training.py",
                        "--model",
                        str(base_dir.resolve()),
                        "--initial-peers",
                        *peers,
                        "--dht-prefix",
                        prefix,
                        "--output",
                        str(profile_path),
                        "--sequence-length",
                        "8",
                        "--accumulation",
                        "2",
                        "--warmup",
                        "1",
                        "--steps",
                        "2",
                        *profile_args,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=180,
                )
            profile = json.loads(profile_path.read_text())
            assert len(profile["samples"]) == 3 and profile["effective_batch_size"] == 2
            assert profile["samples"][0]["cuda_peak_allocated_since_reset_bytes"] > 0
            assert profile["stage_profiles"] and all(s["samples"] == 4 for s in profile["stage_profiles"])
            report["profiling_cli"] = {
                key: profile[key]
                for key in ("steady_median_step_s", "effective_tokens_per_second", "trainable_parameters")
            }
        if args.check_merged:
            merged_dir = args.output / "merged-model"
            with (args.output / "merge.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "examples/merge_lora.py",
                        "--model",
                        str(base_dir),
                        "--adapter",
                        str(args.output / "adapter"),
                        "--output",
                        str(merged_dir),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=120,
                )
            merged_prefix = prefix + "_merged"
            start_workers(merged_dir, merged_prefix, pipeline=True)
            merged = load_client(merged_dir, merged_prefix).eval()
            merged_reference = (
                AutoModelForCausalLM.from_pretrained(merged_dir, dtype=torch.float16, attn_implementation="eager")
                .cuda()
                .eval()
            )
            cache_errors, cache_positions = [], []
            # Ordinary BloomBee KV session: prefill 5 tokens, then 3 decode steps.
            with torch.inference_mode(), merged.inference_session(max_length=16) as session:
                for start, end in ((0, 5), (5, 6), (6, 7), (7, 8)):
                    actual = merged(ids[:, start:end]).logits
                    expected = merged_reference(ids[:, :end], use_cache=False).logits[:, start:end]
                    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.003)
                    cache_errors.append(float((actual - expected).abs().max()))
                    cache_positions.append(session.position)
            assert cache_positions == [5, 6, 7, 8]
            with torch.inference_mode():
                merged_logits = merged_reference(ids, use_cache=False).logits
                generated = merged.generate(ids[:, :3], max_new_tokens=3, do_sample=False)
                expected_tokens = merged_reference.generate(ids[:, :3], max_new_tokens=3, do_sample=False)
            assert torch.equal(generated, expected_tokens)
            torch.testing.assert_close(merged_logits, final.logits, rtol=0.06, atol=0.004)
            worker_lines = [
                line
                for stage in range(args.stages)
                for line in (args.output / f"merged-worker-{stage}.log").read_text().splitlines()
            ]
            s2s_count = sum("[S2S_WIRE]" in line for line in worker_lines)
            microbatch_count = sum("micro-batch metadata" in line for line in worker_lines)
            if args.stages == 2:
                assert s2s_count >= 6 and microbatch_count >= 6
            report["merged_kv"] = dict(
                max_logit_errors=cache_errors,
                cache_positions=cache_positions,
                adapter_merge_logit_error=float((merged_logits - final.logits).abs().max()),
                batch_size=2,
                microbatch_size=1,
                lossless_wrapper=True,
                s2s_log_samples=s2s_count,
                microbatch_log_samples=microbatch_count,
                greedy_generation_exact=True,
                generated_ids=generated.tolist(),
                separate_dht_prefix=True,
                lora_rpc_enabled=False,
            )
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        for manager in managers:
            manager.shutdown()
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in logs:
            log.close()
        dht.shutdown()
    print(json.dumps({k: v for k, v in report.items() if k not in ("losses", "gradient_errors")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
