"""Solve a measured joint-planning catalogue and write an executable static plan.

Input JSON keys: workload (TrainingWorkload), base_lora (LoRAConfig), candidates
(StageChoice list), resource_budgets, minimum_utility, max_parameters; optional
baseline_resources and max_states. See these dataclasses for the strict schema.
Every candidate must describe the SAME model, precision, targets, dropout,
microbatch shape and serial execution mode. Utility is a supplied quality proxy.
Use each stage profile's rpc_seconds field (derived from complete optimizer
steps), not median_forward_backward_rpc_s. Checkpoint modes need workload.hidden_size.
Candidate checkpoint_limit_bytes and activation_cache_limit_bytes default to
the worker defaults. Explicitly set the activation limit to 0 to plan a
cached adapter mode with activation storage disabled.
Set workload.model_revision and workload.model_type from the profile report.
Cached candidates require the measured adapter_cache_limit_bytes and cache_ttl_seconds.

Launch workers using the printed arguments and environment, then pass --plan and a
--worker_peers JSON mapping to examples/train_lora.py. Worker startup is explicit;
this command does not kill or reconfigure an existing swarm.
"""

import argparse
import json
from pathlib import Path

from bloombee.utils.lora import LORA_CACHE_ENV, LoRAConfig
from bloombee.utils.training_plan import StageChoice, TrainingWorkload, solve_training_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.catalogue.read_text())
    data["workload"] = TrainingWorkload(**data["workload"])
    data["base_lora"] = LoRAConfig(**data["base_lora"])
    data["candidates"] = [StageChoice(**s) for s in data["candidates"]]
    plan = solve_training_plan(**data)
    plan.save(args.output)
    revision = plan.workload.model_revision
    revision_args = ["--revision", revision[3:]] if revision and revision.startswith("hf:") else []
    print(
        json.dumps(
            dict(
                estimated_rpc_seconds_per_step=plan.estimated_rpc_seconds_per_step,
                parameters=plan.parameters,
                utility=plan.utility,
                resources=plan.resources,
                worker_args={
                    s.worker: s.worker_args() + ["--torch_dtype", plan.workload.dtype_name] + revision_args
                    for s in plan.stages
                },
                worker_env={
                    s.worker: {
                        name: str(limit)
                        for name, limit in (
                            ("BLOOMBEE_LORA_SPAN_CHECKPOINT_BYTES", s.checkpoint_limit_bytes),
                            *((env, getattr(s, key)) for key, (env, _) in LORA_CACHE_ENV.items()),
                        )
                        if limit is not None
                    }
                    for s in plan.stages
                },
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
