"""Solve a measured joint-planning catalogue and write an executable static plan.

Input JSON keys: workload (TrainingWorkload), base_lora (LoRAConfig), candidates
(StageChoice list), resource_budgets, minimum_utility, max_parameters; optional
baseline_resources and max_states. See these dataclasses for the strict schema.
Every candidate must describe the SAME model, precision, targets, dropout,
microbatch shape and serial execution mode. Utility is a supplied quality proxy.

Launch workers using printed deployment hints, then pass --plan and a
--worker_peers JSON mapping to examples/train_lora.py. Worker startup is explicit;
this command does not kill or reconfigure an existing swarm.
"""

import argparse
import json
from pathlib import Path

from bloombee.utils.lora import LoRAConfig
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
    print(
        json.dumps(
            dict(
                estimated_rpc_seconds_per_step=plan.estimated_rpc_seconds_per_step,
                parameters=plan.parameters,
                utility=plan.utility,
                resources=plan.resources,
                worker_args={s.worker: s.worker_args() for s in plan.stages},
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
