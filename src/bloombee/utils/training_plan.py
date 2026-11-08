"""Static planning for the existing serial, client-relayed LoRA execution path.

Costs are measured forward+backward RPC seconds per microbatch, including the
client/worker link. They are NOT pipeline stage times or full optimizer steps.
Candidates must come from the same workload and execution mode. Resource bytes
are conservative peak reservations (including cold allocations and headroom),
not just parameter sizes. Use the SAME resource key for workers sharing a GPU
or host, and include the client in reservations/baseline_resources.

Utility is an externally calibrated additive rank proxy, not a quality guarantee.
This bounded exact search optimizes only over the supplied candidate catalogue;
it fails explicitly if its search budget is exhausted. No unmeasured extrapolation
or online rank/optimizer-state migration is performed.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bloombee.utils.lora import LORA_EXECUTION_MODES, LoRAConfig, projection_key, projection_shapes


def _integer(value, minimum=0):
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


@dataclass(frozen=True)
class TrainingWorkload:
    model: str
    num_layers: int
    batch_size: int
    sequence_length: int
    parameters_per_rank: int
    input_dtype: str = "torch.float16"
    accumulation: int = 1

    def __post_init__(self):
        if (
            not self.model
            or any(
                not _integer(v, 1)
                for v in (self.num_layers, self.batch_size, self.parameters_per_rank, self.accumulation)
            )
            or not _integer(self.sequence_length, 2)
        ):
            raise ValueError("Invalid training workload")
        if self.input_dtype not in ("torch.float16", "torch.bfloat16", "torch.float32"):
            raise ValueError("Unsupported training input dtype")

    def validate_model(self, config, lora):
        shapes = projection_shapes(config)
        if (
            config._name_or_path != self.model
            or config.num_hidden_layers != self.num_layers
            or sum(sum(shapes[t]) for t in lora.target_modules) != self.parameters_per_rank
        ):
            raise ValueError("Training plan model or projection dimensions do not match")


@dataclass(frozen=True)
class StageChoice:
    worker: str
    start: int
    end: int
    ranks: tuple
    weight_gpu_percent: int
    rpc_seconds: float
    resource_bytes: dict
    utility: float
    profile_id: str
    execution_mode: str = "block"

    def __post_init__(self):
        object.__setattr__(self, "ranks", tuple(self.ranks))
        if self.execution_mode not in LORA_EXECUTION_MODES:
            raise ValueError("Unsupported LoRA execution mode")
        if not self.worker or not self.profile_id or not _integer(self.start) or not _integer(self.end, self.start + 1):
            raise ValueError("Invalid worker, profile or layer span")
        if len(self.ranks) != self.end - self.start or any(not _integer(r, 1) or r > 256 for r in self.ranks):
            raise ValueError("Each layer in a candidate needs a rank in [1, 256]")
        if not _integer(self.weight_gpu_percent) or self.weight_gpu_percent > 100:
            raise ValueError("Weight GPU percent must be an integer in [0, 100]")
        if not math.isfinite(self.rpc_seconds) or self.rpc_seconds <= 0:
            raise ValueError("RPC cost must be finite and positive")
        if not math.isfinite(self.utility) or self.utility < 0:
            raise ValueError("Rank utility must be finite and nonnegative")
        if not self.resource_bytes or any(not k or not _integer(v) for k, v in self.resource_bytes.items()):
            raise ValueError("Candidates require resource peak reservations")
        object.__setattr__(self, "resource_bytes", dict(self.resource_bytes))

    def worker_args(self):
        """Deployment hints; the caller supplies model, DHT and device arguments."""
        return ["--block_indices", f"{self.start}:{self.end}", "--w_gpu_percent", str(self.weight_gpu_percent)]


@dataclass(frozen=True)
class TrainingPlan:
    workload: TrainingWorkload
    base_lora: LoRAConfig
    stages: tuple
    resource_budgets: dict
    minimum_utility: float
    max_parameters: int
    baseline_resources: dict = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "stages", tuple(self.stages))
        if self.schema_version != 1:
            raise ValueError("Unsupported training plan schema")
        if self.base_lora.rank_pattern or self.base_lora.alpha_pattern:
            raise ValueError("Planner base config must be uniform; stages define per-layer ranks")
        if not math.isfinite(self.minimum_utility) or self.minimum_utility < 0 or not _integer(self.max_parameters, 1):
            raise ValueError("Invalid quality floor or parameter budget")
        if not self.resource_budgets or any(not k or not _integer(v, 1) for k, v in self.resource_budgets.items()):
            raise ValueError("Resource budgets must be positive byte counts")
        if any(not _integer(v) for v in self.baseline_resources.values()):
            raise ValueError("Baseline resource reservations must be nonnegative")
        end, workers = 0, set()
        for stage in self.stages:
            if stage.start != end or stage.worker in workers:
                raise ValueError("Plan must cover ordered layers once and use each worker at most once")
            end = stage.end
            workers.add(stage.worker)
        if end != self.workload.num_layers:
            raise ValueError("Plan does not cover every model layer")
        if self.utility < self.minimum_utility or self.parameters > self.max_parameters:
            raise ValueError("Plan violates utility floor or adapter parameter budget")
        if any(k not in self.resource_budgets or v > self.resource_budgets[k] for k, v in self.resources.items()):
            raise ValueError("Plan exceeds a shared resource budget")

    @property
    def lora_config(self):
        ranks, alphas = {}, {}
        for stage in self.stages:
            for layer, rank in enumerate(stage.ranks, stage.start):
                for target in self.base_lora.target_modules:
                    key = projection_key(layer, target)
                    ranks[key] = rank
                    # Keep alpha/r constant when comparing rank allocations.
                    alphas[key] = self.base_lora.lora_alpha * rank / self.base_lora.r
        return LoRAConfig(**self.base_lora.to_dict(), rank_pattern=ranks, alpha_pattern=alphas)

    @property
    def parameters(self):
        return sum(sum(s.ranks) for s in self.stages) * self.workload.parameters_per_rank

    @property
    def utility(self):
        return sum(s.utility for s in self.stages)

    @property
    def estimated_rpc_seconds_per_step(self):
        return sum(s.rpc_seconds for s in self.stages) * self.workload.accumulation

    @property
    def resources(self):
        result = dict(self.baseline_resources)
        for stage in self.stages:
            for key, value in stage.resource_bytes.items():
                result[key] = result.get(key, 0) + value
        return result

    def to_dict(self):
        result = asdict(self)
        result["base_lora"] = self.base_lora.to_dict()
        # Preserve M1 plan/checkpoint serialization for the default executor.
        for stage in result["stages"]:
            if stage["execution_mode"] == "block":
                del stage["execution_mode"]
        return result

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["workload"] = TrainingWorkload(**data["workload"])
        data["base_lora"] = LoRAConfig(**data["base_lora"])
        data["stages"] = tuple(StageChoice(**s) for s in data["stages"])
        return cls(**data)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path):
        with Path(path).open("x") as stream:
            stream.write(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")


def solve_training_plan(
    *,
    workload,
    base_lora,
    candidates,
    resource_budgets,
    minimum_utility,
    max_parameters,
    baseline_resources=None,
    max_states=100000,
):
    """Exact branch-and-bound on contiguous spans; raises rather than silently pruning.

    Worker order is chosen by the search. Network cost is already in each measured
    client-relayed RPC, so no fictitious worker-to-worker edge cost is added.
    """
    baseline_resources = dict(baseline_resources or {})
    if not _integer(max_states, 1):
        raise ValueError("max_states must be positive")
    by_start = {}
    for candidate in candidates:
        if candidate.end > workload.num_layers:
            raise ValueError("Candidate exceeds model layer count")
        if any(key not in resource_budgets for key in candidate.resource_bytes):
            raise ValueError("Missing resource budget for candidate")
        by_start.setdefault(candidate.start, []).append(candidate)
    if any(k not in resource_budgets or v > resource_budgets[k] for k, v in baseline_resources.items()):
        raise ValueError("Baseline exceeds resource budget")
    for choices in by_start.values():
        choices.sort(key=lambda s: (s.rpc_seconds, s.worker, s.end, s.ranks))
    best, best_cost, visited = None, float("inf"), 0

    def visit(end, stages, workers, resources, parameters, utility, cost):
        nonlocal best, best_cost, visited
        visited += 1
        if visited > max_states:
            raise RuntimeError("Planner search budget exhausted; reduce candidates or increase max_states")
        if cost >= best_cost:
            return
        if end == workload.num_layers:
            if utility >= minimum_utility:
                best, best_cost = tuple(stages), cost
            return
        for stage in by_start.get(end, ()):
            count = parameters + sum(stage.ranks) * workload.parameters_per_rank
            if stage.worker in workers or count > max_parameters:
                continue
            memory = dict(resources)
            for key, value in stage.resource_bytes.items():
                memory[key] = memory.get(key, 0) + value
            if any(value > resource_budgets[key] for key, value in memory.items()):
                continue
            visit(
                stage.end,
                (*stages, stage),
                workers | {stage.worker},
                memory,
                count,
                utility + stage.utility,
                cost + stage.rpc_seconds,
            )

    visit(0, (), set(), baseline_resources, 0, 0.0, 0.0)
    if best is None:
        raise ValueError("No feasible joint training plan in the supplied candidate catalogue")
    return TrainingPlan(
        workload, base_lora, best, resource_budgets, minimum_utility, max_parameters, baseline_resources
    )
