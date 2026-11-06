"""Standard alpha/r LoRA, shared by client parameters and worker computations.

Adapters are request-local tensors. No worker parameter or optimizer is mutated.
The context also reaches FlexGen's *actual* functional projection operators.
"""

import math
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

LORA_PROTOCOL_VERSION = 1
LORA_NAMESPACE = "lora-v1"
LORA_NAMESPACES = {1: LORA_NAMESPACE, 2: "lora-v2"}
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_ACTIVE_LORA = ContextVar("bloombee_request_lora", default=None)


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    target_modules: tuple = ("q_proj", "v_proj")
    rank_pattern: dict = field(default_factory=dict)
    alpha_pattern: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        if isinstance(self.r, bool) or not isinstance(self.r, int) or not 1 <= self.r <= 256:
            raise ValueError("LoRA rank must be an integer in [1, 256]")
        if not math.isfinite(self.lora_alpha) or self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be finite and positive")
        if not math.isfinite(self.lora_dropout) or not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        targets = self.target_modules
        if not targets or len(set(targets)) != len(targets) or any(t not in PROJECTIONS for t in targets):
            raise ValueError(f"Choose unique LoRA targets from {PROJECTIONS}")
        for name in ("rank_pattern", "alpha_pattern"):
            pattern = {}
            for key, value in getattr(self, name).items():
                # Exact PEFT suffixes only: regex/overlapping patterns can silently
                # assign a different rank depending on dictionary insertion order.
                match = re.fullmatch(r"layers\.(0|[1-9][0-9]*)\.(self_attn|mlp)\.([a-z_]+)", key)
                if not match or match[3] not in targets or key != projection_key(int(match[1]), match[3]):
                    raise ValueError("LoRA patterns require exact layers.<index>.<group>.<target> keys")
                if name == "rank_pattern":
                    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 256:
                        raise ValueError("Pattern rank must be an integer in [1, 256]")
                elif isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                    raise ValueError("Pattern alpha must be finite and positive")
                pattern[key] = value
            object.__setattr__(self, name, pattern)

    def validate_layers(self, num_layers):
        if any(int(key.split(".")[1]) >= num_layers for p in (self.rank_pattern, self.alpha_pattern) for key in p):
            raise ValueError("LoRA pattern refers to a layer outside this model")

    def rank_for(self, layer, target):
        return self.rank_pattern.get(projection_key(layer, target), self.r)

    def alpha_for(self, layer, target):
        return self.alpha_pattern.get(projection_key(layer, target), self.lora_alpha)

    def to_dict(self):
        result = asdict(self)
        # Preserve the original v1 wire and training checkpoint configuration.
        for key in ("rank_pattern", "alpha_pattern"):
            if not result[key]:
                del result[key]
        return result


def projection_key(layer, target):
    group = "self_attn" if target in PROJECTIONS[:4] else "mlp"
    return f"layers.{layer}.{group}.{target}"


def projection_shapes(config):
    if config.model_type not in ("llama", "qwen3"):
        raise ValueError("Distributed LoRA currently supports LLaMA and Qwen3 dense blocks")
    h, i = config.hidden_size, config.intermediate_size
    d = getattr(config, "head_dim", None) or h // config.num_attention_heads
    q = config.num_attention_heads * d
    kv = getattr(config, "num_key_value_heads", config.num_attention_heads) * d
    return {
        "q_proj": (q, h),
        "k_proj": (kv, h),
        "v_proj": (kv, h),
        "o_proj": (h, q),
        "gate_proj": (i, h),
        "up_proj": (i, h),
        "down_proj": (h, i),
    }


class LoRAProjection(nn.Module):
    def __init__(self, in_features, out_features, rank, *, device=None):
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _apply(self, fn, recurse=True):
        # Preserve values as well as dtype: converting to half and back to float
        # would irreversibly round the optimizer's master weights and gradients.
        def keep_fp32(tensor):
            converted = fn(tensor)
            return (
                converted
                if converted.dtype == torch.float32
                else tensor.to(device=converted.device, dtype=torch.float32)
            )

        return super()._apply(keep_fp32, recurse=recurse)


@contextmanager
def lora_context(config, tensors, *, training, seed, layer_index=0):
    if len(tensors) != 2 * len(config.target_modules):
        raise ValueError("LoRA A/B tensor count does not match target_modules")
    pairs = {name: tensors[2 * j : 2 * j + 2] for j, name in enumerate(config.target_modules)}
    token = _ACTIVE_LORA.set((config, pairs, bool(training), int(seed), layer_index))
    try:
        yield
    finally:
        _ACTIVE_LORA.reset(token)


def lora_delta(inputs, projection):
    state = _ACTIVE_LORA.get()
    if state is None or projection not in state[1]:
        return None
    config, pairs, training, seed, layer = state
    a, b = pairs[projection]
    hidden = inputs.to(dtype=a.dtype)
    if training and config.lora_dropout:
        # A request-local RNG: exact recompute without modifying global CPU/CUDA RNG.
        generator = torch.Generator(device=hidden.device)
        generator.manual_seed((seed + PROJECTIONS.index(projection) * 104729) % (2**63 - 1))
        keep = torch.rand(hidden.shape, device=hidden.device, generator=generator) >= config.lora_dropout
        hidden = hidden * keep / (1.0 - config.lora_dropout)
    return F.linear(F.linear(hidden, a), b) * (config.alpha_for(layer, projection) / config.rank_for(layer, projection))


def lora_linear(inputs, weight, projection):
    output = F.linear(inputs, weight)
    delta = lora_delta(inputs, projection)
    # Add in the adapter's master precision, then round once to the base dtype,
    # matching PEFT's mixed-dtype in-place addition. Rounding delta first changes
    # nonzero adapters in FP16/BF16, particularly near cancellation.
    return output if delta is None else (output + delta).to(output.dtype)


def install_lora_hooks(module):
    """Install once on HF projections. An empty context leaves base inference unchanged."""
    for name, child in module.named_modules():
        target = name.rsplit(".", 1)[-1]
        if target in PROJECTIONS and isinstance(child, nn.Linear) and not getattr(child, "_bb_lora_hook", False):

            def add_delta(linear, args, output, projection=target):
                delta = lora_delta(args[0], projection)
                return output if delta is None else (output + delta).to(output.dtype)

            child.register_forward_hook(add_delta)
            child._bb_lora_hook = True
