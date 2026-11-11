"""Serial span tasks: GPU layer boundaries, per-block graphs and bounded checkpoints.

Protocol v3 uses request-local state; v4 delegates retention to bounded caches.
CPU checkpoints use blocking copies,
so no pinned-buffer lifetime or asynchronous reuse is assumed. The worker's
BLOOMBEE_LORA_SPAN_CHECKPOINT_BYTES limit (default 256 MiB) bounds additional
detached layer boundaries, not total model/working/gradient memory. A planner
must also reserve the measured whole-process peak. span_recompute stores no
additional boundaries and trades memory for repeated prefix computation.
"""

import os
import time

import torch

from bloombee.server.lora_pool import LoRATaskPool
from bloombee.utils.lora import DEFAULT_LORA_CHECKPOINT_BYTES, LORA_EXECUTION_MODES, LoRAConfig
from bloombee.utils.lora_profile import profile_fields, synchronize, tensor_bytes, write_event


def move_tensor(tensor, device, metadata, purpose):
    device = torch.device(device)
    if tensor.device == device:
        return tensor
    fields = profile_fields(metadata)
    cuda_device = device if device.type == "cuda" else tensor.device
    if fields:
        synchronize(cuda_device)
    started = time.perf_counter()
    result = tensor.detach().to(device, non_blocking=False)
    if fields:
        synchronize(cuda_device)
        write_event(
            "span_transfer",
            **fields,
            purpose=purpose,
            seconds=time.perf_counter() - started,
            source=str(tensor.device),
            destination=str(device),
            tensor_bytes=tensor_bytes((tensor,))
        )
    return result


class LoRASpanExecutor:
    def __init__(self, backends, *, checkpoint_bytes):
        if isinstance(checkpoint_bytes, bool) or not isinstance(checkpoint_bytes, int) or checkpoint_bytes < 0:
            raise ValueError("Span checkpoint budget must be a nonnegative byte count")
        self.backends, self.checkpoint_bytes = dict(backends), checkpoint_bytes
        devices = {torch.device(b.lora_pool.device) for b in backends.values()}
        if len(devices) != 1:
            raise ValueError("A LoRA span executor requires a single compute device")
        self.device = devices.pop()

    def clear_cache(self):
        """Release retained tensors after the Runtime has stopped processing requests."""
        if hasattr(self, "cache"):
            self.cache.adapters.entries.clear()
            self.cache.adapters.bytes = 0
            self.cache.activations.entries.clear()
            self.cache.activations.bytes = 0
            del self.cache

    def __call__(self, inputs, grad_outputs, metadata, *parameters):
        if metadata.get("version") == 4:
            from bloombee.server.lora_cache import CachedLoRASpan

            if not hasattr(self, "cache"):
                self.cache = CachedLoRASpan(self)
            return self.cache(inputs, grad_outputs, metadata, parameters)
        return self.run(inputs, grad_outputs, metadata, parameters)

    def run(self, inputs, grad_outputs, metadata, parameters, *, checkpoints=None, save_forward=False):
        from bloombee.server.lora import run_lora_block

        names = metadata["span_uids"]
        backends = [self.backends[name] for name in names]
        indices = [int(name.rsplit(".", 1)[-1]) for name in names]
        if not indices or indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError("Span layers must be contiguous")
        mode = metadata["execution_mode"]
        if mode not in LORA_EXECUTION_MODES[1:4]:
            raise ValueError("Unsupported span execution mode")
        width = 2 * len(LoRAConfig(**metadata["config"]).target_modules)
        if len(parameters) != width * len(names):
            raise ValueError("Span parameter count mismatch")
        backward = metadata["backward"]
        required = (
            (len(names) - 1) * tensor_bytes((inputs,)) if (backward or save_forward) and mode != "span_recompute" else 0
        )
        if required > self.checkpoint_bytes:
            raise RuntimeError("Span activation checkpoint budget exceeded; reprofile/replan or use span_recompute")
        empty = torch.empty(0, device=self.device)

        def block(j, x, grad):
            params = [move_tensor(p, self.device, metadata, "adapter") for p in parameters[j * width : (j + 1) * width]]
            return run_lora_block(
                backends[j], x, grad, {**metadata, "seed": metadata["seed"] + j, "layer_index": indices[j]}, *params
            )

        def prefix(end, checkpoints=None):
            first = checkpoints[0] if checkpoints is not None and checkpoints[0] is not None else inputs
            x = move_tensor(first, self.device, metadata, "input")
            for j in range(end):
                x = block(j, x, empty)[0]
                if checkpoints is not None and j + 1 < len(names):
                    device = "cpu" if mode == "span_cpu" else self.device
                    checkpoints[j + 1] = move_tensor(x, device, metadata, "checkpoint_save")
            return x

        if not backward:
            return (prefix(len(names), checkpoints if save_forward else None),)
        if checkpoints is None and mode != "span_recompute":
            checkpoints = [inputs] + [None] * (len(names) - 1)
            prefix(len(names) - 1, checkpoints)
        grad = move_tensor(grad_outputs, self.device, metadata, "output_gradient")
        parameter_grads = [None] * len(names)
        for j in reversed(range(len(names))):
            if checkpoints is None:
                x = prefix(j)
            else:
                x = move_tensor(checkpoints[j], self.device, metadata, "checkpoint_load")
                checkpoints[j] = None
            result = block(j, x, grad)
            grad = result[0]
            parameter_grads[j] = tuple(move_tensor(g, "cpu", metadata, "adapter_gradient") for g in result[1:])
            del x, result
        fields = profile_fields(metadata)
        if fields:
            write_event(
                "span_execution",
                **fields,
                execution_mode=mode,
                checkpoint_bytes=required,
                checkpoint_limit_bytes=self.checkpoint_bytes
            )
        return (grad, *(g for layer in parameter_grads for g in layer))


def attach_lora_span_pool(backends):
    if not backends or not all(hasattr(b, "lora_pool") for b in backends.values()):
        return
    executor = LoRASpanExecutor(
        backends, checkpoint_bytes=int(os.getenv("BLOOMBEE_LORA_SPAN_CHECKPOINT_BYTES", DEFAULT_LORA_CHECKPOINT_BYTES))
    )
    pool = LoRATaskPool(
        executor,
        max_batch_size=min(b.lora_pool.max_batch_size for b in backends.values()),
        device=None,
        profile_device=executor.device,
        name="lora_span",
    )
    for backend in backends.values():
        backend.lora_span_pool = pool
