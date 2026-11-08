"""Owner-scoped immutable adapter versions and disposable forward checkpoints.

Both stores run exclusively on the serial Runtime thread. Byte and entry limits
bound resident cached tensors; active request/gradient/weight staging memory is
additional. Expired entries are removed on the next request. Cache misses are
recoverable: the client resends the saved adapter, or the worker reconstructs
layer inputs. Workers never apply optimizer updates or accumulate .grad.
"""

import hashlib
import json
import math
import os
import re
import time
from collections import OrderedDict

import torch

from bloombee.utils.lora import LoRAConfig, projection_shapes
from bloombee.utils.lora_profile import profile_fields, tensor_bytes, write_event


class TensorLRU:
    def __init__(self, limit_bytes, ttl, *, max_entries=128, clock=time.monotonic):
        if not isinstance(limit_bytes, int) or limit_bytes < 0 or not math.isfinite(ttl) or ttl <= 0 or max_entries < 1:
            raise ValueError("Invalid LoRA cache limit/TTL")
        self.limit_bytes, self.ttl, self.max_entries, self.clock = limit_bytes, ttl, max_entries, clock
        self.entries, self.bytes = OrderedDict(), 0

    def pop(self, key):
        item = self.entries.pop(key, None)
        if item is not None:
            self.bytes -= item[2]
            return item[1]

    def prune(self):
        now = self.clock()
        for key, (expiry, _, _) in list(self.entries.items()):
            if expiry <= now:
                self.pop(key)

    def get(self, key):
        self.prune()
        item = self.entries.get(key)
        if item is not None:
            self.entries.move_to_end(key)
            return item[1]

    def put(self, key, value, size):
        self.prune()
        if size > self.limit_bytes:
            return False
        self.pop(key)
        while self.entries and (self.bytes + size > self.limit_bytes or len(self.entries) >= self.max_entries):
            self.pop(next(iter(self.entries)))
        self.entries[key] = (self.clock() + self.ttl, value, size)
        self.bytes += size
        return True


def tensor_digest(tensors, description):
    digest = hashlib.sha256(description.encode())
    for tensor in tensors:
        cpu = tensor.detach().cpu().contiguous()
        digest.update(str((tuple(cpu.shape), cpu.dtype)).encode())
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class CachedLoRASpan:
    def __init__(self, executor, *, adapter_bytes=None, activation_bytes=None, ttl=None):
        self.executor = executor
        ttl = float(os.getenv("BLOOMBEE_LORA_CACHE_TTL", "300")) if ttl is None else ttl
        self.adapters = TensorLRU(
            int(os.getenv("BLOOMBEE_LORA_ADAPTER_CACHE_BYTES", 256 * 2**20))
            if adapter_bytes is None
            else adapter_bytes,
            ttl,
        )
        self.activations = TensorLRU(
            int(os.getenv("BLOOMBEE_LORA_ACTIVATION_CACHE_BYTES", 256 * 2**20))
            if activation_bytes is None
            else activation_bytes,
            ttl,
        )

    def __call__(self, inputs, grad_outputs, metadata, parameters):
        from bloombee.server.lora_execution import move_tensor

        self.adapters.prune()
        self.activations.prune()
        owner, job, request = metadata["cache_owner"], metadata["cache_job"], metadata["cache_request"]
        version = metadata["adapter_version"]
        if not owner or any(not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{32}", v) is None for v in (job, request)):
            raise ValueError("Invalid LoRA cache owner/job/request")
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError("Invalid adapter cache version")
        names = tuple(metadata["span_uids"])
        key = (owner, job, version, names)
        checkpoint_key = (owner, job, request, names)
        config = LoRAConfig(**metadata["config"])
        description = json.dumps(config.to_dict(), sort_keys=True)
        adapter = self.adapters.get(key)
        if parameters:
            width = 2 * len(config.target_modules)
            if len(parameters) != len(names) * width:
                raise ValueError("Cached adapter parameter count mismatch")
            for j, name in enumerate(names):
                backend = self.executor.backends[name]
                config.validate_layers(backend.config.num_hidden_layers)
                shapes, layer = projection_shapes(backend.config), int(name.rsplit(".", 1)[-1])
                for k, target in enumerate(config.target_modules):
                    out_features, in_features = shapes[target]
                    rank = config.rank_for(layer, target)
                    for tensor, shape in zip(
                        parameters[j * width + 2 * k : j * width + 2 * k + 2],
                        ((rank, in_features), (out_features, rank)),
                    ):
                        if tensor.shape != shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                            raise ValueError("Invalid cached adapter shape/dtype/value")
            fingerprint = tensor_digest(parameters, description)
            if adapter is not None and (adapter[0] != description or adapter[1] != fingerprint):
                raise ValueError("Adapter version reused with different parameters/configuration")
            if adapter is None:
                adapter = (description, fingerprint, tuple(p.detach().cpu().clone() for p in parameters))
                self.adapters.put(key, adapter, tensor_bytes(adapter[2]))
        elif adapter is None:
            # Internal control tensor, stripped by the RPC service.
            return (torch.tensor([-1, 0], dtype=torch.int64),)
        if adapter[0] != description:
            raise ValueError("Cached adapter configuration does not match")
        mode = metadata["execution_mode"].replace("span_cached_", "span_")
        signature = (adapter[1], metadata["seed"], metadata["training"], mode, tensor_digest((inputs,), "activation"))
        saved = self.activations.get(checkpoint_key)
        if saved is not None and saved[0] != signature:
            raise ValueError("Checkpoint request reused with different input, seed or adapter version")
        backward = metadata["backward"]
        hit = backward and saved is not None
        checkpoints, save_forward = None, False
        if backward and saved is not None:
            self.activations.pop(checkpoint_key)
            checkpoints = list(saved[1])
            saved = None
        elif not backward and metadata.get("cache_requires_backward", False):
            size = len(names) * tensor_bytes((inputs,))
            if (
                size <= self.activations.limit_bytes
                and (len(names) - 1) * tensor_bytes((inputs,)) <= self.executor.checkpoint_bytes
            ):
                device = "cpu" if mode == "span_cpu" else self.executor.device
                checkpoints = [move_tensor(inputs.detach().clone(), device, metadata, "checkpoint_save")] + [None] * (
                    len(names) - 1
                )
                save_forward = True
        result = self.executor.run(
            inputs,
            grad_outputs,
            {**metadata, "execution_mode": mode},
            adapter[2],
            checkpoints=checkpoints,
            save_forward=save_forward,
        )
        if save_forward:
            self.activations.put(checkpoint_key, (signature, tuple(checkpoints)), tensor_bytes(checkpoints))
        fields = profile_fields(metadata)
        if fields:
            write_event(
                "lora_cache",
                **fields,
                backward=backward,
                checkpoint_hit=hit,
                adapter_bytes=self.adapters.bytes,
                adapter_limit_bytes=self.adapters.limit_bytes,
                activation_bytes=self.activations.bytes,
                activation_limit_bytes=self.activations.limit_bytes
            )
        return (*result, torch.tensor([int(key in self.adapters.entries), int(hit)], dtype=torch.int64))
