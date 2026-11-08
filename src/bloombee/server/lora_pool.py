"""LoRA task-pool measurements without modifying the tensor response contract."""

import os
import time

import torch

from bloombee.server.task_pool import PrioritizedTaskPool
from bloombee.utils.lora_profile import enabled, memory_snapshot, profile_fields, synchronize, tensor_bytes, write_event

_RESET_DEVICES = set()


class LoRATaskPool(PrioritizedTaskPool):
    def __init__(self, *args, profile_device=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile_device = profile_device
        self._profiles = {}

    def load_batch_to_runtime(self, timeout=None, device=None):
        target = device if device is not None else self.device
        measurement_device = self.profile_device if self.profile_device is not None else target
        profiling = enabled()
        if profiling:
            synchronize(measurement_device)
            key = (os.getpid(), str(measurement_device))
            if (
                measurement_device is not None
                and torch.device(measurement_device).type == "cuda"
                and key not in _RESET_DEVICES
            ):
                torch.cuda.reset_peak_memory_stats(measurement_device)
                _RESET_DEVICES.add(key)
        start = time.monotonic()
        uid, inputs = super().load_batch_to_runtime(timeout=timeout, device=device)
        task = self._dispatched_tasks[uid]
        metadata = task.args[2] if len(task.args) > 2 and isinstance(task.args[2], dict) else {}
        fields = profile_fields(metadata)
        if fields:
            synchronize(measurement_device)
            self._profiles[uid] = dict(
                **fields,
                pool=self.name,
                rpc_phase="backward" if metadata["backward"] else "forward",
                queue_wait_s=max(0.0, start - task.time_submitted),
                input_transfer_s=time.monotonic() - start,
                input_tensor_bytes=tensor_bytes(task.args),
                h2d_tensor_bytes=sum(
                    a.numel() * a.element_size()
                    for a, b in zip(task.args, inputs)
                    if isinstance(a, torch.Tensor) and a.device.type == "cpu" and b.device.type == "cuda"
                ),
                target=measurement_device,
            )
        return uid, inputs

    def send_outputs_from_runtime(self, uid, batch_outputs):
        event = self._profiles.pop(uid, None)
        if event is None:
            return super().send_outputs_from_runtime(uid, batch_outputs)
        target = event.pop("target")
        synchronize(target)
        memory = memory_snapshot(target)
        event.update(
            output_tensor_bytes=tensor_bytes(batch_outputs),
            d2h_tensor_bytes=sum(t.numel() * t.element_size() for t in batch_outputs if t.device.type == "cuda"),
        )
        start = time.monotonic()
        super().send_outputs_from_runtime(uid, batch_outputs)
        event["output_transfer_s"] = time.monotonic() - start
        # Cumulative process/device peak since the first profiled task, not a
        # per-task peak or the physical footprint of all workers sharing a GPU.
        write_event("runtime_transfer", **event, **memory)

    def send_exception_from_runtime(self, uid, exception):
        self._profiles.pop(uid, None)
        return super().send_exception_from_runtime(uid, exception)
