"""Opt-in LoRA measurements. Durations are local; payload bytes are not NIC bytes.

Set BLOOMBEE_LORA_PROFILE_DIR on the client and each worker. CUDA measurements
synchronize the measured device and perturb overlap; use an isolated profiling run.
No tensor values or training text are recorded.
"""

import json
import os
import resource
import socket
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch

_STEP = ContextVar("lora_profile_step", default={})


def enabled():
    return bool(os.getenv("BLOOMBEE_LORA_PROFILE_DIR"))


def tensor_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors if isinstance(t, torch.Tensor))


@contextmanager
def profile_step(step, microbatch=0, warmup=False):
    token = _STEP.set(dict(step=step, microbatch=microbatch, warmup=warmup))
    try:
        yield
    finally:
        _STEP.reset(token)


def request_metadata():
    return {"profile": {"id": uuid.uuid4().hex, **_STEP.get()}} if enabled() else {}


def profile_fields(metadata):
    if not enabled():
        return {}
    fields = metadata.get("profile", {})
    if not isinstance(fields, dict) or not isinstance(fields.get("id"), str) or len(fields["id"]) != 32:
        return {}
    return {k: v for k, v in fields.items() if k in ("id", "step", "microbatch", "warmup")}


def synchronize(device):
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def memory_snapshot(device=None):
    import psutil

    result = dict(
        rss_bytes=psutil.Process().memory_info().rss,
        process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    )
    if device is not None and torch.device(device).type == "cuda":
        result.update(
            cuda_device=str(device),
            cuda_allocated_bytes=torch.cuda.memory_allocated(device),
            cuda_reserved_bytes=torch.cuda.memory_reserved(device),
            cuda_peak_allocated_since_reset_bytes=torch.cuda.max_memory_allocated(device),
            cuda_peak_reserved_since_reset_bytes=torch.cuda.max_memory_reserved(device),
        )
    return result


def write_event(kind, **fields):
    directory = os.getenv("BLOOMBEE_LORA_PROFILE_DIR")
    if not directory:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    record = dict(schema_version=1, kind=kind, host=socket.gethostname(), pid=os.getpid(), **fields)
    # A file per process prevents different workers interleaving JSON records.
    with (path / f"lora-{socket.gethostname()}-{os.getpid()}.jsonl").open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


@contextmanager
def measure(metadata, phase, device, **fields):
    profile = profile_fields(metadata)
    if not profile:
        yield
        return
    synchronize(device)
    start = time.perf_counter()
    try:
        yield
    except BaseException:
        write_event("block", **profile, phase=phase, status="failed", **fields)
        raise
    else:
        synchronize(device)
        elapsed = time.perf_counter() - start
        write_event("block", **profile, phase=phase, status="ok", seconds=elapsed, **fields, **memory_snapshot(device))


def expected_stateless_cost(layers, boundary_bytes, adapter_bytes):
    """One successful forward/backward pair, same boundary dtype, FP32 A/B.

    Excludes protocol framing and compression. Forward with grad and without
    grad have different costs; this reports invocation counts, not FLOPs.
    """
    if layers < 1 or boundary_bytes < 0 or adapter_bytes < 0:
        raise ValueError("Expected positive layers and nonnegative byte counts")
    return dict(
        request_tensor_bytes=3 * boundary_bytes + 2 * adapter_bytes,
        response_tensor_bytes=2 * boundary_bytes + adapter_bytes,
        forward_calls=3 * layers - 1,
        backward_calls=layers,
    )
