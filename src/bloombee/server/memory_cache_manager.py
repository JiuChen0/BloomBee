
import contextlib
import asyncio
import torch
import os
import threading
from typing import Optional, Tuple, AsyncContextManager, Sequence
from concurrent.futures import ThreadPoolExecutor

from bloombee.server.memory_cache import MemoryCache, AdaptedKVCache, KVCacheMetadata
from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
from bloombee.flexgen_utils.policy import Policy
from bloombee.flexgen_utils.pytorch_backend import DeviceType, TorchDisk, TorchMixedDevice, TorchTensor, general_copy
from hivemind.utils import TensorDescriptor, enter_asynchronously, get_logger

from bloombee.data_structures import Handle
from bloombee.utils.asyncio import shield_and_wait
from bloombee.utils.debug_config import get_env_bool_with_debug_fallback
from bloombee.utils.misc import get_size_in_bytes

from transformers import PretrainedConfig

logger = get_logger(__name__)


class KVCacheManager:
    def __init__(self, cache_max_size_tokens: int, 
                 max_alloc_timeout: int, 
                 policy: Policy, 
                 env: ExecutionEnv,
                 block_config: PretrainedConfig):
        # Initialize as 2D array structure
        self.env = env
        self.runtime_pid = os.getpid()
        self.device = self.get_cache_device(policy)
        self.cache = MemoryCache(cache_max_size_tokens, max_alloc_timeout, policy, block_config, self.device)
        self.offloading_policy = policy
        self.attention_compute = (self.env.cpu if policy.cpu_cache_compute
                                  else self.env.gpu)
        self.block_config = block_config
        self.max_alloc_timeout = max_alloc_timeout
        self._active_cache_tensors_stack = []
        self._reorder_executor = ThreadPoolExecutor(max_workers=1)

        self._verbose_kv_logs = get_env_bool_with_debug_fallback(
            "BLOOMBEE_VERBOSE_KV_LOGS",
            default=False,
            groups=("kv_cache",),
        )
        self._kv_log_once_keys = set()
        logger.info(
            "[KVCACHE_OFFLOAD] GPU multiplexing removed; per-microbatch KV staging is disabled"
        )
        logger.info(
            "[KVCACHE_OFFLOAD] KV verbose logs: %s (set BLOOMBEE_VERBOSE_KV_LOGS=1 for per-layer KV traces)",
            "ON" if self._verbose_kv_logs else "OFF",
        )

        self._enable_kv_wait_timing = os.environ.get("BLOOMBEE_ENABLE_KV_WAIT_TIMING", "1") == "1"
        logger.info(
            "[KVCACHE_OFFLOAD] KV wait timing: %s (set BLOOMBEE_ENABLE_KV_WAIT_TIMING=0 to disable)",
            "ON" if self._enable_kv_wait_timing else "OFF",
        )
        self._kv_timing = {
            "prefetch_wait_ms": 0.0,
            "offload_wait_ms": 0.0,
            "prefetch_wait_calls": 0,
            "offload_wait_calls": 0,
            "prefetch_launch_ms": 0.0,
            "offload_launch_ms": 0.0,
            "prefetch_launch_calls": 0,
            "offload_launch_calls": 0,
        }
        # Request-level offload accounting. This gives a compact signal of whether
        # micro-batch KV offload/prefetch actually happened and how much data moved.
        self._offload_summary = {
            "offload_calls": 0,
            "prefetch_calls": 0,
            "offload_bytes": 0,
            "prefetch_bytes": 0,
            "staging_peak_bytes": 0,
        }
        self._staging_bytes = {}
        
    def _log_kv_detail(self, message: str, *args):
        """High-frequency KV diagnostics: INFO only when verbose flag is enabled."""
        if self._verbose_kv_logs:
            logger.info(message, *args)
        else:
            logger.debug(message, *args)

    def _log_kv_once(self, key, message: str, *args):
        """Emit an INFO diagnostic once per process for important KV mode/state transitions."""
        if key in self._kv_log_once_keys:
            return
        self._kv_log_once_keys.add(key)
        logger.info(message, *args)

    @staticmethod
    def _bytes_to_mb(num_bytes: float) -> float:
        return float(num_bytes) / (1024.0 * 1024.0)

    def _get_gpu_memory_snapshot(self) -> dict:
        snapshot = {
            "gpu_alloc_mb": 0.0,
            "gpu_reserved_mb": 0.0,
            "gpu_max_alloc_mb": 0.0,
        }
        if not torch.cuda.is_available():
            return snapshot
        try:
            device_idx = torch.cuda.current_device()
            snapshot["gpu_alloc_mb"] = self._bytes_to_mb(torch.cuda.memory_allocated(device_idx))
            snapshot["gpu_reserved_mb"] = self._bytes_to_mb(torch.cuda.memory_reserved(device_idx))
            snapshot["gpu_max_alloc_mb"] = self._bytes_to_mb(torch.cuda.max_memory_allocated(device_idx))
        except Exception:
            pass
        return snapshot

    def get_kv_timing_snapshot(self, reset: bool = False) -> dict:
        """
        Return cumulative KV timing counters used to quantify overlap effectiveness.

        Args:
            reset: If True, clear counters after snapshot.
        """
        snapshot = dict(self._kv_timing)
        snapshot["offload_calls"] = float(self._offload_summary.get("offload_calls", 0))
        snapshot["prefetch_calls"] = float(self._offload_summary.get("prefetch_calls", 0))
        snapshot["offload_bytes"] = float(self._offload_summary.get("offload_bytes", 0))
        snapshot["prefetch_bytes"] = float(self._offload_summary.get("prefetch_bytes", 0))
        snapshot["staging_peak_bytes"] = float(self._offload_summary.get("staging_peak_bytes", 0))
        snapshot["staging_live_bytes"] = 0.0
        snapshot["active_staged_entries"] = 0.0
        snapshot["async_kv_transfer"] = 0.0
        snapshot.update(self._get_gpu_memory_snapshot())
        if reset:
            for k in list(self._kv_timing.keys()):
                self._kv_timing[k] = 0 if k.endswith("_calls") else 0.0
        return snapshot
        
        
    def get_cache_device(self, policy):
        if policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed

        if policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device
        return device

    def clear(self):
        # for b in range(self.max_batch_size):
        #     for l in range(self.num_layers):
        #         self.cache[b][l] = None
        # No-op for now; handles are freed by context manager or force_free
        return
    
    @contextlib.asynccontextmanager
    async def allocate_cache(
        self, *descriptors: TensorDescriptor, timeout: float
    ) -> AsyncContextManager[Sequence[Handle]]:
        assert os.getpid() != self.runtime_pid, "must be called by a ConnectionHandler, not runtime"
        assert all(descr.device is not None for descr in descriptors), "please specify allocated devices"
        if self.max_alloc_timeout is not None and timeout is not None:
            timeout = min(timeout, self.max_alloc_timeout)

        allocation_tokens = self.get_allocation_size_tokens(*descriptors)
        allocation_size = allocation_tokens * self.size_per_token() * len(descriptors)

        gib = 1024**3
        cur_tokens, max_tokens = self.current_size_tokens, self.max_size_tokens
        max_size = max_tokens * self.size_per_token() * len(descriptors)
        cur_size = cur_tokens * self.size_per_token() * len(descriptors)
        # logger.info(f"size_per_token: {self.size_per_token()}")
        friendly_max_size = f"{max_size / gib:.2f}" if max_size != 2**64 - 1 else "inf"
        used_pct = (cur_size / max_size * 100.0) if max_size != 0 and max_size != 2**64 - 1 else 0.0
        # logger.info(
        #     f"rpc_inference.wait_for_alloc(size={allocation_size / gib:.2f} GiB), "
        #     f"already used {cur_size / gib:.2f}/{friendly_max_size} GiB ({used_pct:.1f}%)"
        # )

        alloc_task = asyncio.create_task(self.cache._schedule_alloc(allocation_tokens, *descriptors, timeout=timeout))
        try:
            handles = await shield_and_wait(alloc_task)
            # logger.info(f"rpc_inference.alloc_done(size={allocation_size / gib:.2f} GiB)")
            yield handles
        finally:
            self.cache._free(allocation_tokens, alloc_task)
            
            
    @staticmethod
    def get_allocation_size_tokens(*descriptors: TensorDescriptor) -> int:
        allocation_tokens_num = 0
        for descr in descriptors:
            allocation_tokens_num = max(descr.shape[-1], allocation_tokens_num) 
        return allocation_tokens_num
        
    
    def add_cache(self, kvs: AdaptedKVCache, start_position: int):
        self._write_kvs(kvs, start_position)
                
    def update_cache(
        self, new_kvs: AdaptedKVCache, start_position: int,
        batch_offset: int = 0, full_batch_size: int = 0, micro_batch_size: int = 0
    ):
        """
        Update KV cache with new values.
        
        Args:
            new_kvs: New KV tensors to write
            start_position: Start position along sequence dimension
            batch_offset: Start index in the full batch for micro-batch (default 0)
            full_batch_size: Total batch size for micro-batch support (0 = no micro-batch)
            micro_batch_size: Actual size of this micro-batch (0 = use full_batch_size - batch_offset)
        """
        if self._verbose_kv_logs:
            logger.info(
                "[MBPIPE_UPDATE_CACHE] start_position=%s batch_offset=%s full_batch=%s micro_batch=%s new_kvs_type=%s active_cache_depth=%s",
                start_position,
                batch_offset,
                full_batch_size,
                micro_batch_size,
                type(new_kvs).__name__ if new_kvs is not None else "None",
                len(self._active_cache_tensors_stack),
            )
        self._write_kvs(new_kvs, start_position, batch_offset, full_batch_size, micro_batch_size)
    
    def tokens_left(self) -> int:
        return self.cache.tokens_left

    @property
    def current_size_tokens(self) -> int:
        return self.cache.current_size_tokens

    @property
    def max_size_tokens(self) -> int:
        return self.cache.max_size_tokens
    
    def size_per_token(self) -> int:
        cache_values_per_block = 2 * self.block_config.hidden_size
        cache_values_per_block //= self.block_config.num_key_value_groups
        return cache_values_per_block * get_size_in_bytes(torch.float16)
    
    def select_cache(
        self,
        prefix_length: int,
        hypo_ids: Optional[torch.Tensor] = None,
        kv_cache_position_ids: Optional[torch.Tensor] = None,
        batch_offset: int = 0,
        full_batch_size: int = 0,
        micro_batch_size: int = 0,
        cache_tensors: Sequence[torch.Tensor] = None,
    ):
        """
        Return standard KV for computation
        K, V: torch.Tensor, both with shape (B, H, S, D), located on compute_dst (CPU or GPU)
        Convention:
        - Internal cache is stored along dimension (S, B*H, D)
        - If mixed device (MIXED), segments will be merged on compute_dst and returned
        
        Args:
            prefix_length: Length of prefix to read from cache
            hypo_ids: Optional hypothesis IDs for reordering
            kv_cache_position_ids: Optional position IDs for cache
            batch_offset: Start index in the full batch for micro-batch (default 0)
            full_batch_size: Total batch size (0 = no micro-batch, read entire cache)
            micro_batch_size: Actual size of this micro-batch (0 = use full_batch_size - batch_offset)
        """
        if cache_tensors is None:
            assert self._active_cache_tensors_stack, "write_pkv_cache called outside of use_cache context"
            cache_tensors = self._active_cache_tensors_stack[-1]  # TorchTensor
        if prefix_length <= 0:
            return None, None, False
        
        (k_cache, v_cache), = cache_tensors
        S_full, BH_full, D = k_cache.shape
        assert prefix_length <= S_full, f"prefix_length={prefix_length} > seq_len={S_full}"
        
        # [MBPIPE] Debug log - use DEBUG level to avoid log spam
        logger.debug(f"[MBPIPE] select_cache: prefix_length={prefix_length}, "
                    f"cache_shape=(S={S_full}, BH={BH_full}, D={D}), "
                    f"batch_offset={batch_offset}, full_batch_size={full_batch_size}, micro_batch_size={micro_batch_size}")
        
        # Micro-batch support: compute BH slice
        H = getattr(self.block_config, "num_attention_heads", BH_full)
        full_batch_in_cache = BH_full // H  # Actual batch size in allocated cache
        
        # [MBPIPE_KV_VERIFY] High-frequency per-layer read diagnostics
        self._log_kv_detail("[MBPIPE_KV_VERIFY] === KV CACHE READ ===")
        self._log_kv_detail(f"[MBPIPE_KV_VERIFY] Cache tensor shape: (S={S_full}, BH={BH_full}, D={D})")
        self._log_kv_detail(f"[MBPIPE_KV_VERIFY] Cache batch capacity: {full_batch_in_cache} (BH={BH_full} / H={H})")
        self._log_kv_detail(
            f"[MBPIPE_KV_VERIFY] Read params: prefix_len={prefix_length}, batch_offset={batch_offset}, "
            f"full_batch={full_batch_size}, micro_batch={micro_batch_size}"
        )
        
        if full_batch_size > 0 and micro_batch_size > 0:
            # Legacy mode: cache holds full batch, use batch_offset for slicing
            BH_offset_start = batch_offset * H
            BH_offset_end = BH_offset_start + micro_batch_size * H
            BH_offset_end = min(BH_offset_end, BH_full)  # Safety clamp
            BH = BH_offset_end - BH_offset_start
            self._log_kv_once(
                ("kv_read_mode", "legacy"),
                "[MBPIPE_KV_VERIFY] KV READ mode: LEGACY (cache stores full batch; read uses logical batch_offset)",
            )
            self._log_kv_detail(f"[MBPIPE_KV_VERIFY] Mode: LEGACY - reading BH[{BH_offset_start}:{BH_offset_end}]")
        else:
            # Full batch mode
            BH_offset_start = 0
            BH_offset_end = BH_full
            BH = BH_full
            self._log_kv_once(
                ("kv_read_mode", "full"),
                "[MBPIPE_KV_VERIFY] KV READ mode: FULL BATCH",
            )
            self._log_kv_detail(f"[MBPIPE_KV_VERIFY] Mode: FULL BATCH - reading BH[0:{BH_full}]")

        # Target device for computation (CPU/GPU)
        compute_dst = self.attention_compute  # 统一在计算设备上物化

        # Path determination (whether MIXED)
        if self.offloading_policy.cpu_cache_compute and (
            self.device.device_type == DeviceType.MIXED and getattr(k_cache.data[0][0], "shape", None) is not None
        ):
            path = 2
        else:
            path = 0 if not self.offloading_policy.cpu_cache_compute else 1

        # Required slice - MUST include BH slice for micro-batch support
        need_reorder = False
        if kv_cache_position_ids is None or kv_cache_position_ids.numel() == 0:
            idx_all = (slice(0, prefix_length), slice(BH_offset_start, BH_offset_end))
        else:
            root_position = kv_cache_position_ids[0]
            prefix_positions = list(range(root_position))  # [0, 1, 2, ..., root-1]
            s_indices = prefix_positions + kv_cache_position_ids.tolist()  # 完整序列
            expected_continuous = list(range(len(s_indices)))
            need_reorder = False if (s_indices == expected_continuous) else True
            prefix_length = len(s_indices)
            # [MBPIPE FIX] Include BH slice for micro-batch support
            idx_all = (s_indices, slice(BH_offset_start, BH_offset_end))
            logger.debug(f"[MBPIPE] select_cache with kv_cache_position_ids: s_len={len(s_indices)}, "
                        f"BH_slice=[{BH_offset_start}:{BH_offset_end}]")

        # Utility: get underlying torch.Tensor
        def _as_torch(x):
            return x.data if hasattr(x, "data") else x

        # 1) Materialize to (S, BH, D) torch.Tensor (located on compute_dst)
        if path == 0:
            # [KVCACHE_OFFLOAD] Handle TorchMixedDevice reading for micro-batch correctly
            # When cache is MixedDevice and we're reading a BH slice, we need to manually
            # read from each segment and concatenate, because general_copy's cut_indices
            # doesn't work correctly for micro-batch BH offsets
            if hasattr(k_cache, 'device') and getattr(k_cache.device, 'device_type', None) == DeviceType.MIXED:
                # TorchMixedDevice: manually read from each segment and concatenate
                tensors, seg_points = k_cache.data  # ([gpu_tensor, cpu_tensor, ...], [0, seg1, seg2, ...])
                v_tensors, _ = v_cache.data
                
                s_slice, bh_slice = idx_all[:2]
                bh_start, bh_end = bh_slice.start, bh_slice.stop
                
                self._log_kv_detail(
                    f"[KVCACHE_OFFLOAD] select_cache MixedDevice: seg_points={seg_points}, "
                    f"bh_range=[{bh_start}:{bh_end}], prefix_length={prefix_length}"
                )
                
                k_parts = []
                v_parts = []
                
                for i, (k_seg, v_seg) in enumerate(zip(tensors, v_tensors)):
                    if k_seg is None:
                        continue
                    seg_start = seg_points[i]
                    seg_end = seg_points[i + 1]
                    
                    # Calculate overlap between requested BH range and this segment
                    overlap_start = max(bh_start, seg_start)
                    overlap_end = min(bh_end, seg_end)
                    
                    if overlap_start < overlap_end:
                        # There is overlap, read from this segment
                        # Segment-local indices
                        local_bh_start = overlap_start - seg_start
                        local_bh_end = overlap_end - seg_start
                        
                        logger.debug(f"[KVCACHE_OFFLOAD] select_cache seg[{i}]: "
                                    f"seg_range=[{seg_start}:{seg_end}], local_bh=[{local_bh_start}:{local_bh_end}]")
                        
                        # Get underlying tensor data
                        k_data = k_seg.data if hasattr(k_seg, 'data') else k_seg
                        v_data = v_seg.data if hasattr(v_seg, 'data') else v_seg
                        
                        # Read the slice: (S, local_bh, D)
                        k_part = k_data[s_slice, local_bh_start:local_bh_end, :].to(compute_dst.dev, non_blocking=True)
                        v_part = v_data[s_slice, local_bh_start:local_bh_end, :].to(compute_dst.dev, non_blocking=True)
                        
                        k_parts.append(k_part)
                        v_parts.append(v_part)
                
                # Concatenate all parts along BH dimension
                if len(k_parts) == 1:
                    k_sbh = k_parts[0]
                    v_sbh = v_parts[0]
                else:
                    k_sbh = torch.cat(k_parts, dim=1)
                    v_sbh = torch.cat(v_parts, dim=1)
                    
                logger.debug(f"[KVCACHE_OFFLOAD] select_cache MixedDevice result: k_sbh.shape={k_sbh.shape}")
            else:
                # Non-MixedDevice: use original smart_copy path
                k_sel, _ = k_cache.smart_copy(compute_dst, idx_all)
                v_sel, _ = v_cache.smart_copy(compute_dst, idx_all)
                k_sbh = _as_torch(k_sel)
                v_sbh = _as_torch(v_sel)
            # logger.info(f\"k_cache: {k_cache.shape}, k_sbh: {k_sbh.shape}\")

        elif path == 1:
            # Use compute_dst workspace to carry (S, BH, D)
            k_buf, v_buf = compute_dst.next_attention_compute_workspace()
            general_copy(k_buf, idx_all, k_cache, idx_all)
            general_copy(v_buf, idx_all, v_cache, idx_all)
            k_sbh, v_sbh = _as_torch(k_buf), _as_torch(v_buf)

        else:  # path == 2, MIXED: GPU segment + other segments merged to compute_dst
            gpu_k_part = k_cache.data[0][0][:prefix_length]  # (S, BH_gpu, D)
            gpu_v_part = v_cache.data[0][0][:prefix_length]
            BH_gpu = int(gpu_k_part.shape[1])

            # Copy remaining segments to compute_dst workspace
            k_rest, v_rest = compute_dst.next_attention_compute_workspace()
            idx_rest = (slice(0, prefix_length), slice(BH_gpu, BH))
            general_copy(k_rest, idx_rest, k_cache, idx_rest)
            general_copy(v_rest, idx_rest, v_cache, idx_rest)
            k_rest_t, v_rest_t = _as_torch(k_rest), _as_torch(v_rest)

            # If compute_dst is not on GPU, need to move GPU segment to compute_dst then concatenate
            if gpu_k_part.device != k_rest_t.device:
                gpu_k_part = gpu_k_part.to(k_rest_t.device, non_blocking=True)
                gpu_v_part = gpu_v_part.to(v_rest_t.device, non_blocking=True)

            k_sbh = torch.cat([gpu_k_part, k_rest_t[:, BH_gpu:BH, :]], dim=1)
            v_sbh = torch.cat([gpu_v_part, v_rest_t[:, BH_gpu:BH, :]], dim=1)

        # 2) (S, BH, D) -> (B, H, S, D) standard PKV view (zero-copy)
        H = getattr(self.block_config, "num_attention_heads", None)
        assert H is not None, "block_config.num_attention_heads is required"
        BH = k_sbh.shape[1]
        # For MQA/GQA models (e.g. Falcon-7b with num_kv_heads=1), BH = B*num_kv_heads < num_attention_heads.
        # Use BH as H in that case so B = BH/H = 1 is correct.
        if BH < H:
            H = BH
        assert (BH % H) == 0, f"BH={BH} not divisible by H={H}"
        B = BH // H

        def _to_pkv(x_sbh: torch.Tensor) -> torch.Tensor:
            # (S, BH, D) -> (S, B, H, D) -> (B, H, S, D)
            return x_sbh.view(prefix_length, B, H, D).permute(1, 2, 0, 3)

        k_pkv = _to_pkv(k_sbh)
        v_pkv = _to_pkv(v_sbh)
        
        # [MBPIPE] Debug log output shape
        logger.debug(f"[MBPIPE] select_cache OUTPUT: k_pkv={k_pkv.shape}, B={B}, S={prefix_length}")

        # Optional: reorder batch by hypo_ids
        # if hypo_ids is not None:
        #     # hypo_ids: shape (B,)
        #     k_pkv = k_pkv.index_select(0, hypo_ids)
        #     v_pkv = v_pkv.index_select(0, hypo_ids)
        return k_pkv, v_pkv, need_reorder


    
    @contextlib.contextmanager
    def use_cache(self, *handles: Handle) -> Sequence[torch.Tensor]:
        with self.cache.use_cache(*handles) as cache_tensors:
            # Keep underlying tensors in the stack for centralized writes,
            # but yield clones to callers to prevent accidental in-place edits
            # logger.info(f"use cache, cache_tensors: {cache_tensors}, len={len(cache_tensors)}")
            self._active_cache_tensors_stack.append(cache_tensors)
            try:
                # safe_views = tuple(t.detach().clone() for t in cache_tensors)
                yield cache_tensors
            finally:
                self._active_cache_tensors_stack.pop()

    def delete_cache(self, *handles: Handle):
        """Explicitly delete cache handles to free space early."""
        try:
            self.cache.force_free(*handles)
        except Exception as e:
            logger.warning(f"OFFLOAD: delete_cache failed for handles={handles}: {e}")
    
    def clear_offload_state(self):
        """GPU multiplexing removal left no per-request offload state to clear."""
        self._offload_summary = {
            "offload_calls": 0,
            "prefetch_calls": 0,
            "offload_bytes": 0,
            "prefetch_bytes": 0,
            "staging_peak_bytes": 0,
        }
        self._staging_bytes.clear()
    
    def _write_kvs(self, kvs, start_position: int, batch_offset: int = 0, full_batch_size: int = 0, micro_batch_size: int = 0, cache_tensors: Sequence[torch.Tensor] = None) -> None:
        """
        Write new_kvs to current active cache:
        - Target cache_tensors: k_cache, v_cache, both with shape (S_total, B*H, D)
        - Write start position: start_position (along sequence dimension)
        - Source new_kvs:
            key:   (BH_micro, D, s_new) for micro-batch or (BH_full, D, s_new) for full batch
            value: (BH_micro, s_new, D) for micro-batch or (BH_full, s_new, D) for full batch
        - batch_offset: Start index in the full batch for micro-batch (along B dimension)
        - full_batch_size: Total batch size (0 = no micro-batch, use entire cache)
        - micro_batch_size: Actual size of this micro-batch (0 = use full_batch_size - batch_offset)
        """
        if cache_tensors is None:
            assert self._active_cache_tensors_stack, "write_pkv_cache called outside of use_cache context"
            cache_tensors = self._active_cache_tensors_stack[-1]  # TorchTensor
        (k_cache, v_cache), = cache_tensors
        S_total, BH_dst, D_dst = k_cache.shape
        
        # [MBPIPE_WRITE_DEBUG] Log the cache shape and write parameters
        H = getattr(self.block_config, "num_attention_heads", 32)
        cache_batch_size = BH_dst // H
        
        # [MBPIPE_KV_VERIFY] High-frequency per-layer write diagnostics
        self._log_kv_detail("[MBPIPE_KV_VERIFY] === KV CACHE WRITE ===")
        self._log_kv_detail(f"[MBPIPE_KV_VERIFY] Cache tensor shape: (S={S_total}, BH={BH_dst}, D={D_dst})")
        self._log_kv_detail(f"[MBPIPE_KV_VERIFY] Cache batch capacity: {cache_batch_size} (BH={BH_dst} / H={H})")
        self._log_kv_detail(
            f"[MBPIPE_KV_VERIFY] Write params: start_pos={start_position}, batch_offset={batch_offset}, "
            f"full_batch={full_batch_size}, micro_batch={micro_batch_size}"
        )

        if full_batch_size > 0:
            self._log_kv_once(
                ("kv_write_mode", "legacy"),
                "[MBPIPE_KV_VERIFY] KV WRITE mode: LEGACY (cache stores full batch; write uses logical batch_offset)",
            )
            self._log_kv_detail(
                f"[MBPIPE_KV_VERIFY] Mode: LEGACY (cache={cache_batch_size}, full_batch={full_batch_size})"
            )
        else:
            self._log_kv_once(
                ("kv_write_mode", "single"),
                "[MBPIPE_KV_VERIFY] KV WRITE mode: SINGLE BATCH",
            )
            self._log_kv_detail("[MBPIPE_KV_VERIFY] Mode: SINGLE BATCH (no micro-batching)")
        

        # Extract (key, value)
        new_kvs = kvs.kvs if hasattr(kvs, "kvs") else kvs
        key, value = new_kvs

        # If possibly FlexGen wrapper/compression, convert to torch.Tensor
        try:
            def _to_torch(x):
                if hasattr(x, 'device') and (
                    getattr(getattr(x, 'device', None), 'device_type', None) == DeviceType.COMPRESSED
                    or (hasattr(x, 'data') and isinstance(getattr(x, 'data'), tuple) and len(getattr(x, 'data')) == 3)
                ):
                    return x.device.decompress(x)
                return getattr(x, 'data', x)
        except Exception:
            def _to_torch(x):
                return getattr(x, 'data', x)

        key_t = _to_torch(key)       # (BH_micro, D, s_new)
        value_t = _to_torch(value)   # (BH_micro, s_new, D)

        # Shape validation
        assert key_t.ndim == 3 and value_t.ndim == 3, f"new_kvs dims invalid: key {key_t.shape}, value {value_t.shape}"
        BH_src, D_src, s_new = key_t.shape
        assert value_t.shape == (BH_src, s_new, D_src), f"value shape {value_t.shape} != (BH, s_new, D)"
        assert D_src == D_dst, f"D mismatch: src {D_src} vs dst {D_dst}"
        
        if full_batch_size > 0:
            # Micro-batch mode: BH_src is current micro-batch, BH_dst is cache capacity.
            BH_offset_start = batch_offset * H
            BH_offset_end = BH_offset_start + BH_src
            if BH_offset_end > BH_dst:
                logger.error(f"[MBPIPE_DEBUG] BH offset out of bounds: {BH_offset_end} > {BH_dst}, "
                            f"clamping to BH_dst={BH_dst}")
                BH_offset_end = BH_dst
            logger.debug(f"[MBPIPE] _write_kvs LEGACY: BH_slice=[{BH_offset_start}:{BH_offset_end}]")
        else:
            # Full batch mode: if BH_src < BH_dst, auto-adapt by writing to first BH_src entries
            # This handles the case where request batch_size < server cache batch_size
            if BH_src <= BH_dst:
                BH_offset_start = 0
                BH_offset_end = BH_src
                if BH_src < BH_dst:
                    actual_batch_src = BH_src // H
                    actual_batch_dst = BH_dst // H
                    self._log_kv_once(
                        ("kv_write_mode", "auto_adapt"),
                        "[MBPIPE_DEBUG] KV auto-adapt mode observed (request batch smaller than cache capacity)",
                    )
                    self._log_kv_detail(
                        f"[MBPIPE_DEBUG] Auto-adapting: writing first {BH_src} of {BH_dst} BH entries "
                        f"(batch {actual_batch_src} of {actual_batch_dst})"
                    )
                logger.debug(f"[MBPIPE] _write_kvs FULL: BH_slice=[0:{BH_src}]")
            else:
                # This should not happen - source is larger than destination
                raise AssertionError(f"BH mismatch: src {BH_src} > dst {BH_dst}, cannot write")

        end_position = start_position + s_new
        if not (0 <= start_position < S_total and end_position <= S_total):
            # Out of bounds: use overwrite-tail policy
            key_t = key_t[:, :, -S_total:]
            value_t = value_t[:, -S_total:, :]
            s_new = S_total
            start_position = 0
            end_position = S_total

        # Align dtype
        if key_t.dtype != k_cache.dtype:
            key_t = key_t.to(dtype=k_cache.dtype)
        if value_t.dtype != v_cache.dtype:
            value_t = value_t.to(dtype=v_cache.dtype)

        # Transform to internal layout (s_new, BH, D)
        k_write = key_t.permute(2, 0, 1)   # (s_new, BH_src, D)
        v_write = value_t.permute(1, 0, 2) # (s_new, BH_src, D)

        # Target slice - now with BH offset for micro-batch support
        dst_idx = (slice(start_position, start_position + s_new), 
                   slice(BH_offset_start, BH_offset_end), 
                   slice(0, D_src))
        
        # [MBPIPE] Debug log write target
        logger.debug(f"[MBPIPE] _write_kvs TARGET: S=[{start_position}:{start_position + s_new}], "
                    f"BH=[{BH_offset_start}:{BH_offset_end}]")

        # Wrap source into TorchTensor
        k_src_tt = TorchTensor.create_from_torch(k_write, self.attention_compute)
        v_src_tt = TorchTensor.create_from_torch(v_write, self.attention_compute)

        # [KVCACHE_OFFLOAD] Handle TorchMixedDevice (GPU+CPU split) properly for micro-batch slicing
        # The issue: general_copy's cut_indices uses segment boundaries on BOTH src and dst,
        # but for micro-batch, src size != dst slice size (e.g., src=128 but dst_idx spans [128:256])
        def _write_to_cache(cache, src_tt, dst_idx, cache_name):
            """Write source tensor to cache, handling TorchMixedDevice segment splitting."""
            if hasattr(cache, 'device') and getattr(cache.device, 'device_type', None) == DeviceType.MIXED:
                # TorchMixedDevice: manually split writes across segments
                tensors, seg_points = cache.data  # ([gpu_tensor, cpu_tensor, ...], [0, seg1, seg2, ...])
                s_slice, bh_slice, d_slice = dst_idx
                bh_start, bh_end = bh_slice.start, bh_slice.stop
                
                logger.debug(f"[KVCACHE_OFFLOAD] {cache_name} is MixedDevice, seg_points={seg_points}, "
                            f"bh_range=[{bh_start}:{bh_end}]")
                
                # Track position in source tensor
                src_offset = 0
                
                for i, seg_tensor in enumerate(tensors):
                    if seg_tensor is None:
                        continue
                    seg_start = seg_points[i]
                    seg_end = seg_points[i + 1]
                    
                    # Calculate overlap between dst BH range and this segment
                    overlap_start = max(bh_start, seg_start)
                    overlap_end = min(bh_end, seg_end)
                    
                    if overlap_start < overlap_end:
                        # There is overlap, need to write to this segment
                        overlap_len = overlap_end - overlap_start
                        
                        # Source: take next overlap_len elements
                        src_slice_start = src_offset
                        src_slice_end = src_offset + overlap_len
                        
                        # Destination: offset within this segment
                        dst_bh_start = overlap_start - seg_start
                        dst_bh_end = dst_bh_start + overlap_len
                        
                        logger.debug(f"[KVCACHE_OFFLOAD] {cache_name} seg[{i}]: src_bh=[{src_slice_start}:{src_slice_end}], "
                                    f"dst_bh=[{dst_bh_start}:{dst_bh_end}]")
                        
                        # Slice source data
                        src_data = src_tt.data[:, src_slice_start:src_slice_end, :]
                        
                        # Get destination tensor data and write
                        if hasattr(seg_tensor, 'data'):
                            dst_data = seg_tensor.data[s_slice, dst_bh_start:dst_bh_end, d_slice]
                            dst_data.copy_(src_data, non_blocking=True)
                        else:
                            seg_tensor[s_slice, dst_bh_start:dst_bh_end, d_slice].copy_(src_data, non_blocking=True)
                        
                        src_offset += overlap_len
            else:
                # Non-MixedDevice: use regular general_copy
                general_copy(cache, dst_idx, src_tt, None)
        
        # Actual write with MixedDevice handling
        _write_to_cache(k_cache, k_src_tt, dst_idx, "k_cache")
        _write_to_cache(v_cache, v_src_tt, dst_idx, "v_cache")
    
    def write_pkv_cache(self, k_pkv: torch.Tensor, v_pkv: torch.Tensor, start_position: int = 0) -> None:
        assert self._active_cache_tensors_stack, "write_pkv_cache called outside of use_cache context"
        
        B, H, S, D = k_pkv.shape
        BH = B * H
        k_write = k_pkv.reshape(BH, S, D).permute(0, 2, 1)  # (B, H, S, D) -> (B*H, S, D) -> (B*H, D, S)
        v_write = v_pkv.reshape(BH, S, D)                    # (B, H, S, D) -> (B*H, S, D)
        self._write_kvs(
            kvs=(k_write, v_write),
            start_position=start_position
        )
        
    def async_write_pkv_cache(
        self, 
        k_pkv: torch.Tensor, 
        v_pkv: torch.Tensor, 
        start_position: int = 0, 
        cache_tensors: Sequence[torch.Tensor] = None,
        batch_offset: int = 0,
        full_batch_size: int = 0,
        micro_batch_size: int = 0,
    ) -> None:
        B, H, S, D = k_pkv.shape
        BH = B * H
        k_write = k_pkv.reshape(BH, S, D).permute(0, 2, 1)  # (B, H, S, D) -> (B*H, S, D) -> (B*H, D, S)
        v_write = v_pkv.reshape(BH, S, D)                    # (B, H, S, D) -> (B*H, S, D)
        self._write_kvs(
            kvs=(k_write, v_write),
            start_position=start_position,
            batch_offset=batch_offset,
            full_batch_size=full_batch_size,
            micro_batch_size=micro_batch_size,
            cache_tensors=cache_tensors
        )
        
    def update_cache_batched(
        self,
        new_kvs: AdaptedKVCache,
        kv_valid_lengths: torch.Tensor,
    ) -> None:
        """
        Batch speculative decoding 专用：每个 batch 从不同位置写入 KV cache
        """
        # 快速路径：所有 batch 的 start_position 相同
        if (kv_valid_lengths == kv_valid_lengths[0]).all():
            self._write_kvs(new_kvs, kv_valid_lengths[0].item())
            return
        
        # 慢速路径：逐 batch 写入
        assert self._active_cache_tensors_stack, "write called outside of use_cache context"
        cache_tensors = self._active_cache_tensors_stack[-1]
        (k_cache, v_cache), = cache_tensors
        S_total, BH_dst, D_dst = k_cache.shape
        
        new_kvs_data = new_kvs.kvs if hasattr(new_kvs, "kvs") else new_kvs
        key, value = new_kvs_data
        
        def _to_torch(x):
            if hasattr(x, 'device') and (
                getattr(getattr(x, 'device', None), 'device_type', None) == DeviceType.COMPRESSED
                or (hasattr(x, 'data') and isinstance(getattr(x, 'data'), tuple) and len(getattr(x, 'data')) == 3)
            ):
                return x.device.decompress(x)
            return getattr(x, 'data', x)
        
        key_t = _to_torch(key)       # (B*H, D, s_new)
        value_t = _to_torch(value)   # (B*H, s_new, D)
        
        BH_src, D_src, s_new = key_t.shape
        H = getattr(self.block_config, "num_attention_heads", None)
        B = BH_src // H
        
        if key_t.dtype != k_cache.dtype:
            key_t = key_t.to(dtype=k_cache.dtype)
        if value_t.dtype != v_cache.dtype:
            value_t = value_t.to(dtype=v_cache.dtype)
        
        # (B*H, D, s_new) -> (s_new, B*H, D)
        k_write = key_t.permute(2, 0, 1)
        v_write = value_t.permute(1, 0, 2)
        
        for i in range(B):
            start_pos = kv_valid_lengths[i].item()
            end_pos = min(start_pos + s_new, S_total)
            actual_len = end_pos - start_pos
            
            if actual_len <= 0:
                continue
            
            head_start = i * H
            head_end = (i + 1) * H
            
            # 提取第 i 个 batch 的数据并写入
            k_batch = k_write[:actual_len, head_start:head_end, :].contiguous()
            v_batch = v_write[:actual_len, head_start:head_end, :].contiguous()
            
            dst_idx = (slice(start_pos, end_pos), slice(head_start, head_end), slice(0, D_src))
            
            k_src_tt = TorchTensor.create_from_torch(k_batch, self.attention_compute)
            v_src_tt = TorchTensor.create_from_torch(v_batch, self.attention_compute)
            
            general_copy(k_cache, dst_idx, k_src_tt, None)
            general_copy(v_cache, dst_idx, v_src_tt, None)
        
    def reorder_and_write_cache(
        self,
        k_pkv: torch.Tensor,
        v_pkv: torch.Tensor,
        kv_cache_position_ids: torch.Tensor,
        cache_tensors: Sequence[torch.Tensor],
        batch_offset: int = 0,
        full_batch_size: int = 0,
        micro_batch_size: int = 0,
    ) -> Tuple[int, torch.Tensor]:
        B, H, S_old, D = k_pkv.shape
        device = k_pkv.device
        
        if kv_cache_position_ids.dim() == 1:
            kv_cache_position_ids = kv_cache_position_ids.unsqueeze(0)
        
        kv_cache_position_ids = kv_cache_position_ids.to(device)
        
        valid_mask = kv_cache_position_ids >= 0
        has_valid = valid_mask.any(dim=1)
        first_valid_idx = valid_mask.int().argmax(dim=1)
        
        batch_indices = torch.arange(B, device=device)
        root_positions = torch.where(
            has_valid,
            kv_cache_position_ids[batch_indices, first_valid_idx],
            torch.tensor(S_old, device=device)
        )
        
        tree_valid_counts = valid_mask.sum(dim=1)
        valid_lengths = root_positions + tree_valid_counts
        max_new_length = valid_lengths.max().item()
        
        k_new = torch.zeros(B, H, max_new_length, D, dtype=k_pkv.dtype, device=device)
        v_new = torch.zeros(B, H, max_new_length, D, dtype=v_pkv.dtype, device=device)
        
        max_root = root_positions.max().item()
        pos_len = kv_cache_position_ids.shape[1]
        
        # Prefix
        if max_root > 0:
            prefix_idx = torch.arange(max_root, device=device)
            batch_idx_prefix = batch_indices.unsqueeze(1).expand(B, max_root)
            prefix_idx_expanded = prefix_idx.unsqueeze(0).expand(B, max_root)
            prefix_valid = prefix_idx_expanded < root_positions.unsqueeze(1)
            
            valid_batch_prefix = batch_idx_prefix[prefix_valid]
            valid_pos_prefix = prefix_idx_expanded[prefix_valid]
            
            k_new[valid_batch_prefix, :, valid_pos_prefix, :] = k_pkv[valid_batch_prefix, :, valid_pos_prefix, :]
            v_new[valid_batch_prefix, :, valid_pos_prefix, :] = v_pkv[valid_batch_prefix, :, valid_pos_prefix, :]
        
        # Tree
        batch_idx_tree = batch_indices.unsqueeze(1).expand(B, pos_len)
        cumsum_valid = valid_mask.int().cumsum(dim=1) - 1
        dst_positions = root_positions.unsqueeze(1) + cumsum_valid
        
        valid_batch_tree = batch_idx_tree[valid_mask]
        valid_src_tree = kv_cache_position_ids[valid_mask]
        valid_dst_tree = dst_positions[valid_mask]
        
        k_new[valid_batch_tree, :, valid_dst_tree, :] = k_pkv[valid_batch_tree, :, valid_src_tree, :]
        v_new[valid_batch_tree, :, valid_dst_tree, :] = v_pkv[valid_batch_tree, :, valid_src_tree, :]
        
        self.async_write_pkv_cache(
            k_new, 
            v_new, 
            start_position=0, 
            cache_tensors=cache_tensors,
            batch_offset=batch_offset,
            full_batch_size=full_batch_size,
            micro_batch_size=micro_batch_size,
        )
        
        return max_new_length, valid_lengths
    
    def select_cache_for_reorder(
        self,
        kv_cache_position_ids: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        """
        为 reorder 准备：取出所有 batch 需要的 positions 的并集
        """
        assert self._active_cache_tensors_stack, "select_cache called outside of use_cache"
        
        cache_tensors = self._active_cache_tensors_stack[-1]
        (k_cache, v_cache), = cache_tensors
        S_full, BH, D = k_cache.shape
        
        compute_dst = self.attention_compute
        
        H = getattr(self.block_config, "num_attention_heads", None)
        B = BH // H
        
        def _as_torch(x):
            return x.data if hasattr(x, "data") else x
        
        if kv_cache_position_ids.dim() == 1:
            kv_cache_position_ids = kv_cache_position_ids.unsqueeze(0)
        
        # 找出需要的最大 position
        valid_mask = kv_cache_position_ids >= 0
        if not valid_mask.any():
            return None, None, False
        
        max_position = kv_cache_position_ids[valid_mask].max().item()
        
        # 取 [0, max_position] 范围
        prefix_length = int(max_position) + 1
        idx_all = (slice(0, prefix_length), slice(0, BH))
        
        k_sel, _ = k_cache.smart_copy(compute_dst, idx_all)
        v_sel, _ = v_cache.smart_copy(compute_dst, idx_all)
        k_sbh = _as_torch(k_sel)
        v_sbh = _as_torch(v_sel)
        
        def _to_pkv(x_sbh: torch.Tensor) -> torch.Tensor:
            return x_sbh.view(prefix_length, B, H, D).permute(1, 2, 0, 3)
        
        k_pkv = _to_pkv(k_sbh)
        v_pkv = _to_pkv(v_sbh)
        
        # 判断是否需要 reorder
        need_reorder = True  # 只要有 kv_cache_position_ids 就需要
        
        return k_pkv, v_pkv, need_reorder
    
    def select_cache_without_reorder(
        self,
        kv_cache_position_ids: torch.Tensor,  # (B, max_pos_len), -1 是 padding
        batch_offset: int = 0,
        full_batch_size: int = 0,
        micro_batch_size: int = 0,
        cache_tensors: Sequence[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """
        
        Returns:
            k_pkv: (B, H, cache_len, D)
            v_pkv: (B, H, cache_len, D)
            cache_len: 取出的 cache 长度
        """
        if cache_tensors is None:
            assert self._active_cache_tensors_stack, "select_cache called outside of use_cache"
            cache_tensors = self._active_cache_tensors_stack[-1]
        
        (k_cache, v_cache), = cache_tensors
        S_full, BH_full, D = k_cache.shape
        
        H = getattr(self.block_config, "num_attention_heads", None)
        full_batch_in_cache = BH_full // H
        
        # 1. 找到需要取的 cache 范围
        valid_mask = kv_cache_position_ids >= 0  # (B, max_pos_len)
        if not valid_mask.any():
            return None, None, 0
        
        max_position = kv_cache_position_ids[valid_mask].max().item()
        cache_len = int(max_position) + 1
        
        # 2. 计算 BH 切片范围 (与 select_cache 保持一致)
        if full_batch_size > 0 and micro_batch_size > 0:
            # Legacy 模式: cache 存储 full batch，使用 batch_offset 切片
            BH_offset_start = batch_offset * H
            BH_offset_end = BH_offset_start + micro_batch_size * H
            BH_offset_end = min(BH_offset_end, BH_full)
        else:
            # Full batch 模式
            BH_offset_start = 0
            BH_offset_end = BH_full
        
        BH = BH_offset_end - BH_offset_start
        B = BH // H
        
        # 3. 取出 [0, cache_len) 的 cache
        compute_dst = self.attention_compute
        idx_all = (slice(0, cache_len), slice(BH_offset_start, BH_offset_end))
        
        def _as_torch(x):
            return x.data if hasattr(x, "data") else x
        
        k_sel, _ = k_cache.smart_copy(compute_dst, idx_all)
        v_sel, _ = v_cache.smart_copy(compute_dst, idx_all)
        k_sbh = _as_torch(k_sel)  # (cache_len, BH, D)
        v_sbh = _as_torch(v_sel)
        
        def _to_pkv(x_sbh: torch.Tensor) -> torch.Tensor:
            return x_sbh.view(cache_len, B, H, D).permute(1, 2, 0, 3)
        
        k_pkv = _to_pkv(k_sbh)  # (B, H, cache_len, D)
        v_pkv = _to_pkv(v_sbh)
        
        return k_pkv, v_pkv, cache_len

    def update_cache_and_async_reorder(
        self,
        new_kvs: AdaptedKVCache,
        kv_cache_position_ids: Optional[torch.Tensor],  # (B, max_pos_len), -1 是 padding，可能为 None
        cache_tensors: Sequence[torch.Tensor],
        batch_offset: int = 0,
        full_batch_size: int = 0,
        micro_batch_size: int = 0,
    ) -> None:
        cache_manager = self
        
        self._reorder_executor.submit(
            self._do_reorder_task,
            new_kvs,
            kv_cache_position_ids,
            cache_tensors,
            batch_offset,
            full_batch_size,
            micro_batch_size,
            cache_manager,
        )
        
    def _do_reorder_task(
        self,
        new_kvs: AdaptedKVCache,
        kv_cache_position_ids: Optional[torch.Tensor],
        cache_tensors: Sequence[torch.Tensor],
        batch_offset: int,
        full_batch_size: int,
        micro_batch_size: int,
        cache_manager: "KVCacheManager",
    ):
        try:
            with torch.inference_mode():
                if kv_cache_position_ids is None or kv_cache_position_ids.numel() == 0:
                    self._write_kvs(
                        new_kvs, 
                        start_position=0, 
                        batch_offset=batch_offset,
                        full_batch_size=full_batch_size,
                        micro_batch_size=micro_batch_size,
                        cache_tensors=cache_tensors
                    )
                    return
                
                # ============ Generation 阶段 ============
                valid_mask = kv_cache_position_ids >= 0
                
                if not valid_mask.any():
                    self._write_kvs(
                        new_kvs, 
                        start_position=0,
                        batch_offset=batch_offset,
                        full_batch_size=full_batch_size,
                        micro_batch_size=micro_batch_size,
                        cache_tensors=cache_tensors
                    )
                    return
                
                max_position = kv_cache_position_ids[valid_mask].max().item()
                write_position = int(max_position) + 1
                
                # 1. 同步写入新 KV
                self._write_kvs(
                    new_kvs, 
                    write_position,
                    batch_offset=batch_offset,
                    full_batch_size=full_batch_size,
                    micro_batch_size=micro_batch_size,
                    cache_tensors=cache_tensors
                )
                
                # 2. 准备异步重排所需的参数
                new_kvs_data = new_kvs.kvs if hasattr(new_kvs, "kvs") else new_kvs
                key, _ = new_kvs_data
                key_data = key.data if hasattr(key, 'data') else key
                tree_len = key_data.shape[-1]
                
                device = kv_cache_position_ids.device
                B = kv_cache_position_ids.shape[0]
                
                kv_cache_position_ids_copy = kv_cache_position_ids.clone()
                
                # 构建 extended_position_ids
                new_positions = torch.arange(write_position, write_position + tree_len, device=device)
                new_positions = new_positions.unsqueeze(0).expand(B, tree_len)
                extended_position_ids = torch.cat([kv_cache_position_ids_copy, new_positions], dim=1)
                
                # 计算 cache 长度
                ext_valid_mask = extended_position_ids >= 0
                max_ext_position = extended_position_ids[ext_valid_mask].max().item()
                cache_len = int(max_ext_position) + 1
                
                # 直接调用现有的 select_cache
                k_pkv, v_pkv, _ = cache_manager.select_cache(
                    prefix_length=cache_len,
                    hypo_ids=None,
                    kv_cache_position_ids=None,
                    batch_offset=batch_offset,
                    full_batch_size=full_batch_size,
                    micro_batch_size=micro_batch_size,
                    cache_tensors=cache_tensors,
                )
                
                if k_pkv is None:
                    return
                
                # 重排并写回
                cache_manager.reorder_and_write_cache(
                    k_pkv=k_pkv,
                    v_pkv=v_pkv,
                    kv_cache_position_ids=extended_position_ids,
                    cache_tensors=cache_tensors,
                    batch_offset=batch_offset,
                    full_batch_size=full_batch_size,
                    micro_batch_size=micro_batch_size,
                )
                    
        except Exception as e:
            import logging
            logging.error(f"Async cache reorder failed: {e}")
