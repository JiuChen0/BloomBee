"""
Cache coordinator for managing cache operations across model layers
Provides unified interface for cache management and device allocation
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from bloombee.data_structures import KVCache, KVCacheMetadata, Handle, UnifiedCache, DeviceInfo
from bloombee.server.memory_cache_manager import KVCacheManager

# Create dedicated offloading debug logger
offload_logger = logging.getLogger('bloombee.offloading')
offload_logger.setLevel(logging.INFO)


def get_device_allocation_from_policy(policy, fallback_device: str = 'cuda:0') -> Tuple[str, str]:
    """
    Unified device allocation utility function
    
    Args:
        policy: Policy object containing cache configuration
        fallback_device: Default device when no policy is available
        
    Returns:
        (device_type, device_id): Device type and device ID
    """
    if policy is not None:
        # 根据policy的缓存分配比例决定设备
        if policy.cache_gpu_percent > 0:
            # 优先使用GPU
            device_type = 'gpu'
            device_id = 'cuda:0'
        elif policy.cache_cpu_percent > 0:
            # 其次使用CPU
            device_type = 'cpu'
            device_id = 'cpu'
        elif policy.cache_disk_percent > 0:
            # 最后使用磁盘
            device_type = 'disk'
            device_id = '/tmp/disk_cache'
        else:
            # 默认使用GPU
            device_type = 'gpu'
            device_id = 'cuda:0'
    else:
        # 如果没有policy，使用默认设备
        if fallback_device.startswith('cuda'):
            device_type = 'gpu'
            device_id = fallback_device
        else:
            device_type = 'cpu'
            device_id = 'cpu'
    
    return device_type, device_id


def create_device_info_from_policy(policy, fallback_device: str = 'cuda:0', 
                                 compression_config=None) -> DeviceInfo:
    """
    Create DeviceInfo object based on policy
    
    Args:
        policy: Policy object containing cache configuration
        fallback_device: Default device when no policy is available
        compression_config: Compression configuration
        
    Returns:
        DeviceInfo object
    """
    device_type, device_id = get_device_allocation_from_policy(policy, fallback_device)
    
    return DeviceInfo(
        device_type=device_type,
        device_id=device_id,
        compression_config=compression_config,
        offloaded=(device_type != 'gpu')
    )


class CacheCoordinator:
    """
    Unified cache coordinator for model layers
    Provides layer registration and cache operations with device allocation
    """
    
    def __init__(self, cache_manager: KVCacheManager):
        self.cache_manager = cache_manager
        self._layer_registry: Dict[int, Dict[str, Any]] = {}
        
        offload_logger.info("Initializing CacheCoordinator")
        offload_logger.info(f"Cache manager: {type(cache_manager).__name__}")
        if hasattr(cache_manager, 'policy') and cache_manager.policy is not None:
            policy = cache_manager.policy
            offload_logger.info(f"Policy configuration:")
            offload_logger.info(f"  - cache_gpu_percent: {policy.cache_gpu_percent}%")
            offload_logger.info(f"  - cache_cpu_percent: {policy.cache_cpu_percent}%")
            offload_logger.info(f"  - cache_disk_percent: {policy.cache_disk_percent}%")
            offload_logger.info(f"  - compress_cache: {policy.compress_cache}")
            offload_logger.info(f"  - compress_weight: {policy.compress_weight}")
            offload_logger.info(f"  - sep_layer: {policy.sep_layer}")
            offload_logger.info(f"  - cpu_cache_compute: {policy.cpu_cache_compute}")
    
    def register_layer(self, layer_id: int, layer_info: Dict[str, Any] = None):
        """Register a model layer with the coordinator"""
        if layer_info is None:
            layer_info = {}
        
        self._layer_registry[layer_id] = {
            'info': layer_info,
            'registered': True
        }
        
        offload_logger.info(f"Registered layer {layer_id} with coordinator")
    
    def unregister_layer(self, layer_id: int):
        """Unregister a model layer from the coordinator"""
        if layer_id in self._layer_registry:
            del self._layer_registry[layer_id]
            offload_logger.info(f"Unregistered layer {layer_id} from coordinator")
    
    def load_cache(self, layer_id: int, position: int, 
                  target_device: str = 'cuda:0', batch_id: int = 0) -> Optional[UnifiedCache]:
        """
        Load cache for a specific layer and position
        Delegates to KVCacheManager.load_cache()
        """
        offload_logger.info(f"CacheCoordinator.load_cache - layer:{layer_id}, position:{position}")
        
        # 直接委托给KVCacheManager
        return self.cache_manager.load_cache(position, layer_id, batch_id, target_device)
    
    def store_cache(self, layer_id: int, position: int, 
                   past_key_value: Tuple[torch.Tensor, ...],
                   device: torch.device, batch_id: int = 0) -> Optional[Handle]:
        """
        Store cache for a specific layer and position
        Delegates to KVCacheManager.store_cache() with device allocation
        """
        offload_logger.info(f"CacheCoordinator.store_cache - layer:{layer_id}, position:{position}")
        
        # 验证位置一致性 - 更智能的位置处理
        expected_position = self._get_expected_position(layer_id)
        
        # 如果是prefill阶段（position=0），不需要修正
        if position == 0:
            offload_logger.info(f"Prefill stage - position: {position}, layer: {layer_id}")
        elif position != expected_position:
            offload_logger.warning(f"Position mismatch: expected {expected_position}, got {position}")
            # 只有在非prefill阶段才修正位置
            if expected_position > 0:
                position = expected_position
                offload_logger.info(f"Position corrected to: {position}")
        
        # 创建UnifiedCache
        # 根据policy决定设备分配，但优先考虑张量的实际位置
        if hasattr(self.cache_manager, 'policy') and self.cache_manager.policy is not None:
            # 检查张量的实际位置
            if past_key_value and len(past_key_value) > 0:
                first_tensor = past_key_value[0]
                if isinstance(first_tensor, torch.Tensor):
                    actual_device = str(first_tensor.device)
                    offload_logger.info(f"Tensor actual location: {actual_device}")
                    
                    # 根据policy决定目标设备
                    target_device_type, target_device_id = get_device_allocation_from_policy(
                        self.cache_manager.policy, str(device)
                    )
                    offload_logger.info(f"Policy required target device: {target_device_id}")
                    
                    # 如果张量位置与policy要求不符，需要同步
                    if actual_device != target_device_id:
                        offload_logger.info(f"Need to sync tensor from {actual_device} to {target_device_id}")
                        
                        # 同步张量到目标设备
                        synced_tensors = []
                        for i, tensor in enumerate(past_key_value):
                            if isinstance(tensor, torch.Tensor):
                                if str(tensor.device) != target_device_id:
                                    synced_tensor = tensor.to(target_device_id, non_blocking=True)
                                    offload_logger.info(f"Syncing tensor {i}: {tensor.device} -> {synced_tensor.device}")
                                else:
                                    synced_tensor = tensor
                                    offload_logger.info(f"Tensor {i} already on target device, skipping sync")
                                synced_tensors.append(synced_tensor)
                            else:
                                synced_tensors.append(tensor)
                        
                        # 使用同步后的张量
                        past_key_value = tuple(synced_tensors)
                        actual_device = target_device_id
                        offload_logger.info(f"Sync completed, tensor now at: {actual_device}")
                    
                    # 创建设备信息
                    device_info = DeviceInfo(
                        device_type=target_device_type,
                        device_id=target_device_id,
                        compression_config=self.cache_manager.policy.comp_cache_config if self.cache_manager.policy.compress_cache else None,
                        offloaded=(target_device_type != 'gpu')
                    )
                else:
                    device_info = create_device_info_from_policy(self.cache_manager.policy, str(device))
            else:
                device_info = create_device_info_from_policy(self.cache_manager.policy, str(device))
        else:
            # 如果没有policy，使用张量的实际位置
            if past_key_value and len(past_key_value) > 0:
                first_tensor = past_key_value[0]
                if isinstance(first_tensor, torch.Tensor):
                    actual_device = str(first_tensor.device)
                    # Normalize device type
                    normalized_type = 'gpu' if actual_device.startswith('cuda') else (
                        'cpu' if actual_device == 'cpu' else actual_device
                    )
                    device_info = DeviceInfo(
                        device_type=normalized_type,
                        device_id=actual_device,
                        compression_config=None,
                        offloaded=(not actual_device.startswith('cuda'))
                    )
                else:
                    device_info = DeviceInfo(
                        device_type=('gpu' if str(device).startswith('cuda') else ('cpu' if str(device) == 'cpu' else device.type)),
                        device_id=str(device),
                        compression_config=None,
                        offloaded=(not str(device).startswith('cuda'))
                    )
            else:
                device_info = DeviceInfo(
                    device_type=('gpu' if str(device).startswith('cuda') else ('cpu' if str(device) == 'cpu' else device.type)),
                    device_id=str(device),
                    compression_config=None,
                    offloaded=(not str(device).startswith('cuda'))
                )
        
        unified_cache = UnifiedCache(
            past_key_value=past_key_value,
            device_info=device_info
        )
        
        # 直接委托给KVCacheManager
        handle = self.cache_manager.store_cache(unified_cache, position, layer_id, batch_id)
        
        # 更新层状态
        if handle is not None:
            self._update_layer_position(layer_id, position, handle)
            offload_logger.info(f"Successfully stored cache - position:{position}, layer:{layer_id}, handle:{handle}, device:{device_info.device_type} ({device_info.device_id})")
        
        return handle
    
    def update_cache(self, layer_id: int, position: int,
                    new_past_key_value: Tuple[torch.Tensor, ...],
                    device: torch.device, batch_id: int = 0) -> Optional[Handle]:
        """
        Update existing cache for a specific layer and position
        Delegates to KVCacheManager.update_cache()
        """
        offload_logger.info(f"CacheCoordinator.update_cache - layer:{layer_id}, position:{position}")
        
        # 将传入的 Tensor 元组封装为 KVCache（KVCacheManager.update_cache 期望 KVCache 类型）
        try:
            kv_cache = KVCache(
                kvs=new_past_key_value,
                device=KVCacheMetadata(device=None, offloaded=False)
            )
            return self.cache_manager.update_cache(kv_cache, position, layer_id, batch_id)
        except Exception as e:
            offload_logger.warning(f"⚠️ update_cache failed to wrap KVCache: {e}")
            return None
    
    def get_layer_info(self, layer_id: int) -> Dict[str, Any]:
        """Get layer registration information"""
        if layer_id not in self._layer_registry:
            return {}
        
        return {
            'registered': True,
            'layer_info': self._layer_registry[layer_id]['info']
        }
    
    def get_registered_layers(self) -> List[int]:
        """Get list of registered layer IDs"""
        return list(self._layer_registry.keys())
    
    def is_layer_registered(self, layer_id: int) -> bool:
        """Check if a layer is registered"""
        return layer_id in self._layer_registry
    
    def get_cache_manager(self) -> KVCacheManager:
        """Get the underlying cache manager"""
        return self.cache_manager
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from underlying cache manager"""
        return self.cache_manager.get_cache_info()
    
    def _get_expected_position(self, layer_id: int) -> int:
        """Get expected position for a layer based on current state"""
        if layer_id in self._layer_registry:
            return self._layer_registry[layer_id].get('last_position', 0) + 1
        return 0
    
    def _update_layer_position(self, layer_id: int, position: int, handle: Handle):
        """Update layer position tracking"""
        if layer_id in self._layer_registry:
            self._layer_registry[layer_id]['last_position'] = position
            self._layer_registry[layer_id]['last_handle'] = handle
            if 'cache_count' not in self._layer_registry[layer_id]:
                self._layer_registry[layer_id]['cache_count'] = 0
            self._layer_registry[layer_id]['cache_count'] += 1
            
            offload_logger.info(f"Updated layer {layer_id} position to {position} with handle {handle}")
    
    def get_layer_cache_info(self, layer_id: int) -> Dict[str, Any]:
        """Get detailed cache information for a specific layer"""
        if layer_id not in self._layer_registry:
            return {}
        
        info = self._layer_registry[layer_id].copy()
        info['expected_position'] = self._get_expected_position(layer_id)
        
        # Get available positions from cache manager
        if hasattr(self.cache_manager, '_position_tracker') and layer_id in self.cache_manager._position_tracker:
            info['available_positions'] = list(self.cache_manager._position_tracker[layer_id].keys())
        
        return info


# Global cache coordinator instance
_global_cache_coordinator: Optional[CacheCoordinator] = None


def get_cache_coordinator() -> Optional[CacheCoordinator]:
    """Get the global cache coordinator instance"""
    return _global_cache_coordinator


def set_cache_coordinator(cache_manager: KVCacheManager):
    """Set the global cache coordinator"""
    global _global_cache_coordinator
    
    _global_cache_coordinator = CacheCoordinator(cache_manager)
    
    offload_logger.info("Global cache coordinator initialized")
    offload_logger.info(f"Cache manager: {type(cache_manager).__name__}")


def clear_cache_coordinator():
    """Clear the global cache coordinator"""
    global _global_cache_coordinator
    
    _global_cache_coordinator = None
    
    offload_logger.info("Global cache coordinator cleared")


def init_layer_cache_manager(layer_id: int, policy, llama_config, env) -> Optional[CacheCoordinator]:
    """
    Initialize cache manager for a specific layer
    
    Args:
        layer_id: Layer identifier
        policy: Policy object containing cache configuration
        llama_config: LlamaConfig object
        env: Execution environment
        
    Returns:
        CacheCoordinator instance if successful, None otherwise
    """
    cache_interface = get_cache_coordinator()
    if cache_interface is not None:
        # Calculate actual number of layers based on policy.sep_layer
        num_workspaces = 1 if policy.sep_layer else 2
        
        # Register current layer to coordinator with detailed layer information
        layer_info = {
            'layer_type': 'llama_decoder',
            'policy': policy,
            'layer_id': layer_id,
            'num_workspaces': num_workspaces,
            'sep_layer': policy.sep_layer,
            'config': llama_config,
            'env': env
        }
        
        cache_interface.register_layer(layer_id, layer_info)
        offload_logger.info(f" Layer {layer_id} registered to cache coordinator")
        offload_logger.info(f"   - sep_layer: {policy.sep_layer}")
        offload_logger.info(f"   - num_workspaces: {num_workspaces}")
        offload_logger.info(f"   - gpu_batch_size: {policy.gpu_batch_size}")
        offload_logger.info(f"   - num_attention_heads: {llama_config.num_attention_heads}")
    else:
        offload_logger.warning(f" Cache coordinator unavailable, layer {layer_id} will not be able to use cache functionality")
    
    return cache_interface 