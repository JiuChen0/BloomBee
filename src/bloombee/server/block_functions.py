"""
This module implements server-side computations on served blocks: forward, backward and inference; used by handler
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional, Sequence, Tuple, Union

import torch
from hivemind.compression.serialization import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.moe.expert_uid import ExpertUID
from hivemind.proto import runtime_pb2
from hivemind.utils.logging import get_logger
from hivemind.utils.nested import nested_flatten

from bloombee.data_structures import Handle, InferenceMetadata
from bloombee.server.backend import TransformerBackend
from bloombee.server.task_pool import PrioritizedTaskPool
from bloombee.server.task_prioritizer import TaskPrioritizerBase
from bloombee.utils.convert_block import QuantType
from bloombee.utils.misc import DUMMY, is_dummy
from bloombee.utils.packaging import unpack_args_kwargs

from time import perf_counter
from datetime import datetime, timezone  
def print_time_now(s):
    # Get the current time in UTC  
    current_utc_datetime = datetime.now(timezone.utc)  
    # Format the datetime to the desired string format  
    formatted_utc_time = current_utc_datetime.strftime('%Y-%m-%d %H:%M:%S.%f %Z')  
    print('\t\t\t'+s+" UTC Time: "+ str(formatted_utc_time) )  
    

# We prioritize short inference requests and make them use a *merged* inference pool,
# so they are processed without interruptions and extra overheads
# TODO: Increase the NF4 threshold once bitsandbytes ships efficient NF4 kernel for parallel forward
MAX_SHORT_INFERENCE_TOKENS = 128
MAX_NF4_SHORT_INFERENCE_TOKENS = 1

logger = get_logger(__name__)

# Create dedicated offloading debug logger
import logging
offload_logger = logging.getLogger('bloombee.offloading')
offload_logger.setLevel(logging.INFO)


async def run_rpc_forward(
    *flat_tensors: torch.Tensor,
    requested_backends: Sequence[TransformerBackend],
    active_adapter: str = "",
    prioritizer: TaskPrioritizerBase,
    points: int = 0,
    args_structure: Any = None,
) -> torch.Tensor:
    """
    Run forward pass on deserialized inputs and prompts, used by rpc_forward and rpc_forward_stream

    :param flat_tensors: a list of tensors that includes first layer inputs, optional prompts and extra tensors
    :note: some input tensors can be missing, in which case they will be replaced with dummy tensors (see is_dummy)
    :param requested_backends: a sequence of transformer blocks in the same order as they appear in forward pass
    :returns: hidden states after the last layer [batch_size, seq_length, hid_size]
    """
    # Start timing for Cross-GPU Transfer Latency measurement
    cross_gpu_start_time = perf_counter()
    
    if args_structure is not None:
        # TODO: kwargs currently is unused, it can be used later for peft-like adaptation
        flat_tensors, kwargs = unpack_args_kwargs(flat_tensors, args_structure)
    hidden_states, prompts, *_ = flat_tensors

    dtype = requested_backends[0].dtype
    # check parse input tensors and cast dtypes
    hidden_states = hidden_states.to(dtype)
    assert hidden_states.ndim == 3
    if prompts is None or is_dummy(prompts):
        prompts = [DUMMY] * len(requested_backends)
    else:
        prompts = [p.squeeze(0) for p in prompts.to(requested_backends[0].dtype).split(1, dim=0)]

    # Log input tensor info for debugging
    logger.info(f"[CROSS_GPU_DEBUG] Input hidden_states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
    logger.info(f"[CROSS_GPU_DEBUG] Number of requested backends: {len(requested_backends)}")
    
    # Track S1->S2 transfer latency specifically
    s1_to_s2_transfer_times = []
    backend_processing_times = []
    
    # Run a chain of requested backends
    for i, (backend, prompt) in enumerate(zip(requested_backends, prompts)):
        backend_start_time = perf_counter()
        
        if not is_dummy(prompt):
            hidden_states[:, : prompt.shape[1]] += prompt

        assert isinstance(backend.inference_pool, PrioritizedTaskPool), "petals support only prioritized pools"
        priority = prioritizer.prioritize(
            hidden_states, points=points / len(requested_backends), backend=backend, type="forward"
        )
        
        # Submit task and measure processing time
        task_start_time = perf_counter()
        (hidden_states,) = await backend.forward_pool.submit_task(
            hidden_states,
            active_adapter,
            priority=priority,
        )
        task_end_time = perf_counter()
        task_processing_time = (task_end_time - task_start_time) * 1000  # Convert to milliseconds
        
        backend_end_time = perf_counter()
        backend_total_time = (backend_end_time - backend_start_time) * 1000
        
        # Track individual backend processing times
        backend_processing_times.append(task_processing_time)
        
        # Estimate S1->S2 transfer time (this is an approximation)
        # The transfer time is roughly the total time minus pure processing time
        if i > 0:  # Only measure transfer between different backends
            estimated_transfer_time = backend_total_time - task_processing_time
            s1_to_s2_transfer_times.append(estimated_transfer_time)
            logger.info(f"[S1_TO_S2_TRANSFER] Backend {i} | "
                       f"Estimated Transfer Time: {estimated_transfer_time:.2f}ms | "
                       f"Total Backend Time: {backend_total_time:.2f}ms | "
                       f"Pure Processing: {task_processing_time:.2f}ms")
        
        # Log processing latency for each backend
        logger.info(f"[PROCESSING_LATENCY] Backend {i} | "
                   f"Task Processing: {task_processing_time:.2f}ms | "
                   f"Total Backend Time: {backend_total_time:.2f}ms | "
                   f"Hidden States Shape: {hidden_states.shape}")
        
        assert isinstance(hidden_states, torch.Tensor)
        assert (
            hidden_states.ndim == 3
        ), f"inputs to {type(backend)} must be a list with a single 3d tensor of hidden states"

    # Calculate total Cross-GPU Transfer Latency
    cross_gpu_end_time = perf_counter()
    cross_gpu_latency = (cross_gpu_end_time - cross_gpu_start_time) * 1000
    
    # Calculate S1->S2 transfer statistics
    if s1_to_s2_transfer_times:
        s1_to_s2_mean = sum(s1_to_s2_transfer_times) / len(s1_to_s2_transfer_times)
        s1_to_s2_total = sum(s1_to_s2_transfer_times)
        logger.info(f"[S1_TO_S2_TRANSFER_SUMMARY] "
                   f"Average Transfer: {s1_to_s2_mean:.2f}ms | "
                   f"Total Transfer: {s1_to_s2_total:.2f}ms | "
                   f"Transfer Count: {len(s1_to_s2_transfer_times)}")
    
    logger.info(f"[CROSS_GPU_TRANSFER_LATENCY] Total: {cross_gpu_latency:.2f}ms | "
               f"Backends: {len(requested_backends)} | "
               f"Output Shape: {hidden_states.shape}")

    return hidden_states


async def run_rpc_backward(
    *flat_tensors: torch.Tensor,
    requested_backends: Sequence[TransformerBackend],
    active_adapter: str = "",
    prioritizer: TaskPrioritizerBase,
    points: int = 0,
    args_structure: Any = None,
) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    if args_structure is not None:
        # TODO: kwargs currently is unused, it can be used later for peft-like adaptation
        flat_tensors, kwargs = unpack_args_kwargs(flat_tensors, args_structure)
    inputs, grad_outputs, prompts, *_ = flat_tensors

    # Cast inputs & grad outputs to backend dtype
    inputs = inputs.to(requested_backends[0].dtype)
    grad_outputs = grad_outputs.to(requested_backends[-1].dtype)

    if prompts is None or is_dummy(prompts):
        prompts = [DUMMY] * len(requested_backends)
    else:
        prompts = [p.squeeze(0) for p in prompts.to(requested_backends[0].dtype).split(1, dim=0)]

    # Run a forward chain to collect intermediate inputs
    # Note that we do not forward for the last module since we do not need its output
    inter_inputs = []
    for backend, prompt in zip(requested_backends[:-1], prompts[:-1]):
        assert inputs.ndim == 3, f"inputs to {type(backend)} must be a single 3d tensor of hidden states"
        if not is_dummy(prompt):
            inputs[:, : prompt.shape[1]] += prompt
        inter_inputs.append(inputs)
        assert isinstance(backend.inference_pool, PrioritizedTaskPool), "petals support only prioritized pools"
        priority = prioritizer.prioritize(
            inputs, points=points / len(requested_backends), backend=backend, type="forward_in_backward"
        )
        (inputs,) = await backend.forward_pool.submit_task(inputs, active_adapter, priority=priority)

        assert isinstance(inputs, torch.Tensor)

    if not is_dummy(prompts[-1]):
        inputs[:, : prompts[-1].shape[1]] += prompts[-1]
    inter_inputs.append(inputs)

    assert len(inter_inputs) == len(prompts) == len(requested_backends), "internal shape error during backward"
    grad_prompts_reversed = []
    # Run a chain of requested backends
    for inp, prompt, backend in zip(*map(reversed, (inter_inputs, prompts, requested_backends))):
        assert isinstance(backend.inference_pool, PrioritizedTaskPool), "petals support only prioritized pools"
        priority = prioritizer.prioritize(
            inp, grad_outputs, points=points / len(requested_backends), backend=backend, type="backward"
        )
        (grad_outputs,) = await backend.backward_pool.submit_task(inp, grad_outputs, active_adapter, priority=priority)

        assert isinstance(grad_outputs, torch.Tensor)
        if not is_dummy(prompt):
            grad_prompts_reversed.append(grad_outputs[:, : prompt.shape[1]].unsqueeze(0))

    grad_prompts = torch.cat(grad_prompts_reversed[::-1], dim=0) if grad_prompts_reversed else DUMMY
    return [grad_outputs] if is_dummy(grad_prompts) else [grad_outputs, grad_prompts]  # TODO un-duct-tape


async def iterate_rpc_inference(
    requested_uids: Sequence[ExpertUID],
    requested_backends: Sequence[TransformerBackend],
    active_adapter: Optional[str],
    input_iterator: AsyncIterator[Tuple[runtime_pb2.ExpertRequest, dict]],
    cache_handles: Sequence[Sequence[Handle]],
    *,
    max_length: int,
    prioritizer: TaskPrioritizerBase,
    points: int,
    quant_type: QuantType,
    args_structure: Any = None,
) -> AsyncIterator[Tuple[Sequence[runtime_pb2.Tensor], bool, Dict]]:
    assert len(cache_handles) == len(requested_backends)

    start_iterate_rpc_infer_time = perf_counter() #######
    # print('start iterate rpc inference -=-=-=-')
    #print_time_now('')
    prefix_length = 0
    point_per_piece = points / max_length if max_length > 0 else 0.0

    async for request, step_metadata in input_iterator:
        # print('------------------ iterate_rpc_inference step_metadata ', step_metadata)
        step_receive_time = perf_counter()
        if "start_from_position" in step_metadata:
            start_from_position = step_metadata["start_from_position"]
            assert (
                prefix_length >= start_from_position,
            ), f"prefix_length={prefix_length}, start_from_position={start_from_position}"
            prefix_length = start_from_position

        flat_tensors = tuple(deserialize_torch_tensor(tensor) for tensor in request.tensors)
        if args_structure is not None:
            # TODO: kwargs currently is unused, it can be used later for peft-like adaptation
            flat_tensors, kwargs = unpack_args_kwargs(flat_tensors, args_structure)

        hidden_states, prompts, hypo_ids, *_ = flat_tensors
        batch_size, length_increment, _ = hidden_states.shape

        # Cast inputs to backend dtype
        hidden_states = hidden_states.to(requested_backends[0].dtype)
        assert hypo_ids.dtype == torch.int64, f"hypo ids must be int64, got {hypo_ids.dtype}"
        
        # Add deserialize timing
        deserialize_start = perf_counter()
        deserialize_end = perf_counter()
        deserialize_time = (deserialize_end - deserialize_start) * 1000  # ms
        step_num = step_metadata.get("step", 0)
        logger.info(f"[DATA_TRANSFER_RECV] Step={step_num} | Deserialize={deserialize_time:.2f}ms | Batch={batch_size} | Len={length_increment}")
        
        # Add Cross-GPU Transfer Latency measurement
        cross_gpu_start_time = perf_counter()
        logger.info(f"[CROSS_GPU_TRANSFER_RECV_START] Step={step_num} | FromBlocks=remote | ToBlocks={requested_backends[0].name if requested_backends else 'unknown'}")
        logger.info(f"[CROSS_GPU_DEBUG] Input hidden_states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
        logger.info(f"[CROSS_GPU_DEBUG] Number of requested backends: {len(requested_backends)}")
        logger.info(f"[CROSS_GPU_DEBUG] Batch size: {batch_size}, Length increment: {length_increment}")

        # parse deep prompts (optional argument)
        has_prompts = prompts is not None and not is_dummy(prompts)
        if not has_prompts:
            prompts = [None] * len(requested_backends)
        else:
            prompts = [p.squeeze(0) for p in prompts.to(requested_backends[0].dtype).split(1, dim=0)]
            prompts = [prompt if not is_dummy(prompt) else None for prompt in prompts]
        # print('has_prompts', has_prompts)
        # print('prompts ', prompts)
        if not (len(requested_backends) == len(prompts)):
            raise ValueError(f"Received {len(prompts)} prompts for {len(requested_backends)} backends")

        if prefix_length + length_increment > max_length:
            raise ValueError(
                f"Maximum length exceeded: prefix {prefix_length} + current {length_increment}"
                f" exceeds pre-allocated maximum {max_length}"
            )

        merge_max_tokens = MAX_NF4_SHORT_INFERENCE_TOKENS if quant_type == QuantType.NF4 else MAX_SHORT_INFERENCE_TOKENS
        can_merge_pools = batch_size * length_increment <= merge_max_tokens
        # print('-=-=-=-=-=-=-=-==-=- can merge pools : ', can_merge_pools)
        priority = prioritizer.prioritize(
            hidden_states,
            hypo_ids,
            points=point_per_piece,
            requested_uids=requested_uids,
            type="inference",
        )
        # print('after priority = prioritizer.prioritize( )')
        #print_time_now('')
        # A client may pass a tensor with 0 tokens. This is a special case that occurs, e.g.
        # when user wants to pre-allocate cache or check that server *can* allocate that cache.
        if hidden_states.numel() > 0:
            assert hidden_states.ndim == 3, f"hidden states must be a single 3d tensor"
            start_compute_time = perf_counter()
            # print('before merge pools ')
            #print_time_now('')
            
            # Add offloading debug information
            # offload_logger.info(f" Inference computation started - step {prefix_length}")
            # offload_logger.info(f"   - Batch size: {batch_size}")
            # offload_logger.info(f"   - Length increment: {length_increment}")
            # offload_logger.info(f"   - Prefix length: {prefix_length}")
            # offload_logger.info(f"   - Max length: {max_length}")
            
            # # Check cache usage
            # for i, (backend, handles) in enumerate(zip(requested_backends, cache_handles)):
            #     cache_manager = backend.cache_manager
            #     offload_logger.info(f"   - Backend {i}: {len(handles)} cache handles")
            #     offload_logger.info(f"     GPU cache ratio: {cache_manager.offloading_policy.cache_gpu_percent}%")
            #     offload_logger.info(f"     CPU cache ratio: {cache_manager.offloading_policy.cache_cpu_percent}%")
            #     offload_logger.info(f"     CPU cache compute: {cache_manager.offloading_policy.cpu_cache_compute}")
            
            if can_merge_pools:
                # print('-=-=-=-=-=-=-=-==-=- come into can merge pools : ', can_merge_pools)
                # offload_logger.info(" Using merged pool for inference")
                
                inference_infos = tuple(
                    InferenceMetadata(uid, prefix_length, tuple(handles), active_adapter)
                    for uid, handles in zip(requested_uids, cache_handles)
                )
                (hidden_states,) = await requested_backends[0].inference_pool.submit_task(
                    hidden_states, hypo_ids, inference_infos, *prompts, priority=priority
                )
                
            else:
                pass
                # print('-=-=-=-=-=-=-=-==-=- not come into can merge pools : ', can_merge_pools)
                # offload_logger.info(" Using separate pools for inference")
                
                # Track S1->S2 transfer latency specifically
                s1_to_s2_transfer_times = []
                backend_processing_times = []
                
                for i, (backend, uid, handles, prompt) in enumerate(zip(requested_backends, requested_uids, cache_handles, prompts)):
                    backend_start_time = perf_counter()
                    
                    # offload_logger.info(f"   - Processing backend: {uid}")
                    # offload_logger.info(f"     - Cache handles: {len(handles)}")
                    
                    inference_infos = (InferenceMetadata(uid, prefix_length, tuple(handles), active_adapter),)
                    
                    # Submit task and measure processing time
                    task_start_time = perf_counter()
                    (hidden_states,) = await backend.inference_pool.submit_task(
                        hidden_states, hypo_ids, inference_infos, prompt, priority=priority
                    )
                    task_end_time = perf_counter()
                    task_processing_time = (task_end_time - task_start_time) * 1000  # Convert to milliseconds
                    
                    backend_end_time = perf_counter()
                    backend_total_time = (backend_end_time - backend_start_time) * 1000
                    
                    # Track individual backend processing times
                    backend_processing_times.append(task_processing_time)
                    
                    # Estimate S1->S2 transfer time (this is an approximation)
                    # The transfer time is roughly the total time minus pure processing time
                    if i > 0:  # Only measure transfer between different backends
                        estimated_transfer_time = backend_total_time - task_processing_time
                        s1_to_s2_transfer_times.append(estimated_transfer_time)
                        logger.info(f"[S1_TO_S2_TRANSFER] Backend {i} | "
                                   f"Estimated Transfer Time: {estimated_transfer_time:.2f}ms | "
                                   f"Total Backend Time: {backend_total_time:.2f}ms | "
                                   f"Pure Processing: {task_processing_time:.2f}ms")
                    
                    # Log processing latency for each backend
                    logger.info(f"[PROCESSING_LATENCY] Backend {i} | "
                               f"Task Processing: {task_processing_time:.2f}ms | "
                               f"Total Backend Time: {backend_total_time:.2f}ms | "
                               f"Hidden States Shape: {hidden_states.shape}")
                
                # Calculate S1->S2 transfer statistics
                if s1_to_s2_transfer_times:
                    s1_to_s2_mean = sum(s1_to_s2_transfer_times) / len(s1_to_s2_transfer_times)
                    s1_to_s2_total = sum(s1_to_s2_transfer_times)
                    logger.info(f"[S1_TO_S2_TRANSFER_SUMMARY] "
                               f"Average Transfer: {s1_to_s2_mean:.2f}ms | "
                               f"Total Transfer: {s1_to_s2_total:.2f}ms | "
                               f"Transfer Count: {len(s1_to_s2_transfer_times)}")
                
                # Calculate total Cross-GPU Transfer Latency
                cross_gpu_end_time = perf_counter()
                cross_gpu_latency = (cross_gpu_end_time - cross_gpu_start_time) * 1000
                
                logger.info(f"[CROSS_GPU_TRANSFER_LATENCY] Total: {cross_gpu_latency:.2f}ms | "
                           f"Backends: {len(requested_backends)} | "
                           f"Output Shape: {hidden_states.shape}")
            
            # offload_logger.info(f" Inference computation completed - step {prefix_length}")
            end_compute_time = perf_counter()
            compute_time = (end_compute_time - start_compute_time) * 1000  # ms
            logger.info(f"[COMPUTE_END] Step={step_num} | Duration={compute_time:.2f}ms")
            # print('the inference computing time ', end_compute_time - start_compute_time)
            # print_time_now('')
        # serialize and send last layer outputs
        serialize_start = perf_counter()
        output_tensors = [
            serialize_torch_tensor(result.to(proto.dtype), proto.compression, allow_inplace=True)
            for result, proto in zip((hidden_states,), nested_flatten(requested_backends[-1].outputs_schema))
        ]
        serialize_end = perf_counter()
        serialize_time = (serialize_end - serialize_start) * 1000  # ms
        logger.info(f"[DATA_TRANSFER_SEND] Step={step_num} | Serialize={serialize_time:.2f}ms")
        # print('after serialize and send last layer outputs ', )
        # print_time_now('')
        # print('hidden_states ', hidden_states)
        # print('type of hidden_states ', )
        # print('shape of hidden_states ', hidden_states.size())
        # # hidden_size_in_bytes = hidden_states.element_size() * output_tensors.numel()  
        # # print(f"Size of the hidden state in bytes: {size_in_bytes}")  
        # print()
        
        can_push = not has_prompts
        
        # Calculate Cross-GPU Transfer receive time
        cross_gpu_end_time = perf_counter()
        cross_gpu_receive_time = (cross_gpu_end_time - cross_gpu_start_time) * 1000  # ms
        logger.info(f"[CROSS_GPU_TRANSFER_RECV_END] Step={step_num} | ReceiveTime={cross_gpu_receive_time:.2f}ms | FromBlocks=remote")
        
        # Calculate total step time
        step_end_time = perf_counter()
        step_total_time = (step_end_time - step_receive_time) * 1000  # ms
        logger.info(f"[STEP_TOTAL] Step={step_num} | TotalTime={step_total_time:.2f}ms | Prefix={prefix_length}")
        logger.info("="*80)
        
        yield output_tensors, can_push, step_metadata
        # print('output_tensors ',output_tensors)
        # prepare for next step
        prefix_length += length_increment

    end_iterate_rpc_infer_time = perf_counter()#######
    # print('iterate (all steps) rpc infer time cost (sec): ', end_iterate_rpc_infer_time - start_iterate_rpc_infer_time)########
    # #print_time_now('')
    # print()