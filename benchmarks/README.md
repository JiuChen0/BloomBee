# BloomBee Batch Inference

This directory contains the batch inference implementation for BloomBee, enabling efficient processing of multiple input sequences simultaneously.

## Overview

Batch inference allows BloomBee to process multiple input sequences in parallel, significantly improving throughput and resource utilization. This implementation supports:

- **True Batch Processing**: All sequences in a batch are processed simultaneously using PyTorch's vectorized operations
- **Dynamic Batch Sizes**: Support for batch sizes from 1 to 1024+ depending on available memory
- **Memory Optimization**: Efficient KV cache allocation and management for large batches
- **Scaling Analysis**: Comprehensive benchmarking tools to measure performance across different batch sizes

## Key Components

### Client-Side Changes

- **`RemoteGenerationMixin`**: Enhanced `generate_batch()` method for batch inference
- **`InferenceSession`**: Updated to handle batch sizes and validate input dimensions
- **`_ServerInferenceSession`**: Modified to pass batch size information to servers

### Server-Side Changes

- **`TransformerConnectionHandler`**: Extracts batch size from incoming requests
- **`TransformerBackend`**: Allocates KV caches based on batch size
- **`KVCacheManager`**: Integrated memory management system for efficient cache allocation

## Usage

### Basic Batch Inference

```python
from bloombee.utils import AutoDistributedModelForCausalLM
from transformers import AutoTokenizer

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoDistributedModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    initial_peers=["/ip4/127.0.0.1/tcp/31337/p2p/PEER_ID_1", "/ip4/127.0.0.1/tcp/31338/p2p/PEER_ID_2"],
    torch_dtype=torch.float32,
    device_map="cpu"
)

# Prepare batch inputs
prompts = ["Hello, how are you today?"] * 8  # Batch size 8
inputs = tokenizer(prompts, return_tensors="pt", padding=True)

# Run batch inference
outputs = model.generate_batch(
    inputs["input_ids"],
    batch_size=8,
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True
)

# Decode results
generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

### Performance Benchmarking

Run the comprehensive scaling analysis:

```bash
cd BloomBee_integrated/benchmarks
python batch_scaling_analysis.py
```

This will test batch sizes [2, 4, 8, 16, 32, 64, 128] and generate detailed performance metrics.

## Performance Results

### Scaling Analysis (TinyLlama-1.1B-Chat-v1.0)

| Batch Size | Time (s) | Total Tokens | Throughput (tokens/sec) | Memory Δ (GB) 
|------------|----------|--------------|------------------------|---------------|
| 2          | 0.98     | 100          | 102.19                 | 0.00          |
| 4          | 0.98     | 200          | 204.03                 | -0.07         |
| 8          | 1.01     | 400          | 395.77                 | 0.09          |
| 16         | 1.26     | 800          | 637.04                 | -0.01         |
| 32         | 1.75     | 1600         | 912.78                 | -0.05         | 
| 64         | 2.70     | 3200         | 1186.51                | -0.02         |
| 128        | 5.12     | 6400         | 1251.15                | -0.08         | 

### Key Performance Insights

- **Near-Linear Scaling**: Up to batch size 8 with 96.8% efficiency
- **Strong Performance**: Maintains 55.8% efficiency through batch size 32
- **Peak Throughput**: 1,251 tokens/sec at batch size 128
- **Total Speedup**: 12.2x improvement over baseline (batch size 2)
- **Memory Efficiency**: Stable memory usage across all batch sizes

## Technical Implementation

### Batch Processing Flow

1. **Input Preparation**: Multiple prompts are tokenized and padded to the same length
2. **Session Management**: `InferenceSession` creates or reuses sessions with the correct batch size
3. **Server Communication**: Batch size is extracted from input tensors and passed to servers
4. **Cache Allocation**: KV caches are allocated based on batch size and sequence length
5. **Parallel Processing**: All sequences in the batch are processed simultaneously using `torch.bmm`
6. **Output Generation**: Results are returned as a batch of generated sequences

### Memory Management

- **KV Cache Allocation**: Dynamic allocation based on batch size and sequence length
- **Memory Chunking**: Large batches are processed in chunks to prevent OOM
- **Cache Reuse**: Efficient reuse of allocated caches across inference steps
- **Multi-Device Support**: Compatible with CPU, GPU, and mixed-device configurations

### Attention Mechanism

The batch inference leverages PyTorch's `torch.bmm` (batch matrix-matrix product) for efficient parallel attention computation:

```python
# Prefill phase
attn_weights = torch.bmm(q, k.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]
attn_output = torch.bmm(attn_weights, v)          # [batch_size, num_heads, seq_len, head_dim]

# Generation phase  
attn_weights = torch.bmm(q, k.transpose(-2, -1))  # [batch_size, num_heads, 1, seq_len]
attn_output = torch.bmm(attn_weights, v)          # [batch_size, num_heads, 1, head_dim]
```

## Configuration

### Server Configuration

```bash
# Start first worker (blocks 0-11)
python -m bloombee.cli.run_server TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --num_blocks 12 \
    --device cpu \
    --torch_dtype float32 \
    --skip_reachability_check \
    --new_swarm \
    --no_auto_relay \
    --host_maddrs /ip4/127.0.0.1/tcp/31337

# Start second worker (blocks 12-21)
python -m bloombee.cli.run_server TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --block_indices 12:22 \
    --device cpu \
    --torch_dtype float32 \
    --skip_reachability_check \
    --no_auto_relay \
    --host_maddrs /ip4/127.0.0.1/tcp/31338 \
    --initial_peers /ip4/127.0.0.1/tcp/31337/p2p/FIRST_WORKER_PEER_ID
```

### Client Configuration

```python
# Update peer IDs based on your running servers
initial_peers = [
    "/ip4/127.0.0.1/tcp/31337/p2p/FIRST_WORKER_PEER_ID",
    "/ip4/127.0.0.1/tcp/31338/p2p/SECOND_WORKER_PEER_ID"
]
```

## Troubleshooting

### Common Issues

1. **Batch Size Mismatch**: Ensure the batch size passed to `generate_batch()` matches the input tensor's first dimension
2. **Memory Constraints**: Large batch sizes may require more memory; monitor system memory usage
3. **Peer Connection**: Verify that peer IDs are correct and servers are running
4. **Model Loading**: Ensure FlexGen weights are properly downloaded and accessible

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

