#!/usr/bin/env python3
"""
BloomBee Batch Inference Scaling Analysis
Tests batch sizes 2, 4, 8, 16, 32, 64, 128 with throughput tracking
"""

import asyncio
import time
import torch
from bloombee.utils import AutoDistributedModelForCausalLM
from transformers import AutoTokenizer
import psutil
import gc
import json
import os
from datetime import datetime

async def test_batch_scaling():
    """Test batch inference scaling with various batch sizes"""
    
    print("="*80)
    print("BLOOMBEE BATCH INFERENCE SCALING ANALYSIS")
    print("="*80)
    print("Base prompt: 'Hello, how are you today?'")
    print("Batch sizes: [2, 4, 8, 16, 32, 64, 128]")
    print("Max new tokens: 50 per sequence")
    print("="*80)
    
    # Setup
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Note: Update these peer IDs based on your running servers
    model = AutoDistributedModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        initial_peers=["/ip4/127.0.0.1/tcp/31337/p2p/12D3KooWCijViM3xKtDB2r2FKptWLHmeV8Dvts3toXYDfkL7ZKZL", "/ip4/127.0.0.1/tcp/31338/p2p/12D3KooWF6j6ngLkwappreSrzX2o7MRQ1nyje9xfTpfYitu3RwpH"],
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    print("Setup complete!")
    
    # Test configuration
    base_prompt = "Hello, how are you today?"
    batch_sizes = [2, 4, 8, 16, 32, 64, 128]
    max_new_tokens = 50
    
    results = []
    
    print(f"\nStarting batch scaling analysis...")
    print(f"Base prompt: '{base_prompt}'")
    print(f"Max new tokens per sequence: {max_new_tokens}")
    print("-" * 80)
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch_size={batch_size}")
        
        try:
            # Prepare inputs
            prompts = [base_prompt] * batch_size
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cpu")
            
            # Clear memory before test
            gc.collect()
            
            # Measure system memory before
            memory_before = psutil.virtual_memory().used / (1024**3)  # GB
            
            # Run inference
            start_time = time.time()
            outputs = model.generate_batch(
                inputs["input_ids"],
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            end_time = time.time()
            
            # Measure system memory after
            memory_after = psutil.virtual_memory().used / (1024**3)  # GB
            
            # Calculate metrics
            generation_time = end_time - start_time
            total_tokens = batch_size * max_new_tokens
            throughput = total_tokens / generation_time  # tokens/second
            
            # Decode first few outputs for verification
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            result = {
                'batch_size': batch_size,
                'generation_time': generation_time,
                'total_tokens': total_tokens,
                'throughput': throughput,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_delta': memory_after - memory_before,
                'sample_output': generated_texts[0] if generated_texts else "N/A"
            }
            
            results.append(result)
            
            print(f"SUCCESS!")
            print(f"   Generation time: {generation_time:.2f}s")
            print(f"   Total tokens: {total_tokens}")
            print(f"   Throughput: {throughput:.2f} tokens/sec")
            print(f"   Memory delta: {result['memory_delta']:.2f} GB")
            print(f"   Sample output: {result['sample_output'][:100]}...")
            
        except Exception as e:
            print(f"FAILED: {str(e)}")
            result = {
                'batch_size': batch_size,
                'generation_time': None,
                'total_tokens': None,
                'throughput': None,
                'memory_before': None,
                'memory_after': None,
                'memory_delta': None,
                'sample_output': f"ERROR: {str(e)}"
            }
            results.append(result)
            
            # If we hit OOM or similar, break the loop
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                print(f"   Stopping due to memory constraints")
                break
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH SCALING ANALYSIS RESULTS")
    print("="*80)
    
    print(f"{'Batch Size':<12} {'Time (s)':<10} {'Tokens':<8} {'Throughput':<12} {'Memory Δ':<10} {'Status':<8}")
    print("-" * 80)
    
    successful_results = [r for r in results if r['throughput'] is not None]
    
    for result in results:
        if result['throughput'] is not None:
            print(f"{result['batch_size']:<12} {result['generation_time']:<10.2f} {result['total_tokens']:<8} {result['throughput']:<12.2f} {result['memory_delta']:<10.2f} {'PASS':<8}")
        else:
            print(f"{result['batch_size']:<12} {'N/A':<10} {'N/A':<8} {'N/A':<12} {'N/A':<10} {'FAIL':<8}")
    
    if successful_results:
        print("\n" + "="*80)
        print("SCALING ANALYSIS")
        print("="*80)
        
        # Calculate scaling efficiency
        baseline_throughput = successful_results[0]['throughput']
        baseline_batch_size = successful_results[0]['batch_size']
        
        print(f"Baseline (batch_size={baseline_batch_size}): {baseline_throughput:.2f} tokens/sec")
        print()
        
        for result in successful_results[1:]:
            batch_size = result['batch_size']
            throughput = result['throughput']
            
            # Calculate scaling efficiency
            theoretical_throughput = baseline_throughput * (batch_size / baseline_batch_size)
            scaling_efficiency = (throughput / theoretical_throughput) * 100
            
            print(f"Batch {batch_size:3d}: {throughput:8.2f} tokens/sec | "
                  f"Efficiency: {scaling_efficiency:5.1f}% | "
                  f"Speedup: {throughput/baseline_throughput:5.1f}x")
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"batch_scaling_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'test_config': {
                'base_prompt': base_prompt,
                'batch_sizes': batch_sizes,
                'max_new_tokens': max_new_tokens
            },
            'results': results,
            'summary': {
                'successful_tests': len(successful_results),
                'total_tests': len(results),
                'baseline_throughput': successful_results[0]['throughput'] if successful_results else None,
                'peak_throughput': max([r['throughput'] for r in successful_results]) if successful_results else None,
                'max_speedup': max([r['throughput']/successful_results[0]['throughput'] for r in successful_results[1:]]) if len(successful_results) > 1 else None
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    # Note: model.shutdown() doesn't exist in this version
    # await model.shutdown()

if __name__ == "__main__":
    asyncio.run(test_batch_scaling())
