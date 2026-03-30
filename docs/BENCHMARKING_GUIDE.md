# BENCHMARKING_GUIDE.md: How to Measure and Prove Performance

---

## Executive Summary

This guide explains how to measure the two **most important metrics** for LLM inference:

1. **TTFT (Time-to-First-Token)**: Latency from "user submits prompt" to "first output token appears"
   - **Measure**: tokenization + prefill → first decode step
   - **Target**: < 500ms for 100-token prompt
   - **User impact**: Determines perceived responsiveness

2. **TPOT (Time-per-Output-Token)**: Latency for each subsequent token generation
   - **Measure**: Decode iteration duration
   - **Target**: < 50ms per token with batch_size=8
   - **User impact**: Determines output streaming smoothness

---

## Part 1: TTFT (Time-to-First-Token) Measurement

### What TTFT Measures

```
User Input: "What is the capital of France?"
          ↓
       Tokenization (5 tokens)
          ↓
       Prefill Phase (process all 5 tokens, build KV cache)
          ↓
    Sample first output token
          ↓
    ← Timer Stops Here ← TTFT
          ↓
    Return "Paris" to user
          ↓
    (User sees output)
```

### Detailed Breakdown

TTFT spans multiple components:

```
TTFT = Tokenization_Latency
     + Prefill_Compute
     + Softmax_Overhead
     + Sampling_Latency
     + KV_Cache_Allocation
     
For 100-token prompt on 16GB GPU:
  - Tokenization: ~1ms (text→IDs)
  - Prefill compute: 300-400ms (Q,K,V matrix multiplies)
  - KV cache allocation: 5-10ms (allocating blocks)
  - Sampling: 1-2ms (argmax + sample)
  ━━━━━━━━━━━━━━━━━━━━━━━━━
  Total TTFT: ~310-410ms ✓
```

### Code: Measure TTFT

```python
import time
from typing import List

class TTFTBenchmark:
    def __init__(self, engine, tokenizer):
        self.engine = engine
        self.tokenizer = tokenizer
        self.results = []
    
    def measure_single_request(self, prompt: str) -> dict:
        """
        Measure TTFT for a single prompt
        
        Returns: {
            "prompt_length": int,
            "ttft_ms": float,
            "first_token": str,
            "throughput": float (tokens/ms)
        }
        """
        # Step 1: Tokenize
        t_start = time.perf_counter()
        token_ids = self.tokenizer.encode(prompt)
        t_tokenized = time.perf_counter()
        
        # Step 2: Send to engine (prefill phase)
        t_prefill_start = time.perf_counter()
        first_token_id = self.engine.generate_first_token(
            token_ids,
            temperature=0.7
        )
        t_prefill_end = time.perf_counter()
        
        # Step 3: Decode first token
        first_token = self.tokenizer.decode([first_token_id])
        
        # Compute metrics
        tokenization_latency = (t_tokenized - t_start) * 1000  # ms
        prefill_latency = (t_prefill_end - t_prefill_start) * 1000  # ms
        total_ttft = (t_prefill_end - t_start) * 1000  # ms
        
        return {
            "prompt": prompt,
            "prompt_tokens": len(token_ids),
            "ttft_ms": total_ttft,
            "ttft_breakdown": {
                "tokenization_ms": tokenization_latency,
                "prefill_ms": prefill_latency,
            },
            "first_token": first_token,
        }
    
    def benchmark_ttft_sweep(self, test_prompts: List[str], num_runs=5):
        """
        Run TTFT measurement across different prompt lengths
        
        test_prompts: List of prompts ['short', 'medium', 'verylong']
        num_runs: How many times to run each prompt (average the variance)
        """
        results_by_length = {}
        
        for prompt in test_prompts:
            ttft_samples = []
            
            for run in range(num_runs):
                result = self.measure_single_request(prompt)
                ttft_samples.append(result["ttft_ms"])
            
            # Compute statistics
            avg_ttft = sum(ttft_samples) / len(ttft_samples)
            min_ttft = min(ttft_samples)
            max_ttft = max(ttft_samples)
            stdev_ttft = (sum((x - avg_ttft)**2 for x in ttft_samples) / len(ttft_samples))**0.5
            
            prompt_len = len(self.tokenizer.encode(prompt))
            
            results_by_length[prompt_len] = {
                "avg_ttft_ms": avg_ttft,
                "min_ttft_ms": min_ttft,
                "max_ttft_ms": max_ttft,
                "stdev_ms": stdev_ttft,
                "prompt": prompt,
            }
            
            print(f"Prompt length: {prompt_len} tokens")
            print(f"  TTFT: {avg_ttft:.1f}ms (σ={stdev_ttft:.1f}ms, range=[{min_ttft:.1f}, {max_ttft:.1f}])")
        
        return results_by_length

# Usage
engine = InferenceEngine(model, tokenizer)
benchmark = TTFTBenchmark(engine, tokenizer)

test_prompts = [
    "Hi",  # ~2 tokens
    "What is the capital of France?",  # ~8 tokens
    "Explain quantum mechanics in detail. Include double slit experiment, superposition, entanglement, and Heisenberg uncertainty principle.",  # ~30 tokens
]

results = benchmark.benchmark_ttft_sweep(test_prompts, num_runs=5)
```

### Expected Output

```
Prompt length: 2 tokens
  TTFT: 185.3ms (σ=2.1ms, range=[182.1, 189.5])
Prompt length: 8 tokens
  TTFT: 210.7ms (σ=3.4ms, range=[205.2, 215.8])
Prompt length: 30 tokens
  TTFT: 340.2ms (σ=5.2ms, range=[333.1, 348.9])

✓ TTFT scales linearly with prompt length (as expected)
```

### Analysis: What's Good?

- **TTFT < 200ms**: Excellent (Google/OpenAI class)
- **TTFT 200-500ms**: Good (acceptable for most applications)
- **TTFT > 1000ms**: Poor (user perceives lag)

**For vLLM-Lite target**: < 500ms for 100-token prompt

---

## Part 2: TPOT (Time-per-Output-Token) Measurement

### What TPOT Measures

```
After first token generated:

Generate 2nd token: Decode iteration → 40ms
Generate 3rd token: Decode iteration → 35ms
Generate 4th token: Decode iteration → 45ms
...
Generate 100th token: Decode iteration → 38ms

TPOT = Average of (Generate 2nd, 3rd, ..., 100th)
     = ~40ms per token ✓
```

### Detailed Breakdown

TPOT is the latency of one decode iteration:

```
One Decode Iteration:
  1. Load last token ID: 1ms
  2. Embedding lookup: 2ms
  3. Forward pass (MLP only, K,V cached): 25ms
  4. Sample next token: 1ms
  5. Update KV cache: 3ms
  ━━━━━━━━━━━━━━━━━━━
  Total TPOT: ~32ms

Note: Much faster than prefill because:
- Attention only processes 1 new token (not full sequence)
- K,V cache is already computed (reused from previous iterations)
- Just need to compute Q for single token, then Q @ K^T
```

### Code: Measure TPOT

```python
class TPOTBenchmark:
    def __init__(self, engine, tokenizer):
        self.engine = engine
        self.tokenizer = tokenizer
    
    def measure_tpot_for_request(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        batch_size: int = 1
    ) -> dict:
        """
        Generate multiple tokens and measure per-token latency
        
        Returns: {
            "prompt_tokens": int,
            "output_tokens": int,
            "tpot_samples": List[float],  # Per-token latencies (ms)
            "avg_tpot_ms": float,
            "p50_tpot_ms": float,
            "p99_tpot_ms": float,
        }
        """
        # Tokenize and prefill
        token_ids = self.tokenizer.encode(prompt)
        
        # Prefill phase (we care about TPOT, not TTFT, so skip timing)
        self.engine.prefill(token_ids)
        
        # Decode phase: measure each token generation
        tpot_samples = []
        
        for step in range(max_new_tokens):
            t_start = time.perf_counter()
            
            # Generate one token
            next_token_id = self.engine.decode_one_token(batch_size=batch_size)
            
            t_end = time.perf_counter()
            
            tpot_ms = (t_end - t_start) * 1000
            tpot_samples.append(tpot_ms)
            
            # Stop if EOS
            if next_token_id == self.tokenizer.eos_token_id:
                break
        
        # Compute statistics (skip first token as it's often warmer)
        tpot_values = tpot_samples[1:]  # Skip first decode (might be outlier)
        
        if not tpot_values:
            return None
        
        avg_tpot = sum(tpot_values) / len(tpot_values)
        sorted_tpot = sorted(tpot_values)
        p50_tpot = sorted_tpot[len(sorted_tpot) // 2]
        p99_tpot = sorted_tpot[int(len(sorted_tpot) * 0.99)]
        
        return {
            "prompt_tokens": len(token_ids),
            "output_tokens": len(tpot_samples),
            "tpot_samples_ms": tpot_samples,
            "avg_tpot_ms": avg_tpot,
            "p50_tpot_ms": p50_tpot,
            "p99_tpot_ms": p99_tpot,
        }
    
    def benchmark_tpot_vs_batch_size(self, prompt: str):
        """
        Measure how TPOT scales with batch size
        Key insight: Decode is memory-bound, so larger batch can improve throughput
        but might increase per-token latency slightly
        """
        batch_sizes = [1, 2, 4, 8, 16]
        results = {}
        
        for batch_size in batch_sizes:
            result = self.measure_tpot_for_request(
                prompt=prompt,
                max_new_tokens=50,
                batch_size=batch_size
            )
            
            results[batch_size] = result
            
            print(f"Batch size: {batch_size}")
            print(f"  TPOT: {result['avg_tpot_ms']:.1f}ms (p50={result['p50_tpot_ms']:.1f}, p99={result['p99_tpot_ms']:.1f})")
        
        return results

# Usage
tokenizer = load_tokenizer("meta-llama/Llama-2-7b")
engine = InferenceEngine(model, tokenizer)
benchmark = TPOTBenchmark(engine, tokenizer)

prompt = "The future of AI is"
results = benchmark.benchmark_tpot_vs_batch_size(prompt)
```

### Expected Output

```
Batch size: 1
  TPOT: 32.4ms (p50=31.2, p99=38.5)
Batch size: 2
  TPOT: 35.1ms (p50=34.3, p99=42.1)
Batch size: 4
  TPOT: 36.8ms (p50=35.9, p99=44.2)
Batch size: 8
  TPOT: 38.2ms (p50=37.5, p99=46.3)
Batch size: 16
  TPOT: 45.3ms (p50=44.1, p99=52.8)  ← Getting memory bound

✓ Optimal batch size: 8 (good balance of throughput/latency)
```

### Analysis: What's Good?

- **TPOT < 30ms**: Excellent (very fast streaming)
- **TPOT 30-50ms**: Good (smooth streaming, feels responsive)
- **TPOT > 100ms**: Poor (users notice lag between tokens)

**For vLLM-Lite target**: < 50ms per token with batch_size=8

---

## Part 3: Throughput Measurement (Tokens/Second)

### What Throughput Measures

```
Generate 500 tokens with varying batch sizes
Measure total time elapsed
Calculate tokens / second
```

### Code: Measure Throughput

```python
class ThroughputBenchmark:
    def __init__(self, engine, tokenizer):
        self.engine = engine
        self.tokenizer = tokenizer
    
    def throughput_single_batch(self, prompt: str, batch_size: int, num_tokens=500):
        """
        Measure tokens/sec by simulating batch_size concurrent requests
        """
        token_ids = self.tokenizer.encode(prompt)
        
        # Create batch_size copies of same request
        batch = [token_ids] * batch_size
        
        t_start = time.perf_counter()
        output_ids = self.engine.generate_batch(batch, max_new_tokens=num_tokens)
        t_end = time.perf_counter()
        
        elapsed_seconds = t_end - t_start
        total_tokens_generated = batch_size * num_tokens
        
        throughput = total_tokens_generated / elapsed_seconds
        
        return {
            "batch_size": batch_size,
            "tokens_generated": total_tokens_generated,
            "elapsed_seconds": elapsed_seconds,
            "throughput_tps": throughput,
        }
    
    def benchmark_throughput_sweep(self, prompt: str):
        """
        Measure throughput at different batch sizes
        """
        batch_sizes = [1, 2, 4, 8, 16, 32]
        results = {}
        
        for batch_size in batch_sizes:
            try:
                result = self.throughput_single_batch(prompt, batch_size, num_tokens=500)
                results[batch_size] = result
                
                print(f"Batch size: {batch_size} → {result['throughput_tps']:.1f} tok/sec")
            except RuntimeError as e:
                print(f"Batch size: {batch_size} → OOM or error: {e}")
                break
        
        return results

# Usage & Visualization
benchmark = ThroughputBenchmark(engine, tokenizer)
prompt = "Explain the theory of relativity"
results = benchmark.benchmark_throughput_sweep(prompt)

# Find peak throughput
peak_batch, peak_throughput = max(
    results.items(),
    key=lambda x: x[1]["throughput_tps"]
)
print(f"\n🎯 Peak throughput: {peak_throughput['throughput_tps']:.1f} tok/sec at batch_size={peak_batch}")
```

### Expected Output

```
Batch size: 1 → 28.3 tok/sec
Batch size: 2 → 52.1 tok/sec  (nearly 2x!)
Batch size: 4 → 98.7 tok/sec
Batch size: 8 → 145.2 tok/sec ← This is good!
Batch size: 16 → 178.3 tok/sec (memory getting tight)
Batch size: 32 → OOM or error

🎯 Peak throughput: 178.3 tok/sec at batch_size=16
```

---

## Part 4: Comprehensive Benchmark Suite

### Full Benchmark Script

```python
class ComprehensiveBenchmark:
    def __init__(self, engine, tokenizer, model_name="TinyLlama"):
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.results = {}
    
    def run_all_benchmarks(self):
        """Execute complete benchmark suite"""
        
        print("\n" + "="*60)
        print(f"Benchmarking {self.model_name}")
        print("="*60)
        
        # 1. TTFT Benchmark
        print("\n[1/3] Measuring TTFT (Time-to-First-Token)...")
        ttft_bench = TTFTBenchmark(self.engine, self.tokenizer)
        test_prompts = [
            "Hi",
            "What is artificial intelligence?",
            "Write a detailed explanation of quantum computing including qubits, superposition, and error correction strategies."
        ]
        self.results["ttft"] = ttft_bench.benchmark_ttft_sweep(test_prompts, num_runs=3)
        
        # 2. TPOT Benchmark
        print("\n[2/3] Measuring TPOT (Time-per-Output-Token)...")
        tpot_bench = TPOTBenchmark(self.engine, self.tokenizer)
        self.results["tpot"] = tpot_bench.benchmark_tpot_vs_batch_size(
            "The future of artificial intelligence is"
        )
        
        # 3. Throughput Benchmark
        print("\n[3/3] Measuring Throughput...")
        tp_bench = ThroughputBenchmark(self.engine, self.tokenizer)
        self.results["throughput"] = tp_bench.benchmark_throughput_sweep(
            "Once upon a time in a land far away"
        )
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print comprehensive benchmark summary"""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        # TTFT Summary
        print("\n📊 TTFT (Time-to-First-Token) Results:")
        for prompt_len, result in sorted(self.results["ttft"].items()):
            print(f"  {prompt_len:3d} tokens: {result['avg_ttft_ms']:6.1f}ms (σ={result['stdev_ms']:5.1f})")
        
        # TPOT Summary
        print("\n📊 TPOT (Time-per-Output-Token) Results:")
        for batch_size, result in sorted(self.results["tpot"].items()):
            print(f"  Batch {batch_size:2d}: {result['avg_tpot_ms']:6.1f}ms (p99={result['p99_tpot_ms']:6.1f})")
        
        # Throughput Summary
        print("\n📊 Throughput Results:")
        for batch_size, result in sorted(self.results["throughput"].items()):
            print(f"  Batch {batch_size:2d}: {result['throughput_tps']:7.1f} tok/sec")
        
        # Final Score
        print("\n" + "="*60)
        avg_ttft = sum(r["avg_ttft_ms"] for r in self.results["ttft"].values()) / len(self.results["ttft"])
        avg_tpot = sum(r["avg_tpot_ms"] for r in self.results["tpot"].values()) / len(self.results["tpot"])
        max_throughput = max(r["throughput_tps"] for r in self.results["throughput"].values())
        
        print(f"✅ Average TTFT: {avg_ttft:.1f}ms")
        print(f"✅ Average TPOT: {avg_tpot:.1f}ms")
        print(f"✅ Peak Throughput: {max_throughput:.1f} tok/sec")
        print("="*60)
```

### Running the Benchmark

```bash
# In Python REPL
from benchmarks import ComprehensiveBenchmark

engine = InferenceEngine(model, tokenizer)
benchmark = ComprehensiveBenchmark(engine, tokenizer, "TinyLlama-1.1B")
benchmark.run_all_benchmarks()
```

---

## Part 5: Comparison Against Baselines

### Beating LibTransformers

```python
def compare_against_baseline():
    """
    Compare your vLLM-Lite against reference implementation
    """
    
    # Your implementation
    from src.inference_engine import InferenceEngine
    your_engine = InferenceEngine(model, tokenizer)
    
    # Baseline: Standard HuggingFace Transformers
    from transformers import pipeline
    baseline_generator = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B")
    
    prompts = [
        "Hello, how are you?",
        "What is machine learning?",
        # ... more prompts
    ]
    
    print(f"{'Prompt':<30} {'Your TPS':<12} {'Baseline TPS':<12} {'Speedup':<8}")
    print("-" * 65)
    
    for prompt in prompts:
        # Your implementation
        t_start = time.perf_counter()
        your_output = your_engine.generate(prompt, max_new_tokens=100)
        t_your = time.perf_counter() - t_start
        your_tps = 100 / t_your
        
        # Baseline
        t_start = time.perf_counter()
        baseline_output = baseline_generator(prompt, max_new_tokens=100)
        t_baseline = time.perf_counter() - t_start
        baseline_tps = 100 / t_baseline
        
        speedup = your_tps / baseline_tps
        
        print(f"{prompt:<30} {your_tps:<12.1f} {baseline_tps:<12.1f} {speedup:<8.2f}x")
```

### Expected Results

```
Prompt                         Your TPS     Baseline TPS  Speedup
────────────────────────────────────────────────────────────────
Hello, how are you?           145.3        45.2          3.2x
What is machine learning?     142.1        43.8          3.2x
Explain quantum mechanics      128.4        38.5          3.3x
```

---

## Part 6: Monitoring in Production

### Real-time Metrics Collection

```python
class MetricsCollector:
    def __init__(self):
        self.metrics = {
            "ttft_samples": [],
            "tpot_samples": [],
            "batch_sizes": [],
            "memory_usage": [],
        }
    
    def record_request(self, request_metrics):
        """Record metrics from completed request"""
        self.metrics["ttft_samples"].append(request_metrics.ttft_ms)
        self.metrics["tpot_samples"].extend(request_metrics.tpot_per_token)
        self.metrics["batch_sizes"].append(request_metrics.batch_size)
        self.metrics["memory_usage"].append(request_metrics.peak_memory_mb)
    
    def report(self):
        """Print running statistics"""
        if self.metrics["ttft_samples"]:
            avg_ttft = sum(self.metrics["ttft_samples"]) / len(self.metrics["ttft_samples"])
            print(f"Average TTFT: {avg_ttft:.1f}ms")
        
        if self.metrics["tpot_samples"]:
            avg_tpot = sum(self.metrics["tpot_samples"]) / len(self.metrics["tpot_samples"])
            print(f"Average TPOT: {avg_tpot:.1f}ms")
        
        if self.metrics["batch_sizes"]:
            avg_batch = sum(self.metrics["batch_sizes"]) / len(self.metrics["batch_sizes"])
            print(f"Average batch size: {avg_batch:.1f}")
```

---

## Part 7: Targets for vLLM-Lite (20-Day Project)

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| TTFT (100 tokens) | < 500ms | < 300ms |
| TPOT (batch=8) | < 50ms | < 30ms |
| Throughput | > 100 tok/sec | > 150 tok/sec |
| Concurrent requests | 4-8 | 16+ |
| Memory efficiency | < 90% GPU | < 80% GPU |
| TTFT variance | σ < 50ms | σ < 20ms |

---

## Conclusion: What to Include in Your Portfolio

When showing this project to recruiters, include:

1. ✅ Benchmark results table (TTFT, TPOT, throughput)
2. ✅ Graphs of performance vs. batch size
3. ✅ Comparison against baseline (HuggingFace)
4. ✅ Code samples of measurement harness
5. ✅ Discussion of bottlenecks found and how you fixed them

This demonstrates the systems thinking and measurement discipline that ML Platform engineers need.
