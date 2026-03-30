# Week 4: Performance Tuning & Benchmarking (Days 16-20)

## Overview: What You're Building This Week

By the end of Week 4, you'll have a **validated, benchmarked, production-ready inference engine** with:
- Measurable performance metrics (TTFT, TPOT, throughput)
- Optimization techniques applied and validated
- Comprehensive test suite
- Detailed documentation

**The Core Goal**: Prove your system is **fast, correct, and reliable**.

Think of it like building a car:
- Weeks 1-3: Assemble engine, chassis, transmission (functional)
- Week 4: Tune engine, test safety, measure performance (production-ready)

---

## Day 16: KV Cache Quantization (Optional Advanced Technique)

### The Problem: KV Cache is Memory Hungry

For a 1.1B parameter model with 150 concurrent requests:
```
KV Cache Size ≈ 150 requests × 50 tokens avg × 2 (K+V) × 2048 (hidden) × 2 bytes (FP16)
             ≈ ~150 × 50 × 2 × 2048 × 2 / 1e9 GB
             ≈ ~30 GB

But you only have 16 GB GPU! Doesn't fit!
```

### The Solution: Quantize K,V to INT8

Instead of storing K,V as **FP16** (2 bytes per value), store as **INT8** (1 byte):

```
FP16: 0.52893 (2 bytes)
INT8: 53      (1 byte, scaled)

Memory reduction: 2x!
Trade-off: 5-10% quality loss, ~10% latency overhead
```

### What You're Building (Theory)

A **Quantized KV Cache** that:

1. **Quantizes on storage**
   - Compute K,V in FP16 as normal
   - Before storing in block, quantize to INT8
   - Per-channel quantization:
     - For each attention head, compute scale factor
     - scale = max(|K|) / 127  (INT8 range is -128 to 127)
     - K_int8 = round(K / scale)
     - Store: K_int8 (1 byte) + scale (1 FP32, shared across head)

2. **Dequantizes for computation**
   - At attention time, convert INT8 → FP16
   - K_fp16 = K_int8 * scale
   - Use dequantized K,V in attention

3. **Minimal code changes**
   - KVCacheBlock stores quantized values
   - Dequantization happens on-the-fly

### Why This Matters

- **Memory savings**: 2x compression (30 GB → 15 GB!)
- **Handles more requests**: More concurrent users
- **Minimal quality loss**: 5-10% (acceptable trade-off)
- **Real-world**: Used in vLLM, TGI for large-batch serving

### Prerequisites You Need

- **Quantization basics**: How to scale floats to integers
- **Numerical precision**: FP16 vs INT8 trade-offs
- **Per-channel vs per-tensor**: Why per-head quantization?

### How You'll Know It Works (Day 16 Success Criterion)

```
You implement KV Cache Quantization:
  ✓ Quantize K,V to INT8 before storing in block
  ✓ Dequantize on-the-fly during attention
  ✓ Compare output vs. non-quantized reference
    - Difference < 1% (small quality loss)
  ✓ Memory test:
    - Non-quantized: 30 GB
    - Quantized: 15 GB (or close to 2x)
  ✓ Performance test:
    - Quantized slightly slower (~10% overhead) but uses less memory
  ✓ Can now handle 300 concurrent requests vs. 150 before!
```

### Why This Matters

Optional pero importante. Makes system handle **4x more concurrent requests** with same GPU.

---

## Day 17: Latency Profiling & Bottleneck Identification

### The Problem: System Works, But Where's Time Being Spent?

You have a working system, but is it fast? Where are the bottlenecks?

Possibilities:
```
Is attention slow? (30% time)
Is MLP slow? (40% time)
Is memory bandwidth the bottleneck? (20% time)
Is tokenization slow? (5% time)
Is I/O slow? (5% time)
```

You need to **measure** to find out.

### The Solution: Instrument with Profiling

Add timing checkpoints at key locations:

```
Tokenization
  ├─ Encode text → tokens: 0.5ms
  └─ Pad to batch size: 0.1ms
Embedding lookup: 1.2ms
Transformer block 1
  ├─ Attention: 15.3ms  ← This is slow!
  ├─ MLP: 8.2ms
  └─ LayerNorm: 0.2ms
Transformer block 2
  ├─ Attention: 15.2ms
  └─ ...
Logits & sampling: 0.3ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: ~300ms for one decode iteration
```

### What You're Building (Theory)

A **Profiler** that:

1. **Instruments key functions**
   - Time prefill, decode separately
   - Time attention, MLP, layernorm
   - Time memory operations (block allocation, eviction)

2. **Collects statistics**
   - Min/max/avg latency for each component
   - Percentage of total time

3. **Identifies bottleneck**
   - Which component takes most time?
   - Which operation is memory-bound vs. compute-bound?

4. **Produces reports**
   - Breakdown table (component: time %)
   - Suggests optimization targets

### Why This Matters

- **Data-driven optimization**: Don't guess, measure!
- **Prioritization**: Focus on biggest bottleneck first
- **Validation**: Verify optimizations actually help

### Prerequisites You Need

- **Timing APIs**: `time.perf_counter()` in Python, CUDA events
- **Identifying bottlenecks**: What's memory-bound vs. compute-bound?
- **Interpretation**: What does 15ms for attention tell you?

### How You'll Know It Works (Day 17 Success Criterion)

```
You implement profiler:
  ✓ Run inference with profiling enabled
  ✓ Collect latency for each component
  ✓ Print breakdown table:
    |Component|Prefill (ms)|Decode (ms)|% of Total|
    |---------|----------|----------|----------|
    |Attention|    200    |    15    |    40%   |
    |MLP      |    100    |    8     |    25%   |
    |...      |    ...    |   ...    |   ...    |
  ✓ Identify bottleneck (e.g., "Attention is 40% of prefill")
  ✓ Memory test: can you tell memory-bound vs. compute-bound?
```

### Why This Matters

You can now **make data-driven optimization decisions** instead of guessing.

---

## Day 18: Throughput Maximization (Batch Size Tuning)

### The Problem: What Batch Size is Optimal?

Throughput varies with batch size:
```
batch_size=1:  20 tok/sec (low utilization)
batch_size=2:  38 tok/sec
batch_size=4:  72 tok/sec
batch_size=8:  120 tok/sec  ← Maybe optimal?
batch_size=16: 180 tok/sec  ← Or this?
batch_size=32: 200 tok/sec  ← Or this?
batch_size=64: OOM (out of memory!)
```

Too many variables (GPU memory, model size, sequence length). Need to find **empirically**.

### The Solution: Throughput Sweep

Test different batch sizes, measure throughput, find peak:

```
For batch_size in [1, 2, 4, 8, 16, 32]:
  1. Set max_batch_size = batch_size
  2. Queue 100 requests (long enough to reach steady state)
  3. Measure total time
  4. Calculate throughput = 100 * max_tokens / total_time
  5. Record (batch_size, throughput)

Find batch_size with max throughput
```

### What You're Building (Theory)

A **Throughput Benchmark** that:

1. **Sweeps batch sizes**
   - Iterate: batch_size = 1, 2, 4, 8, 16, 32, ...

2. **For each batch size**
   - Queue enough requests (100+)
   - Let system run to completion
   - Measure total tokens/time

3. **Identifies optimal batch size**
   - Which batch_size gives max throughput?
   - Record with GPU memory usage

4. **Reports results**
   - Graph: batch_size vs. throughput
   - Metrics: peak throughput, optimal batch size

5. **Recommends configuration**
   - "Use batch_size=16 for peak throughput (200 tok/sec)"

### Why This Matters

- **Application-specific**: Optimal batch size depends on GPU, model, sequence length
- **No guessing**: Measure and pick best
- **Trade-offs**: Understand latency vs. throughput

### Prerequisites You Need

- **Batch size scaling**: Understand how throughput changes with batch
- **Memory/compute trade-off**: Larger batch uses more memory but better GPU utilization
- **Steady state**: Let system run long enough to stabilize

### How You'll Know It Works (Day 18 Success Criterion)

```
You implement throughput sweep:
  ✓ Test batch_sizes [1, 2, 4, 8, 16, 32]
  ✓ For each, measure throughput
  ✓ Get results like:
    |batch_size|throughput (tok/sec)|GPU mem (%)|
    |----------|------------------|-----------|
    |1         |    20            |    15     |
    |2         |    38            |    25     |
    |4         |    72            |    40     |
    |8         |   120            |    60     |
    |16        |   180            |    80     |
    |32        |   200            |    95     |
  ✓ Identify optimal: batch_size=32 at 200 tok/sec
  ✓ Configure system to use batch_size=32
```

### Why This Matters

You can now **maximize system throughput** for your specific hardware.

---

## Day 19: End-to-End Integration Testing

### The Problem: does everything work together?

You've tested individual components (block manager, attention, scheduler) but does the **entire system** work end-to-end?

Possible failure modes:
```
- Memory leak (blocks not freed, accumulate over time)
- Deadlock (request stuck, never finishes)
- Silent corruption (output is garbage, no error)
- Race condition (timing-dependent bugs)
- Cascade failure (one bug causes many downstreams)
```

### The Solution: Integration Tests

Test the **full system** with realistic workloads:

```
Test 1: Simple Prompt
  Input: "Hello, world"
  Output: Should be coherent English text (not garbage)

Test 2: Multiple Requests
  Input: 5 different prompts simultaneously
  Output: All 5 generate correctly, no interference

Test 3: Long Prompt
  Input: 1000-token prompt (stress test)
  Output: Should handle without OOM or hanging

Test 4: Preemption
  Input: Fill GPU, add more requests
  Expected: Some preempted, others complete
  Output: No crash, graceful handling

Test 5: Concurrent Requests
  Input: 50 requests submitted at random intervals
  Output: All complete eventually, no deadlock

Test 6: Memory Leak
  Input: Generate 10,000 tokens total
  Check: GPU memory stable (doesn't grow unbounded)
```

### What You're Building (Theory)

**Integration Test Suite** that:

1. **Tests basic functionality**
   - Single request generates coherent output
   - Output is not garbage/NaN/Inf

2. **Tests concurrency**
   - Multiple requests without interference
   - All complete eventually

3. **Tests stress**
   - Long sequences, many tokens
   - Doesn't OOM or hang

4. **Tests error handling**
   - Preemption works (gpu full)
   - Timeouts enforced
   - Graceful degradation

5. **Tests robustness**
   - Memory doesn't leak
   - No deadlocks
   - Metrics consistent

### Why This Matters

- **Confidence**: System actually works end-to-end
- **Regression detection**: Spot bugs when you refactor
- **Production readiness**: System is stable

### Prerequisites You Need

- **Testing patterns**: How to structure test cases
- **Debugging tools**: How to identify memory leaks, deadlocks

### How You'll Know It Works (Day 19 Success Criterion)

```
You implement integration test suite:
  ✓ Test 1 passes: Single request generates coherent text
  ✓ Test 2 passes: 5 concurrent requests work
  ✓ Test 3 passes: 1000-token prompt handled
  ✓ Test 4 passes: Preemption works when GPU full
  ✓ Test 5 passes: 50 concurrent requests, no deadlock
  ✓ Test 6 passes: Generate 10K tokens, memory stable
  ✓ All tests run without errors
  ✓ Add to CI/CD: run tests on every commit
```

### Why This Matters

You now have **confidence the system works reliably**.

---

## Day 20: Benchmarking Suite & Final Optimization

### The Problem: How Fast is Your System?

You have metrics (TTFT, TPOT, throughput) but how do they compare to **baselines**?

- Faster than HuggingFace Transformers?
- Slower than vLLM?
- Good enough for production?

### The Solution: Comprehensive Benchmark Suite

Compare your system against:
1. **Reference implementation** (HuggingFace standard inference)
2. **Naive baseline** (simple forward passes, no optimization)
3. **Your optimized system**

### What You're Building (Theory)

**Comprehensive Benchmark Suite** that:

1. **Measures TTFT (Time-to-First-Token)**
   - Time from "submit request" to "first token generated"
   - Repeat 10 times, avg across runs
   - Compare: yours vs. HF Transformers
   ```
   Your system: 250 ms
   HF baseline: 350 ms
   Speedup: 1.4x faster ✓
   ```

2. **Measures TPOT (Time-per-Output-Token)**
   - Time per token in decode phase
   - Vary batch size (1, 2, 4, 8, 16)
   - Compare: yours vs. baseline
   ```
   Your system (batch=8): 18 ms/token
   HF baseline (batch=1): 40 ms/token
   Speedup: 2.2x faster ✓
   ```

3. **Measures Throughput**
   - Tokens/second at optimal batch size
   - Compare: yours vs. baseline
   ```
   Your system: 200 tok/sec
   HF baseline: 80 tok/sec
   Speedup: 2.5x faster ✓
   ```

4. **Produces Final Report**
   - Table of metrics
   - Graphs
   - Summary: "Achieved 2-3x speedup over baseline"

5. **Identifies remaining bottlenecks**
   - Still slow in some area?
   - Ideas for further optimization

### Why This Matters

- **Proof of work**: Quantized improvements
- **Resume ammunition**: "Achieved X speedup"
- **Open questions**: What's left to optimize?

### Prerequisites You Need

- **Statistical rigor**: Run benchmarks multiple times, report variance
- **Fair comparison**: Same model, sequence length, etc.
- **Interpretation**: What's a "good" speedup?

### How You'll Know It Works (Day 20 Success Criterion)

```
You implement benchmarking suite:
  ✓ Run TTFT benchmark (10 runs, average)
    Your: 250 ms, HF: 350 ms → 1.4x faster
  
  ✓ Run TPOT benchmark (batch=1,2,4,8)
    Your (batch=8): 18 ms, HF (batch=1): 40 ms → 2.2x faster
  
  ✓ Run throughput benchmark
    Your: 200 tok/sec, HF: 80 tok/sec → 2.5x faster
  
  ✓ Generate comprehensive report:
    |Metric|Yours|HF Baseline|Speedup|
    |------|-----|-----------|-------|
    |TTFT  |250ms|350ms      |1.4x   |
    |TPOT  |18ms |40ms       |2.2x   |
    |TP    |200  |80         |2.5x   |
  
  ✓ Summary: "Achieved 2-3x speedup over HF baseline"
```

### Why This Matters

This is your **proof of achievement**. Concrete numbers that look great on a resume.

---

## End of Week 4: Your Production-Ready System

At the end of Day 20:

```
vLLM-Lite: Production-Grade LLM Inference Engine
│
├─ Core Features
│  ├─ Custom PagedAttention (scattered KV cache)
│  ├─ Continuous batching scheduler
│  ├─ Token-level request scheduling
│  ├─ Priority queue (fairness + SLA)
│  ├─ RoPE positional embeddings
│  └─ Flash Attention optimization
│
├─ Advanced Features (Week 4)
│  ├─ KV cache quantization (optional, 2x memory savings)
│  ├─ Latency profiling (identify bottlenecks)
│  ├─ Batch size optimization (peak throughput)
│  └─ Comprehensive testing suite
│
├─ Performance
│  ├─ TTFT: 250ms (1.4x faster than baseline)
│  ├─ TPOT: 18ms/token (2.2x faster)
│  ├─ Throughput: 200 tok/sec (2.5x faster)
│  ├─ Concurrent requests: 16+
│  └─ GPU utilization: 80%+
│
├─ Quality
│  ├─ All integration tests pass
│  ├─ No memory leaks
│  ├─ No deadlocks
│  └─ Graceful error handling (preemption)
│
└─ Documentation
   ├─ ROUTEMAP.md (20-day roadmap)
   ├─ ARCHITECTURAL_SPEC.md (deep dives)
   ├─ WEEK1-4.md (beginner guides)
   ├─ RESUME_IMPACT.md (interview talking points)
   └─ BENCHMARKING_GUIDE.md (measurement methodology)
```

### Performance Summary

| Metric | Week 1 | Week 2 | Week 3 | Week 4 |
|--------|--------|--------|--------|---------|
| TTFT | 3-5s | 300-400ms | 200-300ms | 250ms |
| TPOT | 1-2s | 20-30ms | 15-20ms | 18ms |
| Throughput | ~1 tok/sec | 50-80 tok/sec | 100-150 tok/sec | 200 tok/sec |
| Concurrent | 1 | 3-4 | 8-16 | 16+ |
| GPU Util | 10% | 30% | 70% | 80%+ |

**Total speedup**: ~200x from Week 1 to Week 4! 🚀

---

## Architectural Questions?

If you get stuck on:
- **Quantization details?** → See ARCHITECTURAL_SPEC.md > KV Cache Quantization
- **Profiling methodology?** → See BENCHMARKING_GUIDE.md > Profiling
- **Benchmark comparison?** → See BENCHMARKING_GUIDE.md > Comparing Against Baselines

## Prerequisites Learning Path

If you don't know these:

1. **Profiling tools** (20 mins)
   - CUDA events for GPU timing
   - Python's `cProfile` for CPU

2. **Quantization basics** (20 mins)
   - Fixed-point arithmetic
   - Scaling floats to integers

3. **Benchmark methodology** (20 mins)
   - How to run fair comparisons
   - Statistical rigor (multiple runs, average)

## Week 4 Success Criteria

By the end of Friday:
- ✅ Quantization working (optional, 2x memory savings if done)
- ✅ Latency profiler identifies bottlenecks
- ✅ Batch size optimized (peak throughput found)
- ✅ Integration tests pass (all 6 scenarios)
- ✅ Comprehensive benchmarks completed
- ✅ 2-3x speedup vs. HF baseline measured
- ✅ Documentation complete
- ✅ Code clean and well-commented

---

## Your 20-Day Journey Complete! 🎉

### What You've Built

A **complete, optimized LLM inference engine from scratch**:
- Custom PagedAttention (memory-efficient KV cache)
- Continuous batching scheduler (dynamic requests)
- RoPE embeddings (position encoding)
- Priority queue (fairness + SLA)
- Flash Attention (fast kernel)
- Comprehensive benchmarking suite

### What You've Learned

- **Systems architecture**: How production LLM serving works
- **GPU optimization**: Memory management, batching, attention kernels
- **Data structures**: Block allocation, priority queues, request states
- **Performance engineering**: Profiling, benchmarking, optimization trade-offs

### Resume Ammunition

Three killer points:

1. **"I implemented custom PagedAttention from first principles, achieving 3x memory efficiency for KV cache management"**
   - Shows: Deep understanding of attention algorithms and GPU memory

2. **"Built token-level continuous batching scheduler with dynamic request entry/exit, improving GPU utilization from 30% to 80%"**
   - Shows: Systems thinking, request scheduling, concurrency

3. **"Achieved 2-3x throughput improvement over HuggingFace baseline through combined optimizations: RoPE encoding, Flash Attention, and prefill/decode separation"**
   - Shows: Systematic optimization, measurement discipline, real performance gains

---

## Tips for Week 4 Success

1. **Don't over-optimize**
   - Week 4 is about validation and measurement, not new features
   - If something works, leave it

2. **Benchmark comprehensively**
   - Test multiple batch sizes, sequence lengths, request counts
   - Report variance (not just average)

3. **Document everything**
   - Why you made each optimization choice
   - What the trade-offs are

4. **Test thoroughly**
   - Integration tests are your safety net
   - Add tests as you find bugs

5. **Enjoy the finish line!**
   - You've built something impressive
   - Take a moment to appreciate the work

---

## Next Steps (If You Want to Continue)

Beyond Week 4:
- **Speculative decoding**: Generate multiple tokens in parallel (2-4x speedup)
- **Grouped Query Attention (GQA)**: 4x KV cache compression with minimal quality loss
- **Multi-GPU distribution**: Scale to multiple GPUs
- **LoRA on-the-fly**: Fine-tune models during serving
- **Multi-model serving**: Host multiple models simultaneously

But Day 20 is a natural stopping point with a **complete, production-ready system**. 🎊

Congratulations on finishing the 20-day vLLM-Lite project! You now understand how real LLM serving works at the systems level. That's expertise that separates you from 99% of ML engineers. 🌟
