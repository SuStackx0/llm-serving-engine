# Week 3: Continuous Batching & Scheduling (Days 11-15)

## Overview: What You're Building This Week

By the end of Week 3, you'll have a **production-grade continuous batching scheduler** that:
- Automatically manages multiple requests
- Requests enter/exit batch **every iteration** (not pre-planned)
- Achieves high GPU utilization (70%+)
- Handles priority and fairness

**The Core Problem You're Solving**: Week 2 required **static batches** (fixed set of requests). Week 3 enables **dynamic batches** (requests join/leave continuously).

Think of it like a movie theater:
- Week 2: Buy tickets beforehand, everyone watches one movie (static)
- Week 3: Drop-in queue, start movie when enough people, people leave mid-movie (continuous)

---

## Day 11: Token-Level Iteration Scheduler

### The Problem: Static Batching is Inefficient

In Week 2, you'd manually decide:
```
"I have 4 waiting requests, batch them"
Batch = [Req0, Req1, Req2, Req3]

Now process token 1 for all 4
Then token 2 for all 4
Then token 3 for all 4
...
But Req0 finishes at token 100
Req1 finishes at token 200
Req2 finishes at token 300
Req3 finishes at token 400

Problem: Req0 finishes but GPU keeps processing others
GPU can't take new work because batch is "locked"
Inefficient!
```

### The Solution: Continuous Batching

Instead of locking a batch, refresh it **every iteration**:

```
Iteration 0:
  Batch = [Req0, Req1, Req2, Req3]
  All generate token 1
  
Iteration 1:
  Req0 finishes!
  Remove Req0 from batch
  Req4 was waiting, add it
  Batch = [Req1, Req2, Req3, Req4]
  All generate token 2 (from their perspective)
  
Iteration 2:
  Req1 finishes!
  Req5, Req6 waiting, add them
  Batch = [Req2, Req3, Req4, Req5, Req6]
  3 iterations, GPU never idle!
```

### What You're Building (Theory)

A **Token-Level Iteration Scheduler** that:

1. **Maintains waiting queue**
   - Requests waiting to start generation
   - Queue is FIFO (or priority-based, Week 3 Day 13)

2. **Builds batch at each iteration**
   - Remove finished requests
   - Add waiting requests if space
   - Limit batch size (e.g., max 8 requests)

3. **Executes batch**
   - All requests generate one token
   - Update request state (tokens generated, blocks used)

4. **Single iteration = ONE output token for all in batch**
   - Not "generate all tokens for one request then next request"
   - "Generate one token for each request in batch, then repeat"

### Why This Matters

- **GPU utilization**: GPU doesn't idle waiting for slow requests
- **Fairness**: Fast requests don't gate slower ones
- **Real-world**: This is how vLLM, TGI, SGLang work!

### Prerequisites You Need

- **Request lifecycle**: What states does a request go through?
  - WAITING (in queue)
  - PREFILL (processing prompt)
  - DECODE (generating tokens)
  - FINISHED (done)

### How You'll Know It Works (Day 11 Success Criterion)

```
You implement ContinuousBatchingScheduler:
  ✓ Iteration 0: Batch all waiting requests
  ✓ Iteration 1: Remove finished, add new waiting
  ✓ Iterate 100 times: batch composition changes dynamically
  ✓ Performance test:
    - 4 requests with max_tokens=[100, 200, 300, 400]
    - Traditional batching: 400 iterations (wait for slowest)
    - Continuous: ~400 iterations but GPU always full (better utilization)
  ✓ No requests dropped or duplicated
```

### Why This Matters

This is the **scheduler foundation**. Without it, you can't achieve good throughput.

---

## Day 12: Prefill vs. Decode Phase Optimization

### The Problem: Different Phases Have Different Needs

You introduced RoPE and KV cache, but prefill and decode are fundamentally different:

**Prefill Phase** (process entire prompt):
```
Input: "Explain quantum mechanics in 3 sentences"
Tokens: 10 tokens
Compute Q,K,V for all 10 tokens
Attention: Query each token against all 10 tokens
KV cache: Store K,V for all 10 tokens
Latency: ~100-200ms

Characteristic: **High compute**, benefits from **large batch**
```

**Decode Phase** (generate one token at a time):
```
Input: Last token (token 11, a single token)
Compute Q for just this token
Attention: Query this token against cached K,V from tokens 1-10
No new K,V computed (use cached!)
Latency: ~5-10ms per token

Characteristic: **Low compute**, **memory bound**, benefits from **small batch**
```

Why? In decode:
- Attention is O(cached_seq_len) not O(seq_len²)
- Bottleneck is loading K,V from memory, not computation
- Larger batch doesn't help much, just uses more memory

### The Solution: Separate Batches for Prefill and Decode

```
Prefill Queue:
  New requests waiting for their first forward pass
  Preferred batch size: 32 (high compute)
  Example: [Req0, Req1, Req2] (3 requests with prompts of varying length)

Decode Queue:
  Requests mid-generation (in decode phase)
  Preferred batch size: 8 (memory bound)
  Example: [Req3, Req4, Req5, Req6, Req7, Req8] (6 requests generating tokens)

Each iteration:
  1. Process prefill batch (if any)
  2. Process decode batch (if any)
  3. Move requests from prefill → decode after 1 iteration
```

### What You're Building (Theory)

**Phase-Aware Scheduler** that:

1. **Separates request queues**
   - Prefill queue: new requests, haven't started yet
   - Decode queue: active requests, generating tokens

2. **Different batch size limits**
   - Prefill batch size: 64 (can be large)
   - Decode batch size: 8 (smaller to save memory)

3. **Handles transition**
   - After prefill iteration, move request to decode queue
   - Request switches from "processing prompt" to "generating output"

4. **Prioritizes decode**
   - Decode queries use cached K,V (fast!)
   - Prefill queries need to compute K,V (slow!)
   - Treat decode as higher priority

### Why This Matters

- **Memory efficiency**: Don't mix expensive prefill with cheap decode
- **Throughput**: Prefill and decode use GPU differently
- **Real-world**: Production systems do this

### Prerequisites You Need

- **Prefill vs decode characteristics**: Which is compute-bound, which is memory-bound?
- **Why smaller batch in decode?** (Memory bandwidth limit, not compute)

### How You'll Know It Works (Day 12 Success Criterion)

```
You implement phase-aware scheduler:
  ✓ Prefill queue and decode queue separate
  ✓ New requests go to prefill
  ✓ After prefill iteration, move to decode
  ✓ Batch sizes respected: prefill<=64, decode<=8
  ✓ Performance: throughput improves (better memory utilization)
  ✓ Concurrent requests: can handle more in decode (memory efficient)
```

### Why This Matters

You can now **optimize for two different phases** instead of one-size-fits-all.

---

## Day 13: Request Queue & Priority Handling

### The Problem: First-Come-First-Served is Unfair

In simple FIFO:
```
Request 0: High priority (VIP user) arrives
Request 1: Low priority (free user) arrives
Request 2: Low priority (free user) arrives

Queue = [Req0 (high), Req1 (low), Req2 (low)]
If batch size = 1:
  Process Req0: 300ms
  Process Req1: 300ms  ← VIP waits 300ms behind low-priority!
  Process Req2: 300ms
  
Unfair!
```

### The Solution: Priority Queue

Use a heap to order by priority:
```
Queue (priority-aware):
- (priority=-1, arrival_time=t0, Req0)  ← negative so min-heap returns highest priority
- (priority=-2, arrival_time=t1, Req1)
- (priority=-2, arrival_time=t2, Req2)

Process order:
  1. Req0 (higher priority, newer)
  2. Req1 (lower priority, older)
  3. Req2 (lower priority, newer)
```

### What You're Building (Theory)

A **Priority Queue Scheduler** that:

1. **Assigns priority levels**
   - HIGH (0): SLA-critical, VIP users
   - MEDIUM (1): Standard requests
   - LOW (2): Best-effort, background

2. **Orders by priority first, then FIFO**
   - Within same priority, older request goes first
   - Prevents starvation (low-priority eventually processed)

3. **Implements fairness**
   - Even if HIGH queue full, process some MEDIUM/LOW
   - After serving 3 HIGH, serve 1 MEDIUM, etc.
   - Prevents indefinite starvation

4. **Supports dynamic priority**
   - Request's priority can change (e.g., timeout approaching)
   - Adjust priority mid-generation

### Why This Matters

- **SLA compliance**: Meet latency targets for VIP users
- **Fairness**: Regular users don't starve
- **Real-world**: Production systems have SLAs

### Prerequisites You Need

- **Heap data structure** (min-heap for priority)
- **SLA (Service Level Agreement)**: Different users have different requirements
- **Fairness vs priority trade-off**: Can't always satisfy both

### How You'll Know It Works (Day 13 Success Criterion)

```
You implement priority queue scheduler:
  ✓ Add 10 requests with mixed priorities (high, medium, low)
  ✓ Process order respects priority
  ✓ Higher priority processed before lower (most of the time)
  ✓ FIFO within same priority: older before newer
  ✓ Starvation test: low-priority requests eventually process
  ✓ Fairness: low-priority not ignored indefinitely
```

### Why This Matters

You can now **serve different users fairly** based on SLA requirements.

---

## Day 14: Batched Token Generation Loop

### The Problem: Coordinating Everything

You've built:
- Physical blocks (Week 2)
- Paged attention (Week 2)
- Continuous batching (Day 11)
- Priority queue (Day 13)

Now integrate all into ONE **main inference loop**.

### What You're Building (Theory)

A **Token Generation Loop** that:

1. **Selects next batch**
   - Poll prefill queue (priority-based)
   - Poll decode queue (priority-based)
   - Respect batch size limits

2. **Executes prefill** (if any prefill requests)
   - Process all prompt tokens for selected requests
   - Initialize their KV caches
   - Mark for next decode iteration

3. **Executes decode** (if any decode requests)
   - Use cached K,V from prefill
   - Generate one token per request
   - Update KV cache (allocate block if needed)

4. **Updates request state**
   - Increment token counters
   - Check for EOS or max_length
   - Remove finished requests

5. **Handles preemption** (from Week 2 Day 9)
   - If GPU full, preempt low-priority request
   - Free its blocks
   - Add high-priority request instead

6. **Repeats**
   - Next iteration: select new batch, execute

### Why This Matters

This is the **main loop** of your inference engine. Everything works together.

### Prerequisites You Need

- **Request lifecycle**: WAITING → PREFILL → DECODE → FINISHED
- **State machine**: Transitions between states

### How You'll Know It Works (Day 14 Success Criterion)

```
You implement main token generation loop:
  ✓ Queue 5 requests, let system run
  ✓ Batch composition changes dynamically
  ✓ Prefill requests transition to decode
  ✓ Finished requests removed from system
  ✓ All 5 requests eventually complete (no deadlock)
  ✓ Total tokens generated = sum of max_tokens for all requests
  ✓ Output quality: generated text is coherent (not garbage)
  ✓ No memory leaks: blocks freed after request finishes
```

### Why This Matters

You now have a **complete inference system** that handles multiple requests end-to-end!

---

## Day 15: Memory-Efficient Attention (FlashAttention-style)

### The Problem: Attention is Slow and Memory-Hungry

Standard attention:
```
scores = Q @ K^T                 # [query, seq_len] matrix
         Memory: O(seq_len²)     # For seq_len=2048 → 4M entries
attn_probs = softmax(scores)      # Allocate 4M entries
output = attn_probs @ V                   # Read 4M entries
```

For seq_len=2048, this scores matrix is **huge** (4M × 2 bytes = 8 MB per head!)

### The Solution: Tiled Computation (Flash Attention)

Instead of materializing full scores matrix:
```
output = 0
for each block of K,V:
    scores_block = Q @ K_block^T    # Smaller matrix
    probs_block = softmax(scores_block)
    output += probs_block @ V_block

# Tricks:
- Use log-sum-exp to compute softmax incrementally
- Keep scores_block small (fits in GPU cache)
- Reuse GPU cache efficiently
```

Benefits:
- **Memory**: O(seq_len) instead of O(seq_len²)
- **Speed**: 2-4x faster (better cache locality)
- **Accuracy**: Identical to standard attention (just reordered computation)

### What You're Building (Theory)

**Flash Attention Forward Pass** that:

1. **Iterates over K,V blocks**
   - Process one K,V block at a time
   - Keep small, fits in GPU cache (SRAM)

2. **Computes partial attention**
   - Q @ K_block^T (small, fast)
   - Softmax per-block
   - Accumulate output

3. **Handles numerical stability**
   - Log-sum-exp trick (avoid overflow in softmax)
   - Track running max/sum for numerical stability

4. **Returns same result as standard**
   - Bit-for-bit identical (within numerical precision)

### Why This Matters

- **Speed**: Significantly faster attention (2-4x)
- **Memory**: Can handle longer sequences
- **Real-world**: Flash Attention is in all modern LLMs

### Prerequisites You Need

- **Log-sum-exp trick**: softmax(x) = softmax(x - max(x)) for stability
- **GPU cache behavior**: SRAM vs. HBM (high-bandwidth memory)
- **Numerical stability**: Avoiding overflow/underflow

### How You'll Know It Works (Day 15 Success Criterion)

```
You implement Flash Attention:
  ✓ Compute standard attention (dense)
  ✓ Compute flash attention (tiled)
  ✓ Compare outputs: match within 1e-4
  ✓ Benchmark:
    - seq_len=1024: flash should be 2x faster
    - seq_len=4096: flash should be 3-4x faster
  ✓ Memory: flash uses O(seq_len) not O(seq_len²)
  ✓ Works with variable seq_len
```

### Why This Matters

You can now **accelerate the most expensive operation** in transformers!

---

## End of Week 3: What Your System Looks Like

At the end of Day 15:

```
Incoming Requests
    ↓
Request Queue (Priority-based)
    ├─ Prefill Queue (batch≤64)
    └─ Decode Queue (batch≤8)
           ↓
    Iteration Scheduler
    (Refresh batch every iteration)
           ↓
    Prefill Batch Execution OR Decode Batch Execution
    (Whichever has higher priority + more requests)
           ↓
    Batched Paged Attention (Flash variant)
    (2-4x faster, O(seq_len) memory)
           ↓
    Token Sampling
           ↓
    Update Request State
    (Move prefill→decode, mark finished, free blocks)
           ↓
    Check for Preemption
    (If GPU full, stop low-priority request)
           ↓
    Output Tokens
           ↓
    Repeat
```

### Performance at End of Week 3

**Excellent!**
- **TTFT**: ~200-300ms (still improving)
  - Optimized prefill + decode phases
- **TPOT**: ~15-20ms per token (good!)
  - Flash Attention helps
- **Throughput**: ~100-150 tokens/sec (solid!)
  - Continuous batching improves GPU utilization
- **Concurrent requests**: Can handle 8-16 (significant improvement!)
  - Continuous batching + prefill/decode separation
- **GPU utilization**: 70-80% (vs. 30-40% Week 2)

### What's Still Missing (You'll Add in Week 4)

- **Quantization**: Store KV in INT8 (2x memory savings)
- **Profiling**: Identify bottlenecks
- **Benchmarking**: Measure against baselines
- **Fine-tuning**: Optimize batch sizes for your GPU

---

## Architectural Questions?

If you get stuck on:
- **How does continuous batching work?** → See ARCHITECTURAL_SPEC.md > Continuous Batching
- **Why separate prefill/decode?** → See PERFORMANCE_OPTIMIZATION.md
- **Priority queue implementation?** → See ARCHITECTURAL_SPEC.md > Request Queue

## Prerequisites Learning Path

If you don't know these:

1. **Heap data structure** (30 mins)
   - Min-heap for priority queue
   - Heapq in Python

2. **Request state machines** (20 mins)
   - Understand transitions: WAITING → PREFILL → DECODE → FINISHED
   - Why each state exists

3. **Log-sum-exp trick** (20 mins)
   - Read: numerically stable softmax
   - Why: avoid NaN/Inf in exponentiation

## Week 3 Success Criteria

By the end of Friday:
- ✅ Continuous batching scheduler working
- ✅ Requests enter/exit batch dynamically
- ✅ Prefill/decode phases optimized separately
- ✅ Priority queue ensures fairness
- ✅ Main token generation loop complete
- ✅ Flash Attention integrated (2x faster)
- ✅ Throughput 100-150 tok/sec
- ✅ Can handle 8-16 concurrent requests

---

## Week 3 → Ready for Week 4

Week 4 is about **optimization and validation**:

**Problem**: System works, but is it as fast as possible?  
**Solution**: **Profile, measure, optimize** (next week)

But you need Week 3 working first. Don't skip it!

---

## Tips for Week 3

1. **Test continuous batching with delays**
   - Submit requests at intervals (not all at once)
   - Verify batch composition changes over time

2. **Validate priority queue**
   - Test with mixed priorities
   - Verify fairness (no starvation)

3. **Benchmark prefill vs decode**
   - Measure throughput separately
   - Verify decode is faster (should be 5-10x)

4. **Flash Attention debugging**
   - Compare against dense attention carefully
   - Test on various seq_len values

5. **Profile the main loop**
   - Identify bottleneck (prefill, decode, or scheduler)
   - Use print statements initially

You're in the home stretch! Week 3 is where the system becomes production-grade. 🚀
