# Week 2: KV Cache & Memory Architecture (Days 6-10)

## Overview: What You're Building This Week

By the end of Week 2, you'll have a **memory-efficient inference system** that:
- Manages GPU memory like an operating system
- Caches attention computations (K,V) for reuse
- Supports multiple concurrent requests
- Achieves **3x speedup** over Week 1

**The Core Problem You're Solving**: Week 1 regenerates K,V values for every token. Week 2 **caches and reuses** them.

Think of it like a bakery:
- Week 1: Bake fresh bread for every customer
- Week 2: Pre-bake bread once, slice it for each customer

---

## Day 6: Physical Block Manager

### The Problem: GPU Memory is Fragmented

Imagine you have 10 GB of GPU memory. You load:
```
Request 1: needs 2 GB for KV cache
Request 2: needs 1.5 GB for KV cache
Request 3: needs 2.5 GB for KV cache
Request 4: arrives, needs 1.5 GB ...
         ↓ but 1.5 GB is split across multiple holes!
         ↓ fragmented!

Free space: 2.5 GB total, but scattered in [0.1, 0.3, 0.5, 1.2, 0.4] GB chunks
Request 4: Can't fit! Even though we have 2.5GB free!
```

This is **memory fragmentation** - the OS problem that plagued 1980s computers.

### The Solution: Fixed-Size Blocks

Instead of dynamic allocation, use fixed-size **blocks** (like pages in OS virtual memory):

```
10 GB GPU Memory divided into 1000 blocks of 10 MB each

Request 1: occupies blocks [0, 1, 2, 3, ..., 20]       (200 MB allocated)
Request 2: occupies blocks [100, 101, 102, ...]        (150 MB allocated)
Request 3: occupies blocks [500, 501, ..., 750]        (250 MB allocated)
Request 4: occupies blocks [21, 22, 23, ..., 35]       (150 MB allocated)

Free blocks: [36-99], [103-499], [751-999]
             These CAN be fragmented! But allocation is simple.
```

### What You're Building (Theory)

A **Physical Block Manager** that:

1. **Initializes the block pool**
   - Total GPU memory (e.g., 16 GB)
   - Reserve some for model weights (~4.5 GB)
   - Reserve some for compute buffers (~1 GB)
   - Remaining is block pool (~10.5 GB)
   - Divide into fixed-size blocks (e.g., 256 blocks)

2. **Tracks allocation state**
   - Which blocks are free
   - Which blocks are allocated to which request
   - How many tokens are in each block

3. **Allocates on demand**
   - When request 1 generates first token, allocate one block
   - When it fills the block (16 tokens per block), allocate another
   - Continue until request finishes

4. **Frees memory**
   - When request finishes, mark its blocks as free
   - Free blocks become available for next request

### Why This Matters

- **Simplicity**: No complex memory allocation algorithms
- **Speed**: Allocation is O(1) (just pop from free list)
- **Predictability**: No surprise OOM errors mid-generation
- **Real-world**: This is exactly how vLLM does it!

### Prerequisites You Need

- **What's memory fragmentation?** (Free space scattered, hard to allocate)
- **OS virtual memory concepts?** (Pages, allocation tables)
- **Data structures**: Lists, sets (which blocks are free)

### How You'll Know It Works (Day 6 Success Criterion)

```
You implement PhysicalBlockManager:
  ✓ Initialize with 16 GB → creates ~256 blocks
  ✓ Request 1 allocates 5 blocks → verify blocks removed from free pool
  ✓ Request 2 allocates 3 blocks → verify different blocks
  ✓ Free Request 1 → blocks returned to free pool
  ✓ Request 3 allocates 5 blocks → can reuse freed blocks from Request 1
  ✓ Fragmentation test: allocate/free multiple times, verify no OOM
```

### Why This Matters

This is the **foundation for efficient memory management**. Without this, you can't handle concurrent requests.

---

## Day 7: KV Cache Block Table & Per-Request Tracking

### The Problem: Which Blocks Belong to Which Request?

When you have 200 blocks and 10 concurrent requests, you need to track:
```
Request 0: blocks [5, 12, 48, 100]       (tokens 0-63)
Request 1: blocks [6, 13, 49]            (tokens 0-47)
Request 2: blocks [7, 14]                (tokens 0-31)
...
```

If you call `get_kv_for_request(1)`, you need to quickly find blocks [6, 13, 49].

### The Solution: Block Table (Like OS Page Tables)

Create a mapping:
```
block_table = {
    request_0: {
        "block_indices": [5, 12, 48, 100],
        "num_filled": 64,         # How many tokens actually generated
        "max_tokens": 512         # Max this request can generate
    },
    request_1: {
        "block_indices": [6, 13, 49],
        "num_filled": 48,
        "max_tokens": 512
    },
    ...
}
```

### What You're Building (Theory)

A **KV Cache Manager** that:

1. **Tracks per-request state**
   - Which blocks belong to this request
   - How many tokens filled in each block
   - How many total tokens this request has generated

2. **Grows cache as request generates tokens**
   - Token 1: occupies block[0], position 0
   - Token 2: occupies block[0], position 1
   - ...
   - Token 16: occupies block[0], position 15
   - Token 17: occupies block[1], position 0 (new block!)

3. **Allocates blocks on-demand**
   - Don't pre-allocate all blocks at start
   - Allocate as needed during generation
   - When block fills (16 tokens), allocate next block

4. **Retrieves K,V for attention** (you'll use this in Day 8)
   - Given request_id and position
   - Return the physical location of K,V for that token
   - Enables attention computation

### Why This Matters

- **Efficient**: Don't allocate memory until needed
- **Flexible**: Same code works for short and long sequences
- **Real-world**: This is the "block table" in vLLM

### Prerequisites You Need

- **Dictionary/hash table operations** (look up request_id → blocks)
- **How GPU memory is addressed** (block[0] = bytes 0-10MB, block[1] = bytes 10-20MB)

### How You'll Know It Works (Day 7 Success Criterion)

```
You implement KVCacheManager:
  ✓ Request 1 generates token 1 → allocate block 0
  ✓ Request 1 generates tokens 2-16 → fill block 0
  ✓ Request 1 generates token 17 → allocate block 1
  ✓ Request 2 starts → allocate its own block 
  ✓ Retrieve: get_kv(request_1, token_15) → returns block 0, position 15
  ✓ Retrieve: get_kv(request_1, token_17) → returns block 1, position 1
  ✓ Free request 1 → blocks 0,1 returned to pool, available for request 3
```

### Why This Matters

You now have the **data structure** to support multiple concurrent requests. Day 8 uses this to compute attention.

---

## Day 8: Paged Attention Implementation (Core Algorithm)

### The Problem: Attention Over Scattered Blocks

In standard attention:
```
Q = query vector for current token
K = all cached key vectors (dense [num_tokens, hidden_size])
V = all cached value vectors (dense [num_tokens, hidden_size])

Attention = softmax(Q @ K^T) @ V
```

But in paged attention, K and V are **scattered across blocks**:
```
K values are in blocks [5, 12, 48, 100]
V values are in blocks [5, 12, 48, 100]

Not contiguous! Can't just multiply Q @ K^T directly!
```

### The Solution: Block-Level Computation

Instead of computing attention over all tokens at once, compute **per-block**:

```
output = 0
for each block in block_table[request]:
    K_block = physical_memory[block].K
    V_block = physical_memory[block].V
    
    # Attention over this block only
    scores = Q @ K_block^T
    probs = softmax(scores)
    output += probs @ V_block

# Normalize output
output = output / num_blocks
```

This is like:
- Block 1: "What should I pay attention to in block 1?" → weighted sum
- Block 2: "What should I pay attention to in block 2?" → weighted sum
- Combine all weighted sums

### What You're Building (Theory)

A **Paged Attention** forward pass that:

1. **Takes scattered K,V from blocks**
   - Given block_table[request] = [5, 12, 48, 100]
   - Retrieve K and V from each block

2. **Computes attention per-block**
   - For each block: compute Q @ K_block^T
   - Softmax within that block
   - Weighted sum with V_block
   - Accumulate results

3. **Handles partial blocks** (last block might not be full)
   - Block 100 might only have 8 tokens filled (out of 16)
   - Trim K,V to only filled portion
   - Avoid attention over empty positions

4. **Returns final attention output**
   - Same shape as dense attention
   - But computed over scattered blocks

### Why This Matters

- **Memory efficient**: K,V scattered across blocks, no need to gather into dense tensor
- **Enables paging**: If blocks fill up, can page K,V to CPU/NVMe (Week 3-4)
- **Core innovation**: This is the key algorithm that makes vLLM fast!

### Prerequisites You Need

- **How does softmax work?** (Normalizes scores to probabilities)
- **Why normalize across multiple blocks?** (Need to account for partial attention)
- **Log-sum-exp trick** (To avoid numericaloverflow when computing softmax)

### How You'll Know It Works (Day 8 Success Criterion)

```
You implement PagedAttention forward():
  ✓ Compute attention with scattered blocks
  ✓ Compare vs. dense attention (reshape blocks into dense K,V, compute standard attention)
  ✓ Results match within numerical precision (< 1e-4 difference)
  ✓ Variable number of blocks: test with 1, 3, 10 blocks
  ✓ Partial blocks: test with last block not fully filled
  ✓ Performance: faster than dense attention (especially for long sequences)
```

### Why This Matters

You've **implemented the core algorithm** that makes vLLM work. This is a major resume achievement.

---

## Day 9: KV Cache Eviction & Preemption Strategy

### The Problem: GPU Memory Full

Scenario:
```
Request 1: 64 blocks
Request 2: 48 blocks
Request 3: 32 blocks
Total: 144 blocks, but we only have 256 blocks

New request 4 arrives, needs 64 blocks
Free blocks available: 256 - 144 = 112 blocks ✓ Enough!

But wait... later:
Request 1: 200 blocks (still generating)
Request 2: 180 blocks (still generating)
Request 3: 150 blocks (still generating)
Total: 530 blocks needed, but only have 256! ✗ OOM!

What do we do??
```

### The Solution: Preemption

**Stop some ongoing requests** to free their blocks.

Questions:
- Which request to stop?
- What happens to the stopped request?

### What You're Building (Theory)

A **Preemption Manager** that:

1. **Detects when GPU is full**
   - Monitor free blocks
   - Set threshold (e.g., stop when < 10 free blocks)

2. **Selects victim to preempt**
   - **Token-count heuristic**: Stop request with fewest tokens generated (minimize waste)
   - **Priority heuristic**: Stop lowest-priority request
   - **Mixed**: Combine both

3. **Stops the request**
   - Free all its blocks
   - Return partial output to user ("Sorry, interrupted!")
   - Or save state to retry later (advanced)

4. **Logs what happened**
   - Metrics: "Preempted request 5 to free 64 blocks"
   - Helps debugging and monitoring

### Different Preemption Policies

```
Policy 1: FIFO (First-In-First-Out)
  Preempt: oldest request (first to start)
  Why: simple, fair

Policy 2: Priority-Aware
  Preempt: lowest-priority request (ignore age)
  Why: SLA-aware (VIP users don't get interrupted)

Policy 3: Token-Count (Minimize Waste)
  Preempt: request with fewest tokens generated
  Why: don't throw away lots of compute

Policy 4: Mixed
  Preempt: lowest-priority AND fewest tokens
  Why: best of both worlds
```

### Why This Matters

- **Robustness**: System doesn't crash, degrades gracefully
- **Real-world**: Production systems need preemption
- **Fairness**: VIP requests (high priority) finish while low-priority get interrupted

### Prerequisites You Need

- **Fairness vs efficiency trade-offs** (interrupting is bad for latency, good for throughput)
- **SLA (Service Level Agreement)**: Different users have different priorities

### How You'll Know It Works (Day 9 Success Criterion)

```
You implement PreemptionManager:
  ✓ Detect when GPU full (< 10 free blocks)
  ✓ Select victim based on policy
  ✓ Free blocks from preempted request
  ✓ Test scenario: load 256 blocks, add request needing 64 blocks
    - Before preemption: OOM
    - After preemption: frees blocks, accommodates new request
  ✓ Log shows: "Preempted request X, freed Y blocks"
```

### Why This Matters

This is the **safety net** of the system. Without it, you'd crash. With it, you gracefully degrade.

---

## Day 10: Batch Handling for Paged Attention

### The Problem: One Request, Then Another Request

So far you've built:
- PhysicalBlockManager: allocates blocks
- KVCacheManager: tracks per-request blocks
- PagedAttention: computes attention for one request

But in production, you process **multiple requests in a batch**:
```
Batch = [Request_0, Request_1, Request_2, Request_3]
```

Each has:
- Different number of tokens generated
- Different block tables
- Different sequence lengths

How do you compute attention for all 4 at once?

### The Solution: Batched Paged Attention

Instead of looping through requests one-by-one, **batch the computation**:

```
Request 0: blocks [5, 12]        (32 tokens)
Request 1: blocks [6, 13, 49]    (48 tokens)
Request 2: blocks [7]            (16 tokens)
Request 3: blocks [8, 14]        (32 tokens)

Batched forward pass:
  Input: 4 queries from 4 requests
  Output: 4 attention outputs

GPU can process all 4 in parallel! 4x speedup!
```

### What You're Building (Theory)

A **Batched Paged Attention** that:

1. **Takes batch of requests**
   - Each request has its own block_table
   - Each request has its own Q (query for current/last token)

2. **Gathers K,V from blocks**
   - Aggregate blocks from all requests
   - Create "batch block indices"
   - Retrieve K,V for each request's blocks

3. **Computes attention in parallel**
   - PyTorch/GPU handles parallelism
   - All 4 requests' attention computed simultaneously
   - Much faster than serial

4. **Handles variable sequence lengths**
   - Some requests might have 16 tokens, others 100
   - Padding ensures fixed-size computation

### Why This Matters

- **Efficiency**: GPU is designed for batch operations
- **Speed**: 4 requests in parallel = ~4x speedup
- **Real-world**: Production systems always batch

### Prerequisites You Need

- **Broadcasting** in PyTorch (how tensors align)
- **Padding** (making variable-length sequences fixed-size)
- **Batch dimension** (first axis, 4 requests)

### How You'll Know It Works (Day 10 Success Criterion)

```
You implement batched PagedAttention:
  ✓ Create batch of 4 requests with different block tables
  ✓ Compute attention for batch
  ✓ Compare vs. computing 4 requests sequentially
  ✓ Results match (within numerical precision)
  ✓ Performance: batched should be ~3-4x faster
  ✓ Variable sequence lengths: test 1, 3, 5 requests with different token counts
```

### Why This Matters

You've now built the **complete memory-efficient inference system** for Week 2. Multiple requests with scattered blocks, computed in parallel.

---

## End of Week 2: What Your System Looks Like

At the end of Day 10, you have:

```
Request 1 (batch_size=1)
  Tokens generated: 50
  Blocks: [5, 12, 48]
  State: Generating

Request 2 (batch_size=1)
  Tokens generated: 30
  Blocks: [6, 13]
  State: Generating

Request 3 (batch_size=1)
  Tokens generated: 150
  Blocks: [7, 8, 9, 10, 11]
  State: Generating

                ↓
        Block Manager
    Free: 99 blocks, Allocated: 157
                ↓
        KV Cache Manager
    Maps requests → blocks
                ↓
        Batched Paged Attention
        Process all 3 requests in parallel
                ↓
        Output: 3 next tokens
```

### Performance at End of Week 2

**Much better!**
- **TTFT**: ~300-400ms (3-5x faster than Week 1!)
  - Why? KV cache reduces recomputation in prefill
- **TPOT**: ~20-30ms per token (better!)
  - Why? Using cached K,V in decode phase
- **Throughput**: ~50-80 tokens/sec (way better!)
  - Why? Batching requests
- **Concurrent requests**: Can handle 3-4 at once (vs 1 in Week 1)

### What's Still Missing (You'll Add in Week 3-4)

- **Continuous Batching**: Requests enter/exit batch dynamically (not fixed)
- **Prefill/Decode separation**: Different batch sizes for each phase
- **Request prioritization**: VIP requests go first
- **Further optimizations**: Flash attention, INT8 quantization

---

## Architectural Questions?

If you get stuck on:
- **How does block table work?** → See ARCHITECTURAL_SPEC.md > Physical Block Manager
- **Paged attention math?** → See ARCHITECTURAL_SPEC.md > Paged Attention
- **How to batch different sequence lengths?** → See ARCHITECTURAL_SPEC.md > Batched Attention

## Prerequisites Learning Path

If you don't know these:

1. **OS Virtual Memory** (30 mins)
   - Read: Wikipedia "Virtual Memory"
   - Understand: Pages, fragmentation, page tables

2. **Softmax numerical stability** (15 mins)
   - Read: "LogSumExp trick"
   - Why: softmax(x) = softmax(x - max(x)) for stability

3. **GPU memory addressing** (15 mins)
   - How: Byte offsets, fixed allocations
   - Why: Enables scattered KV cache

## Week 2 Success Criteria

By the end of Friday:
- ✅ PhysicalBlockManager allocates/frees blocks
- ✅ KVCacheManager tracks per-request blocks
- ✅ PagedAttention computes over scattered blocks
- ✅ Preemption kicks in when needed
- ✅ Can batch 3-4 requests together
- ✅ TTFT reduced to 300-400ms
- ✅ Can generate longer sequences

---

## Week 2 → Ready for Week 3

Week 3 adds the scheduler:

**Problem**: Current system requires you to manually decide which requests to batch.  
**Solution**: **Continuous Batching Scheduler** - automatically manages requests (next week)

But you need Week 2 working first. Don't skip it!

---

## Tips for Week 2

1. **Test block manager thoroughly**
   - Allocate 100 blocks, free them, allocate again
   - Verify no fragmentation issues

2. **Validate paged attention carefully**
   - Compare against dense attention implementation
   - Test many different block configurations

3. **Start with one request, then batch**
   - Day 6-8: Single request working
   - Day 9-10: Add preemption and batching

4. **Use small tensors for debugging**
   - 4 KB blocks instead of 10 MB (faster tests)
   - Small batch sizes (2-3 requests)

5. **Refer to ARCHITECTURAL_SPEC.md**
   - Has pseudo-code for all components
   - Has validation tests

Great job getting through Week 2! You now have the **memory management foundation** for a real LLM serving system. 🎉
