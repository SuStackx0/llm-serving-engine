# vLLM-Lite-from-Scratch: 20-Day Implementation Roadmap

**Timeline**: 20 days × 1 hour/day = 20 hours total  
**Model**: TinyLlama-1.1B or Phi-2  
**Language**: Pure Python + PyTorch/NumPy (math operations only)  
**Goal**: Resume-worthy LLM inference engine with custom PagedAttention + Continuous Batching

---

## Week 1: Foundation & Model Loading

### Day 1: Project Setup & Model Weights Loading
**Objective**: Load .safetensors weights from HuggingFace, validate shapes  
**Deliverable**: `ModelLoader` class that reads model config and weights  
**Key Skills**: Working with HuggingFace ecosystemm, understanding model architecture

**What You'll Build**:
- Load TinyLlama-1.1B from HuggingFace Hub
- Parse model config (hidden_dim, num_layers, num_heads, etc.)
- Validate weight tensor shapes match config
- Move weights to GPU in FP16

**Test**: Load model, verify forward pass outputs valid logits

---

### Day 2: Tokenizer & Input Preprocessing
**Objective**: Convert text → token IDs, handle special tokens  
**Deliverable**: `Tokenizer` wrapper around BPE tokenizer  
**Key Skills**: Understanding token boundaries, padding, EOS tokens

**What You'll Build**:
- Load tiktoken or sentencepiece tokenizer
- Implement batch tokenization with padding
- Handle special tokens (BOS, EOS, PAD)
- Create attention masks for variable-length sequences

**Test**: Tokenize prompts, verify token counts match reference implementation

---

### Day 3: Basic Transformer Forward Pass (No KV Cache)
**Objective**: Implement transformer forward() without caching  
**Deliverable**: `TransformerBlock` class with MLP, Attention  
**Key Skills**: Understanding transformer mathematics, layer normalization

**What You'll Build**:
- Implement RMSNorm (Layer normalization used in modern LLMs)
- Implement standard multi-head attention
- Implement feedforward network (SwiGLU or GELU)
- Stitch blocks together for full forward pass

**Test**: Single forward pass on random batch, check output shape and numerical stability

---

### Day 4: Rotary Positional Embeddings (RoPE) Deep Dive
**Objective**: Implement RoPE from first principles  
**Deliverable**: `RotaryEmbedding` class with precomputed sin/cos matrices  
**Key Skills**: Understanding rotation matrices, FFT-like embedding tricks

**What You'll Build**:
```
RoPE applies rotation to query/key pairs:
- For dimension pair (2i, 2i+1), apply 2D rotation by m*θ_i
- θ_i = base^(-2i/d) where base=10000
- This encodes absolute position WITHOUT learned embeddings
- Key insight: Diagonal structure allows efficient O(1) computation per head
```

- Precompute freqs = {θ_0, θ_1, ..., θ_{d/2}} once
- For each batch + token, compute sin(m*θ), cos(m*θ)
- Apply rotation in-place to Q, K tensors

**Test**: Verify RoPE preserves relative distances (Q_i · K_j depends on i-j, not absolute position)

---

### Day 5: Multi-Head Attention with RoPE
**Objective**: Integrate RoPE into attention mechanism  
**Deliverable**: `PagedAttention` placeholder + standard attention  
**Key Skills**: Head reshaping, masking logic

**What You'll Build**:
- Reshape Q, K, V for multi-head: [batch, seq, heads, head_dim]
- Apply RoPE to Q and K before attention
- Implement causal masking (future tokens hidden)
- Softmax → apply V → reshape back

**Test**: Attention output should attend only to past/current tokens (test with simple pattern)

---

## Week 2: KV Cache & Memory Architecture

### Day 6: Physical Block Manager Design
**Objective**: Build memory pool for KV-cache blocks  
**Deliverable**: `PhysicalBlockManager` class  
**Key Skills**: Memory management, allocation tracking, preemption

**What You'll Build**:
```
KV-Cache Layout:
Total GPU VRAM: 16GB (for TinyLlama on consumer GPU)
Divide into fixed-size blocks: 4 tokens/block
Each block holds K and V for ALL layers at one block of 4 tokens

Block Table Concept:
request_1: KV stored in blocks [0, 5, 12]  (12 tokens total)
request_2: KV stored in blocks [1, 6]      (8 tokens total)
request_3: Free to use blocks [2, 3, 4...] 

PhysicalBlockManager tracks:
- Which blocks are free/allocated
- Which request owns which blocks
- Eviction policy when full
```

- Allocate VRAM into fixed 4-token blocks
- Track free/allocated blocks with bitmap
- Implement allocation policy (first-fit, best-fit)

**Test**: Allocate 5 blocks to request_1, verify can retrieve them; allocate to request_2; check fragmentation

---

### Day 7: KV Cache Block Table & Per-Request Tracking
**Objective**: Map requests → their KV blocks  
**Deliverable**: `KVCacheManager` class with request tracking  
**Key Skills**: Hash tables, linked lists, request lifecycle

**What You'll Build**:
```
For each request:
  - block_table = {0: [phys_block_id_0], 1: [phys_block_id_1], ...}
  - num_filled = current number of cached tokens
  - max_tokens = max tokens this request can cache
  
On each token generate:
  - Check if space in current block (4 tokens/block)
  - If full, allocate next block OR preempt another request
  - Append new K,V to appropriate block
```

- `RequestState` dataclass: id, block_table, num_filled, tokens
- `KVCacheManager`: tracks all requests, allocates blocks on demand
- Implement reference counting to know when a request's blocks can be freed

**Test**: Generate 4 tokens for request_1, verify fills first block; 5th token allocates second block

---

### Day 8: Paged Attention Implementation (Core Algorithm)
**Objective**: Implement attention that reads K,V from scattered physical blocks  
**Deliverable**: `PagedAttention` forward() that handles block tables  
**Key Skills**: Indexing tricks, scatter/gather operations, GPU memory patterns

**What You'll Build**:
```
Standard attention: attn = softmax(Q @ K^T / sqrt(d)) @ V
K is dense [batch, seq, heads, head_dim]

Paged attention with blocks:
- K/V stored in physical blocks (address space fragmented)
- Need to "virtually" gather them for Q @ K^T
- Efficient approach: Don't actually gather; compute attention per-block

AttnOutput = Σ_block Attention_over_block
where each block processes a fraction of the KV cache
```

- Iterate over blocks in request's block_table
- For each block, compute attention: Q @ K_block^T → softmax → V_block
- Accumulate partial attention outputs
- Normalize and finalize

**Test**: Query attention for request with 3 blocks; verify output matches reshuffled dense K,V

---

### Day 9: KV Cache Eviction & Preemption Strategy
**Objective**: Handle GPU memory full; choose requests to stop  
**Deliverable**: `PreemptionManager` class  
**Key Skills**: Scheduling heuristics, priority queues

**What You'll Build**:
```
When new request arrives but GPU full:

Option 1: FCFS Preemption
  - Stop oldest request (finish it on CPU or reject)
  - Free its blocks
  - Allocate to new request

Option 2: Priority-based Preemption  
  - Stop lowest-priority request
  - Useful for SLA-aware serving

Option 3: Token-saving Preemption
  - Preempt request with fewest tokens generated
  - Minimize wasted work
```

- Implement preemption logic: identify victim, free blocks, handle cleanup
- Log preemption events for monitoring
- Support multiple strategies via config

**Test**: Fill GPU with request_1 + request_2; add request_3; verify request_1 or request_2 preempted

---

### Day 10: Batch Handling for Paged Attention
**Objective**: Handle multiple requests with different block tables simultaneously  
**Deliverable**: Batched `PagedAttention` supporting variable-length sequences  
**Key Skills**: NumPy/PyTorch advanced indexing

**What You'll Build**:
```
Batch of 3 requests with different block counts:
  request_0: blocks [] (prefill, no cache yet)
  request_1: blocks [5, 10, 15] (3 blocks = 12 tokens)
  request_2: blocks [6, 11] (2 blocks = 8 tokens)

Batched attention must handle:
- Variable seq lengths within batch
- Different block table layouts
- Padding for attention computation
```

- Create "block table batch": maps each request to its blocks
- Implement scatter/gather for K,V across physical blocks
- Compute attention with sequence length padding

**Test**: Batch attention with request_1 (1 block) + request_2 (3 blocks) at once; compare with sequential

---

## Week 3: Continuous Batching & Scheduling

### Day 11: Token-Level Iteration Scheduler
**Objective**: Build continuous batching that shifts requests in/out dynamically  
**Deliverable**: `ContinuousBatchingScheduler` class  
**Key Skills**: Async task scheduling, event loops

**What You'll Build**:
```
Traditional batching:
  Group requests 0,1,2,3 → all must generate 512 tokens
  Request 0 finishes at 300 tokens, but must wait for others
  Inefficient!

Continuous batching:
  Iteration 0: Batch = {req0, req1, req2, req3}
               All generate 1 token
  Iteration 1: req0 still generating
               req1 still generating
               req2 finished → remove from batch
               req4 waiting → add to batch
               Batch = {req0, req1, req3, req4}
  Iteration 2: ...
  
Key: Batch composition changes EVERY iteration!
```

- Request pool (waiting) + active batch
- At each iteration:
  1. Generate one token for all in batch
  2. Remove finished requests
  3. Add waiting requests if space
  4. Check for preemption needs

**Test**: Run 4 requests with different max_lengths; verify efficient scheduling (no idle time)

---

### Day 12: Prefill vs. Decode Phase Optimization
**Objective**: Handle two different phases efficiently  
**Deliverable**: `PrefillDecode` scheduler  
**Key Skills**: Phase-aware batching heuristics

**What You'll Build**:
```
Prefill phase (processing input prompt):
  - High compute: Q,K,V computed for all input tokens
  - Benefits from large batch size
  - Fills KV cache blocks
  
Decode phase (generating next token):
  - Low compute: only 1 new token input
  - Memory bandwidth bottleneck (read large K,V)
  - Smaller batch sizes optimal
  
Strategy:
  - Prefill requests: batch size = 32
  - Decode requests: batch size = 8
  - Switch request between prefill → decode after 1 iteration
```

- Separate request queues for prefill vs. decode
- Different batch size limits for each queue
- Track request phase transition

**Test**: Schedule mix of prefill + decode requests; verify batch size limits respected

---

### Day 13: Request Queue & Priority Handling
**Objective**: Implement FIFO + priority queue for request ordering  
**Deliverable**: `RequestQueue` with priority support  
**Key Skills**: Heap queue data structures, fairness

**What You'll Build**:
```
Priority levels:
  HIGH (0): SLA-critical requests
  MEDIUM (1): Standard requests
  LOW (2): Best-effort batch processing

Within same priority: FIFO order
Fairness: After serving 3 high-priority, serve 1 medium, etc.
```

- Use heapq with (priority, arrival_time, request)
- Implement fairness counter to prevent starvation
- Support dynamic priority adjustment

**Test**: Add 10 requests with mixed priorities; verify high-priority first, but medium/low also progress

---

### Day 14: Batched Token Generation Loop
**Objective**: Implement main inference loop combining all components  
**Deliverable**: `InferenceEngine.generate_batch()` method  
**Key Skills**: Coordinating multiple subsystems

**What You'll Build**:
```python
def generate_batch():
    active_requests = select_next_batch()
    
    # Prefill: New requests process entire prompt
    prefill_requests = [r for r in active_requests if r.phase == PREFILL]
    if prefill_requests:
        batch_q = stack([r.q.tokens for r in prefill_requests])
        logits = forward_pass(batch_q)  # Fill KV cache here
        # Mark prefill done, move to decode
        for r in prefill_requests:
            r.phase = DECODE
            r.k_cache.fill = len(r.prompt_tokens)
    
    # Decode: Generate next token
    decode_requests = [r for r in active_requests if r.phase == DECODE]
    if decode_requests:
        last_tokens = [r.tokens[-1] for r in decode_requests]
        block_tables = [r.kv_block_table for r in decode_requests]
        logits = paged_attention_forward(last_tokens, block_tables)
        
        next_tokens = sample(logits, temperature)
        for r, token in zip(decode_requests, next_tokens):
            r.tokens.append(token)
            r.kv_block_table.fill += 1
            if is_eos(token) or len(r.tokens) >= r.max_length:
                r.finished = True
    
    # Remove finished requests
    active_requests = [r for r in active_requests if not r.finished]
    
    return active_requests, generated_tokens
```

- Separate prefill/decode paths
- Coordinate KV cache allocation
- Handle request removal
- Return generated tokens to client

**Test**: Generate 500 tokens from 4 concurrent requests; verify final outputs are coherent

---

## Week 4: Performance Tuning & Benchmarking

### Day 15: Memory-Efficient Attention (FlashAttention-style)
**Objective**: Optimize attention kernel for reduced memory I/O  
**Deliverable**: `OptimizedPagedAttention` with tiling  
**Key Skills**: GPU memory access patterns, kernel fusion

**What You'll Build**:
```
Standard attention O(N^2) memory:
  Compute all Q@K scores → [seq, seq] intermediate (2GB+ for long sequences!)
  Then softmax, then V multiply

Tiled approach (Flash Attention):
  Iterate in blocks:
    For each K,V block:
      Compute Q @ K_block^T (smaller intermediate)
      Softmax incrementally
      Accumulate output
  
Memory: O(N) instead of O(N^2)!
Speed: Better GPU cache locality
```

- Implement block-wise attention loop
- Support variable block sizes (tunable)
- Keep numerical stability (log-sum-exp trick for softmax)

**Test**: Verify attention output matches O(N^2) version; measure 2x speedup on seq_len=2048

---

### Day 16: KV Cache Quantization (Optional Advanced)
**Objective**: Compress K,V to INT8 for memory savings  
**Deliverable**: `QuantizedKVCache` class  
**Key Skills**: Numerical precision trade-offs, quantization schemes

**What You'll Build**:
```
Standard: K,V in FP16 = 2 bytes per value
Quantized: K,V in INT8 = 1 byte per value (2x compression!)

Per-channel quantization:
  For each head, compute scale factor:
    scale = max(|K|) / 127
  Store K_int8 = round(K_fp16 / scale)
  Dequantize on-the-fly during attention

Trade-off: ~5-10% quality loss, 2x memory gain, ~10% latency overhead
```

- Implement quantization/dequantization
- Store scale factors alongside K,V
- Support mixed quantization (only K,V quantized, Q stays FP16)

**Test**: Run inference with quantized KV; compare outputs with FP16 baseline; measure memory saved

---

### Day 17: Latency Profiling & Optimization
**Objective**: Identify bottlenecks; optimize hot paths  
**Deliverable**: Profiling suite with timing breakdowns  
**Key Skills**: Performance analysis, optimization targeting

**What You'll Build**:
```
Profile metrics:
```
1. Prefill latency (ms per token in batch)
2. Decode latency (ms per output token)
3. KV cache allocation overhead
4. Attention compute time
5. Memory bandwidth utilization
6. Per-layer breakdown

- Add CUDA events/timers to key functions
- Log latencies per request + aggregated stats
- Identify if bottleneck is compute vs. memory

**Test**: Profile 10-request batch; identify if attention, MLP, or KV cache is slowest; optimize that

---

### Day 18: Throughput Maximization (Batch Size Tuning)
**Objective**: Find optimal batch size for peak tokens/sec  
**Deliverable**: Auto-tuning batch size selection  
**Key Skills**: Empirical performance tuning

**What You'll Build**:
```
Vary batch sizes and measure throughput:

batch_size=1: low latency, low throughput
batch_size=8: medium latency, good throughput
batch_size=16: higher latency, peak throughput?
batch_size=32: OOM or thermal throttle

Optimal sweet spot: Maximum throughput without queue buildup/OOM
```

- Implement batch size sweep (1, 2, 4, 8, 16, 32...)
- For each, measure tokens/sec over 100 iterations
- Auto-select best batch size at startup
- Monitor GPU temperature/utilization

**Test**: Run throughput sweep; identify optimal batch_size; configure system to use it

---

### Day 19: End-to-End Integration Testing
**Objective**: Test full pipeline: text in → token sampling → text out  
**Deliverable**: Complete inference service  
**Key Skills**: Integration testing, debugging multi-component systems

**What You'll Build**:
- Text preprocessing → tokenization
- Forward pass through all layers (prefill + decode)
- Token sampling with temperature/top-p
- Output post-processing (detokenization)
- Request lifecycle: submit → batch → generate → return → cleanup

**Test**: Generate completions for 10 diverse prompts; verify outputs are coherent, not garbage

---

### Day 20: Benchmarking Suite & Final Optimization
**Objective**: Create reproducible benchmarks; final performance push  
**Deliverable**: `benchmark.py` with TTFT and TPOT measurements  
**Key Skills**: Benchmark design, reproducibility

**What You'll Build**:
```
Benchmark Suite:
1. TTFT (Time-to-First-Token):
   - Submit request
   - Measure time until first output token
   - Profile: tokenization + prefill + first decode
   
2. TPOT (Time-per-Output-Token):
   - Measure decode iteration latency
   - Can vary with batch size
   
3. Throughput (tokens/sec):
   - Generate 1000 tokens with varying batch sizes
   - Measure total time
   
4. Memory efficiency:
   - Peak VRAM usage
   - KV cache efficiency per-token
   
5. Concurrent requests:
   - How many requests can run together?
   - Measure until preemption threshold
```

- Run benchmarks against reference (original model in transformers)
- Document results with 5+ different scenarios
- Identify remaining bottlenecks
- Final optimization pass

**Final Test**: Generate 2000 tokens with 8 concurrent requests; measure TTFT, TPOT, memory, confirm system is stable

---

## Success Criteria by Week

### Week 1: Foundation
- ✅ Model loads and runs forward pass
- ✅ RoPE correctly encodes position
- ✅ Attention masks work (verify causality)

### Week 2: Memory Architecture  
- ✅ Block allocation doesn't fragment
- ✅ Paged attention outputs match dense attention
- ✅ Multiple requests can coexist in VRAM

### Week 3: Continuous Batching
- ✅ Requests shift in/out without recompilation
- ✅ Batch size respects VRAM limits
- ✅ No requests starved (fairness working)

### Week 4: Performance
- ✅ TTFT < 500ms for 100-token prompt
- ✅ TPOT < 50ms per token (with batch_size=8)
- ✅ Throughput > 100 tokens/sec

---

## Key Files to Create

```
llm-serving-engine/
├── docs/
│   ├── ROUTEMAP.md (this file)
│   ├── ARCHITECTURAL_SPEC.md
│   ├── RESUME_IMPACT.md
│   └── BENCHMARKING_GUIDE.md
│
├── src/
│   ├── model_loader.py         (Day 1)
│   ├── tokenizer.py            (Day 2)
│   ├── transformer.py          (Day 3-5)
│   ├── rope.py                 (Day 4)
│   ├── block_manager.py        (Day 6-7)
│   ├── paged_attention.py      (Day 8-10)
│   ├── scheduler.py            (Day 11-14)
│   ├── inference_engine.py     (Day 14)
│   ├── optimizations.py        (Day 15-16)
│   └── profiler.py             (Day 17-18)
│
├── tests/
│   ├── test_rope.py
│   ├── test_attention.py
│   ├── test_block_manager.py
│   ├── test_scheduler.py
│   └── test_inference.py
│
├── benchmarks/
│   ├── benchmark.py            (Day 20)
│   ├── profile.py              (Day 17)
│   └── throughput_sweep.py     (Day 18)
│
└── README.md
```

---

## Commit Strategy (Git Discipline)

One commit per day with clear messages:

```
Day 1: Load model weights and validate shapes
Day 2: Tokenizer wrapper with batch processing
Day 3: Transformer forward pass (no caching)
Day 4: RoPE implementation from first principles
Day 5: Multi-head attention with RoPE integration
Day 6: Physical block manager for KV cache
Day 7: KV cache block table per-request tracking
Day 8: Paged attention read from scattered blocks
Day 9: Preemption and eviction strategies
Day 10: Batched paged attention with variable seqlens
Day 11: Continuous batching scheduler framework
Day 12: Prefill vs decode phase optimization
Day 13: Priority queue with fairness
Day 14: End-to-end generation loop
Day 15: Memory-efficient attention tiling
Day 16: KV cache quantization (optional)
Day 17: Latency profiling infrastructure
Day 18: Throughput maximization & tuning
Day 19: Integration testing
Day 20: Benchmarking suite & final optimization
```

---

## TL;DR: What Makes This Resume Gold

1. **Custom Paged Attention**: You built what vLLM built (using blocks, not dense tensors)
2. **Continuous Batching**: Requests join/leave mid-generation; high utilization
3. **RoPE from Scratch**: Understand rotation matrices, positional encoding deeply
4. **Block Manager**: Memory management skills like systems engineers use
5. **No Library Magic**: Every core component is custom Python + PyTorch

**Recruiter Pitch**: "I built a high-performance LLM inference engine from first principles—custom PagedAttention, continuous batching scheduler, and block-based KV cache management. Zero reliance on high-level libraries. Achieved 100+ tokens/sec single-GPU throughput with < 500ms time-to-first-token."
