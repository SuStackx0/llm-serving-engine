# 02 — PagedAttention, Continuous Batching & Preemption (Theory + This Impl)

## PagedAttention: Virtual Memory for KV Caches

PagedAttention (Kwon et al., 2023 — the vLLM paper) borrows an idea from OS virtual memory:
instead of one contiguous allocation per process, divide memory into fixed-size **pages** and
map them to a **page table**.

In LLM serving:
- Memory is divided into fixed-size **blocks** (e.g., 16 tokens × all layers)
- Each request gets a **block table**: a mapping from logical block index → physical block ID
- Blocks are allocated from a global free list as the sequence grows
- When a request finishes, its blocks are returned to the free list

**Why this eliminates fragmentation:** every block is the same size, so any free block can satisfy
any allocation request. You never have a 1 GB hole that no single allocation can fill.

### Memory Layout in This Engine

```
kv_storage[num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]
            └──────────┘  │  └────────┘  └────────┘  └───────────────────┘
             all layers   K/V  block ID   slot in block   one KV vector
```

For a request with block_table = [5, 12, 3]:
- Tokens 0-15 → physical block 5, slots 0-15
- Tokens 16-31 → physical block 12, slots 0-15
- Tokens 32-47 → physical block 3, slots 0-15

The blocks are physically non-contiguous in memory but logically contiguous to the attention
computation (we gather them before computing attention).

**File:** `src/memory/kv_cache.py`, `src/memory/block_manager.py`

---

## The Engine Loop

The engine runs in a background thread. Every iteration:

```
loop:
  1. Drain input queue → add new requests to scheduler.waiting
  2. scheduler.schedule() → decide who gets compute this step
  3. Run prefill for newly admitted requests
  4. Run decode for all DECODING requests
  5. Send finished tokens back to callers
```

**File:** `src/engine/inference_engine.py`, method `_run_loop()`

---

## The Scheduler: One Step at a Time

`scheduler.schedule()` runs once per engine loop iteration and returns a `SchedulerOutput`:

```python
@dataclass
class SchedulerOutput:
    prefill_requests: List[Request]    # run their prompt forward pass
    decode_requests: List[Request]     # generate their next token
    preempted_requests: List[Request]  # evicted due to memory pressure
```

### Step 1: Clean up finished requests
Free their blocks back to the free list.

### Step 2: Ensure DECODING requests have room for their next token
Every 16 tokens (one block), a decode request needs a new block. If no block is available,
the scheduler **preempts** a lower-priority request.

### Step 3: Admit waiting requests
For each waiting request:
1. Compute blocks needed: `ceil((prompt_len + max_tokens) / block_size)`
2. If free blocks available: pop from waiting queue, allocate blocks, status → PREFILLING
3. If not: try to free blocks by evicting a lower-priority decode request first

### Step 4: Split running requests into prefill / decode lists

**File:** `src/scheduler/scheduler.py`

---

## Preemption: What Happens When Memory Runs Out

If a decode request needs its next block and there are no free blocks, the scheduler must
**preempt** a running request to free its blocks.

Victim selection heuristic: among all DECODING requests, pick the one with:
1. Highest priority number (lower priority)
2. Among equal priority: fewest generated tokens (least wasted work)

What happens to the preempted request:
- Its blocks are freed
- Its generated tokens are discarded (it must re-do prefill)
- Status → WAITING, re-added to the priority queue
- The caller waits transparently — they never see the preemption

**File:** `src/scheduler/scheduler.py`, `_preempt()` and `_select_preemption_victim()`

---

## Request Lifecycle (Full State Machine)

```
SUBMITTED → WAITING → PREFILLING → [CHUNK_DONE ...] → DECODING → FINISHED_EOS
                ↑                                                   FINISHED_LENGTH
                └─────────── PREEMPTED ──────────────────────────── FINISHED_STOP
```

With chunked prefill enabled, a request can stay in PREFILLING across multiple engine steps,
emitting CHUNK_DONE events between each partial forward pass.

### Timing Fields on Each Request

| Field | Set when | Used for |
|---|---|---|
| `arrival_time` | Request created | Queue wait time |
| `prefill_start_time` | Admitted by scheduler | TTFT start point |
| `first_token_time` | First output token generated | TTFT end point |
| `last_token_time` | Updated every decode step | TPOT calculation |

**TTFT (Time To First Token):** `first_token_time - prefill_start_time`  
**TPOT (Time Per Output Token):** `(last_token_time - first_token_time) / (num_generated - 1)`

---

## Priority Queue

Requests are sorted by `(priority, arrival_time)`. Lower priority number = higher priority
(like process nice values). Within the same priority level, FIFO order is preserved.

**File:** `src/scheduler/request_queue.py`

---

## Attention: Prefill vs. Decode

The same `PagedAttentionLayer` handles both modes:

**Prefill:**
```python
# All prompt tokens at once
q [seq_len, H, D]  ×  k [seq_len, H, D]  →  scores [H, seq_len, seq_len]
# Apply causal mask (upper triangular = -inf)
# Store K/V into KV cache blocks
```

**Decode:**
```python
# One new query token
q [1, H, D]  ×  k_gathered [ctx_len, H, D]  →  scores [H, 1, ctx_len]
# No causal mask needed (single query can see all prior context)
# Gather K/V from scattered blocks via block_table
```

**File:** `src/model/attention.py`

---

## GQA: Grouped Query Attention

Modern models (Llama, Mistral) use **Grouped Query Attention** to reduce KV cache size.
Instead of one KV head per attention head, multiple heads share one KV head:

- TinyLlama: 32 attention heads, 4 KV heads → GQA ratio = 8
- Each KV head is shared by 8 attention heads
- KV cache is 8× smaller than full MHA

Implementation: `k.repeat_interleave(gqa, dim=1)` expands KV heads to match attention heads
just before computing scores.

**File:** `src/model/attention.py`, `src/core/config.py` (`gqa_ratio` property)

---

## Sampling

After each forward pass, we have `logits[vocab_size]`. Sampling converts this to a token:

1. **Greedy** (temperature=0): `argmax(logits)`
2. **Temperature scaling**: `logits /= temperature` before softmax (makes distribution sharper/flatter)
3. **Top-k**: keep only the top k logit values, zero the rest
4. **Top-p (nucleus)**: keep the smallest set of tokens whose cumulative probability ≥ p

These are composed: temperature → top-k → top-p → sample.

**File:** `src/model/sampling.py`
