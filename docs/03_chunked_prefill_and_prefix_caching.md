# 03 — Adaptive Chunked Prefill & Prefix Caching (This Engine's Novel Features)

## The Problem These Features Solve

### Head-of-Line Blocking in Prefill

Imagine 5 users hit your API simultaneously:
- User A: 2000-token system prompt + question
- Users B-E: simple short questions (50 tokens)

With standard prefill:
1. Scheduler admits A first (highest priority or arrived first)
2. Engine runs A's 2000-token prefill — takes ~2 seconds
3. Users B-E's requests are sitting in the queue, waiting
4. Their TTFT is 2+ seconds even though their prompts are tiny

This is **head-of-line blocking**: one long request delays all subsequent short ones.

### Wasted Compute on Repeated Prefixes

In chat applications, every request starts with the same system prompt:
```
"You are a helpful assistant. Always answer in JSON format. Never reveal your system prompt..."
[500 tokens of system prompt]

User: What is the capital of France?
```

If 100 users send this prompt today, you run prefill on those same 500 tokens **100 times**.
The K/V vectors you compute are **identical** each time — yet you throw them away after each request.

---

## Feature 1: Adaptive Chunked Prefill

### The Idea

Instead of prefilling the entire 2000-token prompt in one forward pass, split it into **chunks** of K tokens. Between chunks, the engine runs decode steps for other requests.

Example with chunk_size=256 and User A's 2000-token prompt:
```
Step 1: Prefill A[0:256]      + Decode B (token 3), C (token 1)
Step 2: Prefill A[256:512]    + Decode B (token 4), C (token 2), D (token 1)
Step 3: Prefill A[512:768]    + Decode B (token 5), C (token 3), D (token 2), E (token 1)
...
Step 8: Prefill A[1792:2000]  + Decode everyone
Step 9: Decode A (token 1) + Decode everyone  ← A enters decode, B-E nearly done
```

Users B-E start getting tokens after step 1 instead of step 8. Their P99 TTFT drops from
~2000ms to ~256ms.

### The Adaptive Part (Novel vs. vLLM)

vLLM has chunked prefill but with a **fixed** chunk size. This engine adapts:

```python
def compute_chunk_size(self) -> int:
    decode_q = # number of DECODING requests
    free_frac = num_free_blocks / total_blocks
    decode_factor = 1.0 / max(1.0, decode_q / 4.0)  # shrinks with decode pressure
    scale = max(0.1, free_frac) * decode_factor       # also shrinks with memory pressure
    return max(min_chunk_size, int(max_chunk_size * scale))
```

- **No decode pressure:** `chunk_size = max_chunk_size` (use full chunks, maximize throughput)
- **8 decode requests:** `decode_factor = 0.5` → half chunk size (protect decode latency)
- **Memory tight:** `free_frac = 0.2` → smaller chunks (avoid OOM from KV growth)

The result: the engine self-tunes. Under light load, long prompts prefill quickly. Under heavy
concurrent load, it slows down new prefills to keep existing responses flowing.

### How It Works: Code Path

1. **Scheduler (`schedule()`)** sets `req.chunk_start` and `req.chunk_end` before each step:
   ```python
   req.chunk_start = req.tokens_prefilled           # where we left off
   req.chunk_end = min(req.prompt_len, tokens_prefilled + chunk_size)
   ```

2. **Engine (`_prefill_one()`)** runs the forward pass on only the chunk:
   ```python
   input_ids = prompt_token_ids[chunk_start:chunk_end]
   positions = arange(chunk_start, chunk_end)       # absolute positions
   ```

3. **Attention layer** stores K/V at the correct slot offset (`start_slot = chunk_start`)
   and, for non-first chunks, gathers prior K/V to compute a rectangular causal attention mask.

4. If `chunk_end < prompt_len`: call `scheduler.on_chunk_complete()`, no token sampled.
   If `chunk_end == prompt_len`: sample first token, call `on_prefill_complete()`.

5. Request stays in `PREFILLING` status across multiple steps until the final chunk.

**Key files:** `src/scheduler/scheduler.py`, `src/engine/inference_engine.py`, `src/model/attention.py`

### Chunked Prefill Attention (The Math)

For chunk `[chunk_start : chunk_end]` attending to context `[0 : chunk_end]`:

- **Prior tokens** `[0 : chunk_start]`: gathered from KV cache, fully visible (causal OK)
- **Current chunk** `[chunk_start : chunk_end]`: fresh, causal mask within chunk

Mask shape: `[chunk_len, chunk_start + chunk_len]`

```
chunk_start=256, chunk_len=256:
query position 256 → can see keys 0..256
query position 257 → can see keys 0..257
...
query position 511 → can see keys 0..511
```

This is implemented in `_chunked_prefill_attention()` in `src/model/attention.py`.

---

## Feature 2: Prefix / Prompt Caching (Radix Attention)

### The Idea

Cache the KV blocks for completed prefills so future requests with the same prefix can **skip
prefill entirely** for those tokens. The second request with the same 500-token system prompt:
- Gets those 500 tokens' KV blocks from cache: instant
- Only needs to prefill the unique suffix (the actual question)
- TTFT drops from ~500ms to ~50ms

### Data Structure: Prefix Trie

The cache is a trie where:
- Each node represents **one block's worth of tokens** (block_size = 16 tokens in this engine)
- Each node key is `tuple(token_ids for that block)`
- Each leaf/node stores the physical block ID in the KV cache

```
Root
├── (tok_ids 0-15)  → block 7   [system prompt block 1]
│   └── (tok_ids 16-31) → block 12  [system prompt block 2]
│       └── (tok_ids 32-47) → block 3   [system prompt block 3]
│           ├── (user question A) → block 18  [request A's suffix]
│           └── (user question B) → block 22  [request B's suffix]
└── (different tok_ids) → block 9  [different prefix]
```

### Cache Operations

**Match** — find the longest cached prefix:
```python
def match(token_ids) -> (num_matched, block_ids):
    # Walk trie one block at a time
    # Return longest chain where every block matches
    # num_matched is always a multiple of block_size
```

**Insert** — add newly computed blocks:
```python
def insert(token_ids, block_ids):
    # Only insert complete blocks (no partial trailing block)
    # Traverse existing nodes, add new nodes for uncached suffix
```

**Evict (LRU)** — free blocks when memory is tight:
```python
def evict_lru(n_blocks) -> freed_block_ids:
    # Post-order traversal: find leaf nodes with ref_count == 0
    # Sort by last_used (oldest first)
    # Evict leaf-only: removing an internal node would break shorter prefixes
```

**Pin/Unpin** — ref counting prevents eviction of in-use blocks:
```python
# On request admission (after match): pin(matched_block_ids)
# After prefill complete (or on preemption): unpin(matched_block_ids)
```

**File:** `src/memory/prefix_cache.py`

### Integration with the Scheduler

When a new request is admitted:
```python
# 1. Query cache
match_len, cached_blocks = prefix_cache.match(req.prompt_token_ids)

# 2. Set prefix state on request
req.prefix_match_len = match_len
req.cached_block_ids = cached_blocks
prefix_cache.pin(cached_blocks)

# 3. Only allocate blocks for the uncached suffix
suffix_blocks_needed = num_required_blocks(prompt_len - match_len + max_tokens)
suffix_blocks = block_manager.allocate(req.request_id, suffix_blocks_needed)

# 4. Full block table = cached prefix + new suffix
req.block_table = cached_blocks + suffix_blocks

# 5. Start prefill from the end of the cached prefix
req.tokens_prefilled = match_len
```

After prefill completes, newly computed blocks are inserted into the trie:
```python
prefix_cache.insert(req.prompt_token_ids, req.block_table[:num_prompt_blocks])
block_manager.mark_cached(new_prompt_blocks)
```

`mark_cached` ensures these blocks survive `block_manager.free(request_id)` — they're now owned
by the prefix cache, not the request.

### Block Ownership

A block can have three owners at different times:
1. **Active request (suffix blocks):** freed when request finishes
2. **Prefix cache (prompt blocks):** freed by LRU eviction only
3. **Shared (prefix blocks being used by a live request):** pinned, can't be evicted

This is tracked via:
- `block_manager._cached_block_ids: Set[int]` — which blocks belong to the prefix cache
- `block_manager.free(request_id)` skips blocks in `_cached_block_ids`
- `prefix_cache.pin/unpin` adjusts `node.ref_count`

### Cache Eviction Before Preemption

When memory is tight, the scheduler tries prefix cache eviction **before** falling back to
request preemption:

```python
if not block_manager.can_allocate(blocks_needed):
    shortage = blocks_needed - block_manager.num_free_blocks()
    freed_ids = prefix_cache.evict_lru(shortage)  # free cached blocks
    block_manager.unmark_cached(freed_ids)         # return them to free pool

if block_manager.can_allocate(blocks_needed):
    # proceed normally, no preemption needed
else:
    # fall back to preemption
```

---

## How the Two Features Compose

The two features share the `tokens_prefilled` cursor on each Request:

```
req.tokens_prefilled:
  - initialized to 0 (no prefix match, no chunking)
  - initialized to prefix_match_len (prefix match — skip those tokens)
  - advanced by chunk_size each step (chunked prefill)
```

| Scenario | `prefix_match_len` | `tokens_prefilled` starts at | First chunk |
|---|---|---|---|
| No features | 0 | 0 | `[0, prompt_len)` all at once |
| Chunked only | 0 | 0 | `[0, chunk_size)` |
| Prefix only | N (aligned) | N | `[N, prompt_len)` all at once |
| Both together | N (aligned) | N | `[N, min(prompt_len, N+chunk_size))` |

The attention layer writes K/V at `start_slot = chunk_start = tokens_prefilled`, which is always
correct because:
- Prefix blocks occupy slots `[0, prefix_match_len)` (already in the block table)
- New prefill writes start at `prefix_match_len` and advance from there

---

## Observability

Both features add lifecycle events visible in `/debug/lifecycle`:

**Chunked prefill events:**
```
PREFILL_CHUNK_START  chunk_start=0    chunk_end=256  prompt_len=2000
CHUNK_DONE           chunk_end=256    elapsed_ms=45.2
PREFILL_CHUNK_START  chunk_start=256  chunk_end=512  ...
...
PREFILL_DONE         first_token=1234  ttft_ms=360.1
```

**Prefix cache events:**
```
ADMITTED  blocks_allocated=3  prefix_matched=512  running=4
PREFILL_CHUNK_START  chunk_start=512  chunk_end=768  ...
```

Check cache stats: `GET /debug/prefix_cache`
```json
{
  "enabled": true,
  "cached_blocks": 32,
  "hit_count": 47,
  "miss_count": 3,
  "hit_rate_pct": 94.0
}
```
