# 04 — Codebase Walkthrough: Every File Explained

## Directory Structure

```
llm-serving-engine/
├── src/
│   ├── core/          # shared types and config
│   ├── engine/        # main orchestrator (the "brain")
│   ├── memory/        # block manager, KV cache, prefix cache
│   ├── model/         # transformer layers, attention, sampling
│   ├── scheduler/     # continuous batching scheduler
│   ├── observability/ # metrics, structured logging
│   └── api/           # FastAPI routes and schemas
├── tests/             # unit tests per subsystem
├── scripts/           # server runner, quick test
├── benchmarks/        # throughput benchmark
└── docs/              # you are here
```

---

## `src/core/`

### `src/core/types.py`

Every major data structure lives here. The most important is `Request`:

```python
@dataclass
class Request:
    prompt: str                          # original text
    prompt_token_ids: List[int]          # tokenized prompt
    sampling_params: SamplingParams      # temp, top_k, top_p, max_tokens
    request_id: str                      # UUID
    arrival_time: float                  # for queue ordering
    status: SequenceStatus               # state machine (see below)
    output_token_ids: List[int]          # generated tokens so far
    block_table: List[int]               # logical → physical block mapping
    num_cached_tokens: int               # tokens in KV cache (prompt + generated)
    tokens_prefilled: int                # prompt tokens processed so far (chunked prefill)
    chunk_start: int                     # current chunk window start
    chunk_end: int                       # current chunk window end
    prefix_match_len: int                # tokens reused from prefix cache
    cached_block_ids: List[int]          # borrowed prefix cache blocks
    prefill_start_time: float            # for TTFT measurement
    first_token_time: float              # for TTFT measurement
    last_token_time: float               # for TPOT measurement
    _token_queue: queue.Queue            # engine → caller communication
    lifecycle: List[LifecycleEvent]      # full event trace
```

`SequenceStatus` state machine:
```
WAITING → PREFILLING → DECODING → FINISHED_EOS
                                  FINISHED_LENGTH
                                  FINISHED_STOP
         ↑ PREEMPTED ────────────┘
```

`AttentionMetadata` carries per-step batch info to each transformer layer:
```python
@dataclass
class AttentionMetadata:
    prefill_seq_lens: List[int]           # chunk length per prefill request
    prefill_block_tables: List[List[int]] # block tables for prefill requests
    decode_context_lens: List[int]        # context length per decode request
    decode_block_tables: List[List[int]]  # block tables for decode requests
    prefill_chunk_starts: Optional[List[int]]  # slot offset per prefill (chunked prefill)
```

### `src/core/config.py`

Two dataclasses:

`ModelConfig` — loaded from HuggingFace `config.json`:
- `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`
- `head_dim = hidden_size / num_attention_heads`
- `gqa_ratio = num_attention_heads / num_key_value_heads`

`EngineConfig` — server runtime config:
- `block_size=16`, `num_blocks=256` — KV cache block layout
- `max_running_requests=8`, `max_waiting_requests=256`
- `enable_chunked_prefill=True`, `max_chunk_size=256`, `min_chunk_size=64`
- `enable_prefix_caching=True`
- `device="auto"` — resolves to MPS/CUDA/CPU at startup

---

## `src/memory/`

### `src/memory/block_manager.py` — `PhysicalBlockManager`

The free-list allocator. Conceptually a slab allocator for KV blocks.

Internal state:
```python
self._free: List[int]              # stack of available block IDs
self._owned: Dict[str, List[int]]  # request_id → its block IDs
self._cached_block_ids: Set[int]   # blocks owned by prefix cache (not freed on request end)
```

Key operations:
- `allocate(request_id, n)` → pops n blocks from `_free`, adds to `_owned[request_id]`
- `free(request_id)` → returns non-cached blocks to `_free`
- `mark_cached(block_ids)` → marks blocks as prefix cache property
- `unmark_cached(block_ids)` → returns cached blocks to `_free` (eviction path)

### `src/memory/kv_cache.py` — `KVCacheManager`

One giant pre-allocated tensor:
```python
self.storage = torch.zeros(
    num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim,
    device=device, dtype=dtype,
)
```

Operations:
- `store_tokens(layer_idx, block_table, keys, values, start_slot)` — write K/V to blocks
- `gather_tokens(layer_idx, block_table, num_tokens)` → read K/V from scattered blocks

The `block_table` translates logical positions to physical:
```python
slot = start_slot + i
block_idx = slot // block_size       # which block in the table
slot_in_block = slot % block_size    # position within that block
phys_block = block_table[block_idx]  # actual physical block ID
storage[layer_idx, 0, phys_block, slot_in_block] = key
```

### `src/memory/prefix_cache.py` — `PrefixTrieCache`

Trie keyed on token-block tuples. Stores block IDs for completed prompt prefixes.

```python
class PrefixTrieNode:
    children: Dict[tuple, PrefixTrieNode]  # key = 16-token tuple
    block_id: Optional[int]                # physical block in KV cache
    ref_count: int                         # 0 = evictable
    last_used: float                       # for LRU ordering
```

Methods: `match()`, `insert()`, `pin()`, `unpin()`, `evict_lru()`, `stats()`

Thread safety: `threading.Lock` around all mutations. `stats()` reads raw counters
(acceptable eventual consistency for monitoring).

---

## `src/scheduler/`

### `src/scheduler/scheduler.py` — `Scheduler`

The scheduling brain. One `schedule()` call per engine loop iteration.

State:
```python
self.waiting: PriorityRequestQueue   # heap sorted by (priority, arrival_time)
self.running: List[Request]          # all PREFILLING and DECODING requests
self.prefix_cache: PrefixTrieCache   # optional
```

Key methods:
- `add_request(req)` → status=WAITING, push to waiting queue
- `schedule()` → returns SchedulerOutput (see Step-by-step above)
- `on_prefill_complete(req)` → status=DECODING, insert into prefix cache
- `on_chunk_complete(req, chunk_end)` → advance `tokens_prefilled`
- `on_token_generated(req, tok, eos_id)` → append token, check stop conditions
- `compute_chunk_size()` → adaptive chunk size based on decode pressure + memory
- `_preempt(victim, output)` → free blocks, reset state, re-queue

### `src/scheduler/request_queue.py` — `PriorityRequestQueue`

Min-heap. Priority key = `(priority, arrival_time, sequence_counter)`.
Tie-break by arrival_time ensures FIFO within same priority level.

---

## `src/model/`

### `src/model/transformer.py` — `LlamaForCausalLM`

Full Llama forward pass:
```
Embedding → [Layer 0..N-1: RMSNorm → Attention → RMSNorm → MLP] → RMSNorm → LM Head
```

Each layer is a `LlamaLayer` containing:
- `PagedAttentionLayer` (from attention.py)
- RMSNorm (from layers.py)
- SwiGLU MLP (from layers.py)

### `src/model/attention.py` — `PagedAttentionLayer`

The most complex file. Handles both prefill and decode in one `forward()` call.

Key functions:
- `_chunked_prefill_attention(q, k_full, v_full, ..., chunk_start)` — handles any chunk
- `_prefill_attention(q, k, v, ...)` — thin wrapper for `chunk_start=0`
- `_decode_attention_single(q, k, v, ...)` — single-query over full context

The prefill loop in `forward()`:
1. Get chunk_start from `metadata.prefill_chunk_starts[i]`
2. Write K/V at `start_slot=chunk_start`
3. If `chunk_start > 0`: gather prior K/V from cache and prepend
4. Call `_chunked_prefill_attention` with the full context

### `src/model/rope.py` — `RotaryEmbedding`

RoPE positional encoding. Pre-computes `(cos, sin)` tables at startup.
`forward(x, positions)` applies 2D rotation based on absolute positions.
Works correctly for chunked prefill because positions are absolute (not relative to chunk).

### `src/model/sampling.py` — `sample_token`

`logits → token_id`. Applies temperature → top-k → top-p → multinomial sample (or argmax if temperature=0).

### `src/model/loader.py` — `load_model`

Downloads from HuggingFace Hub (or uses cached weights), parses `config.json` into `ModelConfig`,
returns `(state_dict, model_config, tokenizer)`.

---

## `src/engine/`

### `src/engine/inference_engine.py` — `LLMEngine`

The top-level orchestrator. Runs in a background thread.

Factory: `LLMEngine.from_config(model_cfg, engine_cfg)` — creates all components and wires them.

Public API:
- `start()` → launch background thread
- `submit(request)` → push to input queue, return output queue
- `generate(prompt, params)` → synchronous wrapper around submit()
- `stats()` → dict of all metrics

Private loop methods:
- `_run_loop()` → drain input → schedule → execute_step
- `_execute_step(sched_out)` → call `_step_prefill` then `_step_decode`
- `_prefill_one(req, batch_pos, batch_size)` → chunk-aware forward pass
- `_decode_one(req, batch_pos, batch_size)` → single decode step
- `_finalize_request(req)` → send None sentinel to output queue

---

## `src/observability/`

### `src/observability/metrics.py` — `MetricsCollector`

Thread-safe counters:
- `record_token()` — called after every output token (for throughput)
- `record_request_complete(req)` — called when request finishes
- `record_prefix_cache_hit/miss()` — called on first prefill chunk
- `snapshot()` → dict of all current metrics

### `src/observability/prompt_logger.py`

Structured log functions for each lifecycle event. Uses Python `logging` to `"llm.engine"` logger.

Log format: `[COMPONENT] [req-ID] EVENT_NAME  key=value key=value`

---

## `src/api/`

### `src/api/app.py` — `create_app(engine, log_level)`

FastAPI app factory. Configures logging, CORS middleware, attaches engine to `app.state.engine`,
registers all route routers.

### `src/api/routes/completions.py` — `/v1/completions`

OpenAI-compatible text completions. Supports streaming (SSE) and non-streaming.
Tokenizes `prompt`, creates `Request`, calls `engine.submit()`, collects output.

### `src/api/routes/chat.py` — `/v1/chat/completions`

OpenAI-compatible chat completions. Applies `tokenizer.apply_chat_template()` to convert
messages to a single string, then same flow as completions.

### `src/api/routes/health.py` — `/health`, `/v1/stats`

`/health` → simple OK check.
`/v1/stats` → `engine.stats()` formatted as `EngineStats` Pydantic model.

### `src/api/routes/debug.py` — `/debug/*`

`POST /debug/lifecycle` — submit one prompt, get back full lifecycle event trace.
`POST /debug/batch` — submit N prompts, see how continuous batching handled them.
`GET /debug/prefix_cache` — prefix cache hit/miss stats.

### `src/api/schemas.py`

All Pydantic request/response models. `EngineStats` includes optional prefix cache fields.

---

## `tests/`

| File | What it tests |
|---|---|
| `test_block_manager.py` | Allocation, free-list, can_allocate |
| `test_scheduler.py` | Admission, preemption, on_token_generated |
| `test_engine.py` | End-to-end generate() with real model |
| `test_rope.py` | RoPE correctness (relative position invariance) |
| `test_chunked_prefill.py` | Chunk windows, adaptive sizing, preemption reset |
| `test_prefix_cache.py` | Trie match/insert/evict, pin/unpin, scheduler integration |

---

## Request Flow: End to End

```
1. Client POSTs to /v1/chat/completions

2. chat.py route:
   - apply_chat_template() → flat string
   - tokenize → prompt_token_ids
   - create Request(...)
   - engine.submit(req) → returns out_q

3. engine._input_queue.put(req)

4. engine._run_loop() wakes up:
   a. _drain_input_queue() → scheduler.add_request(req)  [status=WAITING]
   b. scheduler.schedule():
      - Step 1: clean finished requests
      - Step 2: ensure decode requests have blocks
      - Step 3: admit req from waiting queue
        * prefix_cache.match() → prefix_match_len, cached_blocks
        * block_manager.allocate() → suffix blocks
        * req.block_table = cached_blocks + suffix_blocks
        * req.tokens_prefilled = prefix_match_len
        * req.status = PREFILLING
      - Step 4: assign chunk window:
        * req.chunk_start = tokens_prefilled
        * req.chunk_end = min(prompt_len, tokens_prefilled + chunk_size)
        * add to prefill_requests
   c. _step_prefill([req]):
      - _prefill_one(req):
        * forward pass on chunk[chunk_start:chunk_end]
        * if final chunk: sample first token, on_prefill_complete()
        * if not final: on_chunk_complete() → tokens_prefilled advances
   d. Repeat until final chunk processed
   e. _step_decode([req]) each step:
      - _decode_one(req): forward pass on last token, sample next token
      - push token to req._token_queue

5. Route handler: reads from out_q until None sentinel
   - decodes token_ids → text
   - returns CompletionResponse (or SSE stream)
```
