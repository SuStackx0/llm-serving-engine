# LLM Serving Engine

A from-scratch LLM inference engine implementing the core algorithms behind [vLLM](https://github.com/vllm-project/vllm) — PagedAttention, continuous batching, adaptive chunked prefill, and prefix caching — running on Apple Silicon (MPS) with an OpenAI-compatible FastAPI server.

> Built to understand and demonstrate how production LLM serving systems work at the algorithm level, not to wrap an existing framework.

---

## Traditional Serving vs This Engine

This is the core problem. When you call `model.generate()` from HuggingFace for multiple users, it works like this:

**Traditional (HuggingFace `generate`):**
```
User A ──► [prefill A] ──► [decode A token 1] ──► [decode A token 2] ──► ... ──► done
User B                                                                              ──► [prefill B] ──► ...
User C                                                                                                   ──► ...
```
- Requests are served **sequentially** — User B waits for User A to fully finish
- Memory is allocated as one **contiguous chunk** per request — fragments badly under load
- The GPU sits **mostly idle** during memory-bound decode steps
- No reuse of shared computation (same system prompt re-computed every time)

**This engine:**
```
Step 1:  [prefill A chunk 1]  [decode B tok 3]  [decode C tok 1]
Step 2:  [prefill A chunk 2]  [decode B tok 4]  [decode C tok 2]  [decode D tok 1]
Step 3:  [prefill A: done]    [decode B tok 5]  [decode C tok 3]  [decode D tok 2]
Step 4:  [decode A tok 1]     [decode B tok 6]  [decode C tok 4]  [decode D tok 3]
```
- All requests run **concurrently** — every engine step processes multiple users
- KV cache is **paged** (fixed-size blocks, no fragmentation, pre-allocated)
- Long prompts are **chunked** so they don't block shorter requests
- Shared prompt prefixes are **cached** — never re-computed across requests

---

## Benchmark — Real Numbers (Apple M-series, MPS, TinyLlama-1.1B)

5 concurrent requests × 64 max tokens, temperature=0.8:

```
  Metric            HuggingFace (sequential)    This engine (concurrent)    Delta
  ─────────────────────────────────────────────────────────────────────────────
  Wall time                        12.28s                      5.15s        -58%
  Throughput                   10.8 tok/s                 15.5 tok/s        +44%
  TTFT (mean)                     727.0ms                    286.2ms        -61%
  TPOT (mean)                     161.7ms                    110.3ms        -32%
  ─────────────────────────────────────────────────────────────────────────────
  Hardware: Apple M-series (MPS)  |  Model: TinyLlama-1.1B-Chat  |  float32
```

**What each number means:**
- **Wall time** — total clock time to finish all 5 requests. Engine is 2.4× faster because requests run in parallel.
- **Throughput** — output tokens per second across all requests. Higher means the GPU is better utilized.
- **TTFT (Time To First Token)** — how long the user waits before seeing any output. Engine is 61% faster because chunked prefill lets decode steps start sooner.
- **TPOT (Time Per Output Token)** — latency of each subsequent token. Engine is 32% faster because continuous batching amortizes the decode cost across requests.

Run it yourself:
```bash
python benchmarks/benchmark.py --num-requests 5 --max-tokens 64
```

---

## What this implements

### From the vLLM paper (Kwon et al., 2023)
| Component | What it does | File |
|---|---|---|
| **PagedAttention** | KV cache in fixed-size blocks — eliminates memory fragmentation | `src/model/attention.py` |
| **Physical Block Manager** | Free-list allocator; any free block fits any request | `src/memory/block_manager.py` |
| **KV Cache** | Pre-allocated `[layers, 2, blocks, block_size, kv_heads, head_dim]` tensor | `src/memory/kv_cache.py` |
| **Continuous Batching** | Batch composition changes every step; no padding, no waiting | `src/scheduler/scheduler.py` |
| **Prefill / Decode separation** | Different compute profiles handled in separate passes per step | `src/engine/inference_engine.py` |
| **Preemption** | Evicts lowest-priority decode request when memory is tight | `src/scheduler/scheduler.py` |
| **GQA (Grouped-Query Attention)** | TinyLlama's 32-head/4-KV-head architecture handled natively | `src/model/attention.py` |
| **RoPE from scratch** | Rotary positional embeddings with cached sin/cos tables | `src/model/rope.py` |
| **Custom Llama forward pass** | Loads HuggingFace weights into our own transformer impl | `src/model/transformer.py` |
| **Token sampling** | Greedy, temperature, top-k, top-p | `src/model/sampling.py` |

### Beyond vLLM — novel additions
| Feature | What makes it different | File |
|---|---|---|
| **Adaptive Chunked Prefill** | Chunk size self-tunes based on decode queue depth + memory pressure. vLLM has a fixed chunk size. Long prompts no longer head-of-line block short requests. | `src/scheduler/scheduler.py` |
| **Prefix / Prompt Caching** | Trie-based KV block reuse for shared prefixes (system prompts, few-shot examples). LRU leaf eviction with ref-counted pinning. | `src/memory/prefix_cache.py` |
| **Per-request lifecycle tracing** | Every state transition logged with timestamps: SUBMITTED → QUEUED → ADMITTED → PREFILL_CHUNK × N → DECODE_STEP × N → FINISHED | `src/core/types.py` |
| **Debug endpoints** | `/debug/lifecycle`, `/debug/batch`, `/debug/prefix_cache` — observe the engine internals live | `src/api/routes/debug.py` |

---

## How a request flows through the engine

```
Client POST /v1/chat/completions
        │
        ▼
  FastAPI route          tokenize → Request object → input queue
        │
        ▼
  LLMEngine (background thread, ~1ms cadence)
  │
  ├── Scheduler.schedule()  — runs every step
  │   ├── 1. Free finished requests' KV blocks
  │   ├── 2. Ensure decode requests have a block for their next token (or preempt)
  │   ├── 3. Admit waiting requests
  │   │       ├── PrefixTrieCache.match() → find cached prefix, skip its prefill
  │   │       └── BlockManager.allocate() → only allocate blocks for the suffix
  │   └── 4. Set chunk window for each PREFILLING request (adaptive chunk size)
  │
  ├── _step_prefill()  — for each prefilling request this step
  │   └── forward pass on [chunk_start : chunk_end] tokens only
  │       └── PagedAttentionLayer: write K/V at start_slot=chunk_start,
  │           gather prior K/V if mid-stream, apply rectangular causal mask
  │
  └── _step_decode()  — for each decoding request this step
      └── forward pass on [last_token] only
          └── PagedAttentionLayer: gather full K/V from scattered blocks,
              compute single-query attention, write new K/V, sample next token
        │
        ▼
  token pushed to per-request queue  →  SSE stream or blocking response
```

**KV cache memory layout** — pre-allocated at startup, zero fragmentation:
```
kv_storage[22 layers, 2 (K/V), 256 blocks, 16 tokens/block, 4 KV heads, 64 head_dim]
           └──────────────────────────────────────────────────────────────────────────┘
                                    184.5 MB on MPS (float32)
```

---

## API

OpenAI-compatible. Works as a drop-in replacement for the OpenAI client pointed at `localhost:8000`.

```
POST /v1/chat/completions    streaming (SSE) and non-streaming
POST /v1/completions
GET  /v1/models
GET  /v1/stats               TTFT · TPOT · throughput · KV utilization · prefix cache hit rate
GET  /health

POST /debug/lifecycle        full event trace for a single request (every state transition)
POST /debug/batch            submit N prompts, see how the batcher handled each engine step
GET  /debug/prefix_cache     trie depth, cached blocks, hit/miss counts, hit rate
```

---

## Quick start

```bash
pip install -r requirements.txt

# Start server (downloads TinyLlama ~2.2 GB on first run)
python scripts/run_server.py --device mps

# Call it like OpenAI
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama",
    "messages": [{"role": "user", "content": "What is PagedAttention?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Watch the engine internals for a single request
curl -X POST http://localhost:8000/debug/lifecycle \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain transformers", "max_tokens": 50}'

# Run the full comparison benchmark
python benchmarks/benchmark.py --num-requests 5 --max-tokens 64

# Run benchmark + prefix cache demo
python benchmarks/benchmark.py --prefix-demo
```

---

## Tests

```bash
pytest tests/ -v
```

Six test suites covering block manager, scheduler, RoPE, engine integration, chunked prefill, and prefix cache.

---

## Project layout

```
src/
  core/           Request dataclass, SequenceStatus, AttentionMetadata, EngineConfig
  engine/         LLMEngine — background loop, prefill/decode orchestration
  memory/         PhysicalBlockManager, KVCacheManager, PrefixTrieCache
  model/          RoPE, PagedAttentionLayer, SwiGLU MLP, LlamaForCausalLM, sampling
  scheduler/      ContinuousBatchingScheduler, PriorityRequestQueue
  observability/  MetricsCollector (TTFT/TPOT/throughput), PromptLogger
  api/            FastAPI app, OpenAI-compatible routes, debug routes
tests/            pytest unit + integration tests
benchmarks/       HF vs engine comparison benchmark
scripts/          run_server.py, quick_test.py, download_model.py
docs/             Theory docs, codebase walkthrough, honest resume guide
```

---

## Docs

| File | Covers |
|---|---|
| `docs/01_inference_engines_101.md` | Autoregressive generation, KV cache, batching — the theory |
| `docs/02_paged_attention_and_scheduling.md` | PagedAttention memory layout, scheduler algorithm, GQA, sampling |
| `docs/03_chunked_prefill_and_prefix_caching.md` | How both novel features work, the math, code paths |
| `docs/04_codebase_walkthrough.md` | Every file explained + full end-to-end request trace |
| `docs/05_resume_and_metrics.md` | Honest benchmark interpretation, what to say in interviews |
