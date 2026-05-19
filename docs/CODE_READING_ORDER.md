# Code Reading Order

Start here if you want to understand the codebase from the ground up. Each stage builds on the previous one.

---

## Stage 1 — Foundations (config + shared types)

These two files define every data structure used across the whole codebase. Read them first so nothing surprises you later.

| # | File | What you get |
|---|------|-------------|
| 1 | `src/core/config.py` | `ModelConfig`, `EngineConfig`, `ServerConfig` — all tunable knobs (block size, chunked prefill, prefix caching, device, etc.) |
| 2 | `src/core/types.py` | `Request`, `SequenceStatus`, `SamplingParams`, `AttentionMetadata`, `SchedulerOutput` — the shared data model that flows through every layer |

---

## Stage 2 — Memory subsystem (KV cache + block management)

The entire engine is built around paged memory. Understand this before touching the model or scheduler.

| # | File | What you get |
|---|------|-------------|
| 3 | `src/memory/kv_cache.py` | Pre-allocated tensor storage for K/V pairs. Understand the `[layer, kv, block, slot, head, head_dim]` layout and `store_tokens` / `gather_tokens` |
| 4 | `src/memory/block_manager.py` | Physical block allocator — allocate, free, fork (copy-on-write for prefix sharing) |
| 5 | `src/memory/prefix_cache.py` | Trie-based prompt cache — how repeated prefixes reuse existing blocks instead of re-computing |

---

## Stage 3 — Model implementation

Custom forward pass that injects paged attention without monkey-patching HuggingFace.

| # | File | What you get |
|---|------|-------------|
| 6 | `src/model/rope.py` | Rotary positional embeddings (RoPE) — position encoding used in every attention layer |
| 7 | `src/model/layers.py` | Low-level primitives: `rms_norm`, `swiglu_mlp` |
| 8 | `src/model/attention.py` | `PagedAttentionLayer` — the core: uses block tables + KVCacheManager instead of naive full-context attention |
| 9 | `src/model/transformer.py` | Full `LlamaForCausalLM` — stacks all layers, runs the forward pass, returns logits |
| 10 | `src/model/sampling.py` | Token sampling: temperature, top-p, top-k |
| 11 | `src/model/loader.py` | Loads HuggingFace weights into our custom model |

---

## Stage 4 — Scheduler

Decides which requests run, in what order, and when to preempt.

| # | File | What you get |
|---|------|-------------|
| 12 | `src/scheduler/request_queue.py` | Priority queue for incoming requests |
| 13 | `src/scheduler/scheduler.py` | Continuous batching scheduler — prefill/decode separation, block allocation, preemption, chunked prefill logic |

---

## Stage 5 — Engine (the orchestrator)

The main loop that ties everything together.

| # | File | What you get |
|---|------|-------------|
| 14 | `src/engine/inference_engine.py` | `LLMEngine` — runs in a background thread; each step: drain input queue → schedule → prefill forward pass → decode forward pass → signal callers |

---

## Stage 6 — API layer

HTTP interface on top of the engine.

| # | File | What you get |
|---|------|-------------|
| 15 | `src/api/schemas.py` | Pydantic request/response models (OpenAI-compatible) |
| 16 | `src/api/routes/health.py` | `/health` endpoint |
| 17 | `src/api/routes/models.py` | `/v1/models` endpoint |
| 18 | `src/api/routes/completions.py` | `/v1/completions` — streaming and non-streaming |
| 19 | `src/api/routes/chat.py` | `/v1/chat/completions` |
| 20 | `src/api/routes/debug.py` | `/debug/*` — exposes scheduler state, block usage, request lifecycle |
| 21 | `src/api/app.py` | FastAPI app factory — wires routes + middleware + engine |

---

## Stage 7 — Observability

Read these alongside the engine and API, or after.

| # | File | What you get |
|---|------|-------------|
| 22 | `src/observability/metrics.py` | `MetricsCollector` — TTFT, TPOT, throughput, queue depth |
| 23 | `src/observability/prompt_logger.py` | Structured lifecycle event logging per request |

---

## Stage 8 — Scripts (how to run it)

| # | File | What you get |
|---|------|-------------|
| 24 | `scripts/download_model.py` | Download and cache a model from HuggingFace |
| 25 | `scripts/run_server.py` | Entry point — parses CLI args, builds configs, starts engine + uvicorn |
| 26 | `scripts/quick_test.py` | Smoke test: send a few requests to a running server |

---

## Stage 9 — Benchmarks

| # | File | What you get |
|---|------|-------------|
| 27 | `benchmarks/benchmark.py` | Load test — concurrent requests, latency/throughput measurements |

---

## Stage 10 — Tests (read to understand edge cases)

| # | File | What it tests |
|---|------|--------------|
| 28 | `tests/test_block_manager.py` | Block alloc/free/fork |
| 29 | `tests/test_prefix_cache.py` | Prefix trie hit/miss/eviction |
| 30 | `tests/test_rope.py` | RoPE correctness |
| 31 | `tests/test_scheduler.py` | Scheduling decisions, preemption |
| 32 | `tests/test_chunked_prefill.py` | Chunked prefill correctness |
| 33 | `tests/test_engine.py` | End-to-end engine integration |

---

## Quick reference — dependency graph

```
config + types
    └── memory (kv_cache → block_manager → prefix_cache)
            └── model (rope → layers → attention → transformer → sampling → loader)
                    └── scheduler (request_queue → scheduler)
                            └── engine (inference_engine)
                                    └── api (schemas → routes → app)
                                            └── scripts / benchmarks
```
