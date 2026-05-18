# LLM Serving Engine — Implementation Plan

## Overview

A production-grade LLM inference engine targeting MacBook M2 (MPS/CPU), implementing every core vLLM feature from scratch. Built with PyTorch + FastAPI.

**Target model**: TinyLlama-1.1B-Chat-v1.0  
**Target hardware**: Apple M2 (MPS) or any CPU  
**API**: OpenAI-compatible REST API  

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                        │
│  POST /v1/completions   POST /v1/chat/completions        │
│  GET  /v1/models        GET  /health    GET /v1/stats    │
└────────────────────────┬────────────────────────────────┘
                         │ submit / stream
┌────────────────────────▼────────────────────────────────┐
│                  Inference Engine                        │
│  ┌─────────────┐  ┌─────────────────────────────────┐   │
│  │  Scheduler  │  │       Worker / Step Loop         │   │
│  │             │  │  prefill_step() → decode_step()  │   │
│  │  waiting    │  └────────────┬────────────────────┘   │
│  │  running    │               │                         │
│  │  preempted  │               │ forward()               │
│  └─────────────┘  ┌────────────▼────────────────────┐   │
│                   │     LlamaForCausalLM             │   │
│                   │  (custom forward pass)           │   │
│                   │  RoPE · PagedAttention · MLP     │   │
│                   └────────────┬────────────────────┘   │
│  ┌────────────────┐            │ read/write KV           │
│  │  Block Manager │  ┌─────────▼──────────────────────┐ │
│  │  free_blocks[] │  │       KV Cache Manager          │ │
│  │  block_table{} │  │  kv_storage[layers,2,blocks,…]  │ │
│  └────────────────┘  └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Core vLLM Features Implemented

| Feature | File | Notes |
|---------|------|-------|
| PagedAttention | `src/model/attention.py` | Block-level KV gather + attention |
| Physical Block Manager | `src/memory/block_manager.py` | Free list, alloc, free, preempt |
| KV Cache Manager | `src/memory/kv_cache.py` | Pre-allocated tensor, store/gather |
| Continuous Batching | `src/scheduler/scheduler.py` | Token-level scheduling |
| Prefill / Decode separation | `src/engine/inference_engine.py` | Different code paths |
| Request preemption | `src/scheduler/scheduler.py` | Free blocks, re-queue |
| Priority scheduling | `src/scheduler/request_queue.py` | heapq with FIFO tie-breaking |
| RoPE (from scratch) | `src/model/rope.py` | Custom sin/cos cache |
| GQA (Grouped Query Attn) | `src/model/attention.py` | repeat_interleave for KV heads |
| Token sampling | `src/model/sampling.py` | greedy, temperature, top-p, top-k |
| Streaming SSE | `src/api/routes/chat.py` | Per-token SSE chunks |
| OpenAI-compat API | `src/api/` | /v1/completions, /v1/chat/completions |
| Metrics: TTFT, TPOT | `src/observability/metrics.py` | Tracked per request |
| Throughput tracking | `src/observability/metrics.py` | tokens/sec window |
| Model loading (HF) | `src/model/loader.py` | safetensors + config.json |
| Custom transformer | `src/model/transformer.py` | Full Llama forward, our weights |

---

## Folder Structure

```
llm-serving-engine/
├── plan/
│   └── IMPLEMENTATION_PLAN.md          ← this file
│
├── src/
│   ├── core/
│   │   ├── config.py                   # ModelConfig, EngineConfig, ServerConfig
│   │   └── types.py                    # Request, SchedulerOutput, SamplingParams, etc.
│   │
│   ├── model/
│   │   ├── loader.py                   # HuggingFace weight loading + tokenizer
│   │   ├── rope.py                     # RotaryEmbedding from scratch
│   │   ├── attention.py                # PagedAttention (prefill + decode paths)
│   │   ├── layers.py                   # RMSNorm, SwiGLU MLP
│   │   ├── transformer.py              # Full LlamaForCausalLM (custom forward)
│   │   └── sampling.py                 # greedy / temperature / top-p / top-k
│   │
│   ├── memory/
│   │   ├── block_manager.py            # PhysicalBlockManager
│   │   └── kv_cache.py                 # KVCacheManager (pre-alloc tensor)
│   │
│   ├── scheduler/
│   │   ├── request_queue.py            # PriorityRequestQueue
│   │   └── scheduler.py               # ContinuousBatchingScheduler
│   │
│   ├── engine/
│   │   └── inference_engine.py        # LLMEngine (orchestrates everything)
│   │
│   ├── observability/
│   │   └── metrics.py                 # MetricsCollector (TTFT, TPOT, throughput)
│   │
│   └── api/
│       ├── schemas.py                  # Pydantic schemas (OpenAI-compatible)
│       ├── app.py                      # FastAPI app factory
│       └── routes/
│           ├── completions.py          # POST /v1/completions
│           ├── chat.py                 # POST /v1/chat/completions (+ streaming)
│           ├── models.py               # GET /v1/models
│           └── health.py              # GET /health, GET /v1/stats
│
├── tests/
│   ├── test_rope.py
│   ├── test_block_manager.py
│   ├── test_scheduler.py
│   └── test_engine.py
│
├── benchmarks/
│   └── benchmark.py                   # TTFT, TPOT, throughput sweep
│
├── scripts/
│   ├── download_model.py              # Download TinyLlama from HuggingFace
│   ├── run_server.py                  # Start the server
│   └── quick_test.py                  # Sanity-check inference
│
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

### 1. Custom Transformer Forward Pass
We load weights from HuggingFace safetensors but implement our own forward pass. This lets us inject our KV cache management (PagedAttention) at the attention layer, which HF's API does not support directly.

### 2. Block-Based KV Cache (PagedAttention)
Pre-allocate one large tensor: `kv_storage[num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]`. The PhysicalBlockManager tracks which physical blocks are free. Each request gets a logical block table mapping logical slot → physical block ID. On attention, we gather physical blocks and compute attention over the concatenated K/V.

### 3. Continuous Batching
The scheduler runs an inner loop:
1. Move finished decode requests out of `running`.
2. Move eligible `waiting` requests into `running` (allocate blocks for them).
3. If memory pressure, preempt lowest-priority decode request.
4. Return `(prefill_requests, decode_requests)` for this step.

The engine processes prefill and decode in one step. Prefill uses a causal mask; decode reads accumulated KV from the cache.

### 4. MPS Compatibility
- Use `float32` on MPS by default (some `float16` ops are unstable on older MPS).
- Avoid `torch.einsum` on critical paths — use `torch.bmm` / `torch.matmul` explicitly.
- Auto-detect device: `mps` → `cpu`.

### 5. Threading Model
The engine loop runs in a background thread. FastAPI submits requests via a thread-safe `queue.Queue` and waits for results via `queue.Queue` (blocking) or an async generator (streaming).

---

## Memory Budget (M2, TinyLlama-1.1B)

```
Model weights (float32):   ~4.4 GB
KV cache (256 blocks):
  22 layers × 2 × 256 × 16 × 4 × 64 × 4 bytes = ~176 MB
Total:                     ~4.6 GB  (M2 8GB has headroom)
```

With float16 weights (loaded as float32 for MPS stability), ~2.2 GB for weights.

---

## Statistics Exposed by /v1/stats

- `ttft_ms`: Time-to-First-Token (prefill latency) per request
- `tpot_ms`: Time-per-Output-Token (decode step latency)
- `throughput_tok_s`: Output tokens per second (sliding window)
- `num_running_requests`: Active requests in engine
- `num_waiting_requests`: Queued, waiting for blocks
- `kv_cache_blocks_used`: Physical blocks in use
- `kv_cache_blocks_free`: Physical blocks available
- `total_requests_served`: Lifetime counter

---

## Implementation Phases

| Phase | Description | Files |
|-------|-------------|-------|
| 1 | Core types & config | `core/` |
| 2 | Model: RoPE, attention, layers, transformer | `model/` |
| 3 | Memory: block manager, KV cache | `memory/` |
| 4 | Scheduler: queue, continuous batching | `scheduler/` |
| 5 | Engine: inference loop, prefill/decode | `engine/` |
| 6 | Observability: TTFT, TPOT, throughput | `observability/` |
| 7 | API: OpenAI-compat endpoints | `api/` |
| 8 | Tests, benchmarks, scripts | `tests/`, `benchmarks/`, `scripts/` |
