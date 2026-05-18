# LLM Serving Engine

A production-grade LLM inference engine implementing every core vLLM feature from scratch — PagedAttention, Continuous Batching, a custom transformer forward pass — running on **MacBook M2 (MPS/CPU)** with an **OpenAI-compatible FastAPI server**.

```
POST /v1/chat/completions   ← OpenAI-compatible, with SSE streaming
POST /v1/completions
GET  /v1/models
GET  /v1/stats              ← TTFT · TPOT · throughput · KV memory
GET  /health
```

---

## Features implemented (from the vLLM paper)

| Feature | Where |
|---------|-------|
| **PagedAttention** | `src/model/attention.py` |
| **Physical Block Manager** | `src/memory/block_manager.py` |
| **KV Cache (pre-allocated)** | `src/memory/kv_cache.py` |
| **Continuous Batching** | `src/scheduler/scheduler.py` |
| **Prefill / Decode separation** | `src/engine/inference_engine.py` |
| **Request preemption** | `src/scheduler/scheduler.py` |
| **Priority queue** | `src/scheduler/request_queue.py` |
| **RoPE from scratch** | `src/model/rope.py` |
| **GQA (Grouped-Query Attention)** | `src/model/attention.py` |
| **Token sampling** (greedy/temp/top-p/top-k) | `src/model/sampling.py` |
| **SSE streaming** | `src/api/routes/chat.py` |
| **OpenAI-compatible API** | `src/api/` |
| **TTFT / TPOT / throughput metrics** | `src/observability/metrics.py` |
| **Custom Llama forward pass** | `src/model/transformer.py` |

---

## Architecture

```
FastAPI  →  LLMEngine  →  Scheduler  →  PhysicalBlockManager
                    ↓                            ↓
              LlamaForCausalLM          KVCacheManager
                    ↓
         PagedAttentionLayer (per transformer layer)
                    ↓
            RotaryEmbedding (RoPE)
```

**Key design**: we load HuggingFace weights but implement our own forward pass, injecting block-based KV cache management at the attention layer. This is the same approach vLLM uses, minus the CUDA kernel (replaced by PyTorch gather ops that run on MPS/CPU).

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download model (~2.2 GB)
python scripts/download_model.py

# 3. Run sanity check (loads model, runs 3 concurrent prompts, prints stats)
python scripts/quick_test.py

# 4. Start the server
python scripts/run_server.py

# 5. Call it like OpenAI
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama",
    "messages": [{"role": "user", "content": "What is PagedAttention?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

---

## Run benchmarks

```bash
python benchmarks/benchmark.py --num-requests 8 --max-tokens 100
```

Sample output on M2 (8 concurrent requests):
```
  Total output tokens : 713
  Wall time           : 41.3s
  Throughput          : 17.3 tok/s
  TTFT                : 1840.2 ms (p50=1721.5)
  TPOT                : 312.1 ms (p50=298.4)
  KV blocks used/free : 0/256
```

---

## Run tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Memory layout

The KV cache is a single pre-allocated tensor:

```
kv_storage[num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]
            22         K/V  256         16           4             64
```

For TinyLlama on M2 with 256 blocks (float32):
- KV cache: ~176 MB
- Model weights: ~4.4 GB
- Total: ~4.6 GB (well within M2 8 GB)

---

## Project layout

```
src/
  core/          # Config dataclasses, Request/SchedulerOutput types
  model/         # RoPE, PagedAttention, SwiGLU MLP, full Llama forward pass
  memory/        # PhysicalBlockManager, KVCacheManager
  scheduler/     # PriorityRequestQueue, ContinuousBatchingScheduler
  engine/        # LLMEngine (background thread, prefill/decode loop)
  observability/ # MetricsCollector (TTFT, TPOT, throughput)
  api/           # FastAPI app, OpenAI-compatible routes
tests/           # pytest unit + integration tests
benchmarks/      # TTFT/TPOT/throughput measurement
scripts/         # download_model, run_server, quick_test
plan/            # Implementation plan
```

---

## Resume bullets

- Implemented **PagedAttention** from scratch: block-based KV cache with physical block manager eliminating memory fragmentation; KV stored in pre-allocated `[layers, 2, blocks, block_size, kv_heads, head_dim]` tensor
- Built **token-level continuous batching scheduler** with dynamic prefill/decode separation, priority queue, and preemption under memory pressure
- Implemented **Rotary Positional Embeddings (RoPE)** from first principles with cached sin/cos tables; verified relative-position invariance property mathematically
- Wrote **custom LlamaForCausalLM forward pass** loading HuggingFace weights into our own attention/MLP implementation, enabling block-based KV injection
- Served with **FastAPI + OpenAI-compatible API** (/v1/chat/completions with SSE streaming, /v1/stats with TTFT/TPOT/throughput)
