# 01 — What Is an LLM Inference Engine? (Theory)

## The Problem: Why Can't We Just Call the Model?

A large language model is a mathematical function. You feed it tokens, it outputs probabilities over the next token. Simple enough — so why does running it at scale require an entire engine?

Because at scale, three things collide:

1. **GPU memory is finite.** A 7B parameter model takes ~14 GB in fp16. A 70B model takes ~140 GB. Every request in flight also needs working memory (the KV cache — more on this). You can easily run out.
2. **Requests arrive continuously and finish at different times.** If you batch naively (wait for N requests, run them together, wait again), you waste GPU cycles whenever some requests finish early.
3. **Autoregressive generation is sequential.** Each new token requires the entire previous context. You cannot parallelize across output positions.

An inference engine's job is to hide all of this from the caller. You POST a prompt, you get a response. Behind that, the engine is doing sophisticated memory management, batching, and scheduling.

---

## Autoregressive Generation

Every modern LLM generates one token at a time. The algorithm:

```
output = []
context = tokenize(prompt)

while not done:
    logits = model(context)          # forward pass
    next_token = sample(logits[-1])  # pick the next token
    context.append(next_token)
    output.append(next_token)
    if next_token == EOS or len(output) >= max_tokens:
        break
```

**The first forward pass** processes the entire prompt — this is called **prefill**.
**Every subsequent pass** processes just one new token — this is called **decode**.

Prefill is compute-bound (many tokens at once, high GPU utilization).
Decode is memory-bandwidth-bound (one token, lots of cache reads).

---

## The KV Cache

During the forward pass, each transformer attention layer computes:
- **Key (K)** and **Value (V)** vectors for every token in the context

These K and V vectors are reused every decode step. Without caching, you'd recompute the K and V of the entire prompt on every single decode step — quadratic cost.

The **KV cache** stores these vectors so each decode step only computes K/V for the one new token.

**Memory cost:** For a 7B model with 32 layers, 32 heads, head_dim=128, and max context 2048:
```
KV cache = 2 × 32 layers × 2048 tokens × 32 heads × 128 dim × 2 bytes (fp16)
         ≈ 1 GB per request
```

With 8 concurrent requests: 8 GB just for KV cache. This is why memory management is central.

---

## The Two Phases in Detail

### Prefill

- Input: full prompt `[t₁, t₂, ..., tₙ]`
- Runs one forward pass through all transformer layers
- Stores K/V for all `n` positions in the KV cache
- Produces logits for position `n` → sample first output token
- Time complexity: O(n²) attention per layer

### Decode

- Input: just the last generated token `[tₙ₊₁]`
- Runs one forward pass
- Attention at this step: one query vector attends to all `n+1` cached K/V pairs
- Stores K/V for position `n+1` in the KV cache
- Produces next token
- Time complexity: O(n) per step (context grows each step)

---

## Batching: The Throughput Trick

If you run one request at a time, the GPU is mostly idle (decode is very low on compute utilization). The fix: run multiple requests simultaneously.

**Static batching** (naive): collect `B` requests, run them all together as a batch.  
Problem: if one request has prompt_len=10 and another has prompt_len=1000, you pad the short one to 1000 — wasted compute.

**Continuous batching** (vLLM's key innovation): the batch composition changes every step.  
- When request A finishes at step 50, immediately add request B to the batch at step 51.
- No padding, no waiting for a fixed batch to fill.
- GPU utilization is much higher.

This is the core innovation that made vLLM 2-24× faster than Hugging Face's naive serving.

---

## Memory Fragmentation: The Hidden Problem

Even with continuous batching, you hit a memory problem. Each request's KV cache grows dynamically — you don't know at the start how long the output will be.

**Naive approach:** allocate a contiguous tensor of size `max_context_length` for each request.  
Problem: a request that only generates 10 tokens still holds `max_context_length` worth of memory. Even if there are 1 GB of free memory scattered in small pieces, you can't allocate a new 1 GB contiguous tensor.

This is **memory fragmentation** — the same problem that plagues OS memory allocators.

**vLLM's solution:** PagedAttention (see doc 02).

---

## Summary

| Concept | Definition | Why It Matters |
|---|---|---|
| Autoregressive | Generate one token at a time, using all previous tokens | Defines the fundamental compute structure |
| Prefill | Process the full prompt in one forward pass | Expensive but done once per request |
| Decode | Generate each output token one at a time | Cheap per step but many steps |
| KV cache | Store attention K/V vectors to avoid recomputation | Essential for decode efficiency |
| Continuous batching | Change batch composition every step | 2-24× throughput over static batching |
| Memory fragmentation | Contiguous allocation fails with variable-length sequences | Why naive KV cache management runs OOM |
