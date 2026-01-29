# Vision

## What is llm-serving-engine?

llm-serving-engine is a **local-first inference and serving system** for Large Language Models.

It is not a model, not a fine-tuning framework, and not a thin wrapper over an existing runtime.
It is an **execution engine** focused on serving multiple inference requests efficiently on constrained hardware.

The core problem it addresses is:
> How do we serve LLMs locally the way production systems do — with batching, scheduling, streaming, and control — without requiring datacenter GPUs?

---

## Why this project exists

Most local LLM tools are designed around a **single prompt → single response** execution model.
This breaks down immediately when:
- Multiple users send requests
- Streaming is required
- Memory must be shared safely
- Throughput matters more than raw latency

Production systems solve this with complex serving engines.
Local setups usually do not.

This project fills that gap.

---

## Design philosophy

The system is built around the following principles:

1. **Serving-first, not model-first**
   - Models are treated as pluggable execution backends
   - The engine owns request lifecycle, not the model

2. **Explicit control over resources**
   - Token limits, memory budgets, batching size
   - No hidden magic, no uncontrolled growth

3. **Architecture before optimization**
   - Clean abstractions first
   - Hardware-specific optimizations come later

4. **Local hardware is a first-class target**
   - CPU and Apple Silicon environments are supported by design
   - No assumption of CUDA or multi-GPU setups

5. **Correctness over peak throughput**
   - Predictable behavior under load
   - Graceful degradation instead of crashes

---

## Intended users

This project is designed for:
- Engineers building local or on-prem LLM systems
- Researchers experimenting with serving strategies
- Developers who need OpenAI-style APIs without cloud dependency
- Anyone who wants to understand how real LLM serving systems work internally

---

## What success looks like

The project is successful if:
- Multiple concurrent requests can be served correctly
- Token streaming works reliably
- Batching improves throughput measurably
- The system remains understandable and hackable

Raw benchmark dominance is **not** the primary goal.

---

## Scope boundaries

This project intentionally avoids:
- Training or fine-tuning
- Distributed multi-node serving
- GPU kernel development
- Chasing benchmark leaderboards

Those concerns are orthogonal to the problem this engine is solving.
