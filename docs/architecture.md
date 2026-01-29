# Architecture

## Overview

llm-serving-engine is structured as a layered system where **serving concerns are separated from model execution**.

The architecture is designed to:
- Isolate hardware-specific logic
- Make scheduling and batching explicit
- Keep the request lifecycle observable and controllable

At a high level, the system is composed of five layers:

1. API Layer
2. Request Router
3. Scheduler
4. Batching Engine
5. Runtime Backends

---

## High-level data flow

Client Request
↓
API Layer (HTTP / OpenAI-compatible)
↓
Request Router
↓
Scheduler
↓
Batching Engine
↓
Runtime Backend
↓
Token Stream / Response


Each layer owns a single responsibility and communicates via well-defined interfaces.

---

## Layer responsibilities

### 1. API Layer

**Responsibility:**
- Expose HTTP endpoints
- Handle request validation
- Support OpenAI-compatible APIs
- Manage streaming responses

**Key characteristics:**
- Stateless
- No model awareness
- No batching or scheduling logic

---

### 2. Request Router

**Responsibility:**
- Normalize incoming requests into internal request objects
- Assign request IDs
- Track request lifecycle state

**Key characteristics:**
- Lightweight
- Does not make execution decisions
- Acts as a boundary between external and internal representations

---

### 3. Scheduler

**Responsibility:**
- Decide *when* a request should be executed
- Decide *which* requests are eligible for batching
- Enforce fairness and resource constraints

**Examples of scheduling policies:**
- First-come-first-served (FCFS)
- Token-budget–aware scheduling
- Priority-based scheduling

The scheduler operates on **requests**, not tokens.

---

### 4. Batching Engine

**Responsibility:**
- Combine multiple active requests into execution batches
- Advance generation one step at a time
- Handle continuous batching

**Key characteristics:**
- Token-level execution
- Requests may enter or leave batches dynamically
- Independent of model implementation

This is where throughput gains are primarily achieved.

---

### 5. Runtime Backends

**Responsibility:**
- Execute model forward passes
- Manage model-specific state (KV cache, logits)
- Produce next-token probabilities

**Examples:**
- CPU runtime
- GGUF runtime (via llama.cpp or equivalent)
- Future GPU or accelerator runtimes

Runtime backends are **pluggable** and must conform to a shared interface.

---

## Control and observability

Cross-cutting concerns include:
- Metrics (latency, throughput, memory usage)
- Logging
- Backpressure and rate limiting

These are implemented outside the core execution path to avoid coupling.

---

## Failure isolation

The architecture is designed to:
- Fail individual requests without crashing the server
- Gracefully reject new requests under resource pressure
- Maintain internal consistency even under partial failures

---

## Architectural invariants

The following rules should always hold:

- The API layer never calls the runtime directly
- The scheduler never executes model code
- Runtime backends never manage HTTP concerns
- Batching logic is centralized and not duplicated

Violating these invariants is considered a design bug.

---

## Evolution path

The architecture allows future extensions such as:
- Alternative scheduling policies
- Speculative decoding
- Hardware-accelerated runtimes

These extensions should not require changes to the API or scheduler layers.