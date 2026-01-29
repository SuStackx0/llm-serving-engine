# Non-Goals

This document explicitly lists what llm-serving-engine does NOT attempt to do.

Defining non-goals is critical to keeping the system focused and maintainable.

---

## Not a training framework

The engine does not support:
- Model training
- Fine-tuning
- Parameter updates

All models are treated as immutable artifacts.

---

## Not a distributed system

The engine does not aim to:
- Run across multiple machines
- Coordinate multi-node execution
- Provide fault tolerance across hosts

Single-node execution is the target.

---

## Not a GPU kernel project

The engine does not:
- Implement custom CUDA or Metal kernels
- Compete with low-level ML frameworks
- Optimize matrix multiplication primitives

It builds on existing runtimes for execution.

---

## Not a benchmark competition

The project does not:
- Chase benchmark leaderboards
- Optimize for synthetic throughput numbers
- Sacrifice correctness for speed

Performance improvements must preserve clarity and correctness.

---

## Not a cloud service

The engine does not:
- Handle authentication or billing
- Provide multi-tenant isolation guarantees
- Offer SLA-level reliability

It is designed for local and controlled environments.
