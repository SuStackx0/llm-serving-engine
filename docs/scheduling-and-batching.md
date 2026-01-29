# Scheduling and Batching

## Motivation

Scheduling and batching are the primary mechanisms for improving throughput in a serving system.

This engine separates:
- **Scheduling**: deciding *which requests* may run
- **Batching**: deciding *how requests are grouped*

---

## Scheduler responsibilities

The scheduler:
- Maintains a queue of pending requests
- Enforces fairness and resource constraints
- Selects requests eligible for execution

The scheduler operates at the **request level**, not the token level.

---

## Initial scheduling policy

The default scheduler implements:
- First-come-first-served (FCFS)
- Optional maximum concurrent request limit
- Optional token budget per request

This policy prioritizes simplicity and predictability.

---

## Batching model

The batching engine:
- Forms batches from scheduled requests
- Executes token generation in steps
- Allows requests to join or leave batches dynamically

This is commonly referred to as **continuous batching**.

---

## Batch constraints

Batches may be constrained by:
- Maximum batch size
- Maximum total tokens per step
- Runtime backend limitations

Batch formation must respect these constraints at all times.

---

## Request eviction

Requests may be evicted from execution if:
- They exceed resource limits
- Higher-priority requests must be admitted
- The system is under memory pressure

Eviction is treated as a controlled failure, not a crash.

---

## Fairness guarantees

The system aims to:
- Prevent starvation
- Avoid long-running requests blocking others
- Maintain reasonable latency under load

Exact fairness guarantees depend on the scheduler implementation.

---

## Future extensions

Planned extensions include:
- Token-aware scheduling
- Priority queues
- Deadline-based scheduling
