# Execution Model

## Overview

The execution model defines how requests move through the system and how tokens are generated over time.

The engine follows a **step-based generation model** where tokens are produced incrementally and streamed back to clients.

---

## Request lifecycle

Each request transitions through the following states:

RECEIVED → QUEUED → SCHEDULED → ACTIVE → COMPLETED
↘
FAILED


### State definitions

- **RECEIVED**  
  Request accepted by the API layer.

- **QUEUED**  
  Request registered with the scheduler but not yet eligible for execution.

- **SCHEDULED**  
  Request selected by the scheduler for batching.

- **ACTIVE**  
  Request is participating in token generation.

- **COMPLETED**  
  Generation finished successfully.

- **FAILED**  
  Request terminated due to error or resource constraint.

---

## Token generation loop

Generation proceeds in discrete steps:

1. Scheduler selects eligible requests
2. Batching engine forms a batch
3. Runtime backend executes one forward step
4. Next tokens are produced
5. Tokens are streamed to clients
6. Requests reaching end conditions are removed

This loop continues until no active requests remain.

---

## Streaming semantics

Token streaming is a first-class concern:

- Tokens are emitted immediately after each generation step
- Clients may disconnect without affecting other requests
- Backpressure is applied if clients cannot consume tokens fast enough

Streaming failures do not terminate model execution unless required.

---

## End conditions

A request completes when:
- An end-of-sequence token is generated
- The maximum token limit is reached
- The client cancels the request
- A runtime error occurs

---

## Error handling

Errors are isolated at the request level:
- Runtime failures affect only the active batch
- Individual request failures do not crash the server
- Failed requests are removed cleanly from active execution

---

## Determinism and reproducibility

The execution model aims to:
- Produce deterministic outputs for identical inputs and seeds
- Ensure consistent behavior across runs, within backend constraints