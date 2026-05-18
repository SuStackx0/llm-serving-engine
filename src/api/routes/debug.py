"""
Debug / observability endpoints.

POST /debug/lifecycle
    Submit a single prompt, wait for it to finish, return the full
    per-event lifecycle trace (every state transition with timestamps
    and metadata).

POST /debug/batch
    Submit N prompts simultaneously, wait for all to finish, return a
    per-request lifecycle summary AND a step-by-step view of how the
    continuous batcher handled each engine step.
"""

import asyncio
import queue
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request as FastAPIRequest
from pydantic import BaseModel, Field

from src.core.types import Request, SamplingParams

router = APIRouter(prefix="/debug", tags=["debug"])


# ── Request / response schemas ─────────────────────────────────────────

class LifecycleRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)


class BatchRequest(BaseModel):
    prompts: List[str] = Field(min_length=1, max_length=16)
    max_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)


class LifecycleEvent(BaseModel):
    event: str
    wall_ts: float
    elapsed_from_submit_ms: float
    data: Dict[str, Any] = {}


class LifecycleResponse(BaseModel):
    request_id: str
    prompt: str
    output: str
    prompt_tokens: int
    output_tokens: int
    total_ms: float
    ttft_ms: Optional[float]
    tpot_ms: Optional[float]
    finish_reason: str
    lifecycle: List[LifecycleEvent]


class BatchRequestResult(BaseModel):
    request_id: str
    prompt: str
    output: str
    prompt_tokens: int
    output_tokens: int
    total_ms: float
    ttft_ms: Optional[float]
    finish_reason: str
    lifecycle_summary: List[str]   # one-liner per event


class BatchStep(BaseModel):
    """What the engine did in a single schedule step across all batch requests."""
    step: int
    prefilling: List[str]   # request_ids being prefilled
    decoding: List[str]     # request_ids being decoded
    preempted: List[str]    # request_ids preempted this step


class BatchResponse(BaseModel):
    num_prompts: int
    total_wall_ms: float
    requests: List[BatchRequestResult]
    engine_steps: List[BatchStep]


# ── Helpers ────────────────────────────────────────────────────────────

def _build_request(prompt: str, engine, max_tokens: int, temperature: float) -> Request:
    token_ids = engine.tokenizer.encode(prompt, add_special_tokens=True)
    return Request(
        prompt=prompt,
        prompt_token_ids=token_ids,
        sampling_params=SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        ),
    )


def _drain_queue(out_q: queue.Queue) -> List[int]:
    """Collect all token ids from the output queue until the sentinel (None)."""
    tokens: List[int] = []
    while True:
        tok = out_q.get()
        if tok is None:
            break
        tokens.append(tok)
    return tokens


def _to_lifecycle_events(req: Request) -> List[LifecycleEvent]:
    if not req.lifecycle:
        return []
    t0 = req.lifecycle[0].wall_ts
    events = []
    for ev in req.lifecycle:
        events.append(LifecycleEvent(
            event=ev.event,
            wall_ts=ev.wall_ts,
            elapsed_from_submit_ms=round((ev.wall_ts - t0) * 1000, 2),
            data=ev.data,
        ))
    return events


def _summarise_lifecycle(req: Request) -> List[str]:
    lines = []
    t0 = req.lifecycle[0].wall_ts if req.lifecycle else 0.0
    for ev in req.lifecycle:
        elapsed = round((ev.wall_ts - t0) * 1000, 1)
        kv = " ".join(f"{k}={v}" for k, v in ev.data.items())
        lines.append(f"+{elapsed:8.1f}ms  {ev.event:<22}  {kv}")
    return lines


# ── Batch step tracker ─────────────────────────────────────────────────

class _StepTracker:
    """Attached to engine via monkey-patch to intercept schedule() output."""

    def __init__(self):
        self.steps: List[BatchStep] = []
        self._counter = 0
        self._target_ids: set = set()

    def record(self, sched_out, step_n: int) -> None:
        prefilling = [r.request_id for r in sched_out.prefill_requests
                      if r.request_id in self._target_ids]
        decoding   = [r.request_id for r in sched_out.decode_requests
                      if r.request_id in self._target_ids]
        preempted  = [r.request_id for r in sched_out.preempted_requests
                      if r.request_id in self._target_ids]
        if prefilling or decoding or preempted:
            self.steps.append(BatchStep(
                step=step_n,
                prefilling=prefilling,
                decoding=decoding,
                preempted=preempted,
            ))


# ── Endpoints ──────────────────────────────────────────────────────────

@router.post("/lifecycle", response_model=LifecycleResponse,
             summary="Full prompt lifecycle trace for a single request")
async def lifecycle(body: LifecycleRequest, http_req: FastAPIRequest):
    """
    Submit one prompt and get back every lifecycle event (SUBMITTED →
    QUEUED → ADMITTED → PREFILL_START → PREFILL_DONE → DECODE_STEP ×N
    → FINISHED) with wall-clock timestamps.
    """
    engine = http_req.app.state.engine
    req = _build_request(body.prompt, engine, body.max_tokens, body.temperature)

    t_wall_start = time.time()
    out_q = engine.submit(req)

    # Run blocking I/O in a thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    output_ids = await loop.run_in_executor(None, _drain_queue, out_q)

    total_ms = (time.time() - t_wall_start) * 1000
    output_text = engine.tokenizer.decode(output_ids, skip_special_tokens=True)

    return LifecycleResponse(
        request_id=req.request_id,
        prompt=body.prompt,
        output=output_text,
        prompt_tokens=req.prompt_len,
        output_tokens=len(output_ids),
        total_ms=round(total_ms, 2),
        ttft_ms=round(req.ttft_ms(), 2) if req.ttft_ms() else None,
        tpot_ms=round(req.tpot_ms(), 2) if req.tpot_ms() else None,
        finish_reason=req.status.value,
        lifecycle=_to_lifecycle_events(req),
    )


@router.post("/batch", response_model=BatchResponse,
             summary="Continuous-batching demo: submit N prompts simultaneously")
async def batch(body: BatchRequest, http_req: FastAPIRequest):
    """
    Submit all prompts at once.  The engine's continuous batcher interleaves
    prefill and decode steps across them.  The response shows per-request
    lifecycle summaries AND a per-engine-step view so you can see exactly
    which requests were prefilling or decoding together at each step.
    """
    engine = http_req.app.state.engine
    loop = asyncio.get_event_loop()

    reqs: List[Request] = []
    out_qs: List[queue.Queue] = []

    for prompt in body.prompts:
        req = _build_request(prompt, engine, body.max_tokens, body.temperature)
        out_q = engine.submit(req)
        reqs.append(req)
        out_qs.append(out_q)

    t_wall_start = time.time()

    # Drain all queues concurrently in the thread pool
    output_ids_list: List[List[int]] = await asyncio.gather(
        *[loop.run_in_executor(None, _drain_queue, q) for q in out_qs]
    )

    total_ms = (time.time() - t_wall_start) * 1000

    # Build per-request results
    results: List[BatchRequestResult] = []
    for req, output_ids in zip(reqs, output_ids_list):
        output_text = engine.tokenizer.decode(output_ids, skip_special_tokens=True)
        req_total_ms = 0.0
        if req.prefill_start_time and req.last_token_time:
            req_total_ms = (req.last_token_time - req.prefill_start_time) * 1000
        results.append(BatchRequestResult(
            request_id=req.request_id,
            prompt=req.prompt,
            output=output_text,
            prompt_tokens=req.prompt_len,
            output_tokens=len(output_ids),
            total_ms=round(req_total_ms, 2),
            ttft_ms=round(req.ttft_ms(), 2) if req.ttft_ms() else None,
            finish_reason=req.status.value,
            lifecycle_summary=_summarise_lifecycle(req),
        ))

    # Reconstruct engine steps from the union of all lifecycle events
    # Each DECODE_STEP / PREFILL_START event carries step and batch_size info.
    engine_steps = _reconstruct_engine_steps(reqs)

    return BatchResponse(
        num_prompts=len(reqs),
        total_wall_ms=round(total_ms, 2),
        requests=results,
        engine_steps=engine_steps,
    )


def _reconstruct_engine_steps(reqs: List[Request]) -> List[BatchStep]:
    """
    Re-derive engine steps from the lifecycle events stored on each request.
    We align events by their monotonic timestamp proximity.
    """
    # Collect (ts, event_name, req_id, data) for all scheduling-relevant events
    events_flat = []
    for req in reqs:
        for ev in req.lifecycle:
            if ev.event in ("PREFILL_START", "DECODE_STEP", "PREEMPTED"):
                events_flat.append((ev.ts, ev.event, req.request_id, ev.data))

    if not events_flat:
        return []

    events_flat.sort(key=lambda x: x[0])

    # Group by the "step" field in data (all events with same step number go together)
    # For PREFILL_START the step is implicit (step=0 for that request)
    steps: Dict[int, BatchStep] = {}
    for ts, event, rid, data in events_flat:
        if event == "PREFILL_START":
            step_n = 0  # prefill is always "step 0" for this request
            key = f"prefill_{rid}"   # unique per request
            step_key = id(key)       # use unique key so prefills don't collide
            # Use ts bucket instead
            bucket = round(ts, 2)
        elif event == "DECODE_STEP":
            step_n = data.get("step", 0)
            bucket = step_n
        elif event == "PREEMPTED":
            bucket = round(ts, 2)
            step_n = -1

        if bucket not in steps:
            steps[bucket] = BatchStep(step=len(steps) + 1,
                                      prefilling=[], decoding=[], preempted=[])
        s = steps[bucket]
        if event == "PREFILL_START":
            s.prefilling.append(rid)
        elif event == "DECODE_STEP":
            if rid not in s.decoding:
                s.decoding.append(rid)
        elif event == "PREEMPTED":
            s.preempted.append(rid)

    # Return steps sorted by their bucket key
    sorted_steps = [steps[k] for k in sorted(steps.keys())]
    # Re-number
    for i, s in enumerate(sorted_steps):
        s.step = i + 1
    return sorted_steps
