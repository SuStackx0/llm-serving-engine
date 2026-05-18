"""
Structured prompt-lifecycle logger.

Every component (engine, scheduler) calls functions here so all log lines share
a consistent format:

  [ENGINE] [req-abc123] PREFILL_START  prompt_tokens=42  batch_position=0/3

The logger writes to the Python "llm.engine" hierarchy so the level and
handler can be configured once in run_server.py.
"""

import logging
import time
from typing import Any, Optional

_log = logging.getLogger("llm.engine")


def _fmt(component: str, request_id: str, event: str, **kv: Any) -> str:
    parts = " ".join(f"{k}={v}" for k, v in kv.items())
    short_id = request_id[:8]
    return f"[{component}] [req-{short_id}] {event:<22}  {parts}"


# ── Submission / queue ─────────────────────────────────────────────────


def log_submitted(request_id: str, prompt_tokens: int, max_tokens: int) -> None:
    _log.info(_fmt("API    ", request_id, "SUBMITTED",
                   prompt_tokens=prompt_tokens, max_tokens=max_tokens))


def log_queued(request_id: str, queue_depth: int) -> None:
    _log.info(_fmt("SCHED  ", request_id, "QUEUED",
                   queue_depth=queue_depth))


# ── Scheduler ─────────────────────────────────────────────────────────


def log_admitted(request_id: str, blocks_allocated: int, running_total: int) -> None:
    _log.info(_fmt("SCHED  ", request_id, "ADMITTED",
                   blocks_allocated=blocks_allocated, running=running_total))


def log_preempted(request_id: str, running_total: int) -> None:
    _log.warning(_fmt("SCHED  ", request_id, "PREEMPTED",
                      running=running_total))


def log_schedule_step(
    step: int,
    prefill_count: int,
    decode_count: int,
    preempted_count: int,
    running_total: int,
    waiting_total: int,
) -> None:
    _log.debug(
        f"[SCHED  ] [step-{step:06d}] SCHEDULE_STEP  "
        f"prefill={prefill_count} decode={decode_count} "
        f"preempted={preempted_count} running={running_total} waiting={waiting_total}"
    )


# ── Engine / prefill ──────────────────────────────────────────────────


def log_prefill_start(request_id: str, seq_len: int, batch_pos: int, batch_size: int) -> None:
    _log.info(_fmt("ENGINE ", request_id, "PREFILL_START",
                   prompt_tokens=seq_len, pos=f"{batch_pos}/{batch_size}"))


def log_prefill_done(request_id: str, first_token_id: int, elapsed_ms: float) -> None:
    _log.info(_fmt("ENGINE ", request_id, "PREFILL_DONE",
                   first_token=first_token_id, ttft_ms=f"{elapsed_ms:.1f}"))


# ── Engine / decode ───────────────────────────────────────────────────


def log_decode_step(
    request_id: str,
    step: int,
    token_id: int,
    ctx_len: int,
    batch_pos: int,
    batch_size: int,
) -> None:
    _log.debug(_fmt("ENGINE ", request_id, "DECODE_STEP",
                    step=step, token=token_id, ctx_len=ctx_len,
                    pos=f"{batch_pos}/{batch_size}"))


def log_finished(
    request_id: str,
    reason: str,
    output_tokens: int,
    total_ms: float,
    ttft_ms: Optional[float],
    tpot_ms: Optional[float],
) -> None:
    _log.info(_fmt("ENGINE ", request_id, "FINISHED",
                   reason=reason, output_tokens=output_tokens,
                   total_ms=f"{total_ms:.1f}",
                   ttft_ms=f"{ttft_ms:.1f}" if ttft_ms else "n/a",
                   tpot_ms=f"{tpot_ms:.1f}" if tpot_ms else "n/a"))
