from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass
class LifecycleEvent:
    """One timestamped event in a request's lifecycle."""
    event: str
    ts: float = field(default_factory=time.monotonic)   # monotonic seconds
    wall_ts: float = field(default_factory=time.time)   # unix time for display
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"event": self.event, "wall_ts": self.wall_ts, **self.data}


class SequenceStatus(Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED_EOS = "finished_eos"
    FINISHED_LENGTH = "finished_length"
    FINISHED_STOP = "finished_stop"
    PREEMPTED = "preempted"

    def is_finished(self) -> bool:
        return self in (
            SequenceStatus.FINISHED_EOS,
            SequenceStatus.FINISHED_LENGTH,
            SequenceStatus.FINISHED_STOP,
        )

    def is_running(self) -> bool:
        return self in (SequenceStatus.PREFILLING, SequenceStatus.DECODING)


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1           # -1 = disabled
    max_tokens: int = 256
    stop: List[str] = field(default_factory=list)
    stream: bool = False

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")


@dataclass
class Request:
    prompt: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    arrival_time: float = field(default_factory=time.monotonic)
    priority: int = 0               # lower = higher priority

    # Runtime state managed by engine/scheduler
    status: SequenceStatus = SequenceStatus.WAITING
    output_token_ids: List[int] = field(default_factory=list)

    # Physical block IDs assigned by BlockManager
    block_table: List[int] = field(default_factory=list)

    # How many tokens have been stored in KV cache (prompt + generated so far)
    num_cached_tokens: int = 0

    # Chunked prefill state (set by scheduler each step)
    tokens_prefilled: int = 0   # prompt tokens stored in KV cache so far
    chunk_start: int = 0        # window start for current prefill step
    chunk_end: int = 0          # window end (exclusive) for current prefill step

    # Prefix cache state
    prefix_match_len: int = 0                                           # tokens matched from cache (multiple of block_size)
    cached_block_ids: List[int] = field(default_factory=list)           # borrowed blocks (not freed on request end)

    # Timing
    prefill_start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None

    # Token streaming: the engine pushes new token ids here
    _token_queue: object = field(default=None, repr=False)

    # Lifecycle event log — each major state transition is appended here
    lifecycle: List[LifecycleEvent] = field(default_factory=list, repr=False)

    def log_event(self, event: str, **data: Any) -> None:
        self.lifecycle.append(LifecycleEvent(event=event, data=data))

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.num_generated_tokens

    @property
    def last_token_id(self) -> int:
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]

    def is_finished(self) -> bool:
        return self.status.is_finished()

    def ttft_ms(self) -> Optional[float]:
        if self.first_token_time and self.prefill_start_time:
            return (self.first_token_time - self.prefill_start_time) * 1000
        return None

    def tpot_ms(self) -> Optional[float]:
        n = self.num_generated_tokens
        if n > 1 and self.first_token_time and self.last_token_time:
            elapsed = self.last_token_time - self.first_token_time
            return elapsed * 1000 / (n - 1)
        return None


@dataclass
class SchedulerOutput:
    prefill_requests: List[Request] = field(default_factory=list)
    decode_requests: List[Request] = field(default_factory=list)
    preempted_requests: List[Request] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.prefill_requests and not self.decode_requests


@dataclass
class AttentionMetadata:
    """Metadata passed to each transformer layer describing the current batch."""
    # Each prefill request's chunk length (tokens to process this step)
    prefill_seq_lens: List[int]
    # Block tables for each prefill request (logical → physical blocks)
    prefill_block_tables: List[List[int]]
    # Total context length (prompt+generated so far) for each decode request
    decode_context_lens: List[int]
    # Block tables for each decode request
    decode_block_tables: List[List[int]]
    # For chunked prefill: absolute slot where each chunk starts writing KV
    # Length matches prefill_seq_lens. None means start_slot=0 (backward compat).
    prefill_chunk_starts: Optional[List[int]] = None

    @property
    def num_prefill_tokens(self) -> int:
        return sum(self.prefill_seq_lens)

    @property
    def num_decode_tokens(self) -> int:
        return len(self.decode_context_lens)

    @property
    def total_tokens(self) -> int:
        return self.num_prefill_tokens + self.num_decode_tokens
