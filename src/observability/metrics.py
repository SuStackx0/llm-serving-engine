"""
Metrics collector: TTFT, TPOT, throughput, request counters.
Thread-safe via a simple lock; readable from the /v1/stats endpoint.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

from src.core.types import Request


@dataclass
class RequestMetrics:
    request_id: str
    ttft_ms: Optional[float]
    tpot_ms: Optional[float]
    num_output_tokens: int
    total_latency_ms: float
    finish_reason: str


class MetricsCollector:
    def __init__(self, throughput_window_secs: float = 10.0):
        self._lock = threading.Lock()
        self._window = throughput_window_secs

        # Recent token timestamps for throughput calculation
        self._token_times: Deque[float] = deque()

        # Counters
        self._total_requests: int = 0
        self._total_tokens_out: int = 0
        self._total_tokens_in: int = 0

        # Per-request history (last 100)
        self._history: Deque[RequestMetrics] = deque(maxlen=100)

        # Running averages
        self._sum_ttft_ms: float = 0.0
        self._sum_tpot_ms: float = 0.0
        self._count_ttft: int = 0
        self._count_tpot: int = 0

        # Prefix cache counters
        self._prefix_cache_hits: int = 0
        self._prefix_cache_misses: int = 0

        self._start_time = time.monotonic()

    def record_token(self) -> None:
        """Call each time a new output token is generated."""
        with self._lock:
            now = time.monotonic()
            self._token_times.append(now)
            self._total_tokens_out += 1
            # Prune old entries outside window
            cutoff = now - self._window
            while self._token_times and self._token_times[0] < cutoff:
                self._token_times.popleft()

    def record_request_complete(self, request: Request) -> None:
        with self._lock:
            self._total_requests += 1
            self._total_tokens_in += request.prompt_len
            self._total_tokens_out  # already tracked token-by-token

            ttft = request.ttft_ms()
            tpot = request.tpot_ms()
            total_ms = 0.0
            if request.prefill_start_time and request.last_token_time:
                total_ms = (request.last_token_time - request.prefill_start_time) * 1000

            if ttft is not None:
                self._sum_ttft_ms += ttft
                self._count_ttft += 1
            if tpot is not None:
                self._sum_tpot_ms += tpot
                self._count_tpot += 1

            self._history.append(RequestMetrics(
                request_id=request.request_id,
                ttft_ms=ttft,
                tpot_ms=tpot,
                num_output_tokens=request.num_generated_tokens,
                total_latency_ms=total_ms,
                finish_reason=request.status.value,
            ))

    def record_prefix_cache_hit(self, matched_tokens: int) -> None:
        with self._lock:
            self._prefix_cache_hits += 1

    def record_prefix_cache_miss(self) -> None:
        with self._lock:
            self._prefix_cache_misses += 1

    def throughput_tok_s(self) -> float:
        """Output tokens per second over the recent window."""
        with self._lock:
            n = len(self._token_times)
            if n < 2:
                return 0.0
            elapsed = self._token_times[-1] - self._token_times[0]
            return n / elapsed if elapsed > 0 else 0.0

    def avg_ttft_ms(self) -> Optional[float]:
        with self._lock:
            if self._count_ttft == 0:
                return None
            return self._sum_ttft_ms / self._count_ttft

    def avg_tpot_ms(self) -> Optional[float]:
        with self._lock:
            if self._count_tpot == 0:
                return None
            return self._sum_tpot_ms / self._count_tpot

    def snapshot(self) -> Dict:
        with self._lock:
            return {
                "total_requests_served": self._total_requests,
                "total_tokens_in": self._total_tokens_in,
                "total_tokens_out": self._total_tokens_out,
                "throughput_tok_s": self.throughput_tok_s(),
                "avg_ttft_ms": self.avg_ttft_ms(),
                "avg_tpot_ms": self.avg_tpot_ms(),
                "uptime_s": time.monotonic() - self._start_time,
                "prefix_cache_hits": self._prefix_cache_hits,
                "prefix_cache_misses": self._prefix_cache_misses,
            }
