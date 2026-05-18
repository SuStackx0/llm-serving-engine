"""
Tests for Adaptive Chunked Prefill.

Verifies that:
- compute_chunk_size adapts to decode pressure and memory
- scheduler sets correct chunk windows on prefill requests
- a long prompt progresses through multiple chunks before transitioning to DECODING
- chunked prefill does not block decode requests from running concurrently
- preemption resets all chunked prefill state
"""

import pytest
from unittest.mock import MagicMock

from src.core.types import Request, SamplingParams, SequenceStatus
from src.memory.block_manager import PhysicalBlockManager
from src.scheduler.scheduler import Scheduler


def make_scheduler(
    num_blocks: int = 64,
    block_size: int = 16,
    max_running: int = 8,
    max_chunk_size: int = 64,
    min_chunk_size: int = 16,
    enable_chunked: bool = True,
    enable_prefix: bool = False,
) -> Scheduler:
    bm = PhysicalBlockManager(num_blocks=num_blocks, block_size=block_size)
    return Scheduler(
        block_manager=bm,
        max_running_requests=max_running,
        enable_chunked_prefill=enable_chunked,
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        enable_prefix_caching=enable_prefix,
    )


def make_request(prompt_len: int = 32, max_tokens: int = 10) -> Request:
    return Request(
        prompt="x " * prompt_len,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


class TestComputeChunkSize:
    def test_no_decode_pressure_returns_max(self):
        s = make_scheduler(max_chunk_size=64, min_chunk_size=16)
        # No decode requests → pressure factor large → returns max
        assert s.compute_chunk_size() == 64

    def test_decode_pressure_reduces_chunk_size(self):
        s = make_scheduler(max_chunk_size=64, min_chunk_size=16)
        # Inject 8 fake DECODING requests
        for _ in range(8):
            req = make_request()
            req.status = SequenceStatus.DECODING
            s.running.append(req)
        chunk = s.compute_chunk_size()
        assert chunk < 64
        assert chunk >= 16

    def test_chunked_disabled_returns_max(self):
        s = make_scheduler(max_chunk_size=64, min_chunk_size=16, enable_chunked=False)
        assert s.compute_chunk_size() == 64

    def test_min_chunk_size_floor(self):
        s = make_scheduler(max_chunk_size=64, min_chunk_size=32)
        # Fill with lots of decode requests to maximise pressure
        for _ in range(100):
            req = make_request()
            req.status = SequenceStatus.DECODING
            s.running.append(req)
        assert s.compute_chunk_size() >= 32


class TestChunkWindows:
    def test_first_schedule_sets_chunk_window(self):
        s = make_scheduler(max_chunk_size=32, min_chunk_size=16)
        req = make_request(prompt_len=80, max_tokens=5)
        s.add_request(req)
        out = s.schedule()
        assert len(out.prefill_requests) == 1
        r = out.prefill_requests[0]
        assert r.chunk_start == 0
        assert 16 <= r.chunk_end <= 32  # adaptive, but bounded by [min, max]
        assert r.chunk_end <= req.prompt_len

    def test_chunk_end_capped_at_prompt_len_for_short_prompts(self):
        s = make_scheduler(max_chunk_size=64, min_chunk_size=16)
        req = make_request(prompt_len=20, max_tokens=5)
        s.add_request(req)
        out = s.schedule()
        r = out.prefill_requests[0]
        assert r.chunk_start == 0
        assert r.chunk_end == 20  # prompt shorter than any chunk size

    def test_subsequent_chunks_advance_window(self):
        s = make_scheduler(max_chunk_size=32, min_chunk_size=16)
        req = make_request(prompt_len=80, max_tokens=5)
        s.add_request(req)

        # First chunk
        out = s.schedule()
        r = out.prefill_requests[0]
        first_end = r.chunk_end
        s.on_chunk_complete(r, first_end)

        # Second chunk must start where the first ended
        out2 = s.schedule()
        r2 = out2.prefill_requests[0]
        assert r2.chunk_start == first_end
        assert r2.chunk_end > first_end  # made forward progress

    def test_final_chunk_reaches_prompt_len(self):
        s = make_scheduler(max_chunk_size=32, min_chunk_size=16)
        req = make_request(prompt_len=80, max_tokens=5)
        s.add_request(req)

        for _ in range(20):  # safety limit
            out = s.schedule()
            if not out.prefill_requests:
                break
            r = out.prefill_requests[0]
            chunk_end = r.chunk_end
            if chunk_end >= req.prompt_len:
                s.on_prefill_complete(r)
                break
            else:
                s.on_chunk_complete(r, chunk_end)

        assert req.status == SequenceStatus.DECODING


class TestChunkedPrefillDoesNotBlockDecode:
    def test_prefill_and_decode_coexist_in_same_step(self):
        s = make_scheduler(max_chunk_size=32, min_chunk_size=16, max_running=8)

        long_req = make_request(prompt_len=80, max_tokens=5)
        short_req = make_request(prompt_len=16, max_tokens=5)

        s.add_request(long_req)
        s.add_request(short_req)

        # First schedule: both admitted, both PREFILLING
        out = s.schedule()
        assert len(out.prefill_requests) == 2

        # Simulate short_req completing prefill in one shot, long_req still chunking
        s.on_prefill_complete(short_req)
        s.on_chunk_complete(long_req, long_req.chunk_end)

        # Next schedule: long_req still PREFILLING, short_req now DECODING
        out2 = s.schedule()
        assert any(r.request_id == long_req.request_id for r in out2.prefill_requests)
        assert any(r.request_id == short_req.request_id for r in out2.decode_requests)


class TestPreemptionResetsChunkState:
    def test_preempt_resets_tokens_prefilled(self):
        s = make_scheduler(max_chunk_size=32, min_chunk_size=16)
        req = make_request(prompt_len=80, max_tokens=5)
        s.add_request(req)

        out = s.schedule()
        r = out.prefill_requests[0]
        s.on_chunk_complete(r, r.chunk_end)  # advance to chunk 2
        assert r.tokens_prefilled == 32

        # Force preemption
        from src.core.types import SchedulerOutput
        dummy_out = SchedulerOutput()
        s._preempt(r, dummy_out)

        assert r.tokens_prefilled == 0
        assert r.chunk_start == 0
        assert r.chunk_end == 0
        assert r.status == SequenceStatus.WAITING

    def test_preempted_request_restarts_from_zero(self):
        s = make_scheduler(max_chunk_size=32, min_chunk_size=16)
        req = make_request(prompt_len=80, max_tokens=5)
        s.add_request(req)

        out = s.schedule()
        r = out.prefill_requests[0]
        first_end = r.chunk_end
        s.on_chunk_complete(r, first_end)

        from src.core.types import SchedulerOutput
        dummy_out = SchedulerOutput()
        s._preempt(r, dummy_out)

        # Re-admit
        out2 = s.schedule()
        r2 = out2.prefill_requests[0]
        assert r2.chunk_start == 0  # starts fresh


class TestOnChunkComplete:
    def test_on_chunk_complete_updates_tokens_prefilled(self):
        s = make_scheduler()
        req = make_request(prompt_len=64, max_tokens=5)
        s.add_request(req)
        s.schedule()
        s.on_chunk_complete(req, 32)
        assert req.tokens_prefilled == 32
        assert req.status == SequenceStatus.PREFILLING  # still prefilling

    def test_on_chunk_complete_logs_event(self):
        s = make_scheduler()
        req = make_request(prompt_len=64, max_tokens=5)
        s.add_request(req)
        s.schedule()
        s.on_chunk_complete(req, 32)
        events = [e.event for e in req.lifecycle]
        assert "CHUNK_DONE" in events
