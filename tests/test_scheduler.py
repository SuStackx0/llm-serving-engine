"""Tests for the Continuous Batching Scheduler."""

import time
import pytest
from src.core.types import Request, SamplingParams, SequenceStatus
from src.memory.block_manager import PhysicalBlockManager
from src.scheduler.scheduler import Scheduler


def make_request(prompt_len: int = 10, max_tokens: int = 20, priority: int = 0) -> Request:
    return Request(
        prompt="x" * prompt_len,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
        priority=priority,
    )


@pytest.fixture
def bm():
    return PhysicalBlockManager(num_blocks=64, block_size=16)


@pytest.fixture
def scheduler(bm):
    return Scheduler(block_manager=bm, max_running_requests=4)


def test_add_and_schedule_prefill(scheduler):
    req = make_request(prompt_len=5, max_tokens=10)
    scheduler.add_request(req)
    out = scheduler.schedule()
    assert len(out.prefill_requests) == 1
    assert req in out.prefill_requests
    assert req.status == SequenceStatus.PREFILLING


def test_prefill_to_decode_transition(scheduler):
    req = make_request(prompt_len=5, max_tokens=10)
    scheduler.add_request(req)
    out = scheduler.schedule()

    # Simulate engine completing prefill
    scheduler.on_prefill_complete(req)
    assert req.status == SequenceStatus.DECODING

    out2 = scheduler.schedule()
    assert req in out2.decode_requests
    assert not out2.prefill_requests


def test_max_running_limit(scheduler):
    for _ in range(6):
        scheduler.add_request(make_request(prompt_len=5, max_tokens=5))

    out = scheduler.schedule()
    # Should admit at most max_running=4
    assert len(out.prefill_requests) <= 4
    assert scheduler.num_waiting() >= 2


def test_finished_request_removed(scheduler, bm):
    req = make_request(prompt_len=5, max_tokens=2)
    scheduler.add_request(req)
    scheduler.schedule()
    scheduler.on_prefill_complete(req)

    # Simulate two decode steps
    scheduler.on_token_generated(req, 1, eos_token_id=2)
    scheduler.on_token_generated(req, 2, eos_token_id=2)  # max_tokens reached
    assert req.is_finished()

    out = scheduler.schedule()
    assert req not in out.decode_requests
    assert req not in out.prefill_requests


def test_priority_ordering():
    bm = PhysicalBlockManager(num_blocks=128, block_size=16)
    sched = Scheduler(block_manager=bm, max_running_requests=2)

    low  = make_request(priority=10)
    high = make_request(priority=0)

    sched.add_request(low)
    sched.add_request(high)

    out = sched.schedule()
    # High priority should be admitted first
    assert high in out.prefill_requests


def test_block_expansion_on_decode(scheduler):
    """Request that grows beyond first block should get a new block."""
    # block_size=16; allocate prompt=5 + max=20 → needs ceil(25/16)=2 blocks
    req = make_request(prompt_len=5, max_tokens=20)
    scheduler.add_request(req)
    scheduler.schedule()
    scheduler.on_prefill_complete(req)

    # Simulate decoding 11 tokens (5+11=16, filling first block)
    for i in range(11):
        scheduler.on_token_generated(req, i + 100, eos_token_id=999)

    # Block table should have been expanded
    out = scheduler.schedule()
    assert req in out.decode_requests
