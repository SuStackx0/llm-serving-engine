"""Tests for PhysicalBlockManager."""

import pytest
from src.memory.block_manager import PhysicalBlockManager


@pytest.fixture
def bm():
    return PhysicalBlockManager(num_blocks=32, block_size=16)


def test_initial_state(bm):
    assert bm.num_free_blocks() == 32
    assert bm.num_used_blocks() == 0


def test_allocate_returns_correct_count(bm):
    blocks = bm.allocate("req1", 4)
    assert len(blocks) == 4
    assert bm.num_free_blocks() == 28
    assert bm.num_used_blocks() == 4


def test_allocate_no_overlap(bm):
    b1 = bm.allocate("req1", 5)
    b2 = bm.allocate("req2", 5)
    assert not set(b1) & set(b2), "Blocks must not overlap between requests"


def test_free_returns_blocks(bm):
    bm.allocate("req1", 10)
    freed = bm.free("req1")
    assert freed == 10
    assert bm.num_free_blocks() == 32


def test_reallocate_after_free(bm):
    bm.allocate("req1", 16)
    bm.free("req1")
    blocks = bm.allocate("req2", 16)
    assert len(blocks) == 16  # should succeed


def test_can_allocate_check(bm):
    assert bm.can_allocate(32)
    assert not bm.can_allocate(33)


def test_allocate_raises_when_full(bm):
    bm.allocate("req1", 32)
    with pytest.raises(RuntimeError):
        bm.allocate("req2", 1)


def test_num_required_blocks(bm):
    assert bm.num_required_blocks(16) == 1
    assert bm.num_required_blocks(17) == 2
    assert bm.num_required_blocks(32) == 2
    assert bm.num_required_blocks(1) == 1


def test_utilization(bm):
    bm.allocate("req1", 16)
    assert abs(bm.utilization() - 0.5) < 1e-6


def test_allocate_one(bm):
    bm.allocate("req1", 2)
    new = bm.allocate_one("req1")
    assert isinstance(new, int)
    assert bm.num_used_blocks() == 3
