"""
Tests for Prefix / Prompt Cache (PrefixTrieCache).

Verifies that:
- match returns empty on cold cache
- insert + match works for single and multi-block prefixes
- partial prefix matches return the longest aligned match
- LRU eviction only removes leaf nodes (not internal ones)
- pinned blocks survive eviction; unpinned ones don't
- hit/miss stats are tracked correctly
- scheduler integration: second request with same prefix gets fewer blocks allocated
"""

import pytest

from src.memory.block_manager import PhysicalBlockManager
from src.memory.prefix_cache import PrefixTrieCache
from src.core.types import Request, SamplingParams, SequenceStatus
from src.scheduler.scheduler import Scheduler


# ── Unit tests for PrefixTrieCache ────────────────────────────────────

BLOCK_SIZE = 4  # small for tests


def make_cache(block_size: int = BLOCK_SIZE) -> PrefixTrieCache:
    return PrefixTrieCache(block_size=block_size)


def tokens(n: int, start: int = 0) -> list:
    return list(range(start, start + n))


class TestMatch:
    def test_miss_on_empty_cache(self):
        cache = make_cache()
        matched, blocks = cache.match(tokens(8))
        assert matched == 0
        assert blocks == []

    def test_miss_increments_counter(self):
        cache = make_cache()
        cache.match(tokens(8))
        assert cache.stats()["miss_count"] == 1
        assert cache.stats()["hit_count"] == 0

    def test_partial_token_count_no_match(self):
        # Only 3 tokens (< block_size=4) — no complete block, no match possible
        cache = make_cache()
        cache.insert(tokens(4), [42])
        matched, blocks = cache.match(tokens(3))
        assert matched == 0

    def test_match_does_not_exceed_complete_blocks(self):
        # 6 tokens with block_size=4: only 1 complete block can be matched
        cache = make_cache()
        cache.insert(tokens(4), [10])
        matched, blocks = cache.match(tokens(6))
        assert matched == 4
        assert blocks == [10]


class TestInsertAndMatch:
    def test_single_block_insert_and_match(self):
        cache = make_cache()
        cache.insert(tokens(4), [5])
        matched, blocks = cache.match(tokens(8))
        assert matched == 4
        assert blocks == [5]

    def test_two_block_chain(self):
        cache = make_cache()
        cache.insert(tokens(8), [10, 11])
        matched, blocks = cache.match(tokens(12))
        assert matched == 8
        assert blocks == [10, 11]

    def test_partial_match_on_longer_trie(self):
        # Insert 3 blocks; query only 2 complete blocks → matches 2
        cache = make_cache()
        cache.insert(tokens(12), [1, 2, 3])
        # Query only covers first 2 blocks
        matched, blocks = cache.match(tokens(8))
        assert matched == 8
        assert blocks == [1, 2]

    def test_insert_same_prefix_twice_no_duplicate(self):
        cache = make_cache()
        cache.insert(tokens(4), [7])
        cache.insert(tokens(4), [99])  # same tokens, different block_id (ignored)
        assert cache.num_cached_blocks() == 1  # not duplicated

    def test_hit_increments_hit_counter(self):
        cache = make_cache()
        cache.insert(tokens(4), [5])
        cache.match(tokens(8))
        assert cache.stats()["hit_count"] == 1

    def test_second_match_also_counts_as_hit(self):
        cache = make_cache()
        cache.insert(tokens(4), [5])
        cache.match(tokens(8))
        cache.match(tokens(8))
        assert cache.stats()["hit_count"] == 2


class TestEviction:
    def test_evict_lru_removes_leaf(self):
        cache = make_cache()
        cache.insert(tokens(4), [10])
        freed = cache.evict_lru(1)
        assert freed == [10]
        assert cache.num_cached_blocks() == 0

    def test_evict_leaf_only_not_internal(self):
        # Chain: root → A (block 1) → B (block 2, leaf)
        cache = make_cache()
        t_a = tokens(4, start=0)
        t_ab = tokens(8, start=0)
        cache.insert(t_a, [1])
        cache.insert(t_ab, [1, 2])  # block 1 already exists; block 2 is new leaf

        freed = cache.evict_lru(1)
        # Only the leaf (block 2) should be freed; block 1 (internal) stays
        assert 2 in freed
        assert 1 not in freed
        assert cache.num_cached_blocks() == 1  # block 1 still cached

        # Shorter prefix should still match after leaf eviction
        matched, blocks = cache.match(t_a + tokens(4, start=100))
        assert matched == 4
        assert blocks == [1]

    def test_evict_empty_cache(self):
        cache = make_cache()
        freed = cache.evict_lru(5)
        assert freed == []

    def test_evict_more_than_available(self):
        cache = make_cache()
        cache.insert(tokens(4), [10])
        freed = cache.evict_lru(100)
        assert freed == [10]


class TestPinUnpin:
    def test_pinned_block_not_evicted(self):
        cache = make_cache()
        cache.insert(tokens(4), [42])
        cache.pin([42])
        freed = cache.evict_lru(10)
        assert 42 not in freed
        assert cache.num_cached_blocks() == 1

    def test_unpin_allows_eviction(self):
        cache = make_cache()
        cache.insert(tokens(4), [42])
        cache.pin([42])
        cache.unpin([42])
        freed = cache.evict_lru(1)
        assert 42 in freed

    def test_pin_unknown_block_no_error(self):
        cache = make_cache()
        cache.pin([999])  # should not raise


class TestStats:
    def test_initial_stats(self):
        cache = make_cache()
        s = cache.stats()
        assert s["hit_count"] == 0
        assert s["miss_count"] == 0
        assert s["hit_rate"] == 0.0
        assert s["cached_blocks"] == 0

    def test_hit_rate_calculation(self):
        cache = make_cache()
        cache.insert(tokens(4), [1])
        cache.match(tokens(8))   # hit
        cache.match(tokens(8))   # hit
        cache.match(tokens(4, start=100))  # miss (different tokens)
        s = cache.stats()
        assert s["hit_count"] == 2
        assert s["miss_count"] == 1
        assert abs(s["hit_rate"] - 2 / 3) < 1e-6


# ── Scheduler integration tests ───────────────────────────────────────

def make_scheduler_with_cache(
    num_blocks: int = 64,
    block_size: int = 4,
) -> Scheduler:
    bm = PhysicalBlockManager(num_blocks=num_blocks, block_size=block_size)
    return Scheduler(
        block_manager=bm,
        max_running_requests=8,
        enable_chunked_prefill=False,  # isolate prefix cache behavior
        enable_prefix_caching=True,
    )


def make_req(token_ids: list, max_tokens: int = 5) -> Request:
    return Request(
        prompt=" ".join(str(t) for t in token_ids),
        prompt_token_ids=token_ids,
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


class TestSchedulerPrefixCacheIntegration:
    def test_cold_cache_no_match(self):
        s = make_scheduler_with_cache()
        req = make_req(tokens(8))
        s.add_request(req)
        s.schedule()
        assert req.prefix_match_len == 0
        assert req.cached_block_ids == []

    def test_second_request_gets_prefix_match(self):
        s = make_scheduler_with_cache(block_size=4)
        # Request A: 8-token prompt (2 complete blocks)
        req_a = make_req(tokens(8), max_tokens=2)
        s.add_request(req_a)
        s.schedule()
        # Simulate prefill complete for req_a (inserts into cache)
        s.on_prefill_complete(req_a)

        # Request B: same 8-token prefix + 4 more tokens
        req_b = make_req(tokens(8) + tokens(4, start=100), max_tokens=2)
        s.add_request(req_b)
        s.schedule()

        assert req_b.prefix_match_len == 8
        assert len(req_b.cached_block_ids) == 2
        # tokens_prefilled starts at match_len (skip re-prefilling cached tokens)
        assert req_b.tokens_prefilled == 8

    def test_second_request_allocated_fewer_blocks(self):
        s = make_scheduler_with_cache(block_size=4)
        req_a = make_req(tokens(8), max_tokens=2)
        s.add_request(req_a)
        out = s.schedule()
        s.on_prefill_complete(req_a)

        # Count blocks before second request
        blocks_before = s.block_manager.num_used_blocks()

        req_b = make_req(tokens(8) + tokens(4, start=200), max_tokens=2)
        s.add_request(req_b)
        s.schedule()

        # req_b should not allocate new blocks for the shared prefix
        assert req_b.prefix_match_len == 8
        # The block table should include the shared prefix blocks
        assert req_b.cached_block_ids == req_b.block_table[:2]

    def test_preemption_unpins_cached_blocks(self):
        s = make_scheduler_with_cache(block_size=4)
        req_a = make_req(tokens(8), max_tokens=2)
        s.add_request(req_a)
        s.schedule()
        s.on_prefill_complete(req_a)

        req_b = make_req(tokens(8) + tokens(4, start=200), max_tokens=2)
        s.add_request(req_b)
        s.schedule()

        assert req_b.prefix_match_len == 8
        cached = list(req_b.cached_block_ids)

        # Check that blocks are pinned (ref_count > 0)
        for bid in cached:
            node = s.prefix_cache._block_to_node.get(bid)
            assert node is not None and node.ref_count > 0

        # Preempt req_b — should unpin
        from src.core.types import SchedulerOutput
        dummy = SchedulerOutput()
        s._preempt(req_b, dummy)

        for bid in cached:
            node = s.prefix_cache._block_to_node.get(bid)
            if node is not None:
                assert node.ref_count == 0

    def test_block_manager_does_not_free_cached_blocks_on_request_free(self):
        s = make_scheduler_with_cache(block_size=4)
        req_a = make_req(tokens(8), max_tokens=2)
        s.add_request(req_a)
        s.schedule()
        s.on_prefill_complete(req_a)  # marks some blocks as cached

        # Mark req_a as done and free its blocks
        req_a.status = SequenceStatus.FINISHED_EOS
        free_count = s.block_manager.free(req_a.request_id)

        # Cached blocks should NOT be in the free pool
        cached_ids = set(s.block_manager._cached_block_ids)
        assert len(cached_ids) > 0  # some blocks were marked as cached

        free_pool = set(s.block_manager._free)
        # No overlap between free pool and cached blocks
        assert cached_ids.isdisjoint(free_pool)
