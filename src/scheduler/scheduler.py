"""
Continuous Batching Scheduler — the core vLLM scheduling algorithm.

Each call to schedule() returns a SchedulerOutput for ONE engine step:
  - prefill_requests: newly admitted requests to process their full prompt
  - decode_requests:  requests already in the decode phase (generate 1 token)

Key behaviours implemented:
  1. Continuous batching: batch composition changes every step.
  2. Prefill / decode separation: different batches, different compute profiles.
  3. Block allocation on prefill: reserve enough physical blocks up-front.
  4. Dynamic expansion on decode: allocate one more block when a request
     reaches the end of its last block.
  5. Preemption: when no blocks are available for a new request, evict the
     lowest-priority decode request (re-queue it as waiting).
"""

import logging
import time
from collections import deque
from typing import List, Optional

from src.core.types import Request, SchedulerOutput, SequenceStatus
from src.memory.block_manager import PhysicalBlockManager
from src.memory.prefix_cache import PrefixTrieCache
from src.observability import prompt_logger as plog
from src.scheduler.request_queue import PriorityRequestQueue

log = logging.getLogger("llm.engine")


class Scheduler:
    def __init__(
        self,
        block_manager: PhysicalBlockManager,
        max_running_requests: int = 8,
        max_waiting_requests: int = 256,
        enable_chunked_prefill: bool = True,
        max_chunk_size: int = 256,
        min_chunk_size: int = 64,
        enable_prefix_caching: bool = True,
    ):
        self.block_manager = block_manager
        self.max_running = max_running_requests
        self.max_waiting = max_waiting_requests

        # Chunked prefill config
        self.enable_chunked_prefill = enable_chunked_prefill
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

        # Prefix / prompt cache
        self.prefix_cache: Optional[PrefixTrieCache] = (
            PrefixTrieCache(block_size=block_manager.block_size)
            if enable_prefix_caching else None
        )

        self.waiting = PriorityRequestQueue()
        self.running: List[Request] = []          # prefilling or decoding
        self.preempted: deque[Request] = deque()  # re-queued for later

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        request.status = SequenceStatus.WAITING
        self.waiting.push(request)
        request.log_event("QUEUED", queue_depth=len(self.waiting))
        plog.log_queued(request.request_id, len(self.waiting))

    def has_work(self) -> bool:
        return bool(self.running) or bool(self.waiting)

    def schedule(self) -> SchedulerOutput:
        """Build the next execution batch.

        Steps:
          1. Remove finished requests, free their blocks.
          2. For each running decode request that needs a new block, allocate it.
             If no block is available, preempt a low-priority request.
          3. Try to admit waiting requests into running (allocate their blocks).
          4. Return split into prefill / decode lists.
        """
        output = SchedulerOutput()

        # ── Step 1: clean up finished requests ──────────────────────
        still_running: List[Request] = []
        for req in self.running:
            if req.is_finished():
                self.block_manager.free(req.request_id)
            else:
                still_running.append(req)
        self.running = still_running

        # ── Step 2: ensure decode requests have a block for the next token ──
        for req in self.running:
            if req.status != SequenceStatus.DECODING:
                continue
            # Does the next slot fit in the current last block?
            next_slot = req.num_cached_tokens  # 0-indexed slot we're about to write
            if next_slot % self.block_manager.block_size == 0:
                # Need a new block
                if self.block_manager.can_allocate(1):
                    new_block = self.block_manager.allocate_one(req.request_id)
                    req.block_table.append(new_block)
                else:
                    # No free blocks — preempt this (or a lower priority) request
                    victim = self._select_preemption_victim()
                    if victim is not None:
                        self._preempt(victim, output)
                        # Retry allocation for req
                        if self.block_manager.can_allocate(1):
                            new_block = self.block_manager.allocate_one(req.request_id)
                            req.block_table.append(new_block)

        # ── Step 3: admit waiting requests ──────────────────────────
        while (
            self.waiting
            and len(self.running) < self.max_running
        ):
            req = self.waiting.peek()

            # Prefix cache lookup — find how many leading tokens are already cached
            match_len: int = 0
            cached_blocks: List[int] = []
            if self.prefix_cache is not None:
                match_len, cached_blocks = self.prefix_cache.match(req.prompt_token_ids)
            req.prefix_match_len = match_len
            req.cached_block_ids = cached_blocks

            # Only allocate blocks for the uncached suffix + generated tokens
            suffix_tokens = req.prompt_len - match_len + req.sampling_params.max_tokens
            blocks_needed = self.block_manager.num_required_blocks(suffix_tokens)

            if not self.block_manager.can_allocate(blocks_needed):
                # Try prefix cache eviction before falling back to preemption
                if self.prefix_cache is not None:
                    shortage = blocks_needed - self.block_manager.num_free_blocks()
                    freed_ids = self.prefix_cache.evict_lru(shortage)
                    if freed_ids:
                        self.block_manager.unmark_cached(freed_ids)

            if self.block_manager.can_allocate(blocks_needed):
                self.waiting.pop()
                suffix_allocated = self.block_manager.allocate(req.request_id, blocks_needed)

                # Pin borrowed prefix blocks so they survive eviction while in use
                if self.prefix_cache is not None and cached_blocks:
                    self.prefix_cache.pin(cached_blocks)

                # Full block table: cached prefix blocks prepended to newly allocated suffix
                req.block_table = cached_blocks + suffix_allocated
                req.status = SequenceStatus.PREFILLING
                req.prefill_start_time = time.monotonic()
                # Chunked prefill starts from the end of the cached prefix
                req.tokens_prefilled = match_len
                self.running.append(req)
                req.log_event("ADMITTED", blocks_allocated=blocks_needed,
                              prefix_matched=match_len, running=len(self.running))
                plog.log_admitted(req.request_id, blocks_needed, len(self.running))
            else:
                victim = self._select_preemption_victim()
                if victim is not None:
                    self._preempt(victim, output)
                else:
                    break

        # ── Step 4: split into prefill / decode (with chunk windows) ─
        chunk_size = self.compute_chunk_size()
        for req in self.running:
            if req.status == SequenceStatus.PREFILLING:
                req.chunk_start = req.tokens_prefilled
                req.chunk_end = min(req.prompt_len, req.tokens_prefilled + chunk_size)
                output.prefill_requests.append(req)
            elif req.status == SequenceStatus.DECODING:
                output.decode_requests.append(req)

        return output

    def on_prefill_complete(self, request: Request) -> None:
        """Called by engine after a request's final prefill chunk finishes."""
        request.status = SequenceStatus.DECODING
        request.num_cached_tokens = request.prompt_len

        # Insert newly computed suffix blocks into the prefix cache
        if self.prefix_cache is not None:
            num_prefix_blocks = len(request.cached_block_ids)
            # New blocks are the suffix portion of the block table
            suffix_blocks = request.block_table[num_prefix_blocks:]
            # Only cache complete blocks that are part of the prompt
            num_prompt_blocks = self.block_manager.num_required_blocks(request.prompt_len)
            new_cacheable = suffix_blocks[:max(0, num_prompt_blocks - num_prefix_blocks)]
            if new_cacheable:
                self.block_manager.mark_cached(new_cacheable)
                # Insert the full prompt's token→block mapping into the trie
                all_prompt_blocks = request.cached_block_ids + new_cacheable
                self.prefix_cache.insert(request.prompt_token_ids, all_prompt_blocks)
            # Unpin borrowed prefix blocks — prefill is done, they can be evicted
            if request.cached_block_ids:
                self.prefix_cache.unpin(request.cached_block_ids)

    def on_chunk_complete(self, request: Request, chunk_end: int) -> None:
        """Called by engine after a non-final prefill chunk finishes."""
        request.tokens_prefilled = chunk_end
        request.log_event("CHUNK_DONE", tokens_prefilled=chunk_end,
                          prompt_len=request.prompt_len)

    def compute_chunk_size(self) -> int:
        """Adaptive chunk size: shrinks when decode queue is deep or memory is tight."""
        if not self.enable_chunked_prefill:
            return self.max_chunk_size
        decode_q = sum(1 for r in self.running if r.status == SequenceStatus.DECODING)
        total = self.block_manager.num_blocks
        free_frac = self.block_manager.num_free_blocks() / total if total > 0 else 1.0
        pressure = max(0.1, free_frac) * max(1.0, 4.0 / max(1, decode_q))
        return max(self.min_chunk_size, int(self.max_chunk_size * min(1.0, pressure)))

    def on_token_generated(
        self, request: Request, token_id: int, eos_token_id: int
    ) -> None:
        """Called by engine after each decode step."""
        request.output_token_ids.append(token_id)
        request.num_cached_tokens += 1
        request.last_token_time = time.monotonic()

        if request.first_token_time is None:
            request.first_token_time = request.last_token_time

        # Check stop conditions
        if token_id == eos_token_id:
            request.status = SequenceStatus.FINISHED_EOS
        elif len(request.output_token_ids) >= request.sampling_params.max_tokens:
            request.status = SequenceStatus.FINISHED_LENGTH
        elif request.sampling_params.stop:
            # Check stop strings (approximate: check decoded suffix)
            pass   # engine-level stop string checking done in InferenceEngine

    # ------------------------------------------------------------------
    # Preemption helpers
    # ------------------------------------------------------------------

    def _select_preemption_victim(self) -> Optional[Request]:
        """Choose a decode request to preempt.

        Heuristic: lowest priority first; within same priority, fewest
        generated tokens (minimise wasted work).
        """
        decode_reqs = [r for r in self.running if r.status == SequenceStatus.DECODING]
        if not decode_reqs:
            return None
        return max(
            decode_reqs,
            key=lambda r: (r.priority, -r.num_generated_tokens),
        )

    def _preempt(self, victim: Request, output: SchedulerOutput) -> None:
        """Free victim's blocks and re-queue it as waiting."""
        # Unpin prefix cache blocks before freeing (allows eviction again)
        if self.prefix_cache is not None and victim.cached_block_ids:
            self.prefix_cache.unpin(victim.cached_block_ids)

        self.block_manager.free(victim.request_id)
        victim.block_table.clear()
        victim.num_cached_tokens = 0
        victim.output_token_ids.clear()
        # Reset chunked prefill state
        victim.tokens_prefilled = 0
        victim.chunk_start = 0
        victim.chunk_end = 0
        # Reset prefix cache state
        victim.prefix_match_len = 0
        victim.cached_block_ids = []
        victim.status = SequenceStatus.WAITING
        victim.prefill_start_time = None
        victim.first_token_time = None

        self.running.remove(victim)
        self.waiting.push(victim)
        output.preempted_requests.append(victim)

        victim.log_event("PREEMPTED", running=len(self.running))
        plog.log_preempted(victim.request_id, len(self.running))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def num_waiting(self) -> int:
        return len(self.waiting)

    def num_running(self) -> int:
        return len(self.running)
