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

import time
from collections import deque
from typing import List, Optional

from src.core.types import Request, SchedulerOutput, SequenceStatus
from src.memory.block_manager import PhysicalBlockManager
from src.scheduler.request_queue import PriorityRequestQueue


class Scheduler:
    def __init__(
        self,
        block_manager: PhysicalBlockManager,
        max_running_requests: int = 8,
        max_waiting_requests: int = 256,
    ):
        self.block_manager = block_manager
        self.max_running = max_running_requests
        self.max_waiting = max_waiting_requests

        self.waiting = PriorityRequestQueue()
        self.running: List[Request] = []          # prefilling or decoding
        self.preempted: deque[Request] = deque()  # re-queued for later

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        request.status = SequenceStatus.WAITING
        self.waiting.push(request)

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
            blocks_needed = self.block_manager.num_required_blocks(
                req.prompt_len + req.sampling_params.max_tokens
            )

            if self.block_manager.can_allocate(blocks_needed):
                self.waiting.pop()
                allocated = self.block_manager.allocate(req.request_id, blocks_needed)
                req.block_table = allocated
                req.status = SequenceStatus.PREFILLING
                req.prefill_start_time = time.monotonic()
                self.running.append(req)
            else:
                # Try preemption to make room
                victim = self._select_preemption_victim()
                if victim is not None:
                    self._preempt(victim, output)
                else:
                    break  # cannot make progress, stop trying

        # ── Step 4: split into prefill / decode ─────────────────────
        for req in self.running:
            if req.status == SequenceStatus.PREFILLING:
                output.prefill_requests.append(req)
            elif req.status == SequenceStatus.DECODING:
                output.decode_requests.append(req)

        return output

    def on_prefill_complete(self, request: Request) -> None:
        """Called by engine after a request's prefill step finishes."""
        request.status = SequenceStatus.DECODING
        request.num_cached_tokens = request.prompt_len

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
        self.block_manager.free(victim.request_id)
        victim.block_table.clear()
        victim.num_cached_tokens = 0
        # Drop generated tokens — will re-do prefill
        victim.output_token_ids.clear()
        victim.status = SequenceStatus.WAITING
        victim.prefill_start_time = None
        victim.first_token_time = None

        self.running.remove(victim)
        self.waiting.push(victim)
        output.preempted_requests.append(victim)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def num_waiting(self) -> int:
        return len(self.waiting)

    def num_running(self) -> int:
        return len(self.running)
