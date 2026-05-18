"""
LLMEngine — the main orchestrator.

Runs in a background thread.  FastAPI submits requests via a thread-safe
queue and receives results via per-request queues (blocking or streaming).

One engine step:
  1. Drain the input queue → add to scheduler.
  2. Call scheduler.schedule() → get prefill + decode lists.
  3. Run prefill forward pass for new requests.
  4. Run decode forward pass for running requests.
  5. For each finished request: signal the caller.
"""

import logging
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from transformers import AutoTokenizer

from src.core.config import EngineConfig, ModelConfig
from src.core.types import (
    AttentionMetadata,
    Request,
    SchedulerOutput,
    SamplingParams,
    SequenceStatus,
)
from src.memory.block_manager import PhysicalBlockManager
from src.memory.kv_cache import KVCacheManager
from src.model.loader import load_model
from src.model.sampling import sample_token
from src.model.transformer import LlamaForCausalLM
from src.observability.metrics import MetricsCollector
from src.observability import prompt_logger as plog
from src.scheduler.scheduler import Scheduler

log = logging.getLogger("llm.engine")


class LLMEngine:
    """The central inference engine.

    Usage:
        engine = LLMEngine.from_config(model_cfg, engine_cfg)
        engine.start()
        result_q = engine.submit(request)
        output = result_q.get()   # blocks until done
    """

    def __init__(
        self,
        model: LlamaForCausalLM,
        model_config: ModelConfig,
        engine_config: EngineConfig,
        block_manager: PhysicalBlockManager,
        kv_cache: KVCacheManager,
        tokenizer: AutoTokenizer,
        device: str,
        dtype: torch.dtype,
    ):
        self.model = model
        self.model_config = model_config
        self.engine_config = engine_config
        self.block_manager = block_manager
        self.kv_cache = kv_cache
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype

        self.scheduler = Scheduler(
            block_manager=block_manager,
            max_running_requests=engine_config.max_running_requests,
            max_waiting_requests=engine_config.max_waiting_requests,
            enable_chunked_prefill=engine_config.enable_chunked_prefill,
            max_chunk_size=engine_config.max_chunk_size,
            min_chunk_size=engine_config.min_chunk_size,
            enable_prefix_caching=engine_config.enable_prefix_caching,
        )
        self.prefix_cache = self.scheduler.prefix_cache  # may be None
        self.metrics = MetricsCollector()

        # Thread-safe communication
        self._input_queue: queue.Queue = queue.Queue()
        # request_id → queue.Queue that receives output token ids (or None = done)
        self._output_queues: Dict[str, queue.Queue] = {}

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._step_counter: int = 0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls, model_config: ModelConfig, engine_config: EngineConfig
    ) -> "LLMEngine":
        device = engine_config.resolve_device()
        dtype = engine_config.resolve_dtype(device)

        print(f"Engine starting on device={device}, dtype={dtype}")

        weights, loaded_cfg, tokenizer = load_model(
            model_id=model_config.model_id,
            device=device,
            dtype=dtype,
        )
        # Override config with loaded values (in case model_id differs)
        model_config = loaded_cfg

        block_manager = PhysicalBlockManager(
            num_blocks=engine_config.num_blocks,
            block_size=engine_config.block_size,
        )
        kv_cache = KVCacheManager(
            num_layers=model_config.num_hidden_layers,
            num_blocks=engine_config.num_blocks,
            block_size=engine_config.block_size,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            device=device,
            dtype=dtype,
        )
        model = LlamaForCausalLM(
            config=model_config,
            weights=weights,
            device=device,
            dtype=dtype,
        )

        return cls(
            model=model,
            model_config=model_config,
            engine_config=engine_config,
            block_manager=block_manager,
            kv_cache=kv_cache,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background engine loop thread."""
        self._thread = threading.Thread(
            target=self._run_loop, name="engine-loop", daemon=True
        )
        self._thread.start()
        log.info("Engine loop started.")
        print("Engine loop started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, request: Request) -> queue.Queue:
        """Submit a request; returns a queue that yields token ids, None when done."""
        out_q: queue.Queue = queue.Queue()
        request._token_queue = out_q
        request.log_event("SUBMITTED", prompt_tokens=request.prompt_len,
                          max_tokens=request.sampling_params.max_tokens)
        plog.log_submitted(request.request_id, request.prompt_len,
                           request.sampling_params.max_tokens)
        self._input_queue.put(request)
        return out_q

    def generate(self, prompt: str, sampling_params: SamplingParams) -> str:
        """Synchronous helper: block until generation completes."""
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        req = Request(
            prompt=prompt,
            prompt_token_ids=token_ids,
            sampling_params=sampling_params,
        )
        out_q = self.submit(req)
        output_token_ids = []
        while True:
            tok = out_q.get()
            if tok is None:
                break
            output_token_ids.append(tok)
        return self.tokenizer.decode(output_token_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Engine loop (runs in background thread)
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self._drain_input_queue()

            if not self.scheduler.has_work():
                time.sleep(0.002)
                continue

            sched_out = self.scheduler.schedule()
            self._step_counter += 1
            plog.log_schedule_step(
                step=self._step_counter,
                prefill_count=len(sched_out.prefill_requests),
                decode_count=len(sched_out.decode_requests),
                preempted_count=len(sched_out.preempted_requests),
                running_total=self.scheduler.num_running(),
                waiting_total=self.scheduler.num_waiting(),
            )

            if sched_out.is_empty():
                time.sleep(0.001)
                continue

            self._execute_step(sched_out)

    def _drain_input_queue(self) -> None:
        while True:
            try:
                req = self._input_queue.get_nowait()
                self.scheduler.add_request(req)
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def _execute_step(self, sched_out: SchedulerOutput) -> None:
        if sched_out.prefill_requests:
            self._step_prefill(sched_out.prefill_requests)

        if sched_out.decode_requests:
            self._step_decode(sched_out.decode_requests)

    def _step_prefill(self, requests: List[Request]) -> None:
        """Process the full prompt for each prefill request (no batching across requests)."""
        for i, req in enumerate(requests):
            self._prefill_one(req, batch_pos=i, batch_size=len(requests))

    def _prefill_one(self, req: Request, batch_pos: int = 0, batch_size: int = 1) -> None:
        chunk_start = req.chunk_start
        chunk_end = req.chunk_end
        chunk_len = chunk_end - chunk_start
        is_final_chunk = (chunk_end == req.prompt_len)

        t0 = time.monotonic()
        req.log_event("PREFILL_CHUNK_START", chunk_start=chunk_start, chunk_end=chunk_end,
                      prompt_len=req.prompt_len, batch_pos=batch_pos, batch_size=batch_size)
        plog.log_prefill_start(req.request_id, chunk_len, batch_pos, batch_size)

        # Record prefix cache hit/miss on the very first chunk of this request
        if chunk_start == req.prefix_match_len:
            if req.prefix_match_len > 0:
                self.metrics.record_prefix_cache_hit(req.prefix_match_len)
            else:
                self.metrics.record_prefix_cache_miss()

        chunk_ids = req.prompt_token_ids[chunk_start:chunk_end]
        input_ids = torch.tensor(chunk_ids, dtype=torch.long, device=self.device)
        positions = torch.arange(chunk_start, chunk_end, dtype=torch.long, device=self.device)

        metadata = AttentionMetadata(
            prefill_seq_lens=[chunk_len],
            prefill_block_tables=[req.block_table],
            decode_context_lens=[],
            decode_block_tables=[],
            prefill_chunk_starts=[chunk_start],
        )

        with torch.no_grad():
            logits = self.model.forward(input_ids, positions, self.kv_cache, metadata)

        elapsed_ms = (time.monotonic() - t0) * 1000

        if is_final_chunk:
            last_logits = logits[-1].float()
            next_tok = sample_token(last_logits, req.sampling_params)

            req.log_event("PREFILL_DONE", first_token=next_tok, ttft_ms=round(elapsed_ms, 2))
            plog.log_prefill_done(req.request_id, next_tok, elapsed_ms)

            self.scheduler.on_prefill_complete(req)
            self.scheduler.on_token_generated(req, next_tok, self.tokenizer.eos_token_id)
            self.metrics.record_token()

            if req._token_queue is not None:
                req._token_queue.put(next_tok)

            if req.is_finished():
                self._finalize_request(req)
        else:
            # Non-final chunk: advance progress, do NOT sample a token
            req.log_event("CHUNK_DONE", chunk_end=chunk_end, elapsed_ms=round(elapsed_ms, 2))
            self.scheduler.on_chunk_complete(req, chunk_end)

    def _step_decode(self, requests: List[Request]) -> None:
        """Generate one token for each decode request — processed individually."""
        for i, req in enumerate(requests):
            if req.is_finished():
                continue
            self._decode_one(req, batch_pos=i, batch_size=len(requests))

    def _decode_one(self, req: Request, batch_pos: int = 0, batch_size: int = 1) -> None:
        ctx_len = req.num_cached_tokens
        step = req.num_generated_tokens  # already has the prefill-generated token

        input_ids = torch.tensor([req.last_token_id], dtype=torch.long, device=self.device)
        positions = torch.tensor([ctx_len - 1], dtype=torch.long, device=self.device)

        metadata = AttentionMetadata(
            prefill_seq_lens=[],
            prefill_block_tables=[],
            decode_context_lens=[ctx_len],
            decode_block_tables=[req.block_table],
        )

        with torch.no_grad():
            logits = self.model.forward(input_ids, positions, self.kv_cache, metadata)

        next_tok = sample_token(logits[0].float(), req.sampling_params)

        self.scheduler.on_token_generated(req, next_tok, self.tokenizer.eos_token_id)
        self.metrics.record_token()

        plog.log_decode_step(req.request_id, step, next_tok, ctx_len, batch_pos, batch_size)
        req.log_event("DECODE_STEP", step=step, token=next_tok, ctx_len=ctx_len,
                      batch_pos=batch_pos, batch_size=batch_size)

        if not req.is_finished() and req.sampling_params.stop:
            decoded = self.tokenizer.decode(req.output_token_ids[-20:], skip_special_tokens=False)
            for stop in req.sampling_params.stop:
                if stop in decoded:
                    req.status = SequenceStatus.FINISHED_STOP
                    break

        if req._token_queue is not None:
            req._token_queue.put(next_tok)

        if req.is_finished():
            self._finalize_request(req)

    def _finalize_request(self, req: Request) -> None:
        self.metrics.record_request_complete(req)
        total_ms = 0.0
        if req.prefill_start_time and req.last_token_time:
            total_ms = (req.last_token_time - req.prefill_start_time) * 1000
        req.log_event(
            "FINISHED",
            reason=req.status.value,
            output_tokens=req.num_generated_tokens,
            total_ms=round(total_ms, 2),
            ttft_ms=round(req.ttft_ms() or 0, 2),
            tpot_ms=round(req.tpot_ms() or 0, 2),
        )
        plog.log_finished(
            req.request_id, req.status.value,
            req.num_generated_tokens, total_ms,
            req.ttft_ms(), req.tpot_ms(),
        )
        if req._token_queue is not None:
            req._token_queue.put(None)

    # ------------------------------------------------------------------
    # Stats helpers for the API
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        m = self.metrics.snapshot()
        m["num_running_requests"] = self.scheduler.num_running()
        m["num_waiting_requests"] = self.scheduler.num_waiting()
        m["kv_cache_blocks_used"] = self.block_manager.num_used_blocks()
        m["kv_cache_blocks_free"] = self.block_manager.num_free_blocks()
        m["kv_cache_utilization_pct"] = round(self.block_manager.utilization() * 100, 1)
        m["device"] = self.device
        m["model_id"] = self.model_config.model_id
        if self.prefix_cache is not None:
            pc = self.prefix_cache.stats()
            m["prefix_cache_hits"] = pc["hit_count"]
            m["prefix_cache_misses"] = pc["miss_count"]
            m["prefix_cache_hit_rate_pct"] = round(pc["hit_rate"] * 100, 1)
            m["prefix_cached_blocks"] = pc["cached_blocks"]
        return m
