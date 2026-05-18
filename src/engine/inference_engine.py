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
from src.scheduler.scheduler import Scheduler


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
        )
        self.metrics = MetricsCollector()

        # Thread-safe communication
        self._input_queue: queue.Queue = queue.Queue()
        # request_id → queue.Queue that receives output token ids (or None = done)
        self._output_queues: Dict[str, queue.Queue] = {}

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

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
        print("Engine loop started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, request: Request) -> queue.Queue:
        """Submit a request; returns a queue that yields token ids, None when done."""
        out_q: queue.Queue = queue.Queue()
        request._token_queue = out_q
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
            # Drain input queue
            self._drain_input_queue()

            if not self.scheduler.has_work():
                time.sleep(0.002)
                continue

            sched_out = self.scheduler.schedule()
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
        for req in requests:
            self._prefill_one(req)

    def _prefill_one(self, req: Request) -> None:
        prompt_ids = req.prompt_token_ids
        seq_len = len(prompt_ids)

        input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
        positions = torch.arange(seq_len, dtype=torch.long, device=self.device)

        metadata = AttentionMetadata(
            prefill_seq_lens=[seq_len],
            prefill_block_tables=[req.block_table],
            decode_context_lens=[],
            decode_block_tables=[],
        )

        with torch.no_grad():
            logits = self.model.forward(input_ids, positions, self.kv_cache, metadata)

        # Sample next token from last-position logits
        last_logits = logits[-1].float()
        next_tok = sample_token(last_logits, req.sampling_params)

        self.scheduler.on_prefill_complete(req)        # → DECODING, num_cached = prompt_len
        self.scheduler.on_token_generated(
            req, next_tok, self.tokenizer.eos_token_id
        )
        self.metrics.record_token()

        # Push token to caller
        if req._token_queue is not None:
            req._token_queue.put(next_tok)

        # If finished right after first token, notify
        if req.is_finished():
            self._finalize_request(req)

    def _step_decode(self, requests: List[Request]) -> None:
        """Generate one token for each decode request — processed individually."""
        for req in requests:
            if req.is_finished():
                continue
            self._decode_one(req)

    def _decode_one(self, req: Request) -> None:
        ctx_len = req.num_cached_tokens   # total tokens in KV cache

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

        # Check stop strings
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
        if req._token_queue is not None:
            req._token_queue.put(None)   # sentinel: generation done

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
        return m
