"""
Integration-level tests for the inference engine.
These tests mock the model forward pass to avoid downloading weights.
"""

import queue
import time
import unittest
from unittest.mock import MagicMock, patch

import torch

from src.core.config import EngineConfig, ModelConfig
from src.core.types import Request, SamplingParams, SequenceStatus
from src.memory.block_manager import PhysicalBlockManager
from src.memory.kv_cache import KVCacheManager
from src.model.transformer import LlamaForCausalLM
from src.engine.inference_engine import LLMEngine
from src.observability.metrics import MetricsCollector
from src.scheduler.scheduler import Scheduler


def _make_engine_no_model() -> LLMEngine:
    """Build a minimal LLMEngine with a mocked model (no real weights)."""
    model_cfg = ModelConfig(
        model_id="test",
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=100,
    )
    engine_cfg = EngineConfig(num_blocks=32, block_size=4, max_running_requests=4)
    device = "cpu"
    dtype = torch.float32

    bm = PhysicalBlockManager(num_blocks=32, block_size=4)
    kv = KVCacheManager(
        num_layers=2, num_blocks=32, block_size=4,
        num_kv_heads=2, head_dim=16, device=device, dtype=dtype,
    )

    # Mock model that returns random logits
    mock_model = MagicMock(spec=LlamaForCausalLM)
    mock_model.forward.return_value = torch.randn(1, 100)

    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    mock_tokenizer.decode.return_value = "hello world"

    engine = LLMEngine(
        model=mock_model,
        model_config=model_cfg,
        engine_config=engine_cfg,
        block_manager=bm,
        kv_cache=kv,
        tokenizer=mock_tokenizer,
        device=device,
        dtype=dtype,
    )
    return engine


class TestMetricsCollector(unittest.TestCase):
    def test_throughput_zero_initially(self):
        m = MetricsCollector()
        assert m.throughput_tok_s() == 0.0

    def test_record_tokens(self):
        m = MetricsCollector()
        for _ in range(10):
            m.record_token()
        # After at least 2 tokens we can compute throughput
        assert m.throughput_tok_s() >= 0.0


class TestBlockManagerInEngine(unittest.TestCase):
    def test_blocks_allocated_on_schedule(self):
        engine = _make_engine_no_model()
        req = Request(
            prompt="hello",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=4),
        )
        engine.scheduler.add_request(req)
        out = engine.scheduler.schedule()
        assert len(out.prefill_requests) == 1
        assert len(req.block_table) > 0

    def test_blocks_freed_on_finish(self):
        engine = _make_engine_no_model()
        req = Request(
            prompt="hello",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=1),
        )
        engine.scheduler.add_request(req)
        engine.scheduler.schedule()
        engine.scheduler.on_prefill_complete(req)
        engine.scheduler.on_token_generated(req, 2, eos_token_id=2)  # EOS
        assert req.is_finished()
        # After schedule, blocks should be freed
        engine.scheduler.schedule()
        # No blocks still owned by finished request
        assert engine.block_manager.get_block_table(req.request_id) == []


class TestEngineStats(unittest.TestCase):
    def test_stats_keys_present(self):
        engine = _make_engine_no_model()
        stats = engine.stats()
        for key in ["num_running_requests", "num_waiting_requests",
                    "kv_cache_blocks_used", "kv_cache_blocks_free",
                    "throughput_tok_s"]:
            assert key in stats, f"Missing stat key: {key}"


if __name__ == "__main__":
    unittest.main()
