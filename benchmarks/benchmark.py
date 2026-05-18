"""
Benchmark suite — measures TTFT, TPOT, throughput, and memory usage.

Usage:
    python benchmarks/benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python benchmarks/benchmark.py --num-requests 10 --max-tokens 128
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import EngineConfig, ModelConfig
from src.core.types import Request, SamplingParams
from src.engine.inference_engine import LLMEngine


PROMPTS = [
    "Explain what a transformer neural network is in simple terms.",
    "Write a short poem about the ocean.",
    "What are the main differences between Python and Go?",
    "Describe the water cycle in three sentences.",
    "How does attention mechanism work in LLMs?",
    "What is gradient descent and why is it used?",
    "Explain the concept of tokenization in NLP.",
    "What makes a good software architecture?",
]


def run_benchmark(
    engine: LLMEngine,
    num_requests: int,
    max_tokens: int,
    temperature: float = 0.0,
    concurrent: bool = True,
) -> dict:
    tokenizer = engine.tokenizer

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_requests)]
    requests = []
    for p in prompts:
        ids = tokenizer.encode(p, add_special_tokens=True)
        req = Request(
            prompt=p,
            prompt_token_ids=ids,
            sampling_params=SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
        requests.append(req)

    print(f"\n{'='*60}")
    print(f"  Benchmark: {num_requests} requests × {max_tokens} max_tokens")
    print(f"  Concurrent: {concurrent}")
    print(f"{'='*60}")

    ttfts = []
    tpots = []
    total_output_tokens = 0
    wall_start = time.monotonic()

    if concurrent:
        # Submit all at once (tests continuous batching)
        queues = []
        for req in requests:
            q = engine.submit(req)
            queues.append((req, q))

        for req, q in queues:
            output_ids = []
            while True:
                tok = q.get()
                if tok is None:
                    break
                output_ids.append(tok)
            total_output_tokens += len(output_ids)
            ttft = req.ttft_ms()
            tpot = req.tpot_ms()
            if ttft:
                ttfts.append(ttft)
            if tpot:
                tpots.append(tpot)
    else:
        # Sequential (baseline)
        for req in requests:
            q = engine.submit(req)
            output_ids = []
            while True:
                tok = q.get()
                if tok is None:
                    break
                output_ids.append(tok)
            total_output_tokens += len(output_ids)
            if req.ttft_ms():
                ttfts.append(req.ttft_ms())
            if req.tpot_ms():
                tpots.append(req.tpot_ms())

    wall_elapsed = time.monotonic() - wall_start
    throughput = total_output_tokens / wall_elapsed if wall_elapsed > 0 else 0

    def fmt(vals, unit="ms"):
        if not vals:
            return "N/A"
        return f"{statistics.mean(vals):.1f} {unit} (p50={statistics.median(vals):.1f})"

    result = {
        "num_requests": num_requests,
        "max_tokens": max_tokens,
        "total_output_tokens": total_output_tokens,
        "wall_time_s": round(wall_elapsed, 2),
        "throughput_tok_s": round(throughput, 1),
        "mean_ttft_ms": round(statistics.mean(ttfts), 1) if ttfts else None,
        "p50_ttft_ms": round(statistics.median(ttfts), 1) if ttfts else None,
        "p95_ttft_ms": round(sorted(ttfts)[int(len(ttfts) * 0.95)], 1) if len(ttfts) > 1 else None,
        "mean_tpot_ms": round(statistics.mean(tpots), 1) if tpots else None,
        "p50_tpot_ms": round(statistics.median(tpots), 1) if tpots else None,
        "kv_blocks_used": engine.block_manager.num_used_blocks(),
        "kv_blocks_free": engine.block_manager.num_free_blocks(),
    }

    print(f"\n  Results:")
    print(f"  ├─ Total output tokens : {total_output_tokens}")
    print(f"  ├─ Wall time           : {wall_elapsed:.2f}s")
    print(f"  ├─ Throughput          : {throughput:.1f} tok/s")
    print(f"  ├─ TTFT                : {fmt(ttfts)}")
    print(f"  ├─ TPOT                : {fmt(tpots)}")
    print(f"  └─ KV blocks used/free : {engine.block_manager.num_used_blocks()}"
          f"/{engine.block_manager.num_free_blocks()}")

    return result


def main():
    parser = argparse.ArgumentParser(description="LLM Serving Engine Benchmark")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--num-requests", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model_cfg = ModelConfig(model_id=args.model)
    engine_cfg = EngineConfig(
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        device=args.device,
        max_running_requests=8,
    )

    print(f"Loading engine with model: {args.model}")
    engine = LLMEngine.from_config(model_cfg, engine_cfg)
    engine.start()

    try:
        # Warm-up
        print("\nWarm-up run...")
        warm = Request(
            prompt="Hello",
            prompt_token_ids=engine.tokenizer.encode("Hello"),
            sampling_params=SamplingParams(max_tokens=10, temperature=0.0),
        )
        q = engine.submit(warm)
        while q.get() is not None:
            pass

        # Concurrent benchmark
        run_benchmark(engine, args.num_requests, args.max_tokens, concurrent=True)

    finally:
        engine.stop()


if __name__ == "__main__":
    main()
