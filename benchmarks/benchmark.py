"""
Comparison benchmark: HuggingFace baseline vs this engine.

Runs the same prompts through:
  1. HuggingFace transformers pipeline (standard, sequential, no batching tricks)
  2. This engine (PagedAttention + continuous batching + chunked prefill + prefix cache)

Then prints a side-by-side table so you can see exactly what the optimizations buy.

Usage:
    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --num-requests 8 --max-tokens 100 --device mps
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.core.config import EngineConfig, ModelConfig
from src.core.types import Request, SamplingParams
from src.engine.inference_engine import LLMEngine


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

PROMPTS = [
    "Explain what a transformer neural network is in simple terms.",
    "Write a short poem about the ocean.",
    "What are the main differences between Python and Go?",
    "Describe the water cycle in three sentences.",
    "How does attention mechanism work in LLMs?",
    "What is gradient descent and why is it used?",
    "Explain the concept of tokenization in NLP.",
    "What makes a good software architecture?",
    "What is the difference between a list and a tuple in Python?",
    "Explain what recursion is with a simple example.",
]

SEP = "─" * 62


# ── Baseline: HuggingFace pipeline, sequential ────────────────────────

def run_hf_baseline(
    prompts: list[str],
    max_tokens: int,
    device: str,
    temperature: float,
) -> dict:
    print(f"\n{'='*62}")
    print(f"  [1/2] HuggingFace baseline (sequential, no custom batching)")
    print(f"{'='*62}")

    hf_device = 0 if device == "cuda" else device
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    if device == "mps":
        model = model.to("mps")
    elif device == "cuda":
        model = model.cuda()

    do_sample = temperature > 0
    ttfts, tpots, output_lens = [], [], []
    wall_start = time.monotonic()

    for i, prompt in enumerate(prompts):
        print(f"  Request {i+1}/{len(prompts)}: {prompt[:50]}...")
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
        if device == "mps":
            input_ids = input_ids.to("mps")
        elif device == "cuda":
            input_ids = input_ids.cuda()

        prompt_len = input_ids.shape[1]

        # TTFT: time until first new token is produced
        t_start = time.monotonic()
        with torch.no_grad():
            # Generate first token only
            out_first = model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        ttft_ms = (time.monotonic() - t_start) * 1000
        ttfts.append(ttft_ms)

        # Generate remaining tokens and measure TPOT
        remaining = max_tokens - 1
        if remaining > 0:
            t_decode = time.monotonic()
            with torch.no_grad():
                out_full = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            decode_ms = (time.monotonic() - t_decode) * 1000
            n_out = out_full.shape[1] - prompt_len
            if n_out > 1:
                tpots.append(decode_ms / (n_out - 1))
            output_lens.append(n_out)
        else:
            output_lens.append(1)

    wall_elapsed = time.monotonic() - wall_start
    total_tokens = sum(output_lens)
    throughput = total_tokens / wall_elapsed if wall_elapsed > 0 else 0

    # Free model memory before loading engine
    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    result = {
        "label": "HuggingFace (sequential)",
        "num_requests": len(prompts),
        "max_tokens": max_tokens,
        "total_output_tokens": total_tokens,
        "wall_time_s": round(wall_elapsed, 2),
        "throughput_tok_s": round(throughput, 1),
        "mean_ttft_ms": round(statistics.mean(ttfts), 1) if ttfts else None,
        "p50_ttft_ms": round(statistics.median(ttfts), 1) if ttfts else None,
        "mean_tpot_ms": round(statistics.mean(tpots), 1) if tpots else None,
        "p50_tpot_ms": round(statistics.median(tpots), 1) if tpots else None,
    }

    _print_result(result)
    return result


# ── Our engine: PagedAttention + continuous batching ─────────────────

def run_our_engine(
    prompts: list[str],
    max_tokens: int,
    device: str,
    num_blocks: int,
    temperature: float,
) -> dict:
    print(f"\n{'='*62}")
    print(f"  [2/2] This engine (PagedAttention + continuous batching)")
    print(f"{'='*62}")

    model_cfg = ModelConfig(model_id=MODEL_ID)
    engine_cfg = EngineConfig(
        num_blocks=num_blocks,
        block_size=16,
        device=device,
        max_running_requests=len(prompts),
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
    )

    print(f"  Loading engine...")
    engine = LLMEngine.from_config(model_cfg, engine_cfg)
    engine.start()

    # Warm-up (one short request, not counted)
    warm = Request(
        prompt="Hi",
        prompt_token_ids=engine.tokenizer.encode("Hi"),
        sampling_params=SamplingParams(max_tokens=5, temperature=0.0),
    )
    q = engine.submit(warm)
    while q.get() is not None:
        pass

    # Build all requests
    requests = []
    for p in prompts:
        ids = engine.tokenizer.encode(p, add_special_tokens=True)
        req = Request(
            prompt=p,
            prompt_token_ids=ids,
            sampling_params=SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
        requests.append(req)

    print(f"  Submitting {len(requests)} requests concurrently...")
    wall_start = time.monotonic()

    # Submit all at once — this exercises continuous batching
    queues = [(req, engine.submit(req)) for req in requests]

    output_lens = []
    ttfts, tpots = [], []

    for req, q in queues:
        output_ids = []
        while True:
            tok = q.get()
            if tok is None:
                break
            output_ids.append(tok)
        output_lens.append(len(output_ids))
        if req.ttft_ms():
            ttfts.append(req.ttft_ms())
        if req.tpot_ms():
            tpots.append(req.tpot_ms())

    wall_elapsed = time.monotonic() - wall_start
    total_tokens = sum(output_lens)
    throughput = total_tokens / wall_elapsed if wall_elapsed > 0 else 0

    pc_stats = engine.prefix_cache.stats() if engine.prefix_cache else None
    engine.stop()

    result = {
        "label": "This engine (continuous batching)",
        "num_requests": len(requests),
        "max_tokens": max_tokens,
        "total_output_tokens": total_tokens,
        "wall_time_s": round(wall_elapsed, 2),
        "throughput_tok_s": round(throughput, 1),
        "mean_ttft_ms": round(statistics.mean(ttfts), 1) if ttfts else None,
        "p50_ttft_ms": round(statistics.median(ttfts), 1) if ttfts else None,
        "mean_tpot_ms": round(statistics.mean(tpots), 1) if tpots else None,
        "p50_tpot_ms": round(statistics.median(tpots), 1) if tpots else None,
        "output_tokens_per_request": output_lens,
        "prefix_cache_hits": pc_stats["hit_count"] if pc_stats else 0,
        "prefix_cache_hit_rate_pct": round(pc_stats["hit_rate"] * 100, 1) if pc_stats else 0,
    }

    _print_result(result)
    return result


# ── Prefix cache benchmark ────────────────────────────────────────────

def run_prefix_cache_benchmark(device: str, num_blocks: int) -> None:
    """
    Demonstrate prefix caching: same 200-token system prompt prefix, 5 different questions.
    First request is a cold miss. Subsequent requests should hit the cache.
    """
    print(f"\n{'='*62}")
    print(f"  Prefix Cache Demo")
    print(f"  (shared 200-token system prompt + unique question each time)")
    print(f"{'='*62}")

    system_prompt = (
        "You are an expert AI assistant specializing in computer science and mathematics. "
        "You always provide clear, accurate, and concise answers. "
        "You cite relevant concepts and give examples where appropriate. "
        "Your tone is professional but approachable. "
        "You break complex topics into digestible parts and ensure the user understands. "
        "When asked about algorithms or systems, you explain both theory and practical implications. "
        "You never make up facts. If unsure, you say so. "
    ) * 2  # repeat to get ~200 tokens

    questions = [
        "What is a hash table?",
        "What is a binary search tree?",
        "What is dynamic programming?",
        "What is a neural network?",
        "What is the CAP theorem?",
    ]

    model_cfg = ModelConfig(model_id=MODEL_ID)
    engine_cfg = EngineConfig(
        num_blocks=num_blocks,
        block_size=16,
        device=device,
        max_running_requests=4,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
    )

    print(f"  Loading engine...")
    engine = LLMEngine.from_config(model_cfg, engine_cfg)
    engine.start()

    ttfts = []
    for i, q_text in enumerate(questions):
        full_prompt = system_prompt + "\n\nQuestion: " + q_text
        ids = engine.tokenizer.encode(full_prompt, add_special_tokens=True)
        req = Request(
            prompt=full_prompt,
            prompt_token_ids=ids,
            sampling_params=SamplingParams(max_tokens=30, temperature=0.0),
        )
        out_q = engine.submit(req)
        while out_q.get() is not None:
            pass
        ttft = req.ttft_ms()
        if ttft:
            ttfts.append(ttft)
        cache_label = "MISS (cold)" if i == 0 else "HIT  (cached)"
        print(f"  Request {i+1}: [{cache_label}]  prefix_matched={req.prefix_match_len} tokens  ttft={ttft:.0f}ms" if ttft else f"  Request {i+1}: ttft=N/A")

    pc = engine.prefix_cache.stats() if engine.prefix_cache else {}
    engine.stop()

    print(f"\n  Prefix cache stats:")
    print(f"  ├─ Cached blocks     : {pc.get('cached_blocks', 0)}")
    print(f"  ├─ Hits / Misses     : {pc.get('hit_count', 0)} / {pc.get('miss_count', 0)}")
    print(f"  └─ Hit rate          : {pc.get('hit_rate', 0)*100:.0f}%")
    if len(ttfts) >= 2:
        cold_ttft = ttfts[0]
        warm_ttft = statistics.mean(ttfts[1:])
        reduction = (1 - warm_ttft / cold_ttft) * 100
        print(f"\n  TTFT: cold={cold_ttft:.0f}ms  warm(avg)={warm_ttft:.0f}ms  reduction={reduction:.0f}%")


# ── Helpers ───────────────────────────────────────────────────────────

def _print_result(r: dict) -> None:
    print(f"\n  Results — {r['label']}")
    print(f"  {SEP}")
    print(f"  Requests            : {r['num_requests']} × {r['max_tokens']} max_tokens")
    print(f"  Total output tokens : {r['total_output_tokens']}")
    print(f"  Wall time           : {r['wall_time_s']}s")
    print(f"  Throughput          : {r['throughput_tok_s']} tok/s")
    ttft_str = f"{r['mean_ttft_ms']}ms (p50={r['p50_ttft_ms']}ms)" if r['mean_ttft_ms'] else "N/A"
    tpot_str = f"{r['mean_tpot_ms']}ms (p50={r['p50_tpot_ms']}ms)" if r['mean_tpot_ms'] else "N/A"
    print(f"  TTFT (mean)         : {ttft_str}")
    print(f"  TPOT (mean)         : {tpot_str}")
    if "prefix_cache_hits" in r:
        print(f"  Prefix cache hits   : {r['prefix_cache_hits']}  ({r['prefix_cache_hit_rate_pct']}%)")
    print(f"  {SEP}")


def _print_comparison(hf: dict, ours: dict) -> None:
    print(f"\n{'='*62}")
    print(f"  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*62}")

    def pct(a, b, lower_is_better=True):
        if a is None or b is None or b == 0:
            return "N/A"
        ratio = a / b
        improvement = (ratio - 1) * 100 if not lower_is_better else (1 - ratio) * 100
        sign = "+" if improvement > 0 else ""
        return f"{sign}{improvement:.0f}%"

    rows = [
        ("Wall time",      f"{hf['wall_time_s']}s",        f"{ours['wall_time_s']}s",
         pct(ours['wall_time_s'], hf['wall_time_s'], lower_is_better=True)),
        ("Throughput",     f"{hf['throughput_tok_s']} tok/s", f"{ours['throughput_tok_s']} tok/s",
         pct(ours['throughput_tok_s'], hf['throughput_tok_s'], lower_is_better=False)),
        ("TTFT (mean)",    f"{hf['mean_ttft_ms']}ms" if hf['mean_ttft_ms'] else "N/A",
                           f"{ours['mean_ttft_ms']}ms" if ours['mean_ttft_ms'] else "N/A",
         pct(ours['mean_ttft_ms'], hf['mean_ttft_ms'], lower_is_better=True)),
        ("TPOT (mean)",    f"{hf['mean_tpot_ms']}ms" if hf['mean_tpot_ms'] else "N/A",
                           f"{ours['mean_tpot_ms']}ms" if ours['mean_tpot_ms'] else "N/A",
         pct(ours['mean_tpot_ms'], hf['mean_tpot_ms'], lower_is_better=True)),
        ("Output tokens",  str(hf['total_output_tokens']),  str(ours['total_output_tokens']), ""),
    ]

    print(f"  {'Metric':<20} {'HuggingFace':>14} {'This engine':>14} {'Delta':>8}")
    print(f"  {SEP}")
    for label, hf_val, our_val, delta in rows:
        print(f"  {label:<20} {hf_val:>14} {our_val:>14} {delta:>8}")
    print(f"  {SEP}")
    print()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HF baseline vs this engine")
    parser.add_argument("--num-requests", type=int, default=5)
    parser.add_argument("--max-tokens",   type=int, default=64)
    parser.add_argument("--num-blocks",   type=int, default=256)
    parser.add_argument("--device",       default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--temperature",  type=float, default=0.8)
    parser.add_argument("--skip-hf",      action="store_true", help="Skip HF baseline (faster)")
    parser.add_argument("--prefix-demo",  action="store_true", help="Also run prefix cache demo")
    args = parser.parse_args()

    # Resolve device
    device = args.device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.num_requests)]

    print(f"\n  Device : {device}")
    print(f"  Model  : {MODEL_ID}")
    print(f"  Config : {args.num_requests} requests × {args.max_tokens} tokens, temp={args.temperature}")

    hf_result = None
    if not args.skip_hf:
        hf_result = run_hf_baseline(prompts, args.max_tokens, device, args.temperature)

    our_result = run_our_engine(prompts, args.max_tokens, device, args.num_blocks, args.temperature)

    if hf_result is not None:
        _print_comparison(hf_result, our_result)

    if args.prefix_demo:
        run_prefix_cache_benchmark(device, args.num_blocks)


if __name__ == "__main__":
    main()
