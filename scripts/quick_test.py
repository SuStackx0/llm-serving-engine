"""
Quick end-to-end sanity test — loads model, runs inference, prints stats.

Usage:
    python scripts/quick_test.py
    python scripts/quick_test.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python scripts/quick_test.py --max-tokens 50 --device cpu
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.core.config import EngineConfig, ModelConfig
from src.core.types import Request, SamplingParams
from src.engine.inference_engine import LLMEngine
from src.model.rope import RotaryEmbedding

console = Console()

PROMPTS = [
    ("What is PagedAttention?",
     "Explain how vLLM's PagedAttention works."),
    ("Python tip",
     "Give me one practical Python tip in two sentences."),
    ("Haiku",
     "Write a haiku about machine learning."),
]


def run_tests(engine: LLMEngine, max_tokens: int):
    console.print("\n[bold cyan]Running inference tests...[/bold cyan]\n")

    results = []
    queues_and_reqs = []

    # Submit all prompts concurrently
    for title, prompt in PROMPTS:
        ids = engine.tokenizer.encode(prompt, add_special_tokens=True)
        req = Request(
            prompt=prompt,
            prompt_token_ids=ids,
            sampling_params=SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=max_tokens,
            ),
        )
        q = engine.submit(req)
        queues_and_reqs.append((title, prompt, req, q))

    # Collect results
    for title, prompt, req, q in queues_and_reqs:
        output_ids = []
        while True:
            tok = q.get()
            if tok is None:
                break
            output_ids.append(tok)
        output_text = engine.tokenizer.decode(output_ids, skip_special_tokens=True)
        results.append({
            "title": title,
            "prompt": prompt,
            "output": output_text,
            "ttft_ms": req.ttft_ms(),
            "tpot_ms": req.tpot_ms(),
            "tokens": len(output_ids),
            "finish": req.status.value,
        })

    # Print results
    for r in results:
        console.print(Panel(
            f"[dim]Prompt:[/dim] {r['prompt']}\n\n"
            f"[green]{r['output']}[/green]\n\n"
            f"[dim]tokens={r['tokens']} | "
            f"TTFT={r['ttft_ms']:.0f}ms | "
            f"TPOT={r['tpot_ms']:.0f}ms | "
            f"finish={r['finish']}[/dim]",
            title=f"[bold]{r['title']}[/bold]",
        ))

    # Stats table
    stats = engine.stats()
    table = Table(title="Engine Stats", show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    rows = [
        ("Model", stats["model_id"]),
        ("Device", stats["device"]),
        ("Throughput (tok/s)", f"{stats['throughput_tok_s']:.1f}"),
        ("Avg TTFT (ms)", f"{stats['avg_ttft_ms']:.1f}" if stats["avg_ttft_ms"] else "—"),
        ("Avg TPOT (ms)", f"{stats['avg_tpot_ms']:.1f}" if stats["avg_tpot_ms"] else "—"),
        ("KV blocks used", str(stats["kv_cache_blocks_used"])),
        ("KV blocks free", str(stats["kv_cache_blocks_free"])),
        ("KV utilization", f"{stats['kv_cache_utilization_pct']:.1f}%"),
        ("Total requests", str(stats["total_requests_served"])),
        ("Total tokens out", str(stats["total_tokens_out"])),
    ]
    for k, v in rows:
        table.add_row(k, v)

    console.print(table)


def verify_rope():
    console.print("\n[bold cyan]Verifying RoPE relative-position invariance...[/bold cyan]")
    RotaryEmbedding.test_relative_invariance(head_dim=64)
    console.print("  [green]✓ RoPE verification passed[/green]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-blocks", type=int, default=128)
    args = parser.parse_args()

    verify_rope()

    model_cfg = ModelConfig(model_id=args.model)
    engine_cfg = EngineConfig(
        num_blocks=args.num_blocks,
        block_size=16,
        device=args.device,
        max_running_requests=4,
    )

    console.print(f"\n[bold]Loading engine: {args.model}[/bold]")
    t0 = time.monotonic()
    engine = LLMEngine.from_config(model_cfg, engine_cfg)
    engine.start()
    load_time = time.monotonic() - t0
    console.print(f"  Model loaded in [green]{load_time:.1f}s[/green]")

    try:
        run_tests(engine, max_tokens=args.max_tokens)
    finally:
        engine.stop()
        console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
