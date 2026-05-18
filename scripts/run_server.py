"""
Start the LLM Serving Engine HTTP server.

Usage:
    python scripts/run_server.py
    python scripts/run_server.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 8000
    python scripts/run_server.py --device cpu --num-blocks 64
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn

from src.api.app import create_app
from src.core.config import EngineConfig, ModelConfig, ServerConfig
from src.engine.inference_engine import LLMEngine


def main():
    parser = argparse.ArgumentParser(description="LLM Serving Engine Server")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-running", type=int, default=8)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    model_cfg = ModelConfig(model_id=args.model)
    engine_cfg = EngineConfig(
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        device=args.device,
        max_running_requests=args.max_running,
    )

    print(f"Starting LLM Serving Engine")
    print(f"  Model     : {args.model}")
    print(f"  Device    : {args.device}")
    print(f"  Blocks    : {args.num_blocks} × {args.block_size} tokens")
    print(f"  Max concurrent: {args.max_running}")

    engine = LLMEngine.from_config(model_cfg, engine_cfg)
    app = create_app(engine, log_level=args.log_level)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
