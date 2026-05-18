"""
Download a model from HuggingFace Hub to local cache.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model ID to download",
    )
    args = parser.parse_args()

    print(f"Downloading {args.model} ...")
    path = snapshot_download(
        repo_id=args.model,
        allow_patterns=["*.safetensors", "config.json", "tokenizer*", "special_tokens_map.json"],
        ignore_patterns=["*.msgpack", "*.h5", "flax_*", "tf_*"],
    )
    print(f"Downloaded to: {path}")


if __name__ == "__main__":
    main()
