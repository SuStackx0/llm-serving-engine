"""
Model weight loading from HuggingFace Hub.
Downloads the model once, caches locally, returns:
  - weights dict (weight_name → Tensor)
  - model config (parsed config.json)
  - tokenizer (HuggingFace AutoTokenizer)
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from src.core.config import ModelConfig


def _find_safetensor_files(model_dir: str) -> list:
    files = []
    for f in sorted(os.listdir(model_dir)):
        if f.endswith(".safetensors"):
            files.append(os.path.join(model_dir, f))
    return files


def load_model(
    model_id: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[Dict[str, torch.Tensor], ModelConfig, AutoTokenizer]:
    """Download (if needed) and load a HuggingFace model.

    Returns:
        weights: dict of weight name → Tensor (on device, in dtype)
        config:  ModelConfig parsed from config.json
        tokenizer: HuggingFace tokenizer
    """
    print(f"Loading model: {model_id}")

    model_dir = snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.safetensors", "config.json", "tokenizer*", "special_tokens_map.json"],
        ignore_patterns=["*.msgpack", "*.h5", "flax_*", "tf_*"],
    )
    print(f"  Model files at: {model_dir}")

    # Parse config
    cfg_path = Path(model_dir) / "config.json"
    with open(cfg_path) as f:
        hf_cfg = json.load(f)
    model_cfg = ModelConfig.from_hf_config(hf_cfg, model_id=model_id)
    print(f"  Config: {model_cfg.num_hidden_layers} layers, "
          f"hidden={model_cfg.hidden_size}, "
          f"heads={model_cfg.num_attention_heads}/{model_cfg.num_key_value_heads} (Q/KV)")

    # Load weights
    sf_files = _find_safetensor_files(model_dir)
    if not sf_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    weights: Dict[str, torch.Tensor] = {}
    for path in sf_files:
        shard = load_file(path, device="cpu")
        weights.update(shard)

    # Move to target device and dtype
    weights = {
        name: tensor.to(dtype=dtype, device=device)
        for name, tensor in weights.items()
    }
    total_params = sum(t.numel() for t in weights.values())
    print(f"  Loaded {len(weights)} tensors, {total_params/1e9:.2f}B parameters")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return weights, model_cfg, tokenizer
