"""
Full LlamaForCausalLM — our own forward pass using HuggingFace weights.

We do NOT use HuggingFace's model.forward().  Instead we:
  1. Load raw weight tensors.
  2. Implement every layer operation ourselves (RMSNorm, SwiGLU MLP,
     PagedAttention with RoPE, residual connections).
  3. Return logits for the final token(s).

This lets us inject our block-based KV cache (PagedAttention) into the
attention computation without monkey-patching.
"""

import torch
from torch import Tensor
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.kv_cache import KVCacheManager

from src.core.config import ModelConfig
from src.core.types import AttentionMetadata
from src.model.layers import rms_norm, swiglu_mlp
from src.model.rope import RotaryEmbedding
from src.model.attention import PagedAttentionLayer


class LlamaForCausalLM:
    """Custom Llama transformer that uses our paged KV cache."""

    def __init__(
        self,
        config: ModelConfig,
        weights: Dict[str, Tensor],
        device: str,
        dtype: torch.dtype,
    ):
        self.config = config
        self.W = weights
        self.device = device
        self.dtype = dtype

        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            base=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            device=device,
        )

        self.attn_layers: List[PagedAttentionLayer] = [
            PagedAttentionLayer(
                layer_idx=i,
                weights=weights,
                rope=self.rope,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                hidden_size=config.hidden_size,
            )
            for i in range(config.num_hidden_layers)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, token_ids: Tensor) -> Tensor:
        """Token embedding lookup. [total_tokens] → [total_tokens, hidden]."""
        return self.W["model.embed_tokens.weight"][token_ids]

    def _mlp(self, x: Tensor, layer_idx: int) -> Tensor:
        prefix = f"model.layers.{layer_idx}.mlp"
        return swiglu_mlp(
            x,
            gate_proj=self.W[f"{prefix}.gate_proj.weight"],
            up_proj=self.W[f"{prefix}.up_proj.weight"],
            down_proj=self.W[f"{prefix}.down_proj.weight"],
        )

    def _layer_norm(self, x: Tensor, key: str) -> Tensor:
        return rms_norm(x, self.W[key], eps=self.config.rms_norm_eps)

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: Tensor,               # [total_tokens]  (flattened batch)
        positions: Tensor,               # [total_tokens]  (absolute positions)
        kv_cache: "KVCacheManager",
        metadata: AttentionMetadata,
    ) -> Tensor:
        """
        Run a full forward pass for a mixed prefill+decode batch.

        The batch is represented as a flat sequence of tokens:
          [prompt_req0_tok0, …, prompt_req0_tokN, last_tok_req1, last_tok_req2, …]

        metadata tells each layer how many tokens belong to each request and
        what block table to use.

        Returns:
            logits [total_tokens, vocab_size]
        """
        x = self._embed(token_ids)   # [T, hidden]

        for i in range(self.config.num_hidden_layers):
            residual = x

            # Pre-attention RMSNorm
            x = self._layer_norm(x, f"model.layers.{i}.input_layernorm.weight")

            # PagedAttention (stores / reads KV from block cache)
            x = self.attn_layers[i].forward(x, positions, kv_cache, metadata)

            # Residual
            x = x + residual
            residual = x

            # Post-attention RMSNorm
            x = self._layer_norm(x, f"model.layers.{i}.post_attention_layernorm.weight")

            # SwiGLU MLP
            x = self._mlp(x, i)

            # Residual
            x = x + residual

        # Final norm
        x = self._layer_norm(x, "model.norm.weight")

        # LM head: [T, hidden] → [T, vocab]
        return x @ self.W["lm_head.weight"].T
