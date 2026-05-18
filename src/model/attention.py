"""
PagedAttention — the core innovation of vLLM.

Key ideas implemented here:
1. KV pairs are stored in fixed-size physical blocks (managed by KVCacheManager).
2. Prefill: attend over the full prompt with a causal mask; store all K/V into blocks.
3. Decode: for each request gather its K/V from scattered blocks, compute attention
   over the full context (one new query token per request).

On real CUDA hardware vLLM uses a custom CUDA kernel so it never materializes the
full gathered K/V tensor — it computes attention block-by-block.  On MPS/CPU we
gather first then call standard attention; the scheduling / memory management layer
is identical.
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.kv_cache import KVCacheManager
from src.core.types import AttentionMetadata
from src.model.rope import RotaryEmbedding


def _causal_mask(seq_len: int, device) -> Tensor:
    """Upper-triangular -inf mask [seq_len, seq_len]."""
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


def _prefill_attention(
    q: Tensor,           # [seq_len, num_heads, head_dim]
    k_full: Tensor,      # [seq_len, num_kv_heads, head_dim]
    v_full: Tensor,      # [seq_len, num_kv_heads, head_dim]
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Tensor:
    """Standard causal multi-head attention for a single prefill sequence."""
    seq_len = q.shape[0]
    gqa = num_heads // num_kv_heads

    # GQA: expand KV heads
    k = k_full.repeat_interleave(gqa, dim=1)  # [seq, num_heads, head_dim]
    v = v_full.repeat_interleave(gqa, dim=1)

    # [num_heads, seq, head_dim] for batched matmul
    q_t = q.permute(1, 0, 2)   # [H, S, D]
    k_t = k.permute(1, 2, 0)   # [H, D, S]
    v_t = v.permute(1, 0, 2)   # [H, S, D]

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.bmm(q_t, k_t) * scale       # [H, S, S]

    mask = _causal_mask(seq_len, q.device)
    scores = scores + mask.unsqueeze(0)         # broadcast over heads

    attn = torch.softmax(scores, dim=-1)        # [H, S, S]
    out = torch.bmm(attn, v_t)                  # [H, S, D]
    return out.permute(1, 0, 2)                 # [S, H, D]


def _decode_attention_single(
    q: Tensor,        # [1, num_heads, head_dim]  — one query token
    k: Tensor,        # [ctx_len, num_kv_heads, head_dim]
    v: Tensor,        # [ctx_len, num_kv_heads, head_dim]
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Tensor:
    """Attention for a single decode request over its full KV context."""
    gqa = num_heads // num_kv_heads

    k = k.repeat_interleave(gqa, dim=1)   # [ctx, H, D]
    v = v.repeat_interleave(gqa, dim=1)

    q_t = q.permute(1, 0, 2)   # [H, 1, D]
    k_t = k.permute(1, 2, 0)   # [H, D, ctx]
    v_t = v.permute(1, 0, 2)   # [H, ctx, D]

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.bmm(q_t, k_t) * scale   # [H, 1, ctx]
    attn = torch.softmax(scores, dim=-1)
    out = torch.bmm(attn, v_t)              # [H, 1, D]
    return out.permute(1, 0, 2)             # [1, H, D]


class PagedAttentionLayer:
    """
    One transformer attention layer using paged KV cache.

    It does NOT inherit nn.Module because we operate on raw weight tensors
    loaded from HuggingFace checkpoints.
    """

    def __init__(
        self,
        layer_idx: int,
        weights: dict,
        rope: RotaryEmbedding,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
    ):
        self.layer_idx = layer_idx
        self.W = weights
        self.rope = rope
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size

        prefix = f"model.layers.{layer_idx}.self_attn"
        self.Wq = weights[f"{prefix}.q_proj.weight"]   # [nh*hd, hidden]
        self.Wk = weights[f"{prefix}.k_proj.weight"]   # [nkv*hd, hidden]
        self.Wv = weights[f"{prefix}.v_proj.weight"]
        self.Wo = weights[f"{prefix}.o_proj.weight"]   # [hidden, nh*hd]

    def forward(
        self,
        x: Tensor,                      # [total_tokens, hidden_size]
        positions: Tensor,              # [total_tokens]
        kv_cache: "KVCacheManager",
        metadata: AttentionMetadata,
    ) -> Tensor:
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim

        # Linear projections
        q = (x @ self.Wq.T).view(-1, nh, hd)    # [T, nh, hd]
        k = (x @ self.Wk.T).view(-1, nkv, hd)   # [T, nkv, hd]
        v = (x @ self.Wv.T).view(-1, nkv, hd)   # [T, nkv, hd]

        # Apply RoPE to Q and K
        q = self.rope.forward(q, positions)
        k = self.rope.forward(k, positions)

        outputs: List[Tensor] = []
        tok_offset = 0

        # ── Prefill path ──────────────────────────────────────────────
        for i, seq_len in enumerate(metadata.prefill_seq_lens):
            q_i = q[tok_offset: tok_offset + seq_len]   # [S, nh, hd]
            k_i = k[tok_offset: tok_offset + seq_len]   # [S, nkv, hd]
            v_i = v[tok_offset: tok_offset + seq_len]

            # Store all prompt tokens into the KV cache blocks
            kv_cache.store_tokens(
                layer_idx=self.layer_idx,
                block_table=metadata.prefill_block_tables[i],
                keys=k_i,
                values=v_i,
                start_slot=0,           # prompt always starts at slot 0
            )

            out_i = _prefill_attention(q_i, k_i, v_i, nh, nkv, hd)
            outputs.append(out_i)
            tok_offset += seq_len

        # ── Decode path ───────────────────────────────────────────────
        for i, ctx_len in enumerate(metadata.decode_context_lens):
            q_i = q[tok_offset: tok_offset + 1]   # [1, nh, hd]
            k_i = k[tok_offset: tok_offset + 1]   # [1, nkv, hd]
            v_i = v[tok_offset: tok_offset + 1]

            # Store the single new token's K/V at slot (ctx_len - 1)
            kv_cache.store_tokens(
                layer_idx=self.layer_idx,
                block_table=metadata.decode_block_tables[i],
                keys=k_i,
                values=v_i,
                start_slot=ctx_len - 1,
            )

            # Gather entire context K/V from blocks for this request
            k_ctx, v_ctx = kv_cache.gather_tokens(
                layer_idx=self.layer_idx,
                block_table=metadata.decode_block_tables[i],
                num_tokens=ctx_len,
            )   # [ctx_len, nkv, hd]

            out_i = _decode_attention_single(q_i, k_ctx, v_ctx, nh, nkv, hd)
            outputs.append(out_i)
            tok_offset += 1

        # Concatenate all outputs: [total_tokens, num_heads, head_dim]
        combined = torch.cat(outputs, dim=0)

        # Merge heads and apply output projection
        combined = combined.reshape(-1, nh * hd)
        return combined @ self.Wo.T     # [total_tokens, hidden_size]
