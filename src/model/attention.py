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


def _chunked_prefill_attention(
    q: Tensor,           # [chunk_len, num_heads, head_dim]
    k_full: Tensor,      # [total_ctx, num_kv_heads, head_dim]  (prior + chunk K)
    v_full: Tensor,      # [total_ctx, num_kv_heads, head_dim]  (prior + chunk V)
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    chunk_start: int,    # how many prior tokens precede this chunk in k_full/v_full
) -> Tensor:
    """Causal attention for a chunk of queries over prior + current context.

    When chunk_start == 0 (first or only chunk), this is identical to standard
    full-sequence causal prefill attention.

    For later chunks, prior tokens (indices 0..chunk_start-1 in k_full) are
    always visible; causal masking only applies within the current chunk
    (indices chunk_start..chunk_start+chunk_len-1).
    """
    chunk_len = q.shape[0]
    total_ctx = k_full.shape[0]  # chunk_start + chunk_len
    gqa = num_heads // num_kv_heads

    k = k_full.repeat_interleave(gqa, dim=1)  # [total_ctx, H, D]
    v = v_full.repeat_interleave(gqa, dim=1)

    q_t = q.permute(1, 0, 2)   # [H, chunk_len, D]
    k_t = k.permute(1, 2, 0)   # [H, D, total_ctx]
    v_t = v.permute(1, 0, 2)   # [H, total_ctx, D]

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.bmm(q_t, k_t) * scale   # [H, chunk_len, total_ctx]

    if chunk_len > 1 or chunk_start > 0:
        # Build [chunk_len, total_ctx] causal mask.
        # Query at position (chunk_start + qi) must not attend to keys at
        # positions > (chunk_start + qi).
        # Prior tokens (j < chunk_start) are always fully visible (mask = 0).
        mask = torch.zeros(chunk_len, total_ctx, device=q.device, dtype=scores.dtype)
        for qi in range(chunk_len):
            future_start = chunk_start + qi + 1
            if future_start < total_ctx:
                mask[qi, future_start:] = float("-inf")
        scores = scores + mask.unsqueeze(0)

    attn = torch.softmax(scores, dim=-1)
    out = torch.bmm(attn, v_t)          # [H, chunk_len, D]
    return out.permute(1, 0, 2)         # [chunk_len, H, D]


def _prefill_attention(
    q: Tensor,           # [seq_len, num_heads, head_dim]
    k_full: Tensor,      # [seq_len, num_kv_heads, head_dim]
    v_full: Tensor,      # [seq_len, num_kv_heads, head_dim]
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Tensor:
    """Full-sequence causal attention (chunk_start=0 fast path)."""
    return _chunked_prefill_attention(
        q, k_full, v_full, num_heads, num_kv_heads, head_dim, chunk_start=0
    )


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
            q_i = q[tok_offset: tok_offset + seq_len]   # [chunk_len, nh, hd]
            k_i = k[tok_offset: tok_offset + seq_len]   # [chunk_len, nkv, hd]
            v_i = v[tok_offset: tok_offset + seq_len]

            # For chunked prefill: where in the block table does this chunk start?
            chunk_start = 0
            if metadata.prefill_chunk_starts is not None:
                chunk_start = metadata.prefill_chunk_starts[i]

            # Store this chunk's K/V at the correct slot offset
            kv_cache.store_tokens(
                layer_idx=self.layer_idx,
                block_table=metadata.prefill_block_tables[i],
                keys=k_i,
                values=v_i,
                start_slot=chunk_start,
            )

            # For mid-stream chunks: gather previously stored K/V and prepend
            if chunk_start > 0:
                k_prior, v_prior = kv_cache.gather_tokens(
                    layer_idx=self.layer_idx,
                    block_table=metadata.prefill_block_tables[i],
                    num_tokens=chunk_start,
                )
                # Cast to match the current chunk's dtype for cat
                k_full = torch.cat([k_prior.to(k_i.dtype), k_i], dim=0)
                v_full = torch.cat([v_prior.to(v_i.dtype), v_i], dim=0)
            else:
                k_full = k_i
                v_full = v_i

            out_i = _chunked_prefill_attention(q_i, k_full, v_full, nh, nkv, hd, chunk_start)
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
