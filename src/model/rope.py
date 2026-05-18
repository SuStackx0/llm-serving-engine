"""
Rotary Positional Embeddings (RoPE) — implemented from scratch.

Key insight: instead of adding position info (like sinusoidal embeddings),
we *rotate* query/key vectors in 2D planes. The rotation angle for dimension
pair (2i, 2i+1) at position m is  m * θᵢ  where  θᵢ = base^(-2i/d).

Property preserved: dot(RoPE(q,m), RoPE(k,n)) depends only on (m-n), not
on absolute positions — perfect for causal attention.
"""

import math
import torch
from torch import Tensor


class RotaryEmbedding:
    def __init__(self, head_dim: int, base: float = 10000.0,
                 max_seq_len: int = 4096, device: str = "cpu"):
        self.head_dim = head_dim
        self.base = base

        # θᵢ = base^(-2i / d)  for i = 0, 1, …, d/2 - 1
        half = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        # inv_freq: [half]

        # Precompute for all positions up to max_seq_len
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        # freqs[m, i] = m * θᵢ
        freqs = torch.outer(positions, inv_freq)          # [max_seq, half]
        emb = torch.cat([freqs, freqs], dim=-1)           # [max_seq, d]  (duplicated for rotate_half trick)

        self.cos_cache = emb.cos().to(device)             # [max_seq, d]
        self.sin_cache = emb.sin().to(device)             # [max_seq, d]

    def to(self, device: str) -> "RotaryEmbedding":
        self.cos_cache = self.cos_cache.to(device)
        self.sin_cache = self.sin_cache.to(device)
        return self

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        """Rotate each vector by splitting it in half and negating the first half.

        For x = [x₀, x₁, x₂, x₃, …, x_{d-2}, x_{d-1}]:
        rotate_half(x) = [-x_{d/2}, …, -x_{d-1}, x₀, …, x_{d/2-1}]

        Combined with the cos/sin split, this implements the 2D rotation
        (x₀, x₁) → (x₀ cos − x₁ sin, x₀ sin + x₁ cos) across all pairs.
        """
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """Apply RoPE to x.

        Args:
            x:         [total_tokens, num_heads, head_dim]
            positions: [total_tokens]  (integer positions in the sequence)

        Returns:
            x_rotated of the same shape.
        """
        cos = self.cos_cache[positions]        # [total_tokens, head_dim]
        sin = self.sin_cache[positions]        # [total_tokens, head_dim]

        # Broadcast over head dimension: [T, 1, D]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        return x * cos + self._rotate_half(x) * sin

    # ------------------------------------------------------------------
    # Unit test helper — verifies relative-position invariance
    # ------------------------------------------------------------------
    @staticmethod
    def test_relative_invariance(head_dim: int = 64, device: str = "cpu"):
        """
        For any two positions (i, j) the dot product
            RoPE(q, i) · RoPE(k, j)
        should equal
            RoPE(q, i+Δ) · RoPE(k, j+Δ)
        for any shift Δ. We verify this for a random q, k.
        """
        rope = RotaryEmbedding(head_dim=head_dim, device=device)
        q = torch.randn(1, 1, head_dim)
        k = torch.randn(1, 1, head_dim)

        i, j, delta = 5, 12, 7
        q_i  = rope.forward(q, torch.tensor([i]))
        k_j  = rope.forward(k, torch.tensor([j]))
        q_id = rope.forward(q, torch.tensor([i + delta]))
        k_jd = rope.forward(k, torch.tensor([j + delta]))

        dot1 = (q_i  * k_j ).sum().item()
        dot2 = (q_id * k_jd).sum().item()

        assert abs(dot1 - dot2) < 1e-4, (
            f"Relative invariance violated: {dot1:.6f} vs {dot2:.6f}"
        )
        print(f"  ✓ RoPE relative invariance: dot1={dot1:.4f}  dot2={dot2:.4f}")
