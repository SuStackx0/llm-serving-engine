"""Tests for the RoPE implementation."""

import math
import pytest
import torch
from src.model.rope import RotaryEmbedding


def test_output_shape():
    rope = RotaryEmbedding(head_dim=64)
    x = torch.randn(10, 8, 64)   # [tokens, heads, head_dim]
    positions = torch.arange(10)
    out = rope.forward(x, positions)
    assert out.shape == x.shape


def test_rotation_preserves_magnitude():
    """RoPE is a rotation — should preserve vector norm."""
    rope = RotaryEmbedding(head_dim=64)
    x = torch.randn(5, 1, 64)
    positions = torch.arange(5)
    out = rope.forward(x, positions)
    norms_in  = x.norm(dim=-1)
    norms_out = out.norm(dim=-1)
    assert torch.allclose(norms_in, norms_out, atol=1e-5), "RoPE changed vector norm"


def test_relative_position_invariance():
    """Core RoPE property: dot product depends only on (i - j), not absolute i, j."""
    rope = RotaryEmbedding(head_dim=64)
    q = torch.randn(1, 1, 64)
    k = torch.randn(1, 1, 64)

    for delta in [0, 3, 7]:
        i, j = 2, 8
        q_i  = rope.forward(q, torch.tensor([i]))
        k_j  = rope.forward(k, torch.tensor([j]))
        q_id = rope.forward(q, torch.tensor([i + delta]))
        k_jd = rope.forward(k, torch.tensor([j + delta]))

        dot1 = (q_i * k_j).sum().item()
        dot2 = (q_id * k_jd).sum().item()
        assert abs(dot1 - dot2) < 1e-4, (
            f"Invariance failed for delta={delta}: {dot1:.6f} vs {dot2:.6f}"
        )


def test_different_positions_give_different_rotations():
    rope = RotaryEmbedding(head_dim=64)
    x = torch.randn(1, 1, 64)
    out0 = rope.forward(x, torch.tensor([0]))
    out5 = rope.forward(x, torch.tensor([5]))
    assert not torch.allclose(out0, out5), "Same output for different positions"


def test_rotate_half_involution():
    """Applying rotate_half twice should negate (it's an involution up to sign)."""
    x = torch.randn(4, 64)
    r1 = RotaryEmbedding._rotate_half(x)
    r2 = RotaryEmbedding._rotate_half(r1)
    assert torch.allclose(r2, -x, atol=1e-6)
