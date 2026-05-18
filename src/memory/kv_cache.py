"""
KV Cache Manager — pre-allocated tensor storage for key/value pairs.

Layout:
    kv_storage[layer, kv, block, slot, head, head_dim]
    kv  : 0 = key, 1 = value
    block: physical block id (0 … num_blocks-1)
    slot : position within block (0 … block_size-1)

Access pattern:
    - store_tokens: write K/V for a range of new tokens into blocks.
    - gather_tokens: read back all K/V for a request from its block table.

The block table (list of physical block ids) translates:
    logical_slot = block_table[block_idx] * block_size + slot_in_block
"""

from typing import List, Tuple

import torch
from torch import Tensor


class KVCacheManager:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: str,
        dtype: torch.dtype,
    ):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Pre-allocate the full KV cache
        # Shape: [num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]
        self.storage = torch.zeros(
            num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim,
            device=device, dtype=dtype,
        )

        mem_mb = self.storage.numel() * self.storage.element_size() / 1e6
        print(f"  KV cache allocated: {mem_mb:.1f} MB "
              f"({num_blocks} blocks × {block_size} tokens × {num_layers} layers)")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store_tokens(
        self,
        layer_idx: int,
        block_table: List[int],
        keys: Tensor,       # [num_tokens, num_kv_heads, head_dim]
        values: Tensor,     # [num_tokens, num_kv_heads, head_dim]
        start_slot: int,    # absolute slot index where writing begins
    ) -> None:
        """Write keys/values starting at start_slot, following block_table."""
        num_tokens = keys.shape[0]
        for i in range(num_tokens):
            slot = start_slot + i
            block_idx = slot // self.block_size
            slot_in_block = slot % self.block_size

            if block_idx >= len(block_table):
                # Block table too short — caller should have pre-allocated
                raise IndexError(
                    f"slot {slot} maps to block index {block_idx} but "
                    f"block_table only has {len(block_table)} entries"
                )

            phys_block = block_table[block_idx]
            self.storage[layer_idx, 0, phys_block, slot_in_block] = keys[i].to(self.dtype)
            self.storage[layer_idx, 1, phys_block, slot_in_block] = values[i].to(self.dtype)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def gather_tokens(
        self,
        layer_idx: int,
        block_table: List[int],
        num_tokens: int,
    ) -> Tuple[Tensor, Tensor]:
        """Gather num_tokens K/V from the block table.

        Returns:
            keys:   [num_tokens, num_kv_heads, head_dim]
            values: [num_tokens, num_kv_heads, head_dim]
        """
        k_slices: List[Tensor] = []
        v_slices: List[Tensor] = []

        remaining = num_tokens
        block_idx = 0

        while remaining > 0 and block_idx < len(block_table):
            phys = block_table[block_idx]
            take = min(remaining, self.block_size)
            k_slices.append(
                self.storage[layer_idx, 0, phys, :take].float()
            )
            v_slices.append(
                self.storage[layer_idx, 1, phys, :take].float()
            )
            remaining -= take
            block_idx += 1

        keys = torch.cat(k_slices, dim=0)      # [num_tokens, kv_heads, head_dim]
        values = torch.cat(v_slices, dim=0)
        return keys, values

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def memory_mb(self) -> float:
        return self.storage.numel() * self.storage.element_size() / 1e6
