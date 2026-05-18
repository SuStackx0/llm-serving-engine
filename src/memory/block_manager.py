"""
Physical Block Manager — the vLLM memory innovation.

GPU (or CPU/MPS) memory for KV cache is divided into fixed-size blocks.
Each block holds `block_size` tokens worth of KV data for ALL layers.

Instead of allocating a contiguous tensor per request (which fragments
memory and causes OOM even when free memory exists), we:
  1. Pre-allocate one big pool of N blocks at startup.
  2. Track a free-list.
  3. Hand out blocks from the free-list as requests grow.
  4. Return blocks to the free-list when a request finishes.

This eliminates fragmentation: every block is the same size, so any free
block fits any request.
"""

from typing import Dict, List, Optional


class PhysicalBlockManager:
    """Tracks free/allocated physical KV cache blocks."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size      # tokens per block

        # All blocks start free; we use a list so pop() is O(1)
        self._free: List[int] = list(range(num_blocks))

        # Maps request_id → list of physical block IDs it holds
        self._owned: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def num_required_blocks(self, num_tokens: int) -> int:
        """Blocks needed to store num_tokens tokens."""
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_blocks_needed: int) -> bool:
        return len(self._free) >= num_blocks_needed

    def allocate(self, request_id: str, num_blocks: int) -> List[int]:
        """Allocate num_blocks blocks for a request. Returns block ids."""
        if not self.can_allocate(num_blocks):
            raise RuntimeError(
                f"Cannot allocate {num_blocks} blocks: only {len(self._free)} free"
            )
        blocks = [self._free.pop() for _ in range(num_blocks)]
        self._owned.setdefault(request_id, []).extend(blocks)
        return blocks

    def allocate_one(self, request_id: str) -> int:
        """Allocate a single additional block for an existing request."""
        blocks = self.allocate(request_id, 1)
        return blocks[0]

    # ------------------------------------------------------------------
    # Deallocation
    # ------------------------------------------------------------------

    def free(self, request_id: str) -> int:
        """Free all blocks belonging to request_id. Returns count freed."""
        blocks = self._owned.pop(request_id, [])
        self._free.extend(blocks)
        return len(blocks)

    def free_blocks(self, block_ids: List[int], request_id: Optional[str] = None) -> None:
        """Free a specific list of block ids."""
        for bid in block_ids:
            self._free.append(bid)
        if request_id and request_id in self._owned:
            owned = self._owned[request_id]
            for bid in block_ids:
                if bid in owned:
                    owned.remove(bid)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def num_free_blocks(self) -> int:
        return len(self._free)

    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    def get_block_table(self, request_id: str) -> List[int]:
        return self._owned.get(request_id, [])

    def utilization(self) -> float:
        return self.num_used_blocks() / self.num_blocks

    def __repr__(self) -> str:
        return (
            f"PhysicalBlockManager("
            f"total={self.num_blocks}, "
            f"free={self.num_free_blocks()}, "
            f"used={self.num_used_blocks()}, "
            f"block_size={self.block_size})"
        )
