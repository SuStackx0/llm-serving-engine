"""
Prefix Trie Cache — radix-tree of token blocks for KV cache reuse.

Each node represents one complete block's worth of tokens (block_size tokens).
When two requests share a token prefix that is aligned to block boundaries,
the second request reuses the already-computed KV blocks and skips prefill
for those tokens — cutting TTFT dramatically for repeated system prompts
and few-shot examples.

Only complete blocks are cached. A partial trailing block is never inserted.

Thread safety: all mutations happen on the engine background thread.
stats() is called from the API thread — it only reads counters (safe enough
for monitoring; worst case is a slightly stale number).
"""

import threading
import time
from typing import Dict, List, Optional, Tuple


class PrefixTrieNode:
    __slots__ = ("children", "block_id", "ref_count", "last_used")

    def __init__(self):
        self.children: Dict[tuple, "PrefixTrieNode"] = {}
        self.block_id: Optional[int] = None
        self.ref_count: int = 0
        self.last_used: float = time.monotonic()


class PrefixTrieCache:
    """
    Trie-based KV block cache keyed on token-block tuples.

    Usage:
        cache = PrefixTrieCache(block_size=16)

        # On request admission:
        match_len, block_ids = cache.match(token_ids)
        cache.pin(block_ids)

        # After prefill complete:
        cache.insert(token_ids, new_block_ids)
        cache.unpin(borrowed_block_ids)

        # When memory is tight:
        freed_ids = cache.evict_lru(n)
        block_manager.unmark_cached(freed_ids)
    """

    def __init__(self, block_size: int):
        self.block_size = block_size
        self._root = PrefixTrieNode()
        self._hit_count = 0
        self._miss_count = 0
        # block_id → node, for O(1) ref-count and eviction lookups
        self._block_to_node: Dict[int, PrefixTrieNode] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Match
    # ------------------------------------------------------------------

    def match(self, token_ids: List[int]) -> Tuple[int, List[int]]:
        """Walk the trie as far as possible matching complete blocks.

        Returns (num_matched_tokens, matched_block_ids).
        num_matched_tokens is always a multiple of block_size.
        """
        with self._lock:
            node = self._root
            matched_blocks: List[int] = []
            num_complete_blocks = len(token_ids) // self.block_size

            for i in range(num_complete_blocks):
                key = tuple(token_ids[i * self.block_size: (i + 1) * self.block_size])
                child = node.children.get(key)
                if child is None:
                    break
                matched_blocks.append(child.block_id)
                child.last_used = time.monotonic()
                node = child

            if matched_blocks:
                self._hit_count += 1
            else:
                self._miss_count += 1

            return len(matched_blocks) * self.block_size, matched_blocks

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, token_ids: List[int], block_ids: List[int]) -> None:
        """Insert newly computed blocks into the trie.

        Only inserts complete blocks. If a node already exists (another request
        had the same prefix), traverses through it without overwriting.
        """
        with self._lock:
            node = self._root
            num_complete_blocks = len(token_ids) // self.block_size
            blocks_to_insert = min(len(block_ids), num_complete_blocks)

            for i in range(blocks_to_insert):
                key = tuple(token_ids[i * self.block_size: (i + 1) * self.block_size])
                if key in node.children:
                    node = node.children[key]
                else:
                    child = PrefixTrieNode()
                    child.block_id = block_ids[i]
                    child.last_used = time.monotonic()
                    node.children[key] = child
                    self._block_to_node[block_ids[i]] = child
                    node = child

    # ------------------------------------------------------------------
    # Pin / unpin (ref counting)
    # ------------------------------------------------------------------

    def pin(self, block_ids: List[int]) -> None:
        """Prevent eviction of blocks currently in use by a live request."""
        with self._lock:
            for bid in block_ids:
                node = self._block_to_node.get(bid)
                if node is not None:
                    node.ref_count += 1

    def unpin(self, block_ids: List[int]) -> None:
        """Allow eviction once a request finishes using cached blocks."""
        with self._lock:
            for bid in block_ids:
                node = self._block_to_node.get(bid)
                if node is not None:
                    node.ref_count = max(0, node.ref_count - 1)

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def evict_lru(self, n_blocks: int) -> List[int]:
        """Evict up to n_blocks least-recently-used leaf nodes with ref_count==0.

        Leaf-first eviction ensures parent nodes remain valid for shorter prefixes.
        Returns list of freed block_ids (caller must unmark them in block_manager).
        """
        with self._lock:
            candidates: List[PrefixTrieNode] = []
            self._collect_evictable_leaves(self._root, candidates)
            candidates.sort(key=lambda n: n.last_used)

            freed: List[int] = []
            for node in candidates[:n_blocks]:
                if node.block_id is not None and node.ref_count == 0:
                    freed.append(node.block_id)
                    self._remove_node(node)

            return freed

    def _collect_evictable_leaves(
        self, node: PrefixTrieNode, out: List[PrefixTrieNode]
    ) -> None:
        """Post-order traversal — only collect leaf nodes with ref_count==0."""
        for child in list(node.children.values()):
            self._collect_evictable_leaves(child, out)
        if not node.children and node.block_id is not None and node.ref_count == 0:
            out.append(node)

    def _remove_node(self, target: PrefixTrieNode) -> None:
        """Remove a leaf node from trie and block_to_node map."""
        if target.block_id is not None:
            self._block_to_node.pop(target.block_id, None)
        self._remove_from_parent(self._root, target)

    def _remove_from_parent(
        self, current: PrefixTrieNode, target: PrefixTrieNode
    ) -> bool:
        for key, child in list(current.children.items()):
            if child is target:
                del current.children[key]
                return True
            if self._remove_from_parent(child, target):
                return True
        return False

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def num_cached_blocks(self) -> int:
        return len(self._block_to_node)

    def stats(self) -> dict:
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total if total > 0 else 0.0
        return {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(hit_rate, 4),
            "cached_blocks": self.num_cached_blocks(),
        }
