"""
Priority request queue with FIFO tie-breaking.

Priority: lower integer = higher priority (same convention as Python's heapq).
Within the same priority level requests are served FIFO (by arrival_time).
"""

import heapq
from typing import List, Optional

from src.core.types import Request


class PriorityRequestQueue:
    def __init__(self):
        self._heap: List[tuple] = []   # (priority, arrival_time, request)
        self._counter = 0              # ensures stable ordering

    def push(self, request: Request) -> None:
        entry = (request.priority, request.arrival_time, self._counter, request)
        self._counter += 1
        heapq.heappush(self._heap, entry)

    def pop(self) -> Request:
        _, _, _, request = heapq.heappop(self._heap)
        return request

    def peek(self) -> Optional[Request]:
        if self._heap:
            return self._heap[0][3]
        return None

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def to_list(self) -> List[Request]:
        """Return all requests sorted by priority (does not consume them)."""
        return [entry[3] for entry in sorted(self._heap)]
