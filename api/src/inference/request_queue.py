from __future__ import annotations
import time
import heapq
from typing import Any


class ChunkSet:
    def __init__(self, request) -> None:
        self.chunks = [request]
        self.request_id = request.request_id
        self.arrival_time = time.time()
    
    def add_chunk(self, chunk) -> None:
        self.chunks.append(chunk)

    def pop_chunk(self):
        return self.chunks.pop(0)
    
    def get_arrival_time(self) -> float:
        return self.chunks[0].arrival_time


class FairRequestQueue:
    """
    A fair queue that supports deque operations.
    """
    def __init__(self) -> None:
        self.requests = {}
        self.processed_chunks = {}
        self._heap: list[tuple[float, float, Any]] = []
    
    def add_request(self, request) -> None:
        # TODO: consider fairness and complete time?
        if request.request_id not in self.requests:
            self.requests[request.request_id] = ChunkSet(request)
            self.processed_chunks[request.request_id] = 0
            heapq.heappush(self._heap, (self.processed_chunks[request.request_id], request.request_id, self.requests[request.request_id]))
        else:
            self.requests[request.request_id].add_chunk(request)
        
    def pop_request(self):
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, _, request = heapq.heappop(self._heap)
        rst = request.pop_chunk()
        if request.chunks:
            self.processed_chunks[request.request_id] += 1
            heapq.heappush(self._heap, (self.processed_chunks[request.request_id], request.request_id, self.requests[request.request_id]))
        else:
            self.requests.pop(request.request_id)
            self.processed_chunks.pop(request.request_id)
        return rst
    
    def empty(self) -> bool:
        return not self._heap
