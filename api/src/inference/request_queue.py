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
        self.tmp_empty = {}
    
    def add_request(self, request) -> None:
        # TODO: consider fairness and complete time?
        req_id = request.request_id
        if req_id not in self.requests:
            self.requests[req_id] = ChunkSet(request)
            self.processed_chunks[req_id] = 0
            heapq.heappush(self._heap, (self.processed_chunks[req_id], req_id, self.requests[req_id]))
        else:
            self.requests[req_id].add_chunk(request)
            if req_id in self.tmp_empty:
                self.tmp_empty.pop(req_id)
                heapq.heappush(self._heap, (self.processed_chunks[req_id], req_id, self.requests[req_id]))
        
    def pop_request(self):
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, req_id, request = heapq.heappop(self._heap)
        rst = request.pop_chunk()
        if request.chunks:
            self.processed_chunks[req_id] += 1
            heapq.heappush(self._heap, (self.processed_chunks[req_id], req_id, self.requests[req_id]))
        elif rst.data_type == "final":
            self.requests.pop(req_id)
            self.processed_chunks.pop(req_id)
            self.tmp_empty[req_id] = False
            self.tmp_empty.pop(req_id)
        else:
            self.processed_chunks[req_id] += 1
            self.tmp_empty[req_id] = True
        return rst
    
    def empty(self) -> bool:
        return not self._heap
