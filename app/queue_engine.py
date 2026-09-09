from dataclasses import dataclass, field
from itertools import count
from typing import Optional


@dataclass(frozen=True)
class Transaction:
    transaction_id: str
    amount: float
    risk_score: float
    payload: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Transaction(id={self.transaction_id}, amount={self.amount}, risk={self.risk_score:.3f})"


class _HeapNode:
    __slots__ = ("risk_score", "sequence_id", "transaction")

    def __init__(self, risk_score: float, sequence_id: int, transaction: Transaction):
        self.risk_score = risk_score
        self.sequence_id = sequence_id
        self.transaction = transaction

    def __lt__(self, other: "_HeapNode") -> bool:
        if self.risk_score != other.risk_score:
            return self.risk_score > other.risk_score
        return self.sequence_id < other.sequence_id


class TransactionPriorityQueue:
    def __init__(self) -> None:
        self._heap: list[_HeapNode] = []
        self._sequence_generator = count()

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def is_empty(self) -> bool:
        return not self._heap

    def insert(self, transaction: Transaction) -> None:
        node = _HeapNode(transaction.risk_score, next(self._sequence_generator), transaction)
        self._heap.append(node)
        self._sift_up(len(self._heap) - 1)

    def extract_max(self) -> Optional[Transaction]:
        if not self._heap:
            return None

        highest_priority = self._heap[0]
        tail_node = self._heap.pop()
        
        if self._heap:
            self._heap[0] = tail_node
            self._sift_down(0)
            
        return highest_priority.transaction

    def peek(self) -> Optional[Transaction]:
        return self._heap[0].transaction if self._heap else None

    def _sift_up(self, index: int) -> None:
        while index > 0:
            parent = (index - 1) // 2
            if self._heap[index] < self._heap[parent]:
                self._heap[index], self._heap[parent] = self._heap[parent], self._heap[index]
                index = parent
            else:
                break

    def _sift_down(self, index: int) -> None:
        limit = len(self._heap)
        while True:
            left = (2 * index) + 1
            right = (2 * index) + 2
            highest_priority_idx = index

            if left < limit and self._heap[left] < self._heap[highest_priority_idx]:
                highest_priority_idx = left
            if right < limit and self._heap[right] < self._heap[highest_priority_idx]:
                highest_priority_idx = right

            if highest_priority_idx == index:
                break

            self._heap[index], self._heap[highest_priority_idx] = (
                self._heap[highest_priority_idx],
                self._heap[index],
            )
            index = highest_priority_idx
