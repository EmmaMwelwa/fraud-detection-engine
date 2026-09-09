import random
from app.queue_engine import Transaction, TransactionPriorityQueue


def make_txn(i: int, risk: float) -> Transaction:
    return Transaction(transaction_id=f"txn_{i}", amount=100.0 + i, risk_score=risk)


def test_empty_queue():
    q = TransactionPriorityQueue()
    assert q.is_empty
    assert q.peek() is None
    assert q.extract_max() is None


def test_single_insert_extract():
    q = TransactionPriorityQueue()
    t = make_txn(1, 0.9)
    q.insert(t)
    assert len(q) == 1
    assert q.peek() is t
    assert q.extract_max() is t
    assert q.is_empty


def test_extracts_in_descending_risk_order():
    q = TransactionPriorityQueue()
    risks = [0.1, 0.9, 0.5, 0.99, 0.3, 0.7, 0.0, 0.42]
    for i, r in enumerate(risks):
        q.insert(make_txn(i, r))
    extracted = [q.extract_max().risk_score for _ in range(len(risks))]
    assert extracted == sorted(risks, reverse=True)
    assert q.is_empty


def test_ties_are_fifo():
    q = TransactionPriorityQueue()
    a = make_txn(1, 0.5)
    b = make_txn(2, 0.5)
    c = make_txn(3, 0.5)
    q.insert(a)
    q.insert(b)
    q.insert(c)
    assert q.extract_max() is a
    assert q.extract_max() is b
    assert q.extract_max() is c


def test_large_random_stress():
    random.seed(42)
    q = TransactionPriorityQueue()
    n = 2000
    risks = [round(random.random(), 4) for _ in range(n)]
    for i, r in enumerate(risks):
        q.insert(make_txn(i, r))
    extracted = [q.extract_max().risk_score for _ in range(n)]
    assert extracted == sorted(risks, reverse=True)
