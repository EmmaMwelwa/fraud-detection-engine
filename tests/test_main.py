import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.queue_engine import TransactionPriorityQueue

NORMAL_TXN = {
    "transaction_id": "txn_normal",
    "amount": 45.0,
    "hour_of_day": 13,
    "merchant_category": "grocery",
    "distance_from_home_km": 3.0,
    "is_foreign_transaction": False,
    "velocity_last_hour": 1,
}

SUSPICIOUS_TXN = {
    "transaction_id": "txn_suspicious",
    "amount": 1900.0,
    "hour_of_day": 3,
    "merchant_category": "travel",
    "distance_from_home_km": 4300.0,
    "is_foreign_transaction": True,
    "velocity_last_hour": 8,
}


@pytest.fixture
def client():
    app.state.queue = TransactionPriorityQueue()
    if not app.state.model:
        pytest.fail("Model artifacts uninitialized. Run training lifecycle script first.")
    return TestClient(app)


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["queue_size"] == 0
    assert body["model_loaded"] is True


def test_submit_transaction_returns_computed_risk_score(client):
    resp = client.post("/transactions", json=NORMAL_TXN)
    assert resp.status_code == 201
    body = resp.json()
    assert body["transaction_id"] == "txn_normal"
    assert 0.0 <= body["risk_score"] <= 1.0


def test_suspicious_transaction_scores_higher_than_normal(client):
    normal_resp = client.post("/transactions", json=NORMAL_TXN).json()
    suspicious_resp = client.post("/transactions", json=SUSPICIOUS_TXN).json()
    assert suspicious_resp["risk_score"] > normal_resp["risk_score"]


def test_submit_rejects_unknown_merchant_category(client):
    bad_txn = dict(NORMAL_TXN, transaction_id="txn_bad", merchant_category="not_a_real_category")
    resp = client.post("/transactions", json=bad_txn)
    assert resp.status_code == 422


def test_submit_rejects_invalid_hour(client):
    bad_txn = dict(NORMAL_TXN, transaction_id="txn_bad_hour", hour_of_day=99)
    resp = client.post("/transactions", json=bad_txn)
    assert resp.status_code == 422


def test_process_next_returns_highest_risk_first(client):
    client.post("/transactions", json=NORMAL_TXN)
    client.post("/transactions", json=SUSPICIOUS_TXN)
    first = client.post("/process-next").json()
    second = client.post("/process-next").json()
    assert first["transaction_id"] == "txn_suspicious"
    assert second["transaction_id"] == "txn_normal"


def test_process_next_on_empty_queue_returns_404(client):
    resp = client.post("/process-next")
    assert resp.status_code == 404
