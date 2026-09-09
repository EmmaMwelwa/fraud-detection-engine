import tempfile
from pathlib import Path
import pytest

from app.ml_model import FraudRiskModel, TransactionFeatures
from scripts.generate_data import generate_synthetic_dataset


@pytest.fixture(scope="module")
def trained_model():
    df = generate_synthetic_dataset(n_samples=1500, fraud_ratio=0.05, random_state=1)
    model = FraudRiskModel(contamination=0.05, random_state=1)
    model.fit(df)
    return model


def test_risk_score_is_bounded(trained_model):
    obviously_normal = TransactionFeatures(
        amount=35.0,
        hour_of_day=13,
        merchant_category="grocery",
        distance_from_home_km=2.0,
        is_foreign_transaction=False,
        velocity_last_hour=1,
    )
    score = trained_model.predict_risk(obviously_normal)
    assert 0.0 <= score <= 1.0


def test_fraudulent_pattern_scores_higher_than_normal_pattern(trained_model):
    normal = TransactionFeatures(
        amount=42.0,
        hour_of_day=13,
        merchant_category="grocery",
        distance_from_home_km=3.0,
        is_foreign_transaction=False,
        velocity_last_hour=1,
    )
    suspicious = TransactionFeatures(
        amount=1800.0,
        hour_of_day=3,
        merchant_category="travel",
        distance_from_home_km=4200.0,
        is_foreign_transaction=True,
        velocity_last_hour=9,
    )
    normal_score = trained_model.predict_risk(normal)
    suspicious_score = trained_model.predict_risk(suspicious)
    assert suspicious_score > normal_score


def test_predict_before_fit_raises():
    model = FraudRiskModel()
    txn = TransactionFeatures(
        amount=10,
        hour_of_day=10,
        merchant_category="grocery",
        distance_from_home_km=1,
        is_foreign_transaction=False,
        velocity_last_hour=0,
    )
    with pytest.raises(RuntimeError):
        model.predict_risk(txn)


def test_save_and_load_roundtrip(trained_model):
    txn = TransactionFeatures(
        amount=1800.0,
        hour_of_day=3,
        merchant_category="travel",
        distance_from_home_km=4200.0,
        is_foreign_transaction=True,
        velocity_last_hour=9,
    )
    original_score = trained_model.predict_risk(txn)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "model.joblib"
        trained_model.save(path)
        reloaded = FraudRiskModel.load(path)
        reloaded_score = reloaded.predict_risk(txn)
        
    assert original_score == reloaded_score

