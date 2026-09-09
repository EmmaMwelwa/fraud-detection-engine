import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger("fraud_engine.model")

FEATURE_COLUMNS = [
    "amount",
    "hour_of_day",
    "merchant_category",
    "distance_from_home_km",
    "is_foreign_transaction",
    "velocity_last_hour",
]

MERCHANT_CATEGORIES = {
    "grocery": 0,
    "electronics": 1,
    "travel": 2,
    "restaurant": 3,
    "gambling": 4,
    "other": 5,
}


@dataclass(frozen=True)
class TransactionFeatures:
    amount: float
    hour_of_day: int
    merchant_category: str
    distance_from_home_km: float
    is_foreign_transaction: bool
    velocity_last_hour: int

    def to_vector(self) -> np.ndarray:
        category_code = MERCHANT_CATEGORIES.get(self.merchant_category, MERCHANT_CATEGORIES["other"])
        return np.array([[
            self.amount,
            self.hour_of_day,
            category_code,
            self.distance_from_home_km,
            int(self.is_foreign_transaction),
            self.velocity_last_hour,
        ]], dtype=float)


class FraudRiskModel:
    def __init__(self, contamination: float = 0.03, random_state: int = 42):
        self._model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
        )
        self._score_min: Optional[float] = None
        self._score_max: Optional[float] = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "FraudRiskModel":
        X = df[FEATURE_COLUMNS].values
        self._model.fit(X)

        # Invert scores so that higher scores represent anomalies
        anomaly_scores = -self._model.score_samples(X)
        self._score_min = float(anomaly_scores.min())
        self._score_max = float(anomaly_scores.max())
        self._fitted = True
        return self

    def predict_risk(self, features: TransactionFeatures) -> float:
        if not self._fitted:
            raise RuntimeError("Execution blocked: Pipeline must be trained or loaded before inference.")

        vector = features.to_vector()
        raw_score = -self._model.score_samples(vector)[0]

        span = self._score_max - self._score_min
        if span == 0:
            return 0.0

        normalized = (raw_score - self._score_min) / span
        return float(np.clip(normalized, 0.0, 1.0))

    def save(self, path: Union[str, Path]) -> None:
        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        payload = {
            "model_state": self._model,
            "normalization_bounds": (self._score_min, self._score_max),
        }
        joblib.dump(payload, export_path)
        logger.info(f"Model artifacts exported to {export_path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FraudRiskModel":
        import_path = Path(path)
        if not import_path.exists():
            raise FileNotFoundError(f"No usable model payload found at {import_path}")
            
        data = joblib.load(import_path)
        instance = cls()
        instance._model = data["model_state"]
        instance._score_min, instance._score_max = data["normalization_bounds"]
        instance._fitted = True
        return instance
