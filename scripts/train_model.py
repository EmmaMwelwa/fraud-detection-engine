import logging
from pathlib import Path
import numpy as np

from app.ml_model import (
    FraudRiskModel,
    TransactionFeatures,
    MERCHANT_CATEGORIES,
)
from scripts.generate_data import generate_synthetic_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "fraud_model.joblib"
CATEGORY_MAPPING = {value: key for key, value in MERCHANT_CATEGORIES.items()}


def evaluate_model(model: FraudRiskModel, dataset, threshold: float = 0.6) -> None:
    logger.info(f"Starting evaluation run (Threshold: {threshold})...")
    
    features_list = [
        TransactionFeatures(
            amount=row["amount"],
            hour_of_day=int(row["hour_of_day"]),
            merchant_category=CATEGORY_MAPPING[row["merchant_category"]],
            distance_from_home_km=row["distance_from_home_km"],
            is_foreign_transaction=bool(row["is_foreign_transaction"]),
            velocity_last_hour=int(row["velocity_last_hour"]),
        )
        for _, row in dataset.iterrows()
    ]
    
    scores = np.array([model.predict_risk(f) for f in features_list])
    actual_labels = dataset["is_fraud"].values
    predicted_labels = (scores >= threshold).astype(int)
    
    true_pos = int(np.sum((predicted_labels == 1) & (actual_labels == 1)))
    false_pos = int(np.sum((predicted_labels == 1) & (actual_labels == 0)))
    false_neg = int(np.sum((predicted_labels == 0) & (actual_labels == 1)))
    
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
    
    fraud_scores = scores[actual_labels == 1]
    normal_scores = scores[actual_labels == 0]
    
    metrics_summary = (
        f"\n{'='*40}\n"
        f"PERFORMANCE EVALUATION SUMMARY\n"
        f"{'='*40}\n"
        f"Total Ingested Log Volume: {len(dataset)}\n"
        f"True Confirmed Fraud (Ground Truth): {int(actual_labels.sum())}\n"
        f"Engine Flagged Exceptions: {int(predicted_labels.sum())}\n"
        f"System Precision Level: {precision:.4f}\n"
        f"System Recall / Sensitivity: {recall:.4f}\n"
        f"Target Class Anomaly Score Average: {fraud_scores.mean():.4f}\n"
        f"Base Class Baseline Score Average: {normal_scores.mean():.4f}\n"
        f"{'='*40}"
    )
    print(metrics_summary)


def run_pipeline() -> None:
    logger.info("Initializing synthetic pipeline transaction synthesis (N=5000)...")
    dataset = generate_synthetic_dataset(n_samples=5000, fraud_ratio=0.03)

    logger.info("Fitting underlying Isolation Forest anomaly parameters...")
    model = FraudRiskModel(contamination=0.03)
    model.fit(dataset)
    
    evaluate_model(model, dataset)
    
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    logger.info(f"Model artifacts exported and sealed at: {MODEL_PATH}")


if __name__ == "__main__":
    run_pipeline()
