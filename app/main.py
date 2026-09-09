import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from app.ml_model import FraudRiskModel, MERCHANT_CATEGORIES, TransactionFeatures
from app.queue_engine import Transaction, TransactionPriorityQueue

logger = logging.getLogger("fraud_engine")

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "fraud_model.joblib"

app = FastAPI(title="Real-Time Fraud Detection Engine", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.queue = TransactionPriorityQueue()

try:
    app.state.model = FraudRiskModel.load(MODEL_PATH)
except FileNotFoundError:
    logger.critical(f"Failed to locate fraud model at {MODEL_PATH}")
    app.state.model = None


class TransactionIn(BaseModel):
    transaction_id: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0)
    hour_of_day: int = Field(..., ge=0, le=23)
    merchant_category: str
    distance_from_home_km: float = Field(..., ge=0)
    is_foreign_transaction: bool = False
    velocity_last_hour: int = Field(0, ge=0)


class TransactionOut(BaseModel):
    transaction_id: str
    amount: float
    risk_score: float


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "queue_size": len(app.state.queue),
        "model_loaded": app.state.model is not None,
    }


@app.post("/transactions", response_model=TransactionOut, status_code=status.HTTP_201_CREATED)
def submit_transaction(txn_in: TransactionIn):
    if not app.state.model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction engine uninitialized. Run model training scripts.",
        )

    if txn_in.merchant_category not in MERCHANT_CATEGORIES:
        valid_categories = list(MERCHANT_CATEGORIES.keys())
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid category. Must be one of: {valid_categories}",
        )

    features = TransactionFeatures(
        amount=txn_in.amount,
        hour_of_day=txn_in.hour_of_day,
        merchant_category=txn_in.merchant_category,
        distance_from_home_km=txn_in.distance_from_home_km,
        is_foreign_transaction=txn_in.is_foreign_transaction,
        velocity_last_hour=txn_in.velocity_last_hour,
    )

    risk_score = app.state.model.predict_risk(features)
    
    txn = Transaction(
        transaction_id=txn_in.transaction_id,
        amount=txn_in.amount,
        risk_score=risk_score,
    )
    
    app.state.queue.insert(txn)
    return TransactionOut(
        transaction_id=txn.transaction_id,
        amount=txn.amount,
        risk_score=txn.risk_score
    )


@app.post("/process-next", response_model=TransactionOut)
def process_next():
    txn = app.state.queue.extract_max()
    if not txn:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="No pending transactions in priority queue."
        )
        
    return TransactionOut(
        transaction_id=txn.transaction_id,
        amount=txn.amount,
        risk_score=txn.risk_score
    )


@app.get("/queue/size")
def queue_size():
    return {"queue_size": len(app.state.queue)}
