import numpy as np
import pandas as pd

MERCHANT_CATEGORIES = {
    "grocery": 0,
    "electronics": 1,
    "travel": 2,
    "restaurant": 3,
    "gambling": 4,
    "other": 5,
}


def generate_synthetic_dataset(n_samples: int = 5000, fraud_ratio: float = 0.03, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    normal = pd.DataFrame({
        "amount": rng.gamma(shape=2.0, scale=40, size=n_normal),
        "hour_of_day": rng.normal(loc=14, scale=4, size=n_normal).clip(0, 23).astype(int),
        "merchant_category": rng.choice(list(MERCHANT_CATEGORIES.values()),
                                         size=n_normal, p=[0.35, 0.2, 0.1, 0.25, 0.02, 0.08]),
        "distance_from_home_km": rng.exponential(scale=8, size=n_normal),
        "is_foreign_transaction": rng.choice([0, 1], size=n_normal, p=[0.95, 0.05]),
        "velocity_last_hour": rng.poisson(lam=1.0, size=n_normal),
        "is_fraud": 0,
    })

    fraud = pd.DataFrame({
        "amount": rng.gamma(shape=3.0, scale=250, size=n_fraud),
        "hour_of_day": rng.normal(loc=3, scale=3, size=n_fraud).clip(0, 23).astype(int),
        "merchant_category": rng.choice(list(MERCHANT_CATEGORIES.values()),
                                         size=n_fraud, p=[0.05, 0.3, 0.25, 0.05, 0.3, 0.05]),
        "distance_from_home_km": rng.exponential(scale=300, size=n_fraud),
        "is_foreign_transaction": rng.choice([0, 1], size=n_fraud, p=[0.3, 0.7]),
        "velocity_last_hour": rng.poisson(lam=6.0, size=n_fraud),
        "is_fraud": 1,
    })

    df = pd.concat([normal, fraud], ignore_index=True)
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)
