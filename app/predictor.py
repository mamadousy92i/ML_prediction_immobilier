"""
Prediction logic — loads the pkl once, exposes regression & classification helpers.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Paths ──
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "Mes_models.pkl"

# Global state (loaded once) 
_store: dict[str, Any] | None = None


def load_models() -> dict[str, Any]:
    """Load the pickle file and cache it in module-level variable."""
    global _store
    if _store is not None:
        return _store

    logger.info("Loading models from %s …", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        _store = pickle.load(f)
    logger.info("Models loaded successfully.")
    return _store


def get_store() -> dict[str, Any]:
    """Return the already-loaded store (call load_models first)."""
    if _store is None:
        raise RuntimeError("Models not loaded yet — call load_models() first.")
    return _store


# Regression 

# The numerical features expected by the regression pipeline
_REG_NUM_FEATURES = [
    "GrLivArea", "TotalBsmtSF", "LotArea", "BedroomAbvGr", "FullBath",
    "TotRmsAbvGrd", "OverallQual", "OverallCond", "YearBuilt",
    "YearRemodAdd", "GarageCars", "GarageArea", "PoolArea", "Fireplaces",
]

_REG_CAT_FEATURES = ["Neighborhood"]


def predict_regression(data: dict, model_name: str = "random_forest") -> float:
    """Run the regression pipeline and return predicted SalePrice."""
    store = get_store()
    reg = store["regression"]

    # Validate model name
    available = list(reg["models"].keys())
    if model_name not in available:
        raise ValueError(f"Unknown regression model '{model_name}'. Available: {available}")

    model = reg["models"][model_name]
    encoder = reg["preprocessors"]["onehot_encoder"]
    scaler = reg["preprocessors"]["scaler"]

    # 1. Build DataFrame
    df = pd.DataFrame([data])

    # 2. One-hot encode categorical columns
    encoded = encoder.transform(df[_REG_CAT_FEATURES])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(_REG_CAT_FEATURES),
    )

    # 3. Assemble numeric + encoded
    num_df = df[_REG_NUM_FEATURES].copy()
    features = pd.concat([num_df, encoded_df], axis=1)

    # 4. Scale all features
    features_scaled = scaler.transform(features)

    # 5. Predict
    prediction = model.predict(features_scaled)[0]
    logger.info("Regression (%s) → %.2f", model_name, prediction)
    return float(prediction)


# Classification 

_CLF_NUM_FEATURES = [
    "GrLivArea", "TotRmsAbvGrd", "OverallQual", "YearBuilt", "GarageCars",
]

_CLF_CAT_FEATURES = ["Neighborhood", "HouseStyle"]


def predict_classification(data: dict, model_name: str = "random_forest") -> tuple[str, int]:
    """Run the classification pipeline and return (label, encoded_value)."""
    store = get_store()
    clf = store["classification"]

    # Validate model name
    available = list(clf["models"].keys())
    if model_name not in available:
        raise ValueError(f"Unknown classification model '{model_name}'. Available: {available}")

    model = clf["models"][model_name]
    encoder = clf["preprocessors"]["onehot_encoder"]
    scaler = clf["preprocessors"]["scaler"]

    # label_encoder may live in preprocessors or at classification root
    label_encoder = clf["preprocessors"].get("label_encoder") or clf.get("label_encoder")
    if label_encoder is None:
        raise RuntimeError("label_encoder not found in pkl structure")

    # 1. Build DataFrame
    df = pd.DataFrame([data])

    # 2. One-hot encode categorical columns
    encoded = encoder.transform(df[_CLF_CAT_FEATURES])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(_CLF_CAT_FEATURES),
    )

    # 3. Assemble numeric + encoded
    num_df = df[_CLF_NUM_FEATURES].copy()
    features = pd.concat([num_df, encoded_df], axis=1)

    # 4. Scale all features
    features_scaled = scaler.transform(features)

    # 5. Predict
    encoded_pred = int(model.predict(features_scaled)[0])

    # 6. Inverse transform to get the human label
    label = label_encoder.inverse_transform([encoded_pred])[0]
    logger.info("Classification (%s) → %s (encoded=%d)", model_name, label, encoded_pred)
    return str(label), encoded_pred
