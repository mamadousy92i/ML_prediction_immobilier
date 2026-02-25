"""
FastAPI application — Immo Predictor API.
"""

import logging
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.predictor import (
    get_store,
    load_models,
    predict_classification,
    predict_regression,
)
from app.schemas import (
    ClassificationInput,
    ClassificationOutput,
    RegressionInput,
    RegressionOutput,
)

# Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Lifespan (load models once at startup)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Immo Predictor API...")
    load_models()
    yield
    logger.info("Shutting down Immo Predictor API.")


# App ────
app = FastAPI(
    title="Immo Predictor API",
    description=(
        "API de Machine Learning immobilier — "
        "Prédiction du prix (régression) et classification du type de bien."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Gradio UI at /ui
from app.ui import demo as gradio_demo

app = gr.mount_gradio_app(app, gradio_demo, path="/ui")


# Routes


@app.get("/", tags=["General"], summary="Redirection vers l'interface")
def root():
    """Redirige vers l'interface utilisateur Gradio."""
    return RedirectResponse(url="/ui")


@app.get("/health", tags=["General"], summary="Statut de l'API")
def health():
    """Vérifie que l'API est opérationnelle et que les modèles sont chargés."""
    try:
        store = get_store()
        return {
            "status": "healthy",
            "models_loaded": True,
            "regression_models": list(store["regression"]["models"].keys()),
            "classification_models": list(store["classification"]["models"].keys()),
        }
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")


@app.get("/models/info", tags=["General"], summary="Informations sur les modèles")
def models_info():
    """Retourne les détails des modèles chargés."""
    try:
        store = get_store()
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    def _model_meta(m):
        return {"type": type(m).__name__, "params": getattr(m, "get_params", lambda: {})()}

    return {
        "regression": {
            "target": store["regression"]["target"],
            "models": {k: _model_meta(v) for k, v in store["regression"]["models"].items()},
        },
        "classification": {
            "target": store["classification"]["target"],
            "models": {k: _model_meta(v) for k, v in store["classification"]["models"].items()},
        },
    }


# Regression


@app.post(
    "/regression/predict",
    response_model=RegressionOutput,
    tags=["Regression"],
    summary="Prédire le prix d'un bien immobilier",
)
def regression_predict(
    data: RegressionInput,
    model: str = Query(
        default="random_forest",
        description="Modèle à utiliser (decision_tree | random_forest)",
    ),
):
    """
    Prédire le **SalePrice** d'un bien immobilier à partir de ses caractéristiques.

    - `model=random_forest` (défaut) ou `model=decision_tree`
    """
    try:
        price = predict_regression(data.model_dump(), model_name=model)
        return RegressionOutput(
            model_used=model,
            predicted_price=round(price, 2),
            currency="USD",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Regression prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")


# Classification 


@app.post(
    "/classification/predict",
    response_model=ClassificationOutput,
    tags=["Classification"],
    summary="Classifier le type de bien immobilier",
)
def classification_predict(
    data: ClassificationInput,
    model: str = Query(
        default="random_forest",
        description="Modèle à utiliser (random_forest | svm)",
    ),
):
    """
    Classifier le **BldgType** (type de bâtiment) à partir de ses caractéristiques.

    - `model=random_forest` (défaut) ou `model=svm`
    """
    try:
        label, encoded = predict_classification(data.model_dump(), model_name=model)
        return ClassificationOutput(
            model_used=model,
            predicted_type=label,
            predicted_type_encoded=encoded,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Classification prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")
