"""
FastAPI Backend for Bank Marketing Predictor.
Exposes REST endpoints for health status and ML inference.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .dataset_service import dataset_service
from .pipeline_data import get_performance_data, get_pipeline_metadata
from .prediction_service import prediction_service
from .schemas import (
    CustomerInputSchema,
    DatasetFeaturesResponseSchema,
    DatasetPreviewResponseSchema,
    DatasetSummaryResponseSchema,
    EDAResponseSchema,
    HealthResponseSchema,
    PerformanceResponseSchema,
    PipelineResponseSchema,
    PredictionResponseSchema,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bank_predictor_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the trained model once during application startup.
    """
    logger.info("Starting Bank Marketing Predictor API...")
    try:
        prediction_service.load_model()
        logger.info("Machine learning model pipeline loaded successfully.")
    except Exception as exc:
        logger.error(f"Failed to load model pipeline on startup: {exc}")
    yield
    logger.info("Shutting down Bank Marketing Predictor API.")


app = FastAPI(
    title="Bank Marketing Predictor API",
    description="REST API serving the scikit-learn Random Forest model for term deposit subscription prediction.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"
)
allowed_origins: List[str] = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponseSchema, tags=["Health"])
async def health_check():
    """
    Confirm the API is healthy and indicate whether the model is loaded in memory.
    """
    is_loaded = prediction_service.is_loaded()
    return HealthResponseSchema(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_name=prediction_service.model_name if is_loaded else "Unavailable",
        timing_constraint="Pre-contact prediction (duration excluded)",
    )


@app.post(
    "/api/predict",
    response_model=PredictionResponseSchema,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
async def predict_customer(payload: CustomerInputSchema):
    """
    Accept validated customer details and return binary prediction and subscription probability.
    """
    if not prediction_service.is_loaded():
        logger.error("Predict called but model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model is currently not loaded. Please verify server setup.",
        )

    try:
        input_data = payload.model_dump()
        result = prediction_service.predict(input_data)
        return PredictionResponseSchema(**result)
    except ValueError as val_err:
        logger.warning(f"Validation or data processing error during prediction: {val_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Data processing error: {str(val_err)}",
        )
    except Exception as exc:
        logger.error(f"Unexpected error during inference: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while calculating the prediction. Please try again.",
        )


@app.get("/api/pipeline", response_model=PipelineResponseSchema, tags=["Pipeline"])
async def get_pipeline():
    """
    Return documented, verified ML pipeline stages, methodology, and model specifications.
    """
    try:
        data = get_pipeline_metadata()
        return PipelineResponseSchema(**data)
    except Exception as exc:
        logger.error(f"Failed to assemble pipeline metadata: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve pipeline workflow details.",
        )


@app.get("/api/performance", response_model=PerformanceResponseSchema, tags=["Performance"])
async def get_performance():
    """
    Return verified model performance metrics, confusion matrix, and feature importances.
    """
    try:
        data = get_performance_data()
        return PerformanceResponseSchema(**data)
    except Exception as exc:
        logger.error(f"Failed to assemble performance metrics: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model performance evaluation.",
        )


@app.get("/api/dataset/summary", response_model=DatasetSummaryResponseSchema, tags=["Dataset"])
async def get_dataset_summary():
    """
    Return high-level verified summary of the primary bank marketing dataset.
    """
    try:
        summary = dataset_service.get_summary()
        return DatasetSummaryResponseSchema(**summary)
    except FileNotFoundError as fnf:
        logger.error(f"Dataset file not found: {fnf}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file data/bank-full.csv not found.",
        )
    except Exception as exc:
        logger.error(f"Error computing dataset summary: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dataset summary.",
        )


@app.get("/api/dataset/features", response_model=DatasetFeaturesResponseSchema, tags=["Dataset"])
async def get_dataset_features():
    """
    Return structured feature catalogue describing every feature, role, type, and model status.
    """
    try:
        catalogue = dataset_service.get_features_catalogue()
        return DatasetFeaturesResponseSchema(
            total_features=len(catalogue),
            features=catalogue,
        )
    except Exception as exc:
        logger.error(f"Error retrieving feature catalogue: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve feature catalogue.",
        )


@app.get("/api/dataset/preview", response_model=DatasetPreviewResponseSchema, tags=["Dataset"])
async def get_dataset_preview(page: int = 1, page_size: int = 10):
    """
    Return paginated read-only preview of rows from data/bank-full.csv.
    """
    if page_size < 1 or page_size > 50:
        page_size = 10
    if page < 1:
        page = 1

    try:
        preview = dataset_service.get_preview(page=page, page_size=page_size)
        return DatasetPreviewResponseSchema(**preview)
    except FileNotFoundError as fnf:
        logger.error(f"Dataset file not found for preview: {fnf}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file data/bank-full.csv not found.",
        )
    except Exception as exc:
        logger.error(f"Error generating dataset preview: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate dataset preview.",
        )


@app.get("/api/dataset/eda", response_model=EDAResponseSchema, tags=["Dataset"])
async def get_dataset_eda():
    """
    Return precalculated verified exploratory data analysis metrics and distributions.
    """
    try:
        eda_data = dataset_service.get_eda_metrics()
        return EDAResponseSchema(**eda_data)
    except Exception as exc:
        logger.error(f"Error compiling EDA distributions: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve exploratory data distributions.",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

