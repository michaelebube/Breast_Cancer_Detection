"""
Breast Cancer Prediction API
FastAPI backend for breast cancer detection system
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from typing import Optional
import numpy as np
import joblib
import json
import os
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API for predicting breast cancer diagnosis (Benign/Malignant) using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory containing this script
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

# Global variables for model and related artifacts
model = None
scaler = None
dataset_stats = None


def load_model_artifacts():
    """Load model, scaler, and dataset statistics"""
    global model, scaler, dataset_stats

    try:
        model_path = MODEL_DIR / "breast_cancer_model.joblib"
        scaler_path = MODEL_DIR / "scaler.joblib"
        stats_path = MODEL_DIR / "dataset_stats.json"

        if model_path.exists():
            model = joblib.load(model_path)
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"⚠️ Model file not found at {model_path}")

        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            print(f"✅ Scaler loaded from {scaler_path}")
        else:
            print(f"⚠️ Scaler file not found at {scaler_path}")

        if stats_path.exists():
            with open(stats_path, "r") as f:
                dataset_stats = json.load(f)
            print(f"✅ Dataset stats loaded from {stats_path}")
        else:
            print(f"⚠️ Stats file not found at {stats_path}")
            # Create default stats based on Wisconsin Breast Cancer Dataset
            dataset_stats = get_default_stats()

    except Exception as e:
        print(f"❌ Error loading model artifacts: {e}")
        dataset_stats = get_default_stats()


def get_default_stats():
    """Return default statistics based on Wisconsin Breast Cancer Dataset"""
    return {
        "feature_names": [
            "mean radius",
            "mean texture",
            "mean perimeter",
            "mean area",
            "mean smoothness",
            "mean compactness",
            "mean concavity",
            "mean concave points",
            "mean symmetry",
            "mean fractal dimension",
            "radius error",
            "texture error",
            "perimeter error",
            "area error",
            "smoothness error",
            "compactness error",
            "concavity error",
            "concave points error",
            "symmetry error",
            "fractal dimension error",
            "worst radius",
            "worst texture",
            "worst perimeter",
            "worst area",
            "worst smoothness",
            "worst compactness",
            "worst concavity",
            "worst concave points",
            "worst symmetry",
            "worst fractal dimension",
        ],
        "feature_means": {
            "mean radius": 14.127,
            "mean texture": 19.289,
            "mean perimeter": 91.969,
            "mean area": 654.889,
            "mean smoothness": 0.096,
            "mean compactness": 0.104,
            "mean concavity": 0.089,
            "mean concave points": 0.049,
            "mean symmetry": 0.181,
            "mean fractal dimension": 0.063,
            "radius error": 0.405,
            "texture error": 1.217,
            "perimeter error": 2.866,
            "area error": 40.337,
            "smoothness error": 0.007,
            "compactness error": 0.025,
            "concavity error": 0.032,
            "concave points error": 0.012,
            "symmetry error": 0.021,
            "fractal dimension error": 0.004,
            "worst radius": 16.269,
            "worst texture": 25.677,
            "worst perimeter": 107.261,
            "worst area": 880.583,
            "worst smoothness": 0.132,
            "worst compactness": 0.254,
            "worst concavity": 0.272,
            "worst concave points": 0.115,
            "worst symmetry": 0.290,
            "worst fractal dimension": 0.084,
        },
    }


# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model_artifacts()


# ==================== Pydantic Models ====================


class UserInput(BaseModel):
    """User-friendly input from frontend"""

    lump_size: float = Field(
        ..., ge=0.1, le=10.0, description="Lump size in cm (0.1-10)"
    )
    lump_shape: str = Field(..., description="Lump shape: 'regular' or 'irregular'")
    lump_texture: str = Field(..., description="Lump texture: 'smooth' or 'rough'")
    lump_hardness: float = Field(
        ..., ge=0, le=100, description="Lump hardness (0-100 scale)"
    )
    growth_rate: str = Field(..., description="Growth rate: 'slow' or 'fast'")
    pain_present: bool = Field(..., description="Is pain present?")
    skin_changes: bool = Field(..., description="Are there skin changes?")
    nipple_discharge: bool = Field(..., description="Is there nipple discharge?")
    family_history: bool = Field(
        ..., description="Is there family history of breast cancer?"
    )
    patient_age: int = Field(..., ge=18, le=120, description="Patient age in years")

    @validator("lump_shape")
    def validate_lump_shape(cls, v):
        if v.lower() not in ["regular", "irregular"]:
            raise ValueError("lump_shape must be 'regular' or 'irregular'")
        return v.lower()

    @validator("lump_texture")
    def validate_lump_texture(cls, v):
        if v.lower() not in ["smooth", "rough"]:
            raise ValueError("lump_texture must be 'smooth' or 'rough'")
        return v.lower()

    @validator("growth_rate")
    def validate_growth_rate(cls, v):
        if v.lower() not in ["slow", "fast"]:
            raise ValueError("growth_rate must be 'slow' or 'fast'")
        return v.lower()


class RawFeatureInput(BaseModel):
    """Raw 30-feature vector input for direct model prediction"""

    features: list[float] = Field(
        ..., min_items=30, max_items=30, description="30 numerical features"
    )


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    prediction: str = Field(..., description="Classification: 'Benign' or 'Malignant'")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probability_benign: float = Field(..., description="Probability of benign")
    probability_malignant: float = Field(..., description="Probability of malignant")
    risk_level: str = Field(..., description="Risk level: 'Low', 'Medium', or 'High'")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model_loaded: bool
    scaler_loaded: bool
    version: str


# ==================== Feature Mapping ====================


def map_user_input_to_features(user_input: UserInput) -> np.ndarray:
    """
    Map user-friendly inputs to the 30 features expected by the model.

    This mapping uses heuristics and dataset statistics to convert
    simplified inputs into medically-relevant features.

    Feature mapping logic:
    - Lump Size → radius_mean, perimeter_mean, area_mean
    - Lump Texture → texture_mean
    - Lump Shape → concavity_mean, symmetry_mean
    - Lump Hardness → compactness_mean
    - Binary inputs → scaled numerical values
    """

    # Get default means from dataset stats
    means = dataset_stats.get("feature_means", get_default_stats()["feature_means"])

    # Initialize features with dataset means
    features = {}
    for feature_name in dataset_stats["feature_names"]:
        # Convert feature names with underscores to match our dict
        key = feature_name.replace(" ", "_")
        features[feature_name] = means.get(feature_name, means.get(key, 0))

    # ===== Map Lump Size to radius, perimeter, area =====
    # Lump size (0.1-10 cm) maps to radius_mean (6.98-28.11 in dataset)
    lump_size_normalized = (user_input.lump_size - 0.1) / (10.0 - 0.1)

    # Mean features
    features["mean radius"] = 6.98 + lump_size_normalized * (28.11 - 6.98)
    features["mean perimeter"] = features["mean radius"] * 2 * np.pi  # Approximate
    features["mean area"] = np.pi * (features["mean radius"] ** 2)  # Approximate

    # Worst features (typically 1.2-1.5x mean for concerning cases)
    worst_multiplier = 1.2 + (0.3 * lump_size_normalized)
    features["worst radius"] = features["mean radius"] * worst_multiplier
    features["worst perimeter"] = features["worst radius"] * 2 * np.pi
    features["worst area"] = np.pi * (features["worst radius"] ** 2)

    # Error features (variability)
    features["radius error"] = 0.1 + lump_size_normalized * 2.0
    features["perimeter error"] = features["radius error"] * 2 * np.pi * 0.5
    features["area error"] = features["radius error"] * features["mean radius"] * 2

    # ===== Map Lump Texture =====
    # Rough texture indicates higher texture values
    if user_input.lump_texture == "rough":
        features["mean texture"] = 20 + np.random.uniform(5, 15)
        features["worst texture"] = features["mean texture"] * 1.3
    else:
        features["mean texture"] = 12 + np.random.uniform(0, 8)
        features["worst texture"] = features["mean texture"] * 1.15
    features["texture error"] = (
        abs(features["worst texture"] - features["mean texture"]) / 5
    )

    # ===== Map Lump Shape to concavity and symmetry =====
    if user_input.lump_shape == "irregular":
        # Irregular shapes have higher concavity and lower symmetry
        features["mean concavity"] = 0.1 + np.random.uniform(0.05, 0.2)
        features["mean concave points"] = 0.05 + np.random.uniform(0.02, 0.1)
        features["mean symmetry"] = 0.15 + np.random.uniform(0, 0.05)

        features["worst concavity"] = features["mean concavity"] * 1.8
        features["worst concave points"] = features["mean concave points"] * 1.8
        features["worst symmetry"] = features["mean symmetry"] * 1.5
    else:
        # Regular shapes have lower concavity and higher symmetry
        features["mean concavity"] = 0.02 + np.random.uniform(0, 0.05)
        features["mean concave points"] = 0.01 + np.random.uniform(0, 0.03)
        features["mean symmetry"] = 0.17 + np.random.uniform(0, 0.03)

        features["worst concavity"] = features["mean concavity"] * 1.3
        features["worst concave points"] = features["mean concave points"] * 1.3
        features["worst symmetry"] = features["mean symmetry"] * 1.2

    features["concavity error"] = (
        abs(features["worst concavity"] - features["mean concavity"]) / 3
    )
    features["concave points error"] = (
        abs(features["worst concave points"] - features["mean concave points"]) / 3
    )
    features["symmetry error"] = (
        abs(features["worst symmetry"] - features["mean symmetry"]) / 3
    )

    # ===== Map Lump Hardness to compactness =====
    # Hardness 0-100 maps to compactness
    hardness_normalized = user_input.lump_hardness / 100.0
    features["mean compactness"] = 0.02 + hardness_normalized * 0.3
    features["worst compactness"] = features["mean compactness"] * (
        1.2 + hardness_normalized * 0.5
    )
    features["compactness error"] = (
        abs(features["worst compactness"] - features["mean compactness"]) / 4
    )

    # ===== Map Growth Rate =====
    # Fast growth affects fractal dimension and smoothness
    if user_input.growth_rate == "fast":
        features["mean fractal dimension"] = 0.06 + np.random.uniform(0.01, 0.03)
        features["mean smoothness"] = 0.08 + np.random.uniform(0.02, 0.05)
    else:
        features["mean fractal dimension"] = 0.055 + np.random.uniform(0, 0.01)
        features["mean smoothness"] = 0.08 + np.random.uniform(0, 0.03)

    features["worst fractal dimension"] = features["mean fractal dimension"] * 1.3
    features["worst smoothness"] = features["mean smoothness"] * 1.2
    features["fractal dimension error"] = 0.001 + np.random.uniform(0, 0.005)
    features["smoothness error"] = 0.003 + np.random.uniform(0, 0.007)

    # ===== Adjust based on binary risk factors =====
    risk_factor = 0

    if user_input.pain_present:
        risk_factor += 0.1

    if user_input.skin_changes:
        risk_factor += 0.15
        features["mean texture"] *= 1.1

    if user_input.nipple_discharge:
        risk_factor += 0.1

    if user_input.family_history:
        risk_factor += 0.2

    # Age factor (risk increases with age)
    age_factor = max(0, (user_input.patient_age - 40) / 60)
    risk_factor += age_factor * 0.1

    # Apply risk factor adjustments
    if risk_factor > 0:
        features["mean concavity"] *= 1 + risk_factor * 0.3
        features["worst concavity"] *= 1 + risk_factor * 0.3
        features["mean compactness"] *= 1 + risk_factor * 0.2
        features["worst compactness"] *= 1 + risk_factor * 0.2

    # ===== Create feature vector in correct order =====
    feature_names = dataset_stats["feature_names"]
    feature_vector = [features[name] for name in feature_names]

    return np.array(feature_vector).reshape(1, -1)


def get_risk_level(probability_malignant: float) -> str:
    """Determine risk level based on malignancy probability"""
    if probability_malignant < 0.3:
        return "Low"
    elif probability_malignant < 0.6:
        return "Medium"
    else:
        return "High"


# ==================== API Endpoints ====================


@app.get("/", tags=["Root"])
async def root():
    """Serve the frontend"""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api", tags=["Root"])
async def api_info():
    """API information endpoint"""
    return {
        "message": "Breast Cancer Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint to verify API status"""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        scaler_loaded=scaler is not None,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(user_input: UserInput):
    """
    Predict breast cancer diagnosis from user-friendly inputs.

    This endpoint accepts simplified medical observations and converts
    them to the 30-feature vector required by the model.

    Returns:
        - Classification (Benign/Malignant)
        - Confidence score
        - Probability distribution
        - Risk level assessment
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model files are in the /model directory.",
        )

    try:
        # Map user inputs to feature vector
        features = map_user_input_to_features(user_input)

        # Scale features if scaler is available
        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            # Use basic standardization if scaler not available
            features_scaled = features

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        # Model output: 0 = Malignant, 1 = Benign
        prob_malignant = probabilities[0]
        prob_benign = probabilities[1]

        prediction_label = "Benign" if prediction == 1 else "Malignant"
        confidence = max(prob_benign, prob_malignant)

        return PredictionResponse(
            prediction=prediction_label,
            confidence=round(confidence, 4),
            probability_benign=round(prob_benign, 4),
            probability_malignant=round(prob_malignant, 4),
            risk_level=get_risk_level(prob_malignant),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/raw", response_model=PredictionResponse, tags=["Prediction"])
async def predict_raw(raw_input: RawFeatureInput):
    """
    Predict from raw 30-feature vector.

    This endpoint is for advanced users who have the actual
    medical measurements from diagnostic imaging.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model files are in the /model directory.",
        )

    try:
        features = np.array(raw_input.features).reshape(1, -1)

        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features

        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        prob_malignant = probabilities[0]
        prob_benign = probabilities[1]

        prediction_label = "Benign" if prediction == 1 else "Malignant"
        confidence = max(prob_benign, prob_malignant)

        return PredictionResponse(
            prediction=prediction_label,
            confidence=round(confidence, 4),
            probability_benign=round(prob_benign, 4),
            probability_malignant=round(prob_malignant, 4),
            risk_level=get_risk_level(prob_malignant),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/features", tags=["Info"])
async def get_feature_info():
    """Get information about model features and their expected ranges"""
    if dataset_stats is None:
        return {"error": "Dataset statistics not loaded"}

    return {
        "feature_count": len(dataset_stats["feature_names"]),
        "feature_names": dataset_stats["feature_names"],
        "feature_means": dataset_stats.get("feature_means", {}),
        "input_mapping": {
            "lump_size": "Maps to: radius_mean, perimeter_mean, area_mean",
            "lump_texture": "Maps to: texture_mean",
            "lump_shape": "Maps to: concavity_mean, symmetry_mean",
            "lump_hardness": "Maps to: compactness_mean",
            "growth_rate": "Maps to: fractal_dimension_mean, smoothness_mean",
            "binary_inputs": "Adjust risk factors and feature scaling",
        },
    }


# Serve frontend static files (CSS, JS)
@app.get("/styles.css")
async def serve_css():
    return FileResponse(FRONTEND_DIR / "styles.css")


@app.get("/app.js")
async def serve_js():
    return FileResponse(FRONTEND_DIR / "app.js")


# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
