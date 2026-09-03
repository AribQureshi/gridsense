"""
GridSense — Step 5: FastAPI Backend
======================================
Serves the trained model (from Step 4) over HTTP so the React frontend
(Step 6) has something real to call.

Endpoints:
    GET  /health              -> simple liveness check
    GET  /historical           -> actual demand + in-sample model fit, for the time-series chart
    POST /predict               -> live "what-if" prediction given weather + calendar inputs
    GET  /models/compare        -> RMSE/MAE/R2 comparison table across all models
    GET  /diagnostics            -> VIF scores + Breusch-Pagan result + Cook's distance summary
    GET  /coefficients            -> OLS coefficients, p-values, confidence intervals

Run this with:  uvicorn src.api:app --reload
Then visit:      http://127.0.0.1:8000/docs   <- auto-generated interactive API docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------------------
# 0. App setup
# ---------------------------------------------------------------------
app = FastAPI(title="GridSense API", description="Delhi energy demand forecasting")

# Allow the React dev server (typically localhost:5173 or :3000) to call this API.
# In production you'd restrict this to your actual frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "outputs/models"
DATA_PATH = "data/processed/gridsense_features.csv"
COMFORT_BASELINE_C = 18.0

# ---------------------------------------------------------------------
# 1. Load model artifacts ONCE at startup (not per-request — that would
#    be slow and pointless, since these files don't change while the
#    server is running)
# ---------------------------------------------------------------------
model = None
scaler = None
feature_list = None
features_df = None


@app.on_event("startup")
def load_artifacts():
    global model, scaler, feature_list, features_df

    for path in [f"{MODELS_DIR}/best_model.pkl", f"{MODELS_DIR}/scaler.pkl",
                 f"{MODELS_DIR}/feature_list.pkl", DATA_PATH]:
        if not os.path.exists(path):
            raise RuntimeError(
                f"Missing required file: {path}. Run src/train_models.py first."
            )

    model = joblib.load(f"{MODELS_DIR}/best_model.pkl")
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    feature_list = joblib.load(f"{MODELS_DIR}/feature_list.pkl")
    features_df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"Loaded model ({type(model).__name__}) with {len(feature_list)} features")


# ---------------------------------------------------------------------
# 2. Health check
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


# ---------------------------------------------------------------------
# 3. Historical actual vs. predicted (for the time-series chart)
# ---------------------------------------------------------------------
@app.get("/historical")
def historical():
    X = scaler.transform(features_df[feature_list])
    predictions = model.predict(X)

    return {
        "dates": features_df["date"].dt.strftime("%Y-%m-%d").tolist(),
        "actual": features_df["demand_mu"].round(2).tolist(),
        "predicted": np.round(predictions, 2).tolist(),
        "is_lockdown": features_df["is_lockdown"].tolist(),
    }


# ---------------------------------------------------------------------
# 4. Live "what-if" prediction
# ---------------------------------------------------------------------
class PredictionRequest(BaseModel):
    date: str = Field(..., description="YYYY-MM-DD, used to derive seasonal features")
    temp_mean_c: float = Field(..., ge=-10, le=55, description="Mean temperature in Celsius")
    humidity_mean_pct: float = Field(..., ge=0, le=100)
    precipitation_mm: float = Field(0.0, ge=0)
    is_lockdown: bool = Field(False, description="Simulate a lockdown-like demand shock")
    # Lag features default to the dataset's recent average if not provided --
    # a what-if simulator user usually doesn't know "yesterday's demand" by heart.
    demand_lag_1: float | None = Field(None, description="Optional: yesterday's demand (MU)")
    demand_lag_7: float | None = Field(None, description="Optional: same day last week's demand (MU)")


@app.post("/predict")
def predict(req: PredictionRequest):
    try:
        date = pd.Timestamp(req.date)
    except ValueError:
        raise HTTPException(400, "date must be in YYYY-MM-DD format")

    recent_avg = features_df["demand_mu"].tail(30).mean()
    lag_1 = req.demand_lag_1 if req.demand_lag_1 is not None else recent_avg
    lag_7 = req.demand_lag_7 if req.demand_lag_7 is not None else recent_avg

    day_of_year = date.dayofyear
    row = {
        "demand_lag_1": lag_1,
        "demand_lag_7": lag_7,
        "is_weekend": int(date.dayofweek in [5, 6]),
        "is_lockdown": int(req.is_lockdown),
        "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
        "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        "temp_mean_c": req.temp_mean_c,
        "temp_mean_c_sq": req.temp_mean_c ** 2,
        "hdd": max(COMFORT_BASELINE_C - req.temp_mean_c, 0),
        "cdd": max(req.temp_mean_c - COMFORT_BASELINE_C, 0),
        "humidity_mean_pct": req.humidity_mean_pct,
        "precipitation_mm": req.precipitation_mm,
    }

    # Build the feature vector in the EXACT order the model was trained on --
    # this ordering must match FULL_FEATURES in train_models.py precisely,
    # or the scaler/model will silently apply the wrong scaling to the wrong
    # column. feature_list.pkl guarantees we use the same order every time.
    X = pd.DataFrame([row])[feature_list]
    X_scaled = scaler.transform(X)
    prediction = float(model.predict(X_scaled)[0])

    return {
        "predicted_demand_mu": round(prediction, 2),
        "inputs_used": row,
        "note": "demand_lag_1/7 defaulted to the trailing 30-day average"
                if req.demand_lag_1 is None else None,
    }


# ---------------------------------------------------------------------
# 5. Model comparison table
# ---------------------------------------------------------------------
@app.get("/models/compare")
def models_compare():
    path = "outputs/model_comparison.csv"
    if not os.path.exists(path):
        raise HTTPException(404, f"{path} not found. Run src/train_models.py first.")
    df = pd.read_csv(path, index_col=0)
    return df.reset_index().rename(columns={"index": "model"}).to_dict(orient="records")


# ---------------------------------------------------------------------
# 6. Diagnostics summary
# ---------------------------------------------------------------------
@app.get("/diagnostics")
def diagnostics():
    vif_path = "outputs/vif_scores.csv"
    summary_path = "outputs/diagnostics_summary.json"
    if not os.path.exists(vif_path) or not os.path.exists(summary_path):
        raise HTTPException(404, "Diagnostics files not found. Run src/train_models.py first.")

    vif_df = pd.read_csv(vif_path)
    # Replace inf with a large finite sentinel for JSON serialization (JSON
    # has no native concept of infinity)
    vif_df["VIF"] = vif_df["VIF"].replace(np.inf, 999999).round(2)

    import json
    with open(summary_path) as f:
        summary = json.load(f)

    return {
        "vif_scores": vif_df.to_dict(orient="records"),
        "breusch_pagan": summary["breusch_pagan"],
        "cooks_distance": summary["cooks_distance"],
        "note": "VIF > 10 indicates significant multicollinearity. "
                "A value of 999999 represents mathematical infinity (perfect collinearity).",
    }


# ---------------------------------------------------------------------
# 7. OLS coefficients (parsed from the saved summary text)
# ---------------------------------------------------------------------
@app.get("/coefficients")
def coefficients():
    path = "outputs/ols_summary.txt"
    if not os.path.exists(path):
        raise HTTPException(404, f"{path} not found. Run src/train_models.py first.")
    with open(path) as f:
        text = f.read()
    return {"ols_summary_text": text}
