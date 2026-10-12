"""
model.py
========
Module for managing ML models: loading pipelines, inference, uncertainty estimation,
and model performance introspection.
"""

import os
import json
from typing import Dict, Any, List, Tuple
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODELS_FILE = os.path.join(MODELS_DIR, "trained_models.joblib")
METRICS_FILE = os.path.join(MODELS_DIR, "model_metrics.json")

_LOADED_MODELS = None
_LOADED_METRICS = None


def load_models() -> Dict[str, Any]:
    """Load pre-trained regression pipelines."""
    global _LOADED_MODELS
    if _LOADED_MODELS is not None:
        return _LOADED_MODELS

    if not os.path.exists(MODELS_FILE):
        # Fallback to train if file not present
        from train_model import train_and_evaluate_models
        train_and_evaluate_models()

    _LOADED_MODELS = joblib.load(MODELS_FILE)
    return _LOADED_MODELS


def load_metrics() -> Dict[str, Any]:
    """Load saved evaluation metrics and parity plot samples."""
    global _LOADED_METRICS
    if _LOADED_METRICS is not None:
        return _LOADED_METRICS

    if not os.path.exists(METRICS_FILE):
        from train_model import train_and_evaluate_models
        train_and_evaluate_models()

    with open(METRICS_FILE, "r") as f:
        _LOADED_METRICS = json.load(f)
    return _LOADED_METRICS


def predict_emissions(user_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run inference across all trained models (Linear Regression, Random Forest, Gradient Boosting).
    Returns predictions, ensemble mean, and ML prediction interval.
    """
    models = load_models()

    # Format into DataFrame with correct schema
    feature_order = [
        "transport_mode", "distance_km", "occupancy", "electricity_kwh",
        "solar_percentage", "household_size", "diet_type", "extra_dairy_servings",
        "waste_kg", "compost_percentage", "recycle_percentage"
    ]

    row_data = {
        "transport_mode": str(user_inputs.get("transport_mode", "petrol_car")),
        "distance_km": float(user_inputs.get("distance_km", 0.0)),
        "occupancy": int(user_inputs.get("occupancy", 1)),
        "electricity_kwh": float(user_inputs.get("electricity_kwh", 5.0)),
        "solar_percentage": float(user_inputs.get("solar_percentage", 0.0)),
        "household_size": int(user_inputs.get("household_size", 1)),
        "diet_type": str(user_inputs.get("diet_type", "vegetarian")),
        "extra_dairy_servings": int(user_inputs.get("extra_dairy_servings", 1)),
        "waste_kg": float(user_inputs.get("waste_kg", 0.5)),
        "compost_percentage": float(user_inputs.get("compost_percentage", 0.0)),
        "recycle_percentage": float(user_inputs.get("recycle_percentage", 0.0)),
    }

    input_df = pd.DataFrame([row_data], columns=feature_order)

    predictions = {}
    preds_list = []

    for name, pipeline in models.items():
        pred_val = float(pipeline.predict(input_df)[0])
        # Non-negative emissions constraint
        pred_val = max(0.0, round(pred_val, 2))
        predictions[name] = pred_val
        preds_list.append(pred_val)

    ensemble_mean = round(float(np.mean(preds_list)), 2)
    ensemble_std = round(float(np.std(preds_list)), 2)

    # Gradient Boosting is our primary model based on lowest test RMSE
    best_model_name = "Gradient Boosting Regressor"
    primary_prediction = predictions.get(best_model_name, ensemble_mean)

    return {
        "predictions_by_model": predictions,
        "primary_model": best_model_name,
        "primary_prediction_kg": primary_prediction,
        "ensemble_mean_kg": ensemble_mean,
        "ensemble_std_kg": ensemble_std,
        "ml_interval_lower_kg": max(0.0, round(ensemble_mean - 1.96 * ensemble_std, 2)),
        "ml_interval_upper_kg": round(ensemble_mean + 1.96 * ensemble_std, 2),
    }


def get_model_comparison_table() -> pd.DataFrame:
    """Format model metrics into a neat DataFrame for UI presentation."""
    metrics_data = load_metrics()["metrics"]
    records = []
    for model_name, vals in metrics_data.items():
        records.append({
            "Model": model_name,
            "Test MAE (kg CO₂e)": vals["test_mae"],
            "Test RMSE (kg CO₂e)": vals["test_rmse"],
            "Test R² Score": vals["test_r2"],
            "Train R² Score": vals["train_r2"],
            "Generalization Gap (R²)": round(vals["train_r2"] - vals["test_r2"], 4),
        })
    df = pd.DataFrame(records)
    # Sort best R2 first
    df = df.sort_values(by="Test R² Score", ascending=False).reset_index(drop=True)
    return df


def get_feature_importances() -> pd.DataFrame:
    """Retrieve top feature importances from Gradient Boosting model."""
    data = load_metrics().get("feature_importances", [])
    df = pd.DataFrame(data)
    # Clean up display names
    name_cleaner = {
        "distance_km": "Distance Travelled (km)",
        "electricity_kwh": "Electricity Consumed (kWh)",
        "household_size": "Household Size",
        "occupancy": "Vehicle Occupancy",
        "solar_percentage": "Solar Share (%)",
        "waste_kg": "Waste Generated (kg)",
        "extra_dairy_servings": "Extra Dairy Servings",
        "compost_percentage": "Composting Share (%)",
        "recycle_percentage": "Recycling Share (%)",
        "transport_mode_petrol_car": "Mode: Petrol Car",
        "transport_mode_diesel_car": "Mode: Diesel Car",
        "transport_mode_petrol_two_wheeler": "Mode: Petrol 2-Wheeler",
        "transport_mode_public_bus": "Mode: Public Bus",
        "transport_mode_metro_rail": "Mode: Metro Rail",
        "transport_mode_ev_two_wheeler": "Mode: EV 2-Wheeler",
        "transport_mode_ev_car": "Mode: EV Car",
        "transport_mode_cycling_walking": "Mode: Cycling/Walking",
        "transport_mode_cng_car_auto": "Mode: CNG Auto/Car",
        "diet_type_nonveg_heavy": "Diet: Heavy Non-Veg",
        "diet_type_nonveg_poultry": "Diet: Non-Veg Poultry",
        "diet_type_vegetarian": "Diet: Vegetarian",
        "diet_type_eggetarian": "Diet: Eggetarian",
        "diet_type_vegan": "Diet: Vegan",
    }
    df["feature_clean"] = df["feature"].map(lambda x: name_cleaner.get(x, x))
    return df


def get_parity_plot_data() -> Dict[str, Any]:
    """Retrieve test sample ground truth and predictions for parity scatter plots."""
    return load_metrics().get("test_sample", {})
