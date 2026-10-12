"""
train_model.py
==============
Script to:
1. Generate a realistic daily activity benchmark dataset based on Indian lifestyles
   and scientific IPCC/CEA/ARAI emission factors.
2. Train and evaluate three regression models:
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
3. Evaluate models on unseen test data using MAE, RMSE, and R2 (with zero data leakage).
4. Save trained models, pipeline transformers, and evaluation metrics for deployment.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from emissions import calculate_total_emissions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def generate_benchmark_dataset(num_samples: int = 2500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic benchmark dataset representing diverse Indian daily activity profiles.
    Target emissions are calculated using the scientific LCA formulas with realistic
    behavioral, driving, and environmental variance (congestion, seasonal AC variation,
    food preparation differences, waste moisture).

    Note: As specified in research gap guidelines, this synthetic dataset is generated
    from verified CEA/ARAI/CPCB factors and explicitly documented as a benchmark.
    """
    np.random.seed(random_seed)

    transport_modes = [
        "petrol_two_wheeler", "petrol_car", "diesel_car", "cng_car_auto",
        "ev_two_wheeler", "ev_car", "public_bus", "metro_rail", "cycling_walking"
    ]
    diet_types = ["vegan", "vegetarian", "eggetarian", "nonveg_poultry", "nonveg_heavy"]

    # Realistic lifestyle probability distributions in urban/semi-urban India
    mode_probs = [0.30, 0.18, 0.12, 0.10, 0.05, 0.03, 0.12, 0.07, 0.03]
    diet_probs = [0.03, 0.45, 0.15, 0.30, 0.07]

    records = []

    for _ in range(num_samples):
        mode = np.random.choice(transport_modes, p=mode_probs)
        diet = np.random.choice(diet_types, p=diet_probs)

        # Distance distribution: multimodal (short local trips vs office commuters vs non-travelers)
        trip_type = np.random.choice(["local", "commuter", "long_distance", "wfh"], p=[0.25, 0.55, 0.15, 0.05])
        if trip_type == "wfh":
            distance_km = 0.0
        elif trip_type == "local":
            distance_km = float(np.round(np.random.uniform(1.0, 6.0), 1))
        elif trip_type == "commuter":
            distance_km = float(np.round(np.random.uniform(8.0, 35.0), 1))
        else:
            distance_km = float(np.round(np.random.uniform(35.0, 75.0), 1))

        if mode in ["petrol_car", "diesel_car", "cng_car_auto", "ev_car"]:
            occupancy = int(np.random.choice([1, 2, 3, 4], p=[0.65, 0.22, 0.09, 0.04]))
        else:
            occupancy = 1

        # Household size
        household_size = int(np.random.choice([1, 2, 3, 4, 5, 6], p=[0.10, 0.18, 0.28, 0.30, 0.10, 0.04]))

        # Daily electricity per household (kWh/day)
        # 1-3 kWh for small household up to 15-25 kWh for high-AC summer usage
        base_elec_per_person = np.random.uniform(1.2, 5.0)
        electricity_kwh = float(np.round(household_size * base_elec_per_person + np.random.normal(0, 0.8), 2))
        electricity_kwh = max(0.5, electricity_kwh)

        # Solar penetration (emerging in India)
        has_solar = np.random.choice([True, False], p=[0.15, 0.85])
        solar_percentage = float(np.round(np.random.uniform(30.0, 90.0), 1)) if has_solar else 0.0

        # Food dairy consumption
        if diet == "vegan":
            dairy_servings = 0
        else:
            dairy_servings = int(np.random.choice([0, 1, 2, 3, 4], p=[0.10, 0.40, 0.35, 0.12, 0.03]))

        # Waste generation per capita (0.25 - 0.90 kg/day)
        waste_kg = float(np.round(np.random.uniform(0.25, 0.95), 2))

        # Waste segregation & composting
        eco_conscious = np.random.choice([True, False], p=[0.20, 0.80])
        if eco_conscious:
            compost_pct = float(np.round(np.random.uniform(30.0, 80.0), 1))
            recycle_pct = float(np.round(np.random.uniform(20.0, 40.0), 1))
        else:
            compost_pct = float(np.round(np.random.uniform(0.0, 15.0), 1)) if np.random.rand() < 0.15 else 0.0
            recycle_pct = float(np.round(np.random.uniform(0.0, 20.0), 1)) if np.random.rand() < 0.25 else 0.0

        # Calculate exact baseline emissions via scientific LCA engine
        base_calc = calculate_total_emissions(
            transport_mode=mode,
            distance_km=distance_km,
            occupancy=occupancy,
            electricity_kwh=electricity_kwh,
            solar_percentage=solar_percentage,
            household_size=household_size,
            diet_type=diet,
            extra_dairy_servings=dairy_servings,
            waste_kg=waste_kg,
            compost_percentage=compost_pct,
            recycle_percentage=recycle_pct,
            include_cooking_lpg=True,
        )

        # Real-world stochastic variations (traffic idling, AC seasonal COP, waste moisture, food cooking style)
        # Simulates real-world sensor variance around standard laboratory factors
        transport_var = np.random.normal(1.0, 0.06) if distance_km > 0 else 1.0
        elec_var = np.random.normal(1.0, 0.04)
        food_var = np.random.normal(1.0, 0.05)
        waste_var = np.random.normal(1.0, 0.04)

        t_emissions = max(0.0, base_calc["breakdown"]["Transportation"] * transport_var)
        e_emissions = max(0.0, base_calc["breakdown"]["Electricity & Cooking"] * elec_var)
        f_emissions = max(0.0, base_calc["breakdown"]["Food & Diet"] * food_var)
        w_emissions = max(0.0, base_calc["breakdown"]["Waste Management"] * waste_var)

        total_noisy = round(t_emissions + e_emissions + f_emissions + w_emissions, 3)

        records.append({
            "transport_mode": mode,
            "distance_km": distance_km,
            "occupancy": occupancy,
            "electricity_kwh": electricity_kwh,
            "solar_percentage": solar_percentage,
            "household_size": household_size,
            "diet_type": diet,
            "extra_dairy_servings": dairy_servings,
            "waste_kg": waste_kg,
            "compost_percentage": compost_pct,
            "recycle_percentage": recycle_pct,
            "transport_emissions_kg": round(t_emissions, 3),
            "electricity_emissions_kg": round(e_emissions, 3),
            "food_emissions_kg": round(f_emissions, 3),
            "waste_emissions_kg": round(w_emissions, 3),
            "total_emissions_kg": total_noisy,
        })

    df = pd.DataFrame(records)
    return df


def train_and_evaluate_models():
    """
    Train Linear Regression, Random Forest, and Gradient Boosting Regressors.
    Evaluate metrics on held-out test split and persist models and evaluation logs.
    """
    print("Generating benchmark daily activities dataset...")
    df = generate_benchmark_dataset(num_samples=2500, random_seed=42)

    csv_path = os.path.join(DATA_DIR, "daily_activities_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved benchmark dataset with {len(df)} rows to: {csv_path}")

    # Feature definitions
    feature_cols = [
        "transport_mode", "distance_km", "occupancy", "electricity_kwh",
        "solar_percentage", "household_size", "diet_type", "extra_dairy_servings",
        "waste_kg", "compost_percentage", "recycle_percentage"
    ]
    target_col = "total_emissions_kg"

    X = df[feature_cols]
    y = df[target_col]

    # Clean 80/20 train-test split avoiding data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Dataset split: Train shape = {X_train.shape}, Test shape = {X_test.shape}")

    cat_cols = ["transport_mode", "diet_type"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Preprocessing transformer (fitted ONLY on X_train)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore"), cat_cols),
        ]
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42
        ),
    }

    metrics_results = {}
    fitted_pipelines = {}
    test_predictions = {}

    for name, model in models.items():
        print(f"Training {name}...")
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model),
        ])

        pipe.fit(X_train, y_train)
        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)

        train_mae = float(mean_absolute_error(y_train, y_pred_train))
        test_mae = float(mean_absolute_error(y_test, y_pred_test))
        train_rmse = float(root_mean_squared_error(y_train, y_pred_train))
        test_rmse = float(root_mean_squared_error(y_test, y_pred_test))
        train_r2 = float(r2_score(y_train, y_pred_train))
        test_r2 = float(r2_score(y_test, y_pred_test))

        metrics_results[name] = {
            "test_mae": round(test_mae, 4),
            "test_rmse": round(test_rmse, 4),
            "test_r2": round(test_r2, 4),
            "train_mae": round(train_mae, 4),
            "train_rmse": round(train_rmse, 4),
            "train_r2": round(train_r2, 4),
        }
        fitted_pipelines[name] = pipe
        test_predictions[name] = y_pred_test.tolist()

        print(f"  [{name}] Test MAE: {test_mae:.4f} | Test RMSE: {test_rmse:.4f} | Test R2: {test_r2:.4f}")

    # Feature Importances from Tree Ensemble (Gradient Boosting)
    gb_pipe = fitted_pipelines["Gradient Boosting Regressor"]
    fitted_preprocessor = gb_pipe.named_steps["preprocessor"]
    fitted_gb = gb_pipe.named_steps["regressor"]

    # Extract transformed feature names
    cat_encoder = fitted_preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + cat_feature_names
    importances = fitted_gb.feature_importances_

    feature_importance_list = [
        {"feature": feat, "importance": round(float(imp), 4)}
        for feat, imp in sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)
    ]

    # Save trained pipelines
    models_file = os.path.join(MODELS_DIR, "trained_models.joblib")
    joblib.dump(fitted_pipelines, models_file)
    print(f"Saved trained models to: {models_file}")

    # Save evaluation summary and test sample for parity plot
    evaluation_payload = {
        "metrics": metrics_results,
        "feature_importances": feature_importance_list[:15],
        "test_sample": {
            "y_true": y_test.iloc[:100].tolist(),
            "preds": {name: test_predictions[name][:100] for name in models.keys()},
        },
        "feature_names": feature_cols,
        "num_train_samples": len(X_train),
        "num_test_samples": len(X_test),
    }

    eval_file = os.path.join(MODELS_DIR, "model_metrics.json")
    with open(eval_file, "w") as f:
        json.dump(evaluation_payload, f, indent=2)
    print(f"Saved evaluation metrics to: {eval_file}")

    return metrics_results


if __name__ == "__main__":
    train_and_evaluate_models()
