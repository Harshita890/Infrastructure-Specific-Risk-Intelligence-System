from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def build_flood_model() -> dict[str, object]:
    df = pd.read_csv(DATA_DIR / "clean_flood_risk_india.csv")
    feature_columns = [
        "latitude",
        "longitude",
        "rainfall_mm",
        "temperature_deg_c",
        "humidity_pct",
        "river_discharge_m3_s",
        "water_level_m",
        "elevation_m",
        "land_cover",
        "soil_type",
        "population_density",
        "infrastructure",
        "historical_floods",
    ]
    target_column = "flood_occurred"

    x = df[feature_columns].copy()
    y = df[target_column].copy()

    numeric_features = x.select_dtypes(include="number").columns.tolist()
    categorical_features = [column for column in feature_columns if column not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, solver="liblinear")),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "flood_risk_model.joblib", compress=3)
    return {
        "name": "Flood Risk Model",
        "target": "Flood Occurred",
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "features": feature_columns,
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }


def build_disaster_model() -> dict[str, object]:
    df = pd.read_csv(DATA_DIR / "clean_disaster_history.csv")
    feature_columns = [
        "start_year",
        "start_month",
        "magnitude",
        "latitude",
        "longitude",
        "total_deaths",
        "total_affected",
        "total_damage_usd_000",
    ]
    target_column = "disaster_type"

    x = df[feature_columns].copy()
    y = df[target_column].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), feature_columns),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    joblib.dump(model, MODELS_DIR / "disaster_type_model.joblib", compress=3)
    return {
        "name": "Disaster Type Model",
        "target": "Disaster Type",
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "features": feature_columns,
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }


def build_building_model() -> dict[str, object]:
    df = pd.read_csv(DATA_DIR / "building_safety_assessment.csv")
    feature_columns = [
        "latitude",
        "longitude",
        "rainfall_mm",
        "water_level_m",
        "soil_type",
        "building_age_years",
        "floors",
        "building_height_m",
        "distance_to_river_km",
        "construction_material",
        "foundation_type",
        "drainage_quality",
        "soil_bearing_capacity_kpa",
        "maintenance_score",
        "occupancy_load",
    ]
    target_column = "building_condition"

    x = df[feature_columns].copy()
    y = df[target_column].copy()

    numeric_features = x.select_dtypes(include="number").columns.tolist()
    categorical_features = [column for column in feature_columns if column not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=160, random_state=42, class_weight="balanced")),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    joblib.dump(model, MODELS_DIR / "building_safety_model.joblib", compress=3)
    return {
        "name": "Building Safety Model",
        "target": "Building Condition",
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "features": feature_columns,
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "flood_model": build_flood_model(),
        "disaster_model": build_disaster_model(),
        "building_model": build_building_model(),
    }
    with (MODELS_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print("Saved trained models and metrics in:", MODELS_DIR)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
