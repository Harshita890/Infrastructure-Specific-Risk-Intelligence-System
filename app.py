from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

app = Flask(__name__)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def risk_label(probability: float) -> str:
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


def safe_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_infrastructure_risk(form_values: dict[str, object]) -> dict[str, float | str]:
    rainfall = min(max(safe_float(form_values["rainfall_mm"], 0.0), 0.0), 400.0) / 400.0
    water_level = min(max(safe_float(form_values["water_level_m"], 0.0), 0.0), 10.0) / 10.0
    discharge = min(max(safe_float(form_values["river_discharge_m3_s"], 0.0), 0.0), 6000.0) / 6000.0
    population = min(max(safe_float(form_values["population_density"], 0.0), 0.0), 10000.0) / 10000.0
    flood_history = min(max(safe_float(form_values["historical_floods"], 0.0), 0.0), 1.0)
    infrastructure = min(max(safe_float(form_values["infrastructure"], 0.0), 0.0), 1.0)
    elevation = 1 - (min(max(safe_float(form_values["elevation_m"], 0.0), 0.0), 9000.0) / 9000.0)

    score = (
        rainfall * 0.2
        + water_level * 0.24
        + discharge * 0.16
        + population * 0.14
        + flood_history * 0.14
        + infrastructure * 0.08
        + elevation * 0.04
    ) * 100
    score = round(score, 2)
    return {"score": score, "label": risk_label(score / 100.0)}


summary = load_json(DATA_DIR / "project_summary.json")
metrics = load_json(MODELS_DIR / "metrics.json")
phagwara_profile = pd.read_csv(DATA_DIR / "phagwara_profile.csv").iloc[0].to_dict()
punjab_damage = pd.read_csv(DATA_DIR / "punjab_damage_history.csv")
punjab_events = pd.read_csv(DATA_DIR / "punjab_disaster_history.csv")
phagwara_samples = pd.read_csv(DATA_DIR / "phagwara_nearby_flood_samples.csv")
infrastructure_points = pd.read_csv(DATA_DIR / "phagwara_infrastructure_risk.csv")
weather_forecast = pd.read_csv(DATA_DIR / "phagwara_weather_forecast.csv")

flood_model = joblib.load(MODELS_DIR / "flood_risk_model.joblib")
disaster_model = joblib.load(MODELS_DIR / "disaster_type_model.joblib")


@app.route("/")
def index():
    feature_cards = [
        {
            "title": "Natural Disaster Predictor",
            "text": "Check disaster type, flood chance, and local risk.",
            "route": "predict",
        },
        {
            "title": "Weather Forecast",
            "text": "Simple weekly view for Phagwara.",
            "route": "weather",
        },
        {
            "title": "Infrastructure Risk Map",
            "text": "OpenStreetMap view of important locations.",
            "route": "infrastructure_map",
        },
    ]
    top_events = (
        punjab_events["disaster_type"]
        .value_counts()
        .rename_axis("disaster_type")
        .reset_index(name="count")
    )
    return render_template(
        "index.html",
        summary=summary,
        metrics=metrics,
        profile=phagwara_profile,
        feature_cards=feature_cards,
        top_events=top_events.head(5).to_dict(orient="records"),
        infrastructure_points=infrastructure_points.head(4).to_dict(orient="records"),
    )


@app.route("/about")
def about():
    return render_template(
        "about.html",
        summary=summary,
        metrics=metrics,
        profile=phagwara_profile,
    )


@app.route("/weather")
def weather():
    return render_template(
        "weather.html",
        forecast_rows=weather_forecast.to_dict(orient="records"),
        profile=phagwara_profile,
        summary=summary,
    )


@app.route("/infrastructure-map")
def infrastructure_map():
    points = infrastructure_points.round(2).to_dict(orient="records")
    average_score = round(float(infrastructure_points["infrastructure_risk_score"].mean()), 2)
    highest_score = round(float(infrastructure_points["infrastructure_risk_score"].max()), 2)
    return render_template(
        "infrastructure_map.html",
        profile=phagwara_profile,
        points=points,
        average_score=average_score,
        highest_score=highest_score,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    defaults = {
        "latitude": phagwara_profile["latitude"],
        "longitude": phagwara_profile["longitude"],
        "rainfall_mm": phagwara_profile["rainfall_mm"],
        "temperature_deg_c": phagwara_profile["temperature_deg_c"],
        "humidity_pct": phagwara_profile["humidity_pct"],
        "river_discharge_m3_s": phagwara_profile["river_discharge_m3_s"],
        "water_level_m": phagwara_profile["water_level_m"],
        "elevation_m": phagwara_profile["elevation_m"],
        "land_cover": phagwara_profile["land_cover"],
        "soil_type": phagwara_profile["soil_type"],
        "population_density": phagwara_profile["population_density"],
        "infrastructure": phagwara_profile["infrastructure"],
        "historical_floods": phagwara_profile["historical_floods"],
        "start_year": 2026,
        "start_month": 7,
        "magnitude": 4.0,
        "total_deaths": 10.0,
        "total_affected": 1200.0,
        "total_damage_usd_000": 500.0,
    }

    result = None
    infra_result = compute_infrastructure_risk(defaults)
    form_values = defaults.copy()

    if request.method == "POST":
        for key, default in defaults.items():
            form_values[key] = request.form.get(key, default)

        flood_input = pd.DataFrame(
            [
                {
                    "latitude": safe_float(form_values["latitude"], defaults["latitude"]),
                    "longitude": safe_float(form_values["longitude"], defaults["longitude"]),
                    "rainfall_mm": safe_float(form_values["rainfall_mm"], defaults["rainfall_mm"]),
                    "temperature_deg_c": safe_float(form_values["temperature_deg_c"], defaults["temperature_deg_c"]),
                    "humidity_pct": safe_float(form_values["humidity_pct"], defaults["humidity_pct"]),
                    "river_discharge_m3_s": safe_float(form_values["river_discharge_m3_s"], defaults["river_discharge_m3_s"]),
                    "water_level_m": safe_float(form_values["water_level_m"], defaults["water_level_m"]),
                    "elevation_m": safe_float(form_values["elevation_m"], defaults["elevation_m"]),
                    "land_cover": request.form.get("land_cover", str(defaults["land_cover"])),
                    "soil_type": request.form.get("soil_type", str(defaults["soil_type"])),
                    "population_density": safe_float(form_values["population_density"], defaults["population_density"]),
                    "infrastructure": int(safe_float(form_values["infrastructure"], defaults["infrastructure"])),
                    "historical_floods": int(safe_float(form_values["historical_floods"], defaults["historical_floods"])),
                }
            ]
        )
        disaster_input = pd.DataFrame(
            [
                {
                    "start_year": safe_float(form_values["start_year"], defaults["start_year"]),
                    "start_month": safe_float(form_values["start_month"], defaults["start_month"]),
                    "magnitude": safe_float(form_values["magnitude"], defaults["magnitude"]),
                    "latitude": safe_float(form_values["latitude"], defaults["latitude"]),
                    "longitude": safe_float(form_values["longitude"], defaults["longitude"]),
                    "total_deaths": safe_float(form_values["total_deaths"], defaults["total_deaths"]),
                    "total_affected": safe_float(form_values["total_affected"], defaults["total_affected"]),
                    "total_damage_usd_000": safe_float(form_values["total_damage_usd_000"], defaults["total_damage_usd_000"]),
                }
            ]
        )

        flood_probability = float(flood_model.predict_proba(flood_input)[0][1])
        disaster_prediction = disaster_model.predict(disaster_input)[0]
        infra_result = compute_infrastructure_risk(form_values)
        result = {
            "flood_probability": round(flood_probability * 100, 2),
            "flood_risk": risk_label(flood_probability),
            "disaster_prediction": disaster_prediction,
        }

    return render_template(
        "predict.html",
        profile=phagwara_profile,
        metrics=metrics,
        form_values=form_values,
        result=result,
        infra_result=infra_result,
        land_cover_options=sorted(phagwara_samples["land_cover"].dropna().unique().tolist()),
        soil_type_options=sorted(phagwara_samples["soil_type"].dropna().unique().tolist()),
    )


if __name__ == "__main__":
    app.run(debug=True)
