from pathlib import Path
import csv
from statistics import mean

from flask import Flask, render_template


app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "phagwara_final.csv"


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def normalize(value, min_value, max_value):
    if max_value == min_value:
        return 0.0
    return (value - min_value) / (max_value - min_value)


def detect_disaster(rainfall, water_level, discharge):
    if rainfall > 180 or water_level > 7 or discharge > 3000:
        return "Flood"
    if rainfall > 100 or water_level > 4.5 or discharge > 1800:
        return "Heavy Rain / Waterlogging"
    return "Low Hazard"


def infrastructure_type(index):
    types = ["Hospital", "Bridge", "Road", "Data Center"]
    return types[index % len(types)]


def infrastructure_name(kind, index):
    return f"{kind} Unit {index + 1}"


def load_phagwara_data():
    rows = []
    with DATA_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        raw_rows = list(reader)

    rain_values = [float(row["Rainfall (mm)"]) for row in raw_rows]
    water_values = [float(row["Water Level (m)"]) for row in raw_rows]
    discharge_values = [float(row["River Discharge (mł/s)"]) for row in raw_rows]
    population_values = [float(row["Population Density"]) for row in raw_rows]
    risk_values = [float(row["Risk_Score"]) for row in raw_rows]

    for index, item in enumerate(raw_rows):
        rainfall = float(item["Rainfall (mm)"])
        water_level = float(item["Water Level (m)"])
        river_discharge = float(item["River Discharge (mł/s)"])
        population_density = float(item["Population Density"])
        source_risk_score = float(item["Risk_Score"])

        kind = infrastructure_type(index)
        structural_factor = 0.30 + (index % 5) * 0.15
        design_age_factor = 0.20 + (index % 7) * 0.10

        hazard_exposure = (
            normalize(rainfall, min(rain_values), max(rain_values)) * 0.40
            + normalize(water_level, min(water_values), max(water_values)) * 0.30
            + normalize(river_discharge, min(discharge_values), max(discharge_values)) * 0.30
        )
        proximity_factor = (
            normalize(population_density, min(population_values), max(population_values)) * 0.60
            + normalize(source_risk_score, min(risk_values), max(risk_values)) * 0.40
        )

        # IVI = structural factors + hazard exposure + design age + proximity factors
        ivi_score = (
            structural_factor * 25
            + hazard_exposure * 35
            + design_age_factor * 20
            + proximity_factor * 20
        )
        ivi_score = round(ivi_score, 2)

        disruption_probability = round(clamp(ivi_score / 100), 2)

        if ivi_score >= 70:
            safety_status = "Not Safe"
            alert_level = "High"
        elif ivi_score >= 45:
            safety_status = "Use With Caution"
            alert_level = "Medium"
        else:
            safety_status = "Safe"
            alert_level = "Low"

        row = {
            "name": infrastructure_name(kind, index),
            "type": kind,
            "location": item["Location"],
            "latitude": round(float(item["Latitude"]), 4),
            "longitude": round(float(item["Longitude"]), 4),
            "rainfall": round(rainfall, 2),
            "water_level": round(water_level, 2),
            "river_discharge": round(river_discharge, 2),
            "population_density": round(population_density, 2),
            "source_risk_score": round(source_risk_score, 2),
            "disaster_type": detect_disaster(rainfall, water_level, river_discharge),
            "structural_factor": round(structural_factor, 2),
            "hazard_exposure": round(hazard_exposure, 2),
            "design_age_factor": round(design_age_factor, 2),
            "proximity_factor": round(proximity_factor, 2),
            "ivi_score": ivi_score,
            "disruption_probability": disruption_probability,
            "safety_status": safety_status,
            "alert_level": alert_level,
        }
        rows.append(row)

    return rows


def build_summary(rows):
    ivi_values = [row["ivi_score"] for row in rows]
    disruption_values = [row["disruption_probability"] for row in rows]
    safe_count = sum(1 for row in rows if row["safety_status"] == "Safe")
    caution_count = sum(1 for row in rows if row["safety_status"] == "Use With Caution")
    unsafe_count = sum(1 for row in rows if row["safety_status"] == "Not Safe")

    disaster_counts = {}
    type_counts = {}
    for row in rows:
        disaster_counts[row["disaster_type"]] = disaster_counts.get(row["disaster_type"], 0) + 1
        type_counts[row["type"]] = type_counts.get(row["type"], 0) + 1

    detected_disaster = max(disaster_counts, key=disaster_counts.get)

    return {
        "total_units": len(rows),
        "safe_count": safe_count,
        "caution_count": caution_count,
        "unsafe_count": unsafe_count,
        "avg_ivi": round(mean(ivi_values), 2),
        "max_ivi": round(max(ivi_values), 2),
        "avg_disruption": round(mean(disruption_values) * 100, 2),
        "detected_disaster": detected_disaster,
        "type_counts": type_counts,
    }


@app.route("/")
def home():
    rows = load_phagwara_data()
    summary = build_summary(rows)
    top_unsafe = sorted(rows, key=lambda item: item["ivi_score"], reverse=True)[:5]
    safe_units = sorted(
        [row for row in rows if row["safety_status"] == "Safe"],
        key=lambda item: item["ivi_score"]
    )[:5]
    return render_template(
        "index.html",
        summary=summary,
        rows=rows,
        top_unsafe=top_unsafe,
        safe_units=safe_units,
    )


if __name__ == "__main__":
    app.run(debug=True)
