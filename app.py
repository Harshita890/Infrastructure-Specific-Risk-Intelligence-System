from __future__ import annotations

import json
import math
from pathlib import Path
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import joblib
import pandas as pd
from flask import Flask, jsonify, make_response, render_template, request


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

app = Flask(__name__)
HTTP_HEADERS = {"User-Agent": "PhagwaraResilienceAtlas/1.0 (student project)"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def risk_label(probability: float) -> str:
    if probability >= 0.7:
        return "Critical"
    if probability >= 0.4:
        return "Medium"
    return "Low"


def fetch_json(url: str) -> dict | list:
    request_obj = Request(url, headers=HTTP_HEADERS)
    with urlopen(request_obj, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def weather_code_label(code: int | None) -> str:
    labels = {
        0: "Clear",
        1: "Mostly Clear",
        2: "Partly Cloudy",
        3: "Cloudy",
        45: "Fog",
        48: "Fog",
        51: "Light Drizzle",
        53: "Drizzle",
        55: "Heavy Drizzle",
        61: "Light Rain",
        63: "Rain",
        65: "Heavy Rain",
        66: "Freezing Rain",
        67: "Freezing Rain",
        71: "Light Snow",
        73: "Snow",
        75: "Heavy Snow",
        80: "Rain Showers",
        81: "Showers",
        82: "Heavy Showers",
        95: "Thunderstorm",
        96: "Thunderstorm",
        99: "Thunderstorm",
    }
    return labels.get(code, "Weather")


def get_live_weather(latitude: float, longitude: float) -> dict:
    params = urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": 7,
            "timezone": "auto",
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{params}"
    payload = fetch_json(url)
    current = payload.get("current", {})
    daily = payload.get("daily", {})
    daily_rows: list[dict[str, object]] = []
    for i, day in enumerate(daily.get("time", [])):
        daily_rows.append(
            {
                "day": day,
                "condition": weather_code_label(daily.get("weather_code", [None])[i]),
                "temp_max": daily.get("temperature_2m_max", [None])[i],
                "temp_min": daily.get("temperature_2m_min", [None])[i],
                "rain_chance": daily.get("precipitation_probability_max", [None])[i],
            }
        )
    return {
        "location": {"latitude": latitude, "longitude": longitude},
        "current": {
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "wind_speed": current.get("wind_speed_10m"),
            "condition": weather_code_label(current.get("weather_code")),
            "time": current.get("time"),
        },
        "daily": daily_rows,
        "source": "Live weather",
    }


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_infrastructure_risk(form_values: dict[str, object]) -> dict[str, float | str]:
    rainfall = min(max(safe_float(form_values["rainfall_mm"], 0.0), 0.0), 400.0) / 400.0
    water_level = min(max(safe_float(form_values["water_level_m"], 0.0), 0.0), 10.0) / 10.0
    discharge = min(max(safe_float(form_values["river_discharge_m3_s"], 0.0), 0.0), 6000.0) / 6000.0
    population = min(max(safe_float(form_values["population_density"], 0.0), 0.0), 10000.0) / 10000.0
    flood_history = min(max(safe_float(form_values["historical_floods"], 0.0), 0.0), 1.0)
    infrastructure = min(max(safe_float(form_values["infrastructure"], 0.0), 0.0), 1.0)
    elevation = 1 - (min(max(safe_float(form_values["elevation_m"], 0.0), 0.0), 9000.0) / 9000.0)
    temperature = (min(max(safe_float(form_values["temperature_deg_c"], 12.0), 12.0), 48.0) - 12.0) / 36.0
    building_age = min(max(safe_float(form_values.get("building_age_years"), 15.0), 0.0), 120.0) / 120.0

    score = (
        rainfall * 0.18
        + water_level * 0.22
        + discharge * 0.14
        + population * 0.12
        + flood_history * 0.12
        + infrastructure * 0.08
        + elevation * 0.04
        + temperature * 0.06
        + building_age * 0.04
    ) * 100
    score = round(score, 2)
    return {"score": score, "label": risk_label(score / 100.0)}


def nearest_flood_context(latitude: float, longitude: float) -> dict:
    samples = clean_flood_risk.copy()
    samples["distance_km"] = samples.apply(
        lambda row: haversine_km(latitude, longitude, row["latitude"], row["longitude"]),
        axis=1,
    )
    nearest = samples.nsmallest(1, "distance_km").iloc[0]
    return {
        "zone_name": str(nearest["zone_name"]),
        "latitude": round(latitude, 6),
        "longitude": round(longitude, 6),
        "rainfall_mm": round(float(nearest["rainfall_mm"]), 2),
        "temperature_deg_c": round(float(nearest["temperature_deg_c"]), 2),
        "humidity_pct": round(float(nearest["humidity_pct"]), 2),
        "river_discharge_m3_s": round(float(nearest["river_discharge_m3_s"]), 2),
        "water_level_m": round(float(nearest["water_level_m"]), 2),
        "elevation_m": round(float(nearest["elevation_m"]), 2),
        "land_cover": str(nearest["land_cover"]),
        "soil_type": str(nearest["soil_type"]),
        "population_density": round(float(nearest["population_density"]), 2),
        "infrastructure": int(nearest["infrastructure"]),
        "historical_floods": int(nearest["historical_floods"]),
        "nearest_dataset_distance_km": round(float(nearest["distance_km"]), 2),
    }


def nearest_hotspot(latitude: float, longitude: float) -> dict:
    points = disaster_hotspots.copy()
    points["distance_km"] = points.apply(
        lambda row: haversine_km(latitude, longitude, row["latitude"], row["longitude"]),
        axis=1,
    )
    nearest = points.nsmallest(1, "distance_km").iloc[0]
    return {
        "dominant_hazard": str(nearest["dominant_hazard"]),
        "alert_level": str(nearest["alert_level"]),
        "alert_score": round(float(nearest["alert_score"]), 2),
        "response_priority": str(nearest["response_priority"]),
        "hotspot_zone": str(nearest["zone_name"]),
        "hotspot_distance_km": round(float(nearest["distance_km"]), 2),
    }


def nearest_asset(latitude: float, longitude: float, preferred_name: str | None = None) -> dict:
    assets = infrastructure_points.copy()
    if preferred_name:
        exact = assets[assets["asset_name"].str.lower() == preferred_name.strip().lower()]
        if not exact.empty:
            chosen = exact.iloc[0].copy()
            chosen["asset_distance_km"] = haversine_km(latitude, longitude, chosen["latitude"], chosen["longitude"])
            return chosen.to_dict()
    assets["asset_distance_km"] = assets.apply(
        lambda row: haversine_km(latitude, longitude, row["latitude"], row["longitude"]),
        axis=1,
    )
    return assets.nsmallest(1, "asset_distance_km").iloc[0].to_dict()


def recommendation_reason(asset: pd.Series | dict, distance_km: float, safest: bool) -> str:
    score = safe_float(asset.get("infrastructure_risk_score"), 0.0)
    age = int(round(safe_float(asset.get("building_age_years"), 0.0)))
    floors = int(round(safe_float(asset.get("floors"), 0.0)))
    asset_type = str(asset.get("asset_type", "Building"))
    base = (
        f"Low nearby risk score ({score}/100), {distance_km:.2f} km away, and useful {asset_type.lower()} access."
        if safest
        else f"High nearby risk score ({score}/100), only {distance_km:.2f} km away, so it needs extra caution."
    )
    if safest and age <= 18:
        return base + f" Its newer {age}-year profile supports it as a safer fallback."
    if safest and floors >= 8:
        return base + f" Its {floors}-floor capacity may help during relocation planning."
    if not safest and age >= 20:
        return base + f" Its older {age}-year structure increases concern."
    if not safest and floors >= 8:
        return base + f" A taller {floors}-floor structure can raise exposure and evacuation pressure."
    return base


def nearby_recommendations(latitude: float, longitude: float, selected_asset_name: str) -> dict[str, list[dict[str, object]]]:
    assets = infrastructure_points.copy()
    assets["distance_km"] = assets.apply(
        lambda row: haversine_km(latitude, longitude, row["latitude"], row["longitude"]),
        axis=1,
    )
    nearby = assets[
        (assets["distance_km"] <= 2.0)
        & (assets["asset_name"].str.lower() != selected_asset_name.strip().lower())
    ].copy()

    if nearby.empty:
        nearby = assets[assets["asset_name"].str.lower() != selected_asset_name.strip().lower()].copy()

    safest_rows = nearby.sort_values(["infrastructure_risk_score", "distance_km"]).head(4)
    danger_rows = nearby.sort_values(["infrastructure_risk_score", "distance_km"], ascending=[False, True]).head(4)

    def pack(rows: pd.DataFrame, safest: bool) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for _, row in rows.iterrows():
            dist = round(float(row["distance_km"]), 2)
            items.append(
                {
                    "asset_name": str(row["asset_name"]),
                    "asset_type": str(row["asset_type"]),
                    "distance_km": dist,
                    "risk_score": round(float(row["infrastructure_risk_score"]), 2),
                    "risk_label": str(row["infrastructure_risk_label"]),
                    "campus_area": str(row.get("campus_area", "")),
                    "reason": recommendation_reason(row, dist, safest),
                }
            )
        return items

    return {
        "safe_buildings": pack(safest_rows, True),
        "danger_buildings": pack(danger_rows, False),
    }


def describe_weather(context: dict[str, object]) -> str:
    rainfall = safe_float(context.get("rainfall_mm"), 0.0)
    temperature = safe_float(context.get("temperature_deg_c"), 0.0)
    humidity = safe_float(context.get("humidity_pct"), 0.0)
    if rainfall >= 170 or humidity >= 84:
        return "Flood Watch"
    if temperature >= 41:
        return "Heat Stress"
    if rainfall >= 95:
        return "Storm Build-Up"
    if humidity <= 35:
        return "Dry Conditions"
    return "Stable"


def weather_scene(condition: str, temperature: float | None = None) -> str:
    text = (condition or "").lower()
    if "thunder" in text or "storm" in text:
        return "storm"
    if "rain" in text or "shower" in text or "drizzle" in text:
        return "rain"
    if temperature is not None and temperature >= 30:
        return "heat"
    if "heat" in text:
        return "heat"
    if "cloud" in text or "fog" in text:
        return "cloud"
    return "clear"


def local_place_match(query: str) -> dict | None:
    normalized = query.strip().lower()
    if not normalized:
        return None
    if "," in normalized:
        parts = [part.strip() for part in normalized.split(",")]
        if len(parts) == 2:
            try:
                return {
                    "display_name": f"Custom location ({query})",
                    "latitude": float(parts[0]),
                    "longitude": float(parts[1]),
                }
            except ValueError:
                pass
    matches = zone_directory[
        zone_directory["place_name"].str.lower().str.contains(normalized, na=False)
    ]
    if matches.empty:
        return None
    best = matches.iloc[0]
    return {
        "display_name": str(best["place_name"]),
        "latitude": float(best["latitude"]),
        "longitude": float(best["longitude"]),
    }


def search_location_by_name(query: str) -> dict | None:
    local_match = local_place_match(query)
    if local_match is not None:
        return local_match
    attempts = [query, f"{query}, Punjab, India"]
    for attempt in attempts:
        try:
            params = urlencode({"q": attempt, "format": "jsonv2", "limit": 1, "addressdetails": 1})
            url = f"https://nominatim.openstreetmap.org/search?{params}"
            results = fetch_json(url)
            if results:
                result = results[0]
                return {
                    "display_name": result.get("display_name", query),
                    "latitude": float(result.get("lat")),
                    "longitude": float(result.get("lon")),
                }
        except Exception:
            continue
    return None


def reverse_geocode(latitude: float, longitude: float) -> dict:
    places = zone_directory.copy()
    places["distance_km"] = places.apply(
        lambda row: haversine_km(latitude, longitude, row["latitude"], row["longitude"]),
        axis=1,
    )
    nearest = places.nsmallest(1, "distance_km").iloc[0]
    if float(nearest["distance_km"]) <= 3:
        return {
            "display_name": str(nearest["place_name"]),
            "latitude": latitude,
            "longitude": longitude,
        }
    params = urlencode({"lat": latitude, "lon": longitude, "format": "jsonv2"})
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?{params}"
        result = fetch_json(url)
        name = result.get("display_name", "Selected location")
    except Exception:
        name = f"Selected location ({latitude:.4f}, {longitude:.4f})"
    return {
        "display_name": name,
        "latitude": latitude,
        "longitude": longitude,
    }


def build_weather_payload(name: str, latitude: float, longitude: float) -> dict:
    try:
        payload = get_live_weather(latitude, longitude)
        payload["location"]["name"] = name
        current_temp = safe_float(payload["current"].get("temperature"), 0.0)
        payload["scene"] = weather_scene(str(payload["current"].get("condition")), current_temp)
        if current_temp >= 38:
            payload["heatwave_alert"] = "Heatwave warning for buildings in this area. Roof heating and indoor discomfort can rise quickly."
        if payload["scene"] in {"rain", "storm"}:
            payload["rain_alert"] = "Rain alert for buildings. Watch roof drainage, seepage points, and low-lying access routes."
        return payload
    except Exception:
        context = nearest_flood_context(latitude, longitude)
        day_rows: list[dict[str, object]] = []
        location_shift = (latitude - phagwara_profile["latitude"]) * 140 + (longitude - phagwara_profile["longitude"]) * 80
        for i, row in weather_forecast.iterrows():
            day_rows.append(
                {
                    "day": row["day"],
                    "condition": row["condition"],
                    "temp_max": round(float(row["temperature_deg_c"]) + 1.2 + location_shift * 0.05, 1),
                    "temp_min": round(float(row["temperature_deg_c"]) - 2.4 + location_shift * 0.03, 1),
                    "rain_chance": int(min(98, max(10, row["humidity_pct"] + context["historical_floods"] * 12))),
                }
            )
            if i == 6:
                break

        current_temp = round(float(context["temperature_deg_c"]) + location_shift * 0.08, 1)
        current_humidity = int(min(99, max(25, context["humidity_pct"])))
        current_precip = round(float(context["rainfall_mm"]) / 24, 2)
        current_wind = round(9 + abs(location_shift) * 0.5 + context["water_level_m"] * 0.9, 1)
        current_condition = describe_weather(context)

        payload = {
            "location": {"latitude": latitude, "longitude": longitude, "name": name},
            "current": {
                "temperature": current_temp,
                "humidity": current_humidity,
                "precipitation": current_precip,
                "wind_speed": current_wind,
                "condition": current_condition,
                "time": "Synthetic demo feed",
            },
            "daily": day_rows,
            "source": "Dummy dataset",
            "scene": weather_scene(current_condition, current_temp),
        }
        if current_temp >= 38:
            payload["heatwave_alert"] = "Heatwave warning for buildings in this area. Roof heating and indoor discomfort can rise quickly."
        if payload["scene"] in {"rain", "storm"}:
            payload["rain_alert"] = "Rain alert for buildings. Watch roof drainage, seepage points, and low-lying access routes."
        return payload


def build_location_context(name: str, latitude: float, longitude: float) -> dict:
    context = nearest_flood_context(latitude, longitude)
    context["name"] = name
    context["weather_condition"] = describe_weather(context)
    context["weather_time"] = "Synthetic demo feed"
    try:
        weather = get_live_weather(latitude, longitude)
        current = weather.get("current", {})
        if current.get("temperature") is not None:
            context["temperature_deg_c"] = round(float(current["temperature"]), 2)
        if current.get("humidity") is not None:
            context["humidity_pct"] = round(float(current["humidity"]), 2)
        if current.get("precipitation") is not None:
            context["rainfall_mm"] = round(float(current["precipitation"]) * 24, 2)
        context["weather_condition"] = str(current.get("condition") or context["weather_condition"])
        context["weather_time"] = str(current.get("time") or context["weather_time"])
    except Exception:
        pass
    asset = nearest_asset(latitude, longitude, name)
    context["asset_name"] = str(asset.get("asset_name", name))
    context["asset_type"] = str(asset.get("asset_type", "Infrastructure"))
    context["campus_area"] = str(asset.get("campus_area", "Phagwara Region"))
    context["building_age_years"] = int(round(safe_float(asset.get("building_age_years"), 15)))
    context["year_built"] = int(round(safe_float(asset.get("year_built"), datetime.now().year - context["building_age_years"])))
    context["floors"] = int(round(safe_float(asset.get("floors"), 3)))
    context["building_height_m"] = round(safe_float(asset.get("building_height_m"), 14.0), 2)
    context["nearest_asset_distance_km"] = round(safe_float(asset.get("asset_distance_km"), 0.0), 2)
    context["infrastructure_risk"] = compute_infrastructure_risk(context)
    context.update(nearest_hotspot(latitude, longitude))
    context.update(nearby_recommendations(latitude, longitude, context["asset_name"]))
    context["sos_triggered"] = bool(context["infrastructure_risk"]["score"] >= 80)
    return context


def scenario_guidance(disaster_type: str, flood_risk: str) -> str:
    if disaster_type == "Earthquake":
        return "Keep evacuation routes, hospital capacity, and communication nodes ready for seismic shocks."
    if disaster_type == "Flood":
        return "Prioritise drainage clearance, canal watch, and shelter preparation for low-lying zones."
    if disaster_type == "Heatwave":
        return "Focus on cooling centres, drinking water points, and heat alerts for dense neighbourhoods."
    if disaster_type == "Fire":
        return "Inspect electrical points, market corridors, and emergency access around crowded built-up areas."
    if disaster_type == "Storm":
        return "Prepare for wind damage, temporary waterlogging, and disruption near transport corridors."
    if flood_risk == "Critical":
        return "Flood-sensitive conditions are elevated, so field checks around canal-side and transport assets are recommended."
    return "Use this result as a demo scenario and compare it with the hotspot map for the nearest zone."


def building_condition(disaster_type: str, infra_score: float, flood_risk: str, weather_condition: str) -> dict[str, object]:
    issues: list[str] = []
    if disaster_type == "Earthquake":
        issues.extend(["foundation vibration", "wall cracking risk", "stairwell evacuation pressure"])
    elif disaster_type == "Flood":
        issues.extend(["basement waterlogging", "ground-floor seepage", "electrical panel exposure"])
    elif disaster_type == "Heatwave":
        issues.extend(["roof heat gain", "indoor overheating", "cooling load stress"])
    elif disaster_type == "Fire":
        issues.extend(["smoke spread", "electrical hazard", "rapid evacuation demand"])
    elif disaster_type == "Storm":
        issues.extend(["roof-sheet uplift", "window impact risk", "temporary drainage overflow"])
    elif disaster_type == "Industrial Accident":
        issues.extend(["toxic exposure concern", "service shutdown pressure", "restricted access"])

    if infra_score >= 80:
        severity = "SOS"
        issues.append("immediate emergency response required")
    elif infra_score >= 70:
        severity = "Critical"
        issues.append("rapid inspection required")
    elif flood_risk == "Critical":
        severity = "Critical"
        issues.append("high flood-sensitive building stress")
    elif weather_condition == "Heat Stress":
        severity = "Warning"
        issues.append("heat safety measures should be activated")
    else:
        severity = "Watch"

    return {
        "severity": severity,
        "summary": ", ".join(dict.fromkeys(issues[:4])),
    }


def emergency_alerts(points_df: pd.DataFrame) -> list[dict[str, object]]:
    alerts: list[dict[str, object]] = []
    for _, row in points_df.iterrows():
        score = float(row["infrastructure_risk_score"])
        if score >= 80:
            alerts.append(
                {
                    "asset_name": str(row["asset_name"]),
                    "asset_type": str(row["asset_type"]),
                    "score": round(score, 2),
                    "action": "SOS dispatch recommended",
                    "message": f"{row['asset_name']} is above 80 risk. Emergency team should be warned immediately.",
                }
            )
        elif score >= 70:
            alerts.append(
                {
                    "asset_name": str(row["asset_name"]),
                    "asset_type": str(row["asset_type"]),
                    "score": round(score, 2),
                    "action": "Critical warning",
                    "message": f"{row['asset_name']} needs urgent inspection and building safety review.",
                }
            )
    return alerts


def emergency_contacts() -> list[dict[str, str]]:
    return [
        {
            "name": "Phagwara Emergency Cell",
            "role": "Primary response desk",
            "phone": "+91 1824 500 101",
            "href": "tel:+911824500101",
        },
        {
            "name": "Punjab Disaster Support",
            "role": "District coordination",
            "phone": "+91 1070",
            "href": "tel:+911070",
        },
        {
            "name": "Building Safety Officer",
            "role": "Structural inspection contact",
            "phone": "+91 1824 500 202",
            "href": "tel:+911824500202",
        },
    ]


def prediction_defaults() -> dict[str, object]:
    return {
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
        "building_age_years": 18,
        "start_year": 2026,
        "start_month": 7,
        "magnitude": 4.8,
        "total_deaths": 8.0,
        "total_affected": 1800.0,
        "total_damage_usd_000": 2200.0,
    }


def coerce_form_values(raw_values: dict[str, object], defaults: dict[str, object]) -> dict[str, object]:
    values: dict[str, object] = {}
    for key, default in defaults.items():
        values[key] = raw_values.get(key, default)
    return values


def build_prediction_result(form_values: dict[str, object], defaults: dict[str, object]) -> tuple[dict[str, float | str], dict[str, object]]:
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
                "land_cover": str(form_values.get("land_cover", defaults["land_cover"])),
                "soil_type": str(form_values.get("soil_type", defaults["soil_type"])),
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
    disaster_prediction = str(disaster_model.predict(disaster_input)[0])
    infra_result = compute_infrastructure_risk(form_values)
    result = {
        "flood_probability": round(flood_probability * 100, 2),
        "flood_risk": risk_label(flood_probability),
        "disaster_prediction": disaster_prediction,
        "guidance": scenario_guidance(disaster_prediction, risk_label(flood_probability)),
    }
    result["building_condition"] = building_condition(
        disaster_prediction,
        infra_result["score"],
        result["flood_risk"],
        describe_weather(form_values),
    )
    return infra_result, result


def build_report_text(form_values: dict[str, object], infra_result: dict[str, float | str], result: dict[str, object]) -> str:
    lines = [
        "Phagwara and LPU Resilience Report",
        "Synthetic dummy dataset for demo use",
        "",
        f"Latitude: {safe_float(form_values['latitude'], 0.0):.6f}",
        f"Longitude: {safe_float(form_values['longitude'], 0.0):.6f}",
        f"Rainfall (mm): {safe_float(form_values['rainfall_mm'], 0.0):.2f}",
        f"Temperature (C): {safe_float(form_values['temperature_deg_c'], 0.0):.2f}",
        f"Water Level (m): {safe_float(form_values['water_level_m'], 0.0):.2f}",
        f"Population Density: {safe_float(form_values['population_density'], 0.0):.2f}",
        f"Building Age (years): {int(round(safe_float(form_values['building_age_years'], 0.0)))}",
        "",
        f"Infrastructure Score: {infra_result['score']}/100",
        f"Infrastructure Label: {infra_result['label']}",
        f"Predicted Disaster: {result['disaster_prediction']}",
        f"Flood Probability: {result['flood_probability']}%",
        f"Flood Risk: {result['flood_risk']}",
        f"Building Status: {result['building_condition']['severity']}",
        f"Building Stress Summary: {result['building_condition']['summary']}",
        "",
        f"Guidance: {result['guidance']}",
    ]
    return "\n".join(lines)


def heatwave_alert(profile: dict[str, object], weather_rows: pd.DataFrame) -> dict[str, object] | None:
    hottest = weather_rows.loc[weather_rows["temperature_deg_c"].idxmax()]
    if float(hottest["temperature_deg_c"]) >= 40:
        return {
            "day": str(hottest["day"]),
            "temperature": round(float(hottest["temperature_deg_c"]), 1),
            "message": "Heatwave warning for exposed buildings. Cooling, water access, and ventilation checks should be activated.",
        }
    if float(profile["temperature_deg_c"]) >= 39:
        return {
            "day": "Current",
            "temperature": round(float(profile["temperature_deg_c"]), 1),
            "message": "Heat stress is already high for dense built-up areas and top floors.",
        }
    heat_row = weather_rows[weather_rows["condition"].astype(str).str.contains("Heat Stress", case=False, na=False)]
    if not heat_row.empty:
        row = heat_row.iloc[0]
        return {
            "day": str(row["day"]),
            "temperature": round(float(row["temperature_deg_c"]), 1),
            "message": "Heatwave-style warning for buildings. Roof surfaces and upper floors may overheat even before extreme temperature peaks.",
        }
    return None


summary = load_json(DATA_DIR / "project_summary.json")
metrics = load_json(MODELS_DIR / "metrics.json")
phagwara_profile = pd.read_csv(DATA_DIR / "phagwara_profile.csv").iloc[0].to_dict()
punjab_damage = pd.read_csv(DATA_DIR / "punjab_damage_history.csv")
punjab_events = pd.read_csv(DATA_DIR / "punjab_disaster_history.csv")
phagwara_samples = pd.read_csv(DATA_DIR / "phagwara_nearby_flood_samples.csv")
infrastructure_points = pd.read_csv(DATA_DIR / "phagwara_infrastructure_risk.csv")
weather_forecast = pd.read_csv(DATA_DIR / "phagwara_weather_forecast.csv")
clean_flood_risk = pd.read_csv(DATA_DIR / "clean_flood_risk_india.csv")
disaster_hotspots = pd.read_csv(DATA_DIR / "phagwara_disaster_hotspots.csv")
zone_directory = pd.read_csv(DATA_DIR / "phagwara_zone_directory.csv")
global_emergency_alerts = emergency_alerts(infrastructure_points)
global_heatwave_alert = heatwave_alert(phagwara_profile, weather_forecast)
global_emergency_contacts = emergency_contacts()

flood_model = joblib.load(MODELS_DIR / "flood_risk_model.joblib")
disaster_model = joblib.load(MODELS_DIR / "disaster_type_model.joblib")


@app.route("/")
def index():
    feature_cards = [
        {
            "title": "Multi-Disaster Predictor",
            "text": "Test earthquakes, floods, heatwaves, storms, fire, and industrial accidents.",
            "route": "predict",
        },
        {
            "title": "Synthetic Weather View",
            "text": "Show location-aware forecast cards generated from the dummy dataset.",
            "route": "weather",
        },
        {
            "title": "Interactive Risk Map",
            "text": "Explore infrastructure risk and disaster hotspots on the same map.",
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
        top_events=top_events.head(6).to_dict(orient="records"),
        infrastructure_points=infrastructure_points.head(4).to_dict(orient="records"),
        hotspots=disaster_hotspots.head(4).to_dict(orient="records"),
        emergency_alerts=global_emergency_alerts[:3],
        heatwave_alert=global_heatwave_alert,
        emergency_contacts=global_emergency_contacts,
    )


@app.route("/about")
def about():
    top_disasters = (
        punjab_events["disaster_type"]
        .value_counts()
        .rename_axis("disaster_type")
        .reset_index(name="count")
        .head(6)
        .to_dict(orient="records")
    )
    return render_template(
        "about.html",
        summary=summary,
        metrics=metrics,
        profile=phagwara_profile,
        top_disasters=top_disasters,
        emergency_alerts=global_emergency_alerts[:3],
        emergency_contacts=global_emergency_contacts,
    )


@app.route("/weather")
def weather():
    return render_template(
        "weather.html",
        profile=phagwara_profile,
        summary=summary,
        heatwave_alert=global_heatwave_alert,
        emergency_contacts=global_emergency_contacts,
    )


@app.route("/infrastructure-map")
def infrastructure_map():
    points = infrastructure_points.round(2).to_dict(orient="records")
    hotspots = disaster_hotspots.round(2).to_dict(orient="records")
    average_score = round(float(infrastructure_points["infrastructure_risk_score"].mean()), 2)
    highest_score = round(float(infrastructure_points["infrastructure_risk_score"].max()), 2)
    return render_template(
        "infrastructure_map.html",
        profile=phagwara_profile,
        points=points,
        hotspots=hotspots,
        average_score=average_score,
        highest_score=highest_score,
        emergency_alerts=global_emergency_alerts[:3],
        emergency_contacts=global_emergency_contacts,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    defaults = prediction_defaults()
    result = None
    infra_result = compute_infrastructure_risk(defaults)
    form_values = defaults.copy()

    if request.method == "POST":
        form_values = coerce_form_values(request.form.to_dict(), defaults)
        infra_result, result = build_prediction_result(form_values, defaults)

    return render_template(
        "predict.html",
        profile=phagwara_profile,
        metrics=metrics,
        form_values=form_values,
        result=result,
        infra_result=infra_result,
        land_cover_options=sorted(phagwara_samples["land_cover"].dropna().unique().tolist()),
        soil_type_options=sorted(phagwara_samples["soil_type"].dropna().unique().tolist()),
        emergency_alerts=global_emergency_alerts[:3],
        emergency_contacts=global_emergency_contacts,
    )


@app.route("/api/weather")
def api_weather():
    latitude = safe_float(request.args.get("lat"), phagwara_profile["latitude"])
    longitude = safe_float(request.args.get("lon"), phagwara_profile["longitude"])
    location_name = request.args.get("name") or "Phagwara"
    return jsonify(build_weather_payload(location_name, latitude, longitude))


@app.route("/api/live-risk")
def api_live_risk():
    defaults = prediction_defaults()
    form_values = coerce_form_values(request.args.to_dict(), defaults)
    return jsonify(compute_infrastructure_risk(form_values))


@app.route("/api/search-location")
def api_search_location():
    query = (request.args.get("q") or "").strip()
    if not query:
        return jsonify({"error": "Missing query"}), 400
    result = search_location_by_name(query)
    if result is None:
        return jsonify({"error": "Location not found"}), 404
    return jsonify(result)


@app.route("/api/location-context")
def api_location_context():
    query = (request.args.get("q") or "").strip()
    if not query:
        return jsonify({"error": "Missing query"}), 400
    result = search_location_by_name(query)
    if result is None:
        return jsonify({"error": "Location not found"}), 404
    context = build_location_context(result["display_name"], result["latitude"], result["longitude"])
    return jsonify(context)


@app.route("/api/coordinate-context")
def api_coordinate_context():
    latitude = request.args.get("lat")
    longitude = request.args.get("lon")
    if latitude is None or longitude is None:
        return jsonify({"error": "Missing coordinates"}), 400
    lat = float(latitude)
    lon = float(longitude)
    location = reverse_geocode(lat, lon)
    context = build_location_context(location["display_name"], lat, lon)
    return jsonify(context)


@app.route("/api/reverse-geocode")
def api_reverse_geocode():
    latitude = request.args.get("lat")
    longitude = request.args.get("lon")
    if latitude is None or longitude is None:
        return jsonify({"error": "Missing coordinates"}), 400
    result = reverse_geocode(float(latitude), float(longitude))
    return jsonify(result)


@app.route("/download-report")
def download_report():
    defaults = prediction_defaults()
    form_values = coerce_form_values(request.args.to_dict(), defaults)
    infra_result, result = build_prediction_result(form_values, defaults)
    report_text = build_report_text(form_values, infra_result, result)
    response = make_response(report_text)
    response.headers["Content-Type"] = "text/plain; charset=utf-8"
    response.headers["Content-Disposition"] = 'attachment; filename="resilience_report.txt"'
    return response


if __name__ == "__main__":
    app.run(debug=True)
