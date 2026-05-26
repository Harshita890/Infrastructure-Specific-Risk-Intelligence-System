from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

PHAGWARA_LAT = 31.2240
PHAGWARA_LON = 75.7708
LPU_LAT = 31.2554
LPU_LON = 75.7052
RNG = np.random.default_rng(42)

ZONE_BLUEPRINTS = [
    {
        "zone_name": "City Centre",
        "latitude": 31.2238,
        "longitude": 75.7720,
        "elevation_m": 247,
        "land_cover": "Built-up",
        "soil_type": "Loam",
        "population_density": 7600,
        "flood_bias": 0.30,
        "heat_bias": 0.35,
        "quake_bias": 0.20,
        "industrial_bias": 0.10,
    },
    {
        "zone_name": "Railway Colony",
        "latitude": 31.2269,
        "longitude": 75.7666,
        "elevation_m": 246,
        "land_cover": "Built-up",
        "soil_type": "Silt Loam",
        "population_density": 6900,
        "flood_bias": 0.24,
        "heat_bias": 0.22,
        "quake_bias": 0.18,
        "industrial_bias": 0.08,
    },
    {
        "zone_name": "Industrial Belt",
        "latitude": 31.2103,
        "longitude": 75.7835,
        "elevation_m": 244,
        "land_cover": "Industrial",
        "soil_type": "Clay Loam",
        "population_density": 5400,
        "flood_bias": 0.36,
        "heat_bias": 0.20,
        "quake_bias": 0.12,
        "industrial_bias": 0.42,
    },
    {
        "zone_name": "Canal Edge",
        "latitude": 31.2365,
        "longitude": 75.7924,
        "elevation_m": 242,
        "land_cover": "Agricultural",
        "soil_type": "Alluvial",
        "population_density": 3100,
        "flood_bias": 0.48,
        "heat_bias": 0.10,
        "quake_bias": 0.10,
        "industrial_bias": 0.04,
    },
    {
        "zone_name": "College District",
        "latitude": 31.2339,
        "longitude": 75.7608,
        "elevation_m": 248,
        "land_cover": "Mixed Use",
        "soil_type": "Loam",
        "population_density": 5900,
        "flood_bias": 0.18,
        "heat_bias": 0.24,
        "quake_bias": 0.15,
        "industrial_bias": 0.06,
    },
    {
        "zone_name": "Peri-Urban Farms",
        "latitude": 31.2428,
        "longitude": 75.7484,
        "elevation_m": 252,
        "land_cover": "Agricultural",
        "soil_type": "Sandy Loam",
        "population_density": 1800,
        "flood_bias": 0.27,
        "heat_bias": 0.16,
        "quake_bias": 0.08,
        "industrial_bias": 0.02,
    },
    {
        "zone_name": "LPU Campus",
        "latitude": LPU_LAT,
        "longitude": LPU_LON,
        "elevation_m": 251,
        "land_cover": "Institutional",
        "soil_type": "Loam",
        "population_density": 5200,
        "flood_bias": 0.16,
        "heat_bias": 0.28,
        "quake_bias": 0.14,
        "industrial_bias": 0.05,
    },
]

INFRASTRUCTURE_BLUEPRINTS = [
    {
        "asset_name": "Phagwara Bus Stand",
        "asset_type": "Transport",
        "latitude": 31.2247,
        "longitude": 75.7710,
        "campus_area": "Phagwara Core",
        "building_age_years": 19,
        "year_built": 2007,
        "floors": 3,
        "building_height_m": 14,
        "distance_to_river_km": 1.8,
        "construction_material": "RCC",
        "foundation_type": "Isolated Footing",
        "drainage_quality": "Moderate",
        "soil_bearing_capacity_kpa": 180,
        "maintenance_score": 72,
        "occupancy_load": 640,
    },
    {
        "asset_name": "Civil Hospital",
        "asset_type": "Healthcare",
        "latitude": 31.2258,
        "longitude": 75.7681,
        "campus_area": "Phagwara Core",
        "building_age_years": 24,
        "year_built": 2002,
        "floors": 5,
        "building_height_m": 23,
        "distance_to_river_km": 2.1,
        "construction_material": "RCC",
        "foundation_type": "Raft Foundation",
        "drainage_quality": "Good",
        "soil_bearing_capacity_kpa": 195,
        "maintenance_score": 78,
        "occupancy_load": 1100,
    },
    {
        "asset_name": "Railway Station Access",
        "asset_type": "Transport",
        "latitude": 31.2266,
        "longitude": 75.7668,
        "campus_area": "Phagwara Core",
        "building_age_years": 17,
        "year_built": 2009,
        "floors": 2,
        "building_height_m": 10,
        "distance_to_river_km": 2.0,
        "construction_material": "Steel",
        "foundation_type": "Pile Foundation",
        "drainage_quality": "Moderate",
        "soil_bearing_capacity_kpa": 175,
        "maintenance_score": 69,
        "occupancy_load": 850,
    },
    {
        "asset_name": "Industrial Gate 2",
        "asset_type": "Industry",
        "latitude": 31.2108,
        "longitude": 75.7830,
        "campus_area": "Industrial Belt",
        "building_age_years": 21,
        "year_built": 2005,
        "floors": 4,
        "building_height_m": 19,
        "distance_to_river_km": 1.2,
        "construction_material": "Steel",
        "foundation_type": "Raft Foundation",
        "drainage_quality": "Poor",
        "soil_bearing_capacity_kpa": 150,
        "maintenance_score": 58,
        "occupancy_load": 920,
    },
    {
        "asset_name": "Canal Bridge",
        "asset_type": "Water Crossing",
        "latitude": 31.2362,
        "longitude": 75.7915,
        "campus_area": "Canal Edge",
        "building_age_years": 31,
        "year_built": 1995,
        "floors": 1,
        "building_height_m": 8,
        "distance_to_river_km": 0.08,
        "construction_material": "RCC",
        "foundation_type": "Pile Foundation",
        "drainage_quality": "Poor",
        "soil_bearing_capacity_kpa": 135,
        "maintenance_score": 54,
        "occupancy_load": 380,
    },
    {
        "asset_name": "Market Road Junction",
        "asset_type": "Commercial",
        "latitude": 31.2231,
        "longitude": 75.7747,
        "campus_area": "City Centre",
        "building_age_years": 16,
        "year_built": 2010,
        "floors": 3,
        "building_height_m": 13,
        "distance_to_river_km": 1.5,
        "construction_material": "Brick Masonry",
        "foundation_type": "Strip Footing",
        "drainage_quality": "Moderate",
        "soil_bearing_capacity_kpa": 165,
        "maintenance_score": 66,
        "occupancy_load": 1400,
    },
    {
        "asset_name": "School Cluster",
        "asset_type": "Education",
        "latitude": 31.2328,
        "longitude": 75.7619,
        "campus_area": "College District",
        "building_age_years": 14,
        "year_built": 2012,
        "floors": 4,
        "building_height_m": 18,
        "distance_to_river_km": 2.7,
        "construction_material": "RCC",
        "foundation_type": "Isolated Footing",
        "drainage_quality": "Good",
        "soil_bearing_capacity_kpa": 205,
        "maintenance_score": 81,
        "occupancy_load": 760,
    },
    {
        "asset_name": "Warehouse Ring",
        "asset_type": "Logistics",
        "latitude": 31.2140,
        "longitude": 75.7808,
        "campus_area": "Industrial Belt",
        "building_age_years": 12,
        "year_built": 2014,
        "floors": 2,
        "building_height_m": 11,
        "distance_to_river_km": 1.4,
        "construction_material": "Steel",
        "foundation_type": "Raft Foundation",
        "drainage_quality": "Poor",
        "soil_bearing_capacity_kpa": 155,
        "maintenance_score": 61,
        "occupancy_load": 520,
    },
    {
        "asset_name": "LPU Main Auditorium",
        "asset_type": "Education",
        "latitude": 31.2558,
        "longitude": 75.7042,
        "campus_area": "LPU Central Spine",
        "building_age_years": 19,
        "year_built": 2007,
        "floors": 9,
        "building_height_m": 39,
        "distance_to_river_km": 3.4,
        "construction_material": "RCC",
        "foundation_type": "Raft Foundation",
        "drainage_quality": "Good",
        "soil_bearing_capacity_kpa": 215,
        "maintenance_score": 83,
        "occupancy_load": 2400,
    },
    {
        "asset_name": "LPU Uni Mall",
        "asset_type": "Commercial",
        "latitude": 31.2565,
        "longitude": 75.7081,
        "campus_area": "LPU Student Hub",
        "building_age_years": 18,
        "year_built": 2008,
        "floors": 9,
        "building_height_m": 40,
        "distance_to_river_km": 3.1,
        "construction_material": "RCC",
        "foundation_type": "Raft Foundation",
        "drainage_quality": "Good",
        "soil_bearing_capacity_kpa": 210,
        "maintenance_score": 79,
        "occupancy_load": 3100,
    },
    {
        "asset_name": "LPU Block 32",
        "asset_type": "Academic",
        "latitude": 31.2548,
        "longitude": 75.7062,
        "campus_area": "LPU Academic Core",
        "building_age_years": 20,
        "year_built": 2006,
        "floors": 9,
        "building_height_m": 42,
        "distance_to_river_km": 3.2,
        "construction_material": "RCC",
        "foundation_type": "Raft Foundation",
        "drainage_quality": "Good",
        "soil_bearing_capacity_kpa": 218,
        "maintenance_score": 82,
        "occupancy_load": 1800,
    },
    {
        "asset_name": "LPU Block 34",
        "asset_type": "Academic",
        "latitude": 31.2541,
        "longitude": 75.7076,
        "campus_area": "LPU Academic Core",
        "building_age_years": 21,
        "year_built": 2005,
        "floors": 9,
        "building_height_m": 42,
        "distance_to_river_km": 3.0,
        "construction_material": "RCC",
        "foundation_type": "Raft Foundation",
        "drainage_quality": "Moderate",
        "soil_bearing_capacity_kpa": 205,
        "maintenance_score": 76,
        "occupancy_load": 1700,
    },
    {
        "asset_name": "LPU School of Civil Engineering",
        "asset_type": "Academic",
        "latitude": 31.2532,
        "longitude": 75.7056,
        "campus_area": "LPU Academic Core",
        "building_age_years": 18,
        "year_built": 2008,
        "floors": 9,
        "building_height_m": 40,
        "distance_to_river_km": 3.3,
        "construction_material": "RCC",
        "foundation_type": "Raft Foundation",
        "drainage_quality": "Good",
        "soil_bearing_capacity_kpa": 220,
        "maintenance_score": 84,
        "occupancy_load": 1600,
    },
    {
        "asset_name": "LPU Boys Hostel BH-1",
        "asset_type": "Residential",
        "latitude": 31.2584,
        "longitude": 75.7014,
        "campus_area": "LPU Hostel Zone",
        "building_age_years": 17,
        "year_built": 2009,
        "floors": 9,
        "building_height_m": 38,
        "distance_to_river_km": 3.7,
        "construction_material": "RCC",
        "foundation_type": "Raft Foundation",
        "drainage_quality": "Moderate",
        "soil_bearing_capacity_kpa": 200,
        "maintenance_score": 74,
        "occupancy_load": 4200,
    },
    {
        "asset_name": "LPU Girls Hostel GH-1",
        "asset_type": "Residential",
        "latitude": 31.2575,
        "longitude": 75.7094,
        "campus_area": "LPU Hostel Zone",
        "building_age_years": 17,
        "year_built": 2009,
        "floors": 9,
        "building_height_m": 38,
        "distance_to_river_km": 2.9,
        "construction_material": "RCC",
        "foundation_type": "Raft Foundation",
        "drainage_quality": "Moderate",
        "soil_bearing_capacity_kpa": 198,
        "maintenance_score": 75,
        "occupancy_load": 3900,
    },
    {
        "asset_name": "LPU Sports Arena",
        "asset_type": "Recreation",
        "latitude": 31.2526,
        "longitude": 75.7028,
        "campus_area": "LPU Sports Belt",
        "building_age_years": 16,
        "year_built": 2010,
        "floors": 9,
        "building_height_m": 37,
        "distance_to_river_km": 3.6,
        "construction_material": "Steel",
        "foundation_type": "Raft Foundation",
        "drainage_quality": "Good",
        "soil_bearing_capacity_kpa": 212,
        "maintenance_score": 80,
        "occupancy_load": 2800,
    },
    {
        "asset_name": "LPU University Hospital",
        "asset_type": "Healthcare",
        "latitude": 31.2569,
        "longitude": 75.7030,
        "campus_area": "LPU Health Block",
        "building_age_years": 16,
        "year_built": 2010,
        "floors": 9,
        "building_height_m": 39,
        "distance_to_river_km": 3.5,
        "construction_material": "RCC",
        "foundation_type": "Raft Foundation",
        "drainage_quality": "Good",
        "soil_bearing_capacity_kpa": 216,
        "maintenance_score": 86,
        "occupancy_load": 1300,
    },
]


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def label_from_score(score: float) -> str:
    if score >= 70:
        return "Critical"
    if score >= 40:
        return "Medium"
    return "Low"


def compute_infrastructure_risk_score(df: pd.DataFrame) -> pd.Series:
    rainfall = df["rainfall_mm"].clip(lower=0, upper=400) / 400
    water_level = df["water_level_m"].clip(lower=0, upper=10) / 10
    discharge = df["river_discharge_m3_s"].clip(lower=0, upper=6000) / 6000
    population = df["population_density"].clip(lower=0, upper=10000) / 10000
    flood_history = df["historical_floods"].clip(lower=0, upper=1)
    infrastructure = df["infrastructure"].clip(lower=0, upper=1)
    elevation = 1 - (df["elevation_m"].clip(lower=0, upper=9000) / 9000)
    temperature = ((df["temperature_deg_c"].clip(lower=12, upper=48) - 12) / 36).clip(lower=0, upper=1)

    score = (
        rainfall * 0.18
        + water_level * 0.22
        + discharge * 0.14
        + population * 0.12
        + flood_history * 0.12
        + infrastructure * 0.08
        + elevation * 0.04
        + temperature * 0.10
    ) * 100
    return score.round(2)


def season_for_month(month: int) -> str:
    if month in {6, 7, 8, 9}:
        return "Monsoon"
    if month in {4, 5}:
        return "Pre-Monsoon"
    if month in {10, 11}:
        return "Post-Monsoon"
    return "Winter"


def month_weather_profile(month: int) -> tuple[float, float, float]:
    if month in {6, 7, 8, 9}:
        return 170.0, 30.5, 80.0
    if month in {4, 5}:
        return 55.0, 37.0, 49.0
    if month in {10, 11}:
        return 38.0, 28.0, 58.0
    return 18.0, 17.0, 61.0


def generate_flood_dataset(samples_per_zone: int = 120) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for zone in ZONE_BLUEPRINTS:
        for _ in range(samples_per_zone):
            month = int(RNG.integers(1, 13))
            season = season_for_month(month)
            base_rainfall, base_temp, base_humidity = month_weather_profile(month)
            rainfall = max(0.0, RNG.normal(base_rainfall + zone["flood_bias"] * 80, 28))
            temperature = RNG.normal(base_temp + zone["heat_bias"] * 4, 2.5)
            humidity = np.clip(RNG.normal(base_humidity + zone["flood_bias"] * 9, 8), 28, 99)
            discharge = max(300.0, RNG.normal(1400 + rainfall * 8 + zone["flood_bias"] * 1400, 240))
            water_level = np.clip(RNG.normal(2.2 + rainfall / 60 + zone["flood_bias"] * 1.8, 0.45), 0.6, 8.5)
            elevation = np.clip(RNG.normal(zone["elevation_m"], 3.5), 236, 257)
            population_density = max(800.0, RNG.normal(zone["population_density"], 650))
            infrastructure = int(RNG.random() < (0.42 + zone["industrial_bias"] + zone["heat_bias"] * 0.2))
            historical_floods = int(RNG.random() < (0.18 + zone["flood_bias"]))
            latitude = zone["latitude"] + RNG.normal(0, 0.0042)
            longitude = zone["longitude"] + RNG.normal(0, 0.0042)
            flood_signal = (
                (rainfall - 90) / 32
                + (water_level - 3.4) * 1.3
                + (discharge - 2000) / 1100
                + zone["flood_bias"] * 2.2
                + historical_floods * 0.8
                - (elevation - 245) / 9
            )
            flood_probability = sigmoid(flood_signal - 1.35)
            rows.append(
                {
                    "zone_name": zone["zone_name"],
                    "season": season,
                    "month": month,
                    "latitude": round(latitude, 6),
                    "longitude": round(longitude, 6),
                    "rainfall_mm": round(rainfall, 2),
                    "temperature_deg_c": round(temperature, 2),
                    "humidity_pct": round(float(humidity), 2),
                    "river_discharge_m3_s": round(float(discharge), 2),
                    "water_level_m": round(float(water_level), 2),
                    "elevation_m": round(float(elevation), 2),
                    "land_cover": zone["land_cover"],
                    "soil_type": zone["soil_type"],
                    "population_density": round(float(population_density), 2),
                    "infrastructure": infrastructure,
                    "historical_floods": historical_floods,
                    "flood_occurred": int(RNG.random() < flood_probability),
                }
            )
    flood_df = pd.DataFrame(rows)
    flood_df["distance_to_phagwara_km"] = flood_df.apply(
        lambda row: haversine_km(PHAGWARA_LAT, PHAGWARA_LON, row["latitude"], row["longitude"]),
        axis=1,
    )
    flood_df["infrastructure_risk_score"] = compute_infrastructure_risk_score(flood_df)
    flood_df["infrastructure_risk_label"] = flood_df["infrastructure_risk_score"].apply(label_from_score)
    return flood_df.sort_values(["distance_to_phagwara_km", "zone_name"]).reset_index(drop=True)


def create_phagwara_profile(flood_df: pd.DataFrame) -> pd.DataFrame:
    profile = {
        "city": "Phagwara",
        "state": "Punjab",
        "latitude": PHAGWARA_LAT,
        "longitude": PHAGWARA_LON,
        "rainfall_mm": round(float(flood_df["rainfall_mm"].median()), 2),
        "temperature_deg_c": round(float(flood_df["temperature_deg_c"].median()), 2),
        "humidity_pct": round(float(flood_df["humidity_pct"].median()), 2),
        "river_discharge_m3_s": round(float(flood_df["river_discharge_m3_s"].median()), 2),
        "water_level_m": round(float(flood_df["water_level_m"].median()), 2),
        "elevation_m": round(float(flood_df["elevation_m"].median()), 2),
        "land_cover": str(flood_df["land_cover"].mode().iat[0]),
        "soil_type": str(flood_df["soil_type"].mode().iat[0]),
        "population_density": round(float(flood_df["population_density"].median()), 2),
        "infrastructure": int(round(float(flood_df["infrastructure"].mean()))),
        "historical_floods": int(round(float(flood_df["historical_floods"].mean()))),
        "nearest_sample_count": int(len(flood_df)),
        "mean_distance_km": round(float(flood_df["distance_to_phagwara_km"].mean()), 2),
        "flood_occurred_ratio": round(float(flood_df["flood_occurred"].mean()), 3),
        "avg_infrastructure_risk_score": round(float(flood_df["infrastructure_risk_score"].mean()), 2),
        "peak_infrastructure_risk_score": round(float(flood_df["infrastructure_risk_score"].max()), 2),
    }
    return pd.DataFrame([profile])


def create_infrastructure_points(flood_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    asset_bonus = {
        "Canal Bridge": 66,
        "Industrial Gate 2": 48,
        "Civil Hospital": 28,
        "Market Road Junction": 26,
    }
    for asset in INFRASTRUCTURE_BLUEPRINTS:
        asset_name = asset["asset_name"]
        asset_type = asset["asset_type"]
        latitude = asset["latitude"]
        longitude = asset["longitude"]
        sample = flood_df.assign(
            distance_to_asset_km=flood_df.apply(
                lambda row: haversine_km(latitude, longitude, row["latitude"], row["longitude"]),
                axis=1,
            )
        ).nsmallest(1, "distance_to_asset_km").iloc[0]
        asset_multiplier = {
            "Healthcare": 1.16,
            "Transport": 1.08,
            "Industry": 1.24,
            "Education": 1.02,
            "Water Crossing": 1.32,
            "Commercial": 1.08,
            "Logistics": 1.14,
        }.get(asset_type, 1.0)
        score = min(
            99.0,
            round(
                float(sample["infrastructure_risk_score"]) * asset_multiplier
                + (6 if asset_type in {"Water Crossing", "Industry"} else 0)
                + min(float(asset["building_age_years"]) * 0.28, 10)
                + min(float(asset["floors"]) * 0.85, 8)
                + asset_bonus.get(asset_name, 0),
                2,
            ),
        )
        rows.append(
            {
                "asset_name": asset_name,
                "asset_type": asset_type,
                "latitude": latitude,
                "longitude": longitude,
                "distance_to_phagwara_km": round(haversine_km(PHAGWARA_LAT, PHAGWARA_LON, latitude, longitude), 2),
                "infrastructure_risk_score": score,
                "infrastructure_risk_label": label_from_score(score),
                "priority_action": (
                    "Immediate response planning"
                    if score >= 70
                    else "Preparedness and inspection"
                    if score >= 40
                    else "Routine monitoring"
                ),
                "rainfall_mm": float(sample["rainfall_mm"]),
                "water_level_m": float(sample["water_level_m"]),
                "population_density": float(sample["population_density"]),
                "temperature_deg_c": float(sample["temperature_deg_c"]),
                "building_age_years": int(asset["building_age_years"]),
                "year_built": int(asset["year_built"]),
                "floors": int(asset["floors"]),
                "building_height_m": float(asset["building_height_m"]),
                "distance_to_river_km": float(asset["distance_to_river_km"]),
                "construction_material": asset["construction_material"],
                "foundation_type": asset["foundation_type"],
                "drainage_quality": asset["drainage_quality"],
                "soil_bearing_capacity_kpa": float(asset["soil_bearing_capacity_kpa"]),
                "maintenance_score": float(asset["maintenance_score"]),
                "occupancy_load": int(asset["occupancy_load"]),
                "campus_area": asset["campus_area"],
            }
        )
    return pd.DataFrame(rows).sort_values("infrastructure_risk_score", ascending=False).reset_index(drop=True)


def building_condition_label(score: float) -> str:
    if score >= 60:
        return "Unsafe"
    if score >= 42:
        return "Major Repair"
    if score >= 25:
        return "Minor Repair"
    return "Good"


def generate_building_assessment_dataset(infrastructure_df: pd.DataFrame, samples_per_asset: int = 35) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    material_risk = {"Brick Masonry": 9, "RCC": 2, "Steel": 4}
    foundation_risk = {"Strip Footing": 8, "Isolated Footing": 5, "Raft Foundation": 2, "Pile Foundation": 1}
    drainage_risk = {"Poor": 10, "Moderate": 5, "Good": 1}

    for _, asset in infrastructure_df.iterrows():
        for _ in range(samples_per_asset):
            floors = int(np.clip(RNG.normal(asset["floors"], 1.3), 1, 14))
            height = float(np.clip(RNG.normal(asset["building_height_m"], 4.0), 4, 55))
            age = float(np.clip(RNG.normal(asset["building_age_years"], 5.0), 0, 70))
            river_distance = float(np.clip(RNG.normal(asset["distance_to_river_km"], 0.35), 0.02, 5.0))
            soil_capacity = float(np.clip(RNG.normal(asset["soil_bearing_capacity_kpa"], 20), 80, 260))
            maintenance = float(np.clip(RNG.normal(asset["maintenance_score"], 12), 15, 100))
            occupancy = int(np.clip(RNG.normal(asset["occupancy_load"], asset["occupancy_load"] * 0.18), 80, 6000))
            drainage = str(RNG.choice(["Poor", "Moderate", "Good"], p=[0.2, 0.45, 0.35]))
            material = str(RNG.choice(["Brick Masonry", "RCC", "Steel"], p=[0.18, 0.66, 0.16]))
            foundation = str(RNG.choice(["Strip Footing", "Isolated Footing", "Raft Foundation", "Pile Foundation"], p=[0.18, 0.32, 0.34, 0.16]))
            rainfall = float(np.clip(RNG.normal(asset["rainfall_mm"], 32), 0, 400))
            water_level = float(np.clip(RNG.normal(asset["water_level_m"], 0.5), 0.1, 10))
            scenario = str(RNG.choice(["good", "normal", "stressed", "unsafe"], p=[0.18, 0.48, 0.24, 0.10]))

            if scenario == "good":
                age = max(0.0, age - 8)
                floors = max(1, floors - 1)
                height = max(4.0, height - 3)
                river_distance = min(5.0, river_distance + 1.0)
                soil_capacity = min(260.0, soil_capacity + 30)
                maintenance = min(100.0, maintenance + 18)
                occupancy = max(80, int(occupancy * 0.72))
                rainfall = max(0.0, rainfall - 35)
                water_level = max(0.1, water_level - 0.8)
                drainage = "Good"
                material = str(RNG.choice(["RCC", "Steel"], p=[0.75, 0.25]))
                foundation = str(RNG.choice(["Raft Foundation", "Pile Foundation"], p=[0.72, 0.28]))
            elif scenario == "unsafe":
                age = min(70.0, age + 18)
                floors = min(14, floors + 2)
                height = min(55.0, height + 6)
                river_distance = max(0.02, river_distance - 1.0)
                soil_capacity = max(80.0, soil_capacity - 45)
                maintenance = max(15.0, maintenance - 26)
                occupancy = min(6000, int(occupancy * 1.35))
                rainfall = min(400.0, rainfall + 65)
                water_level = min(10.0, water_level + 1.4)
                drainage = "Poor"
                material = str(RNG.choice(["Brick Masonry", "RCC"], p=[0.68, 0.32]))
                foundation = str(RNG.choice(["Strip Footing", "Isolated Footing"], p=[0.7, 0.3]))
            elif scenario == "stressed":
                age = min(70.0, age + 8)
                river_distance = max(0.02, river_distance - 0.45)
                soil_capacity = max(80.0, soil_capacity - 18)
                maintenance = max(15.0, maintenance - 12)
                rainfall = min(400.0, rainfall + 25)
                water_level = min(10.0, water_level + 0.5)

            vulnerability_score = (
                age * 0.32
                + floors * 1.35
                + height * 0.18
                + max(0.0, 2.0 - river_distance) * 6.0
                + max(0.0, 180.0 - soil_capacity) * 0.12
                + max(0.0, 70.0 - maintenance) * 0.38
                + min(occupancy / 6000.0, 1.0) * 6.0
                + rainfall / 400.0 * 7.0
                + water_level / 10.0 * 8.0
                + material_risk[material]
                + foundation_risk[foundation]
                + drainage_risk[drainage]
                + RNG.normal(0, 5)
            )
            vulnerability_score = round(float(np.clip(vulnerability_score, 0, 100)), 2)
            rows.append(
                {
                    "asset_name": asset["asset_name"],
                    "asset_type": asset["asset_type"],
                    "latitude": round(float(asset["latitude"]), 6),
                    "longitude": round(float(asset["longitude"]), 6),
                    "rainfall_mm": round(rainfall, 2),
                    "water_level_m": round(water_level, 2),
                    "soil_type": asset.get("soil_type", "Loam"),
                    "building_age_years": round(age, 1),
                    "floors": floors,
                    "building_height_m": round(height, 2),
                    "distance_to_river_km": round(river_distance, 2),
                    "construction_material": material,
                    "foundation_type": foundation,
                    "drainage_quality": drainage,
                    "soil_bearing_capacity_kpa": round(soil_capacity, 2),
                    "maintenance_score": round(maintenance, 2),
                    "occupancy_load": occupancy,
                    "building_vulnerability_score": vulnerability_score,
                    "building_condition": building_condition_label(vulnerability_score),
                }
            )

    return pd.DataFrame(rows)


def create_weather_forecast(profile_df: pd.DataFrame) -> pd.DataFrame:
    profile = profile_df.iloc[0]
    rows: list[dict[str, object]] = []
    sequence = [
        ("Day 1", "Stable Morning", -1.0, 8, 0.1),
        ("Day 2", "Humid Build-Up", 1.8, 18, 0.3),
        ("Day 3", "Heat Stress", 4.5, -4, -0.2),
        ("Day 4", "Storm Watch", 0.9, 34, 0.6),
        ("Day 5", "Light Rain", -0.5, 22, 0.4),
        ("Day 6", "Flood Watch", -1.1, 44, 0.8),
        ("Day 7", "Recovery Window", 0.3, 10, 0.0),
    ]
    for day, condition, temp_shift, rain_shift, water_shift in sequence:
        rainfall = max(0.0, float(profile["rainfall_mm"]) + rain_shift)
        water_level = max(0.0, float(profile["water_level_m"]) + water_shift)
        risk_score = min(
            100.0,
            max(
                0.0,
                float(profile["avg_infrastructure_risk_score"]) + rain_shift * 0.45 + water_shift * 10,
            ),
        )
        rows.append(
            {
                "day": day,
                "condition": condition,
                "temperature_deg_c": round(float(profile["temperature_deg_c"]) + temp_shift, 2),
                "rainfall_mm": round(rainfall, 2),
                "humidity_pct": round(min(99.0, float(profile["humidity_pct"]) + rain_shift * 0.25), 2),
                "water_level_m": round(water_level, 2),
                "infrastructure_risk_score": round(risk_score, 2),
                "risk_label": label_from_score(risk_score),
            }
        )
    return pd.DataFrame(rows)


def pick_type_by_month(month: int) -> str:
    if month in {6, 7, 8, 9}:
        options = ["Flood", "Storm", "Fire", "Industrial Accident", "Earthquake", "Heatwave"]
        weights = [0.34, 0.24, 0.08, 0.10, 0.08, 0.16]
    elif month in {4, 5}:
        options = ["Heatwave", "Fire", "Storm", "Industrial Accident", "Earthquake", "Flood"]
        weights = [0.36, 0.22, 0.10, 0.12, 0.10, 0.10]
    elif month in {10, 11}:
        options = ["Flood", "Fire", "Industrial Accident", "Earthquake", "Storm", "Heatwave"]
        weights = [0.18, 0.18, 0.18, 0.18, 0.18, 0.10]
    else:
        options = ["Earthquake", "Fire", "Industrial Accident", "Storm", "Flood", "Heatwave"]
        weights = [0.26, 0.18, 0.18, 0.16, 0.12, 0.10]
    return str(RNG.choice(options, p=weights))


def generate_disaster_history(records: int = 420) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    zone_names = [zone["zone_name"] for zone in ZONE_BLUEPRINTS]
    zone_lookup = {zone["zone_name"]: zone for zone in ZONE_BLUEPRINTS}
    subtype_map = {
        "Flood": ["Urban Flood", "Flash Flood", "Canal Overflow"],
        "Earthquake": ["Shallow Tremor", "Regional Shock"],
        "Heatwave": ["Extreme Heat", "Humidity Surge"],
        "Fire": ["Market Fire", "Residential Fire", "Crop Fire"],
        "Storm": ["Thunderstorm", "Windstorm"],
        "Industrial Accident": ["Factory Blast", "Chemical Leak"],
    }

    for _ in range(records):
        year = int(RNG.integers(2012, 2027))
        month = int(RNG.integers(1, 13))
        disaster_type = pick_type_by_month(month)
        zone_name = str(RNG.choice(zone_names))
        zone = zone_lookup[zone_name]
        latitude = zone["latitude"] + RNG.normal(0, 0.005)
        longitude = zone["longitude"] + RNG.normal(0, 0.005)

        if disaster_type == "Flood":
            magnitude = RNG.normal(4.6 + zone["flood_bias"], 0.45)
            total_deaths = max(0, int(RNG.normal(8 + zone["flood_bias"] * 14, 4)))
            total_affected = max(500, int(RNG.normal(4200 + zone["flood_bias"] * 3200, 900)))
            total_damage = max(150, int(RNG.normal(2400 + zone["flood_bias"] * 1800, 650)))
        elif disaster_type == "Earthquake":
            magnitude = RNG.normal(5.4 + zone["quake_bias"], 0.35)
            total_deaths = max(0, int(RNG.normal(12 + zone["quake_bias"] * 18, 5)))
            total_affected = max(350, int(RNG.normal(2500 + zone["quake_bias"] * 1800, 650)))
            total_damage = max(200, int(RNG.normal(3200 + zone["quake_bias"] * 1600, 700)))
        elif disaster_type == "Heatwave":
            magnitude = RNG.normal(43.8 + zone["heat_bias"] * 3, 1.2)
            total_deaths = max(0, int(RNG.normal(5 + zone["heat_bias"] * 8, 3)))
            total_affected = max(600, int(RNG.normal(5200 + zone["heat_bias"] * 2500, 850)))
            total_damage = max(90, int(RNG.normal(780 + zone["heat_bias"] * 400, 180)))
        elif disaster_type == "Fire":
            magnitude = RNG.normal(2.9 + zone["industrial_bias"], 0.45)
            total_deaths = max(0, int(RNG.normal(4 + zone["industrial_bias"] * 10, 2.5)))
            total_affected = max(120, int(RNG.normal(900 + zone["industrial_bias"] * 1400, 260)))
            total_damage = max(120, int(RNG.normal(1600 + zone["industrial_bias"] * 1200, 420)))
        elif disaster_type == "Storm":
            magnitude = RNG.normal(4.2 + zone["flood_bias"] * 0.8, 0.4)
            total_deaths = max(0, int(RNG.normal(6 + zone["flood_bias"] * 9, 3)))
            total_affected = max(400, int(RNG.normal(2100 + zone["flood_bias"] * 1800, 520)))
            total_damage = max(160, int(RNG.normal(1900 + zone["flood_bias"] * 1200, 450)))
        else:
            magnitude = RNG.normal(3.5 + zone["industrial_bias"] * 1.2, 0.35)
            total_deaths = max(0, int(RNG.normal(7 + zone["industrial_bias"] * 16, 3.5)))
            total_affected = max(150, int(RNG.normal(1100 + zone["industrial_bias"] * 2200, 420)))
            total_damage = max(150, int(RNG.normal(2200 + zone["industrial_bias"] * 2100, 520)))

        rows.append(
            {
                "disaster_type": disaster_type,
                "disaster_subtype": str(RNG.choice(subtype_map[disaster_type])),
                "location": f"{zone_name}, Phagwara, Punjab",
                "zone_name": zone_name,
                "start_year": year,
                "start_month": month,
                "magnitude": round(float(magnitude), 2),
                "latitude": round(float(latitude), 6),
                "longitude": round(float(longitude), 6),
                "total_deaths": total_deaths,
                "total_affected": total_affected,
                "total_damage_usd_000": total_damage,
                "phagwara_relevance": "High",
            }
        )
    return pd.DataFrame(rows).sort_values(["start_year", "start_month"]).reset_index(drop=True)


def create_punjab_damage(disaster_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in range(2020, 2026):
        year_slice = disaster_df[disaster_df["start_year"] == year]
        rows.append(
            {
                "state": "Punjab",
                "year": str(year),
                "lives_lost": int(year_slice["total_deaths"].sum()),
                "cattle_lost": int((year_slice["total_affected"].sum() * 0.015) + 10),
                "houses_damaged": int((year_slice["total_damage_usd_000"].sum() * 1.7) + 40),
                "crop_area_lakh_ha": round(float(year_slice["total_damage_usd_000"].sum()) / 12000, 2),
            }
        )
    return pd.DataFrame(rows)


def create_hotspots(disaster_df: pd.DataFrame, flood_df: pd.DataFrame) -> pd.DataFrame:
    latest_year = int(disaster_df["start_year"].max())
    recent = disaster_df[disaster_df["start_year"] >= latest_year - 2]
    grouped = (
        recent.groupby("zone_name")
        .agg(
            recent_incidents=("disaster_type", "size"),
            dominant_hazard=("disaster_type", lambda series: series.mode().iat[0]),
            average_magnitude=("magnitude", "mean"),
            average_damage=("total_damage_usd_000", "mean"),
        )
        .reset_index()
    )
    flood_zone = (
        flood_df.groupby("zone_name")
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            flood_probability=("flood_occurred", "mean"),
            infrastructure_risk_score=("infrastructure_risk_score", "mean"),
        )
        .reset_index()
    )
    hotspots = grouped.merge(flood_zone, on="zone_name", how="left")
    hotspots["alert_score"] = (
        hotspots["recent_incidents"] * 4
        + hotspots["flood_probability"] * 30
        + hotspots["infrastructure_risk_score"] * 0.55
    ).clip(upper=100)
    hotspots["alert_level"] = hotspots["alert_score"].apply(label_from_score)
    hotspots["response_priority"] = np.where(
        hotspots["alert_score"] >= 70,
        "Immediate field coordination",
        np.where(hotspots["alert_score"] >= 40, "Preparedness drill and inspection", "Community awareness"),
    )
    return hotspots.round(2).sort_values("alert_score", ascending=False).reset_index(drop=True)


def create_zone_directory() -> pd.DataFrame:
    rows = [
        {"place_name": "Phagwara", "category": "City", "latitude": PHAGWARA_LAT, "longitude": PHAGWARA_LON},
    ]
    for zone in ZONE_BLUEPRINTS:
        rows.append(
            {
                "place_name": zone["zone_name"],
                "category": "Zone",
                "latitude": zone["latitude"],
                "longitude": zone["longitude"],
            }
        )
    for asset in INFRASTRUCTURE_BLUEPRINTS:
        rows.append(
            {
                "place_name": asset["asset_name"],
                "category": asset["asset_type"],
                "latitude": asset["latitude"],
                "longitude": asset["longitude"],
            }
        )
    return pd.DataFrame(rows)


def create_summary(
    profile_df: pd.DataFrame,
    disaster_df: pd.DataFrame,
    punjab_damage_df: pd.DataFrame,
    hotspots_df: pd.DataFrame,
    flood_df: pd.DataFrame,
) -> dict[str, object]:
    profile = profile_df.iloc[0]
    return {
        "city": "Phagwara",
        "state": "Punjab",
        "dataset_mode": "Synthetic dummy dataset for demonstration",
        "synthetic_flood_records": int(len(flood_df)),
        "synthetic_disaster_records": int(len(disaster_df)),
        "disaster_types_covered": int(disaster_df["disaster_type"].nunique()),
        "nearest_flood_samples": int(profile["nearest_sample_count"]),
        "average_nearest_distance_km": float(profile["mean_distance_km"]),
        "estimated_flood_occurrence_ratio": float(profile["flood_occurred_ratio"]),
        "punjab_historic_disaster_events": int(len(disaster_df)),
        "top_punjab_disaster_type": str(disaster_df["disaster_type"].mode().iat[0]),
        "punjab_total_lives_lost_2020_2025": float(punjab_damage_df["lives_lost"].sum()),
        "punjab_total_houses_damaged_2020_2025": float(punjab_damage_df["houses_damaged"].sum()),
        "avg_infrastructure_risk_score": float(profile["avg_infrastructure_risk_score"]),
        "peak_infrastructure_risk_score": float(profile["peak_infrastructure_risk_score"]),
        "hotspot_count": int(len(hotspots_df)),
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    flood_df = generate_flood_dataset()
    profile_df = create_phagwara_profile(flood_df)
    infrastructure_df = create_infrastructure_points(flood_df)
    building_assessment_df = generate_building_assessment_dataset(infrastructure_df)
    weather_df = create_weather_forecast(profile_df)
    disaster_df = generate_disaster_history()
    punjab_damage_df = create_punjab_damage(disaster_df)
    hotspots_df = create_hotspots(disaster_df, flood_df)
    zone_directory_df = create_zone_directory()

    flood_df.to_csv(DATA_DIR / "clean_flood_risk_india.csv", index=False)
    flood_df.to_csv(DATA_DIR / "phagwara_nearby_flood_samples.csv", index=False)
    profile_df.to_csv(DATA_DIR / "phagwara_profile.csv", index=False)
    infrastructure_df.to_csv(DATA_DIR / "phagwara_infrastructure_risk.csv", index=False)
    building_assessment_df.to_csv(DATA_DIR / "building_safety_assessment.csv", index=False)
    weather_df.to_csv(DATA_DIR / "phagwara_weather_forecast.csv", index=False)
    disaster_df.to_csv(DATA_DIR / "clean_disaster_history.csv", index=False)
    disaster_df.to_csv(DATA_DIR / "punjab_disaster_history.csv", index=False)
    punjab_damage_df.to_csv(DATA_DIR / "punjab_damage_history.csv", index=False)
    hotspots_df.to_csv(DATA_DIR / "phagwara_disaster_hotspots.csv", index=False)
    zone_directory_df.to_csv(DATA_DIR / "phagwara_zone_directory.csv", index=False)

    summary = create_summary(profile_df, disaster_df, punjab_damage_df, hotspots_df, flood_df)
    with (DATA_DIR / "project_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Generated synthetic Phagwara datasets in:", DATA_DIR)
    for path in sorted(DATA_DIR.glob("*")):
        print("-", path.name)


if __name__ == "__main__":
    main()
