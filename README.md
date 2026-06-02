# Infrastructure-Specific-Risk-Intelligence-System


A Flask-based disaster risk intelligence website for Phagwara, Punjab. The project uses synthetic local datasets to demonstrate infrastructure risk scoring, flood prediction, disaster classification, live weather context, maps, and preparedness guidance.

The system is designed for academic demonstration, reports, and viva explanation when real operational disaster datasets are unavailable or unsuitable

## Key Features

- Phagwara-focused synthetic datasets for flood risk, disaster history, weather, infrastructure assets, and building safety.
- Machine-learning models for flood risk prediction, disaster type classification, and building safety assessment.
- Interactive web dashboard with local summary statistics and disaster preparedness context.
- Predictor page for infrastructure and location-based risk analysis.
- Weather board with Phagwara zone forecasts and live weather lookup support.
- Infrastructure map showing assets, nearby flood samples, hotspots, emergency contacts, and local recommendations.
- Downloadable risk report generated from predictor inputs.

## Project Structure

```text
.
|-- app.py                         # Main Flask application
|-- README.md                      # Project documentation
|-- metrics.json                   # Top-level model metrics copy
|-- project_summary.json           # Top-level project summary copy
|-- data/                          # Generated CSV datasets and summaries
|-- models/                        # Trained joblib models and metrics
|-- scripts/
|   |-- prepare_data.py            # Generates synthetic Phagwara datasets
|   `-- train_model.py             # Trains and saves ML models
|-- static/css/style.css           # Website styling
`-- templates/                     # Flask HTML templates
```

## Dataset Overview

The project data is synthetic and generated for demonstration. It covers:

- 720 flood-risk records around Phagwara.
- 420 disaster-history records for Punjab-style disaster scenarios.
- 6 disaster categories: Earthquake, Flood, Heatwave, Fire, Storm, and Industrial Accident.
- Infrastructure assets such as transport, health, education, utilities, and emergency facilities.
- Local disaster hotspots and Phagwara zone profiles.

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

This repository does not currently include a `requirements.txt`, so install the required packages directly:

```bash
python -m pip install flask pandas numpy scikit-learn joblib
```

### 3. Generate datasets

```bash
python scripts/prepare_data.py
```

### 4. Train models

```bash
python scripts/train_model.py
```

### 5. Start the website

```bash
python app.py
```

Open the app at:

```text
http://127.0.0.1:5000
```

## Main Pages

- `/` - Dashboard with project summary and local risk indicators.
- `/predict` - Prediction workflow for flood risk, disaster type, infrastructure risk, and building safety.
- `/weather` - Weather and forecast board for Phagwara locations.
- `/infrastructure-map` - Interactive map of infrastructure assets, hotspots, alerts, and recommendations.
- `/about` - Project explanation, dataset notes, and model summary.

## API Routes

- `/api/weather` - Returns live weather context for coordinates.
- `/api/live-risk` - Returns live risk context.
- `/api/search-location` - Searches local or external place names.
- `/api/location-context` - Builds weather and risk context for a named location.
- `/api/coordinate-context` - Builds context from latitude and longitude.
- `/api/reverse-geocode` - Resolves coordinates into a readable place.
- `/download-report` - Downloads the latest prediction report.

## Notes

- The datasets are intentionally synthetic and should not be used for real emergency decisions.
- Live weather and location lookup depend on external network APIs when available.
- If generated files are missing, run `scripts/prepare_data.py` and then `scripts/train_model.py` before launching the app.
- The project is centered on Phagwara, Punjab, including local zones, hotspots, and infrastructure examples.

## Suggested Viva Points

- Explain why synthetic data was used and how it supports repeatable demonstrations.
- Show how flood-risk prediction combines environmental, geographic, and infrastructure features.
- Discuss how the infrastructure map helps visualize preparedness and nearby support assets.
- Mention that the website separates demonstration logic from real-world emergency response requirements.
