# Infrastructure-Specific-Risk-Intelligence-System


This project is a Flask-based interactive disaster website for **Phagwara, Punjab**. It was rebuilt around a **synthetic dummy dataset** so the project can still demonstrate mapping, prediction, and risk analysis even when the original dataset is not suitable.

## What the project includes

1. A generated Phagwara-focused flood and infrastructure dataset.
2. A synthetic disaster-history dataset covering:
   Earthquake, Flood, Heatwave, Fire, Storm, and Industrial Accident.
3. Trained machine-learning models for:
   Flood-risk prediction and disaster-type classification.
4. An interactive website with:
   Home dashboard, predictor, forecast board, and disaster map.

## Project structure

- `scripts/prepare_data.py`
  Generates all dummy CSV datasets for Phagwara.
- `scripts/train_model.py`
  Trains the models and saves `joblib` files plus `metrics.json`.
- `app.py`
  Runs the Flask website.
- `data/`
  Stores generated datasets and summary files.
- `models/`
  Stores trained models and model metrics.

## How to run

### 1. Install packages

```bash
python -m pip install -r requirements.txt
```

### 2. Generate the Phagwara dummy dataset

```bash
python scripts/prepare_data.py
```

### 3. Train the models

```bash
python scripts/train_model.py
```

### 4. Start the website

```bash
python app.py
```

### 5. Open the project

Visit:

```text
http://127.0.0.1:5000
```

## Main pages

- `/`
  Project dashboard and summary.
- `/predict`
  Multi-disaster predictor with flood-risk score.
- `/weather`
  Synthetic forecast board for Phagwara zones.
- `/infrastructure-map`
  Interactive map with infrastructure points and disaster hotspots.
- `/about`
  Project explanation and model summary.

## Notes for report or viva

- The dataset is intentionally synthetic and created for demonstration.
- The website is focused on **Phagwara** and includes local zones, hotspots, and infrastructure assets.
- The map and prediction flow are designed to explain disaster preparedness visually.
