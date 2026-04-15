# Phagwara Disaster Prediction Project

This project builds a small Flask website for a data science project focused on **Phagwara, Punjab**.

It does 3 things:

1. Cleans the raw India disaster datasets.
2. Creates a Phagwara-focused cleaned dataset from the nearest useful records.
3. Trains machine-learning models and shows predictions in a web app.

## Raw files used

The code reads these files from `C:\Users\LiFe\Desktop\New Project\`:

- `disasterIND.csv`
- `flood_risk_dataset_india.csv`
- `RS_Session_248_AU_355.1.csv`

## Project structure

- `scripts/prepare_data.py` cleans and creates the Phagwara datasets.
- `scripts/train_model.py` trains the models and saves their scores.
- `app.py` runs the Flask website.
- `data/` stores cleaned CSV files.
- `models/` stores trained `.joblib` files and `metrics.json`.

## Easy step-by-step run

### 1. Install packages

```bash
python -m pip install -r requirements.txt
```

### 2. Clean the datasets

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

### 5. Open in browser

Open:

```text
http://127.0.0.1:5000
```

## Pages in the website

- `/` shows project summary and model scores.
- `/phagwara-data` shows the cleaned Phagwara dataset.
- `/predict` lets you enter values and predict disaster type and flood risk.

## Notes for project report

- The flood-risk model is trained on the flood dataset.
- The disaster-type model is trained on historical India disaster records.
- Phagwara does not appear directly in the raw files, so the project creates a **derived Phagwara profile** from nearby Punjab-region flood records plus Punjab damage history.
