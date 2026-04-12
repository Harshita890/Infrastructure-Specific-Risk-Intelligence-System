# Infrastructure-Specific Risk Intelligence System

This is a simple Python Flask project for your CAP599 topic.

It uses `phagwara_final.csv` as the main dataset and builds a dashboard for:

- disaster detection
- IVI calculation
- operational disruption probability
- safe / not safe infrastructure recommendation

## Project assumption

Your `phagwara_final.csv` file does not contain real hospital names, bridge names, or road names.

So for this student project:

- each row is treated as one infrastructure unit
- the app assigns unit types as `Hospital`, `Bridge`, `Road`, and `Data Center`

This matches your project limitation that some infrastructure attributes need proxy assumptions.

## IVI formula

```text
IVI = structural factors + hazard exposure + design age + proximity factors
```

## Files

- `app.py` - Python Flask app
- `templates/index.html` - dashboard page
- `static/style.css` - styling
- `phagwara_final.csv` - dataset
- `requirements.txt` - Python package list

## How to run

### 1. Open terminal

```powershell
cd "C:\Users\LiFe\Desktop\New Project"
```

### 2. Install Flask

```powershell
pip install -r requirements.txt
```

### 3. Run the app

```powershell
python app.py
```

### 4. Open browser

```text
http://127.0.0.1:5000
```

## What the dashboard shows

- detected hazard
- number of safe units
- number of caution units
- number of not safe units
- top vulnerable infrastructure
- recommended safe infrastructure
- full operational dashboard table

## Simple code flow

1. Read `phagwara_final.csv`
2. Convert rainfall, water level, discharge, and risk score into numbers
3. Detect disaster type
4. Calculate IVI
5. Calculate disruption probability
6. Mark each unit as safe, caution, or not safe
7. Show results in the Flask dashboard

## Future improvement

Later you can add:

- real OpenStreetMap data
- Overpass API integration
- live weather API
- real hospital and bridge names
- map view
- alert notifications
- machine learning model
