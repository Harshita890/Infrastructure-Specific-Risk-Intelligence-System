# Phagwara Flood Risk Dashboard

This project is a simple Flask website made for the Phagwara flood-risk dataset.

## Project files

- `app.py` - Flask backend
- `templates/index.html` - main webpage
- `static/style.css` - page design
- `phagwara_final.csv` - Phagwara dataset
- `requirements.txt` - Python package list

## Step 1: Open terminal in the project folder

Make sure your terminal is inside:

```powershell
C:\Users\LiFe\Desktop\New Project
```

## Step 2: Install Flask

Run:

```powershell
pip install -r requirements.txt
```

## Step 3: Start the website

Run:

```powershell
python app.py
```

## Step 4: Open in browser

Open this address:

```text
http://127.0.0.1:5000
```

## What this website shows

- Total Phagwara records
- Average rainfall
- Average water level
- Average river discharge
- Lowest and highest risk score
- Risk distribution
- Top 5 highest risk rows
- Full table of all Phagwara data

## Easy explanation of code

### `app.py`

- Reads the CSV file
- Converts numbers from text to float
- Calculates summary values
- Sends data to the HTML page

### `index.html`

- Shows dashboard cards
- Shows top-risk list
- Shows data table

### `style.css`

- Adds colors
- Makes cards and table look better
- Makes the page work on mobile and desktop

## Notes

- Right now the project is focused only on `phagwara_final.csv`
- You can later add maps, charts, login, prediction, or alert features
