# Sail Racing Telemetry Analytics Demo

This project is a polished Streamlit prototype for exploring sail racing telemetry.
It includes realistic sample data generation, KPI analytics, leg comparisons, and
automated tack/gybe detection with estimated speed loss.

## Features

- Streamlit dashboard with generated or uploaded CSV telemetry
- Boat and time-window filters for focused race analysis
- KPI cards for:
  - max speed
  - average speed
  - best VMG
  - average maneuver loss
- Plotly visualizations:
  - speed over time
  - TWA vs VMG scatter
- Leg comparison aggregation table
- Maneuver detection summary and per-event loss breakdown

## Project Structure

- `app.py` - Streamlit entry point and dashboard UI
- `data_generator.py` - sample telemetry data generation
- `analytics.py` - telemetry validation and analytics logic
- `export_sample_csv.py` - helper script to generate CSV for upload testing
- `requirements.txt` - pinned runtime dependencies
- `IMPLEMENTATION_PLAN.md` - implementation checklist

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Generate a Sample CSV

Create a reusable sample telemetry file for the upload workflow:

```bash
python export_sample_csv.py --output sample_telemetry.csv
```

Optional parameters:

```bash
python export_sample_csv.py --output demo_data/telemetry.csv --boats 4 --legs 8 --points-per-leg 140 --freq-seconds 2 --seed 7
```

## Data Schema

Telemetry should include the following columns:

- `timestamp`
- `boat_id`
- `speed_knots`
- `true_wind_speed`
- `true_wind_angle`
- `heading`
- `heel_angle`
- `foil_cant`
- `vmg`
- `maneuver_type`
- `leg_id`

Notes:

- `timestamp` is parsed as datetime.
- `maneuver_type` defaults to `none` when missing.
- `vmg` is computed from speed and true wind angle when missing.

## Verification

Run a compile check:

```bash
python -m compileall .
```

Then run the app:

```bash
streamlit run app.py
```
