# Sail Racing Telemetry Analytics Demo

## Project Purpose

This project is a Streamlit prototype inspired by high-performance SailGP-style analytics.
It shows how simulated on-water telemetry can be transformed into tactical performance
insight using lightweight Python analytics.

The demo is intentionally simple, local, and dependency-light while still feeling like
an executive-ready telemetry review experience.

## Features

- Polished multi-section Streamlit dashboard
- Generated sample telemetry with race-course coordinates and weather context
- KPI overview for speed, VMG, and maneuver loss
- Race replay with:
  - boat tracks
  - current positions
  - timestamp slider
  - boat filtering
  - optional leg coloring
  - optional wind-direction arrows
- Maneuver event detection (tacks/gybes) with markers and metrics:
  - entry speed
  - exit speed
  - speed loss
  - recovery time estimate
- Lightweight anomaly detection:
  - sudden speed drop
  - abnormal heel angle
  - poor VMG for wind-angle regime
  - possible foil instability pattern
- Live streaming simulation mode:
  - start/stop/reset
  - playback speed
  - current timestamp and live KPI panel
  - rolling live chart window
- Team-vs-team comparison:
  - side-by-side metric cards
  - comparison table
  - grouped comparison chart
- Predictive "Optimal VMG Zone" analysis:
  - TWA vs VMG with highlighted target bands
  - best upwind/downwind target ranges
  - best TWA band by wind regime
- Weather overlays and weather timeline chart (simulated)

## Project Structure

- `app.py` - Streamlit app and UI sections
- `data_generator.py` - simulated telemetry generation (boats, race track, weather)
- `analytics.py` - validation, maneuvers, anomalies, comparison, and predictive analytics
- `export_sample_csv.py` - helper script to export generated telemetry CSV
- `requirements.txt` - runtime dependencies
- `IMPLEMENTATION_PLAN.md` - original implementation plan

## How to Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app runs with generated sample data by default, so no upload is required.

## Generate a Sample CSV

Create a CSV file for testing the upload path:

```bash
python export_sample_csv.py --output sample_telemetry.csv
```

Custom example:

```bash
python export_sample_csv.py --output demo_data/telemetry.csv --boats 6 --legs 8 --points-per-leg 140 --freq-seconds 2 --seed 7
```

## Simulated Data Schema

Generated telemetry includes:

- `timestamp`
- `boat_id`
- `team_name`
- `leg_id`
- `leg_mode`
- `speed_knots`
- `vmg`
- `true_wind_speed`
- `true_wind_angle`
- `true_wind_direction`
- `gust_knots`
- `heading`
- `heel_angle`
- `foil_cant`
- `x_position`
- `y_position`
- `maneuver_type`
- `course_side`
- `sea_state`

Notes:

- This is simulated telemetry for demo and prototyping use.
- Analytics outputs are intentionally simplified and should not be treated as production race models.

## Why This Matters

SailGP-class teams ingest large telemetry streams in real time and post-race review.
This demo mirrors that workflow at a lightweight level: it links raw sensor-style data
to clear tactical questions such as maneuver quality, wind-adjusted VMG performance,
anomaly identification, and team-vs-team deltas.
