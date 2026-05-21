# Sail Racing Telemetry Analytics Demo

**v1.0 - Open Telemetry Analytics Framework Prototype**

## What this is

This repository is a lightweight, portfolio-ready prototype of an open telemetry
analytics framework for high-performance sailing. It combines:

- a canonical telemetry schema
- a modular ingestion layer
- reusable analytics modules
- replay and visualization helpers
- a Streamlit example app that demonstrates end-to-end workflows

The current implementation uses simulated telemetry and is designed to be clear,
extensible, and practical for rapid sports-technology prototyping.

## Why telemetry matters in high-performance sailing

Modern high-performance sailing teams rely on telemetry to answer tactical and
coaching questions quickly: where speed was gained/lost, how maneuvers affected
race position, how wind shifts changed outcomes, and which settings delivered
consistent VMG in each race mode.

### Why this matters

High-performance sailing teams, coaches, analysts, race organizers, and fan-
experience teams need ways to transform raw telemetry into actionable insight.

## Features

- Canonical telemetry schema validation and normalization
- Simulated multi-boat race telemetry generation
- CSV ingestion with a unified ingest interface
- Modular analytics for:
  - KPIs and leg summaries
  - maneuver detection and loss estimation
  - anomaly detection
  - VMG banding and target zone recommendations
  - weather overlays
  - boat/team comparison
- Replay-ready track helpers
- Reusable Plotly visualization functions independent of Streamlit
- Streamlit demo app with live simulation controls

## Framework architecture

Project structure:

```text
sail-racing-telemetry-analytics-demo/
  app.py
  README.md
  requirements.txt
  runtime.txt
  sample_data/
    sailing_telemetry_sample.csv
  sailing_telemetry/
    __init__.py
    schema.py
    generator.py
    ingestion.py
    replay.py
    visualization.py
    utils.py
    analytics/
      __init__.py
      metrics.py
      maneuvers.py
      anomalies.py
      vmg.py
      weather.py
      comparison.py
  tests/
    test_schema.py
    test_metrics.py
    test_maneuvers.py
    test_anomalies.py
```

System flow diagram:

```text
+---------------------------------------------------------------+
|                     Streamlit Example App                     |
|                           app.py                              |
|  (Sidebar controls, sections, live simulation, user workflow) |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|                 Ingestion + Schema Normalization              |
|  sailing_telemetry/ingestion.py                               |
|  - ingest_telemetry()                                         |
|  - load_sample_data() / load_csv()                            |
|                                                               |
|  sailing_telemetry/schema.py                                  |
|  - validate_telemetry_dataframe()                             |
|  - normalize_telemetry_dataframe()                            |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|                    Canonical Telemetry DataFrame              |
|   timestamp, boat_id, team_name, leg_id, leg_mode, ...        |
+-------------------+-------------------+-----------------------+
                    |                   |
                    v                   v
      +---------------------------+   +-------------------------+
      |      Analytics Engine     |   |       Replay Layer      |
      | sailing_telemetry/analytics/ | sailing_telemetry/replay.py |
      | - metrics.py              |   | - get_replay_history()  |
      | - maneuvers.py            |   | - get_replay_frame()    |
      | - anomalies.py            |   | - track bounds helpers  |
      | - vmg.py                  |   +-------------------------+
      | - weather.py              |
      | - comparison.py           |
      +-------------+-------------+
                    |
                    v
+---------------------------------------------------------------+
|                    Visualization Layer                        |
|             sailing_telemetry/visualization.py               |
|  - plot_speed_over_time()                                    |
|  - plot_twa_vs_vmg()                                         |
|  - plot_race_replay()                                        |
|  - plot_weather_over_time()                                  |
|  - plot_team_comparison()                                    |
|  - plot_anomalies()                                          |
|  - plot_vmg_bands()                                          |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|                     Plotly Figures -> UI                      |
|            Rendered in app sections/tabs/panels              |
+---------------------------------------------------------------+

Data utilities:
- sailing_telemetry/generator.py  (simulated telemetry source)
- sample_data/sailing_telemetry_sample.csv
- tests/ (schema + analytics validation)
```

Mobile quick view:

```text
[Data Source]
    |
    v
[Ingestion] -> [Schema Normalize/Validate]
    |
    v
[Canonical Telemetry DataFrame]
    |
    +--> [Analytics Modules]
    |       - metrics / maneuvers / anomalies
    |       - vmg / weather / comparison
    |
    +--> [Replay Layer]
    |
    v
[Visualization Layer (Plotly Figures)]
    |
    v
[Streamlit App Sections + Live Simulation]
```

## Telemetry schema

Canonical columns:

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

The schema module provides:

- `get_required_columns()`
- `get_optional_columns()`
- `validate_telemetry_dataframe(df)`
- `normalize_telemetry_dataframe(df)`

## Analytics modules

- `metrics.py`: KPI summaries, boat/leg aggregations, VMG efficiency
- `maneuvers.py`: explicit/inferred maneuver detection and recovery estimation
- `anomalies.py`: robust detection for speed, heel, VMG, and foil-instability
- `vmg.py`: TWA/wind banding, optimal zone estimation, target recommendations
- `weather.py`: weather summaries, gust windows, wind-shift periods
- `comparison.py`: boat/team comparison tables

## How to run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Optional: regenerate sample CSV

```bash
python export_sample_csv.py --output sample_data/sailing_telemetry_sample.csv
```

Optional: run tests

```bash
python -m pytest
```

## How to deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from this repo.
3. Use:
   - Branch: `master`
   - Main file: `app.py`
4. Confirm `runtime.txt` and `requirements.txt` are detected.
5. Deploy and monitor logs.

This repo pins Python via `runtime.txt` (`python-3.12`) to avoid wheel build
issues on unsupported interpreter versions.

## Example use cases

- Coach and crew maneuver debriefs
- Team-on-team comparative performance reviews
- Event analytics demos for race organizers
- Fan-facing replay prototypes with telemetry overlays

## Roadmap

- NMEA 0183 / NMEA 2000 adapter
- GPX import
- Expedition/B&G/Garmin/Raymarine/Sailmon import adapters
- WebSocket live telemetry ingestion
- cloud streaming backend
- coach debrief reports
- fan-facing race replay mode
- ML-assisted anomaly detection
- tactical routing simulation
- 3D replay

## Disclaimer

This project uses simulated data for demonstration and prototyping only.
It is not affiliated with SailGP.
