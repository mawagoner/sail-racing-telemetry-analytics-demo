## Sail Racing Telemetry Analytics Demo - Implementation Plan

### Goal
Build a polished Streamlit dashboard prototype named `Sail Racing Telemetry Analytics Demo` using Python, pandas, plotly, and streamlit.

### Project Structure
- `app.py` - Streamlit entry point and dashboard UI
- `data_generator.py` - sample telemetry data generation
- `analytics.py` - KPI calculations, leg comparison, maneuver detection/loss logic
- `requirements.txt` - runtime dependencies
- `README.md` - setup, run, and feature documentation

### Data Model
Sample telemetry must include these columns:
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

### Implementation Steps
1. Create telemetry generation module
   - Generate realistic race-session time series
   - Simulate alternating upwind/downwind legs
   - Inject tack/gybe transitions and temporary speed drops
   - Compute VMG from speed and true wind angle
   - Return clean pandas DataFrame with required columns

2. Create analytics module
   - Validate and normalize telemetry schema
   - Compute dashboard KPI cards:
     - max speed
     - average speed
     - best VMG
     - average maneuver loss
   - Build leg comparison aggregation table
   - Implement simple maneuver detection (tacks/gybes) from TWA sign changes
   - Estimate maneuver loss from pre-event vs post-event speed dip

3. Build Streamlit app
   - Page title and polished layout
   - Data source control (generated data or uploaded CSV)
   - Filters for boat and time window
   - KPI summary cards
   - Speed-over-time line chart (Plotly)
   - TWA vs VMG scatter plot (Plotly)
   - Leg comparison table
   - Maneuver detection summary panel (tacks/gybes + losses)
   - Short explanation panel linking metrics to high-performance sailing analytics

4. Add docs and dependencies
   - `requirements.txt` with minimal pinned dependencies
   - `README.md` with quickstart, features, and data schema details

5. Verify
   - Run a syntax/compile check (`python -m compileall .`)
   - Confirm files and app entry point are ready (`streamlit run app.py`)
