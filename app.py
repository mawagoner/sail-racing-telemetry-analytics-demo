"""Streamlit dashboard for sail racing telemetry analytics."""

from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics import (
    REQUIRED_COLUMNS,
    build_leg_comparison,
    build_maneuver_summary,
    compute_kpis,
    detect_maneuvers,
    validate_and_normalize_telemetry,
)
from data_generator import generate_sample_telemetry


st.set_page_config(page_title="Sail Racing Telemetry Analytics Demo", layout="wide")

st.markdown(
    """
<style>
@import url("https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap");

html, body, [class*="st-"] {
    font-family: "Barlow", "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 5%, rgba(238, 247, 255, 0.85) 0%, rgba(238, 247, 255, 0.0) 34%),
        linear-gradient(140deg, #f6f9fc 0%, #e3edf4 55%, #d4e1eb 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2f43 0%, #1b4e6c 100%);
}

section[data-testid="stSidebar"] * {
    color: #f2f8fc;
}

div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid #d6e1eb;
    border-radius: 12px;
    padding: 0.4rem 0.7rem;
}

div[data-testid="stMetricLabel"] {
    text-transform: uppercase;
    letter-spacing: 0.04rem;
    font-size: 0.78rem;
}

.chart-panel {
    background: rgba(255, 255, 255, 0.7);
    border-radius: 14px;
    border: 1px solid #d4dee8;
    padding: 0.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_generated_data(
    num_boats: int,
    leg_count: int,
    points_per_leg: int,
    freq_seconds: int,
    seed: int,
) -> pd.DataFrame:
    return generate_sample_telemetry(
        num_boats=num_boats,
        leg_count=leg_count,
        points_per_leg=points_per_leg,
        freq_seconds=freq_seconds,
        seed=seed,
    )


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def render() -> None:
    st.title("Sail Racing Telemetry Analytics Demo")
    st.caption(
        "Analyze race-session telemetry with KPI tracking, leg-by-leg comparisons, "
        "and automated tack/gybe loss detection."
    )

    with st.sidebar:
        st.header("Telemetry Source")
        source_mode = st.radio(
            "Choose data input",
            options=["Generated sample data", "Upload CSV"],
            index=0,
        )

        if source_mode == "Generated sample data":
            num_boats = st.slider("Boats", min_value=1, max_value=6, value=3)
            leg_count = st.slider("Legs per boat", min_value=4, max_value=12, value=6)
            points_per_leg = st.slider(
                "Samples per leg", min_value=60, max_value=240, value=120, step=10
            )
            freq_seconds = st.selectbox("Sampling interval (seconds)", [1, 2, 5], index=1)
            seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42)

            raw_data = load_generated_data(
                num_boats=num_boats,
                leg_count=leg_count,
                points_per_leg=points_per_leg,
                freq_seconds=freq_seconds,
                seed=int(seed),
            )
        else:
            uploaded_file = st.file_uploader("Upload telemetry CSV", type=["csv"])
            if uploaded_file is None:
                st.info("Upload a CSV file to start the dashboard.")
                st.stop()
            raw_data = load_uploaded_csv(uploaded_file.getvalue())

    try:
        telemetry = validate_and_normalize_telemetry(raw_data)
    except ValueError as exc:
        st.error(f"Telemetry validation error: {exc}")
        st.write("Required schema columns:")
        st.code("\n".join(REQUIRED_COLUMNS))
        st.stop()

    if telemetry.empty:
        st.warning("No valid telemetry rows are available after normalization.")
        st.stop()

    st.subheader("Session Filters")
    filter_col_a, filter_col_b = st.columns([1.0, 2.0])

    available_boats = sorted(telemetry["boat_id"].unique().tolist())
    with filter_col_a:
        selected_boats = st.multiselect(
            "Boat filter",
            options=available_boats,
            default=available_boats,
        )

    with filter_col_b:
        min_timestamp = telemetry["timestamp"].min().to_pydatetime()
        max_timestamp = telemetry["timestamp"].max().to_pydatetime()
        selected_window = st.slider(
            "Time window",
            min_value=min_timestamp,
            max_value=max_timestamp,
            value=(min_timestamp, max_timestamp),
            format="YYYY-MM-DD HH:mm:ss",
        )

    filtered = telemetry[telemetry["boat_id"].isin(selected_boats)].copy()
    filtered = filtered[
        (filtered["timestamp"] >= pd.Timestamp(selected_window[0]))
        & (filtered["timestamp"] <= pd.Timestamp(selected_window[1]))
    ]

    if filtered.empty:
        st.warning("No data remains after filters. Adjust boat or time selections.")
        st.stop()

    maneuvers = detect_maneuvers(filtered)
    kpis = compute_kpis(filtered, maneuvers)
    leg_comparison = build_leg_comparison(filtered)
    maneuver_summary = build_maneuver_summary(maneuvers)

    st.subheader("KPI Summary")
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Max Speed", f"{kpis['max_speed']:.2f} kn")
    kpi_cols[1].metric("Average Speed", f"{kpis['avg_speed']:.2f} kn")
    kpi_cols[2].metric("Best VMG", f"{kpis['best_vmg']:.2f} kn")
    kpi_cols[3].metric("Avg Maneuver Loss", f"{kpis['avg_maneuver_loss']:.2f} kn")

    color_sequence = ["#0f4060", "#1f7a8c", "#e0a458", "#4f6d7a", "#2c4a52", "#b16f3e"]

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        speed_fig = px.line(
            filtered,
            x="timestamp",
            y="speed_knots",
            color="boat_id",
            color_discrete_sequence=color_sequence,
            title="Speed Over Time",
            hover_data=["leg_id", "true_wind_angle", "vmg"],
        )
        speed_fig.update_traces(line={"width": 2})
        speed_fig.update_layout(
            template="plotly_white",
            height=390,
            xaxis_title="Time",
            yaxis_title="Speed (knots)",
            legend_title="Boat",
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
        )
        st.plotly_chart(speed_fig, use_container_width=True)

    with chart_col_2:
        scatter_fig = px.scatter(
            filtered,
            x="true_wind_angle",
            y="vmg",
            color="boat_id",
            symbol="leg_id",
            color_discrete_sequence=color_sequence,
            title="TWA vs VMG",
            hover_data=["timestamp", "speed_knots", "leg_id"],
            opacity=0.75,
        )
        scatter_fig.update_layout(
            template="plotly_white",
            height=390,
            xaxis_title="True Wind Angle (deg)",
            yaxis_title="VMG (knots)",
            legend_title="Boat",
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    st.subheader("Leg Comparison")
    st.dataframe(leg_comparison, use_container_width=True, hide_index=True)

    st.subheader("Maneuver Detection Summary")
    m_col_1, m_col_2, m_col_3, m_col_4 = st.columns(4)
    m_col_1.metric("Detected Tacks", f"{maneuver_summary['tack_count']}")
    m_col_2.metric("Detected Gybes", f"{maneuver_summary['gybe_count']}")
    m_col_3.metric("Average Loss", f"{maneuver_summary['avg_loss']:.2f} kn")
    m_col_4.metric("Worst Loss", f"{maneuver_summary['max_loss']:.2f} kn")

    if maneuvers.empty:
        st.info("No maneuvers detected in the current filter window.")
    else:
        display_maneuvers = maneuvers.copy()
        numeric_cols = ["speed_loss_knots", "baseline_speed_knots", "min_speed_knots"]
        display_maneuvers[numeric_cols] = display_maneuvers[numeric_cols].round(2)
        st.dataframe(display_maneuvers, use_container_width=True, hide_index=True)

    with st.expander("How these metrics support high-performance sailing analytics", expanded=True):
        st.markdown(
            """
- **Speed profile** highlights raw boat handling quality through each leg.
- **VMG behavior** reveals efficiency in converting speed into progress toward race marks.
- **Leg comparison** surfaces setup and trim differences between upwind and downwind phases.
- **Maneuver loss** quantifies the speed penalty of each tack or gybe so teams can improve transitions.
"""
        )


if __name__ == "__main__":
    render()
