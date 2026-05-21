"""Streamlit dashboard for Sail Racing Telemetry Analytics Demo."""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    REQUIRED_COLUMNS,
    build_anomaly_summary,
    build_leg_comparison,
    build_maneuver_summary,
    build_team_comparison,
    compute_kpis,
    compute_optimal_vmg_zones,
    detect_anomalies,
    detect_maneuvers,
    detect_weather_periods,
    extract_weather_timeseries,
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
    color: #0f2536;
}

.stApp {
    background:
        radial-gradient(circle at 8% 5%, rgba(235, 248, 255, 0.88) 0%, rgba(235, 248, 255, 0.0) 34%),
        linear-gradient(140deg, #f6f9fc 0%, #e3edf4 55%, #d3e0ea 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2f43 0%, #1a4f6d 100%);
}

section[data-testid="stSidebar"] * {
    color: #f3f8fb;
}

.main .block-container {
    background: rgba(248, 252, 255, 0.74);
    border-radius: 18px;
    padding: 1.2rem 1.5rem 1.6rem 1.5rem;
}

h1, h2, h3, h4, h5, h6,
p,
label,
li,
span,
div[data-testid="stMarkdownContainer"],
div[data-testid="stWidgetLabel"] {
    color: #0f2536;
}

button[role="tab"] {
    color: #4f6474 !important;
}

button[role="tab"][aria-selected="true"] {
    color: #0f4060 !important;
}

div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.84);
    border: 1px solid #d7e2ec;
    border-radius: 12px;
    padding: 0.4rem 0.7rem;
}

div[data-testid="stMetricLabel"] {
    text-transform: uppercase;
    letter-spacing: 0.05rem;
    font-size: 0.78rem;
}

div[data-testid="stMetricValue"] {
    color: #0f2536;
}

div[data-baseweb="select"] > div {
    background: #f7fbff !important;
    border: 1px solid #9eb4c5 !important;
    color: #0f2536 !important;
}

div[data-baseweb="select"] input,
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {
    color: #0f2536 !important;
    -webkit-text-fill-color: #0f2536 !important;
}

div[data-baseweb="select"] svg {
    fill: #1f4b66 !important;
}

div[data-baseweb="popover"] [role="listbox"],
[role="listbox"] {
    background: #0f2536 !important;
    border: 1px solid #2e5a74 !important;
}

div[data-baseweb="popover"] [role="option"],
[role="listbox"] [role="option"],
[role="option"] {
    background: #0f2536 !important;
    color: #f4fbff !important;
    -webkit-text-fill-color: #f4fbff !important;
}

[role="listbox"] [role="option"]:hover,
[role="option"]:hover {
    background: #23455d !important;
    color: #ffffff !important;
}

[role="listbox"] [role="option"][aria-selected="true"],
[role="option"][aria-selected="true"] {
    background: #315a76 !important;
    color: #ffffff !important;
}

div[data-testid="stTabs"] button {
    font-weight: 600;
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


def get_color_map(boats: list[str]) -> dict[str, str]:
    palette = [
        "#0f4060",
        "#1f7a8c",
        "#e0a458",
        "#4f6d7a",
        "#2c4a52",
        "#b16f3e",
        "#4e7d6f",
        "#7f5a58",
    ]
    return {boat: palette[idx % len(palette)] for idx, boat in enumerate(sorted(boats))}


def add_weather_overlays(fig: go.Figure, weather_periods: dict[str, pd.DataFrame]) -> None:
    gust_periods = weather_periods.get("gust_periods", pd.DataFrame())
    shift_periods = weather_periods.get("shift_periods", pd.DataFrame())

    for _, row in gust_periods.iterrows():
        fig.add_vrect(
            x0=row["start"],
            x1=row["end"],
            fillcolor="#f4b942",
            opacity=0.10,
            line_width=0,
            annotation_text="Gust",
            annotation_position="top left",
        )

    for _, row in shift_periods.iterrows():
        fig.add_vrect(
            x0=row["start"],
            x1=row["end"],
            fillcolor="#8bb2c9",
            opacity=0.08,
            line_width=0,
        )


def build_speed_chart(
    telemetry: pd.DataFrame,
    maneuvers: pd.DataFrame,
    anomalies: pd.DataFrame,
    weather_periods: dict[str, pd.DataFrame],
    show_maneuver_markers: bool,
    show_anomaly_markers: bool,
    show_weather_shading: bool,
) -> go.Figure:
    boats = sorted(telemetry["boat_id"].unique().tolist())
    color_map = get_color_map(boats)
    fig = go.Figure()

    for boat in boats:
        boat_df = telemetry[telemetry["boat_id"] == boat]
        fig.add_trace(
            go.Scatter(
                x=boat_df["timestamp"],
                y=boat_df["speed_knots"],
                mode="lines",
                name=boat,
                line={"color": color_map[boat], "width": 2.2},
                hovertemplate=(
                    "Boat: %{customdata[0]}<br>"
                    "Time: %{x}<br>"
                    "Speed: %{y:.2f} kn<br>"
                    "VMG: %{customdata[1]:.2f} kn<br>"
                    "TWA: %{customdata[2]:.1f} deg<extra></extra>"
                ),
                customdata=np.column_stack(
                    [
                        boat_df["boat_id"],
                        boat_df["vmg"],
                        boat_df["true_wind_angle"],
                    ]
                ),
            )
        )

    if show_maneuver_markers and not maneuvers.empty:
        marker_colors = maneuvers["maneuver_type"].map(
            {"tack": "#ef7f45", "gybe": "#2e8b57"}
        ).fillna("#394f62")

        customdata = np.column_stack(
            [
                maneuvers["boat_id"],
                maneuvers["maneuver_type"],
                maneuvers["entry_speed_knots"],
                maneuvers["exit_speed_knots"],
                maneuvers["speed_loss_knots"],
                maneuvers["recovery_time_sec"],
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=maneuvers["timestamp"],
                y=maneuvers["min_speed_knots"],
                mode="markers",
                marker={"size": 10, "symbol": "diamond", "color": marker_colors.tolist()},
                name="Maneuver Events",
                customdata=customdata,
                hovertemplate=(
                    "Boat: %{customdata[0]}<br>"
                    "Time: %{x}<br>"
                    "Type: %{customdata[1]}<br>"
                    "Entry Speed: %{customdata[2]:.2f} kn<br>"
                    "Exit Speed: %{customdata[3]:.2f} kn<br>"
                    "Speed Loss: %{customdata[4]:.2f} kn<br>"
                    "Recovery: %{customdata[5]:.1f} s<extra></extra>"
                ),
            )
        )

        for _, event in maneuvers.iterrows():
            fig.add_vline(
                x=event["timestamp"],
                line={
                    "color": "#ef7f45" if event["maneuver_type"] == "tack" else "#2e8b57",
                    "dash": "dot",
                    "width": 1,
                },
                opacity=0.25,
            )

    if show_anomaly_markers and not anomalies.empty:
        speed_anomalies = anomalies[
            anomalies["anomaly_type"].isin(["speed_drop", "foil_instability"])
        ]
        if not speed_anomalies.empty:
            fig.add_trace(
                go.Scatter(
                    x=speed_anomalies["timestamp"],
                    y=speed_anomalies["speed_knots"],
                    mode="markers",
                    name="Speed Anomalies",
                    marker={"symbol": "x", "size": 11, "color": "#a93226", "line": {"width": 1}},
                    customdata=np.column_stack(
                        [
                            speed_anomalies["boat_id"],
                            speed_anomalies["anomaly_type"],
                            speed_anomalies["description"],
                        ]
                    ),
                    hovertemplate=(
                        "Boat: %{customdata[0]}<br>"
                        "Time: %{x}<br>"
                        "Anomaly: %{customdata[1]}<br>"
                        "%{customdata[2]}<extra></extra>"
                    ),
                )
            )

    if show_weather_shading:
        add_weather_overlays(fig, weather_periods)

    fig.update_layout(
        template="plotly_white",
        height=420,
        title="Speed Over Time with Maneuver/Anomaly Markers",
        xaxis_title="Timestamp",
        yaxis_title="Speed (knots)",
        legend_title="Trace",
        margin={"l": 20, "r": 20, "t": 68, "b": 26},
    )
    return fig


def build_twa_vmg_chart(
    telemetry: pd.DataFrame,
    anomalies: pd.DataFrame,
    recommendations: pd.DataFrame,
    show_anomaly_markers: bool,
) -> go.Figure:
    boats = sorted(telemetry["boat_id"].unique().tolist())
    color_map = get_color_map(boats)
    fig = go.Figure()

    for boat in boats:
        boat_df = telemetry[telemetry["boat_id"] == boat]
        chart_customdata = np.array(
            list(
                zip(
                    boat_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                    boat_df["speed_knots"],
                    boat_df["leg_mode"],
                )
            ),
            dtype=object,
        )
        fig.add_trace(
            go.Scatter(
                x=boat_df["true_wind_angle"],
                y=boat_df["vmg"],
                mode="markers",
                name=boat,
                marker={"size": 7, "opacity": 0.64, "color": color_map[boat]},
                customdata=chart_customdata,
                hovertemplate=(
                    "Boat: " + boat + "<br>"
                    "TWA: %{x:.1f} deg<br>"
                    "VMG: %{y:.2f} kn<br>"
                    "Speed: %{customdata[1]:.2f} kn<br>"
                    "Leg Mode: %{customdata[2]}<br>"
                    "Time: %{customdata[0]}<extra></extra>"
                ),
            )
        )

    if show_anomaly_markers and not anomalies.empty:
        poor_vmg = anomalies[anomalies["anomaly_type"] == "poor_vmg"]
        if not poor_vmg.empty:
            merged = poor_vmg.merge(
                telemetry[["timestamp", "boat_id", "true_wind_angle"]],
                on=["timestamp", "boat_id"],
                how="left",
            )
            poor_vmg_customdata = np.array(
                list(
                    zip(
                        merged["boat_id"],
                        merged["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                ),
                dtype=object,
            )
            fig.add_trace(
                go.Scatter(
                    x=merged["true_wind_angle"],
                    y=merged["vmg"],
                    mode="markers",
                    name="Poor VMG Alerts",
                    marker={"symbol": "x", "size": 11, "color": "#8e2a2a"},
                    hovertemplate=(
                        "Boat: %{customdata[0]}<br>"
                        "Time: %{customdata[1]}<br>"
                        "Poor VMG anomaly<extra></extra>"
                    ),
                    customdata=poor_vmg_customdata,
                )
            )

    if not recommendations.empty:
        for _, rec in recommendations.iterrows():
            range_text = str(rec["twa_range"]).replace(" deg", "")
            start_str, end_str = range_text.split("-")
            start_val = float(start_str)
            end_val = float(end_str)
            color = "#5f8f3d" if "Upwind" in rec["zone"] else "#8b6b38"

            fig.add_vrect(
                x0=start_val,
                x1=end_val,
                fillcolor=color,
                opacity=0.08,
                line_width=0,
            )
            fig.add_vrect(
                x0=-end_val,
                x1=-start_val,
                fillcolor=color,
                opacity=0.08,
                line_width=0,
            )

    fig.update_layout(
        template="plotly_white",
        height=420,
        title="True Wind Angle vs VMG with Suggested Target Zones",
        xaxis_title="True Wind Angle (deg)",
        yaxis_title="VMG (knots)",
        margin={"l": 20, "r": 20, "t": 68, "b": 26},
    )
    return fig


def build_weather_chart(weather: pd.DataFrame, weather_periods: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weather["timestamp"],
            y=weather["true_wind_speed"],
            mode="lines",
            name="True Wind Speed",
            line={"color": "#1f7a8c", "width": 2.1},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=weather["timestamp"],
            y=weather["gust_knots"],
            mode="lines",
            name="Gust Estimate",
            line={"color": "#f49c37", "width": 1.8, "dash": "dash"},
        )
    )

    add_weather_overlays(fig, weather_periods)
    fig.update_layout(
        template="plotly_white",
        height=300,
        title="Simulated Course Weather Overlay",
        xaxis_title="Timestamp",
        yaxis_title="Wind (knots)",
        margin={"l": 20, "r": 20, "t": 60, "b": 22},
    )
    return fig


def build_race_replay_figure(
    telemetry: pd.DataFrame,
    selected_timestamp: pd.Timestamp,
    boats: list[str],
    color_by_leg: bool,
    show_wind_arrows: bool,
    weather: pd.DataFrame,
) -> go.Figure:
    history = telemetry[(telemetry["boat_id"].isin(boats)) & (telemetry["timestamp"] <= selected_timestamp)].copy()
    history = history.sort_values(["boat_id", "timestamp"])
    latest_positions = history.groupby("boat_id", as_index=False).tail(1)

    fig = go.Figure()
    boat_colors = get_color_map(boats)

    if color_by_leg:
        leg_palette = ["#0f4060", "#2e8b57", "#e0a458", "#9d6f7a", "#6b7ea5", "#4f6d7a"]
        leg_ids = sorted(history["leg_id"].unique().tolist())
        leg_colors = {leg: leg_palette[idx % len(leg_palette)] for idx, leg in enumerate(leg_ids)}

        for (boat, leg_id), leg_df in history.groupby(["boat_id", "leg_id"], sort=False):
            fig.add_trace(
                go.Scatter(
                    x=leg_df["x_position"],
                    y=leg_df["y_position"],
                    mode="lines",
                    name=f"{boat} L{leg_id}",
                    line={"width": 2.0, "color": leg_colors.get(leg_id, "#4f6d7a")},
                    legendgroup=boat,
                    showlegend=False,
                )
            )
    else:
        for boat, boat_df in history.groupby("boat_id", sort=False):
            fig.add_trace(
                go.Scatter(
                    x=boat_df["x_position"],
                    y=boat_df["y_position"],
                    mode="lines",
                    name=boat,
                    line={"width": 2.3, "color": boat_colors.get(boat, "#0f4060")},
                )
            )

    fig.add_trace(
        go.Scatter(
            x=latest_positions["x_position"],
            y=latest_positions["y_position"],
            mode="markers+text",
            text=latest_positions["boat_id"],
            textposition="top center",
            marker={"size": 13, "symbol": "circle", "color": "#102a43", "line": {"width": 1.5, "color": "#ffffff"}},
            name="Current Position",
            customdata=np.column_stack(
                [
                    latest_positions["speed_knots"],
                    latest_positions["vmg"],
                    latest_positions["leg_id"],
                ]
            ),
            hovertemplate=(
                "Boat: %{text}<br>"
                "Speed: %{customdata[0]:.2f} kn<br>"
                "VMG: %{customdata[1]:.2f} kn<br>"
                "Leg: %{customdata[2]}<extra></extra>"
            ),
        )
    )

    if show_wind_arrows and not history.empty and not weather.empty:
        current_weather = weather[weather["timestamp"] <= selected_timestamp].tail(1)
        if not current_weather.empty:
            wind_dir = float(current_weather.iloc[0]["true_wind_direction"])
            wind_to = (wind_dir + 180.0) % 360.0

            x_min = float(history["x_position"].min())
            x_max = float(history["x_position"].max())
            y_min = float(history["y_position"].min())
            y_max = float(history["y_position"].max())

            x_span = max(x_max - x_min, 200.0)
            y_span = max(y_max - y_min, 200.0)
            arrow_len = x_span * 0.12

            dx = arrow_len * np.sin(np.deg2rad(wind_to))
            dy = arrow_len * np.cos(np.deg2rad(wind_to))
            anchor_y = y_max + 0.12 * y_span

            for frac in [0.2, 0.5, 0.8]:
                anchor_x = x_min + frac * x_span
                fig.add_annotation(
                    x=anchor_x + dx,
                    y=anchor_y + dy,
                    ax=anchor_x,
                    ay=anchor_y,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1.2,
                    arrowwidth=1.5,
                    arrowcolor="#3f5f73",
                    text="",
                )

            fig.add_annotation(
                x=x_min + 0.5 * x_span,
                y=anchor_y + dy + 0.06 * y_span,
                text=f"Wind {wind_dir:.0f} deg",
                showarrow=False,
                font={"size": 12, "color": "#29495d"},
            )

    fig.update_layout(
        template="plotly_white",
        title="Race Replay - Boat Tracks and Current Positions",
        xaxis_title="Course X Position (m)",
        yaxis_title="Course Y Position (m)",
        height=520,
        margin={"l": 20, "r": 20, "t": 68, "b": 26},
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def anomaly_explanations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "anomaly_type": "speed_drop",
                "meaning": "Sharp speed decrease compared to local rolling baseline.",
            },
            {
                "anomaly_type": "abnormal_heel",
                "meaning": "Heel angle exceeds robust control envelope for current session.",
            },
            {
                "anomaly_type": "poor_vmg",
                "meaning": "VMG underperforms compared to similar TWA/wind conditions.",
            },
            {
                "anomaly_type": "foil_instability",
                "meaning": "Combined foil-cant oscillation, heel load, and speed drop signal.",
            },
        ]
    )


def initialize_live_state() -> None:
    if "live_running" not in st.session_state:
        st.session_state.live_running = False
    if "live_index" not in st.session_state:
        st.session_state.live_index = 0
    if "live_boats" not in st.session_state:
        st.session_state.live_boats = tuple()


def render() -> None:
    st.title("Sail Racing Telemetry Analytics Demo")
    st.caption(
        "SailGP-inspired prototype: transforming simulated telemetry into actionable race-performance insight."
    )

    with st.sidebar:
        st.header("Telemetry Source")
        source_mode = st.radio(
            "Choose data input",
            options=["Generated sample data", "Upload CSV"],
            index=0,
        )

        if source_mode == "Generated sample data":
            num_boats = st.slider("Boats", min_value=2, max_value=8, value=4)
            leg_count = st.slider("Legs per boat", min_value=4, max_value=12, value=6)
            points_per_leg = st.slider(
                "Samples per leg", min_value=60, max_value=260, value=120, step=10
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
        st.write("Expected schema columns:")
        st.code("\n".join(REQUIRED_COLUMNS))
        st.stop()

    if telemetry.empty:
        st.warning("No valid telemetry rows are available after normalization.")
        st.stop()

    st.subheader("Session Filters")
    filter_col_a, filter_col_b = st.columns([1.0, 2.2])

    available_boats = sorted(telemetry["boat_id"].unique().tolist())
    with filter_col_a:
        selected_boats = st.multiselect(
            "Boat selection",
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
    anomalies = detect_anomalies(filtered)
    kpis = compute_kpis(filtered, maneuvers)
    leg_comparison = build_leg_comparison(filtered)
    maneuver_summary = build_maneuver_summary(maneuvers)
    anomaly_summary = build_anomaly_summary(anomalies)
    weather = extract_weather_timeseries(filtered)
    weather_periods = detect_weather_periods(weather)
    optimal_zone_result = compute_optimal_vmg_zones(filtered)
    optimal_recommendations = optimal_zone_result.get("recommendations", pd.DataFrame())

    tab_overview, tab_replay, tab_live, tab_compare, tab_optimal = st.tabs(
        [
            "Performance Overview",
            "Race Replay",
            "Live Streaming Simulation",
            "Team-vs-Team Comparison",
            "Predictive Optimal VMG Zone",
        ]
    )

    with tab_overview:
        st.markdown(
            "This view turns raw telemetry into race-performance insight: speed, VMG quality, maneuvers, weather context, and anomaly alerts."
        )

        kpi_cols = st.columns(5)
        kpi_cols[0].metric("Max Speed", f"{kpis['max_speed']:.2f} kn")
        kpi_cols[1].metric("Average Speed", f"{kpis['avg_speed']:.2f} kn")
        kpi_cols[2].metric("Best VMG", f"{kpis['best_vmg']:.2f} kn")
        kpi_cols[3].metric("Avg Maneuver Loss", f"{kpis['avg_maneuver_loss']:.2f} kn")
        kpi_cols[4].metric("Anomalies", f"{anomaly_summary['total']}")

        toggles = st.columns(3)
        show_maneuver_markers = toggles[0].checkbox("Show maneuver markers", value=True)
        show_anomaly_markers = toggles[1].checkbox("Show anomaly markers", value=True)
        show_weather_shading = toggles[2].checkbox("Shade gust/shift periods", value=True)

        chart_col_a, chart_col_b = st.columns(2)
        with chart_col_a:
            speed_fig = build_speed_chart(
                telemetry=filtered,
                maneuvers=maneuvers,
                anomalies=anomalies,
                weather_periods=weather_periods,
                show_maneuver_markers=show_maneuver_markers,
                show_anomaly_markers=show_anomaly_markers,
                show_weather_shading=show_weather_shading,
            )
            st.plotly_chart(speed_fig, use_container_width=True)

        with chart_col_b:
            twa_fig = build_twa_vmg_chart(
                telemetry=filtered,
                anomalies=anomalies,
                recommendations=optimal_recommendations,
                show_anomaly_markers=show_anomaly_markers,
            )
            st.plotly_chart(twa_fig, use_container_width=True)

        if not weather.empty:
            weather_fig = build_weather_chart(weather, weather_periods)
            st.plotly_chart(weather_fig, use_container_width=True)

        st.subheader("Leg Comparison")
        st.dataframe(leg_comparison, use_container_width=True, hide_index=True)

        st.subheader("Maneuver Detection Summary")
        m_cols = st.columns(5)
        m_cols[0].metric("Tacks", f"{maneuver_summary['tack_count']}")
        m_cols[1].metric("Gybes", f"{maneuver_summary['gybe_count']}")
        m_cols[2].metric("Avg Loss", f"{maneuver_summary['avg_loss']:.2f} kn")
        m_cols[3].metric("Worst Loss", f"{maneuver_summary['max_loss']:.2f} kn")
        m_cols[4].metric("Avg Recovery", f"{maneuver_summary['avg_recovery']:.1f} s")

        if maneuvers.empty:
            st.info("No maneuvers detected in the current filter window.")
        else:
            st.dataframe(maneuvers.round(2), use_container_width=True, hide_index=True)

        st.subheader("Anomaly Detection")
        a_cols = st.columns(5)
        a_cols[0].metric("Total", f"{anomaly_summary['total']}")
        a_cols[1].metric("Speed Drops", f"{anomaly_summary['speed_drop']}")
        a_cols[2].metric("Abnormal Heel", f"{anomaly_summary['abnormal_heel']}")
        a_cols[3].metric("Poor VMG", f"{anomaly_summary['poor_vmg']}")
        a_cols[4].metric("Foil Instability", f"{anomaly_summary['foil_instability']}")

        if anomalies.empty:
            st.info("No anomalies flagged with current filters.")
        else:
            st.dataframe(anomalies.round(3), use_container_width=True, hide_index=True)
        st.dataframe(anomaly_explanations(), use_container_width=True, hide_index=True)

    with tab_replay:
        st.markdown(
            "Interactive race replay based on simulated course coordinates. Use this to inspect tactical lines and timing under changing wind."
        )

        replay_boats = st.multiselect(
            "Replay boats",
            options=available_boats,
            default=selected_boats if selected_boats else available_boats,
            key="replay_boats",
        )

        if not replay_boats:
            st.info("Select at least one boat for race replay.")
        else:
            replay_timestamps = filtered[filtered["boat_id"].isin(replay_boats)]["timestamp"].sort_values().unique()
            replay_index = st.slider(
                "Replay timestamp",
                min_value=0,
                max_value=len(replay_timestamps) - 1,
                value=len(replay_timestamps) - 1,
                key="replay_index",
            )
            replay_ts = pd.Timestamp(replay_timestamps[replay_index])

            replay_toggles = st.columns(2)
            color_by_leg = replay_toggles[0].checkbox("Color by leg", value=False)
            show_wind_arrows = replay_toggles[1].checkbox("Show wind direction arrows", value=True)

            replay_fig = build_race_replay_figure(
                telemetry=filtered,
                selected_timestamp=replay_ts,
                boats=replay_boats,
                color_by_leg=color_by_leg,
                show_wind_arrows=show_wind_arrows,
                weather=weather,
            )
            st.plotly_chart(replay_fig, use_container_width=True)

            current_positions = (
                filtered[
                    (filtered["boat_id"].isin(replay_boats))
                    & (filtered["timestamp"] <= replay_ts)
                ]
                .sort_values("timestamp")
                .groupby("boat_id", as_index=False)
                .tail(1)
                .sort_values("boat_id")
            )

            if not current_positions.empty:
                st.dataframe(
                    current_positions[
                        [
                            "timestamp",
                            "boat_id",
                            "team_name",
                            "leg_id",
                            "leg_mode",
                            "speed_knots",
                            "vmg",
                            "x_position",
                            "y_position",
                            "true_wind_speed",
                            "true_wind_direction",
                            "gust_knots",
                        ]
                    ].round(3),
                    use_container_width=True,
                    hide_index=True,
                )

    with tab_live:
        initialize_live_state()
        st.markdown(
            "Simulated live mode replays telemetry as if packets are arriving in real time."
        )

        live_boats = st.multiselect(
            "Live boats",
            options=available_boats,
            default=selected_boats if selected_boats else available_boats,
            key="live_boat_selection",
        )
        live_enabled = st.checkbox("Enable live streaming simulation", value=False, key="live_enabled")
        playback_speed = st.slider("Playback speed", min_value=1, max_value=8, value=2)
        live_window_points = st.slider("Live chart window (samples)", 40, 600, 220)

        if tuple(live_boats) != st.session_state.live_boats:
            st.session_state.live_boats = tuple(live_boats)
            st.session_state.live_index = 0
            st.session_state.live_running = False

        if not live_boats:
            st.info("Select at least one boat to run live simulation.")
        else:
            live_df = filtered[filtered["boat_id"].isin(live_boats)].sort_values("timestamp")
            live_timestamps = live_df["timestamp"].sort_values().unique()

            if len(live_timestamps) == 0:
                st.info("No timestamps available for live simulation.")
            else:
                max_index = len(live_timestamps) - 1
                st.session_state.live_index = int(min(st.session_state.live_index, max_index))

                btn_cols = st.columns(3)
                if btn_cols[0].button("Start", use_container_width=True):
                    st.session_state.live_running = True
                if btn_cols[1].button("Stop", use_container_width=True):
                    st.session_state.live_running = False
                if btn_cols[2].button("Reset", use_container_width=True):
                    st.session_state.live_running = False
                    st.session_state.live_index = 0

                manual_index = st.slider(
                    "Current replay step",
                    min_value=0,
                    max_value=max_index,
                    value=st.session_state.live_index,
                    key="manual_live_step",
                    disabled=st.session_state.live_running and live_enabled,
                )
                if not (st.session_state.live_running and live_enabled):
                    st.session_state.live_index = int(manual_index)

                current_ts = pd.Timestamp(live_timestamps[st.session_state.live_index])
                st.write(f"Current live timestamp: `{current_ts}`")

                live_snapshot = live_df[live_df["timestamp"] == current_ts]
                live_history = live_df[live_df["timestamp"] <= current_ts]

                live_kpis = compute_kpis(live_history)
                live_kpi_cols = st.columns(4)
                live_kpi_cols[0].metric("Latest Max Speed", f"{live_kpis['max_speed']:.2f} kn")
                live_kpi_cols[1].metric("Latest Avg Speed", f"{live_kpis['avg_speed']:.2f} kn")
                live_kpi_cols[2].metric("Latest Best VMG", f"{live_kpis['best_vmg']:.2f} kn")
                live_kpi_cols[3].metric(
                    "Telemetry Rows",
                    f"{len(live_history)}",
                )

                history_timestamps = live_history["timestamp"].sort_values().unique()
                start_idx = max(0, len(history_timestamps) - live_window_points)
                window_start = pd.Timestamp(history_timestamps[start_idx])
                live_window = live_history[live_history["timestamp"] >= window_start]

                live_fig = px.line(
                    live_window,
                    x="timestamp",
                    y="speed_knots",
                    color="boat_id",
                    title="Live Speed Window",
                    hover_data=["vmg", "leg_id", "true_wind_angle"],
                )
                live_fig.update_layout(
                    template="plotly_white",
                    height=360,
                    xaxis_title="Timestamp",
                    yaxis_title="Speed (knots)",
                    margin={"l": 20, "r": 20, "t": 56, "b": 20},
                )
                st.plotly_chart(live_fig, use_container_width=True)

                st.dataframe(
                    live_snapshot[
                        [
                            "timestamp",
                            "boat_id",
                            "team_name",
                            "speed_knots",
                            "vmg",
                            "true_wind_speed",
                            "gust_knots",
                            "heel_angle",
                            "foil_cant",
                        ]
                    ].round(3),
                    use_container_width=True,
                    hide_index=True,
                )

                if live_enabled and st.session_state.live_running:
                    if st.session_state.live_index < max_index:
                        st.session_state.live_index = min(
                            max_index,
                            st.session_state.live_index + playback_speed,
                        )
                        time.sleep(max(0.05, 0.42 / playback_speed))
                        st.rerun()
                    else:
                        st.session_state.live_running = False

    with tab_compare:
        st.markdown(
            "Choose two boats/teams for tactical and performance benchmarking across speed, VMG, maneuvers, and anomaly burden."
        )

        if len(available_boats) < 2:
            st.info("Need at least two boats to run comparison.")
        else:
            default_a = available_boats[0]
            default_b = available_boats[1]
            comp_col_a, comp_col_b = st.columns(2)
            with comp_col_a:
                boat_a = st.selectbox("Team A", options=available_boats, index=0)
            with comp_col_b:
                second_index = 1 if len(available_boats) > 1 else 0
                boat_b = st.selectbox("Team B", options=available_boats, index=second_index)

            if boat_a == boat_b:
                st.warning("Select two different boats for comparison.")
            else:
                comparison_df = build_team_comparison(filtered, maneuvers, anomalies, boat_a, boat_b)

                metric_map = {row["metric"]: row for _, row in comparison_df.iterrows()}
                left_col, right_col = st.columns(2)
                with left_col:
                    st.markdown(f"#### {boat_a}")
                    st.metric("Average Speed", f"{metric_map['Average Speed'][boat_a]:.2f} kn")
                    st.metric("Max Speed", f"{metric_map['Max Speed'][boat_a]:.2f} kn")
                    st.metric("Average VMG", f"{metric_map['Average VMG'][boat_a]:.2f} kn")
                    st.metric("Maneuver Count", f"{metric_map['Maneuver Count'][boat_a]:.0f}")
                    st.metric("Anomaly Count", f"{metric_map['Anomaly Count'][boat_a]:.0f}")

                with right_col:
                    st.markdown(f"#### {boat_b}")
                    st.metric("Average Speed", f"{metric_map['Average Speed'][boat_b]:.2f} kn")
                    st.metric("Max Speed", f"{metric_map['Max Speed'][boat_b]:.2f} kn")
                    st.metric("Average VMG", f"{metric_map['Average VMG'][boat_b]:.2f} kn")
                    st.metric("Maneuver Count", f"{metric_map['Maneuver Count'][boat_b]:.0f}")
                    st.metric("Anomaly Count", f"{metric_map['Anomaly Count'][boat_b]:.0f}")

                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                grouped_bar_df = comparison_df.melt(
                    id_vars="metric", var_name="boat_id", value_name="value"
                )
                comparison_fig = px.bar(
                    grouped_bar_df,
                    x="metric",
                    y="value",
                    color="boat_id",
                    barmode="group",
                    title="Team-vs-Team Comparison",
                )
                comparison_fig.update_layout(
                    template="plotly_white",
                    height=430,
                    xaxis_title="Metric",
                    yaxis_title="Value",
                    margin={"l": 20, "r": 20, "t": 60, "b": 60},
                )
                st.plotly_chart(comparison_fig, use_container_width=True)

    with tab_optimal:
        st.markdown(
            "Simplified predictive model estimating target TWA bands that maximize progress VMG in upwind and downwind modes."
        )

        bucket_stats = optimal_zone_result.get("bucket_stats", pd.DataFrame())
        recommendations = optimal_zone_result.get("recommendations", pd.DataFrame())
        wind_band_recommendations = optimal_zone_result.get("wind_band_recommendations", pd.DataFrame())

        optimal_fig = build_twa_vmg_chart(
            telemetry=filtered,
            anomalies=pd.DataFrame(),
            recommendations=recommendations,
            show_anomaly_markers=False,
        )
        st.plotly_chart(optimal_fig, use_container_width=True)

        if not recommendations.empty:
            st.subheader("Recommended Target TWA Ranges")
            st.dataframe(recommendations, use_container_width=True, hide_index=True)

        if not wind_band_recommendations.empty:
            st.subheader("Best TWA Band by Wind Regime")
            st.dataframe(wind_band_recommendations, use_container_width=True, hide_index=True)

        if not bucket_stats.empty:
            st.subheader("Average VMG by TWA Bucket")
            st.dataframe(bucket_stats, use_container_width=True, hide_index=True)

        st.markdown(
            "In a real campaign, analysts can use this style of view to align target angles, trim modes, and maneuver timing with the wind state observed in race replay."
        )


if __name__ == "__main__":
    render()
