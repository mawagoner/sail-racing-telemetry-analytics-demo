"""Reusable Plotly visualizations for telemetry analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .analytics.weather import summarize_weather
from .replay import calculate_track_bounds, get_replay_frame, get_replay_history


def _boat_color_map(boats: list[str]) -> dict[str, str]:
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


def plot_speed_over_time(
    df: pd.DataFrame,
    maneuvers: pd.DataFrame | None = None,
    anomalies: pd.DataFrame | None = None,
) -> go.Figure:
    """Plot speed timeseries with optional maneuver and anomaly markers."""
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="Speed Over Time")
        return fig

    boats = sorted(df["boat_id"].unique().tolist())
    color_map = _boat_color_map(boats)

    for boat in boats:
        boat_df = df[df["boat_id"] == boat]
        fig.add_trace(
            go.Scatter(
                x=boat_df["timestamp"],
                y=boat_df["speed_knots"],
                mode="lines",
                name=boat,
                line={"color": color_map[boat], "width": 2.1},
            )
        )

    if maneuvers is not None and not maneuvers.empty:
        marker_colors = maneuvers["maneuver_type"].map({"tack": "#ef7f45", "gybe": "#2e8b57"}).fillna("#445a6d")
        fig.add_trace(
            go.Scatter(
                x=maneuvers["timestamp"],
                y=maneuvers["min_speed_knots"],
                mode="markers",
                marker={"size": 9, "symbol": "diamond", "color": marker_colors.tolist()},
                name="Maneuvers",
                hovertemplate=(
                    "Boat: %{customdata[0]}<br>"
                    "Type: %{customdata[1]}<br>"
                    "Entry: %{customdata[2]:.2f} kn<br>"
                    "Exit: %{customdata[3]:.2f} kn<br>"
                    "Loss: %{customdata[4]:.2f} kn<br>"
                    "Recovery: %{customdata[5]:.1f} s<extra></extra>"
                ),
                customdata=np.array(
                    list(
                        zip(
                            maneuvers["boat_id"],
                            maneuvers["maneuver_type"],
                            maneuvers["entry_speed_knots"],
                            maneuvers["exit_speed_knots"],
                            maneuvers["speed_loss_knots"],
                            maneuvers["recovery_time_sec"],
                        )
                    ),
                    dtype=object,
                ),
            )
        )

    if anomalies is not None and not anomalies.empty:
        speed_anomalies = anomalies[anomalies["anomaly_type"].isin(["speed_drop", "foil_instability"])]
        if not speed_anomalies.empty:
            merged = speed_anomalies.merge(
                df[["timestamp", "boat_id", "speed_knots"]], on=["timestamp", "boat_id"], how="left"
            )
            fig.add_trace(
                go.Scatter(
                    x=merged["timestamp"],
                    y=merged["speed_knots"],
                    mode="markers",
                    marker={"symbol": "x", "size": 10, "color": "#9d2f2f"},
                    name="Speed Anomalies",
                    hovertemplate=(
                        "Boat: %{customdata[0]}<br>"
                        "Anomaly: %{customdata[1]}<br>"
                        "%{customdata[2]}<extra></extra>"
                    ),
                    customdata=np.array(
                        list(zip(merged["boat_id"], merged["anomaly_type"], merged["explanation"])),
                        dtype=object,
                    ),
                )
            )

    fig.update_layout(
        template="plotly_white",
        title="Speed Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Speed (knots)",
        height=420,
    )
    return fig


def plot_twa_vs_vmg(df: pd.DataFrame, optimal_zones: dict[str, object] | None = None) -> go.Figure:
    """Plot TWA vs VMG with optional highlighted optimal ranges."""
    if df.empty:
        return go.Figure()

    fig = px.scatter(
        df,
        x="true_wind_angle",
        y="vmg",
        color="boat_id",
        opacity=0.7,
        hover_data=["timestamp", "speed_knots", "leg_mode"],
        title="TWA vs VMG",
    )

    if optimal_zones:
        for range_key in ["optimal_upwind_twa_range", "optimal_downwind_twa_range"]:
            range_text = str(optimal_zones.get(range_key, "N/A"))
            if range_text == "N/A" or "-" not in range_text:
                continue
            clean = range_text.replace(" deg", "")
            start_str, end_str = clean.split("-")
            start = float(start_str)
            end = float(end_str)
            color = "#4f8b2f" if "upwind" in range_key else "#8b6b38"
            fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.08, line_width=0)
            fig.add_vrect(x0=-end, x1=-start, fillcolor=color, opacity=0.08, line_width=0)

    fig.update_layout(template="plotly_white", height=420, xaxis_title="TWA (deg)", yaxis_title="VMG (knots)")
    return fig


def plot_race_replay(
    df: pd.DataFrame,
    current_timestamp: pd.Timestamp | None = None,
    show_weather: bool = True,
) -> go.Figure:
    """Plot race replay tracks and latest boat positions at a timestamp."""
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="Race Replay")
        return fig

    replay_ts = pd.Timestamp(current_timestamp) if current_timestamp is not None else pd.Timestamp(df["timestamp"].max())
    history = get_replay_history(df, replay_ts)
    frame = get_replay_frame(df, replay_ts)
    boats = sorted(history["boat_id"].unique().tolist())
    color_map = _boat_color_map(boats)

    for boat in boats:
        boat_history = history[history["boat_id"] == boat]
        fig.add_trace(
            go.Scatter(
                x=boat_history["x_position"],
                y=boat_history["y_position"],
                mode="lines",
                name=boat,
                line={"color": color_map[boat], "width": 2.1},
            )
        )

    fig.add_trace(
        go.Scatter(
            x=frame["x_position"],
            y=frame["y_position"],
            mode="markers+text",
            text=frame["boat_id"],
            textposition="top center",
            marker={"size": 12, "color": "#15384f", "line": {"width": 1.2, "color": "#ffffff"}},
            name="Current Position",
        )
    )

    if show_weather and not frame.empty and "true_wind_direction" in frame.columns:
        wind_dir = float(frame["true_wind_direction"].mean())
        wind_to = (wind_dir + 180.0) % 360.0
        bounds = calculate_track_bounds(history)
        span = max(bounds["x_max"] - bounds["x_min"], 100.0)
        arrow_len = span * 0.12
        anchor_x = bounds["x_min"] + span * 0.7
        anchor_y = bounds["y_max"] - (bounds["y_max"] - bounds["y_min"]) * 0.1
        dx = arrow_len * np.sin(np.deg2rad(wind_to))
        dy = arrow_len * np.cos(np.deg2rad(wind_to))

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
            arrowcolor="#4d6d82",
            text=f"Wind {wind_dir:.0f} deg",
        )

    bounds = calculate_track_bounds(history)
    fig.update_layout(
        template="plotly_white",
        title="Race Replay",
        xaxis_title="Course X Position (m)",
        yaxis_title="Course Y Position (m)",
        xaxis={"range": [bounds["x_min"], bounds["x_max"]]},
        yaxis={"range": [bounds["y_min"], bounds["y_max"]], "scaleanchor": "x", "scaleratio": 1},
        height=520,
    )
    return fig


def plot_weather_over_time(df: pd.DataFrame) -> go.Figure:
    """Plot wind speed and gust timeseries."""
    weather = summarize_weather(df) if "boat_id" in df.columns else df
    fig = go.Figure()
    if weather.empty:
        fig.update_layout(title="Weather Over Time")
        return fig

    fig.add_trace(
        go.Scatter(
            x=weather["timestamp"],
            y=weather["true_wind_speed"],
            mode="lines",
            name="True Wind Speed",
            line={"color": "#1f7a8c", "width": 2.0},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=weather["timestamp"],
            y=weather["gust_knots"],
            mode="lines",
            name="Gust",
            line={"color": "#e09f3e", "width": 1.8, "dash": "dash"},
        )
    )

    fig.update_layout(
        template="plotly_white",
        title="Weather Overlay",
        xaxis_title="Timestamp",
        yaxis_title="Wind (knots)",
        height=320,
    )
    return fig


def plot_team_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Plot grouped bar comparison for two boats/teams."""
    fig = go.Figure()
    if comparison_df.empty or "metric" not in comparison_df.columns or len(comparison_df.columns) < 3:
        fig.update_layout(title="Team Comparison")
        return fig

    metric_col = "metric"
    left_col, right_col = comparison_df.columns[1], comparison_df.columns[2]
    melted = comparison_df.melt(id_vars=metric_col, var_name="entity", value_name="value")

    fig = px.bar(
        melted,
        x=metric_col,
        y="value",
        color="entity",
        barmode="group",
        title="Team-vs-Team Comparison",
    )
    fig.update_layout(template="plotly_white", height=420)
    return fig


def plot_anomalies(df: pd.DataFrame, anomalies_df: pd.DataFrame) -> go.Figure:
    """Plot anomaly markers against speed timeseries context."""
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="Anomalies")
        return fig

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["speed_knots"],
            mode="lines",
            name="Speed",
            line={"color": "#274c66", "width": 1.8},
        )
    )

    if anomalies_df is not None and not anomalies_df.empty:
        merged = anomalies_df.merge(
            df[["timestamp", "boat_id", "speed_knots"]],
            on=["timestamp", "boat_id"],
            how="left",
        )
        fig.add_trace(
            go.Scatter(
                x=merged["timestamp"],
                y=merged["speed_knots"],
                mode="markers",
                marker={"size": 9, "symbol": "x", "color": "#a93226"},
                name="Anomalies",
                hovertemplate=(
                    "Boat: %{customdata[0]}<br>"
                    "Type: %{customdata[1]}<br>"
                    "Metric: %{customdata[2]} = %{customdata[3]}<extra></extra>"
                ),
                customdata=np.array(
                    list(
                        zip(
                            merged["boat_id"],
                            merged["anomaly_type"],
                            merged["related_metric"],
                            merged["value"],
                        )
                    ),
                    dtype=object,
                ),
            )
        )

    fig.update_layout(template="plotly_white", title="Anomaly Timeline", xaxis_title="Timestamp", yaxis_title="Speed (knots)", height=360)
    return fig


def plot_vmg_bands(vmg_band_df: pd.DataFrame) -> go.Figure:
    """Plot average progress VMG by TWA buckets and wind regime."""
    fig = go.Figure()
    if vmg_band_df.empty:
        fig.update_layout(title="VMG Bands")
        return fig

    fig = px.bar(
        vmg_band_df,
        x="twa_bucket_label",
        y="avg_progress_vmg",
        color="wind_bucket",
        barmode="group",
        title="VMG Bands by TWA and Wind Regime",
        hover_data=["samples", "avg_vmg"],
    )
    fig.update_layout(template="plotly_white", height=420, xaxis_title="TWA Bucket", yaxis_title="Avg Progress VMG")
    return fig
