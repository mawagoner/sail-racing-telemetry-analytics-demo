"""KPI and aggregation metrics for sailing telemetry."""

from __future__ import annotations

import pandas as pd

from ..utils import progress_vmg
from .maneuvers import detect_maneuvers


def calculate_maneuver_loss(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-event maneuver losses derived from detected maneuvers."""
    events = detect_maneuvers(df)
    if events.empty:
        return pd.DataFrame(columns=["timestamp", "boat_id", "maneuver_type", "speed_loss_knots"])
    return events[["timestamp", "boat_id", "maneuver_type", "speed_loss_knots"]].copy()


def summarize_kpis(df: pd.DataFrame) -> dict[str, float]:
    """Compute headline KPI values for dashboard cards."""
    if df.empty:
        return {
            "max_speed": 0.0,
            "avg_speed": 0.0,
            "best_vmg": 0.0,
            "avg_maneuver_loss": 0.0,
            "data_points": 0.0,
        }

    maneuver_events = detect_maneuvers(df)
    avg_loss = float(maneuver_events["speed_loss_knots"].mean()) if not maneuver_events.empty else 0.0
    return {
        "max_speed": float(df["speed_knots"].max()),
        "avg_speed": float(df["speed_knots"].mean()),
        "best_vmg": float(df["vmg"].max()),
        "avg_maneuver_loss": avg_loss,
        "data_points": float(len(df)),
    }


def summarize_by_leg(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate performance metrics by boat and leg."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "boat_id",
                "team_name",
                "leg_id",
                "leg_mode",
                "samples",
                "avg_speed_knots",
                "max_speed_knots",
                "avg_vmg",
                "best_vmg",
                "avg_true_wind_speed",
                "avg_heel_angle",
            ]
        )

    summary = (
        df.groupby(["boat_id", "team_name", "leg_id", "leg_mode"], as_index=False)
        .agg(
            samples=("timestamp", "count"),
            avg_speed_knots=("speed_knots", "mean"),
            max_speed_knots=("speed_knots", "max"),
            avg_vmg=("vmg", "mean"),
            best_vmg=("vmg", "max"),
            avg_true_wind_speed=("true_wind_speed", "mean"),
            avg_heel_angle=("heel_angle", "mean"),
        )
        .sort_values(["boat_id", "leg_id"])
        .reset_index(drop=True)
    )

    numeric_cols = [
        "avg_speed_knots",
        "max_speed_knots",
        "avg_vmg",
        "best_vmg",
        "avg_true_wind_speed",
        "avg_heel_angle",
    ]
    summary[numeric_cols] = summary[numeric_cols].round(3)
    return summary


def summarize_by_boat(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate core telemetry metrics by boat and team."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "boat_id",
                "team_name",
                "samples",
                "avg_speed_knots",
                "max_speed_knots",
                "avg_vmg",
                "best_vmg",
                "avg_true_wind_speed",
                "avg_heel_angle",
            ]
        )

    summary = (
        df.groupby(["boat_id", "team_name"], as_index=False)
        .agg(
            samples=("timestamp", "count"),
            avg_speed_knots=("speed_knots", "mean"),
            max_speed_knots=("speed_knots", "max"),
            avg_vmg=("vmg", "mean"),
            best_vmg=("vmg", "max"),
            avg_true_wind_speed=("true_wind_speed", "mean"),
            avg_heel_angle=("heel_angle", "mean"),
        )
        .sort_values("boat_id")
        .reset_index(drop=True)
    )

    numeric_cols = [
        "avg_speed_knots",
        "max_speed_knots",
        "avg_vmg",
        "best_vmg",
        "avg_true_wind_speed",
        "avg_heel_angle",
    ]
    summary[numeric_cols] = summary[numeric_cols].round(3)
    return summary


def calculate_vmg_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate VMG efficiency as progress VMG ratio versus average speed."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "boat_id",
                "team_name",
                "avg_progress_vmg",
                "avg_speed_knots",
                "vmg_efficiency_ratio",
            ]
        )

    working = df.copy()
    working["progress_vmg"] = progress_vmg(working)

    summary = (
        working.groupby(["boat_id", "team_name"], as_index=False)
        .agg(
            avg_progress_vmg=("progress_vmg", "mean"),
            avg_speed_knots=("speed_knots", "mean"),
        )
        .sort_values("boat_id")
        .reset_index(drop=True)
    )

    summary["vmg_efficiency_ratio"] = summary["avg_progress_vmg"] / summary["avg_speed_knots"].replace(0, pd.NA)
    summary["vmg_efficiency_ratio"] = summary["vmg_efficiency_ratio"].fillna(0.0)
    summary[["avg_progress_vmg", "avg_speed_knots", "vmg_efficiency_ratio"]] = summary[
        ["avg_progress_vmg", "avg_speed_knots", "vmg_efficiency_ratio"]
    ].round(4)
    return summary
