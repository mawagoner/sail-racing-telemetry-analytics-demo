"""Boat and team comparison analytics."""

from __future__ import annotations

import pandas as pd

from ..utils import progress_vmg
from .anomalies import detect_anomalies
from .maneuvers import detect_maneuvers


def _boat_metrics(df: pd.DataFrame, boat_id: str, maneuvers: pd.DataFrame, anomalies: pd.DataFrame) -> dict[str, float]:
    boat_df = df[df["boat_id"] == boat_id]
    boat_maneuvers = maneuvers[maneuvers["boat_id"] == boat_id] if not maneuvers.empty else maneuvers
    boat_anomalies = anomalies[anomalies["boat_id"] == boat_id] if not anomalies.empty else anomalies

    if boat_df.empty:
        return {
            "average_speed": 0.0,
            "max_speed": 0.0,
            "average_vmg": 0.0,
            "best_vmg": 0.0,
            "maneuver_count": 0.0,
            "average_maneuver_loss": 0.0,
            "anomaly_count": 0.0,
            "upwind_performance": 0.0,
            "downwind_performance": 0.0,
        }

    working = boat_df.copy()
    working["progress_vmg"] = progress_vmg(working)
    upwind = working[working["leg_mode"] == "Upwind"]
    downwind = working[working["leg_mode"] == "Downwind"]

    return {
        "average_speed": float(working["speed_knots"].mean()),
        "max_speed": float(working["speed_knots"].max()),
        "average_vmg": float(working["vmg"].mean()),
        "best_vmg": float(working["vmg"].max()),
        "maneuver_count": float(len(boat_maneuvers)),
        "average_maneuver_loss": float(boat_maneuvers["speed_loss_knots"].mean()) if len(boat_maneuvers) else 0.0,
        "anomaly_count": float(len(boat_anomalies)),
        "upwind_performance": float(upwind["progress_vmg"].mean()) if not upwind.empty else 0.0,
        "downwind_performance": float(downwind["progress_vmg"].mean()) if not downwind.empty else 0.0,
    }


def compare_boats(df: pd.DataFrame, boat_a: str, boat_b: str) -> pd.DataFrame:
    """Compare two boats across core performance indicators."""
    maneuvers = detect_maneuvers(df)
    anomalies = detect_anomalies(df)

    a_metrics = _boat_metrics(df, boat_a, maneuvers, anomalies)
    b_metrics = _boat_metrics(df, boat_b, maneuvers, anomalies)

    metric_map = [
        ("Average Speed", "average_speed"),
        ("Max Speed", "max_speed"),
        ("Average VMG", "average_vmg"),
        ("Best VMG", "best_vmg"),
        ("Maneuver Count", "maneuver_count"),
        ("Average Maneuver Loss", "average_maneuver_loss"),
        ("Anomaly Count", "anomaly_count"),
        ("Upwind Performance", "upwind_performance"),
        ("Downwind Performance", "downwind_performance"),
    ]

    rows = []
    for label, key in metric_map:
        rows.append({"metric": label, boat_a: round(a_metrics[key], 3), boat_b: round(b_metrics[key], 3)})
    return pd.DataFrame(rows)


def compare_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate performance by team for broad benchmarking."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "team_name",
                "boats",
                "samples",
                "avg_speed_knots",
                "max_speed_knots",
                "avg_vmg",
                "best_vmg",
            ]
        )

    summary = (
        df.groupby("team_name", as_index=False)
        .agg(
            boats=("boat_id", "nunique"),
            samples=("timestamp", "count"),
            avg_speed_knots=("speed_knots", "mean"),
            max_speed_knots=("speed_knots", "max"),
            avg_vmg=("vmg", "mean"),
            best_vmg=("vmg", "max"),
        )
        .sort_values("avg_speed_knots", ascending=False)
        .reset_index(drop=True)
    )
    numeric = ["avg_speed_knots", "max_speed_knots", "avg_vmg", "best_vmg"]
    summary[numeric] = summary[numeric].round(3)
    return summary


def create_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build reusable boat-level comparison table with maneuver/anomaly counts."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "boat_id",
                "team_name",
                "samples",
                "avg_speed_knots",
                "avg_vmg",
                "maneuver_count",
                "avg_maneuver_loss",
                "anomaly_count",
            ]
        )

    maneuvers = detect_maneuvers(df)
    anomalies = detect_anomalies(df)

    base = (
        df.groupby(["boat_id", "team_name"], as_index=False)
        .agg(
            samples=("timestamp", "count"),
            avg_speed_knots=("speed_knots", "mean"),
            avg_vmg=("vmg", "mean"),
        )
        .sort_values("boat_id")
        .reset_index(drop=True)
    )

    maneuver_summary = (
        maneuvers.groupby("boat_id", as_index=False)
        .agg(
            maneuver_count=("timestamp", "count"),
            avg_maneuver_loss=("speed_loss_knots", "mean"),
        )
        if not maneuvers.empty
        else pd.DataFrame(columns=["boat_id", "maneuver_count", "avg_maneuver_loss"])
    )

    anomaly_summary = (
        anomalies.groupby("boat_id", as_index=False)
        .agg(anomaly_count=("timestamp", "count"))
        if not anomalies.empty
        else pd.DataFrame(columns=["boat_id", "anomaly_count"])
    )

    merged = base.merge(maneuver_summary, on="boat_id", how="left").merge(anomaly_summary, on="boat_id", how="left")
    merged[["maneuver_count", "avg_maneuver_loss", "anomaly_count"]] = merged[
        ["maneuver_count", "avg_maneuver_loss", "anomaly_count"]
    ].fillna(0.0)
    merged[["avg_speed_knots", "avg_vmg", "avg_maneuver_loss"]] = merged[
        ["avg_speed_knots", "avg_vmg", "avg_maneuver_loss"]
    ].round(3)
    return merged
