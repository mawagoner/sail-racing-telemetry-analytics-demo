"""Race replay data preparation independent of Streamlit UI."""

from __future__ import annotations

import pandas as pd


def prepare_race_replay_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return replay-ready telemetry sorted by boat and timestamp."""
    if df.empty:
        return df.copy()
    return df.sort_values(["boat_id", "timestamp"]).reset_index(drop=True)


def get_replay_history(df: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    """Return telemetry history up to a replay timestamp."""
    replay_df = prepare_race_replay_data(df)
    return replay_df[replay_df["timestamp"] <= pd.Timestamp(timestamp)].copy()


def get_replay_frame(df: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    """Return the latest row per boat at a replay timestamp."""
    history = get_replay_history(df, timestamp)
    if history.empty:
        return history
    return history.groupby("boat_id", as_index=False).tail(1).sort_values("boat_id").reset_index(drop=True)


def calculate_track_bounds(df: pd.DataFrame) -> dict[str, float]:
    """Calculate replay track bounds with margin padding."""
    if df.empty:
        return {"x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0}

    x_min = float(df["x_position"].min())
    x_max = float(df["x_position"].max())
    y_min = float(df["y_position"].min())
    y_max = float(df["y_position"].max())

    x_pad = max((x_max - x_min) * 0.08, 10.0)
    y_pad = max((y_max - y_min) * 0.08, 10.0)

    return {
        "x_min": x_min - x_pad,
        "x_max": x_max + x_pad,
        "y_min": y_min - y_pad,
        "y_max": y_max + y_pad,
    }
