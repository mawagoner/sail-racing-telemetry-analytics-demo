"""Maneuver detection and summary logic."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..utils import wrapped_angle_delta


def calculate_recovery_time(df: pd.DataFrame, event: dict[str, Any] | pd.Series) -> float:
    """Estimate seconds needed to recover to 95% of entry speed after event."""
    if df.empty:
        return 0.0

    event_data = dict(event)
    boat_id = str(event_data.get("boat_id", ""))
    event_ts = pd.Timestamp(event_data.get("timestamp"))

    boat_df = df[df["boat_id"] == boat_id].sort_values("timestamp").reset_index(drop=True)
    if boat_df.empty:
        return 0.0

    ts_values = boat_df["timestamp"]
    idx = int(ts_values.searchsorted(event_ts, side="left"))
    idx = min(max(idx, 0), len(boat_df) - 1)
    if idx > 0:
        prev_delta = abs((ts_values.iloc[idx - 1] - event_ts).total_seconds())
        curr_delta = abs((ts_values.iloc[idx] - event_ts).total_seconds())
        if prev_delta < curr_delta:
            idx -= 1

    entry_speed = float(event_data.get("entry_speed_knots", boat_df.iloc[idx]["speed_knots"]))
    recovery_target = entry_speed * 0.95

    for future_idx in range(idx + 1, len(boat_df)):
        if float(boat_df.iloc[future_idx]["speed_knots"]) >= recovery_target:
            recovery = (boat_df.iloc[future_idx]["timestamp"] - ts_values.iloc[idx]).total_seconds()
            return float(max(0.0, recovery))
    return 0.0


def detect_maneuvers(df: pd.DataFrame, cooldown_points: int = 8) -> pd.DataFrame:
    """Detect tack and gybe events from explicit labels and inferred changes."""
    columns = [
        "timestamp",
        "boat_id",
        "team_name",
        "leg_id",
        "maneuver_type",
        "entry_speed_knots",
        "exit_speed_knots",
        "baseline_speed_knots",
        "min_speed_knots",
        "speed_loss_knots",
        "recovery_time_sec",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    telemetry = df.copy()
    if "maneuver_type" not in telemetry.columns:
        telemetry["maneuver_type"] = "none"

    events: list[dict[str, Any]] = []

    for _, boat_df in telemetry.groupby("boat_id", sort=False):
        boat_df = boat_df.sort_values("timestamp").reset_index(drop=True)

        explicit_idx = boat_df.index[
            boat_df["maneuver_type"].astype(str).str.lower().isin(["tack", "gybe"])
        ].tolist()

        smooth_twa = boat_df["true_wind_angle"].rolling(window=5, center=True, min_periods=1).mean()
        twa_sign = np.sign(smooth_twa).replace(0, np.nan).ffill().bfill()
        sign_change_idx = np.flatnonzero(twa_sign.ne(twa_sign.shift(1)).to_numpy())

        heading_change = wrapped_angle_delta(boat_df["heading"]).abs().fillna(0.0)
        heading_threshold = max(float(heading_change.quantile(0.92)), 8.0)
        heading_idx = np.flatnonzero((heading_change > heading_threshold).to_numpy())

        candidate_idx = sorted(set(explicit_idx + list(sign_change_idx) + list(heading_idx)))

        speeds = boat_df["speed_knots"].to_numpy(dtype=float)
        last_event_idx = -cooldown_points

        for idx in candidate_idx:
            if idx == 0 or (idx - last_event_idx) < cooldown_points:
                continue

            row = boat_df.iloc[idx]
            label = str(row.get("maneuver_type", "none")).lower()
            if label not in {"tack", "gybe"}:
                left = max(0, idx - 5)
                right = min(len(boat_df), idx + 6)
                mean_abs_twa = float(boat_df.iloc[left:right]["true_wind_angle"].abs().mean())
                label = "gybe" if mean_abs_twa >= 90.0 else "tack"

            pre_slice = speeds[max(0, idx - 8) : idx]
            post_slice = speeds[idx + 1 : idx + 9]
            event_slice = speeds[max(0, idx - 4) : idx + 5]

            entry_speed = float(np.nanmean(pre_slice[-3:])) if pre_slice.size else float(speeds[idx])
            exit_speed = float(np.nanmean(post_slice[:3])) if post_slice.size else float(speeds[idx])
            if pre_slice.size and post_slice.size:
                baseline = max(entry_speed, float(np.nanmean(np.concatenate([pre_slice[-3:], post_slice[:3]]))))
            else:
                baseline = entry_speed
            min_speed = float(np.nanmin(event_slice)) if event_slice.size else float(speeds[idx])
            speed_loss = max(0.0, baseline - min_speed)

            event_record = {
                "timestamp": row["timestamp"],
                "boat_id": row["boat_id"],
                "team_name": row.get("team_name", row["boat_id"]),
                "leg_id": row.get("leg_id", ""),
                "maneuver_type": label,
                "entry_speed_knots": round(entry_speed, 3),
                "exit_speed_knots": round(exit_speed, 3),
                "baseline_speed_knots": round(baseline, 3),
                "min_speed_knots": round(min_speed, 3),
                "speed_loss_knots": round(speed_loss, 3),
            }
            event_record["recovery_time_sec"] = round(calculate_recovery_time(boat_df, event_record), 2)
            events.append(event_record)
            last_event_idx = idx

    if not events:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(events).sort_values(["timestamp", "boat_id"]).reset_index(drop=True)


def summarize_maneuvers(maneuvers_df: pd.DataFrame) -> dict[str, float]:
    """Summarize maneuver frequency and average loss/recovery."""
    if maneuvers_df.empty:
        return {
            "total_maneuvers": 0,
            "tack_count": 0,
            "gybe_count": 0,
            "avg_speed_loss": 0.0,
            "worst_speed_loss": 0.0,
            "avg_recovery_time": 0.0,
        }

    return {
        "total_maneuvers": int(len(maneuvers_df)),
        "tack_count": int((maneuvers_df["maneuver_type"] == "tack").sum()),
        "gybe_count": int((maneuvers_df["maneuver_type"] == "gybe").sum()),
        "avg_speed_loss": float(maneuvers_df["speed_loss_knots"].mean()),
        "worst_speed_loss": float(maneuvers_df["speed_loss_knots"].max()),
        "avg_recovery_time": float(maneuvers_df["recovery_time_sec"].mean()),
    }
