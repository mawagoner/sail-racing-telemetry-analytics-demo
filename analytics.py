"""Analytics utilities for SailGP-style telemetry dashboards."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp",
    "boat_id",
    "team_name",
    "leg_id",
    "leg_mode",
    "speed_knots",
    "vmg",
    "true_wind_speed",
    "true_wind_angle",
    "true_wind_direction",
    "gust_knots",
    "heading",
    "heel_angle",
    "foil_cant",
    "x_position",
    "y_position",
    "maneuver_type",
    "course_side",
    "sea_state",
]

MANDATORY_COLUMNS = [
    "timestamp",
    "boat_id",
    "leg_id",
    "speed_knots",
    "true_wind_speed",
    "true_wind_angle",
    "heading",
    "heel_angle",
    "foil_cant",
]

NUMERIC_COLUMNS = [
    "speed_knots",
    "vmg",
    "true_wind_speed",
    "true_wind_angle",
    "true_wind_direction",
    "gust_knots",
    "heading",
    "heel_angle",
    "foil_cant",
    "x_position",
    "y_position",
    "sea_state",
]


def _median_absolute_deviation(values: pd.Series | np.ndarray) -> float:
    values_array = np.asarray(values, dtype=float)
    values_array = values_array[np.isfinite(values_array)]
    if values_array.size == 0:
        return 0.0
    median = float(np.median(values_array))
    mad = float(np.median(np.abs(values_array - median)))
    return mad


def _derive_xy_positions(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    derived_x = pd.Series(np.nan, index=df.index, dtype=float)
    derived_y = pd.Series(np.nan, index=df.index, dtype=float)

    for _, boat_df in df.groupby("boat_id", sort=False):
        boat_df = boat_df.sort_values("timestamp")
        dt_seconds = boat_df["timestamp"].diff().dt.total_seconds().fillna(0.0).clip(lower=0.0)
        dt_array = dt_seconds.to_numpy(dtype=float)
        if dt_array.size:
            dt_array[0] = 0.0

        speed_mps = boat_df["speed_knots"].to_numpy(dtype=float) * 0.514444
        heading_rad = np.deg2rad(boat_df["heading"].to_numpy(dtype=float))
        distance = speed_mps * dt_array

        dx = distance * np.sin(heading_rad)
        dy = distance * np.cos(heading_rad)

        derived_x.loc[boat_df.index] = np.cumsum(dx)
        derived_y.loc[boat_df.index] = np.cumsum(dy)

    return derived_x, derived_y


def validate_and_normalize_telemetry(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize telemetry, filling optional demo fields when missing."""
    if telemetry is None or telemetry.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    missing_columns = [col for col in MANDATORY_COLUMNS if col not in telemetry.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required telemetry columns: {missing_text}")

    df = telemetry.copy()

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    df["boat_id"] = df["boat_id"].astype(str).str.strip()
    df["leg_id"] = df["leg_id"].astype(str).str.strip()

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df["vmg"].isna().any():
        df.loc[df["vmg"].isna(), "vmg"] = df["speed_knots"] * np.cos(
            np.deg2rad(df["true_wind_angle"])
        )

    if df["true_wind_direction"].isna().any():
        df.loc[df["true_wind_direction"].isna(), "true_wind_direction"] = (
            df["heading"] + df["true_wind_angle"]
        ) % 360.0

    if df["gust_knots"].isna().any():
        df.loc[df["gust_knots"].isna(), "gust_knots"] = df["true_wind_speed"] + 1.2

    if df["sea_state"].isna().any():
        df.loc[df["sea_state"].isna(), "sea_state"] = np.clip(
            1.2 + (df["true_wind_speed"] - 10.0) * 0.22,
            1.0,
            5.0,
        )

    if df["team_name"].isna().any() or (df["team_name"].astype(str).str.strip() == "").any():
        default_names = df["boat_id"].map(lambda boat: f"{boat} Team")
        df["team_name"] = df["team_name"].fillna(default_names)
        df.loc[df["team_name"].astype(str).str.strip() == "", "team_name"] = default_names
    df["team_name"] = df["team_name"].astype(str).str.strip()

    df["maneuver_type"] = (
        df["maneuver_type"].fillna("none").astype(str).str.strip().str.lower()
    )
    df["course_side"] = np.where(df["true_wind_angle"] >= 0.0, "starboard", "port")
    df["leg_mode"] = np.where(df["true_wind_angle"].abs() < 90.0, "Upwind", "Downwind")

    if df["x_position"].isna().any() or df["y_position"].isna().any():
        derived_x, derived_y = _derive_xy_positions(df)
        df.loc[df["x_position"].isna(), "x_position"] = derived_x[df["x_position"].isna()]
        df.loc[df["y_position"].isna(), "y_position"] = derived_y[df["y_position"].isna()]

    required_for_rows = [
        "timestamp",
        "boat_id",
        "team_name",
        "leg_id",
        "speed_knots",
        "vmg",
        "true_wind_speed",
        "true_wind_angle",
        "heading",
        "heel_angle",
        "foil_cant",
        "x_position",
        "y_position",
    ]
    df = df.dropna(subset=required_for_rows)
    df = df.sort_values(["boat_id", "timestamp"]).reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    return df[REQUIRED_COLUMNS]


def _estimate_maneuver_metrics(
    speed_values: np.ndarray,
    timestamps: np.ndarray,
    event_index: int,
    baseline_points: int = 8,
    event_points: int = 4,
) -> dict[str, float]:
    pre_slice = speed_values[max(0, event_index - baseline_points) : event_index]
    post_slice = speed_values[event_index + 1 : event_index + 1 + baseline_points]
    event_slice = speed_values[
        max(0, event_index - event_points) : event_index + event_points + 1
    ]

    if pre_slice.size == 0 or post_slice.size == 0 or event_slice.size == 0:
        return {
            "entry_speed_knots": 0.0,
            "exit_speed_knots": 0.0,
            "baseline_speed_knots": 0.0,
            "min_speed_knots": 0.0,
            "speed_loss_knots": 0.0,
            "recovery_time_sec": 0.0,
        }

    entry_speed = float(np.nanmean(pre_slice[-3:]))
    exit_speed = float(np.nanmean(post_slice[:3]))
    baseline_speed = max(entry_speed, float(np.nanmean(np.concatenate([pre_slice[-3:], post_slice[:3]]))))
    min_speed = float(np.nanmin(event_slice))
    speed_loss = max(0.0, baseline_speed - min_speed)

    recovery_target = entry_speed * 0.95
    recovery_time_sec = np.nan
    for idx in range(event_index + 1, len(speed_values)):
        if speed_values[idx] >= recovery_target:
            delta = (timestamps[idx] - timestamps[event_index]) / np.timedelta64(1, "s")
            recovery_time_sec = float(max(0.0, delta))
            break
    if np.isnan(recovery_time_sec):
        recovery_time_sec = float(0.0)

    return {
        "entry_speed_knots": round(entry_speed, 3),
        "exit_speed_knots": round(exit_speed, 3),
        "baseline_speed_knots": round(baseline_speed, 3),
        "min_speed_knots": round(min_speed, 3),
        "speed_loss_knots": round(speed_loss, 3),
        "recovery_time_sec": round(recovery_time_sec, 2),
    }


def detect_maneuvers(
    telemetry: pd.DataFrame,
    cooldown_points: int = 8,
    min_mean_abs_twa: float = 18.0,
) -> pd.DataFrame:
    """Detect tacks/gybes and enrich each event with speed-loss context."""
    columns = [
        "timestamp",
        "boat_id",
        "team_name",
        "leg_id",
        "maneuver_type",
        "entry_speed_knots",
        "exit_speed_knots",
        "speed_loss_knots",
        "recovery_time_sec",
        "baseline_speed_knots",
        "min_speed_knots",
    ]
    if telemetry.empty:
        return pd.DataFrame(columns=columns)

    events: list[dict[str, Any]] = []

    for _, group in telemetry.groupby("boat_id", sort=False):
        boat_df = group.sort_values("timestamp").reset_index(drop=True)
        explicit_idx = boat_df.index[
            boat_df["maneuver_type"].isin(["tack", "gybe"])
        ].to_list()

        smooth_twa = boat_df["true_wind_angle"].rolling(window=5, center=True, min_periods=1).mean()
        twa_sign = np.sign(smooth_twa).replace(0, np.nan).ffill().bfill()
        sign_change = twa_sign.ne(twa_sign.shift(1))
        candidate_sign_idx = np.flatnonzero(sign_change.to_numpy())

        inferred_types: dict[int, str] = {}
        for pos in candidate_sign_idx:
            if pos == 0:
                continue
            window_start = max(0, pos - 6)
            window_end = min(len(boat_df), pos + 7)
            pre_window = smooth_twa.iloc[window_start:pos]
            post_window = smooth_twa.iloc[pos:window_end]
            if pre_window.empty or post_window.empty:
                continue
            pre_abs = float(pre_window.abs().mean())
            post_abs = float(post_window.abs().mean())
            if min(pre_abs, post_abs) < min_mean_abs_twa:
                continue
            inferred_types[pos] = "gybe" if (pre_abs + post_abs) / 2.0 >= 90.0 else "tack"

        candidate_idx = sorted(set(explicit_idx + list(inferred_types.keys())))
        last_event = -cooldown_points

        speed_values = boat_df["speed_knots"].to_numpy(dtype=float)
        time_values = boat_df["timestamp"].to_numpy(dtype="datetime64[ns]")

        for pos in candidate_idx:
            if pos == 0 or (pos - last_event) < cooldown_points:
                continue

            explicit_label = boat_df.iloc[pos]["maneuver_type"]
            if explicit_label in {"tack", "gybe"}:
                maneuver_type = explicit_label
            else:
                maneuver_type = inferred_types.get(pos, "tack")

            metrics = _estimate_maneuver_metrics(speed_values, time_values, pos)
            event_row = boat_df.iloc[pos]
            events.append(
                {
                    "timestamp": event_row["timestamp"],
                    "boat_id": event_row["boat_id"],
                    "team_name": event_row["team_name"],
                    "leg_id": event_row["leg_id"],
                    "maneuver_type": maneuver_type,
                    **metrics,
                }
            )
            last_event = pos

    if not events:
        return pd.DataFrame(columns=columns)

    detected = pd.DataFrame(events).sort_values(["timestamp", "boat_id"]).reset_index(drop=True)
    return detected


def compute_kpis(telemetry: pd.DataFrame, maneuvers: pd.DataFrame | None = None) -> dict[str, float]:
    """Compute headline KPI values for dashboard cards."""
    if telemetry.empty:
        return {
            "max_speed": 0.0,
            "avg_speed": 0.0,
            "best_vmg": 0.0,
            "avg_maneuver_loss": 0.0,
        }

    maneuver_df = maneuvers if maneuvers is not None else detect_maneuvers(telemetry)
    avg_loss = float(maneuver_df["speed_loss_knots"].mean()) if not maneuver_df.empty else 0.0

    return {
        "max_speed": float(telemetry["speed_knots"].max()),
        "avg_speed": float(telemetry["speed_knots"].mean()),
        "best_vmg": float(telemetry["vmg"].max()),
        "avg_maneuver_loss": avg_loss,
    }


def build_leg_comparison(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Aggregate leg metrics for tactical comparisons by boat/team."""
    columns = [
        "team_name",
        "boat_id",
        "leg_id",
        "leg_mode",
        "duration_sec",
        "samples",
        "avg_speed_knots",
        "max_speed_knots",
        "avg_vmg",
        "best_vmg",
        "avg_true_wind_speed",
        "avg_heel_angle",
    ]
    if telemetry.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        telemetry.groupby(["team_name", "boat_id", "leg_id", "leg_mode"], as_index=False)
        .agg(
            leg_start=("timestamp", "min"),
            leg_end=("timestamp", "max"),
            samples=("timestamp", "count"),
            avg_speed_knots=("speed_knots", "mean"),
            max_speed_knots=("speed_knots", "max"),
            avg_vmg=("vmg", "mean"),
            best_vmg=("vmg", "max"),
            avg_true_wind_speed=("true_wind_speed", "mean"),
            avg_heel_angle=("heel_angle", "mean"),
        )
        .reset_index(drop=True)
    )

    grouped["duration_sec"] = (grouped["leg_end"] - grouped["leg_start"]).dt.total_seconds()
    grouped["_leg_sort"] = pd.to_numeric(grouped["leg_id"], errors="coerce")
    grouped = grouped.sort_values(["boat_id", "_leg_sort", "leg_id"]).drop(columns=["_leg_sort"])

    rounded = grouped[columns].copy()
    numeric_cols = [
        "duration_sec",
        "avg_speed_knots",
        "max_speed_knots",
        "avg_vmg",
        "best_vmg",
        "avg_true_wind_speed",
        "avg_heel_angle",
    ]
    rounded[numeric_cols] = rounded[numeric_cols].round(2)
    return rounded.reset_index(drop=True)


def build_maneuver_summary(maneuvers: pd.DataFrame) -> dict[str, float]:
    """Summarize maneuver frequency and quality metrics."""
    if maneuvers.empty:
        return {
            "tack_count": 0,
            "gybe_count": 0,
            "avg_loss": 0.0,
            "max_loss": 0.0,
            "avg_recovery": 0.0,
        }

    return {
        "tack_count": int((maneuvers["maneuver_type"] == "tack").sum()),
        "gybe_count": int((maneuvers["maneuver_type"] == "gybe").sum()),
        "avg_loss": float(maneuvers["speed_loss_knots"].mean()),
        "max_loss": float(maneuvers["speed_loss_knots"].max()),
        "avg_recovery": float(maneuvers["recovery_time_sec"].mean()),
    }


def detect_anomalies(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Detect unusual telemetry patterns using robust, lightweight statistics.

    Logic notes:
    - Speed drops use rolling robust z-score plus a quantile-based delta threshold.
    - Heel anomalies use median + MAD instead of mean/std so outliers do not hide risk.
    - Poor VMG uses baseline progress in matched TWA and wind bands.
    - Foil instability combines speed drop + heel load + foil cant change.
    """

    columns = [
        "timestamp",
        "boat_id",
        "team_name",
        "leg_id",
        "anomaly_type",
        "severity",
        "metric_value",
        "threshold",
        "speed_knots",
        "vmg",
        "heel_angle",
        "foil_cant",
        "description",
    ]
    if telemetry.empty:
        return pd.DataFrame(columns=columns)

    anomalies: list[dict[str, Any]] = []
    working = telemetry.copy().sort_values(["boat_id", "timestamp"]).reset_index(drop=True)

    for _, boat_df in working.groupby("boat_id", sort=False):
        speed = boat_df["speed_knots"]
        heel = boat_df["heel_angle"].abs()
        foil = boat_df["foil_cant"]

        speed_diff = speed.diff()
        rolling_median = speed.rolling(window=12, min_periods=6).median()
        rolling_mad = (speed - rolling_median).abs().rolling(window=12, min_periods=6).median()
        robust_scale = (1.4826 * rolling_mad).replace(0, np.nan)
        robust_z = (speed - rolling_median) / robust_scale

        drop_threshold = min(float(speed_diff.quantile(0.08)), -1.5)
        speed_drop_mask = (speed_diff < drop_threshold) | (robust_z < -2.4)

        heel_median = float(heel.median())
        heel_mad = max(_median_absolute_deviation(heel), 0.5)
        heel_threshold = heel_median + 3.0 * heel_mad + 1.5
        high_heel_mask = heel > heel_threshold

        foil_delta = foil.diff().abs().fillna(0.0)
        foil_delta_threshold = max(float(foil_delta.quantile(0.90)), 4.0)
        foil_instability_mask = speed_drop_mask & (heel > (heel_median + 1.8 * heel_mad)) & (
            foil_delta > foil_delta_threshold
        )

        for idx in np.flatnonzero(speed_drop_mask.fillna(False).to_numpy()):
            row = boat_df.iloc[idx]
            anomalies.append(
                {
                    "timestamp": row["timestamp"],
                    "boat_id": row["boat_id"],
                    "team_name": row["team_name"],
                    "leg_id": row["leg_id"],
                    "anomaly_type": "speed_drop",
                    "severity": round(abs(float(speed_diff.iloc[idx])), 3),
                    "metric_value": round(float(speed_diff.iloc[idx]), 3),
                    "threshold": round(drop_threshold, 3),
                    "speed_knots": round(float(row["speed_knots"]), 3),
                    "vmg": round(float(row["vmg"]), 3),
                    "heel_angle": round(float(row["heel_angle"]), 3),
                    "foil_cant": round(float(row["foil_cant"]), 3),
                    "description": "Rapid speed loss vs recent telemetry trend.",
                }
            )

        for idx in np.flatnonzero(high_heel_mask.fillna(False).to_numpy()):
            row = boat_df.iloc[idx]
            anomalies.append(
                {
                    "timestamp": row["timestamp"],
                    "boat_id": row["boat_id"],
                    "team_name": row["team_name"],
                    "leg_id": row["leg_id"],
                    "anomaly_type": "abnormal_heel",
                    "severity": round(float(heel.iloc[idx] - heel_threshold), 3),
                    "metric_value": round(float(row["heel_angle"]), 3),
                    "threshold": round(heel_threshold, 3),
                    "speed_knots": round(float(row["speed_knots"]), 3),
                    "vmg": round(float(row["vmg"]), 3),
                    "heel_angle": round(float(row["heel_angle"]), 3),
                    "foil_cant": round(float(row["foil_cant"]), 3),
                    "description": "Heel angle above robust control envelope.",
                }
            )

        for idx in np.flatnonzero(foil_instability_mask.fillna(False).to_numpy()):
            row = boat_df.iloc[idx]
            anomalies.append(
                {
                    "timestamp": row["timestamp"],
                    "boat_id": row["boat_id"],
                    "team_name": row["team_name"],
                    "leg_id": row["leg_id"],
                    "anomaly_type": "foil_instability",
                    "severity": round(float(foil_delta.iloc[idx]), 3),
                    "metric_value": round(float(foil_delta.iloc[idx]), 3),
                    "threshold": round(foil_delta_threshold, 3),
                    "speed_knots": round(float(row["speed_knots"]), 3),
                    "vmg": round(float(row["vmg"]), 3),
                    "heel_angle": round(float(row["heel_angle"]), 3),
                    "foil_cant": round(float(row["foil_cant"]), 3),
                    "description": "Speed drop with foil-cant oscillation and heel load.",
                }
            )

    # Build expected progress VMG baseline by matching wind-angle and wind-speed regimes.
    baseline_df = working.copy()
    baseline_df["abs_twa"] = baseline_df["true_wind_angle"].abs().clip(0.0, 179.9)
    baseline_df["progress_vmg"] = np.where(
        baseline_df["abs_twa"] < 90.0,
        baseline_df["vmg"],
        -baseline_df["vmg"],
    )

    twa_bins = np.arange(0.0, 181.0, 10.0)
    baseline_df["twa_bucket"] = pd.cut(
        baseline_df["abs_twa"], bins=twa_bins, right=False, include_lowest=True
    )

    wind_quantiles = baseline_df["true_wind_speed"].quantile([0.0, 0.33, 0.66, 1.0]).to_numpy()
    wind_bins = np.unique(np.round(wind_quantiles, 3))
    if wind_bins.size < 4:
        wind_bins = np.linspace(
            float(baseline_df["true_wind_speed"].min()),
            float(baseline_df["true_wind_speed"].max()) + 1e-6,
            4,
        )
    baseline_df["wind_bucket"] = pd.cut(
        baseline_df["true_wind_speed"], bins=wind_bins, include_lowest=True
    )

    benchmark = (
        baseline_df.groupby(["twa_bucket", "wind_bucket"], observed=True)["progress_vmg"]
        .agg(["median", "count"])
        .reset_index()
        .rename(columns={"median": "progress_median", "count": "sample_count"})
    )
    benchmark_mad = (
        baseline_df.groupby(["twa_bucket", "wind_bucket"], observed=True)["progress_vmg"]
        .apply(_median_absolute_deviation)
        .reset_index(name="progress_mad")
    )
    benchmark = benchmark.merge(benchmark_mad, on=["twa_bucket", "wind_bucket"], how="left")

    baseline_df = baseline_df.merge(benchmark, on=["twa_bucket", "wind_bucket"], how="left")
    fallback_scale = max(float(baseline_df["progress_vmg"].std(ddof=0)) * 0.25, 0.25)
    spread = baseline_df["progress_mad"].fillna(fallback_scale).replace(0, fallback_scale)
    baseline_df["poor_vmg_threshold"] = baseline_df["progress_median"] - 1.9 * spread

    poor_vmg_mask = (
        baseline_df["progress_vmg"] < baseline_df["poor_vmg_threshold"]
    ) & (baseline_df["sample_count"].fillna(0) >= 8)

    for idx in np.flatnonzero(poor_vmg_mask.fillna(False).to_numpy()):
        row = baseline_df.iloc[idx]
        anomalies.append(
            {
                "timestamp": row["timestamp"],
                "boat_id": row["boat_id"],
                "team_name": row["team_name"],
                "leg_id": row["leg_id"],
                "anomaly_type": "poor_vmg",
                "severity": round(float(row["poor_vmg_threshold"] - row["progress_vmg"]), 3),
                "metric_value": round(float(row["progress_vmg"]), 3),
                "threshold": round(float(row["poor_vmg_threshold"]), 3),
                "speed_knots": round(float(row["speed_knots"]), 3),
                "vmg": round(float(row["vmg"]), 3),
                "heel_angle": round(float(row["heel_angle"]), 3),
                "foil_cant": round(float(row["foil_cant"]), 3),
                "description": "VMG underperforms peer runs in same wind-angle regime.",
            }
        )

    if not anomalies:
        return pd.DataFrame(columns=columns)

    anomaly_df = pd.DataFrame(anomalies).sort_values(["timestamp", "boat_id"]).reset_index(drop=True)
    return anomaly_df[columns]


def build_anomaly_summary(anomalies: pd.DataFrame) -> dict[str, int]:
    """Roll up anomaly counts for KPI cards."""
    if anomalies.empty:
        return {
            "total": 0,
            "speed_drop": 0,
            "abnormal_heel": 0,
            "poor_vmg": 0,
            "foil_instability": 0,
        }

    return {
        "total": int(len(anomalies)),
        "speed_drop": int((anomalies["anomaly_type"] == "speed_drop").sum()),
        "abnormal_heel": int((anomalies["anomaly_type"] == "abnormal_heel").sum()),
        "poor_vmg": int((anomalies["anomaly_type"] == "poor_vmg").sum()),
        "foil_instability": int((anomalies["anomaly_type"] == "foil_instability").sum()),
    }


def _mask_to_periods(
    timestamps: pd.Series,
    mask: pd.Series,
    label: str,
) -> pd.DataFrame:
    periods: list[dict[str, Any]] = []
    if timestamps.empty:
        return pd.DataFrame(columns=["period_type", "start", "end", "points"])

    timestamps = timestamps.reset_index(drop=True)
    mask = mask.fillna(False).reset_index(drop=True)

    start_idx: int | None = None
    for idx, flag in enumerate(mask.to_list()):
        if flag and start_idx is None:
            start_idx = idx
        if (not flag or idx == len(mask) - 1) and start_idx is not None:
            end_idx = idx if flag and idx == len(mask) - 1 else idx - 1
            periods.append(
                {
                    "period_type": label,
                    "start": timestamps.iloc[start_idx],
                    "end": timestamps.iloc[end_idx],
                    "points": int(end_idx - start_idx + 1),
                }
            )
            start_idx = None

    return pd.DataFrame(periods)


def extract_weather_timeseries(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Aggregate course-level weather per timestamp for overlays and replay."""
    if telemetry.empty:
        return pd.DataFrame(
            columns=["timestamp", "true_wind_speed", "true_wind_direction", "gust_knots", "sea_state"]
        )

    weather = (
        telemetry.groupby("timestamp", as_index=False)
        .agg(
            true_wind_speed=("true_wind_speed", "mean"),
            true_wind_direction=("true_wind_direction", "mean"),
            gust_knots=("gust_knots", "mean"),
            sea_state=("sea_state", "mean"),
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    weather[["true_wind_speed", "true_wind_direction", "gust_knots", "sea_state"]] = weather[
        ["true_wind_speed", "true_wind_direction", "gust_knots", "sea_state"]
    ].round(3)
    return weather


def detect_weather_periods(weather: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Find high-gust and wind-shift periods for chart overlays."""
    if weather.empty:
        empty = pd.DataFrame(columns=["period_type", "start", "end", "points"])
        return {"gust_periods": empty, "shift_periods": empty}

    gust_threshold = float(weather["gust_knots"].quantile(0.90))
    gust_mask = weather["gust_knots"] >= gust_threshold
    gust_periods = _mask_to_periods(weather["timestamp"], gust_mask, "high_gust")

    angle_delta = ((weather["true_wind_direction"].diff() + 180.0) % 360.0) - 180.0
    shift_threshold = max(float(angle_delta.abs().quantile(0.92)), 4.5)
    shift_mask = angle_delta.abs() >= shift_threshold
    shift_periods = _mask_to_periods(weather["timestamp"], shift_mask, "wind_shift")

    return {
        "gust_periods": gust_periods,
        "shift_periods": shift_periods,
    }


def _boat_summary(telemetry: pd.DataFrame, maneuvers: pd.DataFrame, anomalies: pd.DataFrame, boat_id: str) -> dict[str, float]:
    boat_df = telemetry[telemetry["boat_id"] == boat_id]
    boat_maneuvers = maneuvers[maneuvers["boat_id"] == boat_id] if not maneuvers.empty else maneuvers
    boat_anomalies = anomalies[anomalies["boat_id"] == boat_id] if not anomalies.empty else anomalies

    upwind_df = boat_df[boat_df["leg_mode"] == "Upwind"]
    downwind_df = boat_df[boat_df["leg_mode"] == "Downwind"]

    upwind_perf = float(upwind_df["vmg"].mean()) if not upwind_df.empty else 0.0
    downwind_perf = float((-downwind_df["vmg"]).mean()) if not downwind_df.empty else 0.0

    return {
        "average_speed": float(boat_df["speed_knots"].mean()) if not boat_df.empty else 0.0,
        "max_speed": float(boat_df["speed_knots"].max()) if not boat_df.empty else 0.0,
        "average_vmg": float(boat_df["vmg"].mean()) if not boat_df.empty else 0.0,
        "best_vmg": float(boat_df["vmg"].max()) if not boat_df.empty else 0.0,
        "maneuver_count": float(len(boat_maneuvers)),
        "avg_maneuver_loss": float(boat_maneuvers["speed_loss_knots"].mean()) if len(boat_maneuvers) else 0.0,
        "anomaly_count": float(len(boat_anomalies)),
        "upwind_performance": upwind_perf,
        "downwind_performance": downwind_perf,
    }


def build_team_comparison(
    telemetry: pd.DataFrame,
    maneuvers: pd.DataFrame,
    anomalies: pd.DataFrame,
    boat_a: str,
    boat_b: str,
) -> pd.DataFrame:
    """Build side-by-side comparison metrics for two selected teams."""
    if telemetry.empty:
        return pd.DataFrame(columns=["metric", boat_a, boat_b])

    summary_a = _boat_summary(telemetry, maneuvers, anomalies, boat_a)
    summary_b = _boat_summary(telemetry, maneuvers, anomalies, boat_b)

    metrics = [
        ("Average Speed", "average_speed"),
        ("Max Speed", "max_speed"),
        ("Average VMG", "average_vmg"),
        ("Best VMG", "best_vmg"),
        ("Maneuver Count", "maneuver_count"),
        ("Average Maneuver Loss", "avg_maneuver_loss"),
        ("Anomaly Count", "anomaly_count"),
        ("Upwind Performance", "upwind_performance"),
        ("Downwind Performance", "downwind_performance"),
    ]

    comparison_rows: list[dict[str, Any]] = []
    for label, key in metrics:
        comparison_rows.append(
            {
                "metric": label,
                boat_a: round(float(summary_a[key]), 3),
                boat_b: round(float(summary_b[key]), 3),
            }
        )

    return pd.DataFrame(comparison_rows)


def compute_optimal_vmg_zones(telemetry: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Estimate best-performing TWA bands as a simplified demo optimization model."""
    if telemetry.empty:
        empty = pd.DataFrame()
        return {
            "bucket_stats": empty,
            "recommendations": empty,
            "wind_band_recommendations": empty,
        }

    working = telemetry.copy()
    working["abs_twa"] = working["true_wind_angle"].abs().clip(0.0, 179.9)
    working["progress_vmg"] = np.where(working["abs_twa"] < 90.0, working["vmg"], -working["vmg"])

    bucket_size = 10
    working["twa_bucket_start"] = (working["abs_twa"] // bucket_size) * bucket_size
    working["twa_bucket_end"] = working["twa_bucket_start"] + bucket_size
    working["twa_bucket_label"] = (
        working["twa_bucket_start"].astype(int).astype(str)
        + "-"
        + working["twa_bucket_end"].astype(int).astype(str)
        + " deg"
    )

    bucket_stats = (
        working.groupby(["twa_bucket_start", "twa_bucket_end", "twa_bucket_label"], as_index=False)
        .agg(
            samples=("progress_vmg", "count"),
            avg_vmg=("vmg", "mean"),
            avg_progress_vmg=("progress_vmg", "mean"),
            best_progress_vmg=("progress_vmg", "max"),
            avg_true_wind_speed=("true_wind_speed", "mean"),
        )
        .sort_values("twa_bucket_start")
        .reset_index(drop=True)
    )

    upwind_candidates = bucket_stats[bucket_stats["twa_bucket_start"] < 90.0]
    downwind_candidates = bucket_stats[bucket_stats["twa_bucket_start"] >= 90.0]

    recommendation_rows: list[dict[str, Any]] = []
    if not upwind_candidates.empty:
        best_upwind = upwind_candidates.sort_values("avg_progress_vmg", ascending=False).iloc[0]
        recommendation_rows.append(
            {
                "zone": "Optimal Upwind TWA",
                "twa_range": best_upwind["twa_bucket_label"],
                "avg_progress_vmg": round(float(best_upwind["avg_progress_vmg"]), 3),
                "avg_true_wind_speed": round(float(best_upwind["avg_true_wind_speed"]), 3),
            }
        )

    if not downwind_candidates.empty:
        best_downwind = downwind_candidates.sort_values("avg_progress_vmg", ascending=False).iloc[0]
        recommendation_rows.append(
            {
                "zone": "Optimal Downwind TWA",
                "twa_range": best_downwind["twa_bucket_label"],
                "avg_progress_vmg": round(float(best_downwind["avg_progress_vmg"]), 3),
                "avg_true_wind_speed": round(float(best_downwind["avg_true_wind_speed"]), 3),
            }
        )

    wind_quantiles = working["true_wind_speed"].quantile([0.0, 0.33, 0.66, 1.0]).to_numpy()
    wind_bins = np.unique(np.round(wind_quantiles, 3))
    if wind_bins.size < 4:
        wind_bins = np.linspace(
            float(working["true_wind_speed"].min()),
            float(working["true_wind_speed"].max()) + 1e-6,
            4,
        )

    wind_labels = ["Low Wind", "Mid Wind", "High Wind"]
    working["wind_band"] = pd.cut(working["true_wind_speed"], bins=wind_bins, labels=wind_labels, include_lowest=True)

    by_wind = (
        working.groupby(["wind_band", "twa_bucket_label", "twa_bucket_start"], observed=True, as_index=False)
        .agg(avg_progress_vmg=("progress_vmg", "mean"), samples=("progress_vmg", "count"))
    )
    wind_band_recommendations = (
        by_wind.sort_values(["wind_band", "avg_progress_vmg"], ascending=[True, False])
        .groupby("wind_band", as_index=False, observed=True)
        .head(1)
        .reset_index(drop=True)
    )

    recommendation_df = pd.DataFrame(recommendation_rows)
    return {
        "bucket_stats": bucket_stats.round(3),
        "recommendations": recommendation_df,
        "wind_band_recommendations": wind_band_recommendations.round(3),
    }
