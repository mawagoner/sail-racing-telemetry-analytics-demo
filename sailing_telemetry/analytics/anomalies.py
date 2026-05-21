"""Lightweight anomaly detection for telemetry QA and performance review."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..utils import median_absolute_deviation, progress_vmg


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Detect speed, heel, VMG, and foil-instability anomalies.

    Returns a clean dataframe with columns:
    timestamp, boat_id, anomaly_type, severity, explanation, related_metric, value
    """

    columns = [
        "timestamp",
        "boat_id",
        "anomaly_type",
        "severity",
        "explanation",
        "related_metric",
        "value",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    telemetry = df.sort_values(["boat_id", "timestamp"]).reset_index(drop=True)
    records: list[dict[str, Any]] = []

    for boat_id, boat_df in telemetry.groupby("boat_id", sort=False):
        speed = boat_df["speed_knots"]
        heel = boat_df["heel_angle"].abs()
        foil = boat_df["foil_cant"]

        speed_diff = speed.diff()
        rolling_median = speed.rolling(window=12, min_periods=6).median()
        rolling_std = speed.rolling(window=12, min_periods=6).std(ddof=0).replace(0, np.nan)
        z_score = (speed - rolling_median) / rolling_std

        speed_drop_threshold = min(float(speed_diff.quantile(0.08)), -1.4)
        speed_drop_mask = (speed_diff < speed_drop_threshold) | (z_score < -2.5)

        heel_median = float(heel.median())
        heel_mad = max(median_absolute_deviation(heel), 0.45)
        heel_threshold = heel_median + 3.0 * heel_mad + 1.0
        heel_mask = heel > heel_threshold

        foil_delta = foil.diff().abs().fillna(0.0)
        foil_threshold = max(float(foil_delta.quantile(0.9)), 4.0)
        foil_mask = speed_drop_mask & (heel > heel_median + 1.8 * heel_mad) & (foil_delta > foil_threshold)

        for idx in np.flatnonzero(speed_drop_mask.fillna(False).to_numpy()):
            row = boat_df.iloc[idx]
            records.append(
                {
                    "timestamp": row["timestamp"],
                    "boat_id": boat_id,
                    "anomaly_type": "speed_drop",
                    "severity": round(abs(float(speed_diff.iloc[idx])), 3),
                    "explanation": "Sudden speed drop relative to rolling baseline.",
                    "related_metric": "speed_knots_delta",
                    "value": round(float(speed_diff.iloc[idx]), 3),
                }
            )

        for idx in np.flatnonzero(heel_mask.fillna(False).to_numpy()):
            row = boat_df.iloc[idx]
            records.append(
                {
                    "timestamp": row["timestamp"],
                    "boat_id": boat_id,
                    "anomaly_type": "abnormal_heel",
                    "severity": round(float(heel.iloc[idx] - heel_threshold), 3),
                    "explanation": "Heel angle exceeds robust operating envelope.",
                    "related_metric": "heel_angle",
                    "value": round(float(row["heel_angle"]), 3),
                }
            )

        for idx in np.flatnonzero(foil_mask.fillna(False).to_numpy()):
            row = boat_df.iloc[idx]
            records.append(
                {
                    "timestamp": row["timestamp"],
                    "boat_id": boat_id,
                    "anomaly_type": "foil_instability",
                    "severity": round(float(foil_delta.iloc[idx]), 3),
                    "explanation": "Speed drop with heel load and foil-cant oscillation.",
                    "related_metric": "foil_cant_delta",
                    "value": round(float(foil_delta.iloc[idx]), 3),
                }
            )

    # Poor VMG baseline by wind-angle and wind-speed regimes.
    baseline = telemetry.copy()
    baseline["abs_twa"] = baseline["true_wind_angle"].abs().clip(0.0, 179.9)
    baseline["progress_vmg"] = progress_vmg(baseline)
    baseline["twa_bucket"] = pd.cut(
        baseline["abs_twa"],
        bins=np.arange(0.0, 181.0, 10.0),
        include_lowest=True,
        right=False,
    )

    wind_quantiles = baseline["true_wind_speed"].quantile([0.0, 0.33, 0.66, 1.0]).to_numpy()
    wind_bins = np.unique(np.round(wind_quantiles, 3))
    if wind_bins.size < 4:
        wind_bins = np.linspace(
            float(baseline["true_wind_speed"].min()),
            float(baseline["true_wind_speed"].max()) + 1e-6,
            4,
        )
    baseline["wind_bucket"] = pd.cut(baseline["true_wind_speed"], bins=wind_bins, include_lowest=True)

    benchmark = (
        baseline.groupby(["twa_bucket", "wind_bucket"], observed=True)["progress_vmg"]
        .agg(["median", "count"])
        .reset_index()
        .rename(columns={"median": "progress_median", "count": "sample_count"})
    )
    benchmark["progress_mad"] = (
        baseline.groupby(["twa_bucket", "wind_bucket"], observed=True)["progress_vmg"]
        .apply(median_absolute_deviation)
        .reset_index(drop=True)
    )

    baseline = baseline.merge(benchmark, on=["twa_bucket", "wind_bucket"], how="left")
    fallback_spread = max(float(baseline["progress_vmg"].std(ddof=0)) * 0.25, 0.25)
    spread = baseline["progress_mad"].fillna(fallback_spread).replace(0.0, fallback_spread)
    baseline["poor_vmg_threshold"] = baseline["progress_median"] - 1.9 * spread

    poor_mask = (
        (baseline["progress_vmg"] < baseline["poor_vmg_threshold"])
        & (baseline["sample_count"].fillna(0) >= 8)
    )

    for idx in np.flatnonzero(poor_mask.fillna(False).to_numpy()):
        row = baseline.iloc[idx]
        records.append(
            {
                "timestamp": row["timestamp"],
                "boat_id": row["boat_id"],
                "anomaly_type": "poor_vmg",
                "severity": round(float(row["poor_vmg_threshold"] - row["progress_vmg"]), 3),
                "explanation": "Poor VMG for the current TWA and wind-speed regime.",
                "related_metric": "progress_vmg",
                "value": round(float(row["progress_vmg"]), 3),
            }
        )

    if not records:
        return pd.DataFrame(columns=columns)

    anomaly_df = pd.DataFrame(records).sort_values(["timestamp", "boat_id"]).reset_index(drop=True)
    return anomaly_df[columns]
