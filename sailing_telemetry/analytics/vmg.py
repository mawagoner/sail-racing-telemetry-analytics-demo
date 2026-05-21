"""VMG banding, zoning, and target recommendation analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import progress_vmg


def classify_upwind_downwind(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure leg_mode classification exists from true wind angle."""
    if df.empty:
        return df.copy()
    classified = df.copy()
    classified["leg_mode"] = np.where(classified["true_wind_angle"].abs() < 90.0, "Upwind", "Downwind")
    return classified


def calculate_vmg_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate VMG by true-wind-angle and wind-speed buckets."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "twa_bucket_start",
                "twa_bucket_end",
                "twa_bucket_label",
                "wind_bucket",
                "samples",
                "avg_vmg",
                "avg_progress_vmg",
            ]
        )

    working = classify_upwind_downwind(df)
    working["abs_twa"] = working["true_wind_angle"].abs().clip(0.0, 179.9)
    working["progress_vmg"] = progress_vmg(working)

    bucket_size = 10
    working["twa_bucket_start"] = (working["abs_twa"] // bucket_size) * bucket_size
    working["twa_bucket_end"] = working["twa_bucket_start"] + bucket_size
    working["twa_bucket_label"] = (
        working["twa_bucket_start"].astype(int).astype(str)
        + "-"
        + working["twa_bucket_end"].astype(int).astype(str)
        + " deg"
    )

    wind_bins = np.quantile(working["true_wind_speed"], [0.0, 0.33, 0.66, 1.0])
    wind_bins = np.unique(np.round(wind_bins, 3))
    if wind_bins.size < 4:
        wind_bins = np.linspace(
            float(working["true_wind_speed"].min()),
            float(working["true_wind_speed"].max()) + 1e-6,
            4,
        )
    working["wind_bucket"] = pd.cut(
        working["true_wind_speed"],
        bins=wind_bins,
        labels=["Low Wind", "Mid Wind", "High Wind"],
        include_lowest=True,
    )

    bands = (
        working.groupby(
            ["twa_bucket_start", "twa_bucket_end", "twa_bucket_label", "wind_bucket"],
            observed=True,
            as_index=False,
        )
        .agg(
            samples=("progress_vmg", "count"),
            avg_vmg=("vmg", "mean"),
            avg_progress_vmg=("progress_vmg", "mean"),
        )
        .sort_values(["twa_bucket_start", "wind_bucket"])
        .reset_index(drop=True)
    )

    bands[["avg_vmg", "avg_progress_vmg"]] = bands[["avg_vmg", "avg_progress_vmg"]].round(3)
    return bands


def estimate_optimal_vmg_zones(df: pd.DataFrame) -> dict[str, object]:
    """Estimate optimal upwind/downwind TWA zones with confidence note."""
    bands = calculate_vmg_bands(df)
    if bands.empty:
        return {
            "optimal_upwind_twa_range": "N/A",
            "optimal_downwind_twa_range": "N/A",
            "supporting_table": bands,
            "confidence_note": "No data available.",
        }

    supporting = (
        bands.groupby(["twa_bucket_start", "twa_bucket_end", "twa_bucket_label"], as_index=False)
        .agg(samples=("samples", "sum"), avg_progress_vmg=("avg_progress_vmg", "mean"))
        .sort_values("twa_bucket_start")
        .reset_index(drop=True)
    )

    upwind = supporting[supporting["twa_bucket_start"] < 90.0]
    downwind = supporting[supporting["twa_bucket_start"] >= 90.0]

    optimal_upwind = "N/A"
    optimal_downwind = "N/A"
    if not upwind.empty:
        top_upwind = upwind.sort_values("avg_progress_vmg", ascending=False).iloc[0]
        optimal_upwind = str(top_upwind["twa_bucket_label"])
    if not downwind.empty:
        top_downwind = downwind.sort_values("avg_progress_vmg", ascending=False).iloc[0]
        optimal_downwind = str(top_downwind["twa_bucket_label"])

    sample_count = int(len(df))
    if sample_count >= 3000:
        confidence_note = "High confidence: ample sample size for demo-level range recommendations."
    elif sample_count >= 1200:
        confidence_note = "Medium confidence: useful directional guidance, add more sessions for stability."
    else:
        confidence_note = "Low confidence: recommendation is illustrative with limited telemetry volume."

    return {
        "optimal_upwind_twa_range": optimal_upwind,
        "optimal_downwind_twa_range": optimal_downwind,
        "supporting_table": supporting,
        "confidence_note": confidence_note,
    }


def recommend_target_twa_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Return compact table of recommended upwind/downwind TWA targets."""
    zones = estimate_optimal_vmg_zones(df)
    return pd.DataFrame(
        [
            {
                "zone": "Upwind Target TWA",
                "recommended_range": zones["optimal_upwind_twa_range"],
                "confidence_note": zones["confidence_note"],
            },
            {
                "zone": "Downwind Target TWA",
                "recommended_range": zones["optimal_downwind_twa_range"],
                "confidence_note": zones["confidence_note"],
            },
        ]
    )
