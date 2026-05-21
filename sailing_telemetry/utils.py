"""Utility helpers shared across telemetry framework modules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def median_absolute_deviation(values: pd.Series | np.ndarray) -> float:
    """Return robust spread estimate (MAD) for a numeric sequence."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    median = float(np.median(array))
    return float(np.median(np.abs(array - median)))


def wrapped_angle_delta(series: pd.Series) -> pd.Series:
    """Return wrapped heading/wind-direction deltas in range [-180, 180]."""
    return ((series.diff() + 180.0) % 360.0) - 180.0


def progress_vmg(frame: pd.DataFrame) -> pd.Series:
    """Return VMG signed for progress toward upwind/downwind marks.

    Upwind progress uses VMG directly, downwind uses -VMG so both regimes can
    be compared on a positive-is-better scale.
    """
    abs_twa = frame["true_wind_angle"].abs()
    return pd.Series(
        np.where(abs_twa < 90.0, frame["vmg"], -frame["vmg"]),
        index=frame.index,
        dtype=float,
    )


def contiguous_periods(
    timestamps: pd.Series,
    mask: pd.Series,
    period_type: str,
) -> pd.DataFrame:
    """Convert boolean masks into contiguous start/end period tables."""
    if timestamps.empty:
        return pd.DataFrame(columns=["period_type", "start", "end", "points"])

    times = timestamps.reset_index(drop=True)
    flags = mask.fillna(False).reset_index(drop=True)
    records: list[dict[str, Any]] = []

    start_idx: int | None = None
    for idx, is_active in enumerate(flags.to_list()):
        if is_active and start_idx is None:
            start_idx = idx

        at_last = idx == len(flags) - 1
        if start_idx is not None and (not is_active or at_last):
            end_idx = idx if is_active and at_last else idx - 1
            records.append(
                {
                    "period_type": period_type,
                    "start": times.iloc[start_idx],
                    "end": times.iloc[end_idx],
                    "points": int(end_idx - start_idx + 1),
                }
            )
            start_idx = None

    return pd.DataFrame(records)
