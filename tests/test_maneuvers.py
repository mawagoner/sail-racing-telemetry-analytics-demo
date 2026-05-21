"""Maneuver analytics tests."""

from __future__ import annotations

import pandas as pd

from sailing_telemetry.analytics.maneuvers import detect_maneuvers, summarize_maneuvers
from sailing_telemetry.generator import generate_sample_telemetry


def test_detect_maneuvers_returns_dataframe() -> None:
    df = generate_sample_telemetry(num_boats=3, legs_per_boat=4, samples_per_leg=80, seed=17)
    maneuvers = detect_maneuvers(df)
    assert isinstance(maneuvers, pd.DataFrame)


def test_summarize_maneuvers_returns_expected_fields() -> None:
    df = generate_sample_telemetry(num_boats=3, legs_per_boat=4, samples_per_leg=80, seed=19)
    summary = summarize_maneuvers(detect_maneuvers(df))
    expected = {"total_maneuvers", "tack_count", "gybe_count", "avg_speed_loss", "avg_recovery_time"}
    assert expected.issubset(set(summary.keys()))
