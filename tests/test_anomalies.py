"""Anomaly detection tests."""

from __future__ import annotations

import pandas as pd

from sailing_telemetry.analytics.anomalies import detect_anomalies
from sailing_telemetry.generator import generate_sample_telemetry


def test_detect_anomalies_returns_dataframe() -> None:
    df = generate_sample_telemetry(num_boats=3, legs_per_boat=5, samples_per_leg=70, seed=23)
    anomalies = detect_anomalies(df)
    assert isinstance(anomalies, pd.DataFrame)


def test_anomaly_dataframe_has_expected_columns() -> None:
    df = generate_sample_telemetry(num_boats=3, legs_per_boat=5, samples_per_leg=70, seed=29)
    anomalies = detect_anomalies(df)
    expected = {
        "timestamp",
        "boat_id",
        "anomaly_type",
        "severity",
        "explanation",
        "related_metric",
        "value",
    }
    assert expected.issubset(set(anomalies.columns))
