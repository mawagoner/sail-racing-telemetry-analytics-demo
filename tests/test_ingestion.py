"""Ingestion pipeline tests for v1.1 features."""

from __future__ import annotations

import pandas as pd

from sailing_telemetry.ingestion import detect_input_format, ingest_telemetry


def test_detect_input_format_basic_cases() -> None:
    assert detect_input_format("session.csv") == "csv"
    assert detect_input_format("track.json") == "json"
    assert detect_input_format("generated") == "sample"


def test_ingestion_returns_artifact_report() -> None:
    artifact = ingest_telemetry(
        "sample",
        config={
            "num_boats": 3,
            "legs_per_boat": 4,
            "samples_per_leg": 80,
            "sampling_interval_seconds": 2,
            "seed": 7,
            "sample_source_profile": "bg_csv",
            "inject_data_issues": True,
            "issue_rate": 0.03,
            "resample_enabled": True,
            "resample_seconds": 2,
        },
        return_artifact=True,
    )

    assert len(artifact.gold_df) > 0
    assert "rows" in artifact.report
    assert "quality" in artifact.report
    assert artifact.report["source_profile"] == "bg_csv"
    assert "row_quality_score" in artifact.quality_flags_df.columns


def test_resample_aggregates_duplicate_timestamps_per_boat() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "boat_id": "Boat-A",
                "leg_id": "1",
                "speed_knots": 20.0,
                "true_wind_speed": 12.0,
                "true_wind_angle": -30.0,
                "heading": 90.0,
                "heel_angle": 12.0,
                "foil_cant": 70.0,
                "x_position": 100.0,
                "y_position": 200.0,
            },
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "boat_id": "Boat-A",
                "leg_id": "1",
                "speed_knots": 24.0,
                "true_wind_speed": 14.0,
                "true_wind_angle": -34.0,
                "heading": 110.0,
                "heel_angle": 14.0,
                "foil_cant": 74.0,
                "x_position": 104.0,
                "y_position": 204.0,
            },
            {
                "timestamp": "2026-01-01T00:00:02Z",
                "boat_id": "Boat-A",
                "leg_id": "1",
                "speed_knots": 30.0,
                "true_wind_speed": 15.0,
                "true_wind_angle": -20.0,
                "heading": 120.0,
                "heel_angle": 15.0,
                "foil_cant": 75.0,
                "x_position": 120.0,
                "y_position": 220.0,
            },
        ]
    )

    artifact = ingest_telemetry(
        frame,
        config={
            "source_profile": "canonical",
            "resample_enabled": True,
            "resample_seconds": 2,
        },
        return_artifact=True,
    )

    assert artifact.report["rows"]["silver"] == 2
    first_timestamp_row = artifact.silver_df.loc[
        artifact.silver_df["timestamp"] == pd.Timestamp("2026-01-01 00:00:00")
    ].iloc[0]
    assert first_timestamp_row["speed_knots"] == 22.0
    assert first_timestamp_row["true_wind_speed"] == 13.0
    assert first_timestamp_row["heading"] == 100.0
