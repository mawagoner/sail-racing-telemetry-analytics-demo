"""Schema and generator integration tests."""

from __future__ import annotations

from sailing_telemetry.generator import generate_sample_telemetry
from sailing_telemetry.schema import get_required_columns, validate_telemetry_dataframe


def test_generated_data_has_required_columns() -> None:
    df = generate_sample_telemetry(num_boats=3, legs_per_boat=4, samples_per_leg=60, seed=7)
    required = set(get_required_columns())
    assert required.issubset(set(df.columns))


def test_generated_data_validates() -> None:
    df = generate_sample_telemetry(num_boats=3, legs_per_boat=4, samples_per_leg=60, seed=11)
    is_valid, errors = validate_telemetry_dataframe(df)
    assert is_valid, f"Validation errors: {errors}"
