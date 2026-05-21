"""Metric and VMG analytics tests."""

from __future__ import annotations

from sailing_telemetry.analytics.metrics import summarize_kpis
from sailing_telemetry.analytics.vmg import estimate_optimal_vmg_zones
from sailing_telemetry.generator import generate_sample_telemetry


def test_kpi_summary_contains_expected_fields() -> None:
    df = generate_sample_telemetry(num_boats=3, legs_per_boat=4, samples_per_leg=60, seed=3)
    kpis = summarize_kpis(df)
    expected_fields = {"max_speed", "avg_speed", "best_vmg", "avg_maneuver_loss", "data_points"}
    assert expected_fields.issubset(set(kpis.keys()))


def test_vmg_zone_estimation_handles_normal_input() -> None:
    df = generate_sample_telemetry(num_boats=4, legs_per_boat=6, samples_per_leg=80, seed=13)
    zones = estimate_optimal_vmg_zones(df)
    assert "optimal_upwind_twa_range" in zones
    assert "optimal_downwind_twa_range" in zones
    assert "supporting_table" in zones
