"""Analytics subpackage for sailing telemetry framework."""

from .anomalies import detect_anomalies
from .comparison import compare_boats, compare_teams, create_comparison_table
from .maneuvers import calculate_recovery_time, detect_maneuvers, summarize_maneuvers
from .metrics import (
    calculate_maneuver_loss,
    calculate_vmg_efficiency,
    summarize_by_boat,
    summarize_by_leg,
    summarize_kpis,
)
from .vmg import (
    calculate_vmg_bands,
    classify_upwind_downwind,
    estimate_optimal_vmg_zones,
    recommend_target_twa_ranges,
)
from .weather import detect_gust_periods, detect_wind_shifts, prepare_weather_overlay, summarize_weather

__all__ = [
    "calculate_maneuver_loss",
    "calculate_recovery_time",
    "calculate_vmg_bands",
    "calculate_vmg_efficiency",
    "classify_upwind_downwind",
    "compare_boats",
    "compare_teams",
    "create_comparison_table",
    "detect_anomalies",
    "detect_gust_periods",
    "detect_maneuvers",
    "detect_wind_shifts",
    "estimate_optimal_vmg_zones",
    "prepare_weather_overlay",
    "recommend_target_twa_ranges",
    "summarize_by_boat",
    "summarize_by_leg",
    "summarize_kpis",
    "summarize_maneuvers",
    "summarize_weather",
]
