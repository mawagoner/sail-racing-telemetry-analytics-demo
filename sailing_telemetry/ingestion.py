"""Telemetry ingestion and cleansing pipeline (v1.1).

This module implements a lightweight, framework-style ingestion flow with:
- adapter registry parsing (bronze)
- source profile mapping and unit/sign normalization (silver)
- canonical schema normalization (gold)
- data quality scoring and integration reporting artifact

Future adapters can be added for:
- NMEA 0183 sentence streams
- NMEA 2000 gateway payloads
- GPX tracks and points
- Expedition/B&G/Garmin/Raymarine/Sailmon vendor formats
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .adapters import get_adapter_for_format, list_registered_formats
from .generator import KNOTS_PER_MPS, generate_sample_telemetry
from .schema import CANONICAL_COLUMNS, NUMERIC_COLUMNS, REQUIRED_COLUMNS, normalize_telemetry_dataframe
from .source_profiles import (
    SourceProfile,
    detect_profile_from_columns,
    get_source_profile,
    list_source_profiles,
)
from .utils import median_absolute_deviation, wrapped_angle_delta


@dataclass
class IntegrationArtifact:
    """Structured artifact of ingestion and cleansing stages."""

    source_name: str
    source_format: str
    adapter_used: str
    source_profile: str
    bronze_df: pd.DataFrame
    silver_df: pd.DataFrame
    gold_df: pd.DataFrame
    quality_flags_df: pd.DataFrame
    report: dict[str, Any]


def detect_input_format(file_name: str) -> str:
    """Infer source format from extension or source label."""
    lowered = str(file_name).strip().lower()
    if lowered in {"sample", "generated", "generated sample telemetry"}:
        return "sample"

    suffix = Path(lowered).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix == ".gpx":
        return "gpx"
    if suffix in {".nmea", ".log", ".txt"}:
        return "nmea"
    return "unknown"


def _source_name_from_input(source: Any) -> str:
    if source is None:
        return "generated_sample"
    if isinstance(source, str):
        return source
    if hasattr(source, "name"):
        return str(source.name)
    if isinstance(source, pd.DataFrame):
        return "dataframe_input"
    return "unknown_source"


def _apply_column_map(frame: pd.DataFrame, profile: SourceProfile) -> tuple[pd.DataFrame, list[str]]:
    """Rename profile source columns into canonical names."""
    if not profile.column_map:
        return frame.copy(), ["column_map: none (canonical profile)"]

    rename_map = {source: target for source, target in profile.column_map.items() if source in frame.columns}
    renamed = frame.rename(columns=rename_map).copy()
    rule = f"column_map: renamed {len(rename_map)} fields from profile '{profile.name}'"
    return renamed, [rule]


def _apply_unit_and_convention_normalization(frame: pd.DataFrame, profile: SourceProfile) -> tuple[pd.DataFrame, list[str]]:
    """Convert units/sign conventions to canonical representation."""
    cleaned = frame.copy()
    applied: list[str] = []

    for column, unit in profile.units.items():
        if column not in cleaned.columns:
            continue
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

        if unit == "mps":
            cleaned[column] = cleaned[column] * KNOTS_PER_MPS
            applied.append(f"unit_conversion: {column} mps -> knots")
        elif unit == "rad":
            cleaned[column] = np.degrees(cleaned[column])
            applied.append(f"unit_conversion: {column} radians -> degrees")

    if profile.twa_sign_convention == "port_positive" and "true_wind_angle" in cleaned.columns:
        cleaned["true_wind_angle"] = -pd.to_numeric(cleaned["true_wind_angle"], errors="coerce")
        applied.append("sign_conversion: true_wind_angle port-positive -> starboard-positive")

    if profile.value_maps:
        for column, mapping in profile.value_maps.items():
            if column not in cleaned.columns:
                continue
            lowered_map = {str(key).lower(): value for key, value in mapping.items()}
            normalized_text = cleaned[column].astype(str).str.strip().str.lower()
            cleaned[column] = normalized_text.map(lambda value: lowered_map.get(value, value))
            applied.append(f"value_map: normalized categorical values for {column}")

    return cleaned, applied


def _apply_basic_cleansing(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply generic cleansing before strict schema normalization."""
    cleaned = frame.copy()
    applied: list[str] = []

    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    applied.append("cleansing: trimmed column labels")

    if "timestamp" in cleaned.columns:
        cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
        applied.append("cleansing: parsed timestamp to datetime")
        cleaned = cleaned.sort_values(["timestamp"])
        applied.append("cleansing: sorted by timestamp")

    before_dedup = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    dropped = before_dedup - len(cleaned)
    applied.append(f"cleansing: dropped {dropped} exact duplicate rows")
    return cleaned, applied


def _resample_if_configured(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Optionally align telemetry to a regular cadence with interpolation."""
    if not bool(config.get("resample_enabled", False)):
        return frame, []

    if "timestamp" not in frame.columns or "boat_id" not in frame.columns:
        return frame, ["resample: skipped (timestamp/boat_id unavailable)"]

    seconds = int(config.get("resample_seconds", 2))
    seconds = max(1, seconds)
    frequency = f"{seconds}s"

    numeric_cols = [column for column in frame.columns if column in NUMERIC_COLUMNS or pd.api.types.is_numeric_dtype(frame[column])]
    string_cols = [column for column in frame.columns if column not in numeric_cols and column != "timestamp"]

    groups: list[pd.DataFrame] = []
    for boat_id, boat_df in frame.groupby("boat_id", sort=False):
        boat_df = boat_df.sort_values("timestamp").copy()
        if boat_df["timestamp"].isna().all():
            continue

        if boat_df["timestamp"].duplicated().any():
            agg_map: dict[str, str] = {}
            for column in boat_df.columns:
                if column == "timestamp":
                    continue
                agg_map[column] = "mean" if column in numeric_cols else "last"
            boat_df = boat_df.groupby("timestamp", as_index=False).agg(agg_map)

        boat_df = boat_df.set_index("timestamp")
        full_index = pd.date_range(boat_df.index.min(), boat_df.index.max(), freq=frequency)
        aligned = boat_df.reindex(full_index)

        for column in string_cols:
            if column == "boat_id":
                aligned[column] = boat_id
            else:
                aligned[column] = aligned[column].ffill().bfill()

        for column in numeric_cols:
            aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
            aligned[column] = aligned[column].interpolate(method="time", limit_direction="both")

        aligned["timestamp"] = aligned.index
        groups.append(aligned.reset_index(drop=True))

    if not groups:
        return frame, ["resample: skipped (no valid boat groups)"]

    resampled = pd.concat(groups, ignore_index=True)
    ordered_columns = [column for column in frame.columns if column in resampled.columns]
    resampled = resampled[ordered_columns].sort_values(["timestamp", "boat_id"]).reset_index(drop=True)
    return resampled, [f"resample: aligned to {seconds}-second cadence with interpolation"]


def _compute_quality_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute row-level data quality flags and score."""
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "boat_id",
                "missing_required_fields",
                "out_of_range_speed",
                "timestamp_jump",
                "duplicate_packet",
                "row_quality_score",
            ]
        )

    working = frame.copy()
    if "timestamp" in working.columns:
        working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    if "boat_id" in working.columns:
        working["boat_id"] = working["boat_id"].astype(str).str.strip()

    required_present = [column for column in REQUIRED_COLUMNS if column in working.columns]
    missing_required = working[required_present].isna().any(axis=1) if required_present else pd.Series(True, index=working.index)

    if "speed_knots" in working.columns:
        speed_values = pd.to_numeric(working["speed_knots"], errors="coerce")
        out_of_range_speed = (speed_values < 0.0) | (speed_values > 70.0)
    else:
        out_of_range_speed = pd.Series(False, index=working.index)

    if "timestamp" in working.columns and "boat_id" in working.columns:
        duplicate_packet = working.duplicated(subset=["timestamp", "boat_id"], keep="first")
    else:
        duplicate_packet = pd.Series(False, index=working.index)

    timestamp_jump = pd.Series(False, index=working.index)
    if "timestamp" in working.columns and "boat_id" in working.columns:
        for _, group_idx in working.groupby("boat_id").groups.items():
            group = working.loc[group_idx].sort_values("timestamp")
            delta = group["timestamp"].diff().dt.total_seconds().fillna(0.0)
            positive = delta[delta > 0]
            if positive.empty:
                continue
            threshold = max(float(positive.median()) * 5.0, 20.0)
            jump_mask = delta > threshold
            timestamp_jump.loc[group.index] = jump_mask.to_numpy()

    flags = pd.DataFrame(
        {
            "timestamp": working.get("timestamp", pd.Series(pd.NaT, index=working.index)),
            "boat_id": working.get("boat_id", pd.Series("unknown", index=working.index)),
            "missing_required_fields": missing_required.astype(bool),
            "out_of_range_speed": out_of_range_speed.fillna(False).astype(bool),
            "timestamp_jump": timestamp_jump.fillna(False).astype(bool),
            "duplicate_packet": duplicate_packet.fillna(False).astype(bool),
        }
    )

    flag_columns = [
        "missing_required_fields",
        "out_of_range_speed",
        "timestamp_jump",
        "duplicate_packet",
    ]
    flags["row_quality_score"] = 1.0 - flags[flag_columns].sum(axis=1) / float(len(flag_columns))
    flags["row_quality_score"] = flags["row_quality_score"].clip(0.0, 1.0)
    return flags


def _summarize_quality(flags: pd.DataFrame) -> dict[str, Any]:
    """Summarize quality flags into compact report metrics."""
    if flags.empty:
        return {
            "mean_quality_score": 0.0,
            "rows_with_issues": 0,
            "flag_counts": {
                "missing_required_fields": 0,
                "out_of_range_speed": 0,
                "timestamp_jump": 0,
                "duplicate_packet": 0,
            },
        }

    flag_columns = [
        "missing_required_fields",
        "out_of_range_speed",
        "timestamp_jump",
        "duplicate_packet",
    ]
    counts = {column: int(flags[column].sum()) for column in flag_columns}
    rows_with_issues = int(flags[flag_columns].any(axis=1).sum())
    mean_quality = float(flags["row_quality_score"].mean())
    return {
        "mean_quality_score": round(mean_quality, 4),
        "rows_with_issues": rows_with_issues,
        "flag_counts": counts,
    }


def _simulate_profile_variant(canonical_df: pd.DataFrame, profile: SourceProfile) -> pd.DataFrame:
    """Convert canonical sample telemetry into profile-like raw source fields."""
    if profile.name == "canonical":
        return canonical_df.copy()

    source_df = canonical_df.copy()

    if profile.twa_sign_convention == "port_positive" and "true_wind_angle" in source_df.columns:
        source_df["true_wind_angle"] = -source_df["true_wind_angle"]

    for column, unit in profile.units.items():
        if column not in source_df.columns:
            continue
        if unit == "mps":
            source_df[column] = source_df[column] / KNOTS_PER_MPS
        elif unit == "rad":
            source_df[column] = np.radians(source_df[column])

    if profile.value_maps:
        for column, mapping in profile.value_maps.items():
            if column not in source_df.columns:
                continue
            reverse = {target: source for source, target in mapping.items()}
            source_df[column] = source_df[column].map(lambda value: reverse.get(str(value), value))

    reverse_column_map = {canonical: source for source, canonical in profile.column_map.items()}
    rename_map = {canonical: source for canonical, source in reverse_column_map.items() if canonical in source_df.columns}
    source_df = source_df.rename(columns=rename_map)
    source_df["source_profile"] = profile.name
    return source_df


def _load_sample_bronze(config: dict[str, Any]) -> tuple[pd.DataFrame, str, str]:
    """Generate sample telemetry and optionally convert it into source-profile variants."""
    profile_name = str(config.get("sample_source_profile", "canonical"))
    profile = get_source_profile(profile_name)

    canonical_raw = generate_sample_telemetry(
        num_boats=int(config.get("num_boats", 5)),
        legs_per_boat=int(config.get("legs_per_boat", 6)),
        samples_per_leg=int(config.get("samples_per_leg", 120)),
        sampling_interval_seconds=int(config.get("sampling_interval_seconds", 2)),
        seed=int(config.get("seed", 42)),
        inject_data_issues=bool(config.get("inject_data_issues", False)),
        issue_rate=float(config.get("issue_rate", 0.02)),
        normalize=False,
    )

    bronze = _simulate_profile_variant(canonical_raw, profile)
    return bronze, "sample", profile.name


def _detect_or_select_profile(bronze: pd.DataFrame, config: dict[str, Any], mode: str) -> str:
    """Determine source profile from config or column heuristics."""
    if mode == "sample":
        return str(config.get("sample_source_profile", "canonical"))

    selected = str(config.get("source_profile", "auto"))
    if selected != "auto":
        return selected
    return detect_profile_from_columns(list(bronze.columns))


def _build_integration_report(
    source_name: str,
    source_format: str,
    adapter_name: str,
    profile_name: str,
    bronze: pd.DataFrame,
    silver: pd.DataFrame,
    gold: pd.DataFrame,
    quality_flags: pd.DataFrame,
    applied_transformations: list[str],
    warnings: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Create an integration report artifact dictionary."""
    quality_summary = _summarize_quality(quality_flags)
    dropped_rows = max(0, len(bronze) - len(gold))
    outlier_speed = 0
    if "speed_knots" in silver.columns:
        speed = pd.to_numeric(silver["speed_knots"], errors="coerce")
        median_speed = float(speed.median()) if speed.notna().any() else 0.0
        mad_speed = median_absolute_deviation(speed)
        threshold = median_speed + 4.5 * max(mad_speed, 0.4)
        outlier_speed = int((speed > threshold).sum())

    return {
        "source_name": source_name,
        "source_format": source_format,
        "adapter_used": adapter_name,
        "source_profile": profile_name,
        "rows": {
            "bronze": int(len(bronze)),
            "silver": int(len(silver)),
            "gold": int(len(gold)),
            "dropped": int(dropped_rows),
        },
        "quality": quality_summary,
        "quality_outliers": {
            "speed_outlier_rows": outlier_speed,
        },
        "applied_transformations": applied_transformations,
        "warnings": warnings,
        "time_alignment": {
            "enabled": bool(config.get("resample_enabled", False)),
            "resample_seconds": int(config.get("resample_seconds", 2)),
        },
        "available_formats": list_registered_formats(),
        "available_profiles": list_source_profiles(),
    }


def _parse_non_sample_source(source: Any, config: dict[str, Any]) -> tuple[pd.DataFrame, str, str]:
    """Parse non-sample source using registered adapters."""
    source_name = _source_name_from_input(source)
    source_format = detect_input_format(source_name)
    if source_format == "unknown":
        raise ValueError(f"Unsupported input format for source '{source_name}'.")

    adapter = get_adapter_for_format(source_format)
    bronze = adapter.parse(source)
    return bronze, source_format, adapter.name


def load_csv(file: Any, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load and normalize telemetry from CSV using v1.1 ingestion pipeline."""
    return ingest_telemetry(file, config=config)


def load_sample_data(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load generated sample telemetry via v1.1 ingestion pipeline."""
    return ingest_telemetry("sample", config=config)


def ingest_telemetry(
    source: Any,
    config: dict[str, Any] | None = None,
    return_artifact: bool = False,
) -> pd.DataFrame | IntegrationArtifact:
    """Unified ingestion entrypoint with bronze/silver/gold stages.

    Parameters
    ----------
    source:
        - "sample" / "generated" for synthetic telemetry
        - path to CSV/JSON/GPX/NMEA (supported adapters may vary)
        - file-like object from upload
        - pandas DataFrame
    config:
        Optional ingestion configuration (profile selection, resampling, etc).
    return_artifact:
        When True, returns IntegrationArtifact with stage data and report.
    """
    config = config or {}
    source_name = _source_name_from_input(source)
    warnings: list[str] = []
    applied_transformations: list[str] = []

    mode = "sample" if (source is None or str(source).strip().lower() in {"sample", "generated", "generated sample telemetry"}) else "external"

    if mode == "sample":
        bronze, source_format, profile_hint = _load_sample_bronze(config)
        adapter_name = "sample_generator"
    elif isinstance(source, pd.DataFrame):
        bronze = source.copy()
        source_format = "dataframe"
        profile_hint = "canonical"
        adapter_name = "dataframe_pass_through"
    else:
        bronze, source_format, adapter_name = _parse_non_sample_source(source, config)
        profile_hint = "auto"

    profile_name = _detect_or_select_profile(bronze, config=config, mode=mode)
    profile = get_source_profile(profile_name)
    applied_transformations.append(f"profile_selected: {profile.name}")

    silver, column_rules = _apply_column_map(bronze, profile)
    applied_transformations.extend(column_rules)

    silver, unit_rules = _apply_unit_and_convention_normalization(silver, profile)
    applied_transformations.extend(unit_rules)

    silver, cleanse_rules = _apply_basic_cleansing(silver)
    applied_transformations.extend(cleanse_rules)

    silver, resample_rules = _resample_if_configured(silver, config)
    applied_transformations.extend(resample_rules)

    quality_flags = _compute_quality_flags(silver)

    try:
        gold = normalize_telemetry_dataframe(silver)
    except ValueError as exc:
        raise ValueError(
            f"Ingestion failed after cleansing for profile '{profile.name}': {exc}"
        ) from exc

    report = _build_integration_report(
        source_name=source_name,
        source_format=source_format,
        adapter_name=adapter_name,
        profile_name=profile.name,
        bronze=bronze,
        silver=silver,
        gold=gold,
        quality_flags=quality_flags,
        applied_transformations=applied_transformations,
        warnings=warnings,
        config=config,
    )

    artifact = IntegrationArtifact(
        source_name=source_name,
        source_format=source_format,
        adapter_used=adapter_name,
        source_profile=profile.name,
        bronze_df=bronze,
        silver_df=silver,
        gold_df=gold,
        quality_flags_df=quality_flags,
        report=report,
    )

    if return_artifact:
        return artifact
    return gold
