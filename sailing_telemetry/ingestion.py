"""Telemetry ingestion layer for sample data and CSV input.

Future adapters can be added here for common sailing telemetry standards:
- NMEA 0183 sentence parsing
- NMEA 2000 gateway payloads
- GPX track and waypoint exports
- Expedition/B&G/Garmin/Raymarine/Sailmon vendor CSV/JSON exports

The current module intentionally focuses on generated sample data and CSV files.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd

from .generator import generate_sample_telemetry
from .schema import normalize_telemetry_dataframe


def detect_input_format(file_name: str) -> str:
    """Infer supported input format from file extension."""
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".json", ".gpx", ".nmea", ".log"}:
        return "future_adapter"
    return "unknown"


def load_csv(file: Any) -> pd.DataFrame:
    """Load telemetry from a CSV file path, bytes, or file-like object."""
    if hasattr(file, "read"):
        raw = pd.read_csv(file)
    elif isinstance(file, (bytes, bytearray)):
        raw = pd.read_csv(io.BytesIO(file))
    else:
        raw = pd.read_csv(file)
    return normalize_telemetry_dataframe(raw)


def load_sample_data(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load generated sample telemetry using optional configuration overrides."""
    config = config or {}
    return generate_sample_telemetry(
        num_boats=int(config.get("num_boats", 5)),
        legs_per_boat=int(config.get("legs_per_boat", 6)),
        samples_per_leg=int(config.get("samples_per_leg", 120)),
        sampling_interval_seconds=int(config.get("sampling_interval_seconds", 2)),
        seed=int(config.get("seed", 42)),
    )


def ingest_telemetry(source: Any, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Unified ingestion entrypoint for sample or CSV telemetry.

    Parameters
    ----------
    source:
        - "sample" or "generated" for synthetic telemetry
        - CSV file-like object (e.g., Streamlit upload)
        - CSV path string
        - pre-loaded pandas DataFrame
    config:
        Optional generator configuration for sample mode.
    """

    config = config or {}

    if isinstance(source, pd.DataFrame):
        return normalize_telemetry_dataframe(source)

    if source is None:
        return load_sample_data(config)

    if isinstance(source, str):
        mode = source.strip().lower()
        if mode in {"sample", "generated"}:
            return load_sample_data(config)

        format_hint = detect_input_format(source)
        if format_hint == "csv":
            return load_csv(source)
        raise ValueError(f"Unsupported input source '{source}'. Use sample mode or CSV.")

    if hasattr(source, "name"):
        format_hint = detect_input_format(str(source.name))
        if format_hint == "csv":
            return load_csv(source)
        raise ValueError(f"Unsupported uploaded file format for '{source.name}'.")

    if hasattr(source, "read"):
        return load_csv(source)

    raise ValueError(
        "Unable to ingest telemetry from source. "
        "Use sample mode, CSV path, upload object, or pandas DataFrame."
    )
