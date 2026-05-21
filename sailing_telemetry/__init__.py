"""Open telemetry analytics framework for high-performance sailing."""

from .generator import generate_sample_telemetry
from .ingestion import ingest_telemetry, load_csv, load_sample_data
from .schema import (
    CANONICAL_COLUMNS,
    get_optional_columns,
    get_required_columns,
    normalize_telemetry_dataframe,
    validate_telemetry_dataframe,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "generate_sample_telemetry",
    "get_optional_columns",
    "get_required_columns",
    "ingest_telemetry",
    "load_csv",
    "load_sample_data",
    "normalize_telemetry_dataframe",
    "validate_telemetry_dataframe",
]

__version__ = "1.0.0"
