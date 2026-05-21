"""Open telemetry analytics framework for high-performance sailing."""

from .generator import generate_sample_telemetry
from .ingestion import IntegrationArtifact, detect_input_format, ingest_telemetry, load_csv, load_sample_data
from .schema import (
    CANONICAL_COLUMNS,
    get_optional_columns,
    get_required_columns,
    normalize_telemetry_dataframe,
    validate_telemetry_dataframe,
)
from .source_profiles import list_source_profiles

__all__ = [
    "CANONICAL_COLUMNS",
    "generate_sample_telemetry",
    "get_optional_columns",
    "get_required_columns",
    "IntegrationArtifact",
    "detect_input_format",
    "ingest_telemetry",
    "list_source_profiles",
    "load_csv",
    "load_sample_data",
    "normalize_telemetry_dataframe",
    "validate_telemetry_dataframe",
]

__version__ = "1.0.0"
