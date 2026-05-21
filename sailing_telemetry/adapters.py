"""Adapter registry for parsing telemetry from different source formats."""

from __future__ import annotations

import io
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseTelemetryAdapter(ABC):
    """Base adapter interface for telemetry sources."""

    name: str
    supported_formats: set[str]

    @abstractmethod
    def parse(self, source: Any) -> pd.DataFrame:
        """Parse raw source into a pandas DataFrame (bronze stage)."""


class CsvTelemetryAdapter(BaseTelemetryAdapter):
    """CSV adapter supporting paths, bytes, and file-like objects."""

    name = "csv"
    supported_formats = {"csv"}

    def parse(self, source: Any) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source.copy()
        if hasattr(source, "read"):
            return pd.read_csv(source)
        if isinstance(source, (bytes, bytearray)):
            return pd.read_csv(io.BytesIO(source))
        return pd.read_csv(source)


class JsonTelemetryAdapter(BaseTelemetryAdapter):
    """Basic JSON adapter for list/dict structures."""

    name = "json"
    supported_formats = {"json"}

    def parse(self, source: Any) -> pd.DataFrame:
        if isinstance(source, (bytes, bytearray)):
            payload = json.loads(source.decode("utf-8"))
        elif hasattr(source, "read"):
            payload = json.load(source)
        else:
            payload = json.loads(Path(source).read_text(encoding="utf-8"))

        if isinstance(payload, dict):
            if "records" in payload and isinstance(payload["records"], list):
                return pd.DataFrame(payload["records"])
            return pd.DataFrame([payload])
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        raise ValueError("Unsupported JSON payload structure for telemetry adapter.")


class GpxTelemetryAdapter(BaseTelemetryAdapter):
    """Placeholder GPX adapter.

    A future implementation can parse GPX trackpoints (<trkpt>) and map fields
    like timestamp, latitude/longitude, speed, and heading to the canonical
    telemetry schema.
    """

    name = "gpx"
    supported_formats = {"gpx"}

    def parse(self, source: Any) -> pd.DataFrame:
        raise NotImplementedError(
            "GPX adapter is not implemented yet. "
            "Use CSV/JSON for now or add a GPX parser in this adapter."
        )


class NmeaTelemetryAdapter(BaseTelemetryAdapter):
    """Placeholder NMEA adapter.

    A future implementation can parse NMEA 0183 sentences (e.g., RMC, VHW, MWV)
    and NMEA 2000 gateway exports into canonical telemetry columns.
    """

    name = "nmea"
    supported_formats = {"nmea", "log"}

    def parse(self, source: Any) -> pd.DataFrame:
        raise NotImplementedError(
            "NMEA adapter is not implemented yet. "
            "Use CSV/JSON for now or add an NMEA parser in this adapter."
        )


_ADAPTER_REGISTRY: dict[str, BaseTelemetryAdapter] = {}


def register_adapter(adapter: BaseTelemetryAdapter) -> None:
    """Register an adapter for one or more input formats."""
    for fmt in adapter.supported_formats:
        _ADAPTER_REGISTRY[fmt] = adapter


def get_adapter_for_format(input_format: str) -> BaseTelemetryAdapter:
    """Return adapter registered for format or raise helpful error."""
    adapter = _ADAPTER_REGISTRY.get(input_format)
    if adapter is None:
        known = ", ".join(sorted(_ADAPTER_REGISTRY.keys()))
        raise ValueError(f"No adapter registered for format '{input_format}'. Known: {known}")
    return adapter


def list_registered_formats() -> list[str]:
    """List formats with registered adapters."""
    return sorted(_ADAPTER_REGISTRY.keys())


def register_default_adapters() -> None:
    """Register built-in adapters."""
    register_adapter(CsvTelemetryAdapter())
    register_adapter(JsonTelemetryAdapter())
    register_adapter(GpxTelemetryAdapter())
    register_adapter(NmeaTelemetryAdapter())


register_default_adapters()
