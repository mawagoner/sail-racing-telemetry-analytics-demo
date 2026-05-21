"""Source profile metadata for telemetry column/units conventions.

Profiles capture how external sources may name fields and represent units.
They enable ingestion to map source data into the canonical telemetry schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import CANONICAL_COLUMNS


@dataclass(frozen=True)
class SourceProfile:
    """Describes source column naming and unit conventions."""

    name: str
    description: str
    column_map: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    twa_sign_convention: str = "starboard_positive"
    heading_reference: str = "true"
    wind_direction_reference: str = "true"
    value_maps: dict[str, dict[str, str]] = field(default_factory=dict)


SOURCE_PROFILES: dict[str, SourceProfile] = {
    "canonical": SourceProfile(
        name="canonical",
        description="Canonical schema fields and units.",
    ),
    "expedition_csv": SourceProfile(
        name="expedition_csv",
        description="Expedition-like CSV naming conventions.",
        column_map={
            "UTC": "timestamp",
            "Boat": "boat_id",
            "Team": "team_name",
            "Leg": "leg_id",
            "Mode": "leg_mode",
            "BSP": "speed_knots",
            "VMG": "vmg",
            "TWS": "true_wind_speed",
            "TWA": "true_wind_angle",
            "TWD": "true_wind_direction",
            "Gust": "gust_knots",
            "HDG": "heading",
            "Heel": "heel_angle",
            "Foil": "foil_cant",
            "X": "x_position",
            "Y": "y_position",
            "Maneuver": "maneuver_type",
            "Side": "course_side",
            "Sea": "sea_state",
        },
    ),
    "bg_csv": SourceProfile(
        name="bg_csv",
        description="B&G-style export with metric speed columns.",
        column_map={
            "Time": "timestamp",
            "Vessel": "boat_id",
            "Crew": "team_name",
            "LegNo": "leg_id",
            "PointOfSail": "leg_mode",
            "BoatSpeed_mps": "speed_knots",
            "VMG_mps": "vmg",
            "WindSpeed_mps": "true_wind_speed",
            "TWA_deg": "true_wind_angle",
            "TWD_deg": "true_wind_direction",
            "Gust_mps": "gust_knots",
            "Heading_deg": "heading",
            "Heel_deg": "heel_angle",
            "FoilCant_deg": "foil_cant",
            "PosX_m": "x_position",
            "PosY_m": "y_position",
            "Maneuver": "maneuver_type",
            "TackSide": "course_side",
            "SeaState": "sea_state",
        },
        units={
            "speed_knots": "mps",
            "vmg": "mps",
            "true_wind_speed": "mps",
            "gust_knots": "mps",
        },
        value_maps={
            "course_side": {"S": "starboard", "P": "port"},
        },
    ),
    "garmin_csv": SourceProfile(
        name="garmin_csv",
        description="Garmin-like telemetry dump with condensed labels.",
        column_map={
            "ts": "timestamp",
            "vessel_id": "boat_id",
            "team": "team_name",
            "leg": "leg_id",
            "mode": "leg_mode",
            "speed_mps": "speed_knots",
            "vmg_mps": "vmg",
            "tws_mps": "true_wind_speed",
            "twa": "true_wind_angle",
            "twd": "true_wind_direction",
            "gust_mps": "gust_knots",
            "hdg": "heading",
            "heel": "heel_angle",
            "foil": "foil_cant",
            "x_m": "x_position",
            "y_m": "y_position",
            "maneuver": "maneuver_type",
            "side": "course_side",
            "sea": "sea_state",
        },
        units={
            "speed_knots": "mps",
            "vmg": "mps",
            "true_wind_speed": "mps",
            "gust_knots": "mps",
        },
    ),
    "raymarine_csv": SourceProfile(
        name="raymarine_csv",
        description="Raymarine-like export with radians for angle values.",
        column_map={
            "time_utc": "timestamp",
            "boat": "boat_id",
            "team_name": "team_name",
            "leg_id": "leg_id",
            "point_of_sail": "leg_mode",
            "boat_speed_kn": "speed_knots",
            "vmg_kn": "vmg",
            "wind_speed_kn": "true_wind_speed",
            "twa_rad": "true_wind_angle",
            "twd_rad": "true_wind_direction",
            "gust_kn": "gust_knots",
            "heading_rad": "heading",
            "heel_deg": "heel_angle",
            "foil_deg": "foil_cant",
            "track_x": "x_position",
            "track_y": "y_position",
            "turn": "maneuver_type",
            "side": "course_side",
            "sea_state": "sea_state",
        },
        units={
            "true_wind_angle": "rad",
            "true_wind_direction": "rad",
            "heading": "rad",
        },
    ),
    "sailmon_csv": SourceProfile(
        name="sailmon_csv",
        description="Sailmon-style feed with port-positive TWA sign convention.",
        column_map={
            "timestamp": "timestamp",
            "boat": "boat_id",
            "team": "team_name",
            "leg_number": "leg_id",
            "sailing_mode": "leg_mode",
            "boat_speed": "speed_knots",
            "vmg": "vmg",
            "tws": "true_wind_speed",
            "twa": "true_wind_angle",
            "twd": "true_wind_direction",
            "gust": "gust_knots",
            "heading": "heading",
            "heel": "heel_angle",
            "foil_cant": "foil_cant",
            "x": "x_position",
            "y": "y_position",
            "maneuver_type": "maneuver_type",
            "course_side": "course_side",
            "sea_state": "sea_state",
        },
        twa_sign_convention="port_positive",
    ),
}


def list_source_profiles() -> list[str]:
    """List known source profile names."""
    return sorted(SOURCE_PROFILES.keys())


def get_source_profile(name: str | None) -> SourceProfile:
    """Return profile by name, defaulting to canonical."""
    if not name:
        return SOURCE_PROFILES["canonical"]
    return SOURCE_PROFILES.get(name, SOURCE_PROFILES["canonical"])


def detect_profile_from_columns(columns: list[str]) -> str:
    """Heuristically detect source profile from available columns."""
    column_set = set(columns)

    canonical_overlap = len(column_set.intersection(set(CANONICAL_COLUMNS)))
    if canonical_overlap >= 10:
        return "canonical"

    best_profile = "canonical"
    best_score = -1
    for name, profile in SOURCE_PROFILES.items():
        if name == "canonical":
            continue
        source_columns = set(profile.column_map.keys())
        score = len(source_columns.intersection(column_set))
        if score > best_score:
            best_profile = name
            best_score = score

    if best_score <= 0:
        return "canonical"
    return best_profile
