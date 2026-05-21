"""Example Streamlit app for the Open Telemetry Analytics Framework."""

from __future__ import annotations

import io
import time

import pandas as pd
import streamlit as st

from sailing_telemetry import __version__, ingest_telemetry, list_source_profiles
from sailing_telemetry.analytics import (
    calculate_vmg_bands,
    compare_boats,
    compare_teams,
    create_comparison_table,
    detect_anomalies,
    detect_maneuvers,
    estimate_optimal_vmg_zones,
    prepare_weather_overlay,
    recommend_target_twa_ranges,
    summarize_by_boat,
    summarize_by_leg,
    summarize_kpis,
    summarize_maneuvers,
)
from sailing_telemetry.replay import get_replay_frame, get_replay_history
from sailing_telemetry.visualization import (
    plot_anomalies,
    plot_race_replay,
    plot_speed_over_time,
    plot_team_comparison,
    plot_twa_vs_vmg,
    plot_vmg_bands,
    plot_weather_over_time,
)


st.set_page_config(page_title="Sail Racing Telemetry Analytics Demo", layout="wide")


@st.cache_data(show_spinner=False)
def load_sample_data(config: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    artifact = ingest_telemetry("sample", config=config, return_artifact=True)
    return artifact.gold_df, artifact.report, artifact.quality_flags_df


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes: bytes, config: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    artifact = ingest_telemetry(io.BytesIO(file_bytes), config=config, return_artifact=True)
    return artifact.gold_df, artifact.report, artifact.quality_flags_df


def _initialize_live_state() -> None:
    if "live_running" not in st.session_state:
        st.session_state.live_running = False
    if "live_index" not in st.session_state:
        st.session_state.live_index = 0
    if "live_signature" not in st.session_state:
        st.session_state.live_signature = ""


def _configure_sidebar_source() -> tuple[str, dict[str, object], bytes | None]:
    profiles = list_source_profiles()
    source_mode = st.sidebar.radio(
        "Data source",
        options=["Generated sample telemetry", "Upload CSV telemetry"],
        index=0,
    )

    sample_config: dict[str, object] = {
        "num_boats": 5,
        "legs_per_boat": 6,
        "samples_per_leg": 120,
        "sampling_interval_seconds": 2,
        "seed": 42,
        "sample_source_profile": "canonical",
        "source_profile": "auto",
        "inject_data_issues": False,
        "issue_rate": 0.02,
        "resample_enabled": False,
        "resample_seconds": 2,
    }
    upload_bytes: bytes | None = None

    if source_mode == "Generated sample telemetry":
        sample_config["num_boats"] = st.sidebar.slider("Number of boats", 2, 8, 5)
        sample_config["legs_per_boat"] = st.sidebar.slider("Legs per boat", 4, 12, 6)
        sample_config["samples_per_leg"] = st.sidebar.slider("Samples per leg", 60, 260, 120, step=10)
        sample_config["sampling_interval_seconds"] = st.sidebar.selectbox("Sampling interval (seconds)", [1, 2, 5], index=1)
        sample_config["seed"] = int(st.sidebar.number_input("Random seed", min_value=0, max_value=99999, value=42))
        sample_config["sample_source_profile"] = st.sidebar.selectbox(
            "Simulated source profile",
            options=profiles,
            index=profiles.index("canonical") if "canonical" in profiles else 0,
            help="Generate canonical data or profile-shaped source variants for ingestion testing.",
        )
        sample_config["inject_data_issues"] = st.sidebar.checkbox(
            "Inject realistic data issues",
            value=False,
            help="Inject missing values, duplicates, and out-of-order rows.",
        )
        sample_config["issue_rate"] = st.sidebar.slider(
            "Issue injection rate",
            min_value=0.0,
            max_value=0.20,
            value=0.02,
            step=0.01,
            disabled=not bool(sample_config["inject_data_issues"]),
        )
    else:
        uploaded = st.sidebar.file_uploader("Upload telemetry CSV", type=["csv"])
        if uploaded is not None:
            upload_bytes = uploaded.getvalue()
        profile_options = ["auto"] + profiles
        sample_config["source_profile"] = st.sidebar.selectbox(
            "Source profile override",
            options=profile_options,
            index=0,
            help="Use auto-detection or force a known source profile.",
        )

    sample_config["resample_enabled"] = st.sidebar.checkbox(
        "Enable time alignment/resampling",
        value=False,
        help="Align telemetry to fixed cadence and interpolate numeric channels.",
    )
    sample_config["resample_seconds"] = st.sidebar.slider(
        "Resample seconds",
        min_value=1,
        max_value=10,
        value=2,
        disabled=not bool(sample_config["resample_enabled"]),
    )

    return source_mode, sample_config, upload_bytes


def _configure_sidebar_filters(df: pd.DataFrame) -> dict[str, object]:
    _initialize_live_state()

    options: dict[str, object] = {}
    available_boats = sorted(df["boat_id"].unique().tolist())
    selected_boats = st.sidebar.multiselect("Selected boats", options=available_boats, default=available_boats)
    options["selected_boats"] = selected_boats

    filtered = df[df["boat_id"].isin(selected_boats)].copy()
    replay_timestamps = filtered["timestamp"].sort_values().unique()
    max_index = max(0, len(replay_timestamps) - 1)

    live_signature = f"{selected_boats}-{len(replay_timestamps)}"
    if st.session_state.live_signature != live_signature:
        st.session_state.live_signature = live_signature
        st.session_state.live_running = False
        st.session_state.live_index = 0

    live_enabled = st.sidebar.checkbox("Enable live simulation", value=False)
    playback_step = st.sidebar.slider("Live playback step", 1, 10, 2)
    live_window_steps = st.sidebar.slider("Live speed window (steps)", 40, 600, 200)

    if live_enabled:
        col_a, col_b, col_c = st.sidebar.columns(3)
        if col_a.button("Start"):
            st.session_state.live_running = True
        if col_b.button("Stop"):
            st.session_state.live_running = False
        if col_c.button("Reset"):
            st.session_state.live_running = False
            st.session_state.live_index = 0

        manual_idx = st.sidebar.slider(
            "Replay timestamp step",
            min_value=0,
            max_value=max_index,
            value=min(st.session_state.live_index, max_index),
            disabled=st.session_state.live_running,
            key="live_replay_idx",
        )
        if not st.session_state.live_running:
            st.session_state.live_index = manual_idx
    else:
        st.session_state.live_running = False
        manual_idx = st.sidebar.slider(
            "Replay timestamp step",
            min_value=0,
            max_value=max_index,
            value=min(st.session_state.live_index, max_index),
            key="manual_replay_idx",
        )
        st.session_state.live_index = manual_idx

    chart_col_1, chart_col_2, chart_col_3 = st.sidebar.columns(3)
    options["show_maneuver_markers"] = chart_col_1.checkbox("M", value=True, help="Show maneuver markers")
    options["show_anomaly_markers"] = chart_col_2.checkbox("A", value=True, help="Show anomaly markers")
    options["show_weather_in_replay"] = chart_col_3.checkbox("W", value=True, help="Show weather in replay")

    options["live_enabled"] = live_enabled
    options["playback_step"] = playback_step
    options["live_window_steps"] = live_window_steps
    options["replay_timestamps"] = replay_timestamps
    options["filtered_df"] = filtered
    options["current_replay_index"] = min(st.session_state.live_index, max_index)
    return options


def render() -> None:
    st.title("Sail Racing Telemetry Analytics Demo")
    st.caption("v1.0 — Open Telemetry Analytics Framework Prototype")
    st.info(
        "This is simulated data designed to demonstrate telemetry analytics workflows "
        "for high-performance sailing."
    )

    source_mode, sample_config, upload_bytes = _configure_sidebar_source()

    try:
        if source_mode == "Generated sample telemetry":
            telemetry, integration_report, quality_flags = load_sample_data(sample_config)
        else:
            if upload_bytes is None:
                st.warning("Upload a CSV file to continue.")
                st.stop()
            telemetry, integration_report, quality_flags = load_uploaded_csv(upload_bytes, sample_config)
    except ValueError as error:
        st.error(f"Telemetry ingestion failed: {error}")
        st.stop()

    if telemetry.empty:
        st.warning("Telemetry dataset is empty after ingestion.")
        st.stop()

    sidebar_options = _configure_sidebar_filters(telemetry)
    filtered = sidebar_options["filtered_df"]
    selected_boats = sidebar_options["selected_boats"]

    if not selected_boats:
        st.warning("Select at least one boat in the sidebar.")
        st.stop()
    if filtered.empty:
        st.warning("No rows remain after filtering. Adjust selected boats.")
        st.stop()

    replay_timestamps = sidebar_options["replay_timestamps"]
    if len(replay_timestamps) == 0:
        st.warning("No replay timestamps are available after filtering.")
        st.stop()

    current_idx = int(sidebar_options["current_replay_index"])
    current_timestamp = pd.Timestamp(replay_timestamps[current_idx])

    maneuvers = detect_maneuvers(filtered)
    anomalies = detect_anomalies(filtered)
    weather_overlay = prepare_weather_overlay(filtered)
    kpis = summarize_kpis(filtered)
    leg_summary = summarize_by_leg(filtered)
    boat_summary = summarize_by_boat(filtered)
    maneuver_summary = summarize_maneuvers(maneuvers)
    vmg_bands = calculate_vmg_bands(filtered)
    vmg_zones = estimate_optimal_vmg_zones(filtered)
    target_ranges = recommend_target_twa_ranges(filtered)
    comparison_table = create_comparison_table(filtered)

    st.header("Overview")
    st.write(
        "This app is an example client for a modular telemetry framework. "
        "It demonstrates schema normalization, ingestion, analytics, replay, and visualization "
        "using simulated race sessions."
    )

    st.header("Data Source")
    st.write(f"Source mode: `{source_mode}`")
    st.write(f"Rows: `{len(telemetry)}` | Filtered rows: `{len(filtered)}` | Boats in filter: `{len(selected_boats)}`")

    quality = integration_report.get("quality", {}) if isinstance(integration_report, dict) else {}
    flag_counts = quality.get("flag_counts", {}) if isinstance(quality, dict) else {}
    quality_cols = st.columns(5)
    quality_cols[0].metric("Mean Quality", f"{quality.get('mean_quality_score', 0.0):.3f}")
    quality_cols[1].metric("Rows w/ Issues", f"{quality.get('rows_with_issues', 0)}")
    quality_cols[2].metric("Missing Required", f"{flag_counts.get('missing_required_fields', 0)}")
    quality_cols[3].metric("Out-of-Range Speed", f"{flag_counts.get('out_of_range_speed', 0)}")
    quality_cols[4].metric("Timestamp Jumps", f"{flag_counts.get('timestamp_jump', 0)}")

    with st.expander("Integration report artifact", expanded=False):
        st.json(integration_report)
        if not quality_flags.empty:
            st.subheader("Quality flag sample")
            st.dataframe(quality_flags.head(20), use_container_width=True, hide_index=True)

    st.header("KPI Summary")
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Max Speed", f"{kpis['max_speed']:.2f} kn")
    kpi_cols[1].metric("Average Speed", f"{kpis['avg_speed']:.2f} kn")
    kpi_cols[2].metric("Best VMG", f"{kpis['best_vmg']:.2f} kn")
    kpi_cols[3].metric("Avg Maneuver Loss", f"{kpis['avg_maneuver_loss']:.2f} kn")
    kpi_cols[4].metric("Data Points", f"{int(kpis['data_points'])}")

    st.header("Race Replay")
    replay_fig = plot_race_replay(
        filtered,
        current_timestamp=current_timestamp,
        show_weather=bool(sidebar_options["show_weather_in_replay"]),
    )
    st.plotly_chart(replay_fig, use_container_width=True)

    replay_frame = get_replay_frame(filtered, current_timestamp)
    st.dataframe(
        replay_frame[
            [
                "timestamp",
                "boat_id",
                "team_name",
                "leg_id",
                "leg_mode",
                "speed_knots",
                "vmg",
                "x_position",
                "y_position",
            ]
        ].round(3),
        use_container_width=True,
        hide_index=True,
    )

    if bool(sidebar_options["live_enabled"]):
        st.subheader("Live Streaming Simulation")
        replay_history = get_replay_history(filtered, current_timestamp)
        live_kpis = summarize_kpis(replay_history)
        live_cols = st.columns(4)
        live_cols[0].metric("Live Max Speed", f"{live_kpis['max_speed']:.2f} kn")
        live_cols[1].metric("Live Avg Speed", f"{live_kpis['avg_speed']:.2f} kn")
        live_cols[2].metric("Live Best VMG", f"{live_kpis['best_vmg']:.2f} kn")
        live_cols[3].metric("Live Rows", f"{int(live_kpis['data_points'])}")

        start_idx = max(0, current_idx - int(sidebar_options["live_window_steps"]))
        start_ts = pd.Timestamp(replay_timestamps[start_idx])
        live_window = replay_history[replay_history["timestamp"] >= start_ts]
        st.plotly_chart(plot_speed_over_time(live_window), use_container_width=True)

    st.header("Speed & Maneuver Analysis")
    speed_fig = plot_speed_over_time(
        filtered,
        maneuvers=maneuvers if bool(sidebar_options["show_maneuver_markers"]) else None,
        anomalies=anomalies if bool(sidebar_options["show_anomaly_markers"]) else None,
    )
    st.plotly_chart(speed_fig, use_container_width=True)

    maneuver_cols = st.columns(5)
    maneuver_cols[0].metric("Maneuvers", f"{maneuver_summary['total_maneuvers']}")
    maneuver_cols[1].metric("Tacks", f"{maneuver_summary['tack_count']}")
    maneuver_cols[2].metric("Gybes", f"{maneuver_summary['gybe_count']}")
    maneuver_cols[3].metric("Avg Loss", f"{maneuver_summary['avg_speed_loss']:.2f} kn")
    maneuver_cols[4].metric("Avg Recovery", f"{maneuver_summary['avg_recovery_time']:.1f} s")
    st.dataframe(maneuvers.round(3), use_container_width=True, hide_index=True)

    st.header("Anomaly Detection")
    anomaly_fig = plot_anomalies(filtered, anomalies if bool(sidebar_options["show_anomaly_markers"]) else pd.DataFrame())
    st.plotly_chart(anomaly_fig, use_container_width=True)

    if anomalies.empty:
        st.info("No anomalies detected in current filter scope.")
    else:
        counts = anomalies["anomaly_type"].value_counts().to_dict()
        anomaly_cols = st.columns(4)
        anomaly_cols[0].metric("Speed Drops", f"{counts.get('speed_drop', 0)}")
        anomaly_cols[1].metric("Abnormal Heel", f"{counts.get('abnormal_heel', 0)}")
        anomaly_cols[2].metric("Poor VMG", f"{counts.get('poor_vmg', 0)}")
        anomaly_cols[3].metric("Foil Instability", f"{counts.get('foil_instability', 0)}")
        st.dataframe(anomalies, use_container_width=True, hide_index=True)

    st.header("Weather Overlay")
    weather_fig = plot_weather_over_time(weather_overlay["weather"])
    st.plotly_chart(weather_fig, use_container_width=True)
    st.write(
        f"Detected gust periods: `{len(weather_overlay['gust_periods'])}` | "
        f"Detected wind shifts: `{len(weather_overlay['shift_periods'])}`"
    )
    if len(weather_overlay["gust_periods"]):
        st.dataframe(weather_overlay["gust_periods"], use_container_width=True, hide_index=True)
    if len(weather_overlay["shift_periods"]):
        st.dataframe(weather_overlay["shift_periods"], use_container_width=True, hide_index=True)

    st.header("Optimal VMG Zones")
    vmg_cols = st.columns(2)
    vmg_cols[0].metric("Optimal Upwind TWA", str(vmg_zones["optimal_upwind_twa_range"]))
    vmg_cols[1].metric("Optimal Downwind TWA", str(vmg_zones["optimal_downwind_twa_range"]))
    st.info(str(vmg_zones["confidence_note"]))
    st.plotly_chart(plot_twa_vs_vmg(filtered, vmg_zones), use_container_width=True)
    st.plotly_chart(plot_vmg_bands(vmg_bands), use_container_width=True)
    st.dataframe(target_ranges, use_container_width=True, hide_index=True)
    st.dataframe(vmg_zones["supporting_table"], use_container_width=True, hide_index=True)

    st.header("Team-vs-Team Comparison")
    compare_boat_options = sorted(filtered["boat_id"].unique().tolist())
    if len(compare_boat_options) >= 2:
        comp_col_1, comp_col_2 = st.columns(2)
        boat_a = comp_col_1.selectbox("Boat A", options=compare_boat_options, index=0)
        boat_b = comp_col_2.selectbox("Boat B", options=compare_boat_options, index=1)

        if boat_a == boat_b:
            st.warning("Choose two different boats for direct comparison.")
        else:
            boat_comp = compare_boats(filtered, boat_a, boat_b)
            st.dataframe(boat_comp, use_container_width=True, hide_index=True)
            st.plotly_chart(plot_team_comparison(boat_comp), use_container_width=True)
    else:
        st.info("Need at least two boats in current selection for comparison.")

    st.subheader("Team Aggregate Table")
    st.dataframe(compare_teams(filtered), use_container_width=True, hide_index=True)

    st.header("Leg Comparison")
    st.dataframe(leg_summary, use_container_width=True, hide_index=True)
    st.subheader("Boat Summary")
    st.dataframe(boat_summary, use_container_width=True, hide_index=True)
    st.subheader("Comparison Table")
    st.dataframe(comparison_table, use_container_width=True, hide_index=True)

    st.header("About This Framework")
    st.markdown(f"**Version:** v1.0 — Open Telemetry Analytics Framework Prototype (`{__version__}`)")
    st.info(
        "This project demonstrates how raw sailing telemetry can be normalized, analyzed, "
        "replayed, and visualized through a reusable analytics framework. The current "
        "implementation uses simulated race data, but the architecture is designed to "
        "support future adapters for real-world sailing telemetry formats."
    )

    if bool(sidebar_options["live_enabled"]) and st.session_state.live_running:
        max_index = len(replay_timestamps) - 1
        if current_idx < max_index:
            st.session_state.live_index = min(max_index, current_idx + int(sidebar_options["playback_step"]))
            time.sleep(0.18)
            st.rerun()
        else:
            st.session_state.live_running = False


if __name__ == "__main__":
    render()
