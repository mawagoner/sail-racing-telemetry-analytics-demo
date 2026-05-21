"""Inspect ingestion and cleansing stages from the terminal."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from sailing_telemetry.ingestion import IntegrationArtifact, ingest_telemetry

KEY_COLUMNS = ["timestamp", "boat_id", "leg_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect ingestion stages, transformations, and data quality in the terminal.",
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="sample",
        help="Path to source file or 'sample'/'generated'",
    )
    parser.add_argument(
        "--source-profile",
        default="auto",
        help="Source profile for external files (default: auto)",
    )
    parser.add_argument(
        "--sample-source-profile",
        default="canonical",
        help="Profile variant when source is sample (default: canonical)",
    )
    parser.add_argument(
        "--inject-data-issues",
        action="store_true",
        help="Inject missing/duplicate/out-of-order packets for sample mode",
    )
    parser.add_argument(
        "--issue-rate",
        type=float,
        default=0.03,
        help="Issue injection rate in sample mode (default: 0.03)",
    )
    parser.add_argument(
        "--resample-enabled",
        action="store_true",
        help="Enable silver-stage time alignment/resampling",
    )
    parser.add_argument(
        "--resample-seconds",
        type=int,
        default=2,
        help="Resample cadence in seconds when enabled (default: 2)",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=3,
        help="Rows to preview per section (default: 3)",
    )
    parser.add_argument(
        "--preview-format",
        choices=["vertical", "table"],
        default="vertical",
        help="Preview format for rows and diffs",
    )
    parser.add_argument(
        "--diff-preview-rows",
        type=int,
        default=3,
        help="Changed rows to preview in stage diffs (default: 3)",
    )
    parser.add_argument(
        "--show-quality-flags",
        action="store_true",
        help="Print preview of quality flags rows",
    )
    return parser.parse_args()


def _resolve_source(source: str) -> str:
    lowered = source.strip().lower()
    if lowered in {"sample", "generated", "generated sample telemetry"}:
        return "sample"

    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Input source not found: {path}")
    return str(path.resolve())


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    return str(value)


def _print_preview(frame: pd.DataFrame, rows: int, title: str, fmt: str) -> None:
    if rows <= 0 or frame.empty:
        return

    preview = frame.head(rows)
    if fmt == "table":
        print(f"\n{title} (first {len(preview)} rows, table format):")
        print(preview.to_string(index=False))
        return

    print(f"\n{title} (first {len(preview)} rows, vertical format):")
    for idx, (_, row) in enumerate(preview.iterrows(), start=1):
        print(f"\nRow {idx}")
        for column in preview.columns:
            print(f"  {column:<22}: {_format_value(row[column])}")


def _prepare_for_merge(frame: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    prepared = frame.copy()
    if "timestamp" in key_cols and "timestamp" in prepared.columns:
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    if "boat_id" in key_cols and "boat_id" in prepared.columns:
        prepared["boat_id"] = prepared["boat_id"].astype(str).str.strip()
    if "leg_id" in key_cols and "leg_id" in prepared.columns:
        prepared["leg_id"] = prepared["leg_id"].astype(str).str.strip()

    if key_cols:
        prepared = prepared.sort_values(key_cols)
        prepared["_dup_idx"] = prepared.groupby(key_cols, dropna=False).cumcount()
    else:
        prepared["_row_idx"] = range(len(prepared))
    return prepared


def _print_stage_diff(
    before: pd.DataFrame,
    after: pd.DataFrame,
    before_name: str,
    after_name: str,
    preview_rows: int,
    preview_format: str,
) -> None:
    print(f"\nStage diff: {before_name} -> {after_name}")

    key_cols = [column for column in KEY_COLUMNS if column in before.columns and column in after.columns]
    left = _prepare_for_merge(before, key_cols)
    right = _prepare_for_merge(after, key_cols)

    merge_keys = key_cols + ["_dup_idx"] if key_cols else ["_row_idx"]
    merged = left.merge(right, on=merge_keys, how="inner", suffixes=("_before", "_after"))

    shared_columns = [
        column
        for column in before.columns
        if column in after.columns and column not in key_cols
    ]
    added_columns = [column for column in after.columns if column not in before.columns]
    removed_columns = [column for column in before.columns if column not in after.columns]

    print(f"- {before_name} rows: {len(before)}")
    print(f"- {after_name} rows: {len(after)}")
    print(f"- Comparable rows: {len(merged)}")
    print(f"- Key columns: {', '.join(key_cols) if key_cols else '(row index fallback)'}")
    print(f"- Added columns: {', '.join(added_columns) if added_columns else 'None'}")
    print(f"- Removed columns: {', '.join(removed_columns) if removed_columns else 'None'}")

    changed_counts: dict[str, int] = {}
    changed_any = pd.Series(False, index=merged.index)
    for column in shared_columns:
        before_col = f"{column}_before"
        after_col = f"{column}_after"
        if before_col not in merged.columns or after_col not in merged.columns:
            continue

        changed_mask = (
            merged[before_col].fillna("<NA>").astype(str)
            != merged[after_col].fillna("<NA>").astype(str)
        )
        changed_counts[column] = int(changed_mask.sum())
        changed_any = changed_any | changed_mask

    changed_columns = {column: count for column, count in changed_counts.items() if count > 0}
    if not changed_columns:
        print("- Changed cells by column: none")
        return

    print("- Changed cells by column:")
    for column, count in sorted(changed_columns.items(), key=lambda item: item[1], reverse=True):
        print(f"  - {column}: {count}")

    if preview_rows <= 0:
        return

    preview = merged.loc[changed_any].head(preview_rows)
    if preview.empty:
        return

    if preview_format == "table":
        pair_columns: list[str] = []
        for column in changed_columns.keys():
            pair_columns.extend([f"{column}_before", f"{column}_after"])
        display_columns = merge_keys + pair_columns
        print(f"\nChanged row preview (first {len(preview)} rows, table format):")
        print(preview[display_columns].to_string(index=False))
        return

    print(f"\nChanged row preview (first {len(preview)} rows, vertical format):")
    for idx, (_, row) in enumerate(preview.iterrows(), start=1):
        print(f"\nRow {idx}")
        for key in merge_keys:
            print(f"  {key:<22}: {_format_value(row[key])}")

        for column in changed_columns.keys():
            before_value = _format_value(row[f"{column}_before"])
            after_value = _format_value(row[f"{column}_after"])
            if before_value == after_value:
                continue
            print(f"  {column:<22}: {before_value}  ->  {after_value}")


def _print_quality_summary(artifact: IntegrationArtifact) -> None:
    quality = artifact.report.get("quality", {})
    flag_counts = quality.get("flag_counts", {})
    print("\nQuality summary:")
    print(f"- Mean row quality score: {quality.get('mean_quality_score', 0.0)}")
    print(f"- Rows with issues: {quality.get('rows_with_issues', 0)}")
    if flag_counts:
        print("- Flag counts:")
        for flag_name, flag_count in flag_counts.items():
            print(f"  - {flag_name}: {flag_count}")


def _print_quality_flags_preview(artifact: IntegrationArtifact, rows: int, fmt: str) -> None:
    if rows <= 0 or artifact.quality_flags_df.empty:
        return

    flag_columns = [
        "missing_required_fields",
        "out_of_range_speed",
        "timestamp_jump",
        "duplicate_packet",
    ]
    available_flag_columns = [column for column in flag_columns if column in artifact.quality_flags_df.columns]
    if not available_flag_columns:
        return

    issue_mask = artifact.quality_flags_df[available_flag_columns].any(axis=1)
    preview = artifact.quality_flags_df.loc[issue_mask]
    if preview.empty:
        print("\nQuality flags preview: no flagged rows")
        return

    display_columns = ["timestamp", "boat_id"] + available_flag_columns + ["row_quality_score"]
    display_columns = [column for column in display_columns if column in preview.columns]
    _print_preview(preview[display_columns], rows, "Quality flags preview", fmt)


def main() -> None:
    args = parse_args()
    source = _resolve_source(args.source)
    config = {
        "source_profile": args.source_profile,
        "sample_source_profile": args.sample_source_profile,
        "inject_data_issues": args.inject_data_issues,
        "issue_rate": float(args.issue_rate),
        "resample_enabled": bool(args.resample_enabled),
        "resample_seconds": max(1, int(args.resample_seconds)),
    }

    artifact = ingest_telemetry(source, config=config, return_artifact=True)
    rows = artifact.report.get("rows", {})

    print("Ingestion inspection")
    print(f"- Source: {artifact.source_name}")
    print(f"- Source format: {artifact.source_format}")
    print(f"- Adapter: {artifact.adapter_used}")
    print(f"- Profile: {artifact.source_profile}")
    print(
        "- Rows (bronze/silver/gold/dropped): "
        f"{rows.get('bronze', 0)}/{rows.get('silver', 0)}/{rows.get('gold', 0)}/{rows.get('dropped', 0)}"
    )

    transformations = artifact.report.get("applied_transformations", [])
    print("\nApplied transformations:")
    if not transformations:
        print("- None")
    else:
        for step in transformations:
            print(f"- {step}")

    _print_quality_summary(artifact)

    _print_stage_diff(
        artifact.bronze_df,
        artifact.silver_df,
        before_name="bronze",
        after_name="silver",
        preview_rows=max(0, int(args.diff_preview_rows)),
        preview_format=args.preview_format,
    )
    _print_stage_diff(
        artifact.silver_df,
        artifact.gold_df,
        before_name="silver",
        after_name="gold",
        preview_rows=max(0, int(args.diff_preview_rows)),
        preview_format=args.preview_format,
    )

    _print_preview(artifact.bronze_df, max(0, int(args.preview_rows)), "Bronze preview", args.preview_format)
    _print_preview(artifact.silver_df, max(0, int(args.preview_rows)), "Silver preview", args.preview_format)
    _print_preview(artifact.gold_df, max(0, int(args.preview_rows)), "Gold preview", args.preview_format)

    if args.show_quality_flags:
        _print_quality_flags_preview(artifact, max(0, int(args.preview_rows)), args.preview_format)


if __name__ == "__main__":
    main()
