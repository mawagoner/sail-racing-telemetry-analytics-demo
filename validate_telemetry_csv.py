"""CLI helper to validate and normalize telemetry CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sailing_telemetry.schema import normalize_telemetry_dataframe, validate_telemetry_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and normalize a telemetry CSV against the canonical schema.",
    )
    parser.add_argument("input_csv", help="Path to input telemetry CSV")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for normalized CSV",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="How many normalized rows to print as preview (default: 5)",
    )
    parser.add_argument(
        "--preview-format",
        choices=["vertical", "table"],
        default="vertical",
        help="Preview output format: vertical key/value or table",
    )
    parser.add_argument(
        "--show-diff-summary",
        action="store_true",
        help="Show a column-level summary of normalization changes",
    )
    parser.add_argument(
        "--diff-preview-rows",
        type=int,
        default=5,
        help="How many changed rows to preview in diff mode (default: 5)",
    )
    return parser.parse_args()


def _print_diff_summary(
    raw: pd.DataFrame,
    normalized: pd.DataFrame,
    preview_rows: int,
    preview_format: str,
) -> None:
    """Print normalization change summary and changed-row preview."""
    raw_work = raw.copy()
    norm_work = normalized.copy()

    key_cols = ["timestamp", "boat_id", "leg_id"]
    for frame in [raw_work, norm_work]:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
        frame["boat_id"] = frame["boat_id"].astype(str).str.strip()
        frame["leg_id"] = frame["leg_id"].astype(str).str.strip()
        frame.sort_values(key_cols, inplace=True)
        frame["_dup_idx"] = frame.groupby(key_cols, dropna=False).cumcount()

    merged = raw_work.merge(
        norm_work,
        on=key_cols + ["_dup_idx"],
        suffixes=("_raw", "_norm"),
        how="inner",
    )

    common_columns = [column for column in raw.columns if column in normalized.columns and column not in key_cols]
    added_columns = [column for column in normalized.columns if column not in raw.columns]

    print("\nNormalization diff summary:")
    print(f"- Input rows: {len(raw)}")
    print(f"- Normalized rows: {len(normalized)}")
    print(f"- Dropped rows: {max(0, len(raw) - len(normalized))}")
    print(f"- Added columns: {', '.join(added_columns) if added_columns else 'None'}")
    print(f"- Comparable rows (matched by timestamp/boat_id/leg_id): {len(merged)}")

    changed_counts: dict[str, int] = {}
    changed_any = pd.Series(False, index=merged.index)

    for column in common_columns:
        raw_col = f"{column}_raw"
        norm_col = f"{column}_norm"
        if raw_col not in merged.columns or norm_col not in merged.columns:
            continue

        changed_mask = (
            merged[raw_col].fillna("<NA>").astype(str)
            != merged[norm_col].fillna("<NA>").astype(str)
        )
        changed_counts[column] = int(changed_mask.sum())
        changed_any = changed_any | changed_mask

    changed_columns = {key: value for key, value in changed_counts.items() if value > 0}
    if not changed_columns:
        print("- Changed cells by column: none")
        return

    print("- Changed cells by column:")
    for column, count in sorted(changed_columns.items(), key=lambda item: item[1], reverse=True):
        print(f"  - {column}: {count}")

    preview = merged.loc[changed_any].head(preview_rows)
    if not preview.empty and preview_format == "table":
        preview_column_pairs = []
        for column in changed_columns.keys():
            preview_column_pairs.extend([f"{column}_raw", f"{column}_norm"])

        preview_cols = key_cols + preview_column_pairs
        print(f"\nChanged row preview (first {len(preview)} rows, table format):")
        print(preview[preview_cols].to_string(index=False))
        return

    if not preview.empty:
        print(f"\nChanged row preview (first {len(preview)} rows, vertical format):")

        def _fmt(value: object) -> str:
            if pd.isna(value):
                return "<NA>"
            return str(value)

        for idx, (_, row) in enumerate(preview.iterrows(), start=1):
            print(f"\nRow {idx}")
            print(f"  {'timestamp':<22}: {_fmt(row['timestamp'])}")
            print(f"  {'boat_id':<22}: {_fmt(row['boat_id'])}")
            print(f"  {'leg_id':<22}: {_fmt(row['leg_id'])}")

            row_changed_cols = []
            for column in changed_columns.keys():
                raw_col = f"{column}_raw"
                norm_col = f"{column}_norm"
                if raw_col not in row or norm_col not in row:
                    continue
                if _fmt(row[raw_col]) != _fmt(row[norm_col]):
                    row_changed_cols.append(column)

            if not row_changed_cols:
                print("  (No changed fields in this preview row)")
                continue

            for column in row_changed_cols:
                raw_value = _fmt(row[f"{column}_raw"])
                norm_value = _fmt(row[f"{column}_norm"])
                print(f"  {column:<22}: {raw_value}  ->  {norm_value}")


def _print_vertical_rows(frame: pd.DataFrame, row_count: int, section_title: str) -> None:
    """Print dataframe rows in vertical key/value format."""
    if row_count <= 0 or frame.empty:
        return

    preview = frame.head(row_count)
    print(f"\n{section_title} (first {len(preview)} rows, vertical format):")
    for idx, (_, row) in enumerate(preview.iterrows(), start=1):
        print(f"\nRow {idx}")
        for column in preview.columns:
            value = row[column]
            display_value = "<NA>" if pd.isna(value) else str(value)
            print(f"  {column:<22}: {display_value}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    raw = pd.read_csv(input_path)
    is_valid, errors = validate_telemetry_dataframe(raw)

    print(f"Input file: {input_path}")
    print(f"Input rows: {len(raw)}")
    print(f"Input columns: {len(raw.columns)}")

    if not is_valid:
        print("Validation status: FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    normalized = normalize_telemetry_dataframe(raw)
    print("Validation status: PASSED")
    print(f"Normalized rows: {len(normalized)}")
    print(f"Normalized columns: {len(normalized.columns)}")

    if args.preview_rows > 0:
        if args.preview_format == "table":
            print(f"\nPreview (first {min(args.preview_rows, len(normalized))} rows, table format):")
            print(normalized.head(args.preview_rows).to_string(index=False))
        else:
            _print_vertical_rows(normalized, args.preview_rows, section_title="Preview")

    if args.show_diff_summary:
        _print_diff_summary(
            raw,
            normalized,
            preview_rows=args.diff_preview_rows,
            preview_format=args.preview_format,
        )

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        normalized.to_csv(output_path, index=False)
        print(f"\nWrote normalized CSV to: {output_path}")


if __name__ == "__main__":
    main()
