"""Export generated sample telemetry to CSV for demo/testing workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from sailing_telemetry.generator import generate_sample_telemetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sample sailing telemetry and save as CSV.",
    )
    parser.add_argument(
        "--output",
        default="sample_data/sailing_telemetry_sample.csv",
        help="Output CSV path",
    )
    parser.add_argument("--boats", type=int, default=5, help="Number of boats")
    parser.add_argument("--legs", type=int, default=6, help="Legs per boat")
    parser.add_argument(
        "--points-per-leg",
        type=int,
        default=120,
        dest="points_per_leg",
        help="Telemetry samples per leg",
    )
    parser.add_argument(
        "--freq-seconds",
        type=int,
        default=2,
        dest="freq_seconds",
        help="Sampling interval in seconds",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    telemetry = generate_sample_telemetry(
        num_boats=args.boats,
        legs_per_boat=args.legs,
        samples_per_leg=args.points_per_leg,
        sampling_interval_seconds=args.freq_seconds,
        seed=args.seed,
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry.to_csv(output_path, index=False)

    print(f"Wrote {len(telemetry)} rows to {output_path}")


if __name__ == "__main__":
    main()
