"""Command-line entry point for LID-DS 2021 feature extraction."""
import argparse
from .lidds import export_windows_csv


def main():
    parser = argparse.ArgumentParser(description="Stream LID-DS nested ZIP recordings into window features.")
    parser.add_argument("--archive", required=True, help="Path to CVE-2018-3760.zip")
    parser.add_argument("--output", default="results/lidds_windows.csv")
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--splits", nargs="+", choices=["training", "validation", "test"],
                        default=["training", "validation", "test"])
    args = parser.parse_args()
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    count = export_windows_csv(args.archive, args.output,
                               window_seconds=args.window_seconds,
                               splits=set(args.splits))
    print(f"Wrote {count} windows to {args.output}")
    print("Labels: benign metadata recordings=0; attack windows=1 only when the full window starts at/after exploit marker; boundary windows omitted.")
    print("Use training for fitting, validation for threshold/model selection, and test once for final evaluation.")


if __name__ == "__main__":
    main()
