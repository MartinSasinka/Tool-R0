"""Print the Pilot4.2 vs Pilot4.3 comparison table as a readable column view."""
from __future__ import annotations

import argparse
import csv
import pathlib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    args = ap.parse_args()
    path = pathlib.Path(args.out_dir) / "PILOT42_VS_PILOT43_METRICS.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            print(f"{row['metric']:<38} {row['pilot42']:>12} -> "
                  f"{row['pilot43']:>12}   {row['wanted_direction']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
