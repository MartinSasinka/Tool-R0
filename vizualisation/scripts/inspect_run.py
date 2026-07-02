#!/usr/bin/env python3
"""Quick inspection of a completed analysis run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vizualisation.scripts.lib.io_utils import load_jsonl_list, log  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        log("inspect", f"ERROR: not a directory: {run_dir}")
        return 2

    cfg_path = run_dir / "config.json"
    run_name = run_dir.name
    if cfg_path.is_file():
        import json

        with open(cfg_path, encoding="utf-8") as fh:
            run_name = json.load(fh).get("run_name", run_name)

    print(f"Run: {run_name}")
    print(f"Directory: {run_dir}")

    raw_path = run_dir / "trajectories_raw.jsonl"
    if raw_path.is_file():
        raw = load_jsonl_list(raw_path)
        cps = sorted({r.get("checkpoint") for r in raw if r.get("checkpoint")})
        print(f"Checkpoints ({len(cps)}): {', '.join(cps)}")
        for cp in cps:
            n = sum(1 for r in raw if r.get("checkpoint") == cp)
            print(f"  {cp}: {n} rows")

    figures = list((run_dir / "figures").glob("*.*"))
    print(f"Figures: {len(figures)}")
    for f in sorted(figures)[:10]:
        print(f"  {f.name}")
    if len(figures) > 10:
        print(f"  ... and {len(figures) - 10} more")

    summary_path = run_dir / "checkpoint_summary.csv"
    if summary_path.is_file():
        import pandas as pd

        df = pd.read_csv(summary_path)
        primary = df[df.get("summary_policy", pd.Series(dtype=str)) == "best_score"]
        if not primary.empty:
            print("\nPrimary checkpoint summary (best_score):")
            cols = ["checkpoint", "final_answer_score", "reference_score", "format_score"]
            cols = [c for c in cols if c in primary.columns]
            print(primary[cols].to_string(index=False))

    metrics_path = run_dir / "trajectory_metrics.jsonl"
    if metrics_path.is_file():
        import pandas as pd

        m = pd.DataFrame(load_jsonl_list(metrics_path))
        model = m[m["checkpoint"] != "gold"]
        print("\nTop error types:")
        print(model["error_type"].value_counts().head(8).to_string())

    warnings = []
    if not (run_dir / "feature_matrix.csv").is_file():
        warnings.append("missing feature_matrix.csv")
    if not (run_dir / "reports" / "analysis_report.md").is_file():
        warnings.append("missing analysis_report.md")
    if warnings:
        print("\nWarnings:", ", ".join(warnings))
    else:
        print("\nStatus: run artifacts look complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
