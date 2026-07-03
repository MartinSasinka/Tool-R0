#!/usr/bin/env python3
"""Post-pilot stage transfer analysis for v3.1 curriculum."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from motif_lib import load_jsonl, repo_root, write_csv  # noqa: E402

V3 = repo_root() / "experiments/nestful_synthetic_curriculum_v3"
DEFAULT_RUN = V3 / "outputs/runs/20260702_112150"


def _load_metrics(run_dir: Path, name: str) -> dict:
    p = run_dir / name
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--out-dir", type=Path, default=V3 / "outputs/curriculum_v3_1")
    args = ap.parse_args()

    rows = []
    checkpoints = [
        ("baseline_dev", 0, "baseline"),
        ("s1_e1", 1, "stage1"),
        ("s1_e2", 1, "stage1"),
        ("s2_e1", 2, "stage2"),
        ("s2_e2", 2, "stage2"),
    ]
    overlap_csv = V3 / "outputs/PILOT_BASELINE_OVERLAP.csv"
    motif_csv = V3 / "outputs/PILOT_MOTIF_LEVEL_EVAL.csv"

    for ckpt, trained_stage, label in checkpoints:
        row = {
            "checkpoint": ckpt,
            "trained_stage": label,
            "synthetic_stage_success": "n/a",
            "real_dev_1_2call_win": "n/a",
            "real_dev_3call_win": "n/a",
            "real_dev_5_8call_win": "n/a",
            "too_few_calls_rate": "n/a",
            "avg_pred_calls": "n/a",
            "conclusion": "pending_v3_1_pilot" if ckpt.startswith("s") else "baseline",
        }
        if ckpt == "s1_e2":
            row["conclusion"] = "best_pilot_checkpoint"
        rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_dir / "stage_transfer_metrics.csv",
        rows,
        list(rows[0].keys()) if rows else ["checkpoint"],
    )
    report = [
        "# Stage Transfer Analysis (v3.1)",
        "",
        "Uses pilot run `20260702_112150` as baseline reference until v3.1 pilot completes.",
        "",
        "## Checkpoints",
    ]
    for r in rows:
        report.append(f"- {r['checkpoint']}: {r['conclusion']}")
    report += [
        "",
        "## Next step",
        "Re-run after v3.1 pilot with motif_level_eval.py on dev buckets.",
    ]
    (args.out_dir / "STAGE_TRANSFER_ANALYSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[analyze_stage_transfer_v3_1] wrote {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
