#!/usr/bin/env python3
"""Generate Markdown report and JSON summary for an analysis run."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vizualisation.scripts.lib.io_utils import (  # noqa: E402
    FEATURE_COLUMNS,
    checkpoints_order,
    load_config,
    load_jsonl_list,
    log,
    run_dir_from_config,
)


CAVEATS = """
These visualizations analyze **behavioral tool-use trajectories**, not direct internal model knowledge.

**2D projections are exploratory**; quantitative claims are based on component metrics and distances in the original standardized feature space.

Observed shifts should be interpreted as changes in model outputs on a fixed evaluation set.
""".strip()


def compute_shift_alignment(run_dir: Path, order: list) -> pd.DataFrame:
    std_path = run_dir / "feature_matrix_standardized.csv"
    if not std_path.is_file():
        return pd.DataFrame()
    std = pd.read_csv(std_path)
    feat_cols = [c for c in FEATURE_COLUMNS if c in std.columns]
    gold = std[std["checkpoint"] == "gold"].set_index("sample_id")
    model_cps = [c for c in order if c != "gold"]
    rows = []
    for i in range(len(model_cps) - 1):
        a, b = model_cps[i], model_cps[i + 1]
        left = std[std["checkpoint"] == a]
        right = std[std["checkpoint"] == b]
        merged = left.merge(right, on=["sample_id", "rollout_idx"], suffixes=("_a", "_b"))
        alignments = []
        for _, row in merged.iterrows():
            sid = row["sample_id"]
            if sid not in gold.index:
                continue
            va = row[[f"{c}_a" for c in feat_cols]].astype(float).values
            vb = row[[f"{c}_b" for c in feat_cols]].astype(float).values
            vg = gold.loc[sid, feat_cols].astype(float).values
            shift = vb - va
            toward = vg - va
            n_shift = np.linalg.norm(shift)
            n_toward = np.linalg.norm(toward)
            if n_shift < 1e-9 or n_toward < 1e-9:
                continue
            alignments.append(float(np.dot(shift, toward) / (n_shift * n_toward)))
        rows.append(
            {
                "transition": f"{a} -> {b}",
                "mean_cosine_alignment_toward_gold": float(np.mean(alignments)) if alignments else np.nan,
                "n_pairs": len(alignments),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(run_dir / "shift_alignment.csv", index=False)
    return df


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run_dir", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else run_dir_from_config(cfg)
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    order = checkpoints_order(cfg)
    raw = load_jsonl_list(run_dir / "trajectories_raw.jsonl")
    metrics = pd.DataFrame(load_jsonl_list(run_dir / "trajectory_metrics.jsonl"))
    summary = pd.read_csv(run_dir / "checkpoint_summary.csv") if (run_dir / "checkpoint_summary.csv").is_file() else pd.DataFrame()
    stability = pd.read_csv(run_dir / "stability_metrics.csv") if (run_dir / "stability_metrics.csv").is_file() else pd.DataFrame()
    dist = pd.read_csv(run_dir / "distance_to_gold.csv") if (run_dir / "distance_to_gold.csv").is_file() else pd.DataFrame()
    alignment = compute_shift_alignment(run_dir, order + ["gold"])

    malformed = sum(1 for r in raw if "missing_prediction_output" in (r.get("parse_flags") or []))
    figures = sorted(str(p.relative_to(run_dir)) for p in (run_dir / "figures").glob("*.*"))

    primary = summary[summary.get("summary_policy", pd.Series(dtype=str)) == "best_score"]

    lines = [
        "# Trajectory analysis report",
        "",
        f"**Run:** `{cfg.get('run_name')}`  ",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Methodological caveats",
        "",
        CAVEATS,
        "",
        "## Run metadata",
        "",
        f"- Config: `{cfg.get('_config_path', 'config.json')}`",
        f"- Output: `{run_dir}`",
        f"- Checkpoints: {', '.join(order)}",
        f"- Rollout policy (raw dataset): `{cfg.get('rollout_policy', 'all_rollouts')}`",
        f"- Primary summary aggregation: `{cfg.get('summary_aggregation', 'best_score')}`",
        "",
        "## Sample counts",
        "",
    ]
    model_raw = [r for r in raw if r.get("checkpoint") != "gold"]
    for cp in order:
        n = sum(1 for r in model_raw if r.get("checkpoint") == cp)
        lines.append(f"- `{cp}`: {n} rollout rows")
    lines.append(f"- Malformed / missing prediction_output flags: {malformed}")
    lines.append("")

    def _md_table(frame: pd.DataFrame) -> str:
        try:
            return frame.to_markdown(index=False)
        except Exception:
            return "```\n" + frame.to_string(index=False) + "\n```"

    lines += ["## Quantitative component metrics (standardized feature space separate)", ""]
    lines.append("### Primary checkpoint summary (sample-level, best_score)")
    if not primary.empty:
        lines.append(_md_table(primary))
    else:
        lines.append("_No primary summary available._")
    lines.append("")
    lines.append("### All aggregation policies")
    if not summary.empty:
        lines.append(_md_table(summary))
    lines.append("")

    lines += ["## Distance and alignment metrics (original standardized feature space)", ""]
    if not dist.empty:
        overall = dist[dist["num_calls_gold"] == "all"] if "num_calls_gold" in dist.columns else dist
        lines.append(_md_table(overall))
    if not alignment.empty:
        lines.append("")
        lines.append("### Shift alignment toward gold (cosine)")
        lines.append(_md_table(alignment))
    lines.append("")

    lines += ["## Rollout stability", ""]
    if not stability.empty:
        lines.append(_md_table(stability))
    lines.append("")

    lines += ["## Exploratory 2D projections", ""]
    lines.append("See figures below. Do not treat 2D layout as ground-truth geometry.")
    lines.append("")
    lines += ["## Generated figures", ""]
    for fig in figures:
        lines.append(f"- `{fig}`")
    lines.append("")

    if not metrics.empty:
        err = metrics[metrics["checkpoint"] != "gold"]["error_type"].value_counts().head(10)
        lines += ["## Top error types (all rollout rows)", "", _md_table(err.reset_index(name="count")), ""]

    report_path = reports_dir / "analysis_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    summary_json = {
        "run_name": cfg.get("run_name"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoints": order,
        "malformed_count": malformed,
        "figures": figures,
        "caveats": CAVEATS,
    }
    with open(reports_dir / "analysis_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary_json, fh, indent=2, ensure_ascii=False)

    log("report", f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
