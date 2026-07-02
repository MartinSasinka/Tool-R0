#!/usr/bin/env python3
"""Compute per-row trajectory metrics and checkpoint summaries."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vizualisation.scripts.lib.aggregation import (  # noqa: E402
    checkpoint_summary_table,
    compute_stability_metrics,
)
from vizualisation.scripts.lib.io_utils import (  # noqa: E402
    COMPONENT_LABELS,
    FEATURE_COLUMNS,
    checkpoints_order,
    load_config,
    load_jsonl_list,
    log,
    run_dir_from_config,
    write_jsonl,
)
from vizualisation.scripts.lib.rewards_bridge import rewards_using_fallback  # noqa: E402
from vizualisation.scripts.lib.trajectory_metrics import compute_row_metrics  # noqa: E402

SCORE_COMPONENTS = [c for c in FEATURE_COLUMNS if c.endswith("_score") or c in (
    "valid_json",
    "exact_trajectory_match",
    "final_answer_exact_match",
)]


def compute_gain_summary(sample_summary: pd.DataFrame, order: list) -> pd.DataFrame:
    model_cps = [c for c in order if c != "gold"]
    rows = []
    for i in range(1, len(model_cps)):
        prev, cur = model_cps[i - 1], model_cps[i]
        prev_row = sample_summary[sample_summary["checkpoint"] == prev]
        cur_row = sample_summary[sample_summary["checkpoint"] == cur]
        if prev_row.empty or cur_row.empty:
            continue
        prev_row = prev_row.iloc[0]
        cur_row = cur_row.iloc[0]
        for col in SCORE_COMPONENTS:
            if col in prev_row and col in cur_row:
                rows.append(
                    {
                        "transition": f"{cur} - {prev}",
                        "component": col,
                        "gain": float(cur_row[col]) - float(prev_row[col]),
                        "aggregation_level": "sample_level_primary",
                    }
                )
    return pd.DataFrame(rows)


def compute_error_transitions(metrics_df: pd.DataFrame, order: list, aggregation: str) -> pd.DataFrame:
    from vizualisation.scripts.lib.aggregation import aggregate_by_sample

    model_cps = [c for c in order if c != "gold"]
    records = []

    # By rollout_idx (approximate pairing)
    for i in range(len(model_cps) - 1):
        a, b = model_cps[i], model_cps[i + 1]
        left = metrics_df[metrics_df["checkpoint"] == a]
        right = metrics_df[metrics_df["checkpoint"] == b]
        merged = left.merge(
            right,
            on=["sample_id", "rollout_idx"],
            suffixes=("_from", "_to"),
        )
        for _, row in merged.iterrows():
            records.append(
                {
                    "transition": f"{a} -> {b}",
                    "pairing": "rollout_idx",
                    "from_error": row["error_type_from"],
                    "to_error": row["error_type_to"],
                    "count": 1,
                }
            )

    # Sample-level via aggregation policy
    agg_from = aggregate_by_sample(metrics_df[metrics_df["checkpoint"].isin(model_cps)], aggregation)
    for i in range(len(model_cps) - 1):
        a, b = model_cps[i], model_cps[i + 1]
        left = agg_from[agg_from["checkpoint"] == a][["sample_id", "error_type"]]
        right = agg_from[agg_from["checkpoint"] == b][["sample_id", "error_type"]]
        merged = left.merge(right, on="sample_id", suffixes=("_from", "_to"))
        for _, row in merged.iterrows():
            records.append(
                {
                    "transition": f"{a} -> {b}",
                    "pairing": f"sample_{aggregation}",
                    "from_error": row["error_type_from"],
                    "to_error": row["error_type_to"],
                    "count": 1,
                }
            )

    if not records:
        return pd.DataFrame(
            columns=["transition", "pairing", "from_error", "to_error", "count"]
        )
    df = pd.DataFrame(records)
    return (
        df.groupby(["transition", "pairing", "from_error", "to_error"], as_index=False)["count"]
        .sum()
        .sort_values(["transition", "count"], ascending=[True, False])
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run_dir", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else run_dir_from_config(cfg)
    canon_path = run_dir / "trajectories_canonical.jsonl"
    if not canon_path.is_file():
        log("metrics", f"ERROR: missing {canon_path}")
        return 2

    if rewards_using_fallback():
        log("metrics", "WARNING: using fallback reward helpers (import from rewards_nestful failed)")

    order = checkpoints_order(cfg) + ["gold"]
    canonical = load_jsonl_list(canon_path)
    metric_rows = [compute_row_metrics(r) for r in canonical]
    metrics_path = run_dir / "trajectory_metrics.jsonl"
    write_jsonl(metrics_path, metric_rows)
    log("metrics", f"wrote {len(metric_rows)} rows -> {metrics_path}")

    df = pd.DataFrame(metric_rows)
    model_df = df[df["checkpoint"] != "gold"].copy()

    rollout_summary = checkpoint_summary_table(model_df, order, SCORE_COMPONENTS, level="rollout_level")
    sample_summaries = []
    for policy in ("all_rollouts", "best_score", "rollout_idx_0", "mean_over_rollouts"):
        if policy == "all_rollouts":
            sub = model_df
        else:
            from vizualisation.scripts.lib.aggregation import aggregate_by_sample

            sub = aggregate_by_sample(model_df, policy)
        tbl = checkpoint_summary_table(sub, order, SCORE_COMPONENTS, level=f"sample_{policy}")
        tbl["summary_policy"] = policy
        sample_summaries.append(tbl)

    summary = pd.concat([rollout_summary] + sample_summaries, ignore_index=True)
    summary_path = run_dir / "checkpoint_summary.csv"
    summary.to_csv(summary_path, index=False)
    log("metrics", f"wrote {summary_path}")

    primary = summary[summary.get("summary_policy", pd.Series(dtype=str)) == "best_score"]
    if primary.empty:
        primary = rollout_summary

    gain = compute_gain_summary(primary, order)
    gain_path = run_dir / "gain_summary.csv"
    gain.to_csv(gain_path, index=False)

    stability = compute_stability_metrics(model_df)
    stability_path = run_dir / "stability_metrics.csv"
    stability.to_csv(stability_path, index=False)

    agg_policy = cfg.get("summary_aggregation", "best_score")
    err_trans = compute_error_transitions(df, order, agg_policy)
    err_path = run_dir / "error_transitions.csv"
    err_trans.to_csv(err_path, index=False)
    log("metrics", f"wrote gain, stability, error transitions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
