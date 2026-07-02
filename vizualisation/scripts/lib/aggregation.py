"""Feature scaling and aggregation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from vizualisation.scripts.lib.io_utils import FEATURE_COLUMNS, log


def fit_scaler(df: pd.DataFrame, feature_cols: List[str], random_seed: int = 42) -> Tuple[StandardScaler, Dict[str, Any]]:
    del random_seed
    X = df[feature_cols].astype(float).values
    scaler = StandardScaler()
    scaler.fit(X)
    meta = {
        "feature_columns": feature_cols,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_samples_fit": int(len(df)),
    }
    return scaler, meta


def transform_features(df: pd.DataFrame, scaler: StandardScaler, feature_cols: List[str]) -> np.ndarray:
    return scaler.transform(df[feature_cols].astype(float).values)


def save_scaler_metadata(path: Path, scaler_meta: Dict[str, Any], reducer_meta: Optional[Dict] = None) -> None:
    payload = {"scaler": scaler_meta}
    if reducer_meta:
        payload["reducer"] = reducer_meta
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def aggregate_by_sample(
    df: pd.DataFrame,
    policy: str,
    score_col: str = "score",
) -> pd.DataFrame:
    """Pick one rollout per (sample_id, checkpoint) according to policy."""
    col = score_col if score_col in df.columns else (
        "final_answer_score" if "final_answer_score" in df.columns else score_col
    )
    rows = []
    for (sid, cp), grp in df.groupby(["sample_id", "checkpoint"], sort=False):
        if policy == "best_score" and col in grp.columns:
            pick = grp.loc[grp[col].astype(float).fillna(-1).idxmax()]
        elif policy == "rollout_idx_0":
            sub = grp[grp["rollout_idx"] == 0]
            pick = sub.iloc[0] if len(sub) else grp.iloc[0]
        elif policy == "pass_first":
            passed = grp[(grp.get("status") == "completed") | (grp.get("verdict") == "pass")]
            if len(passed):
                pick = passed.sort_values("rollout_idx").iloc[0]
            elif col in grp.columns:
                pick = grp.loc[grp[col].astype(float).fillna(-1).idxmax()]
            else:
                pick = grp.iloc[0]
        elif policy == "mean_over_rollouts":
            pick = grp.sort_values("rollout_idx").iloc[0].copy()
            numeric = grp.select_dtypes(include=[np.number])
            for ncol in numeric.columns:
                if ncol in ("rollout_idx",):
                    continue
                pick[ncol] = numeric[ncol].mean()
        else:
            pick = grp.iloc[0]
        rows.append(pick)
    return pd.DataFrame(rows)


def checkpoint_summary_table(
    df: pd.DataFrame,
    checkpoints: List[str],
    component_cols: List[str],
    *,
    level: str,
) -> pd.DataFrame:
    records = []
    for cp in checkpoints:
        sub = df[df["checkpoint"] == cp]
        if sub.empty:
            continue
        rec = {"checkpoint": cp, "aggregation_level": level, "n_rows": len(sub)}
        if "sample_id" in sub.columns:
            rec["n_samples"] = sub["sample_id"].nunique()
        for col in component_cols:
            if col in sub.columns:
                rec[col] = float(sub[col].mean())
        records.append(rec)
    return pd.DataFrame(records)


def compute_stability_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-checkpoint rollout variance and within-sample dispersion."""
    records = []
    for cp, cp_df in df.groupby("checkpoint"):
        if cp == "gold":
            continue
        score_vars = []
        feature_disp = []
        same_error_pct = []
        component_cols = [
            c for c in FEATURE_COLUMNS
            if c in cp_df.columns and (c.endswith("_score") or c in (
                "valid_json", "exact_trajectory_match", "final_answer_exact_match", "trajectory_edit_distance"
            ))
        ]
        for sid, grp in cp_df.groupby("sample_id"):
            if "score" in grp.columns:
                scores = grp["score"].astype(float)
                if len(scores) > 1:
                    score_vars.append(float(scores.var()))
            if len(grp) > 1 and component_cols:
                vals = grp[component_cols].astype(float)
                feature_disp.append(float(vals.var(axis=0).mean()))
            if len(grp) > 1 and "error_type" in grp.columns:
                same_error_pct.append(float(grp["error_type"].nunique() == 1))
        records.append(
            {
                "checkpoint": cp,
                "mean_rollout_score_variance": float(np.mean(score_vars)) if score_vars else 0.0,
                "mean_within_sample_feature_dispersion": float(np.mean(feature_disp)) if feature_disp else 0.0,
                "pct_samples_uniform_error_type": float(np.mean(same_error_pct) * 100) if same_error_pct else 100.0,
            }
        )
    return pd.DataFrame(records)
