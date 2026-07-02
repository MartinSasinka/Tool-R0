#!/usr/bin/env python3
"""Reduce standardized feature space to 2D (PCA or UMAP)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vizualisation.scripts.lib.aggregation import fit_scaler, save_scaler_metadata, transform_features  # noqa: E402
from vizualisation.scripts.lib.io_utils import FEATURE_COLUMNS, load_config, log, run_dir_from_config  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run_dir", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else run_dir_from_config(cfg)
    feat_path = run_dir / "feature_matrix.csv"
    if not feat_path.is_file():
        log("reduce", f"ERROR: missing {feat_path}")
        return 2

    dr_cfg = cfg.get("dimensionality_reduction") or {}
    method = dr_cfg.get("method", "pca")
    seed = int(dr_cfg.get("random_seed", 42))

    df = pd.read_csv(feat_path)
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    scaler, scaler_meta = fit_scaler(df, feature_cols, seed)
    X = transform_features(df, scaler, feature_cols)

    reducer_meta = {"method": method, "random_seed": seed, "feature_columns": feature_cols}
    if method == "umap":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_components=2, random_state=seed)
            coords = reducer.fit_transform(X)
            reducer_meta["method_used"] = "umap"
        except ImportError:
            log("reduce", "WARNING: umap-learn not installed; falling back to PCA")
            method = "pca"

    if method == "pca":
        pca = PCA(n_components=2, random_state=seed)
        coords = pca.fit_transform(X)
        reducer_meta["method_used"] = "pca"
        reducer_meta["explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
        reducer_meta["explained_variance_total"] = float(pca.explained_variance_ratio_.sum())

    embed = df[["sample_id", "checkpoint", "rollout_idx", "error_type"]].copy()
    for col in ("final_answer_score", "reference_score", "dependency_depth_score", "valid_json"):
        if col in df.columns:
            embed[col] = df[col]
    embed["x"] = coords[:, 0]
    embed["y"] = coords[:, 1]

    embed_path = run_dir / "embedding_2d.csv"
    embed.to_csv(embed_path, index=False)

    # Standardized features for downstream distance/centroid
    std_df = df[["sample_id", "checkpoint", "rollout_idx"]].copy()
    for i, col in enumerate(feature_cols):
        std_df[col] = X[:, i]
    std_path = run_dir / "feature_matrix_standardized.csv"
    std_df.to_csv(std_path, index=False)

    meta_path = run_dir / "reducer_metadata.json"
    save_scaler_metadata(meta_path, scaler_meta, reducer_meta)
    log("reduce", f"wrote {embed_path} and scaler metadata -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
