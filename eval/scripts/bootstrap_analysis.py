#!/usr/bin/env python3
"""
Bootstrap resampling analysis for confidence intervals.

Computes mean ± std (and 95% CI) for all benchmarks by resampling
from existing prediction files. Standard practice for ML papers —
no need to re-run expensive inference.

Usage:
    python -m eval.scripts.bootstrap_analysis [--n-bootstrap 1000] [--seed 42]
    python -m eval.scripts.bootstrap_analysis --results-dir eval/results --output eval/results/bootstrap_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval.ast_eval import ast_match, ast_match_single  # noqa: F401 — available for manual re-scoring


def load_predictions(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _bootstrap_metric(
    scores: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> Dict[str, float]:
    """Bootstrap a binary/continuous score array → mean, std, 95% CI."""
    n = len(scores)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = scores[idx].mean()

    point = float(scores.mean())
    std = float(means.std())
    ci_low = float(np.percentile(means, 2.5))
    ci_high = float(np.percentile(means, 97.5))

    return {
        "mean": round(100.0 * point, 2),
        "std": round(100.0 * std, 2),
        "ci95_low": round(100.0 * ci_low, 2),
        "ci95_high": round(100.0 * ci_high, 2),
        "n_samples": n,
    }


# ── Benchmark-specific scoring ──────────────────────────────────

def _use_score_field(preds: List[Dict]) -> np.ndarray:
    """Use the pre-computed 'score' field from prediction records."""
    return np.array([float(p.get("score", 0.0)) for p in preds])


def score_bfcl(preds: List[Dict]) -> np.ndarray:
    return _use_score_field(preds)


def score_bfcl_by_category(preds: List[Dict]) -> Dict[str, np.ndarray]:
    cats: Dict[str, list] = {}
    for p in preds:
        cat = p.get("category", "unknown")
        cats.setdefault(cat, []).append(float(p.get("score", 0.0)))
    return {k: np.array(v) for k, v in cats.items()}


def score_toolalpaca(preds: List[Dict]) -> np.ndarray:
    return _use_score_field(preds)


def score_apibank(preds: List[Dict]) -> np.ndarray:
    return _use_score_field(preds)


def score_tooltalk(preds: List[Dict]) -> np.ndarray:
    return _use_score_field(preds)


SCORERS = {
    "bfcl": score_bfcl,
    "bfcl_exec": score_bfcl,
    "toolalpaca": score_toolalpaca,
    "apibank": score_apibank,
    "tooltalk": score_tooltalk,
    "nestful": score_tooltalk,
}


# ── Main logic ───────────────────────────────────────────────────

def find_prediction_files(results_dir: str) -> List[Tuple[str, str, str]]:
    """Returns list of (benchmark, profile, path)."""
    found = []
    for bench_dir in sorted(os.listdir(results_dir)):
        full = os.path.join(results_dir, bench_dir)
        if not os.path.isdir(full) or bench_dir.endswith("_smoke"):
            continue
        for fname in sorted(os.listdir(full)):
            if fname.endswith("_predictions.jsonl"):
                profile = fname.replace("_predictions.jsonl", "")
                found.append((bench_dir, profile, os.path.join(full, fname)))
    return found


def run_bootstrap(
    results_dir: str = "eval/results",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    files = find_prediction_files(results_dir)

    if not files:
        print(f"No prediction files found in {results_dir}")
        return {}

    all_results: Dict[str, Dict[str, Any]] = {}

    for bench, profile, path in files:
        scorer = SCORERS.get(bench)
        if scorer is None:
            print(f"  [skip] No scorer for {bench}/{profile}")
            continue

        preds = load_predictions(path)
        if not preds:
            continue

        scores = scorer(preds)
        stats = _bootstrap_metric(scores, n_bootstrap, rng)

        key = f"{bench}/{profile}"
        all_results[key] = {
            "benchmark": bench,
            "profile": profile,
            **stats,
        }

        print(
            f"  {bench:15s} / {profile:15s}: "
            f"{stats['mean']:6.2f}% +/- {stats['std']:.2f}%  "
            f"[95% CI: {stats['ci95_low']:.2f} - {stats['ci95_high']:.2f}%]  "
            f"(n={stats['n_samples']})"
        )

        if bench in ("bfcl", "bfcl_exec"):
            cat_scores = score_bfcl_by_category(preds)
            for cat, cat_arr in sorted(cat_scores.items()):
                cat_stats = _bootstrap_metric(cat_arr, n_bootstrap, rng)
                cat_key = f"{bench}/{profile}/{cat}"
                all_results[cat_key] = {
                    "benchmark": bench,
                    "profile": profile,
                    "category": cat,
                    **cat_stats,
                }
                print(
                    f"    -- {cat:20s}: "
                    f"{cat_stats['mean']:6.2f}% +/- {cat_stats['std']:.2f}%  "
                    f"[{cat_stats['ci95_low']:.2f} - {cat_stats['ci95_high']:.2f}%]  "
                    f"(n={cat_stats['n_samples']})"
                )

    return all_results


def format_comparison_table(results: Dict[str, Any]) -> str:
    """Build a markdown table comparing baseline vs finetuned."""
    benchmarks = set()
    for key, data in results.items():
        if "/" in key and "category" not in data:
            benchmarks.add(data["benchmark"])

    lines = [
        "| Benchmark | Baseline | Finetuned | Delta |",
        "|-----------|----------|-----------|-------|",
    ]

    for bench in ["bfcl", "bfcl_exec", "toolalpaca", "apibank", "tooltalk", "nestful"]:
        base_key = f"{bench}/baseline"
        tune_key = f"{bench}/finetuned"
        if bench == "bfcl_exec":
            base_key = f"{bench}/baseline_exec"
            tune_key = f"{bench}/finetuned_exec"

        base = results.get(base_key)
        tune = results.get(tune_key)
        if not base or not tune:
            continue

        delta = tune["mean"] - base["mean"]
        lines.append(
            f"| {bench} | {base['mean']:.2f} +/- {base['std']:.2f} | "
            f"{tune['mean']:.2f} +/- {tune['std']:.2f} | {delta:+.2f} |"
        )

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Bootstrap resampling analysis.")
    p.add_argument("--results-dir", default="eval/results", help="Results directory")
    p.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap iterations")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--output", default=None, help="Output JSON path")
    p.add_argument("--table", default=None, help="Output markdown table path")
    args = p.parse_args()

    print("=" * 70)
    print(f"  Bootstrap Analysis (n={args.n_bootstrap}, seed={args.seed})")
    print("=" * 70)
    print()

    results = run_bootstrap(args.results_dir, args.n_bootstrap, args.seed)

    if not results:
        return

    print()
    table = format_comparison_table(results)
    print(table)
    print()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Full results: {args.output}")

    if args.table:
        os.makedirs(os.path.dirname(args.table) or ".", exist_ok=True)
        with open(args.table, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"Comparison table: {args.table}")


if __name__ == "__main__":
    main()
