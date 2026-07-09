"""Unified metrics schema (v1) for NESTFUL eval cells.

Builds ``metrics_unified.json`` from the two files every final_eval cell already
produces (``metrics.json`` internal diagnostics, ``metrics_official.json`` official
scorer) plus decoding/dataset context. Additive only — never modifies existing files.

Naming rules (METRIC_STANDARD_PROPOSAL.md):
- the primary, headline-eligible number is ``primary.official_nestful_win_rate``;
- the internal executed-trajectory win is exported ONLY as
  ``diagnostics.internal_final_answer_win`` — a bare ``win_rate`` key never appears
  at the top level, so the inflated internal number (audited +6-7 pp vs official)
  cannot masquerade as the paper metric.

Per-sample official wins (for paired gained/regressed counts) are read from
``final_eval_trajectories.jsonl`` rows: ``_traj.official_win`` keyed by ``sample_id``.

Usage:
  python metrics_schema.py --cell-dir <dir> [--cell NAME ...] [--out <file>]
  python metrics_schema.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, Optional

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def binomial_ci95(p: float, n: int) -> Optional[list]:
    """Normal-approximation 95% CI for a proportion; None if n == 0."""
    if n <= 0:
        return None
    half = 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / n)
    return [round(max(0.0, p - half), 4), round(min(1.0, p + half), 4)]


def load_per_sample_official_wins(trajectories_path: str) -> Dict[str, float]:
    """sample_id -> official_win (0.0/1.0) from a final_eval_trajectories.jsonl."""
    wins: Dict[str, float] = {}
    with open(trajectories_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            traj = row.get("_traj") or {}
            sid = row.get("sample_id") or traj.get("task_id")
            w = traj.get("official_win")
            if sid is not None and w is not None:
                wins[str(sid)] = float(w)
    return wins


def paired_counts(cell_wins: Dict[str, float], baseline_wins: Dict[str, float]) -> Dict[str, Any]:
    """Paired gained/regressed vs baseline over the shared sample-id set."""
    shared = sorted(set(cell_wins) & set(baseline_wins))
    gained = sum(1 for s in shared if cell_wins[s] > baseline_wins[s])
    regressed = sum(1 for s in shared if cell_wins[s] < baseline_wins[s])
    return {
        "n_shared": len(shared),
        "gained": gained,
        "regressed": regressed,
        "net": gained - regressed,
    }


def build_unified_metrics(
    *,
    cell: str,
    checkpoint: Optional[str],
    dataset: Dict[str, Any],
    decoding: Dict[str, Any],
    internal_metrics: Dict[str, Any],
    official_metrics: Dict[str, Any],
    paired_vs_baseline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the unified metrics dict (schema v1).

    ``internal_metrics`` = parsed metrics.json; ``official_metrics`` = parsed
    metrics_official.json (the official scorer's ``win_rate`` key IS the official
    NESTFUL win — that file is scorer output, not the internal replica).
    """
    ours = internal_metrics.get("our_metrics", {})
    idiag = internal_metrics.get("internal_metrics_diagnostic", {})
    n = int(official_metrics.get("num_examples") or internal_metrics.get("num_tasks") or 0)
    official_win = official_metrics.get("win_rate")

    diagnostics = {
        # internal executed-trajectory win — DIAGNOSTIC ONLY (inflates official ~6-7 pp)
        "internal_final_answer_win": idiag.get("win_rate"),
        "final_answer_pass": ours.get("final_answer_pass"),
        "solution_equivalent_pass": ours.get("solution_equivalent_pass"),
        "strict_gold_trace_pass": ours.get("strict_gold_trace_pass"),
        "alternative_valid_solution_pass": ours.get("alternative_valid_solution_pass"),
        "correct_answer_but_unsupported_trace": ours.get("correct_answer_but_unsupported_trace"),
        "internal_f1_func": idiag.get("f1_func"),
        "internal_f1_param": idiag.get("f1_param"),
        "internal_partial_sequence_accuracy": idiag.get("partial_sequence_accuracy"),
        "internal_full_sequence_accuracy": idiag.get("full_sequence_accuracy"),
    }
    # call-count behavior when the producer supplies it (recomputed from
    # trajectories by the eval runner; run.py's metrics.json does not aggregate it)
    for k in ("too_few_calls_rate", "avg_predicted_calls", "no_tool_call_rate",
              "parse_error_rate"):
        if k in internal_metrics:
            diagnostics[k] = internal_metrics[k]

    return {
        "schema_version": SCHEMA_VERSION,
        "cell": cell,
        "checkpoint": checkpoint,
        "dataset": dataset,          # {name, path, sha256, n_rows}
        "decoding": decoding,        # {temperature, top_p, max_new_tokens, seed, backend}
        "primary": {
            "official_nestful_win_rate": official_win,
            "n": n,
            "ci95": binomial_ci95(float(official_win), n) if official_win is not None else None,
        },
        "official": {
            "f1_func": official_metrics.get("f1_func"),
            "f1_param": official_metrics.get("f1_param"),
            "partial_sequence_accuracy": official_metrics.get("partial_sequence_accuracy"),
            "full_sequence_accuracy": official_metrics.get("full_sequence_accuracy"),
            "num_pred_parsing_errors": official_metrics.get("num_pred_parsing_errors"),
            "paradigm": official_metrics.get("paradigm"),
        },
        "diagnostics": diagnostics,
        "paired_vs_baseline": paired_vs_baseline,  # None for the baseline cell
    }


def build_from_cell_dir(
    cell_dir: str,
    *,
    cell: Optional[str] = None,
    checkpoint: Optional[str] = None,
    dataset: Optional[Dict[str, Any]] = None,
    decoding: Optional[Dict[str, Any]] = None,
    baseline_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Build unified metrics from an existing on-disk eval cell directory."""
    def _load(name: str) -> Dict[str, Any]:
        p = os.path.join(cell_dir, name)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"required file missing in cell dir: {p}")
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    internal = _load("metrics.json")
    official = _load("metrics_official.json")

    paired = None
    if baseline_dir and os.path.abspath(baseline_dir) != os.path.abspath(cell_dir):
        cw = load_per_sample_official_wins(os.path.join(cell_dir, "final_eval_trajectories.jsonl"))
        bw = load_per_sample_official_wins(os.path.join(baseline_dir, "final_eval_trajectories.jsonl"))
        paired = paired_counts(cw, bw)

    return build_unified_metrics(
        cell=cell or os.path.basename(os.path.normpath(cell_dir)),
        checkpoint=checkpoint,
        dataset=dataset or {"name": None, "path": None, "sha256": None, "n_rows": None},
        decoding=decoding or {"temperature": None, "top_p": None,
                              "max_new_tokens": None, "seed": None},
        internal_metrics=internal,
        official_metrics=official,
        paired_vs_baseline=paired,
    )


# ---------------------------------------------------------------------------
# self-test on synthetic inputs (no repo files needed)
# ---------------------------------------------------------------------------

def _self_test() -> int:
    internal = {
        "num_tasks": 100,
        "internal_metrics_diagnostic": {"win_rate": 0.61, "f1_func": 0.9, "f1_param": 0.4,
                                        "partial_sequence_accuracy": 0.2,
                                        "full_sequence_accuracy": 0.02},
        "our_metrics": {"final_answer_pass": 0.6, "solution_equivalent_pass": 0.55,
                        "strict_gold_trace_pass": 0.2,
                        "alternative_valid_solution_pass": 0.5,
                        "correct_answer_but_unsupported_trace": 0.05},
        "too_few_calls_rate": 0.42,
    }
    official = {"win_rate": 0.54, "num_examples": 100, "f1_func": 0.88, "f1_param": 0.45,
                "partial_sequence_accuracy": 0.19, "full_sequence_accuracy": 0.02,
                "num_pred_parsing_errors": 0, "paradigm": "react"}
    u = build_unified_metrics(
        cell="s3_e1", checkpoint="ckpt/path",
        dataset={"name": "nestful_test.jsonl", "path": "x", "sha256": "0" * 64, "n_rows": 100},
        decoding={"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 2560, "seed": 42},
        internal_metrics=internal, official_metrics=official,
        paired_vs_baseline=paired_counts({"a": 1.0, "b": 0.0, "c": 1.0},
                                         {"a": 0.0, "b": 0.0, "c": 1.0}),
    )
    assert u["schema_version"] == SCHEMA_VERSION
    assert u["primary"]["official_nestful_win_rate"] == 0.54
    assert u["diagnostics"]["internal_final_answer_win"] == 0.61
    assert "win_rate" not in u and "win_rate" not in u["diagnostics"]
    assert u["primary"]["ci95"] == [0.4423, 0.6377]
    assert u["paired_vs_baseline"] == {"n_shared": 3, "gained": 1, "regressed": 0, "net": 1}
    assert u["diagnostics"]["too_few_calls_rate"] == 0.42
    json.dumps(u)  # must be serializable
    print("[metrics_schema] self-test OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--cell-dir", help="existing eval cell dir with metrics.json + metrics_official.json")
    ap.add_argument("--cell", default=None, help="cell name (default: dir basename)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--baseline-dir", default=None,
                    help="baseline cell dir for paired counts (needs trajectories in both)")
    ap.add_argument("--out", default=None, help="output path (default: <cell-dir>/metrics_unified.json)")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if not args.cell_dir:
        ap.error("--cell-dir is required unless --self-test")

    unified = build_from_cell_dir(args.cell_dir, cell=args.cell,
                                  checkpoint=args.checkpoint,
                                  baseline_dir=args.baseline_dir)
    out = args.out or os.path.join(args.cell_dir, "metrics_unified.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(unified, fh, indent=2, ensure_ascii=False)
    print(f"[metrics_schema] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
