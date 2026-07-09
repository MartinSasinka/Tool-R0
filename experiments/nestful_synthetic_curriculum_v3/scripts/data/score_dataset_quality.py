"""Dataset quality scorer for the agentic (or any NESTFUL-like) corpus.

Sections (spec §8): validity, contamination, distribution similarity,
solver gap, GRPO signal (from a stage-probe report when available).

Verdict ladder — a dataset is only ever:
  technically_acceptable : validity + contamination hard gates pass,
                           gold replay pass rate = 1.0;
  training_candidate     : + distribution closer to NESTFUL than v3.1 on most
                           dimensions, positive solver gap, better probe signal;
  actually_useful        : ONLY decided by a same-batch official NESTFUL eval
                           after training — never claimed by this script.

Usage (repo root):
  python .../score_dataset_quality.py                      # agentic corpus
  python .../score_dataset_quality.py --dataset-dir <dir>  # any corpus dir
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(V3_ROOT, "scripts", "lib"))
sys.path.insert(0, V3_ROOT)

from paths import CANONICAL_STAGE_FILES, NESTFUL_DATASETS, REPO_ROOT  # noqa: E402

from lib.agentic_data.contamination import load_nestful_hashes  # noqa: E402
from lib.agentic_data.distribution import (DIMENSIONS, corpus_stats,  # noqa: E402
                                           distance_report, norm_row)
from lib.agentic_data.schema import STAGES  # noqa: E402
from lib.nestful_like_generator import question_hash, replay_task, trace_hash  # noqa: E402

DEFAULT_DATASET = os.path.join(
    V3_ROOT, "data", "curriculum_v4_nestful_like_agentic_openrouter")
DET_V4_FILTERED = os.path.join(V3_ROOT, "data", "curriculum_v4_nestful_like",
                               "filtered")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_corpus(dataset_dir: str) -> List[Dict[str, Any]]:
    filtered = os.path.join(dataset_dir, "filtered")
    root = filtered if os.path.isdir(filtered) else dataset_dir
    rows: List[Dict[str, Any]] = []
    for f in sorted(os.listdir(root)):
        if f.endswith(".jsonl"):
            rows.extend(_load_jsonl(os.path.join(root, f)))
    return rows


# ---------------------------------------------------------------- sections
def validity_section(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    replay_pass = schema_pass = 0
    null_answers = unresolved = 0
    seen_q: Counter = Counter()
    seen_t: Counter = Counter()
    seen_id: Counter = Counter()
    replayable = 0
    for row in rows:
        r = norm_row(row)
        required_ok = all(row.get(k) is not None for k in
                          ("sample_id", "question", "gold_calls", "num_calls",
                           "stage"))
        stage_ok = True
        stage = row.get("stage")
        if stage in STAGES:
            lo, hi = STAGES[stage]
            stage_ok = lo <= len(r["gold_calls"]) <= hi
        schema_pass += int(required_ok and stage_ok)
        if row.get("gold_answer") is None:
            null_answers += 1
        elif isinstance(row["gold_answer"], str) and "$" in row["gold_answer"]:
            unresolved += 1
        seen_q[question_hash(r["question"])] += 1
        try:
            seen_t[trace_hash(r["gold_calls"])] += 1
        except (KeyError, TypeError):
            pass
        seen_id[r["sample_id"]] += 1
        # replay only rows in our executable-registry format
        if row.get("observations") is not None and row.get("gold_answer") is not None:
            replayable += 1
            ok, _ = replay_task(row)
            replay_pass += int(ok)
    dup_q = sum(c - 1 for c in seen_q.values() if c > 1)
    dup_t = sum(c - 1 for c in seen_t.values() if c > 1)
    dup_id = sum(c - 1 for c in seen_id.values() if c > 1)
    replay_rate = replay_pass / replayable if replayable else 0.0
    out = {
        "n_rows": n,
        "gold_replay_pass_rate": round(replay_rate, 4),
        "schema_pass_rate": round(schema_pass / n, 4) if n else 0.0,
        "null_answer_rate": round(null_answers / n, 4) if n else 0.0,
        "unresolved_var_rate": round(unresolved / n, 4) if n else 0.0,
        "duplicate_question_rate": round(dup_q / n, 4) if n else 0.0,
        "duplicate_trace_rate": round(dup_t / n, 4) if n else 0.0,
        "duplicate_sample_id_rate": round(dup_id / n, 4) if n else 0.0,
    }
    out["hard_gates_pass"] = bool(
        n > 0 and out["gold_replay_pass_rate"] == 1.0
        and out["schema_pass_rate"] == 1.0 and null_answers == 0
        and unresolved == 0 and dup_q == 0 and dup_t == 0 and dup_id == 0)
    return out


def contamination_section(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    nest_q, nest_t, nest_ids = load_nestful_hashes()
    q_hits = t_hits = id_hits = 0
    for row in rows:
        r = norm_row(row)
        if question_hash(r["question"]) in nest_q:
            q_hits += 1
        try:
            if trace_hash(r["gold_calls"]) in nest_t:
                t_hits += 1
        except (KeyError, TypeError):
            pass
        if r["sample_id"] in nest_ids:
            id_hits += 1
    return {
        "question_hash_overlap": q_hits,
        "trace_hash_overlap": t_hits,
        "sample_id_overlap": id_hits,
        "hard_gates_pass": q_hits == 0 and t_hits == 0 and id_hits == 0,
    }


def distribution_section(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats_by = {"candidate": corpus_stats(rows)}
    v31: List[Dict[str, Any]] = []
    for p in CANONICAL_STAGE_FILES.values():
        if os.path.isfile(p):
            v31.extend(_load_jsonl(p))
    if v31:
        stats_by["v3_1"] = corpus_stats(v31)
    nf = NESTFUL_DATASETS.get("nestful_full")
    if nf and os.path.isfile(nf):
        stats_by["nestful"] = corpus_stats(_load_jsonl(nf))
    out: Dict[str, Any] = {"stats": stats_by}
    if "nestful" in stats_by and "v3_1" in stats_by and rows:
        dist = distance_report(stats_by)
        closer = sum(
            1 for dim in DIMENSIONS
            if dist["dimensions"][dim]["candidate"]
            < dist["dimensions"][dim]["v3_1"])
        out["distance"] = dist
        out["candidate_closer_than_v3_1_on"] = f"{closer}/{len(DIMENSIONS)}"
        out["closer_than_v3_1"] = closer > len(DIMENSIONS) / 2
    else:
        out["closer_than_v3_1"] = None
    return out


def solver_gap_section(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    gaps = [row.get("solver_gap") for row in rows
            if isinstance(row.get("solver_gap"), dict)]
    if not gaps:
        return {"available": False}
    weak = [float(g.get("weak_score", 0)) for g in gaps]
    strong = [float(g.get("strong_score", 0)) for g in gaps]
    gap_vals = [float(g.get("gap", 0)) for g in gaps]
    wfsp = sum(1 for g in gaps if float(g.get("weak_score", 1)) <= 0.5
               and float(g.get("strong_score", 0)) >= 0.8)
    return {
        "available": True,
        "n": len(gaps),
        "weak_fail_strong_pass_rate": round(wfsp / len(gaps), 4),
        "avg_weak_score": round(sum(weak) / len(weak), 4),
        "avg_strong_score": round(sum(strong) / len(strong), 4),
        "avg_gap": round(sum(gap_vals) / len(gap_vals), 4),
        "positive": (sum(gap_vals) / len(gap_vals)) >= 0.25,
    }


def grpo_signal_section(dataset_dir: str) -> Dict[str, Any]:
    """Read PROBE_REPORT.json produced by scripts/probe (never runs the probe)."""
    candidates = [
        os.path.join(dataset_dir, "reports", "PROBE_REPORT.json"),
        os.path.join(dataset_dir, "PROBE_REPORT.json"),
    ]
    probes_root = os.path.join(V3_ROOT, "outputs", "probes")
    if os.path.isdir(probes_root):
        tag = os.path.basename(os.path.normpath(dataset_dir))
        for d in sorted(os.listdir(probes_root)):
            if tag in d:
                candidates.append(os.path.join(probes_root, d, "PROBE_REPORT.json"))
    for path in candidates:
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                rep = json.load(fh)
            agg = rep.get("aggregate", rep)
            return {
                "available": True,
                "probe_report": os.path.relpath(path, REPO_ROOT),
                "dead_group_rate": agg.get("dead_group_rate"),
                "mean_unique_rewards_per_group":
                    agg.get("mean_unique_rewards_per_group"),
                "too_few_calls_rate": agg.get("too_few_calls_rate"),
                "avg_predicted_calls": agg.get("avg_predicted_calls"),
            }
    return {"available": False,
            "note": "no stage-probe report found — run scripts/probe/probe_stage.sh "
                    "on the pod (this scorer never launches it)"}


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score a NESTFUL-like dataset (validity/contamination/"
                    "distribution/solver-gap/GRPO-signal).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset-dir", default=DEFAULT_DATASET)
    ap.add_argument("--out", default=None,
                    help="report basename (default <dataset>/reports/DATASET_QUALITY)")
    args = ap.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    rows = load_corpus(dataset_dir)
    print(f"[score] dataset: {dataset_dir} ({len(rows)} rows)")
    if not rows:
        print("[score] ERROR: no rows found", file=sys.stderr)
        return 1

    validity = validity_section(rows)
    contamination = contamination_section(rows)
    distribution = distribution_section(rows)
    solver_gap = solver_gap_section(rows)
    grpo = grpo_signal_section(dataset_dir)

    technically_acceptable = bool(validity["hard_gates_pass"]
                                  and contamination["hard_gates_pass"])
    training_candidate = bool(
        technically_acceptable
        and distribution.get("closer_than_v3_1") is True
        and solver_gap.get("positive") is True
        and grpo.get("available") is True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": os.path.relpath(dataset_dir, REPO_ROOT),
        "validity": validity,
        "contamination": contamination,
        "distribution": {k: v for k, v in distribution.items() if k != "stats"},
        "solver_gap": solver_gap,
        "grpo_signal": grpo,
        "verdict": {
            "technically_acceptable": technically_acceptable,
            "training_candidate": training_candidate,
            "actually_useful": None,   # only a same-batch official eval decides
        },
    }

    out_base = args.out or os.path.join(dataset_dir, "reports", "DATASET_QUALITY")
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    with open(out_base + ".json", "w", encoding="utf-8") as fh:
        json.dump({**report, "distribution_stats": distribution.get("stats")},
                  fh, indent=2, ensure_ascii=False)

    md = [
        "# Dataset quality report", "",
        f"Dataset: `{report['dataset_dir']}` | rows: {validity['n_rows']} | "
        f"generated {report['generated_at']}", "",
        "## Verdict", "",
        f"- technically_acceptable: **{technically_acceptable}**",
        f"- training_candidate: **{training_candidate}**"
        + ("" if training_candidate else
           " (needs: distribution closer than v3.1, positive solver gap, "
           "probe signal available and better)"),
        "- actually_useful: **undetermined** — only training + same-batch "
        "official NESTFUL eval can decide this. Do not claim the dataset is "
        "good before that.", "",
        "## Validity", "",
    ]
    for k, v in validity.items():
        md.append(f"- {k}: {v}")
    md += ["", "## Contamination", ""]
    for k, v in contamination.items():
        md.append(f"- {k}: {v}")
    md += ["", "## Distribution similarity", ""]
    if "distance" in distribution:
        dist = distribution["distance"]
        names = list(dist["mean_distance"].keys())
        md += ["| dimension | " + " | ".join(names) + " |",
               "|---" * (len(names) + 1) + "|"]
        for dim in DIMENSIONS:
            md.append("| " + dim + " | " + " | ".join(
                str(dist["dimensions"][dim][n]) for n in names) + " |")
        md += ["", f"Mean distance to NESTFUL: "
               + ", ".join(f"{n}={v}" for n, v in dist["mean_distance"].items()),
               f"Candidate closer than v3.1 on "
               f"{distribution['candidate_closer_than_v3_1_on']} dimensions."]
    else:
        md.append("(NESTFUL or v3.1 reference not available)")
    md += ["", "## Solver gap", ""]
    for k, v in solver_gap.items():
        md.append(f"- {k}: {v}")
    md += ["", "## GRPO signal (stage probe)", ""]
    for k, v in grpo.items():
        md.append(f"- {k}: {v}")
    with open(out_base + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    print(f"[score] verdict: technically_acceptable={technically_acceptable} "
          f"training_candidate={training_candidate}")
    print(f"[score] reports: {out_base}.md / .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
