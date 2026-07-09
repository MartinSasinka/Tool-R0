"""Build curriculum_v4_nestful_like (Phase 1j) — generation + validation + audit.

Research dataset, NOT a replacement for canonical v3.1. Generates NESTFUL-style
synthetic stages via lib/nestful_like_generator.py, then enforces the hard
validation gates:

  * gold replay pass rate = 1.0 (independent re-execution of every row);
  * expected call count per stage;
  * no null gold answers; no unresolved $var references in gold_answer;
  * no metadata leakage (stage/motif/cluster tokens) in questions;
  * no duplicate sample_id / question / gold trace;
  * ZERO overlap with NESTFUL dev/test/full by question hash, trace hash and
    sample_id (contamination gate).

Also writes a v4 vs v3.1 vs NESTFUL distribution report with a simple
distance score (mean L1/total-variation distance to NESTFUL across dimensions).

Usage (repo root):
    python .../build_curriculum_v4_nestful_like.py --pilot          # 40/stage
    python .../build_curriculum_v4_nestful_like.py                  # 800/stage
    python .../build_curriculum_v4_nestful_like.py --examples-per-stage 200
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(V3_ROOT, "scripts", "lib"))
sys.path.insert(0, V3_ROOT)

from paths import (  # noqa: E402
    CANONICAL_STAGE_FILES, NESTFUL_DATASETS, REPO_ROOT, dataset_info,
)
from run_manifest import build_manifest, write_manifest  # noqa: E402

from lib.nestful_like_generator import (  # noqa: E402
    GENERATOR_VERSION, PROVENANCE, STAGES, question_hash, replay_task, trace_hash,
)

DEFAULT_OUT = os.path.join(V3_ROOT, "data", "curriculum_v4_nestful_like")
_LEAK_TOKENS = ("motif", "stage", "cluster", "curriculum", "sample_id", "v3_1", "v4_")
_VAR_RE = re.compile(r"\$[A-Za-z_]\w*(\.\w+)?\$")


# ---------------------------------------------------------------------------
# corpus loading for overlap + distribution comparison
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _coerce(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return v
    return v


def _norm_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize NESTFUL/v3.1/v4 rows to {question, tools, gold_calls, gold_answer}."""
    q = row.get("question") or row.get("input") or ""
    tools = _coerce(row.get("tools")) or []
    calls = _coerce(row.get("gold_calls") or row.get("output")) or []
    return {"sample_id": str(row.get("sample_id") or ""), "question": str(q),
            "tools": tools if isinstance(tools, list) else [],
            "gold_calls": calls if isinstance(calls, list) else [],
            "gold_answer": row.get("gold_answer")}


def _tool_arity(tool: Dict[str, Any]) -> int:
    params = tool.get("parameters") or {}
    props = params.get("properties", params)  # NESTFUL uses flat parameters dict
    return len(props) if isinstance(props, dict) else 0


def _arg_type(v: Any) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "reference" if _VAR_RE.fullmatch(v.strip() or "") else "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "other"


def _answer_type(v: Any) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "scalar"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "object"
    return "null"


def corpus_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    call_counts: Counter = Counter()
    offered: Counter = Counter()
    arity: Counter = Counter()
    arg_types: Counter = Counter()
    ans_types: Counter = Counter()
    q_lens: List[int] = []
    for raw in rows:
        r = _norm_row(raw)
        call_counts[min(len(r["gold_calls"]), 8)] += 1
        offered[min(len(r["tools"]), 30)] += 1
        for t in r["tools"]:
            if isinstance(t, dict):
                arity[min(_tool_arity(t), 6)] += 1
        for c in r["gold_calls"]:
            for v in (c.get("arguments") or {}).values():
                arg_types[_arg_type(v)] += 1
        ans_types[_answer_type(r["gold_answer"])] += 1
        q_lens.append(len(r["question"].split()))
    return {
        "n_rows": len(rows),
        "call_count_dist": dict(sorted(call_counts.items())),
        "offered_tools_dist": dict(sorted(offered.items())),
        "tool_arity_dist": dict(sorted(arity.items())),
        "arg_type_dist": dict(sorted(arg_types.items())),
        "answer_type_dist": dict(sorted(ans_types.items())),
        "mean_question_words": round(sum(q_lens) / len(q_lens), 1) if q_lens else None,
    }


def _l1_distance(d1: Dict, d2: Dict) -> float:
    """Total-variation distance between two count distributions (0=identical, 1=disjoint)."""
    keys = set(d1) | set(d2)
    n1, n2 = sum(d1.values()) or 1, sum(d2.values()) or 1
    return round(0.5 * sum(abs(d1.get(k, 0) / n1 - d2.get(k, 0) / n2) for k in keys), 4)


# ---------------------------------------------------------------------------
# validation gates
# ---------------------------------------------------------------------------

def validate_corpus(rows_by_stage: Dict[str, List[Dict[str, Any]]],
                    nestful_rows: List[Dict[str, Any]]) -> List[str]:
    problems: List[str] = []
    nest_q = {question_hash(_norm_row(r)["question"]) for r in nestful_rows}
    nest_t = set()
    nest_ids = set()
    for r in nestful_rows:
        n = _norm_row(r)
        if n["gold_calls"]:
            nest_t.add(trace_hash(n["gold_calls"]))
        if n["sample_id"]:
            nest_ids.add(n["sample_id"])

    seen_ids: set = set()
    seen_q: set = set()
    seen_t: set = set()
    n_replayed = 0
    for stage, rows in rows_by_stage.items():
        lo, hi = STAGES[stage]["n_calls"]
        for row in rows:
            sid = row["sample_id"]
            # duplicates
            if sid in seen_ids:
                problems.append(f"duplicate sample_id: {sid}")
            seen_ids.add(sid)
            qh = question_hash(row["question"])
            if qh in seen_q:
                problems.append(f"duplicate question: {sid}")
            seen_q.add(qh)
            th = trace_hash(row["gold_calls"])
            if th in seen_t:
                problems.append(f"duplicate gold trace: {sid}")
            seen_t.add(th)
            # call count per stage
            if not (lo <= row["num_calls"] <= hi) or len(row["gold_calls"]) != row["num_calls"]:
                problems.append(f"call count out of range for {stage}: {sid}")
            # answers
            if row["gold_answer"] is None:
                problems.append(f"null gold_answer: {sid}")
            if isinstance(row["gold_answer"], str) and _VAR_RE.search(row["gold_answer"]):
                problems.append(f"unresolved $var$ in gold_answer: {sid}")
            # leakage
            q_low = row["question"].lower()
            for tok in _LEAK_TOKENS:
                if tok in q_low:
                    problems.append(f"metadata token '{tok}' leaked into question: {sid}")
            if "$var" in q_low:
                problems.append(f"$var reference syntax leaked into question: {sid}")
            # NESTFUL overlap
            if qh in nest_q:
                problems.append(f"question overlaps NESTFUL: {sid}")
            if th in nest_t:
                problems.append(f"gold trace overlaps NESTFUL: {sid}")
            if sid in nest_ids:
                problems.append(f"sample_id overlaps NESTFUL: {sid}")
            # independent gold replay
            ok, obs = replay_task(row)
            if not ok:
                problems.append(f"gold replay FAILED: {sid} ({obs})")
            n_replayed += 1
    print(f"[v4-build] validation: {n_replayed} rows replayed, "
          f"{len(problems)} problems")
    return problems


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate + validate curriculum_v4_nestful_like.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--examples-per-stage", type=int, default=800)
    ap.add_argument("--pilot", action="store_true", help="small pilot (40/stage)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    args = ap.parse_args()

    n_per_stage = 40 if args.pilot else args.examples_per_stage
    out_dir = os.path.abspath(args.output_dir)
    filtered_dir = os.path.join(out_dir, "filtered")
    os.makedirs(filtered_dir, exist_ok=True)

    print(f"[v4-build] generator {GENERATOR_VERSION} | seed={args.seed} | "
          f"{n_per_stage} examples/stage -> {out_dir}")

    # NESTFUL corpora: used ONLY for overlap checking + aggregate distribution
    # comparison. Never copied into generated rows.
    nestful_rows: List[Dict[str, Any]] = []
    for name in ("nestful_full",):
        p = NESTFUL_DATASETS[name]
        if os.path.isfile(p):
            nestful_rows.extend(_load_jsonl(p))
    if not nestful_rows:
        print("[v4-build] ERROR: NESTFUL data not found — cannot run the "
              "contamination gate, refusing to write a corpus.", file=sys.stderr)
        return 1
    nest_qh = {question_hash(_norm_row(r)["question"]) for r in nestful_rows}
    nest_th = {trace_hash(_norm_row(r)["gold_calls"]) for r in nestful_rows
               if _norm_row(r)["gold_calls"]}

    # --- generate (dedup against NESTFUL hashes during generation) ----------
    rows_by_stage: Dict[str, List[Dict[str, Any]]] = {}
    from lib.nestful_like_generator import generate_stage
    for stage in STAGES:
        rows_by_stage[stage] = generate_stage(
            stage, n_per_stage, args.seed,
            forbidden_question_hashes=nest_qh,
            forbidden_trace_hashes=nest_th)
        print(f"[v4-build] generated {stage}: {len(rows_by_stage[stage])} rows")

    # --- validation gates -----------------------------------------------------
    problems = validate_corpus(rows_by_stage, nestful_rows)
    if problems:
        print(f"[v4-build] ERROR: {len(problems)} validation problems; first 20:",
              file=sys.stderr)
        for p in problems[:20]:
            print(f"  - {p}", file=sys.stderr)
        return 2

    # --- write stage files ------------------------------------------------------
    stage_files: Dict[str, str] = {}
    for stage, rows in rows_by_stage.items():
        path = os.path.join(filtered_dir, f"{stage}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        stage_files[stage] = path
        print(f"[v4-build] wrote {os.path.relpath(path, REPO_ROOT)} ({len(rows)} rows)")

    # --- distribution report: v4 vs v3.1 vs NESTFUL ------------------------------
    v4_rows = [r for rows in rows_by_stage.values() for r in rows]
    v31_rows: List[Dict[str, Any]] = []
    for p in CANONICAL_STAGE_FILES.values():
        if os.path.isfile(p):
            v31_rows.extend(_load_jsonl(p))
    stats = {
        "v4": corpus_stats(v4_rows),
        "v3_1": corpus_stats(v31_rows),
        "nestful": corpus_stats(nestful_rows),
    }
    dims = ("call_count_dist", "offered_tools_dist", "tool_arity_dist",
            "arg_type_dist", "answer_type_dist")
    distances = {}
    v4_closer = 0
    for dim in dims:
        d_v4 = _l1_distance(stats["v4"][dim], stats["nestful"][dim])
        d_v31 = _l1_distance(stats["v3_1"][dim], stats["nestful"][dim])
        distances[dim] = {"v4_to_nestful": d_v4, "v3_1_to_nestful": d_v31,
                          "v4_closer": d_v4 < d_v31}
        v4_closer += int(d_v4 < d_v31)
    score = {
        "dimensions": distances,
        "v4_closer_on": f"{v4_closer}/{len(dims)}",
        "v4_mean_distance": round(sum(d["v4_to_nestful"] for d in distances.values())
                                  / len(dims), 4),
        "v3_1_mean_distance": round(sum(d["v3_1_to_nestful"] for d in distances.values())
                                    / len(dims), 4),
        "technically_acceptable": v4_closer > len(dims) / 2,
    }

    # --- manifest + reports -------------------------------------------------------
    manifest = build_manifest(
        kind="curriculum_v4_nestful_like",
        datasets=list(stage_files.values()),
        seed=args.seed,
        extra={
            "generator_version": GENERATOR_VERSION,
            "provenance": PROVENANCE,
            "examples_per_stage": n_per_stage,
            "pilot": bool(args.pilot),
            "stages": {s: dataset_info(p) for s, p in stage_files.items()},
            "validation": {"problems": 0, "gold_replay_pass_rate": 1.0,
                           "nestful_overlap": 0},
            "distribution_score": score,
        },
    )
    write_manifest(manifest, os.path.join(out_dir, "manifest.json"))

    with open(os.path.join(out_dir, "DISTRIBUTION_REPORT.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"stats": stats, "score": score}, fh, indent=2, ensure_ascii=False)

    md = [
        "# curriculum_v4_nestful_like — audit & distribution report",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} | generator "
        f"{GENERATOR_VERSION} | seed {args.seed} | {n_per_stage} examples/stage"
        + (" (PILOT)" if args.pilot else ""),
        "",
        "## Contamination gate",
        "",
        "- NESTFUL questions / gold traces / tool schemas copied: **NONE** "
        "(tool library written from scratch; only aggregate statistics used — "
        "see `manifest.json:extra.provenance`).",
        f"- Overlap with NESTFUL dev/test/full (question hash, trace hash, "
        f"sample_id): **0** across {len(v4_rows)} rows.",
        "- Gold replay pass rate: **1.0** (independent re-execution).",
        "",
        "## Validation gates (all passed)",
        "",
        "- expected call count per stage; no null answers; no unresolved `$var$` "
        "in answers; no metadata leakage in questions; no duplicate "
        "sample_id/question/trace.",
        "",
        "## Distribution distance to NESTFUL (total variation, lower = closer)",
        "",
        "| dimension | v4 -> NESTFUL | v3.1 -> NESTFUL | v4 closer? |",
        "|---|---|---|---|",
    ]
    for dim, d in distances.items():
        md.append(f"| {dim} | {d['v4_to_nestful']} | {d['v3_1_to_nestful']} | "
                  f"{'YES' if d['v4_closer'] else 'no'} |")
    md += [
        "",
        f"**v4 closer to NESTFUL on {score['v4_closer_on']} dimensions** "
        f"(mean distance {score['v4_mean_distance']} vs v3.1's "
        f"{score['v3_1_mean_distance']}). Technically acceptable: "
        f"**{score['technically_acceptable']}**.",
        "",
        "## Interpretation limits",
        "",
        "Passing these gates does NOT make v4 'good'. It is a better candidate "
        "than v3.1 only if, additionally: the stage probe shows better GRPO "
        "signal on v4, AND a same-batch official NESTFUL eval improves after "
        "training on it. Neither has been run yet.",
        "",
        "## Corpus summary",
        "",
        f"- v4 rows: {len(v4_rows)} ({', '.join(f'{s}: {len(r)}' for s, r in rows_by_stage.items())})",
        f"- mean question length (words): v4={stats['v4']['mean_question_words']} "
        f"v3.1={stats['v3_1']['mean_question_words']} "
        f"nestful={stats['nestful']['mean_question_words']}",
    ]
    with open(os.path.join(out_dir, "AUDIT_REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    print(f"[v4-build] distribution: v4 closer on {score['v4_closer_on']} dims "
          f"(v4 mean dist {score['v4_mean_distance']} vs v3.1 {score['v3_1_mean_distance']})")
    print(f"[v4-build] reports: {os.path.join(out_dir, 'AUDIT_REPORT.md')}")
    print("[v4-build] done. NO training was launched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
