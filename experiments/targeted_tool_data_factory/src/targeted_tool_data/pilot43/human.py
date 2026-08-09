"""Human audit package: stratified sample, guide, import template, results.

``HUMAN_VALIDATED`` may only become true after real ratings are imported, so this
module deliberately splits into two halves: ``prepare`` writes the sample and the
empty template, ``import_results`` reads filled-in ratings back and computes
agreement. With no ratings file present the status stays false and the report says
so instead of pretending the review happened.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import RUN_ID
from .export import MASTER_FILE
from .pipeline import read_jsonl

SAMPLE_FILE = "human_audit_sample.csv"
GUIDE_FILE = "human_audit_guide.md"
TEMPLATE_FILE = "human_audit_import_template.csv"
RESULTS_FILE = "human_audit_results.json"

#: One column per question in section 29 of the specification.
QUESTIONS = (
    ("plausible_user_request", "Is the query a plausible user request?"),
    ("program_solves_query", "Does the program solve the query?"),
    ("all_nodes_necessary", "Is every program node necessary?"),
    ("graph_disclosed", "Does the query disclose the computation graph?"),
    ("all_facts_used", "Are all facts used?"),
    ("values_realistic", "Are units and values realistic?"),
    ("target_unambiguous", "Is the target unambiguous?"),
    ("answer_answers_target", "Does the final answer answer the target?"),
    ("distractors_realistic", "Are distractors realistic?"),
    ("obvious_shorter_solution", "Is there an obvious shorter solution?"),
)
#: Questions where "yes" is the defect, not the pass.
NEGATIVE = ("graph_disclosed", "obvious_shorter_solution")

THRESHOLDS = {
    "query_program_alignment": 0.98,
    "naturalness": 0.92,
    "unambiguous": 0.95,
    "target_alignment": 1.00,
    "unit_correctness": 1.00,
    "graph_leak_implicit_max": 0.02,
}


def _strata(row: Dict[str, Any]) -> List[str]:
    """Every stratum a task belongs to; the sampler must cover all of them."""
    out = [
        f"bucket={row['call_bucket']}",
        f"pattern={row['declared']['structural_pattern']}",
        f"mode={row['actual_query_mode']}",
        f"track={row['surface_track']}",
        f"answer={row['answer_type']}",
        f"tier={row['cell_tier']}",
    ]
    if any(c.get("coding_like") for c in row["gold_calls"]):
        out.append("coding")
    if row["call_bucket"] == "6+":
        out.append("long_horizon")
    critic = row["validation"].get("critic") or {}
    if critic.get("rewrites"):
        out.append("writer_rewrite")
    if row["validation"].get("second_critic", {}).get("disagreement"):
        out.append("critic_disagreement")
    if row["answer_type"] == "boolean" and row["validation"].get(
            "counterfactuals", {}).get("near_boundary"):
        out.append("boolean_near_boundary")
    return out


def sample(rows: Sequence[Dict[str, Any]], *, size: int = 400,
           min_six_plus: int = 120, seed: int = 4711
           ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Cover every stratum first, then fill up to ``size`` with a spread draw."""
    rng = random.Random(seed)
    by_stratum: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        for key in _strata(row):
            by_stratum.setdefault(key, []).append(row)

    picked: Dict[str, Dict[str, Any]] = {}
    for key in sorted(by_stratum):
        pool = sorted(by_stratum[key], key=lambda r: r["task_id"])
        rng.shuffle(pool)
        for row in pool[:2]:                     # two per stratum for agreement
            picked[row["task_id"]] = row

    six = [r for r in rows if r["call_bucket"] == "6+"]
    rng.shuffle(six)
    for row in six:
        if sum(1 for r in picked.values() if r["call_bucket"] == "6+") >= min_six_plus:
            break
        picked[row["task_id"]] = row

    rest = [r for r in rows if r["task_id"] not in picked]
    rng.shuffle(rest)
    for row in rest:
        if len(picked) >= size:
            break
        picked[row["task_id"]] = row

    chosen = sorted(picked.values(), key=lambda r: r["task_id"])
    covered = {k: sum(1 for r in chosen if k in _strata(r)) for k in by_stratum}
    stats = {
        "n": len(chosen),
        "requested": size,
        "six_plus": sum(1 for r in chosen if r["call_bucket"] == "6+"),
        "min_six_plus": min_six_plus,
        "strata_total": len(by_stratum),
        "strata_covered": sum(1 for v in covered.values() if v),
        "uncovered_strata": [k for k, v in covered.items() if not v],
        "per_stratum": dict(sorted(covered.items())),
    }
    return chosen, stats


def _row_for_reviewer(row: Dict[str, Any]) -> Dict[str, Any]:
    calls = " | ".join(
        f"{c['label']}={c['name']}({', '.join(f'{k}={v!r}' for k, v in c['arguments'].items())})"
        for c in row["gold_calls"])
    facts = " | ".join(f"{f['role']}={f['value']!r} ({f['semantic_type']})"
                       for f in row["stated_facts"])
    return {
        "task_id": row["task_id"],
        "tier": row["cell_tier"],
        "workflow_id": row["workflow_id"],
        "natural_user_goal": row["natural_user_goal"],
        "query": row["question"],
        "query_mode": row["actual_query_mode"],
        "surface_track": row["surface_track"],
        "call_count": row["call_count"],
        "answer_type": row["answer_type"],
        "gold_answer": json.dumps(row["gold_answer"], ensure_ascii=False),
        "stated_facts": facts,
        "gold_program": calls,
        "offered_tools": ", ".join(t["name"] for t in row["tools"]),
        "structural_pattern": row["declared"]["structural_pattern"],
        "query_source": row["query_source"],
    }


def prepare(out_dir: Path, *, size: int = 400, seed: int = 4711,
            reviewers: Sequence[str] = ("R1", "R2")) -> Dict[str, Any]:
    rows = read_jsonl(out_dir / MASTER_FILE)
    if not rows:
        raise FileNotFoundError(f"{out_dir / MASTER_FILE} is empty; export first")
    chosen, stats = sample(rows, size=size, seed=seed)
    display = [_row_for_reviewer(r) for r in chosen]
    columns = list(display[0].keys())
    _csv(out_dir / SAMPLE_FILE, columns, display)

    template_cols = (["task_id", "reviewer_id"]
                     + [q for q, _text in QUESTIONS] + ["notes"])
    _csv(out_dir / TEMPLATE_FILE, template_cols,
         [{"task_id": r["task_id"], "reviewer_id": rev,
           **{q: "" for q, _t in QUESTIONS}, "notes": ""}
          for r in chosen for rev in reviewers])
    (out_dir / GUIDE_FILE).write_text(_guide(stats, reviewers), encoding="utf-8")
    stats["files"] = [SAMPLE_FILE, TEMPLATE_FILE, GUIDE_FILE]
    stats["reviewers"] = list(reviewers)
    return stats


def _csv(path: Path, columns: Sequence[str],
         rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _guide(stats: Dict[str, Any], reviewers: Sequence[str]) -> str:
    qs = "\n".join(f"{i}. **{q}** — {text}"
                   + ("  \n   *(yes here means a defect)*" if q in NEGATIVE else "")
                   for i, (q, text) in enumerate(QUESTIONS, 1))
    return f"""# Pilot4.3 human audit guide

run_id: `{RUN_ID}`

`{stats['n']}` tasks sampled from `{MASTER_FILE}`, covering
{stats['strata_covered']}/{stats['strata_total']} strata, including
{stats['six_plus']} tasks with six or more calls.

Two independent reviewers ({', '.join(reviewers)}) rate **every** sampled task so
agreement can be computed. Do not discuss tasks before both passes are finished.

## How to rate

Open `{SAMPLE_FILE}`, and for each row fill one line per reviewer in
`{TEMPLATE_FILE}`. Answer every question with `yes`, `no`, or `unsure`.

{qs}

## What each question means here

* *Plausible user request*: would a real person send this message to an assistant?
  Stilted phrasing is a defect even when the content is correct.
* *Program solves the query*: read the gold program and check it computes what the
  query asks for, using only the facts stated in the query.
* *Every node necessary*: if you can delete a call and still answer the question,
  answer `no` and name the call in `notes`.
* *Query discloses the computation graph*: does the query name the tools, the
  number of steps, or spell out the dependency order? For implicit modes this
  must be `no`.
* *All facts used*: every number, entity and unit in the query is consumed by the
  program, and the program needs no value the query does not state.
* *Units and values realistic*: prices, durations, quantities and paths look like
  something from the domain, not filler.
* *Target unambiguous*: exactly one reading of what to return.
* *Answer answers the target*: the gold answer is the thing the query asked for,
  in the right type and unit.
* *Distractors realistic*: the non-gold tools are plausible confusions, not noise.
* *Obvious shorter solution*: could you answer with strictly fewer calls?

## Import

```powershell
$env:PYTHONPATH="src"; python -m targeted_tool_data.cli import-human-audit-pilot43 `
  --ratings outputs/{RUN_ID}/human_audit_ratings.csv
```

Thresholds for `HUMAN_VALIDATED=true`: query-program alignment
>= {THRESHOLDS['query_program_alignment']:.0%}, naturalness
>= {THRESHOLDS['naturalness']:.0%}, unambiguous
>= {THRESHOLDS['unambiguous']:.0%}, target alignment
= {THRESHOLDS['target_alignment']:.0%}, unit correctness
= {THRESHOLDS['unit_correctness']:.0%}, implicit graph leak
<= {THRESHOLDS['graph_leak_implicit_max']:.0%}.
"""


# ── import and agreement ─────────────────────────────────────────────────
def _yes(value: str) -> Optional[bool]:
    text = (value or "").strip().lower()
    if text in ("yes", "y", "true", "1", "pass"):
        return True
    if text in ("no", "n", "false", "0", "fail"):
        return False
    return None


def cohens_kappa(a: Sequence[Optional[bool]],
                 b: Sequence[Optional[bool]]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    n = len(pairs)
    agree = sum(1 for x, y in pairs if x == y) / n
    pa = sum(1 for x, _y in pairs if x) / n
    pb = sum(1 for _x, y in pairs if y) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected >= 1.0:
        return 1.0 if agree >= 1.0 else 0.0
    return round((agree - expected) / (1 - expected), 4)


def import_results(out_dir: Path, ratings_path: Path) -> Dict[str, Any]:
    """Read real reviewer ratings and decide ``HUMAN_VALIDATED``."""
    master = {r["task_id"]: r for r in read_jsonl(out_dir / MASTER_FILE)}
    with ratings_path.open(encoding="utf-8-sig", newline="") as fh:
        raw = list(csv.DictReader(fh))
    if not raw:
        raise ValueError(f"no rating rows in {ratings_path}")

    per_task: Dict[str, Dict[str, List[Optional[bool]]]] = {}
    reviewers: Counter = Counter()
    for line in raw:
        tid = (line.get("task_id") or "").strip()
        rev = (line.get("reviewer_id") or "").strip() or "R?"
        if not tid:
            continue
        reviewers[rev] += 1
        slot = per_task.setdefault(tid, {q: [] for q, _t in QUESTIONS})
        for q, _text in QUESTIONS:
            slot[q].append(_yes(line.get(q, "")))

    def _pass(q: str, values: Sequence[Optional[bool]]) -> Optional[bool]:
        seen = [v for v in values if v is not None]
        if not seen:
            return None
        want = q not in NEGATIVE
        return all(v == want for v in seen)

    rates: Dict[str, Dict[str, Any]] = {}
    for q, _text in QUESTIONS:
        decided = [(tid, _pass(q, slot[q])) for tid, slot in per_task.items()]
        rated = [v for _t, v in decided if v is not None]
        rates[q] = {
            "n_rated": len(rated),
            "pass_rate": round(sum(1 for v in rated if v) / len(rated), 4)
            if rated else None,
            "failing_task_ids": sorted(t for t, v in decided if v is False)[:50],
        }

    kappas: Dict[str, Optional[float]] = {}
    ordered = sorted(per_task)
    for q, _text in QUESTIONS:
        first = [per_task[t][q][0] if per_task[t][q] else None for t in ordered]
        second = [per_task[t][q][1] if len(per_task[t][q]) > 1 else None
                  for t in ordered]
        kappas[q] = cohens_kappa(first, second)
    disagreements = {
        q: sum(1 for t in ordered
               if len(per_task[t][q]) > 1
               and per_task[t][q][0] is not None
               and per_task[t][q][1] is not None
               and per_task[t][q][0] != per_task[t][q][1])
        for q, _text in QUESTIONS}

    implicit_leak = _implicit_leak(master, per_task)
    observed = {
        "query_program_alignment": rates["program_solves_query"]["pass_rate"],
        "naturalness": rates["plausible_user_request"]["pass_rate"],
        "unambiguous": rates["target_unambiguous"]["pass_rate"],
        "target_alignment": rates["answer_answers_target"]["pass_rate"],
        "unit_correctness": rates["values_realistic"]["pass_rate"],
        "graph_leak_implicit": implicit_leak,
    }
    unmet = []
    for key, floor in THRESHOLDS.items():
        if key == "graph_leak_implicit_max":
            got = observed["graph_leak_implicit"]
            if got is None or got > floor:
                unmet.append(f"graph_leak_implicit={got} > {floor}")
            continue
        got = observed.get(key)
        if got is None or got < floor:
            unmet.append(f"{key}={got} < {floor}")

    by_stratum = _by_stratum(master, per_task)
    result = {
        "run_id": RUN_ID,
        "ratings_file": str(ratings_path),
        "n_tasks_rated": len(per_task),
        "n_rating_rows": len(raw),
        "reviewers": dict(reviewers),
        "double_rated_tasks": sum(1 for t in ordered
                                  if len(per_task[t][QUESTIONS[0][0]]) > 1),
        "question_pass_rates": rates,
        "cohens_kappa": kappas,
        "disagreements": disagreements,
        "observed": observed,
        "thresholds": THRESHOLDS,
        "unmet_thresholds": unmet,
        "thresholds_met": not unmet,
        "by_stratum": by_stratum,
    }
    (out_dir / RESULTS_FILE).write_text(
        json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result


def _implicit_leak(master: Dict[str, Dict[str, Any]],
                   per_task: Dict[str, Dict[str, List[Optional[bool]]]]
                   ) -> Optional[float]:
    implicit = [t for t in per_task
                if (master.get(t) or {}).get("actual_query_mode")
                in ("GOAL_BASED_IMPLICIT", "DOMAIN_GROUNDED_IMPLICIT")]
    if not implicit:
        return None
    leaked = sum(1 for t in implicit
                 if any(v is True for v in per_task[t]["graph_disclosed"]))
    return round(leaked / len(implicit), 4)


def _by_stratum(master: Dict[str, Dict[str, Any]],
                per_task: Dict[str, Dict[str, List[Optional[bool]]]]
                ) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for tid, slot in per_task.items():
        row = master.get(tid)
        if not row:
            continue
        ok = all(all(v == (q not in NEGATIVE) for v in slot[q] if v is not None)
                 for q, _t in QUESTIONS)
        for key in _strata(row):
            cell = out.setdefault(key, {"n": 0, "pass": 0})
            cell["n"] += 1
            cell["pass"] += int(ok)
    for cell in out.values():
        cell["pass_rate"] = round(cell["pass"] / cell["n"], 4) if cell["n"] else None
    return dict(sorted(out.items()))


def pending_notice(out_dir: Path) -> Dict[str, Any]:
    """Written when no ratings exist, so the status is false *and* explained."""
    payload = {
        "run_id": RUN_ID,
        "thresholds_met": False,
        "reason": "no reviewer ratings imported yet",
        "n_tasks_rated": 0,
        "thresholds": THRESHOLDS,
        "next_command": ("python -m targeted_tool_data.cli "
                         "import-human-audit-pilot43 --ratings <filled csv>"),
        "sample": SAMPLE_FILE,
        "template": TEMPLATE_FILE,
        "guide": GUIDE_FILE,
    }
    (out_dir / RESULTS_FILE).write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return payload
