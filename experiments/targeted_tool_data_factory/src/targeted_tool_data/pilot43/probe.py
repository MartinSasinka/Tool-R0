"""Model-relative GRPO-signal probe over the base model.

This is neither training nor a NESTFUL evaluation: it asks whether a group of
rollouts on one of our prompts would carry a usable advantage signal. A dataset can
pass every offline gate and still be useless for GRPO if every group is uniformly
solved or uniformly hopeless, so the groups are classified by outcome spread.

If no inference backend is reachable the module writes the artifacts with
``thresholds_met=false``, the reason, and the exact command to run later. It never
invents rollouts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import RUN_ID
from .export import MASTER_FILE
from .ops import build_ops
from .pipeline import read_jsonl, write_jsonl

ROLLOUTS_FILE = "model_probe_rollouts.jsonl"
GROUPS_FILE = "model_probe_groups.csv"
REPORT_FILE = "model_probe_report.json"

ALL_CORRECT = "ALL_CORRECT"
MIXED_TERMINAL = "MIXED_TERMINAL"
ALL_FAIL_WITH_PROGRESS = "ALL_FAIL_WITH_PROGRESS"
ALL_FAIL_NO_PROGRESS = "ALL_FAIL_NO_PROGRESS"
INVALID = "INVALID"
CLASSES = (ALL_CORRECT, MIXED_TERMINAL, ALL_FAIL_WITH_PROGRESS,
           ALL_FAIL_NO_PROGRESS, INVALID)

THRESHOLDS = {
    "effective_group_rate_min": 0.60,
    "dead_group_rate_max": 0.40,
    "all_fail_no_progress_max": 0.25,
    "all_correct_max": 0.30,
    "invalid_group_rate_max": 0.02,
    "per_cell_effective_min": 0.25,
}

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
RUN_COMMAND = (
    "python -m targeted_tool_data.cli probe-pilot43-grpo-signal "
    "--sample-size 2000 --initial-rollouts 4 --max-rollouts 8 "
    "--provider openai_compatible_local --base-url http://127.0.0.1:1234/v1 "
    f"--model {DEFAULT_MODEL}"
)

PROMPT = """You have these tools:
{tools}

Task: {query}

Reply with ONLY a JSON array of tool calls. Each element must be
{{"name": "<tool>", "arguments": {{...}}, "label": "$var1"}}. To pass a previous
result as an argument, use "$varN.<output_field>$". Do not add explanations.
"""


# ── prompt / parse / grade ───────────────────────────────────────────────
def build_prompt(row: Dict[str, Any]) -> str:
    tools = [{"name": t["name"], "description": t["description"],
              "parameters": t["parameters"],
              "output": {t["output_field"]: t["output_type"]}}
             for t in row["tools"]]
    return PROMPT.format(tools=json.dumps(tools, ensure_ascii=False, indent=1),
                         query=row["question"])


def parse_calls(text: str) -> Optional[List[Dict[str, Any]]]:
    match = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not match:
        return None
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, list) or not raw:
        return None
    calls: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or "name" not in item:
            return None
        calls.append({"name": str(item["name"]),
                      "arguments": item.get("arguments") or {},
                      "label": str(item.get("label") or f"$var{len(calls) + 1}$")})
    return calls


_REF = re.compile(r"^\$?(var\d+)(?:\.([A-Za-z0-9_]+))?\$?$")


def _label_key(label: str) -> str:
    return str(label).strip().strip("$").split(".")[0].replace("_", "").lower()


def grade(row: Dict[str, Any], calls: Sequence[Dict[str, Any]]
          ) -> Dict[str, Any]:
    """Execute a predicted program with our own primitives and grade it.

    ``progress`` means the rollout reproduced at least one non-trivial oracle
    intermediate value, which is what separates a group worth learning from
    (partial credit exists) from one that is pure noise.
    """
    ops = build_ops()
    by_name = {t["name"]: t for t in row["tools"]}
    gold_values = [c.get("observation") for c in row["gold_calls"]]
    gold_answer = row["gold_answer"]

    values: Dict[str, Any] = {}
    last: Any = None
    for call in calls:
        spec = by_name.get(call["name"])
        if spec is None:
            return {"executable": False, "correct": False, "progress": False,
                    "failure": "unknown_tool", "n_calls": len(calls)}
        op = ops.get(spec["primitive_id"])
        if op is None:
            return {"executable": False, "correct": False, "progress": False,
                    "failure": "unknown_primitive", "n_calls": len(calls)}
        kwargs: Dict[str, Any] = {}
        try:
            for param, canonical in zip(spec["parameters"], op.params):
                given = call["arguments"].get(param)
                if given is None:
                    raise ValueError(f"missing argument {param}")
                if isinstance(given, str) and _REF.match(given.strip()):
                    key = _label_key(given)
                    if key not in values:
                        raise ValueError(f"unresolved reference {given}")
                    given = values[key]
                kwargs[canonical.name] = given
            produced = op.fn(**kwargs)
        except Exception as exc:                       # noqa: BLE001
            return {"executable": False, "correct": False,
                    "progress": _progress(values, gold_values),
                    "failure": f"execution_error:{type(exc).__name__}",
                    "n_calls": len(calls)}
        values[_label_key(call["label"]) or f"var{len(values) + 1}"] = produced
        last = produced

    correct = _equal(last, gold_answer)
    return {
        "executable": True,
        "correct": correct,
        "progress": correct or _progress(values, gold_values),
        "failure": "" if correct else ("too_few_calls"
                                       if len(calls) < row["call_count"]
                                       else "executable_wrong_answer"),
        "n_calls": len(calls),
    }


def _equal(got: Any, want: Any) -> bool:
    if isinstance(got, bool) or isinstance(want, bool):
        return got is want
    if isinstance(got, (int, float)) and isinstance(want, (int, float)):
        return abs(float(got) - float(want)) <= 1e-6
    return got == want


def _progress(values: Dict[str, Any], gold_values: Sequence[Any]) -> bool:
    """Did the rollout hit any oracle intermediate value worth partial credit?"""
    interesting = [v for v in gold_values
                   if v is not None and not isinstance(v, bool)
                   and not (isinstance(v, (int, float)) and abs(float(v)) <= 1)]
    for produced in values.values():
        for gold in interesting:
            if _equal(produced, gold):
                return True
    return False


# ── group classification ─────────────────────────────────────────────────
def classify(rollouts: Sequence[Dict[str, Any]]) -> str:
    if not rollouts:
        return INVALID
    parsed = [r for r in rollouts if r.get("parsed")]
    if len(parsed) < max(1, len(rollouts) // 2):
        return INVALID
    correct = sum(1 for r in rollouts if r.get("correct"))
    if correct == len(rollouts):
        return ALL_CORRECT
    if correct:
        return MIXED_TERMINAL
    if any(r.get("progress") for r in rollouts):
        return ALL_FAIL_WITH_PROGRESS
    return ALL_FAIL_NO_PROGRESS


EFFECTIVE = (MIXED_TERMINAL, ALL_FAIL_WITH_PROGRESS)
DEAD = (ALL_CORRECT, ALL_FAIL_NO_PROGRESS, INVALID)


def uncertain(rollouts: Sequence[Dict[str, Any]]) -> bool:
    """Groups worth spending the second batch of rollouts on."""
    cls = classify(rollouts)
    return cls in (ALL_CORRECT, ALL_FAIL_WITH_PROGRESS, ALL_FAIL_NO_PROGRESS)


# ── sampling ─────────────────────────────────────────────────────────────
def stratified_sample(rows: Sequence[Dict[str, Any]], size: int,
                      seed: int = 909) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = (f"{row['cell_tier']}|{row['call_bucket']}|{row['answer_type']}|"
               f"{row['declared']['structural_pattern']}")
        groups.setdefault(key, []).append(row)
    for pool in groups.values():
        rng.shuffle(pool)
    picked: List[Dict[str, Any]] = []
    keys = sorted(groups)
    index = 0
    while len(picked) < min(size, len(rows)):
        progressed = False
        for key in keys:
            pool = groups[key]
            if index < len(pool):
                picked.append(pool[index])
                progressed = True
                if len(picked) >= min(size, len(rows)):
                    break
        if not progressed:
            break
        index += 1
    return picked


# ── driver ───────────────────────────────────────────────────────────────
Sampler = Callable[[str, int, int], List[str]]


def run(out_dir: Path, *, sampler: Optional[Sampler] = None,
        sample_size: int = 2000, initial_rollouts: int = 4,
        max_rollouts: int = 8, seed: int = 909,
        model: str = DEFAULT_MODEL, provider_id: str = "",
        unavailable_reason: str = "") -> Dict[str, Any]:
    """Run the probe, or record honestly that it could not run."""
    rows = read_jsonl(out_dir / MASTER_FILE)
    chosen = stratified_sample(rows, sample_size, seed=seed)
    if sampler is None:
        return _unavailable(out_dir, chosen,
                            unavailable_reason or "no inference backend reachable",
                            model=model, sample_size=sample_size,
                            initial_rollouts=initial_rollouts,
                            max_rollouts=max_rollouts)

    t0 = time.perf_counter()
    records: List[Dict[str, Any]] = []
    groups: List[Dict[str, Any]] = []
    for row in chosen:
        prompt = build_prompt(row)
        rollouts: List[Dict[str, Any]] = []
        spent = 0
        for batch in (initial_rollouts, max_rollouts - initial_rollouts):
            if batch <= 0:
                break
            if spent and not uncertain(rollouts):
                break
            texts = sampler(prompt, batch, seed + spent)
            for offset, text in enumerate(texts):
                calls = parse_calls(text)
                result = ({"executable": False, "correct": False,
                           "progress": False, "failure": "unparseable",
                           "n_calls": 0} if calls is None
                          else grade(row, calls))
                rollouts.append({
                    "task_id": row["task_id"],
                    "rollout_index": spent + offset,
                    "parsed": calls is not None,
                    "completion_sha256": hashlib.sha256(
                        (text or "").encode("utf-8")).hexdigest(),
                    **result,
                })
            spent += batch
        records.extend(rollouts)
        cls = classify(rollouts)
        groups.append({
            "task_id": row["task_id"],
            "cell_id": row["cell_id"],
            "tier": row["cell_tier"],
            "call_bucket": row["call_bucket"],
            "answer_type": row["answer_type"],
            "pattern": row["declared"]["structural_pattern"],
            "rollouts": len(rollouts),
            "correct": sum(1 for r in rollouts if r["correct"]),
            "progress": sum(1 for r in rollouts if r["progress"]),
            "parsed": sum(1 for r in rollouts if r["parsed"]),
            "group_class": cls,
            "effective": cls in EFFECTIVE,
        })

    write_jsonl(out_dir / ROLLOUTS_FILE, records)
    _write_groups(out_dir, groups)
    report = _report(groups, model=model, provider_id=provider_id,
                     seconds=round(time.perf_counter() - t0, 1),
                     initial_rollouts=initial_rollouts,
                     max_rollouts=max_rollouts, sample_size=sample_size)
    (out_dir / REPORT_FILE).write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


def _write_groups(out_dir: Path, groups: Sequence[Dict[str, Any]]) -> None:
    columns = ["task_id", "cell_id", "tier", "call_bucket", "answer_type",
               "pattern", "rollouts", "correct", "progress", "parsed",
               "group_class", "effective"]
    with (out_dir / GROUPS_FILE).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(groups)


def _report(groups: Sequence[Dict[str, Any]], **meta: Any) -> Dict[str, Any]:
    n = len(groups)
    counts = Counter(g["group_class"] for g in groups)
    rate = (lambda k: round(counts.get(k, 0) / n, 5)) if n else (lambda k: 0.0)
    effective = round(sum(1 for g in groups if g["effective"]) / n, 5) if n else 0.0
    dead = round(sum(1 for g in groups if g["group_class"] in DEAD) / n, 5) if n else 0.0

    cells: Dict[str, Dict[str, int]] = {}
    for g in groups:
        cell = cells.setdefault(g["cell_id"], {"n": 0, "effective": 0})
        cell["n"] += 1
        cell["effective"] += int(g["effective"])
    per_cell = {k: {**v, "effective_rate": round(v["effective"] / v["n"], 4)}
                for k, v in sorted(cells.items())}
    weak_cells = {k: v for k, v in per_cell.items()
                  if v["effective_rate"] < THRESHOLDS["per_cell_effective_min"]}

    observed = {
        "n_groups": n,
        "effective_group_rate": effective,
        "dead_group_rate": dead,
        "class_distribution": {c: rate(c) for c in CLASSES},
        "all_fail_no_progress_rate": rate(ALL_FAIL_NO_PROGRESS),
        "all_correct_rate": rate(ALL_CORRECT),
        "invalid_group_rate": rate(INVALID),
    }
    unmet: List[str] = []
    if effective < THRESHOLDS["effective_group_rate_min"]:
        unmet.append(f"effective_group_rate={effective} < "
                     f"{THRESHOLDS['effective_group_rate_min']}")
    if dead > THRESHOLDS["dead_group_rate_max"]:
        unmet.append(f"dead_group_rate={dead} > {THRESHOLDS['dead_group_rate_max']}")
    if rate(ALL_FAIL_NO_PROGRESS) > THRESHOLDS["all_fail_no_progress_max"]:
        unmet.append(f"all_fail_no_progress={rate(ALL_FAIL_NO_PROGRESS)} > "
                     f"{THRESHOLDS['all_fail_no_progress_max']}")
    if rate(ALL_CORRECT) > THRESHOLDS["all_correct_max"]:
        unmet.append(f"all_correct={rate(ALL_CORRECT)} > "
                     f"{THRESHOLDS['all_correct_max']}")
    if rate(INVALID) > THRESHOLDS["invalid_group_rate_max"]:
        unmet.append(f"invalid_groups={rate(INVALID)} > "
                     f"{THRESHOLDS['invalid_group_rate_max']}")
    if weak_cells:
        unmet.append(f"{len(weak_cells)} cells below per-cell effective floor")

    return {
        "run_id": RUN_ID,
        "executed": True,
        "note": ("model-relative signal probe; not a NESTFUL evaluation and not "
                 "training"),
        **{k: v for k, v in meta.items()},
        "observed": observed,
        "thresholds": THRESHOLDS,
        "per_cell": per_cell,
        "weak_cells": weak_cells,
        "unmet_thresholds": unmet,
        "thresholds_met": not unmet,
        "remediation": _remediation(per_cell, groups),
        "sampler_metadata": {
            "model": meta.get("model"),
            "provider": meta.get("provider_id"),
            "initial_rollouts": meta.get("initial_rollouts"),
            "max_rollouts": meta.get("max_rollouts"),
            "temperature": 0.7,
            "grading": "own-primitive replay, terminal answer equivalence",
        },
    }


def _remediation(per_cell: Dict[str, Dict[str, Any]],
                 groups: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    by_cell: Dict[str, Counter] = {}
    for g in groups:
        by_cell.setdefault(g["cell_id"], Counter())[g["group_class"]] += 1
    harder, easier, fix = [], [], []
    for cell, counts in by_cell.items():
        total = sum(counts.values())
        if counts[ALL_CORRECT] / total >= 0.7:
            harder.append(cell)
        if counts[ALL_FAIL_NO_PROGRESS] / total >= 0.7:
            easier.append(cell)
        if counts[INVALID] / total >= 0.1:
            fix.append(cell)
    return {"needs_harder_siblings": sorted(harder),
            "needs_easier_siblings_or_downweight": sorted(easier),
            "needs_data_or_environment_fix": sorted(fix)}


def _unavailable(out_dir: Path, chosen: Sequence[Dict[str, Any]], reason: str,
                 **meta: Any) -> Dict[str, Any]:
    """No backend: write the sample and an explicit not-run report."""
    write_jsonl(out_dir / ROLLOUTS_FILE, [])
    _write_groups(out_dir, [{"task_id": r["task_id"], "cell_id": r["cell_id"],
                             "tier": r["cell_tier"],
                             "call_bucket": r["call_bucket"],
                             "answer_type": r["answer_type"],
                             "pattern": r["declared"]["structural_pattern"],
                             "rollouts": 0, "correct": 0, "progress": 0,
                             "parsed": 0, "group_class": "NOT_RUN",
                             "effective": False} for r in chosen])
    payload = {
        "run_id": RUN_ID,
        "executed": False,
        "thresholds_met": False,
        "reason": reason,
        "sample_prepared": len(chosen),
        "thresholds": THRESHOLDS,
        "next_command": RUN_COMMAND,
        **{k: v for k, v in meta.items()},
        "note": ("The stratified probe sample is written to "
                 f"{GROUPS_FILE} so the run can be reproduced exactly once an "
                 "inference backend is available."),
    }
    (out_dir / REPORT_FILE).write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return payload


def provider_sampler(provider: Any, *, max_tokens: int = 900,
                     temperature: float = 0.7) -> Sampler:
    """Adapt a ``providers.BaseProvider`` to the sampler signature."""
    def sample(prompt: str, n: int, seed: int) -> List[str]:
        out: List[str] = []
        for i in range(n):
            got = provider.complete(prompt, max_tokens=max_tokens,
                                    temperature=temperature, n=1, seed=seed + i)
            out.append(got[0] if got else "")
        return out
    return sample
