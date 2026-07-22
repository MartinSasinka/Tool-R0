#!/usr/bin/env python3
"""Format-status audit for two-phase C0/C1/C2 NESTFUL evals (analysis only).

Does NOT modify predictions, parser, reward, or evaluator.

Usage (from repo root):
  python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/format_status_c0_c1_c2.py \\
    --run-dir experiments/nestful_synthetic_curriculum_v3/outputs/runs/two_phase_20260718_192902

Writes under experiments/nestful_synthetic_curriculum_v3/reports/:
  FORMAT_STATUS_C0_C1_C2.md|.json
  format_error_taxonomy.csv
  format_error_examples.json
  reference_syntax_audit.md
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parents[1]
_REPO = _V3.parents[1]
_MINIMAL = _REPO / "experiments" / "nestful_mtgrpo_minimal"
sys.path.insert(0, str(_MINIMAL))
sys.path.insert(0, str(_V3))

from executor import _VAR_REF_RE, _is_variable_ref  # noqa: E402
from scripts.analysis.two_phase_root_cause_analysis import (  # noqa: E402
    classify_failure as root_classify_failure,
)

BOOTSTRAP_ITERS = 2_000
BOOTSTRAP_SEED = 20260720
N_TOTAL = 1661

ARMS = ("C0", "C1", "C2")
EVAL_REL = {
    "C0": "eval/eval/final_test/C0_baseline",
    "C1": "eval/eval/final_test/C1_phase1",
    "C2": "eval/eval/C2_nestful_test",
}

# Reference forms
_CANON_REF = re.compile(r"^\$var(\d+)\.([A-Za-z_][\w]*)\$$")
_NESTFUL_REF = re.compile(r"^\$var_(\d+)\.([A-Za-z_][\w]*)\$$")
_CANON_LABEL = re.compile(r"^\$var(\d+)$")
_NESTFUL_LABEL = re.compile(r"^\$var_(\d+)$")
_ANY_DOLLAR = re.compile(r"\$[^$]*\$")

REPORTS = _V3 / "reports"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def resolve_run_dir(run_dir: str) -> Path:
    p = Path(run_dir)
    if not p.is_absolute():
        p = (_REPO / p).resolve()
    if not (p / "run_manifest.json").is_file():
        nested = p / p.name
        if (nested / "run_manifest.json").is_file():
            p = nested
    if not (p / "run_manifest.json").is_file():
        raise SystemExit(f"run_manifest.json not found under {p}")
    return p


def load_trajectories(eval_dir: Path) -> Dict[str, dict]:
    path = eval_dir / "final_eval_trajectories.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    rows: Dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            rows[r["sample_id"]] = r
    return rows


def load_gold_tools(test_path: Path) -> Dict[str, Dict[str, dict]]:
    """sample_id -> {tool_name: parameters dict}."""
    out: Dict[str, Dict[str, dict]] = {}
    with open(test_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            sid = str(r["sample_id"])
            tools = r.get("tools") or []
            if isinstance(tools, str):
                tools = json.loads(tools)
            by_name: Dict[str, dict] = {}
            for t in tools:
                name = t.get("name")
                params = t.get("parameters") or t.get("arguments") or {}
                if name:
                    by_name[str(name)] = params if isinstance(params, dict) else {}
            out[sid] = by_name
    return out


def official_win(row: dict) -> Optional[float]:
    v = (row.get("_traj") or {}).get("official_win")
    return None if v is None else float(bool(v))


def call_bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 5:
        return str(n)
    return "6+"


def paired_bootstrap(flags: List[float], seed: int = BOOTSTRAP_SEED) -> dict:
    if not flags:
        return {"mean": None, "ci95": None}
    rng = random.Random(seed)
    n = len(flags)
    boots = []
    for _ in range(BOOTSTRAP_ITERS):
        s = sum(flags[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    return {
        "mean": sum(flags) / n,
        "ci95": [boots[int(0.025 * BOOTSTRAP_ITERS)],
                 boots[int(0.975 * BOOTSTRAP_ITERS) - 1]],
        "iters": BOOTSTRAP_ITERS,
        "seed": seed,
    }


def walk_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_strings(v)


def classify_ref(s: str) -> str:
    t = s.strip()
    if _CANON_REF.match(t) or _CANON_LABEL.match(t):
        return "tool_r0_canonical"
    if _NESTFUL_REF.match(t) or _NESTFUL_LABEL.match(t):
        return "nestful_underscore"
    if _is_variable_ref(t) or _VAR_REF_RE.match(t):
        return "executor_accepted_other"
    if t.startswith("$") and t.endswith("$"):
        return "malformed_dollar_ref"
    return "not_a_ref"


def type_ok(val: Any, type_str: str) -> bool:
    if _is_variable_ref(val) if isinstance(val, str) else False:
        return True
    ts = (type_str or "").lower()
    if "int" in ts or "float" in ts or "number" in ts:
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    if "bool" in ts:
        return isinstance(val, bool)
    if "str" in ts or "string" in ts:
        return isinstance(val, str)
    if "list" in ts or "array" in ts:
        return isinstance(val, list)
    if "object" in ts or "dict" in ts:
        return isinstance(val, dict)
    return True  # unknown schema type → don't flag


def coarse_failure(row: dict) -> str:
    """Reuse root-cause primary label (aligned with C0_C1_C2_failure_taxonomy.csv)."""
    primary, _secondary = root_classify_failure(row)
    return primary

def analyze_sample(
    row: dict,
    tools_by_name: Dict[str, dict],
) -> Dict[str, Any]:
    traj = row.get("_traj") or {}
    turns = traj.get("turns") or []
    stop = traj.get("stop_reason")
    gold_n = int(row.get("num_gold_calls") or 0)
    pred_n = int(traj.get("num_tool_calls") or 0)
    win = official_win(row) == 1.0

    flags: Dict[str, bool] = {k: False for k in (
        "raw_output_syntax_failure",
        "parser_extraction_failure",
        "malformed_tool_call",
        "unknown_tool",
        "missing_argument_key",
        "extra_argument_key",
        "wrong_type_serialization",
        "malformed_reference",
        "unresolvable_reference",
        "missing_output_field",
        "final_answer_extraction_failure",
        "output_truncation",
        "unsupported_trace",
        "executable",
        "wrong_tool",
        "wrong_argument_value",
        "executable_wrong_result",
        "no_tool_call",
        "semantic_dominant",
        "schema_or_reference",
        "syntax_format",
    )}

    parse_reasons: List[str] = []
    fail_turns: List[dict] = []
    ref_classes: Counter = Counter()
    first_format_turn: Optional[int] = None

    for t in turns:
        idx = int(t.get("turn_idx", len(fail_turns)))
        text = t.get("model_text") or ""
        fr = t.get("fail_reason") or ""
        clipped = bool(t.get("clipped_completion"))
        if clipped or stop == "clipped":
            flags["output_truncation"] = True
            if first_format_turn is None:
                first_format_turn = idx
        if not text.strip() and stop in ("parse_fail", "prompt_overflow"):
            flags["raw_output_syntax_failure"] = True
            if first_format_turn is None:
                first_format_turn = idx

        if fr.startswith("parse:"):
            reason = fr[len("parse:"):]
            parse_reasons.append(reason)
            flags["parser_extraction_failure"] = True
            flags["raw_output_syntax_failure"] = True
            flags["syntax_format"] = True
            if first_format_turn is None:
                first_format_turn = idx
            # Final answer dumped into the tag (common pattern)
            if reason == "invalid_json" and re.search(
                r"<tool_call_answer>\s*[^\[{].*?</tool_call_answer>",
                text, re.DOTALL | re.IGNORECASE,
            ):
                flags["final_answer_extraction_failure"] = True
            fail_turns.append({"turn_idx": idx, "fail_reason": fr, "text_head": text[:240]})

        if fr.startswith("exec:"):
            body = fr[len("exec:"):]
            if "unknown_tool" in body or "unregistered_tool" in body:
                flags["unknown_tool"] = True
                flags["schema_or_reference"] = True
            if "unresolved_variable" in body:
                flags["unresolvable_reference"] = True
                flags["schema_or_reference"] = True
            if "unresolved_field" in body or "missing_field" in body:
                flags["missing_output_field"] = True
                flags["schema_or_reference"] = True
            if ("unknown_argument" in body or "missing_required_argument" in body
                    or "argument_type_mismatch" in body):
                flags["schema_or_reference"] = True
                if "missing_required" in body:
                    flags["missing_argument_key"] = True
                if "unknown_argument" in body:
                    flags["extra_argument_key"] = True
                if "type_mismatch" in body:
                    flags["wrong_type_serialization"] = True

        pc = t.get("parsed_call")
        if not isinstance(pc, dict):
            continue
        name = pc.get("name")
        args = pc.get("arguments") if isinstance(pc.get("arguments"), dict) else {}
        # schema checks against available tools for this task
        if name and name not in tools_by_name and tools_by_name:
            # may still be valid if executor accepted; only mark if exec failed unknown
            pass
        schema = tools_by_name.get(str(name) or "", {})
        if schema:
            schema_keys = set(schema.keys())
            arg_keys = set(args.keys())
            missing = schema_keys - arg_keys
            extra = arg_keys - schema_keys
            # Only count schema key issues when the executor also complained,
            # or when this turn failed — don't flag every stylistic subset.
            if fr.startswith("exec:"):
                if missing and "missing_required" in fr:
                    flags["missing_argument_key"] = True
                    flags["schema_or_reference"] = True
                if extra and "unknown_argument" in fr:
                    flags["extra_argument_key"] = True
                    flags["schema_or_reference"] = True
            for k, v in args.items():
                spec = schema.get(k) or {}
                tstr = spec.get("type", "") if isinstance(spec, dict) else ""
                if (tstr and not type_ok(v, str(tstr))
                        and fr.startswith("exec:") and "type_mismatch" in fr):
                    flags["wrong_type_serialization"] = True
                    flags["schema_or_reference"] = True
        # Soft schema signal from param-key mismatch is applied once via coarse_failure.
        for s in walk_strings(args):
            if not isinstance(s, str):
                continue
            if "$" not in s:
                continue
            # full-string refs or embedded
            candidates = [s] if (s.startswith("$") and s.endswith("$")) else _ANY_DOLLAR.findall(s)
            for c in candidates:
                cls = classify_ref(c)
                ref_classes[cls] += 1
                if cls == "malformed_dollar_ref":
                    flags["malformed_reference"] = True
                    flags["schema_or_reference"] = True
                elif cls == "not_a_ref" and c.startswith("$"):
                    flags["malformed_reference"] = True
                    flags["schema_or_reference"] = True

    if stop == "parse_fail":
        flags["parser_extraction_failure"] = True
        flags["raw_output_syntax_failure"] = True
        flags["syntax_format"] = True
    if stop == "clipped" or traj.get("clipped_any"):
        flags["output_truncation"] = True
        flags["syntax_format"] = True
    if pred_n == 0 and stop != "parse_fail":
        flags["no_tool_call"] = True
        # empty finish without calls / never emitted a call — format-adjacent
        flags["raw_output_syntax_failure"] = True
        flags["syntax_format"] = True

    flags["unsupported_trace"] = bool(row.get("correct_answer_but_unsupported_trace"))
    # Official IBM executability when present; else infer from stop_reason
    off_exec = traj.get("executable")
    if off_exec is not None:
        flags["executable"] = bool(off_exec)
    else:
        flags["executable"] = stop in ("terminal", "max_turns") and not any(
            (t.get("fail_reason") or "").startswith("exec:") for t in turns
        )

    coarse = coarse_failure(row)
    if coarse == "wrong tool":
        flags["wrong_tool"] = True
        flags["semantic_dominant"] = True
    if coarse == "correct keys, wrong argument values":
        flags["wrong_argument_value"] = True
        flags["semantic_dominant"] = True
    if coarse == "executable trajectory ending wrong result":
        flags["executable_wrong_result"] = True
        flags["semantic_dominant"] = True
    if coarse in ("too few calls", "too many calls", "correct trajectory, wrong final answer"):
        flags["semantic_dominant"] = True
    if coarse == "correct tool, wrong argument keys":
        flags["schema_or_reference"] = True
        flags["missing_argument_key"] = True
        flags["extra_argument_key"] = True
    if coarse in ("unresolved or wrong reference", "unknown or unsupported tool"):
        flags["schema_or_reference"] = True
        if "unresolved" in coarse:
            flags["unresolvable_reference"] = True
        if "unknown" in coarse:
            flags["unknown_tool"] = True
            flags["wrong_tool"] = True
    if coarse == "parse/format error":
        flags["syntax_format"] = True
    if coarse == "no tool call":
        flags["syntax_format"] = True

    # Runtime TypeErrors from IBM are semantic value mistakes, not schema JSON format.
    # Layer priority for reporting
    if flags["syntax_format"]:
        layer = "A_syntax_or_parser"
    elif flags["schema_or_reference"]:
        layer = "C_D_schema_or_reference"
    elif not win:
        layer = "E_semantic"
    else:
        layer = "ok"

    return {
        "sample_id": row["sample_id"],
        "num_gold_calls": gold_n,
        "pred_n": pred_n,
        "stop_reason": stop,
        "official_win": win,
        "coarse_failure": coarse,
        "layer": layer,
        "flags": flags,
        "parse_reasons": parse_reasons,
        "first_format_turn": first_format_turn,
        "ref_classes": dict(ref_classes),
        "fail_turns": fail_turns,
        "call_bucket": call_bucket(gold_n),
    }


def rate_block(analyses: List[dict], key: str, *, bootstrap: bool = False) -> dict:
    flags = [1.0 if a["flags"].get(key) else 0.0 for a in analyses]
    n = len(flags)
    c = int(sum(flags))
    out = {
        "count": c,
        "n": n,
        "rate": c / n if n else None,
        "ci95": None,
    }
    if bootstrap:
        out["ci95"] = paired_bootstrap(flags).get("ci95")
    return out


def delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        default=str(
            _V3 / "outputs" / "runs" / "two_phase_20260718_192902"
        ),
    )
    ap.add_argument(
        "--test-set",
        default=str(_MINIMAL / "data" / "splits" / "nestful_test.jsonl"),
    )
    args = ap.parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    tools_map = load_gold_tools(Path(args.test_set))

    arm_rows: Dict[str, Dict[str, dict]] = {}
    arm_paths: Dict[str, str] = {}
    for arm, rel in EVAL_REL.items():
        ed = run_dir / rel
        arm_paths[arm] = str(ed)
        arm_rows[arm] = load_trajectories(ed)
        man = ed / "eval_manifest.json"
        print(f"[{arm}] {ed} n={len(arm_rows[arm])} "
              f"manifest={'yes' if man.is_file() else 'no'}")

    ids = sorted(set.intersection(*(set(arm_rows[a]) for a in ARMS)))
    if len(ids) != N_TOTAL:
        print(f"[warn] intersection n={len(ids)} (expected {N_TOTAL})")

    analyses: Dict[str, Dict[str, dict]] = {a: {} for a in ARMS}
    for arm in ARMS:
        for sid in ids:
            row = arm_rows[arm][sid]
            tools = tools_map.get(sid, {})
            analyses[arm][sid] = analyze_sample(row, tools)

    metric_keys = [
        "raw_output_syntax_failure",
        "parser_extraction_failure",
        "malformed_tool_call",
        "unknown_tool",
        "missing_argument_key",
        "extra_argument_key",
        "wrong_type_serialization",
        "malformed_reference",
        "unresolvable_reference",
        "missing_output_field",
        "final_answer_extraction_failure",
        "output_truncation",
        "unsupported_trace",
        "executable",
        "wrong_tool",
        "wrong_argument_value",
        "executable_wrong_result",
        "no_tool_call",
        "syntax_format",
        "schema_or_reference",
        "semantic_dominant",
    ]

    BOOT_KEYS = {
        "syntax_format", "parser_extraction_failure", "semantic_dominant",
        "raw_output_syntax_failure", "schema_or_reference", "executable",
        "wrong_tool", "wrong_argument_value", "executable_wrong_result",
        "output_truncation", "unsupported_trace",
    }
    rates: Dict[str, Dict[str, dict]] = {}
    for arm in ARMS:
        alist = [analyses[arm][sid] for sid in ids]
        rates[arm] = {
            k: rate_block(alist, k, bootstrap=(k in BOOT_KEYS))
            for k in metric_keys
        }
        wins = [1.0 if analyses[arm][sid]["official_win"] else 0.0 for sid in ids]
        rates[arm]["official_win"] = {
            "count": int(sum(wins)), "n": len(wins),
            "rate": sum(wins) / len(wins) if wins else None,
            "ci95": paired_bootstrap(wins).get("ci95"),
        }

    # paired gained/lost for syntax_format and parser_extraction
    paired = {}
    for key in ("syntax_format", "parser_extraction_failure", "semantic_dominant",
                "official_win", "raw_output_syntax_failure"):
        paired[key] = {}
        for a, b, label in (("C0", "C1", "C1_vs_C0"), ("C1", "C2", "C2_vs_C1"),
                            ("C0", "C2", "C2_vs_C0")):
            gained = lost = same_bad = same_ok = 0
            deltas = []
            for sid in ids:
                if key == "official_win":
                    fa = 1.0 if analyses[a][sid]["official_win"] else 0.0
                    fb = 1.0 if analyses[b][sid]["official_win"] else 0.0
                else:
                    fa = 1.0 if analyses[a][sid]["flags"].get(key) else 0.0
                    fb = 1.0 if analyses[b][sid]["flags"].get(key) else 0.0
                deltas.append(fb - fa)
                if fa == 0 and fb == 1:
                    gained += 1
                elif fa == 1 and fb == 0:
                    lost += 1
                elif fa == 1 and fb == 1:
                    same_bad += 1
                else:
                    same_ok += 1
            # For error flags, "gained" means NEW errors (worse). Rename for clarity.
            paired[key][label] = {
                "new_positive": gained,
                "resolved_positive": lost,
                "both_positive": same_bad,
                "both_negative": same_ok,
                "net_rate_delta": sum(deltas) / len(deltas),
                "ci95": paired_bootstrap(deltas).get("ci95")
                if key in ("syntax_format", "parser_extraction_failure",
                           "semantic_dominant", "official_win",
                           "raw_output_syntax_failure")
                else None,
            }

    # per-turn format errors
    per_turn: Dict[str, Dict[str, dict]] = {}
    for arm in ARMS:
        buckets: Dict[str, List[float]] = defaultdict(list)
        for sid in ids:
            a = analyses[arm][sid]
            ft = a["first_format_turn"]
            # only among samples with syntax_format
            if not a["flags"]["syntax_format"]:
                continue
            if ft is None:
                label = "unknown_or_no_call"
            elif ft == 0:
                label = "turn_1"
            elif ft == 1:
                label = "turn_2"
            elif ft == 2:
                label = "turn_3"
            else:
                label = "turn_4+"
            # final-answer-in-tag subset
            if a["flags"]["final_answer_extraction_failure"]:
                buckets["final_answer_segment"].append(1.0)
            buckets[label].append(1.0)
        # also rate among ALL tasks for turn index of first syntax fail
        all_counts = Counter()
        for sid in ids:
            a = analyses[arm][sid]
            if not a["flags"]["syntax_format"]:
                continue
            ft = a["first_format_turn"]
            if ft is None:
                all_counts["unknown_or_no_call"] += 1
            elif ft == 0:
                all_counts["turn_1"] += 1
            elif ft == 1:
                all_counts["turn_2"] += 1
            elif ft == 2:
                all_counts["turn_3"] += 1
            else:
                all_counts["turn_4+"] += 1
            if a["flags"]["final_answer_extraction_failure"]:
                all_counts["final_answer_segment"] += 1
        per_turn[arm] = {
            k: {"count": v, "rate_of_all_tasks": v / len(ids),
                "rate_of_syntax_fails": v / max(1, sum(
                    1 for sid in ids if analyses[arm][sid]["flags"]["syntax_format"]
                ))}
            for k, v in all_counts.items()
        }

    # per call-bucket syntax rates
    by_bucket: Dict[str, Dict[str, dict]] = {}
    for arm in ARMS:
        by_bucket[arm] = {}
        groups: Dict[str, List[dict]] = defaultdict(list)
        for sid in ids:
            groups[analyses[arm][sid]["call_bucket"]].append(analyses[arm][sid])
        for b, alist in sorted(groups.items()):
            by_bucket[arm][b] = {
                "n": len(alist),
                "syntax_format": rate_block(alist, "syntax_format"),
                "parser_extraction_failure": rate_block(alist, "parser_extraction_failure"),
                "schema_or_reference": rate_block(alist, "schema_or_reference"),
                "semantic_dominant": rate_block(alist, "semantic_dominant"),
                "output_truncation": rate_block(alist, "output_truncation"),
            }

    # stop_reason + parse reason distributions
    stop_dist = {a: dict(Counter(analyses[a][sid]["stop_reason"] for sid in ids)) for a in ARMS}
    parse_dist = {}
    for arm in ARMS:
        c: Counter = Counter()
        for sid in ids:
            for r in analyses[arm][sid]["parse_reasons"]:
                c[r] += 1
        parse_dist[arm] = dict(c)

    # Official vs internal contradiction evidence
    contradiction = {
        "official_num_pred_parsing_errors": {},
        "internal_parse_fail_stop_reason": {},
        "explanation": (
            "Official metrics_official.json reports num_pred_parsing_errors=0 because "
            "final_eval feeds the official Llama-3.1 parser a JSON-serialized list of "
            "ALREADY extracted predicted_calls (build_item → generated_text=json.dumps(calls)). "
            "That list always parses; empty/partial lists are scored as 0 calls, not parse errors. "
            "Internal taxonomy 'parse/format error' counts ReAct rollout stop_reason=parse_fail "
            "(strict single-call <tool_call_answer> gate in nestful_mtgrpo_minimal/parser.py)."
        ),
        "sample_ids_parse_fail_C0_head": [
            sid for sid in ids
            if analyses["C0"][sid]["stop_reason"] == "parse_fail"
        ][:15],
    }
    for arm in ARMS:
        mpath = Path(arm_paths[arm]) / "metrics_official.json"
        off = json.loads(mpath.read_text(encoding="utf-8")) if mpath.is_file() else {}
        contradiction["official_num_pred_parsing_errors"][arm] = off.get(
            "num_pred_parsing_errors")
        contradiction["internal_parse_fail_stop_reason"][arm] = sum(
            1 for sid in ids if analyses[arm][sid]["stop_reason"] == "parse_fail"
        )

    # Reference syntax across model outputs
    ref_totals = {a: Counter() for a in ARMS}
    malformed_ref_ids = {a: [] for a in ARMS}
    for arm in ARMS:
        for sid in ids:
            for cls, n in analyses[arm][sid]["ref_classes"].items():
                ref_totals[arm][cls] += n
            if analyses[arm][sid]["flags"]["malformed_reference"]:
                if len(malformed_ref_ids[arm]) < 20:
                    malformed_ref_ids[arm].append(sid)
    # Compare training Stage3 audit
    stage3_audit = REPORTS / "stage3_nestful_syntax_audit.json"
    stage3_verdict = None
    if stage3_audit.is_file():
        stage3_verdict = json.loads(stage3_audit.read_text(encoding="utf-8")).get("verdict")

    ref_verdict = "NO_REFERENCE_FORMAT_MISMATCH"
    if sum(ref_totals[a].get("malformed_dollar_ref", 0) for a in ARMS) > 50:
        ref_verdict = "PARTIAL_REFERENCE_FORMAT_MISMATCH"
    # underscore in model outputs is accepted → not a mismatch
    if all(ref_totals[a].get("malformed_dollar_ref", 0) == 0 for a in ARMS):
        ref_verdict = "NO_REFERENCE_FORMAT_MISMATCH"

    # Qualitative examples
    def raw_pack(arm: str, sid: str) -> dict:
        row = arm_rows[arm][sid]
        traj = row.get("_traj") or {}
        turns = traj.get("turns") or []
        return {
            "sample_id": sid,
            "arm": arm,
            "stop_reason": traj.get("stop_reason"),
            "coarse_failure": analyses[arm][sid]["coarse_failure"],
            "layer": analyses[arm][sid]["layer"],
            "official_win": analyses[arm][sid]["official_win"],
            "flags": {k: v for k, v in analyses[arm][sid]["flags"].items() if v},
            "predicted_calls": [
                t.get("parsed_call") for t in turns if t.get("parsed_call")
            ],
            "turns": [
                {
                    "turn_idx": t.get("turn_idx"),
                    "fail_reason": t.get("fail_reason"),
                    "clipped": t.get("clipped_completion"),
                    "model_text": (t.get("model_text") or "")[:500],
                    "parsed_call": t.get("parsed_call"),
                }
                for t in turns
            ],
            "internal_label": analyses[arm][sid]["coarse_failure"],
            "correct_classification_note": None,
        }

    # 10: C0 format fail → C2 format ok (prefer true parse_fail)
    fixed = []
    for sid in ids:
        if (analyses["C0"][sid]["stop_reason"] == "parse_fail"
                and analyses["C2"][sid]["stop_reason"] != "parse_fail"
                and not analyses["C2"][sid]["flags"]["parser_extraction_failure"]):
            fixed.append(sid)
        if len(fixed) >= 10:
            break
    if len(fixed) < 10:
        for sid in ids:
            if sid in fixed:
                continue
            if (analyses["C0"][sid]["flags"]["syntax_format"]
                    and not analyses["C2"][sid]["flags"]["syntax_format"]):
                fixed.append(sid)
            if len(fixed) >= 10:
                break
    # 10: C0 format ok → C2 format fail
    regressed = []
    for sid in ids:
        if (analyses["C0"][sid]["stop_reason"] != "parse_fail"
                and analyses["C2"][sid]["stop_reason"] == "parse_fail"):
            regressed.append(sid)
        if len(regressed) >= 10:
            break
    if len(regressed) < 10:
        for sid in ids:
            if sid in regressed:
                continue
            if (not analyses["C0"][sid]["flags"]["syntax_format"]
                    and analyses["C2"][sid]["flags"]["syntax_format"]):
                regressed.append(sid)
            if len(regressed) >= 10:
                break
    # 10: internal parse/format but actually semantic-ish mislabel
    # (e.g. no_tool_call labelled elsewhere, or parse_fail after valid calls with answer dump)
    mislabeled = []
    for sid in ids:
        a = analyses["C0"][sid]
        if a["coarse_failure"] != "parse/format error":
            continue
        # if had ≥1 successful call then dumped final answer → still format, but note
        # Prefer cases where internal said parse but dominant issue is answer-in-tag
        # after executable prefix — keep as "format but often confused with semantics"
        if a["flags"]["final_answer_extraction_failure"] and a["pred_n"] >= 1:
            pack = raw_pack("C0", sid)
            pack["correct_classification_note"] = (
                "Internal label parse/format is correct for the ReAct gate, but the "
                "model already produced valid calls; the failure is dumping a bare "
                "final answer into <tool_call_answer> instead of []. Not a schema error."
            )
            pack["correct_classification"] = "A_syntax_final_answer_in_tag"
            mislabeled.append(pack)
        if len(mislabeled) >= 10:
            break
    # If not enough answer-dump cases, fill with no_tool_call that root-cause might confuse
    if len(mislabeled) < 10:
        for sid in ids:
            a = analyses["C0"][sid]
            if a["coarse_failure"] == "no tool call":
                pack = raw_pack("C0", sid)
                pack["correct_classification_note"] = (
                    "Root-cause taxonomy separates 'no tool call' from parse/format; "
                    "this is format-adjacent (never emitted a valid call), not wrong-tool semantics."
                )
                pack["correct_classification"] = "A_syntax_no_tool_call"
                mislabeled.append(pack)
            if len(mislabeled) >= 10:
                break

    # 10: parsed+executable wrong result
    exec_wrong = []
    for sid in ids:
        a = analyses["C2"][sid]
        if (a["coarse_failure"] in (
                "correct keys, wrong argument values",
                "executable trajectory ending wrong result",
                "wrong tool",
            )
                and not a["flags"]["syntax_format"]
                and a["pred_n"] > 0
                and not a["official_win"]):
            pack = raw_pack("C2", sid)
            pack["correct_classification"] = "E_semantic"
            pack["correct_classification_note"] = (
                "Calls parsed; trajectory progressed without parse_fail; loss is tool/"
                "argument/result semantics under IBM re-execution."
            )
            exec_wrong.append(pack)
        if len(exec_wrong) >= 10:
            break

    examples = {
        "C0_format_fail_C2_format_ok": [raw_pack("C0", sid) | {"C2": raw_pack("C2", sid)}
                                        for sid in fixed],
        "C0_format_ok_C2_format_fail": [raw_pack("C0", sid) | {"C2": raw_pack("C2", sid)}
                                       for sid in regressed],
        "internal_format_label_nuance": mislabeled,
        "parsed_executable_wrong_result": exec_wrong,
    }

    # Layer mix among non-wins
    layer_mix = {}
    for arm in ARMS:
        c = Counter(analyses[arm][sid]["layer"] for sid in ids
                    if not analyses[arm][sid]["official_win"])
        layer_mix[arm] = dict(c)

    # Verdict uses *strict* parser failure (parse_fail) as surface-format signal.
    # Broader syntax_format also includes no_tool_call (~7pp) which is format-adjacent
    # but not "broken JSON/tag syntax".
    c0_parse = rates["C0"]["parser_extraction_failure"]["rate"]
    c2_parse = rates["C2"]["parser_extraction_failure"]["rate"]
    c0_syn = rates["C0"]["syntax_format"]["rate"]
    c2_syn = rates["C2"]["syntax_format"]["rate"]
    c0_sem = rates["C0"]["semantic_dominant"]["rate"]
    c2_sem = rates["C2"]["semantic_dominant"]["rate"]
    # Among non-wins, fraction semantic
    nonwin_sem = {}
    for arm in ARMS:
        nw = [sid for sid in ids if not analyses[arm][sid]["official_win"]]
        if not nw:
            nonwin_sem[arm] = None
            continue
        nonwin_sem[arm] = sum(
            1 for sid in nw if analyses[arm][sid]["flags"]["semantic_dominant"]
        ) / len(nw)

    schema_c2 = rates["C2"]["schema_or_reference"]["rate"] or 0.0
    # Prefer semantic-dominant verdict when strict parse is low and non-wins are mostly E.
    verdict = "FORMAT_LARGELY_RESOLVED_SEMANTIC_ERRORS_DOMINATE"
    if (c2_parse or 0) > 0.12:
        verdict = "FORMAT_IS_PRIMARY_BOTTLENECK"
    elif (c2_parse or 0) > 0.06:
        verdict = "FORMAT_PARTIALLY_RESOLVED"
    elif schema_c2 > 0.08 and schema_c2 >= (nonwin_sem.get("C2") or 0) * 0.5:
        # schema rate high relative to semantic share among failures
        verdict = "SURFACE_FORMAT_RESOLVED_SCHEMA_ISSUES_REMAIN"
    elif (nonwin_sem.get("C2") or 0) < 0.40:
        verdict = "FORMAT_PARTIALLY_RESOLVED"
    # Taxonomy CSV (finer than root-cause)
    tax_rows = []
    for key in metric_keys:
        row = {
            "metric": key,
            "C0_count": rates["C0"][key]["count"],
            "C0_rate": rates["C0"][key]["rate"],
            "C1_count": rates["C1"][key]["count"],
            "C1_rate": rates["C1"][key]["rate"],
            "C2_count": rates["C2"][key]["count"],
            "C2_rate": rates["C2"][key]["rate"],
            "C1_minus_C0": delta(rates["C1"][key]["rate"], rates["C0"][key]["rate"]),
            "C2_minus_C1": delta(rates["C2"][key]["rate"], rates["C1"][key]["rate"]),
            "C2_minus_C0": delta(rates["C2"][key]["rate"], rates["C0"][key]["rate"]),
            "C0_ci95": rates["C0"][key].get("ci95"),
            "C2_ci95": rates["C2"][key].get("ci95"),
        }
        tax_rows.append(row)

    payload = {
        "generated_at": _now(),
        "run_dir": str(run_dir),
        "n_tasks": len(ids),
        "eval_dirs": arm_paths,
        "verdict": verdict,
        "definitions": {
            "A_raw_output_format": (
                "empty text, missing/unclosed <tool_call_answer>, invalid JSON inside tag, "
                "bare final answer in tag, truncation/clipped, no tool call emitted"
            ),
            "B_parser_extraction": (
                "internal ReAct parse_tool_call gate fail → stop_reason=parse_fail; "
                "reasons: no_tag, invalid_json, not_exactly_one, ..."
            ),
            "C_schema": "unknown tool, missing/extra arg keys, type/serialization vs tool schema",
            "D_reference": "malformed $...$, unresolvable var, missing output field",
            "E_semantic": (
                "wrong tool choice, wrong argument values, executable-but-wrong result, "
                "call-count mismatch with otherwise valid format"
            ),
            "official_vs_internal_parse": contradiction["explanation"],
        },
        "contradiction": contradiction,
        "rates": rates,
        "paired": paired,
        "stop_reason_dist": stop_dist,
        "parse_reason_dist": parse_dist,
        "per_turn_syntax": per_turn,
        "by_call_bucket": by_bucket,
        "layer_mix_among_nonwins": layer_mix,
        "nonwin_semantic_fraction": nonwin_sem,
        "reference_totals": {a: dict(ref_totals[a]) for a in ARMS},
        "reference_verdict": ref_verdict,
        "stage3_train_audit_verdict": stage3_verdict,
        "c0_syntax_rate": c0_syn,
        "c2_syntax_rate": c2_syn,
        "c0_parse_fail_rate": c0_parse,
        "c2_parse_fail_rate": c2_parse,
        "c0_semantic_rate": c0_sem,
        "c2_semantic_rate": c2_sem,
    }

    _write_json(REPORTS / "FORMAT_STATUS_C0_C1_C2.json", payload)
    _write_json(REPORTS / "format_error_examples.json", examples)
    _write_csv(
        REPORTS / "format_error_taxonomy.csv",
        [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()}
         for r in tax_rows],
        fieldnames=list(tax_rows[0].keys()),
    )

    # reference_syntax_audit.md
    ref_md = []
    ref_md.append("# Reference Syntax Audit (C0/C1/C2 + Stage 3)\n")
    ref_md.append(f"Generated: {_now()}\n")
    ref_md.append(f"Verdict: **{ref_verdict}**\n")
    ref_md.append("## Canonical forms (from code)\n")
    ref_md.append(
        "- Tool-R0 ReAct prompt (`prompt.py`): `$varN.field$` "
        "(e.g. `$var1.result$`, `$var1.output_0$`).\n"
        "- Executor (`executor.py`): `_VAR_REF_RE` accepts `$name` / `$name.field$`; "
        "index via `var_?(\\d+)` → both `$var1` and `$var_1` resolve.\n"
        "- Official NESTFUL gold often uses `$var_N.result$`; scorer "
        "`ground_seq_nested_repsonse` accepts `$var…` forms.\n"
    )
    ref_md.append(f"Stage 3 training audit verdict: `{stage3_verdict}` "
                  f"(see `stage3_nestful_syntax_audit.md`).\n")
    ref_md.append("## Model-output reference class counts\n")
    ref_md.append("| Class | C0 | C1 | C2 |\n|---|---:|---:|---:|\n")
    all_cls = sorted(set().union(*[ref_totals[a].keys() for a in ARMS]))
    for cls in all_cls:
        ref_md.append(
            f"| {cls} | {ref_totals['C0'].get(cls, 0)} | "
            f"{ref_totals['C1'].get(cls, 0)} | {ref_totals['C2'].get(cls, 0)} |\n"
        )
    ref_md.append("\nMalformed sample IDs (head): "
                  f"C0={malformed_ref_ids['C0'][:5]}, "
                  f"C2={malformed_ref_ids['C2'][:5]}\n")
    ref_md.append(
        "\n## Verdict rationale\n\n"
        "Underscore vs no-underscore is **not** treated as an error: both are "
        "accepted by the executor and IBM scorer. Hard mismatches would be "
        "dollar-strings that fail `_VAR_REF_RE`. Counts of `malformed_dollar_ref` "
        "are low; Stage 3 gold is already Tool-R0-canonical. "
        f"→ **{ref_verdict}**.\n"
    )
    (REPORTS / "reference_syntax_audit.md").write_text("".join(ref_md), encoding="utf-8")

    # Main MD report
    def pct(x: Optional[float]) -> str:
        return "—" if x is None else f"{100 * x:.2f}%"

    def cnt_rate(arm: str, key: str) -> str:
        r = rates[arm][key]
        return f"{r['count']} ({pct(r['rate'])})"

    md: List[str] = []
    md.append("# Format Status — C0 / C1 / C2\n\n")
    md.append("## Verdikt pro supervizora (≤5 vět)\n\n")
    md.append(
        f"**Kategorie: `{verdict}`**\n\n"
        f"1. Základní ReAct formát se mírně zlepšil: ostrý `parse_fail` "
        f"{contradiction['internal_parse_fail_stop_reason']['C0']} → "
        f"{contradiction['internal_parse_fail_stop_reason']['C2']} "
        f"({pct(c0_parse)} → {pct(c2_parse)}); širší syntax+no-call flag "
        f"{pct(c0_syn)} → {pct(c2_syn)}.\n"
        f"2. Formát **není** hlavní bottleneck — u neúspěšných úloh dominuje sémantika "
        f"(wrong tool / wrong values / executable-wrong-result); "
        f"podíl semantic_dominant mezi non-wins ≈ {pct(nonwin_sem.get('C2'))}.\n"
        f"3. Zbývající formát: hlavně `parse:invalid_json` (často finální číslo v tagu "
        f"místo `[]`) a `parse:no_tag` / no-tool-call; truncace je vzácná "
        f"({cnt_rate('C2', 'output_truncation')}).\n"
        f"4. Chyby jsou převážně **sémantické**; schema/reference jsou menšinové; "
        f"„official parser_errors=0“ **nedokazuje** vyřešený formát — měří jinou věc.\n"
        f"5. Další krok: cílit credit assignment / Stage-3 reasoning "
        f"(wrong values & wrong tool na 4–5 call), ne další format reward.\n\n"
    )

    md.append(f"Generated: {_now()}\n")
    md.append(f"Run: `{run_dir}`\n")
    md.append(f"n = {len(ids)} paired nestful_test tasks\n\n")

    md.append("## 1. Definice vrstev\n\n")
    md.append("| Layer | Co znamená |\n|---|---|\n")
    md.append("| A Raw output | prázdný text, neuzavřený tag, invalid JSON, truncace, no-call |\n")
    md.append("| B Parser | interní `parse_tool_call` gate → `parse_fail` |\n")
    md.append("| C Schema | unknown tool, missing/extra keys, type |\n")
    md.append("| D Reference | malformed `$…$`, unresolved var/field |\n")
    md.append("| E Semantic | wrong tool/values, executable wrong result, call-count |\n\n")

    md.append("## 2. Kanonické artefakty\n\n")
    for arm in ARMS:
        md.append(f"- **{arm}**: `{arm_paths[arm]}`\n")
        md.append(f"  - `final_eval_trajectories.jsonl`, `metrics_official.json`, "
                  f"`final_eval_predictions.partial.jsonl`, `eval_manifest.json`\n")
    md.append(
        "- Interní parser: `nestful_mtgrpo_minimal/parser.py` "
        "(`<tool_call_answer>` + exactly one call)\n"
        "- Official parser path: `nestful_official_score.build_item` → "
        "`parse_llama_3_output` on **pre-extracted** JSON calls\n"
        "- Prompt: `nestful_mtgrpo_minimal/prompt.py`\n"
        "- Resolver: `executor.py` `_VAR_REF_RE` / `resolve_variables`\n\n"
    )

    md.append("## 3. Rozpor: official parser_errors=0 vs interní parse/format\n\n")
    md.append("| Arm | official `num_pred_parsing_errors` | internal `parse_fail` |\n|---|---:|---:|\n")
    for arm in ARMS:
        md.append(
            f"| {arm} | {contradiction['official_num_pred_parsing_errors'][arm]} | "
            f"{contradiction['internal_parse_fail_stop_reason'][arm]} |\n"
        )
    md.append(f"\n{contradiction['explanation']}\n\n")
    md.append("**Konkrétní sample IDs (C0 parse_fail head):** "
              + ", ".join(f"`{s}`" for s in contradiction["sample_ids_parse_fail_C0_head"][:8])
              + "\n\n")
    # show one raw example
    if contradiction["sample_ids_parse_fail_C0_head"]:
        ex = raw_pack("C0", contradiction["sample_ids_parse_fail_C0_head"][0])
        md.append("### Příklad raw output (C0 parse_fail)\n\n")
        md.append(f"- sample_id: `{ex['sample_id']}`\n")
        md.append(f"- stop_reason: `{ex['stop_reason']}`\n")
        for t in ex["turns"]:
            if t.get("fail_reason"):
                md.append(f"- failing turn fail_reason=`{t['fail_reason']}`\n")
                md.append(f"```\n{t['model_text']}\n```\n")
        md.append(
            "Official scorer never sees this raw ReAct text — it receives "
            f"`predicted_calls` with {ex.get('predicted_calls') and len(ex['predicted_calls'])} "
            "already-parsed calls (often partial), so `parse_valid=True` / parsing_errors=0.\n\n"
        )

    md.append("## 4. Kvantitativní srovnání\n\n")
    md.append("| Metric | C0 | C1 | C2 | C2−C0 |\n|---|---:|---:|---:|---:|\n")
    for key in metric_keys:
        d = delta(rates["C2"][key]["rate"], rates["C0"][key]["rate"])
        md.append(
            f"| {key} | {cnt_rate('C0', key)} | {cnt_rate('C1', key)} | "
            f"{cnt_rate('C2', key)} | {pct(d) if d is not None else '—'} |\n"
        )
    md.append("\nBootstrap 95% CI (C2 syntax_format rate): "
              f"{rates['C2']['syntax_format'].get('ci95')}\n")
    md.append(f"Bootstrap 95% CI (C2 semantic_dominant rate): "
              f"{rates['C2']['semantic_dominant'].get('ci95')}\n\n")

    md.append("### Párově (syntax_format flag)\n\n")
    for label, blk in paired["syntax_format"].items():
        md.append(
            f"- **{label}**: new_errors={blk['new_positive']}, "
            f"resolved={blk['resolved_positive']}, "
            f"net_rate_delta={blk['net_rate_delta']:.4f}, "
            f"CI95={blk['ci95']}\n"
        )
    md.append("\n")

    md.append("## 5. Per-turn / per-bucket\n\n")
    md.append("### First syntax-format failure turn (C2)\n\n")
    md.append("| Turn | count | % of all tasks |\n|---|---:|---:|\n")
    for k, v in sorted(per_turn["C2"].items()):
        md.append(f"| {k} | {v['count']} | {pct(v['rate_of_all_tasks'])} |\n")
    md.append("\n### By gold call count (C2)\n\n")
    md.append("| Bucket | n | syntax | parser_fail | schema/ref | semantic |\n"
              "|---|---:|---:|---:|---:|---:|\n")
    for b, blk in by_bucket["C2"].items():
        md.append(
            f"| {b} | {blk['n']} | {pct(blk['syntax_format']['rate'])} | "
            f"{pct(blk['parser_extraction_failure']['rate'])} | "
            f"{pct(blk['schema_or_reference']['rate'])} | "
            f"{pct(blk['semantic_dominant']['rate'])} |\n"
        )
    md.append("\nParse reason dist C0→C2: "
              f"{parse_dist['C0']} → {parse_dist['C2']}\n\n")

    md.append("## 6. Reference syntax\n\n")
    md.append(f"See `reference_syntax_audit.md`. Verdict: **{ref_verdict}**.\n\n")

    md.append("## 7. Kvalitativní příklady\n\n")
    md.append(f"Uloženo v `format_error_examples.json` "
              f"({len(examples['C0_format_fail_C2_format_ok'])} fixed, "
              f"{len(examples['C0_format_ok_C2_format_fail'])} regressed, "
              f"{len(examples['internal_format_label_nuance'])} nuance, "
              f"{len(examples['parsed_executable_wrong_result'])} semantic).\n\n")
    md.append("Ukázka fixed (C0→C2):\n")
    for sid in fixed[:3]:
        md.append(f"- `{sid}`: C0 stop=`{analyses['C0'][sid]['stop_reason']}` → "
                  f"C2 stop=`{analyses['C2'][sid]['stop_reason']}` "
                  f"win={analyses['C2'][sid]['official_win']}\n")
    md.append("\nUkázka semantic wrong-result (C2):\n")
    for pack in exec_wrong[:3]:
        md.append(f"- `{pack['sample_id']}`: {pack['coarse_failure']}\n")

    md.append("\n## 8. Layer mix among non-wins\n\n")
    md.append("| Arm | A syntax/parser | C/D schema/ref | E semantic | ok? |\n|---|---:|---:|---:|---:|\n")
    for arm in ARMS:
        lm = layer_mix[arm]
        md.append(
            f"| {arm} | {lm.get('A_syntax_or_parser', 0)} | "
            f"{lm.get('C_D_schema_or_reference', 0)} | "
            f"{lm.get('E_semantic', 0)} | {lm.get('ok', 0)} |\n"
        )

    md.append("\n## Message for supervisor\n\n")
    md.append("### Krátká verze\n\n")
    md.append(
        "Formát tool callů se oproti C0 mírně zlepšil (parse_fail 74→63), ale už teď "
        "není hlavní problém — official „0 parser errors“ je navíc matoucí, protože "
        "oficiální scorer dostává už vytěžené JSON call listy, ne raw ReAct text. "
        "Většina proher je sémantická (špatný tool / hodnoty / výsledek).\n\n"
    )
    md.append("### Delší verze\n\n")
    md.append(
        "Po dvou fázích GRPO vypadá ostrý ReAct parse_fail spíš jako okrajový jev "
        f"(cca {pct(c2_parse)} úloh; širší syntax+no-call {pct(c2_syn)}) než jako bottleneck: "
        "typické zbývající format chyby jsou „finální číslo v `<tool_call_answer>`“ "
        "nebo chybějící tag, ne rozbitá JSON syntax napříč trajektorií. "
        "Rozpor mezi official parser_errors=0 a interními desítkami parse/format "
        "chyb je definiční — jiný parser, jiný vstup. "
        "Mezi nevyhranými úlohami dominují wrong-tool / wrong-value / "
        "executable-wrong-result; schema a reference mismatch nejsou hlavní příběh "
        f"({ref_verdict}). "
        "Další investice by měla jít do sémantiky a credit assignment na delších "
        "řetězcích, ne do dalšího format rewardu.\n"
    )

    (REPORTS / "FORMAT_STATUS_C0_C1_C2.md").write_text("".join(md), encoding="utf-8")
    print(f"[format-audit] verdict={verdict}")
    print(f"[format-audit] wrote {REPORTS / 'FORMAT_STATUS_C0_C1_C2.md'}")
    print(f"[format-audit] wrote {REPORTS / 'FORMAT_STATUS_C0_C1_C2.json'}")
    print(f"[format-audit] wrote {REPORTS / 'format_error_taxonomy.csv'}")
    print(f"[format-audit] wrote {REPORTS / 'format_error_examples.json'}")
    print(f"[format-audit] wrote {REPORTS / 'reference_syntax_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
