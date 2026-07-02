#!/usr/bin/env python3
"""Clean / repair the synthetic curriculum training set (no reward change).

Goal (see docs/STABILIZED_CURRICULUM_PLAN.md):
  - Produce a clean/repaired variant of the per-stage curriculum JSONL files.
  - Remove ONLY objectively invalid / non-executable examples.
  - NEVER drop hard examples just because the model cannot solve them.

Removal criteria (objective only):
  - unparsable JSONL line
  - missing essential field: input/question, tools, output/gold trace, gold_answer
  - gold trace not parseable as a list of tool calls (or empty)
  - a tool call uses a function not present in the row's `tools`
  - a call provides an argument name not declared by the tool schema
  - a call to a tool with parameters provides zero arguments (missing required)
  - a `$var...` reference is invalid (self / future / out-of-range / wrong field)
  - gold execution fails (only when an IBM executor is available)
  - gold_answer holds an unresolved `$var...` reference that execution cannot fix

Safe repairs:
  - `tools` / `output` stored as JSON strings -> parsed + stored normalized
  - `gold_answer` with unresolved `$var...` -> replaced with the concrete value
    obtained by executing the gold trace (only when executor available)

Never removed:
  - long / hard / many-call tasks
  - tasks the baseline/model cannot solve
  - tasks with an alternative valid path, as long as the gold trace is valid

Outputs (default under nestful_mtgrpo_minimal/data/clean_curriculum/):
  - epoch_<N>_<N>call.jsonl        (clean rows, same schema as input)
  - CLEANING_REPORT.md
  - removed_examples.csv
  - repaired_examples.csv
  - validation_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_MIN = os.path.join(_REPO, "experiments", "nestful_mtgrpo_minimal")
_DEFAULT_IN = os.path.join(_MIN, "data", "filtered_toolr0_synthetic")
_DEFAULT_OUT = os.path.join(_MIN, "data", "clean_curriculum")

for _p in (_MIN, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the training-time normalizers so the cleaned rows match what the loader
# (data.py) will see at train time.
from data import (  # noqa: E402
    _coerce_jsonish,
    _normalize_calls,
    _normalize_tool_schema,
)

# Reference syntax accepted by the executor/scorer:
#   $var1.result$ / $var_1.result$ / $var1.output_0$ / $var_1.output_0$ / $var_1$
_VAR_REF_RE = re.compile(r"^\$([A-Za-z_]*?_?)(\d+)(?:\.([A-Za-z_][\w]*))?\$$")
_VAR_ANYWHERE_RE = re.compile(r"\$[A-Za-z_]*?_?\d+(?:\.[A-Za-z_][\w]*)?\$")

_INPUT_FIELDS = ("input", "prompt", "query", "question")
_OUTPUT_FIELDS = ("gold_output", "gold_outputs", "output", "gold_calls")
_ANSWER_FIELDS = ("gold_answer", "answer", "final_answer")
_ID_FIELDS = ("sample_id", "task_id", "id")


# ── Optional IBM gold executor (best-effort; structural checks run regardless) ──
def _try_get_executor():
    try:
        from curricullum.data.exec_trajectory import (  # noqa: E402
            execute_trajectory,
            get_ibm_registry,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[clean] IBM executor import failed ({type(exc).__name__}: {exc}); "
              "execution checks/repairs disabled.", flush=True)
        return None, None, None
    try:
        registry = get_ibm_registry()
    except Exception as exc:  # noqa: BLE001
        print(f"[clean] get_ibm_registry failed ({type(exc).__name__}: {exc}); "
              "execution checks/repairs disabled.", flush=True)
        return None, None, None
    if registry is None:
        print("[clean] IBM registry unavailable; execution checks/repairs disabled.", flush=True)
        return None, None, None

    # _matches_gold for verifying executed value vs declared gold_answer.
    try:
        from nestful_evaluation.run import _matches_gold  # noqa: E402
    except Exception:  # noqa: BLE001
        _matches_gold = None
    return execute_trajectory, registry, _matches_gold


def _first(row: Dict[str, Any], fields) -> Any:
    for f in fields:
        if f in row and row[f] is not None:
            return row[f]
    return None


def _output_param_keys(tool: Dict[str, Any]) -> List[str]:
    op = tool.get("output_parameters")
    if isinstance(op, dict):
        return list(op.keys())
    return []


def _declared_param_names(tool: Dict[str, Any]) -> List[str]:
    params = tool.get("parameters", {})
    if isinstance(params, dict) and "properties" in params:
        props = params.get("properties", {})
        return list(props.keys()) if isinstance(props, dict) else []
    if isinstance(params, dict):
        return list(params.keys())
    return []


def _refs_in_value(value: Any) -> List[Tuple[int, Optional[str]]]:
    out: List[Tuple[int, Optional[str]]] = []
    if isinstance(value, str):
        m = _VAR_REF_RE.match(value.strip())
        if m:
            out.append((int(m.group(2)), m.group(3)))
    elif isinstance(value, list):
        for v in value:
            out.extend(_refs_in_value(v))
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(_refs_in_value(v))
    return out


def _validate_calls_structural(
    calls: List[Dict[str, Any]], tools: List[Dict[str, Any]]
) -> Tuple[bool, str]:
    """Objective structural validation: tool existence, arg names, references."""
    toolmap = {t.get("name"): t for t in tools if isinstance(t.get("name"), str)}
    if not calls:
        return False, "empty_gold_trace"
    for i, c in enumerate(calls, start=1):
        name = c.get("name")
        args = c.get("arguments")
        if not isinstance(name, str) or name not in toolmap:
            return False, "tool_not_available"
        if not isinstance(args, dict):
            return False, "call_args_not_object"
        tool = toolmap[name]
        declared = set(_declared_param_names(tool))
        if declared and len(args) == 0:
            return False, "missing_required_args"
        for arg_name in args:
            if declared and arg_name not in declared:
                return False, "unknown_arg"
    # Reference validity.
    for call_idx, c in enumerate(calls, start=1):
        args = c.get("arguments", {})
        if not isinstance(args, dict):
            continue
        for v in args.values():
            for ref_idx, field in _refs_in_value(v):
                if ref_idx == call_idx:
                    return False, "invalid_reference_self"
                if ref_idx > call_idx:
                    return False, "invalid_reference_future"
                if ref_idx < 1 or ref_idx > len(calls):
                    return False, "invalid_reference_range"
                ref_tool = toolmap.get(calls[ref_idx - 1].get("name", ""))
                if ref_tool is None:
                    return False, "invalid_reference"
                out_params = _output_param_keys(ref_tool)
                if field is not None and out_params and field not in out_params:
                    return False, "invalid_reference_field"
    return True, "ok"


def _gold_answer_has_unresolved_ref(gold_answer: Any) -> bool:
    if gold_answer is None:
        return False
    try:
        s = gold_answer if isinstance(gold_answer, str) else json.dumps(gold_answer)
    except (TypeError, ValueError):
        s = str(gold_answer)
    return bool(_VAR_ANYWHERE_RE.search(s))


def clean_row(
    raw_line: str,
    idx: int,
    executor=None,
    registry=None,
    matches_gold=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], List[str]]:
    """Returns (clean_row | None, removal_reason | None, repairs_applied)."""
    repairs: List[str] = []

    try:
        row = json.loads(raw_line)
    except (json.JSONDecodeError, TypeError):
        return None, "unparsable_json", repairs
    if not isinstance(row, dict):
        return None, "row_not_object", repairs

    question = _first(row, _INPUT_FIELDS)
    tools_raw = _first(row, ("tools",))
    output_raw = _first(row, _OUTPUT_FIELDS)
    answer_raw = _first(row, _ANSWER_FIELDS)

    missing = []
    if question is None:
        missing.append("input")
    if tools_raw is None:
        missing.append("tools")
    if output_raw is None:
        missing.append("output")
    if answer_raw is None:
        missing.append("gold_answer")
    if missing:
        return None, f"missing_field:{'+'.join(missing)}", repairs

    # Parse tools / output (repair: JSON-string -> native).
    if isinstance(tools_raw, str):
        repairs.append("parsed_tools_json")
    tools = _normalize_tool_schema(tools_raw)
    if not tools:
        return None, "tools_unparsable", repairs

    if isinstance(output_raw, str):
        repairs.append("parsed_output_json")
    parsed_output = _coerce_jsonish(output_raw)
    if not isinstance(parsed_output, list):
        return None, "gold_trace_not_list", repairs
    calls = _normalize_calls(output_raw)

    ok, reason = _validate_calls_structural(calls, tools)
    if not ok:
        return None, reason, repairs

    # Gold answer coercion.
    gold_answer = _coerce_jsonish(answer_raw)

    # Execution-based checks / repairs (best-effort).
    if executor is not None and registry is not None:
        try:
            final_value, _traces, err = executor(calls, ibm_registry=registry)
        except Exception as exc:  # noqa: BLE001
            return None, f"gold_execution_error:{type(exc).__name__}", repairs
        if err:
            return None, f"gold_execution_failed:{err}", repairs

        unresolved = _gold_answer_has_unresolved_ref(gold_answer)
        if unresolved:
            # Only accept the executed value if it is itself fully resolved.
            # Some traces short-circuit and hand back a value that STILL contains
            # `$var...` references — that is effectively non-resolvable → remove.
            if final_value is None or _gold_answer_has_unresolved_ref(final_value):
                return None, "gold_answer_unresolved", repairs
            gold_answer = final_value
            repairs.append("resolved_gold_answer")
        elif matches_gold is not None:
            # Keep declared gold_answer; only flag (never remove) on mismatch
            # if it already executes — declared value may use different rounding.
            pass
    else:
        # No executor: cannot resolve unresolved refs -> objective removal.
        if _gold_answer_has_unresolved_ref(gold_answer):
            return None, "gold_answer_unresolved_no_executor", repairs

    # Rebuild a normalized clean row (preserve id; store native objects).
    clean: "OrderedDict[str, Any]" = OrderedDict()
    sid = _first(row, _ID_FIELDS) or f"task_{idx}"
    clean["sample_id"] = str(sid)
    clean["input"] = str(question)
    clean["tools"] = tools
    clean["output"] = calls
    clean["gold_answer"] = gold_answer
    return clean, None, repairs


def _discover_stage_files(in_dir: str, stages: List[int]) -> "OrderedDict[int, str]":
    found: "OrderedDict[int, str]" = OrderedDict()
    for n in stages:
        path = os.path.join(in_dir, f"epoch_{n}_{n}call.jsonl")
        if os.path.isfile(path):
            found[n] = path
        else:
            print(f"[clean] stage {n}: file not found, skipping: {path}", flush=True)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_dir", default=_DEFAULT_IN, help="dir with epoch_N_Ncall.jsonl files")
    ap.add_argument("--out_dir", default=_DEFAULT_OUT, help="output dir for clean files + report")
    ap.add_argument("--stages", default="1,2,3,4,5,6", help="comma-separated stage numbers")
    ap.add_argument("--no_exec", action="store_true", help="disable IBM gold execution checks/repairs")
    args = ap.parse_args()

    stages = [int(s) for s in args.stages.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    executor = registry = matches_gold = None
    if not args.no_exec:
        executor, registry, matches_gold = _try_get_executor()
    exec_available = executor is not None and registry is not None

    stage_files = _discover_stage_files(args.in_dir, stages)
    if not stage_files:
        print(f"[clean] ERROR: no stage files found in {args.in_dir}", file=sys.stderr)
        return 1

    removed_rows: List[Dict[str, Any]] = []
    repaired_rows: List[Dict[str, Any]] = []
    per_stage: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()

    for n, path in stage_files.items():
        kept = 0
        total = 0
        remove_reasons: Counter = Counter()
        repair_reasons: Counter = Counter()
        out_path = os.path.join(args.out_dir, f"epoch_{n}_{n}call.jsonl")
        with open(path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for idx, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue
                total += 1
                clean, reason, repairs = clean_row(
                    line, idx, executor=executor, registry=registry, matches_gold=matches_gold,
                )
                if clean is None:
                    remove_reasons[reason] += 1
                    removed_rows.append({
                        "stage": n, "line_index": idx, "reason": reason,
                        "sample_id": _safe_sid(line),
                    })
                    continue
                if repairs:
                    for r in repairs:
                        repair_reasons[r] += 1
                    repaired_rows.append({
                        "stage": n, "line_index": idx,
                        "sample_id": clean.get("sample_id", ""),
                        "repairs": ";".join(repairs),
                    })
                fout.write(json.dumps(_json_safe(clean), ensure_ascii=False) + "\n")
                kept += 1
        per_stage[n] = {
            "input_file": path,
            "output_file": out_path,
            "total": total,
            "kept": kept,
            "removed": total - kept,
            "remove_reasons": dict(remove_reasons),
            "repair_reasons": dict(repair_reasons),
        }
        print(f"[clean] stage {n}: total={total} kept={kept} removed={total - kept}", flush=True)

    _write_csv(os.path.join(args.out_dir, "removed_examples.csv"),
               ["stage", "line_index", "sample_id", "reason"], removed_rows)
    _write_csv(os.path.join(args.out_dir, "repaired_examples.csv"),
               ["stage", "line_index", "sample_id", "repairs"], repaired_rows)

    summary = {
        "exec_available": exec_available,
        "in_dir": args.in_dir,
        "out_dir": args.out_dir,
        "stages": list(stage_files.keys()),
        "per_stage": per_stage,
        "totals": {
            "total": sum(s["total"] for s in per_stage.values()),
            "kept": sum(s["kept"] for s in per_stage.values()),
            "removed": sum(s["removed"] for s in per_stage.values()),
            "repaired": len(repaired_rows),
        },
    }
    with open(os.path.join(args.out_dir, "validation_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    _write_report(os.path.join(args.out_dir, "CLEANING_REPORT.md"), summary, exec_available)

    t = summary["totals"]
    print(f"[clean] DONE total={t['total']} kept={t['kept']} removed={t['removed']} "
          f"repaired={t['repaired']} exec_available={exec_available}", flush=True)
    print(f"[clean] outputs in {args.out_dir}", flush=True)
    return 0


def _json_safe(value: Any) -> Any:
    """Recursively coerce a value into a JSON-serializable structure.

    The IBM gold executor can return bytes / tuples / exotic objects; gold_answer
    repaired from execution must still serialize cleanly into the clean JSONL.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return _json_safe(value.tolist())
    except ImportError:
        pass
    return str(value)


def _safe_sid(line: str) -> str:
    try:
        row = json.loads(line)
        return str(row.get("sample_id") or row.get("task_id") or row.get("id") or "")
    except (json.JSONDecodeError, TypeError):
        return ""


def _write_csv(path: str, cols: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_report(path: str, summary: Dict[str, Any], exec_available: bool) -> None:
    lines = [
        "# Clean curriculum — cleaning report",
        "",
        f"- Vstup: `{summary['in_dir']}`",
        f"- Výstup: `{summary['out_dir']}`",
        f"- IBM gold executor dostupný: **{exec_available}** "
        + ("(spuštěny i execution checks/repairs)" if exec_available
           else "(jen strukturální kontroly; unresolved `$var` v gold_answer → odstraněno)"),
        "",
        "## Zásada",
        "",
        "Odstraňují se **pouze objektivně vadné / neexekuovatelné** příklady. "
        "**Neodstraňují se** těžké, dlouhé ani multi-call úlohy, ani úlohy, které model neumí, "
        "ani úlohy s alternativní cestou — pokud je gold trace validní a spustitelná.",
        "",
        "## Počty po stage",
        "",
        "| stage | total | kept | removed | repaired |",
        "|-------|-------|------|---------|----------|",
    ]
    for n, s in summary["per_stage"].items():
        rep = sum(s["repair_reasons"].values())
        lines.append(f"| {n} | {s['total']} | {s['kept']} | {s['removed']} | {rep} |")
    t = summary["totals"]
    lines.append(f"| **Σ** | **{t['total']}** | **{t['kept']}** | **{t['removed']}** | **{t['repaired']}** |")

    lines += ["", "## Důvody odstranění (per stage)", ""]
    for n, s in summary["per_stage"].items():
        if not s["remove_reasons"]:
            continue
        lines.append(f"### stage {n}")
        for reason, cnt in sorted(s["remove_reasons"].items(), key=lambda x: -x[1]):
            lines.append(f"- `{reason}`: {cnt}")
        lines.append("")

    lines += ["## Provedené opravy (per stage)", ""]
    any_repair = False
    for n, s in summary["per_stage"].items():
        if not s["repair_reasons"]:
            continue
        any_repair = True
        lines.append(f"### stage {n}")
        for reason, cnt in sorted(s["repair_reasons"].items(), key=lambda x: -x[1]):
            lines.append(f"- `{reason}`: {cnt}")
        lines.append("")
    if not any_repair:
        lines.append("- (žádné opravy nebyly potřeba)")
        lines.append("")

    lines += [
        "## Detaily",
        "",
        "- `removed_examples.csv` — odstraněné příklady (stage, line_index, sample_id, reason).",
        "- `repaired_examples.csv` — opravené příklady (stage, line_index, sample_id, repairs).",
        "- `validation_summary.json` — strojově čitelný souhrn.",
        "",
        "Reward se touto úpravou **nemění**; jde pouze o čistotu trénovacích dat.",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
