#!/usr/bin/env python3
"""Audit Stage 3 reference syntax vs Tool-R0 / NESTFUL implementations.

Canonical sources (code, not paper):
  * Tool-R0 ReAct prompt: ``nestful_mtgrpo_minimal/prompt.py``
    → teaches ``$varN.field$`` (no underscore), e.g. ``$var1.result$``.
  * Synthetic/IBM executor: ``nestful_mtgrpo_minimal/executor.py``
    → ``_VAR_REF_RE`` accepts ``$name`` or ``$name.field$``; index lookup
    via ``var_?(\\d+)`` so both ``$var1`` and ``$var_1`` resolve.
  * Official NESTFUL scorer: accepts ``$…$`` / ``$var…`` forms; gold data
    predominantly uses ``$var_N.result$`` (underscore).

This audit checks every gold reference in the pure Stage 3 dataset against
the Tool-R0 training stack (prompt + synthetic executor). A *stylistic*
difference vs NESTFUL gold underscore labels is reported but is NOT a
hard mismatch for this pipeline, because eval uses the Tool-R0 ReAct prompt
and the official scorer accepts both forms.

Verdicts:
  MISMATCH_CONFIRMED — refs fail Tool-R0 executor/parser round-trip or
                       violate Stage 3 contract (→ derive normalized file)
  NO_MISMATCH        — all refs compatible; no derived dataset written
  AMBIGUOUS          — mixed / undecidable (hard abort for overnight)

Usage:
  python scripts/data/audit_stage3_nestful_syntax.py \\
    --input data/training_ready_v5/filtered/stage3_train_ready.jsonl \\
    --report-dir reports/stage3_syntax_audit
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3 = os.path.normpath(os.path.join(_HERE, "..", ".."))
_MINIMAL = os.path.normpath(os.path.join(_V3, "..", "nestful_mtgrpo_minimal"))
for p in (_V3, _MINIMAL):
    if p not in sys.path:
        sys.path.insert(0, p)

from executor import (  # noqa: E402
    _VAR_REF_RE,
    ToolExecutor,
    _is_variable_ref,
)
from lib.agentic_data.exec_bridge import (  # noqa: E402
    REGISTRY_VERSION,
    TOOLS,
    execute_gold_trace,
    registry_hash,
)
from lib.agentic_data.trace_validation import hard_trace_errors  # noqa: E402
from parser import parse_tool_calls_all  # noqa: E402

# Tool-R0 ReAct prompt canonical form (prompt.py)
_CANONICAL_LABEL = re.compile(r"^\$var(\d+)$")
_CANONICAL_REF = re.compile(r"^\$var(\d+)\.([A-Za-z_][\w]*)\$$")
# NESTFUL-native gold style (underscore)
_NESTFUL_LABEL = re.compile(r"^\$var_(\d+)$")
_NESTFUL_REF = re.compile(r"^\$var_(\d+)\.([A-Za-z_][\w]*)\$$")

EXPECTED_ROWS = 326


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk_refs(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, str):
        if _is_variable_ref(obj) or (obj.startswith("$") and obj.endswith("$")):
            out.append(obj)
    elif isinstance(obj, list):
        for x in obj:
            out.extend(_walk_refs(x))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_walk_refs(v))
    return out


def _classify_ref(ref: str) -> str:
    s = ref.strip()
    if _CANONICAL_REF.match(s):
        return "tool_r0_canonical"
    if _NESTFUL_REF.match(s):
        return "nestful_underscore"
    if _VAR_REF_RE.match(s):
        return "executor_accepted_other"
    return "invalid"


def _classify_label(lab: str) -> str:
    if _CANONICAL_LABEL.match(lab or ""):
        return "tool_r0_canonical"
    if _NESTFUL_LABEL.match(lab or ""):
        return "nestful_underscore"
    return "other"


def _render_tool_call_answer(calls: List[dict]) -> str:
    return "<tool_call_answer>" + json.dumps(calls, ensure_ascii=False) + "</tool_call_answer>"


def audit_row(row: Dict[str, Any], *, line_no: int) -> Dict[str, Any]:
    sid = str(row.get("sample_id") or f"line_{line_no}")
    gold = row.get("gold_calls") or []
    issues: List[str] = []
    refs: List[Dict[str, Any]] = []
    hard = False

    if row.get("num_calls") != 3 or len(gold) != 3:
        issues.append(f"expected 3 gold calls, got num_calls={row.get('num_calls')} len={len(gold)}")
        hard = True

    stage = str(row.get("stage") or "")
    if "stage2" in stage:
        issues.append(f"stage2 contamination: {stage}")
        hard = True
    if "stage3" not in stage:
        issues.append(f"not stage3: {stage}")
        hard = True

    # Trace validation vs synthetic registry
    terr = hard_trace_errors(row, TOOLS, (3, 3))
    if terr:
        issues.append(f"trace: {terr[0]}")
        hard = True

    labels = []
    for i, call in enumerate(gold):
        lab = call.get("label")
        labels.append(lab)
        lc = _classify_label(str(lab or ""))
        if lc == "other":
            issues.append(f"call[{i}] non-canonical label {lab!r}")
            hard = True
        for ref in _walk_refs(call.get("arguments") or {}):
            cls = _classify_ref(ref)
            refs.append({"call_idx": i, "ref": ref, "class": cls, "label": lab})
            if cls == "invalid":
                issues.append(f"call[{i}] invalid ref {ref!r}")
                hard = True
            # backward-only: referenced var index must be < current
            m = _CANONICAL_REF.match(ref.strip()) or _NESTFUL_REF.match(ref.strip())
            if m:
                src = int(m.group(1))
                if src >= i + 1:
                    issues.append(f"call[{i}] forward/self ref {ref!r}")
                    hard = True

    # Variable numbering: $var1, $var2, $var3 sequential
    for i, lab in enumerate(labels, start=1):
        m = _CANONICAL_LABEL.match(str(lab or "")) or _NESTFUL_LABEL.match(str(lab or ""))
        if m and int(m.group(1)) != i:
            issues.append(f"label numbering: expected var{i}, got {lab}")
            hard = True

    # Parser round-trip (Tool-R0)
    text = _render_tool_call_answer(gold)
    parsed = parse_tool_calls_all(text)
    if parsed is None or len(parsed) != len(gold):
        issues.append("parser round-trip: parse_tool_calls_all failed")
        hard = True
    else:
        for i, (a, b) in enumerate(zip(gold, parsed)):
            if a.get("name") != b.get("name"):
                issues.append(f"parser round-trip name mismatch at {i}")
                hard = True

    # Synthetic executor replay
    obs, err = execute_gold_trace(gold)
    if err is not None:
        issues.append(f"executor replay: {err}")
        hard = True
    elif obs != row.get("observations"):
        issues.append("observations mismatch vs replay")
        hard = True
    elif obs and obs[-1] != row.get("gold_answer"):
        issues.append("gold_answer mismatch vs replay")
        hard = True

    # Field resolution: execute step-by-step via ToolExecutor
    try:
        ex = ToolExecutor(row, registry=None, mode="synthetic")
        for call in gold:
            res = ex.execute(call)
            if res.error:
                issues.append(f"ToolExecutor.execute: {res.error}")
                hard = True
                break
    except Exception as exc:  # noqa: BLE001
        issues.append(f"ToolExecutor: {type(exc).__name__}: {exc}")
        hard = True

    # Question must not leak $var$
    q = str(row.get("question") or row.get("input") or "")
    if "$var" in q:
        issues.append("question contains $var leak")
        hard = True

    return {
        "sample_id": sid,
        "n_refs": len(refs),
        "refs": refs,
        "issues": issues,
        "hard_fail": hard,
        "ref_classes": dict(Counter(r["class"] for r in refs)),
        "label_classes": dict(Counter(_classify_label(str(l or "")) for l in labels)),
    }


def maybe_normalize_row(row: Dict[str, Any]) -> Tuple[Dict[str, Any], List[dict]]:
    """Normalize ``$var_N.field$`` → ``$varN.field$`` if needed.

    Only syntactic rewrite of labels/refs. Returns (new_row, changes).
    """
    changes: List[dict] = []
    out = copy.deepcopy(row)

    def norm_label(lab: str) -> str:
        m = _NESTFUL_LABEL.match(lab or "")
        if m:
            return f"$var{int(m.group(1))}"
        return lab

    def norm_value(v: Any, *, path: str) -> Any:
        if isinstance(v, str):
            m = _NESTFUL_REF.match(v.strip())
            if m:
                new = f"$var{int(m.group(1))}.{m.group(2)}$"
                if new != v:
                    changes.append({"path": path, "before": v, "after": new})
                return new
            return v
        if isinstance(v, list):
            return [norm_value(x, path=f"{path}[]") for x in v]
        if isinstance(v, dict):
            return {k: norm_value(x, path=f"{path}.{k}") for k, x in v.items()}
        return v

    new_calls = []
    for i, call in enumerate(out.get("gold_calls") or []):
        c = dict(call)
        old_lab = c.get("label")
        new_lab = norm_label(str(old_lab or ""))
        if new_lab != old_lab:
            changes.append({"path": f"gold_calls[{i}].label",
                            "before": old_lab, "after": new_lab})
        c["label"] = new_lab
        c["arguments"] = norm_value(c.get("arguments") or {},
                                    path=f"gold_calls[{i}].arguments")
        new_calls.append(c)
    out["gold_calls"] = new_calls
    return out, changes


def write_reports(report: dict, report_dir: str) -> None:
    os.makedirs(report_dir, exist_ok=True)
    jp = os.path.join(report_dir, "stage3_nestful_syntax_audit.json")
    mp = os.path.join(report_dir, "stage3_nestful_syntax_audit.md")
    # also mirror under reports/ at v3 root for runbook paths
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    lines = [
        "# Stage 3 NESTFUL Syntax Audit",
        "",
        f"Generated: {report['created_at']}",
        f"Verdict: **{report['verdict']}**",
        "",
        "## Canonical syntax (from code)",
        "",
        report["canonical_syntax"]["summary"],
        "",
        "### Supported alternatives",
        "",
    ]
    for a in report["canonical_syntax"]["alternatives"]:
        lines.append(f"- {a}")
    lines += [
        "",
        "## Counts",
        "",
        f"- Rows: {report['n_rows']}",
        f"- References checked: {report['n_refs_total']}",
        f"- Hard-fail rows: {report['n_hard_fail']}",
        f"- Incompatible refs: {report['n_incompatible_refs']}",
        f"- Rows changed by normalization: {report['n_rows_changed']}",
        f"- Input SHA-256: `{report['input_sha256']}`",
        f"- Output SHA-256: `{report.get('output_sha256') or '(same as input / no-op)'}`",
        "",
        "## Ref class distribution",
        "",
    ]
    for k, v in report["ref_class_counts"].items():
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Problem sample IDs", ""]
    if report["problem_sample_ids"]:
        for sid in report["problem_sample_ids"][:50]:
            lines.append(f"- `{sid}`")
    else:
        lines.append("(none)")
    lines += ["", "## Before/after examples", ""]
    for ex in report.get("normalize_examples", [])[:10]:
        lines.append(f"- `{ex['sample_id']}`: `{ex['before']}` → `{ex['after']}`")
    if not report.get("normalize_examples"):
        lines.append("(no-op — no changes)")
    lines += ["", "## Verdict rationale", "", report["verdict_rationale"], ""]
    with open(mp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    # Mirror to reports/ for runbook convenience
    mirror = os.path.join(_V3, "reports")
    os.makedirs(mirror, exist_ok=True)
    for src, name in ((jp, "stage3_nestful_syntax_audit.json"),
                      (mp, "stage3_nestful_syntax_audit.md")):
        dst = os.path.join(mirror, name)
        with open(src, encoding="utf-8") as fh:
            data = fh.read()
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--report-dir", default=os.path.join(_V3, "reports", "stage3_syntax_audit"))
    ap.add_argument("--derived-out", default=os.path.join(
        _V3, "data", "training_ready_v5", "derived",
        "stage3_nestful_syntax_v1.jsonl"))
    ap.add_argument("--force-normalize", action="store_true",
                    help="force underscore→canonical rewrite even if NO_MISMATCH")
    args = ap.parse_args()

    inp = os.path.abspath(args.input)
    if not os.path.isfile(inp):
        raise SystemExit(f"[audit] missing input: {inp}")

    rows = []
    with open(inp, encoding="utf-8-sig") as fh:
        for i, line in enumerate(fh, 1):
            if line.strip():
                rows.append((i, json.loads(line)))

    if len(rows) != EXPECTED_ROWS:
        raise SystemExit(
            f"[audit] ABORT: expected {EXPECTED_ROWS} rows, got {len(rows)}")

    per_row = [audit_row(r, line_no=ln) for ln, r in rows]
    n_hard = sum(1 for a in per_row if a["hard_fail"])
    ref_counts: Counter = Counter()
    n_refs = 0
    n_incompat = 0
    for a in per_row:
        n_refs += a["n_refs"]
        for r in a["refs"]:
            ref_counts[r["class"]] += 1
            if r["class"] == "invalid":
                n_incompat += 1

    nestful_style = ref_counts.get("nestful_underscore", 0)
    invalid = ref_counts.get("invalid", 0)

    # Decide verdict
    if n_hard or invalid:
        verdict = "MISMATCH_CONFIRMED" if invalid or n_hard else "AMBIGUOUS"
        if not invalid and n_hard:
            # structural issues — don't silent-normalize
            verdict = "AMBIGUOUS" if n_hard else verdict
    elif nestful_style > 0 and args.force_normalize:
        verdict = "MISMATCH_CONFIRMED"
    else:
        # Tool-R0 canonical + executor OK. Underscore style absent or
        # stylistic-only vs NESTFUL gold → NO_MISMATCH for this pipeline.
        verdict = "NO_MISMATCH"

    # If hard structural fails exist, abort overnight (AMBIGUOUS / hard)
    if n_hard and invalid == 0 and nestful_style == 0:
        verdict = "AMBIGUOUS"

    rationale_parts = [
        f"Trainer registry v{REGISTRY_VERSION} hash={registry_hash()[:12]}…",
        f"Tool-R0 ReAct prompt teaches `$varN.field$` (prompt.py).",
        f"Executor accepts `$varN` and `$var_N` via var_?(\\d+).",
        f"Official NESTFUL gold predominantly uses `$var_N.result$` "
        f"(stylistic; scorer accepts both).",
        f"Stage 3 refs: {dict(ref_counts)}.",
    ]
    if verdict == "NO_MISMATCH":
        rationale_parts.append(
            "All Stage 3 rows replay through synthetic executor and match "
            "Tool-R0 canonical syntax. No derived dataset written.")
    elif verdict == "MISMATCH_CONFIRMED":
        rationale_parts.append(
            "Incompatible or underscore-only refs require normalization.")
    else:
        rationale_parts.append(
            "Hard structural failures prevent safe normalization.")

    derived_path = None
    derived_sha = None
    n_changed = 0
    examples: List[dict] = []
    normalize_meta: List[dict] = []

    if verdict == "MISMATCH_CONFIRMED" or args.force_normalize:
        os.makedirs(os.path.dirname(args.derived_out), exist_ok=True)
        out_rows = []
        for (ln, row), audit in zip(rows, per_row):
            new_row, changes = maybe_normalize_row(row)
            if changes:
                n_changed += 1
                for ch in changes[:3]:
                    examples.append({
                        "sample_id": audit["sample_id"],
                        "before": ch["before"],
                        "after": ch["after"],
                    })
                # Re-validate semantics
                obs, err = execute_gold_trace(new_row["gold_calls"])
                if err is not None:
                    raise SystemExit(
                        f"[audit] ABORT: normalize broke replay "
                        f"{audit['sample_id']}: {err}")
                if obs != row.get("observations"):
                    raise SystemExit(
                        f"[audit] ABORT: normalize changed observations "
                        f"{audit['sample_id']}")
                if obs[-1] != row.get("gold_answer"):
                    raise SystemExit(
                        f"[audit] ABORT: normalize changed gold_answer "
                        f"{audit['sample_id']}")
                new_row.setdefault("provenance", {})
                new_row["provenance"] = dict(new_row.get("provenance") or {})
                new_row["provenance"]["syntax_normalized_from"] = row.get("sample_id")
                new_row["provenance"]["syntax_normalize"] = "var_underscore_to_varN"
                normalize_meta.append({
                    "sample_id": audit["sample_id"],
                    "n_changes": len(changes),
                    "changes": changes,
                })
            out_rows.append(new_row)

        with open(args.derived_out, "w", encoding="utf-8") as fh:
            for r in out_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        derived_path = os.path.abspath(args.derived_out)
        derived_sha = _sha256(derived_path)
        # Re-audit derived quickly
        for i, r in enumerate(out_rows, 1):
            a = audit_row(r, line_no=i)
            if a["hard_fail"]:
                raise SystemExit(
                    f"[audit] ABORT: derived still hard-fails {a['sample_id']}: "
                    f"{a['issues'][:2]}")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": inp,
        "input_sha256": _sha256(inp),
        "output_path": derived_path,
        "output_sha256": derived_sha,
        "n_rows": len(rows),
        "n_refs_total": n_refs,
        "n_hard_fail": n_hard,
        "n_incompatible_refs": n_incompat,
        "n_rows_changed": n_changed,
        "ref_class_counts": dict(ref_counts),
        "problem_sample_ids": [a["sample_id"] for a in per_row if a["hard_fail"]],
        "normalize_examples": examples,
        "normalize_meta_path": None,
        "verdict": verdict,
        "verdict_rationale": " ".join(rationale_parts),
        "canonical_syntax": {
            "summary": (
                "Tool-R0 training stack: labels `$varN`, references "
                "`$varN.<output_key>$` (prompt.py + executor.py). "
                "Field is required for object outputs; optional for scalars."
            ),
            "label": "$varN",
            "reference": "$varN.field$",
            "alternatives": [
                "$var_N.field$ — NESTFUL gold / IBM scorer (accepted by executor)",
                "$varN$ without field — accepted by executor for scalar obs",
            ],
            "sources": [
                "experiments/nestful_mtgrpo_minimal/prompt.py",
                "experiments/nestful_mtgrpo_minimal/executor.py (_VAR_REF_RE)",
                "experiments/nestful_mtgrpo_minimal/data/NESTFUL-main/src/scorer.py",
            ],
        },
        "registry_version": REGISTRY_VERSION,
        "registry_hash": registry_hash(),
        "per_row": [{k: a[k] for k in ("sample_id", "n_refs", "hard_fail",
                                        "issues", "ref_classes", "label_classes")}
                    for a in per_row],
    }

    if normalize_meta:
        meta_path = os.path.join(args.report_dir, "normalize_changes.jsonl")
        os.makedirs(args.report_dir, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as fh:
            for m in normalize_meta:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")
        report["normalize_meta_path"] = meta_path

    write_reports(report, args.report_dir)
    print(f"[audit] verdict={verdict}")
    print(f"[audit] rows={len(rows)} refs={n_refs} hard_fail={n_hard} "
          f"changed={n_changed}")
    print(f"[audit] report -> {args.report_dir}")
    if derived_path:
        print(f"[audit] derived -> {derived_path} sha256={derived_sha}")
    if verdict == "AMBIGUOUS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
