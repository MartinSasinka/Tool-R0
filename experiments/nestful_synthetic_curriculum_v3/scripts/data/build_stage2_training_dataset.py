#!/usr/bin/env python3
"""Rematerialize legacy v4 Stage 2 agentic rows through the v5 tool registry.

Reads ``curriculum_v4_nestful_like_agentic_openrouter`` Stage 2 JSONL, then for
each row:

  * rewrites embedded tool schemas to authoritative v5 ``tool_schema()`` output;
  * canonicalizes ``$var.field$`` references to v5 ``out_key`` / ``out_fields``;
  * replays gold calls through ``executor.mode=synthetic``;
  * refreshes ``observations`` and ``gold_answer`` from replay;
  * stamps registry version/hash and v5 ``source`` for training.

Usage (repo root):
  python experiments/nestful_synthetic_curriculum_v3/scripts/data/build_stage2_training_dataset.py
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if V3_ROOT not in sys.path:
    sys.path.insert(0, V3_ROOT)
sys.path.insert(0, os.path.join(V3_ROOT, "scripts", "lib"))

from paths import sha256_file  # noqa: E402
from lib.agentic_data.distribution import corpus_stats  # noqa: E402
from lib.agentic_data.exec_bridge import (  # noqa: E402
    REGISTRY_SOURCE, REGISTRY_VERSION, TOOLS, execute_gold_trace,
    registry_hash, replay_task,
)
from lib.agentic_data.schema import (  # noqa: E402
    SOURCE_NAME, STAGE_FILES, TOOL_SCHEMA_SOURCE_POLICY,
)
from lib.agentic_data.semantics import semantic_errors  # noqa: E402
from lib.agentic_data.trace_validation import hard_trace_errors, _valid_ref_fields  # noqa: E402
from lib.synthetic_tools import tool_schema  # noqa: E402

STAGE = "stage2_2call_agentic_openrouter"
DEFAULT_IN = os.path.join(
    V3_ROOT, "data", "curriculum_v4_nestful_like_agentic_openrouter",
    "filtered", STAGE_FILES[STAGE],
)
DEFAULT_OUT = os.path.join(V3_ROOT, "data", "curriculum_v5_stage2_training")

_VAR_REF_RE = re.compile(r"^\$([A-Za-z_]\w*)\.(\w+)\$$")


def _tool_names_from_row(row: Dict[str, Any]) -> List[str]:
    tools = row.get("tools") or []
    names: List[str] = []
    for t in tools:
        if isinstance(t, dict) and isinstance(t.get("name"), str):
            names.append(t["name"])
        elif isinstance(t, str):
            names.append(t)
    return names


def _rewrite_ref(value: str, label_to_tool: Dict[str, str]) -> str:
    m = _VAR_REF_RE.match(value.strip())
    if not m:
        return value
    lbl, field = m.group(1), m.group(2)
    producer = label_to_tool.get(lbl)
    if not producer or producer not in TOOLS:
        return value
    valid = _valid_ref_fields(TOOLS[producer])
    if valid is None or field in valid:
        return value
    # Scalar tools: legacy decorative keys (value/result/output_0) → v5 out_key.
    if TOOLS[producer].get("out_fields"):
        return value
    canon = TOOLS[producer]["out_key"]
    return f"${lbl}.{canon}$"


def _rewrite_gold_calls(gold_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    label_to_tool = {
        str(c.get("label", "")).strip("$"): c["name"]
        for c in gold_calls
        if isinstance(c, dict) and isinstance(c.get("name"), str)
    }
    out: List[Dict[str, Any]] = []
    for call in gold_calls:
        if not isinstance(call, dict):
            out.append(call)
            continue
        args = call.get("arguments")
        if not isinstance(args, dict):
            out.append(copy.deepcopy(call))
            continue
        new_args = {}
        for k, v in args.items():
            if isinstance(v, str):
                new_args[k] = _rewrite_ref(v, label_to_tool)
            else:
                new_args[k] = v
        nc = copy.deepcopy(call)
        nc["arguments"] = new_args
        out.append(nc)
    return out


def _v5_tool_menu(tool_names: List[str]) -> List[Dict[str, Any]]:
    unknown = [n for n in tool_names if n not in TOOLS]
    if unknown:
        raise ValueError(f"unknown tools: {unknown[:3]}")
    return [tool_schema(n) for n in tool_names]


def rematerialize_row(row: Dict[str, Any], *, now: str,
                      apply_semantic_filter: bool = False
                      ) -> Tuple[Dict[str, Any], Optional[str]]:
    tool_names = _tool_names_from_row(row)
    if not tool_names:
        return row, "empty tool menu"

    gold_calls = _rewrite_gold_calls(row.get("gold_calls") or [])
    trace_errs = hard_trace_errors(
        {"gold_calls": gold_calls, "num_calls": len(gold_calls)},
        TOOLS, (2, 2),
    )
    if trace_errs:
        return row, f"trace: {trace_errs[0]}"

    sem_errs = semantic_errors(gold_calls, TOOLS)
    if sem_errs and apply_semantic_filter:
        return row, f"semantic: {sem_errs[0]}"

    observations, err = execute_gold_trace(gold_calls)
    if err is not None:
        return row, f"replay: {err}"

    gold_answer = observations[-1] if observations else None
    prov = dict(row.get("provenance") or {})
    prov.update({
        "tool_schema_source_policy": TOOL_SCHEMA_SOURCE_POLICY,
        "registry_source": REGISTRY_SOURCE,
        "registry_version": REGISTRY_VERSION,
        "registry_hash": registry_hash(),
        "rematerialized_from": {
            "source": row.get("source"),
            "sample_id": row.get("sample_id"),
            "at": now,
        },
    })

    quality = dict(row.get("quality") or {})
    quality["gold_replay_passed"] = True
    quality["validation_passed"] = True

    out = copy.deepcopy(row)
    out.update({
        "tools": _v5_tool_menu(tool_names),
        "gold_calls": gold_calls,
        "observations": observations,
        "gold_answer": gold_answer,
        "num_calls": len(gold_calls),
        "source": SOURCE_NAME,
        "provenance": prov,
        "quality": quality,
    })
    return out, None


def _schema_signature(schema: Dict[str, Any]) -> Tuple[Any, ...]:
    params = schema.get("parameters") or {}
    out = schema.get("output_parameters") or {}
    req = tuple(sorted(params.get("required") or []))
    pin = tuple(sorted((k, (v or {}).get("type")) for k, v in (params.get("properties") or {}).items()))
    pout = tuple(sorted((k, (v or {}).get("type")) for k, v in out.items()))
    return (req, pin, pout)


def schema_drift_report(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare embedded v4 schemas vs v5 for tools present in the corpus."""
    drift: List[Dict[str, Any]] = []
    seen: set = set()
    for row in rows:
        for t in row.get("tools") or []:
            if not isinstance(t, dict):
                continue
            name = t.get("name")
            if not name or name in seen or name not in TOOLS:
                continue
            seen.add(name)
            v5 = tool_schema(name)
            if _schema_signature(t) != _schema_signature(v5):
                drift.append({
                    "tool": name,
                    "v4_required": sorted((t.get("parameters") or {}).get("required") or []),
                    "v5_required": sorted((v5.get("parameters") or {}).get("required") or []),
                    "v4_outputs": sorted((t.get("output_parameters") or {}).keys()),
                    "v5_outputs": sorted((v5.get("output_parameters") or {}).keys()),
                })
    return sorted(drift, key=lambda d: d["tool"])


def write_manifest(out_root: str, *, n_rows: int, replay_failed: List[Dict[str, str]],
                   drift: List[Dict[str, Any]], tier_counts: Dict[str, int],
                   input_path: str) -> str:
    filtered_path = os.path.join(out_root, "filtered", STAGE_FILES[STAGE])
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "curriculum_v5_stage2_training",
        "stage": STAGE,
        "executor_mode": "synthetic",
        "registry_version": REGISTRY_VERSION,
        "registry_hash": registry_hash(),
        "rows": n_rows,
        "training_candidate": True,
        "rematerialized_from": os.path.abspath(input_path),
        "schema_drift_tools_fixed": [d["tool"] for d in drift],
        "replay_failures_dropped": len(replay_failed),
        "replay_failure_samples": replay_failed[:20],
        "tier_counts": tier_counts,
        "files": {
            STAGE_FILES[STAGE]: {
                "path": os.path.abspath(filtered_path),
                "rows": n_rows,
                "sha256": sha256_file(filtered_path) if n_rows else None,
            },
        },
    }
    man_dir = os.path.join(out_root, "manifests")
    os.makedirs(man_dir, exist_ok=True)
    man_path = os.path.join(man_dir, "curriculum_v5_stage2_training_manifest.json")
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return man_path


def _row_tier(row: Dict[str, Any]) -> str:
    q = row.get("quality") or {}
    rs = row.get("rollout_signal") or {}
    tier = q.get("quality_tier") or rs.get("quality_tier")
    if tier:
        return tier
    fsr = rs.get("full_success_rate") or 0
    if fsr >= 0.999:
        return "easy_anchor"
    if fsr > 0:
        return "frontier"
    return "partial_frontier"


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_IN)
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    ap.add_argument("--apply-semantic-filter", action="store_true",
                    help="drop rows that fail v5 semantic compatibility (off by "
                         "default — matches stage3 build, which replay-filters only)")
    args = ap.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"[stage2-train] ERROR: missing input {input_path}", file=sys.stderr)
        return 2

    raw: List[Dict[str, Any]] = []
    with open(input_path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    print(f"[stage2-train] loaded {len(raw)} v4 rows from {input_path}")

    drift = schema_drift_report(raw)
    print(f"[stage2-train] schema drift vs v5 on {len(drift)} tool(s)")
    for d in drift:
        print(f"  - {d['tool']}: req {d['v4_required']} -> {d['v5_required']}, "
              f"out {d['v4_outputs']} -> {d['v5_outputs']}")

    now = datetime.now(timezone.utc).isoformat()
    kept: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []
    semantic_warnings = 0
    for row in raw:
        out, err = rematerialize_row(row, now=now,
                                     apply_semantic_filter=args.apply_semantic_filter)
        if err:
            failed.append({"sample_id": row.get("sample_id", "?"), "detail": err})
            continue
        if semantic_errors(out.get("gold_calls") or [], TOOLS):
            semantic_warnings += 1
        ok, detail = replay_task(out)
        if not ok:
            failed.append({
                "sample_id": row.get("sample_id", "?"),
                "detail": f"post_replay_mismatch: {detail}",
            })
            continue
        kept.append(out)

    for i, row in enumerate(kept):
        row["sample_id"] = f"agentic_v5_stage2_{i + 1:06d}"

    out_root = os.path.abspath(args.output_dir)
    out_path = os.path.join(out_root, "filtered", STAGE_FILES[STAGE])
    _write_jsonl(out_path, kept)

    tier_counts = dict(Counter(_row_tier(r) for r in kept))
    man_path = write_manifest(out_root, n_rows=len(kept), replay_failed=failed,
                              drift=drift, tier_counts=tier_counts,
                              input_path=input_path)

    summary_path = os.path.join(out_root, "reports", "STAGE2_TRAINING_DATASET.md")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    stats = corpus_stats(kept) if kept else {}
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("# Stage 2 v5 training dataset (rematerialized)\n\n")
        fh.write(f"- **Rows:** {len(kept)} / {len(raw)} input\n")
        fh.write(f"- **Registry:** {REGISTRY_VERSION} `{registry_hash()[:16]}…`\n")
        fh.write(f"- **Source field:** `{SOURCE_NAME}`\n")
        fh.write(f"- **Input:** `{input_path}`\n")
        fh.write(f"- **Output:** `{out_path}`\n")
        fh.write(f"- **Manifest:** `{man_path}`\n\n")
        fh.write("## Schema drift fixed\n\n")
        for d in drift:
            fh.write(f"- `{d['tool']}`: required `{d['v4_required']}` → "
                     f"`{d['v5_required']}`; outputs `{d['v4_outputs']}` → "
                     f"`{d['v5_outputs']}`\n")
        fh.write("\n## Tier mix\n\n")
        for k, v in sorted(tier_counts.items()):
            pct = 100.0 * v / len(kept) if kept else 0
            fh.write(f"- {k}: {v} ({pct:.1f}%)\n")
        if failed:
            fh.write(f"\n## Dropped ({len(failed)})\n\n")
            for rf in failed[:15]:
                fh.write(f"- `{rf['sample_id']}`: {rf['detail']}\n")
        if semantic_warnings:
            fh.write(f"\n## Semantic warnings (kept, {semantic_warnings})\n\n")
            fh.write("These rows replay correctly but chain semantically "
                     "incompatible quantity families (legacy v4 acceptances). "
                     "Pass `--apply-semantic-filter` to exclude them.\n")
        if stats:
            fh.write("\n## Diversity\n\n")
            fh.write(f"- motif dominance: {stats.get('dominance', {}).get('motif')}\n")
            fh.write(f"- tool_family dominance: "
                     f"{stats.get('dominance', {}).get('tool_family')}\n")

    print(f"[stage2-train] rematerialized {len(kept)}/{len(raw)} rows -> {out_path}")
    if semantic_warnings:
        print(f"[stage2-train] note: {semantic_warnings} rows have v5 semantic "
              f"warnings (kept; use --apply-semantic-filter to drop)")
    if failed:
        print(f"[stage2-train] dropped {len(failed)} rows (see report)")
    print(f"[stage2-train] manifest -> {man_path}")
    print(f"[stage2-train] summary -> {summary_path}")
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
