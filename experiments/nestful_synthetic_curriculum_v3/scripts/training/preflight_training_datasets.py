#!/usr/bin/env python3
"""Preflight validation for v5 GRPO training datasets.

Validates one or more JSONL files before training:

  * registry version/hash vs the trainer's live ``lib/synthetic_tools`` registry;
  * 100 % gold-trace replay through ``executor.mode=synthetic``;
  * stored ``observations`` and ``gold_answer`` match replay;
  * trace references and embedded tool schemas match the v5 registry;
  * ``num_calls`` equals ``len(gold_calls)``.

Exits non-zero on the **first** failure. Prints SHA-256 and row counts on success.

Usage (repo root):
  python experiments/nestful_synthetic_curriculum_v3/scripts/training/preflight_training_datasets.py \\
    experiments/nestful_synthetic_curriculum_v3/data/training_ready_v5/filtered/phase1_stage2_train.jsonl \\
    experiments/nestful_synthetic_curriculum_v3/data/training_ready_v5/filtered/phase2_stage3_plus_stage2_replay.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3 = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _V3 not in sys.path:
    sys.path.insert(0, _V3)

from lib.agentic_data.exec_bridge import (  # noqa: E402
    REGISTRY_VERSION, TOOLS, execute_gold_trace, registry_hash,
)
from lib.agentic_data.trace_validation import hard_trace_errors  # noqa: E402
from lib.synthetic_tools import tool_schema  # noqa: E402


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_registry(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    prov = row.get("provenance") or {}
    return (
        row.get("registry_hash") or prov.get("registry_hash"),
        row.get("registry_version") or prov.get("registry_version"),
    )


def _schema_sig(schema: Dict[str, Any]) -> Tuple[Any, ...]:
    params = schema.get("parameters") or {}
    out = schema.get("output_parameters") or {}
    req = tuple(sorted(params.get("required") or []))
    pin = tuple(sorted((k, (v or {}).get("type"))
                     for k, v in (params.get("properties") or {}).items()))
    pout = tuple(sorted((k, (v or {}).get("type")) for k, v in out.items()))
    return req, pin, pout


def _validate_row(row: Dict[str, Any], *, path: str, line_no: int) -> Optional[str]:
    sid = row.get("sample_id", f"line_{line_no}")
    gold_calls = row.get("gold_calls")
    if not isinstance(gold_calls, list) or not gold_calls:
        return f"{path}:{line_no} {sid}: missing gold_calls"

    n_calls = row.get("num_calls")
    if n_calls != len(gold_calls):
        return (f"{path}:{line_no} {sid}: num_calls={n_calls} != "
                f"len(gold_calls)={len(gold_calls)}")

    stage = row.get("stage", "")
    if "stage2" in stage:
        bounds = (2, 2)
    elif "stage3" in stage:
        bounds = (3, 3)
    else:
        bounds = (len(gold_calls), len(gold_calls))

    trace_errs = hard_trace_errors(row, TOOLS, bounds)
    if trace_errs:
        return f"{path}:{line_no} {sid}: trace: {trace_errs[0]}"

    tools = row.get("tools") or []
    if not isinstance(tools, list):
        return f"{path}:{line_no} {sid}: tools is not a list"
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        if not name or name not in TOOLS:
            return f"{path}:{line_no} {sid}: unknown tool in menu: {name!r}"
        v5 = tool_schema(name)
        if _schema_sig(t) != _schema_sig(v5):
            return (f"{path}:{line_no} {sid}: schema drift for {name} "
                    f"(embedded != v5 registry)")

    observations, err = execute_gold_trace(gold_calls)
    if err is not None:
        return f"{path}:{line_no} {sid}: replay: {err}"

    stored_obs = row.get("observations")
    if stored_obs != observations:
        return f"{path}:{line_no} {sid}: observations mismatch vs replay"

    final = observations[-1] if observations else None
    if final != row.get("gold_answer"):
        return f"{path}:{line_no} {sid}: gold_answer mismatch vs replay final obs"

    rh, rv = _row_registry(row)
    cur_hash = registry_hash()
    if rh and rh != cur_hash:
        return (f"{path}:{line_no} {sid}: registry_hash {rh[:16]}… != "
                f"trainer {cur_hash[:16]}…")
    if rv and rv != REGISTRY_VERSION:
        return (f"{path}:{line_no} {sid}: registry_version {rv} != "
                f"trainer {REGISTRY_VERSION}")

    return None


def validate_file(path: str) -> Dict[str, Any]:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise SystemExit(f"[preflight] ABORT: missing file: {path}")

    cur_hash = registry_hash()
    rows = 0
    reg_hashes: set = set()
    reg_versions: set = set()

    with open(path, encoding="utf-8-sig") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows += 1
            rh, rv = _row_registry(row)
            if rh:
                reg_hashes.add(rh)
            if rv:
                reg_versions.add(rv)
            err = _validate_row(row, path=path, line_no=line_no)
            if err:
                print(f"[preflight] FAIL: {err}", file=sys.stderr)
                raise SystemExit(1)

    if len(reg_hashes) > 1:
        raise SystemExit(f"[preflight] ABORT: {path} mixes registry hashes")
    if reg_hashes and next(iter(reg_hashes)) != cur_hash:
        raise SystemExit(
            f"[preflight] ABORT: {path} registry_hash "
            f"{next(iter(reg_hashes))[:16]}… != trainer {cur_hash[:16]}… "
            f"(v{REGISTRY_VERSION})")

    id_audit = {"path": path}
    ids = []
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                sid = row.get("sample_id") or row.get("task_id")
                if sid:
                    ids.append(str(sid))
    from collections import Counter
    c = Counter(ids)
    dups = {k: v for k, v in c.items() if v > 1}
    if dups:
        raise SystemExit(f"[preflight] ABORT: {path} duplicate sample_ids: "
                         f"{list(dups.items())[:3]}")

    digest = _sha256(path)
    report = {
        "path": path,
        "sha256": digest,
        "rows": rows,
        "registry_hash": next(iter(reg_hashes), cur_hash),
        "registry_version": next(iter(reg_versions), REGISTRY_VERSION),
        "trainer_registry_hash": cur_hash,
        "trainer_registry_version": REGISTRY_VERSION,
        "status": "ok",
    }
    print(f"[preflight] OK {os.path.basename(path)}: {rows} rows, sha256={digest}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", help="training JSONL file(s)")
    ap.add_argument("--report", default=None,
                    help="write combined JSON report to this path")
    args = ap.parse_args()

    print(f"[preflight] trainer registry v{REGISTRY_VERSION} "
          f"hash={registry_hash()[:16]}…")
    reports = [validate_file(p) for p in args.datasets]
    combined = {
        "registry_version": REGISTRY_VERSION,
        "registry_hash": registry_hash(),
        "datasets": reports,
        "total_rows": sum(r["rows"] for r in reports),
    }
    print(f"[preflight] ALL OK — {combined['total_rows']} rows across "
          f"{len(reports)} file(s)")
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(combined, fh, indent=2, ensure_ascii=False)
        print(f"[preflight] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
