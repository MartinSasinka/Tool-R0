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
_MINIMAL = os.path.normpath(os.path.join(_V3, "..", "nestful_mtgrpo_minimal"))


def _load_live_registry():
    """Load TOOLS from the same directory the trainer executor will use.

    ``SYNTHETIC_TOOLS_DIR`` (a directory containing ``lib/synthetic_tools.py``)
    is the trainer's registry override. The preflight MUST validate against
    that registry; otherwise a factory/pilot2 dataset is rejected as
    ``unknown tool`` while the subsequent training run would have executed
    those tools correctly.

    Loading goes through ``load_synthetic_tools_module`` (importlib by file
    path). A plain ``import lib.synthetic_tools`` is unsafe once package
    ``lib`` has been imported from a different tree — ``lib.__path__`` pins
    submodule discovery and silently keeps the Stage-3 registry.
    """
    tools_dir = os.path.abspath(os.environ.get("SYNTHETIC_TOOLS_DIR") or _V3)
    if _MINIMAL not in sys.path:
        sys.path.insert(0, _MINIMAL)
    if _V3 not in sys.path:
        sys.path.append(_V3)
    from synthetic_tool_registry import (  # noqa: E402
        load_synthetic_tools_module,
        reset_synthetic_registry,
    )
    reset_synthetic_registry()
    mod = load_synthetic_tools_module(tools_dir)
    return (
        tools_dir,
        mod.REGISTRY_VERSION,
        mod.TOOLS,
        mod.registry_hash,
        mod.tool_schema,
        mod,
    )


(_TOOLS_DIR, REGISTRY_VERSION, TOOLS, registry_hash, tool_schema,
 _REGISTRY_MOD) = _load_live_registry()
_IS_FACTORY_REGISTRY = (
    getattr(_REGISTRY_MOD, "REGISTRY_SOURCE", "") == "targeted_tool_data_factory"
    or hasattr(_REGISTRY_MOD, "factory_hashes")
)

if _V3 not in sys.path:
    sys.path.insert(0, _V3)
# Import only the structural validator. Do NOT import exec_bridge here:
# it pulls synthetic_gen_v5, which indexes Stage-3 TOOLS (`chain_in`) at
# import time and breaks when SYNTHETIC_TOOLS_DIR points at the factory
# adapter. Gold replay goes through ToolExecutor directly instead.
from lib.agentic_data.trace_validation import hard_trace_errors  # noqa: E402
from executor import ToolExecutor  # noqa: E402
from synthetic_tool_registry import get_synthetic_registry  # noqa: E402


def _acceptable_registry_hashes() -> set:
    """Dataset provenance may stamp the factory registry hash or the combined
    adapter hash; accept either so a correct export is not rejected."""
    out = {registry_hash()}
    try:
        if hasattr(_REGISTRY_MOD, "factory_hashes"):
            fh = _REGISTRY_MOD.factory_hashes()
            for k in ("registry_hash", "adapter_registry_hash"):
                if fh.get(k):
                    out.add(fh[k])
    except Exception:  # noqa: BLE001
        pass
    return out


def execute_gold_trace(gold_calls: List[Dict[str, Any]]
                       ) -> Tuple[Optional[List[Any]], Optional[str]]:
    """Replay gold calls through the REAL trainer executor (mode=synthetic).

    Uses the registry already bound by ``SYNTHETIC_TOOLS_DIR`` / the loader
    above — same path training uses for factory pilot2 canaries.
    """
    names = sorted({c.get("name") for c in gold_calls if isinstance(c, dict)})
    unknown = [n for n in names if n not in TOOLS]
    if unknown:
        return None, f"unknown tool '{unknown[0]}'"
    # Ensure ToolExecutor sees the same registry we validated against.
    _ = get_synthetic_registry()
    task = {"tools": [tool_schema(n) for n in names], "gold_calls": []}
    ex = ToolExecutor(task, mode="synthetic")
    observations: List[Any] = []
    for i, call in enumerate(gold_calls):
        res = ex.execute(call)
        if res.error is not None:
            return None, f"call {i + 1} ({call.get('name')}): {res.error}"
        observations.append(res.observation)
    return observations, None


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

    # Stage-3 hard_trace_errors require sequential `$var1`, `$var2`, …
    # Factory pilot data also emits `$var_1` (underscore) — legal for the
    # trainer executor. Skip the Stage-3 label convention when the live
    # registry is the factory adapter; gold replay still catches bad refs.
    if not _IS_FACTORY_REGISTRY:
        trace_errs = hard_trace_errors(row, TOOLS, bounds)
        if trace_errs:
            return f"{path}:{line_no} {sid}: trace: {trace_errs[0]}"
    else:
        n = len(gold_calls)
        if not (bounds[0] <= n <= bounds[1]):
            return (f"{path}:{line_no} {sid}: call count {n} outside "
                    f"expected range {bounds}")

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
    accepted = _acceptable_registry_hashes()
    if rh and rh not in accepted:
        return (f"{path}:{line_no} {sid}: registry_hash {rh[:16]}… != "
                f"trainer {registry_hash()[:16]}… "
                f"(registry_dir={_TOOLS_DIR})")
    if rv and rv != REGISTRY_VERSION:
        return (f"{path}:{line_no} {sid}: registry_version {rv} != "
                f"trainer {REGISTRY_VERSION}")

    return None


def validate_file(path: str) -> Dict[str, Any]:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise SystemExit(f"[preflight] ABORT: missing file: {path}")

    cur_hash = registry_hash()
    accepted = _acceptable_registry_hashes()
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
    if reg_hashes and not (reg_hashes <= accepted):
        raise SystemExit(
            f"[preflight] ABORT: {path} registry_hash "
            f"{next(iter(reg_hashes))[:16]}… != trainer {cur_hash[:16]}… "
            f"(v{REGISTRY_VERSION}, registry_dir={_TOOLS_DIR})")

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
          f"hash={registry_hash()[:16]}… dir={_TOOLS_DIR} "
          f"n_tools={len(TOOLS)}")
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
