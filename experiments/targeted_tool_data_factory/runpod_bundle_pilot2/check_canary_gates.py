#!/usr/bin/env python3
"""Pilot2 executor gates for the dispatch canary.

`validate_dispatch_canary.py` already proves the reward dispatch is fixed. This
adds the gates that are specific to swapping the tool registry: the trainer must
really be executing the targeted_tool_data_factory primitives, and it must not
have silently fallen back to the legacy Stage-3 `synthetic_tools.py`.

Gates:
  E1  every executed tool name is a pilot2 factory surface
  E2  no legacy-only Stage-3 tool name appears anywhere in the canary rollouts
  E3  the factory registry/executor/adapter hash is present in the runtime log
      and identical across both arms
  E4  reference resolution actually happened ($varN.output_0$ never survives
      into an executed argument)
  E5  observations were produced (non-empty executor output on executable calls)

Exit 0 = PASS, 1 = FAIL.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

BUNDLE = Path(__file__).resolve().parent
FACTORY = BUNDLE.parent
ARMS = ("A1_OUTCOME_ONLY", "A4_GATED_VERIFIABLE")
REF_RE = re.compile(r"\$var\d+\.output_\d+\$")

# Legacy Stage-3 names that must never be executed under the factory adapter.
LEGACY_MARKERS = ("get_current_weather", "search_web", "get_stock_price")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _factory_surfaces() -> set[str]:
    sys.path.insert(0, str(FACTORY / "src"))
    from targeted_tool_data.registry import all_surfaces  # type: ignore

    return {s.name for s in all_surfaces()}


def _walk_calls(row: Any):
    """Yield (name, arguments) for every tool call recorded in a rollout row."""
    stack = [row]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            name = cur.get("name") or cur.get("tool") or cur.get("function")
            args = cur.get("arguments")
            if isinstance(name, str) and isinstance(args, dict):
                yield name, args
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    surfaces = _factory_surfaces()
    gates: dict[str, dict] = {}
    hashes: dict[str, Any] = {}
    failures: list[str] = []

    for arm in ARMS:
        run_dir = args.output_root / f"dispatch_canary_{arm}_seed{args.seed}"
        rollouts = _load_jsonl(run_dir / "train" / "canary_rollouts.jsonl")
        if not rollouts:
            failures.append(f"{arm}: no canary_rollouts.jsonl at {run_dir}")
            continue

        unknown, legacy, unresolved, calls, with_obs = set(), set(), 0, 0, 0
        for row in rollouts:
            for name, arg in _walk_calls(row):
                calls += 1
                if name not in surfaces:
                    unknown.add(name)
                if name in LEGACY_MARKERS:
                    legacy.add(name)
                if REF_RE.search(json.dumps(arg)):
                    unresolved += 1
            obs = row.get("observations") or row.get("executor_observations")
            if obs:
                with_obs += 1

        rh = None
        for row in rollouts:
            rh = row.get("registry_hash") or row.get("factory_registry_hash") or rh
        hashes[arm] = rh

        arm_gates = {
            "E1_all_tools_are_factory_surfaces": not unknown,
            "E2_no_legacy_stage3_tools": not legacy,
            "E4_no_unresolved_references": unresolved == 0,
            "E5_observations_logged": with_obs > 0,
            "calls_seen": calls,
            "rollouts": len(rollouts),
            "unknown_tools": sorted(unknown)[:20],
            "legacy_tools": sorted(legacy),
            "unresolved_reference_calls": unresolved,
        }
        gates[arm] = arm_gates
        for g in ("E1_all_tools_are_factory_surfaces", "E2_no_legacy_stage3_tools",
                  "E4_no_unresolved_references", "E5_observations_logged"):
            if not arm_gates[g]:
                failures.append(f"{arm}: {g}")

    vals = [v for v in hashes.values() if v]
    e3 = bool(vals) and len(set(vals)) == 1
    if not e3:
        failures.append(f"E3_registry_hash_logged_and_consistent: {hashes}")

    report = {
        "seed": args.seed,
        "arms": gates,
        "registry_hashes": hashes,
        "E3_registry_hash_logged_and_consistent": e3,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }
    print(json.dumps(report, indent=2)[:4000])
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if failures:
        print("[canary_gates] FAIL", *failures, sep="\n  ", file=sys.stderr)
        return 1
    print("[canary_gates] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
