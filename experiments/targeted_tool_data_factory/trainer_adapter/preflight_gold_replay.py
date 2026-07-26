"""Gold-replay preflight through the REAL trainer executor.

Runs every gold trajectory of a GRPO-format dataset through
``nestful_mtgrpo_minimal.executor.ToolExecutor(mode="synthetic")`` with the
factory registry adapter installed. This is the check that the trainer can
actually execute pilot2 tools: it exercises tool-name resolution, strict
argument validation, ``$varN.output_0$`` reference resolution, observation
shape and final-answer parity.

Exit code is non-zero on a single failure.

    python trainer_adapter/preflight_gold_replay.py \
        --data outputs/selected/export_pilot2/train_grpo_pilot2.jsonl \
        --expect 160
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
FACTORY = HERE.parent
EXPERIMENTS = FACTORY.parent
MINIMAL = EXPERIMENTS / "nestful_mtgrpo_minimal"

# the trainer must load OUR registry, never the legacy one
os.environ["SYNTHETIC_TOOLS_DIR"] = str(HERE)
for p in (str(MINIMAL), str(FACTORY / "src"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from executor import ToolExecutor, matches_gold, normalize_arguments  # noqa: E402
from lib.synthetic_tools import (ALL_TOOL_NAMES, REGISTRY_SOURCE,  # noqa: E402
                                 REGISTRY_VERSION, factory_hashes, tool_schema)


def _schema_sig(tool: Dict[str, Any]) -> Tuple:
    params = (tool.get("parameters") or {}).get("properties") or {}
    return (tool.get("name"),
            tuple(sorted((k, v.get("type")) for k, v in params.items())))


def check_row(row: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    sid = row.get("sample_id", "?")
    tools = row.get("tools") or []
    gold = row.get("gold_calls") or []

    # 1. every offered tool exists in the runtime registry with the SAME schema
    for t in tools:
        name = t.get("name")
        if name not in ALL_TOOL_NAMES:
            errs.append(f"{sid}: offered tool {name!r} missing from registry")
            continue
        if _schema_sig(t) != _schema_sig(tool_schema(name)):
            errs.append(f"{sid}: schema drift for tool {name!r}")

    if errs:
        return errs

    # 2. execute the gold trajectory for real
    task = {"tools": tools, "gold_calls": gold, "gold_answer": row.get("gold_answer")}
    ex = ToolExecutor(task, mode="synthetic")
    observations: List[Any] = []
    n_refs = 0
    for i, call in enumerate(gold):
        args = call.get("arguments") or {}
        n_refs += sum(1 for v in args.values()
                      if isinstance(v, str) and v.strip().startswith("$"))
        res = ex.execute(call)
        if res.error is not None:
            errs.append(f"{sid}: call {i + 1} ({call.get('name')}) -> {res.error}")
            return errs
        observations.append(res.observation)

    # 3. observation and final-answer parity against the frozen oracle
    expected_obs = row.get("observations") or []
    if len(observations) != len(expected_obs):
        errs.append(f"{sid}: {len(observations)} observations, expected {len(expected_obs)}")
    else:
        for i, (got, want) in enumerate(zip(observations, expected_obs)):
            if not matches_gold(got, want):
                errs.append(f"{sid}: observation {i + 1} mismatch: {got!r} != {want!r}")
    if observations and not matches_gold(observations[-1], row.get("gold_answer")):
        errs.append(f"{sid}: final answer {observations[-1]!r} != {row.get('gold_answer')!r}")
    row["_n_refs"] = n_refs
    return errs


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True,
                    help="GRPO-format jsonl (repeatable)")
    ap.add_argument("--expect", action="append", type=int, default=None,
                    help="expected row count per --data (repeatable)")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    hashes = factory_hashes()
    print(f"[preflight] registry_source={REGISTRY_SOURCE} version={REGISTRY_VERSION}")
    for k, v in sorted(hashes.items()):
        print(f"[preflight] {k}={v}")
    print(f"[preflight] tools_in_registry={len(ALL_TOOL_NAMES)}")

    summary: Dict[str, Any] = {"hashes": hashes, "files": {}, "ok": True}
    total_errs = 0
    for idx, path_s in enumerate(args.data):
        path = Path(path_s)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        rows = [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]
        errs: List[str] = []
        refs = 0
        ref_rows = 0
        for row in rows:
            e = check_row(row)
            errs.extend(e)
            r = row.get("_n_refs", 0)
            refs += r
            ref_rows += 1 if r else 0
        expect = (args.expect or [])[idx] if args.expect and idx < len(args.expect) else None
        count_ok = expect is None or len(rows) == expect
        ok = not errs and count_ok
        summary["files"][str(path)] = {
            "rows": len(rows), "expected": expect, "replayed_ok": len(rows) - len(errs),
            "errors": errs[:20], "n_reference_args": refs,
            "rows_with_references": ref_rows, "passed": ok,
        }
        total_errs += len(errs) + (0 if count_ok else 1)
        status = "PASS" if ok else "FAIL"
        print(f"[preflight] {status} {path.name}: {len(rows) - len(errs)}/{len(rows)} "
              f"replayed, refs={refs} in {ref_rows} rows"
              + ("" if count_ok else f" (expected {expect} rows)"))
        for e in errs[:10]:
            print(f"           ! {e}")

    summary["ok"] = total_errs == 0
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(summary, indent=2) + "\n",
                                     encoding="utf-8")
    print(f"[preflight] {'PASS' if summary['ok'] else 'FAIL'} "
          f"({total_errs} problems)")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
