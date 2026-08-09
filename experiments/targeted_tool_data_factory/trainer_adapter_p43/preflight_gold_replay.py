"""Gold-replay preflight for Pilot4.3 datasets through the P43 trainer adapter.

    python trainer_adapter_p43/preflight_gold_replay.py \
        --data outputs/pilot4_3_nestful_profile_1000/train_nestful_profile_1000.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
FACTORY = HERE.parent
EXPERIMENTS = FACTORY.parent
MINIMAL = EXPERIMENTS / "nestful_mtgrpo_minimal"

os.environ["SYNTHETIC_TOOLS_DIR"] = str(HERE)
for p in (str(MINIMAL), str(FACTORY / "src"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from executor import ToolExecutor, matches_gold  # noqa: E402
from lib.synthetic_tools import (  # noqa: E402
    ALL_TOOL_NAMES, REGISTRY_SOURCE, REGISTRY_VERSION, factory_hashes,
)
from synthetic_tool_registry import reset_synthetic_registry  # noqa: E402

reset_synthetic_registry()


def check_row(row: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    sid = row.get("task_id") or row.get("sample_id") or "?"
    tools = row.get("tools") or []
    gold = row.get("gold_calls") or []

    for t in tools:
        name = t.get("name")
        if name not in ALL_TOOL_NAMES:
            errs.append(f"{sid}: offered tool {name!r} missing from P43 registry")

    for c in gold:
        name = c.get("name")
        if name not in ALL_TOOL_NAMES:
            errs.append(f"{sid}: gold tool {name!r} missing from P43 registry")

    if errs:
        return errs

    task = {
        "tools": tools,
        "gold_calls": gold,
        "gold_answer": row.get("gold_answer"),
    }
    # Normalize list-params schemas the same way the trainer does.
    sys.path.insert(0, str(MINIMAL))
    from data import _normalize_tool_schema  # noqa: WPS433
    task["tools"] = _normalize_tool_schema(tools)

    ex = ToolExecutor(task, mode="synthetic")
    observations: List[Any] = []
    for i, call in enumerate(gold):
        res = ex.execute(call)
        if res.error is not None:
            errs.append(f"{sid}: call {i + 1} ({call.get('name')}) -> {res.error}")
            return errs
        observations.append(res.observation)

    # Prefer per-call frozen observations when present; else final answer only.
    for i, call in enumerate(gold):
        want = call.get("observation")
        if want is None or i >= len(observations):
            continue
        if not matches_gold(observations[i], want):
            errs.append(
                f"{sid}: observation {i + 1} mismatch: "
                f"{observations[i]!r} != {want!r}")
    if observations and not matches_gold(observations[-1], row.get("gold_answer")):
        errs.append(
            f"{sid}: final answer {observations[-1]!r} != {row.get('gold_answer')!r}")
    return errs


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--max-errors", type=int, default=30)
    args = ap.parse_args(argv)

    hashes = factory_hashes()
    print(f"[preflight-p43] registry_source={REGISTRY_SOURCE} version={REGISTRY_VERSION}")
    for k, v in sorted(hashes.items()):
        print(f"[preflight-p43] {k}={v}")
    print(f"[preflight-p43] tools_in_registry={len(ALL_TOOL_NAMES)}")

    summary: Dict[str, Any] = {"hashes": hashes, "files": {}, "ok": True}
    total_errs = 0
    for path_s in args.data:
        path = Path(path_s)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        rows = [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]
        errs: List[str] = []
        n_ok = 0
        for row in rows:
            e = check_row(row)
            if e:
                errs.extend(e)
            else:
                n_ok += 1
        ok = not errs
        summary["files"][str(path)] = {
            "rows": len(rows),
            "replayed_ok": n_ok,
            "errors": errs[: args.max_errors],
            "n_errors": len(errs),
            "passed": ok,
        }
        total_errs += len(errs)
        status = "PASS" if ok else "FAIL"
        print(f"[preflight-p43] {status} {path.name}: {n_ok}/{len(rows)} replayed")
        for e in errs[:10]:
            print(f"           ! {e}")

    summary["ok"] = total_errs == 0
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(summary, indent=2) + "\n",
                                     encoding="utf-8")
    print(f"[preflight-p43] {'PASS' if summary['ok'] else 'FAIL'} "
          f"({total_errs} problems)")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
