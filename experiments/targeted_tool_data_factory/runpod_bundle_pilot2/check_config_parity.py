#!/usr/bin/env python3
"""D0 vs D1 config parity checker.

The whole experiment is only interpretable if the two arms differ in EXACTLY
one dimension: the training dataset. This checker diffs the two run configs and
fails (exit 2) on any difference outside the allow-list.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The only keys that are ALLOWED to differ between D0 and D1.
#
# `synthetic_tools_dir` is on this list and should be understood as a confound,
# not as a free choice: a Stage-3 task can only be executed by the legacy
# registry and a pilot2 task only by the factory adapter, so the executor
# implementation necessarily travels with the dataset. Everything else -- base
# checkpoint, seed, reward, optimizer, LoRA, credit assignment, decoding,
# rollout count, step budget -- must be byte-identical.
ALLOWED_DIFF = {
    "label",
    "train_subset",
    "train_subset_sha256",
    "train_subset_source",
    "synthetic_tools_dir",
    "run_id",
    "notes",
}


def flatten(obj, prefix: str = "") -> dict:
    out: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}." if not prefix else f"{prefix}{k}."))
    return out


def _flat(obj, prefix: str = "") -> dict:
    out: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            out.update(_flat(v, key))
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj, sort_keys=True)
    else:
        out[prefix] = obj
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d0", type=Path, required=True)
    ap.add_argument("--d1", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    d0 = _flat(json.loads(args.d0.read_text(encoding="utf-8")))
    d1 = _flat(json.loads(args.d1.read_text(encoding="utf-8")))

    keys = sorted(set(d0) | set(d1))
    diffs = [(k, d0.get(k, "<absent>"), d1.get(k, "<absent>")) for k in keys if d0.get(k) != d1.get(k)]
    violations = [d for d in diffs if d[0].split(".")[-1] not in ALLOWED_DIFF]

    print("[parity] intentional differences (dataset only):")
    for k, a, b in diffs:
        tag = "OK " if k.split(".")[-1] in ALLOWED_DIFF else "BAD"
        print(f"  [{tag}] {k}\n         D0={a}\n         D1={b}")
    print(f"[parity] {len(keys)} keys, {len(diffs)} differ, {len(violations)} violations")

    report = {
        "keys_compared": len(keys),
        "differences": [{"key": k, "d0": a, "d1": b} for k, a, b in diffs],
        "violations": [{"key": k, "d0": a, "d1": b} for k, a, b in violations],
        "verdict": "PASS" if not violations else "FAIL",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if violations:
        print("[parity] FAIL: D0 and D1 differ outside the dataset", file=sys.stderr)
        return 2
    print("[parity] PASS: D0 and D1 differ only in the training dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
