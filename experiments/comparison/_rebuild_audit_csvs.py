#!/usr/bin/env python3
"""Re-emit the two malformed audit CSVs through the CSV-hygiene writer.

The original `audit_findings.csv` and `parser_executor_audit_summary.csv` were
written without quoting, so free-text `finding`/`result`/`evidence` fields with
commas broke `pandas.read_csv`. We parse them back using fixed-vocabulary anchors
(status / confidence) and rewrite via `nestful_core.logging_utils.write_csv`.
Idempotent: re-running on already-fixed (quoted) files is a no-op-equivalent.
"""
from __future__ import annotations

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.dirname(_HERE)
if _EXPERIMENTS not in sys.path:
    sys.path.insert(0, _EXPERIMENTS)

from nestful_core.logging_utils import write_csv  # noqa: E402

_STATUS = r"(PASS|WARNING|FAIL|NEEDS_MANUAL_REVIEW)"
_CONF = r"(high|medium|low)"


def _read_lines(path):
    with open(path, encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


def rebuild_findings(path):
    lines = _read_lines(path)[1:]  # skip header
    rows = []
    pat = re.compile(r"^([^,]+),([^,]+),([^,]+),(.*)," + _STATUS + r"," + _CONF + r",(.*)$")
    for ln in lines:
        m = pat.match(ln)
        if not m:
            continue
        rows.append({
            "id": m.group(1), "pass": m.group(2), "area": m.group(3),
            "finding": m.group(4), "status": m.group(5),
            "confidence": m.group(6), "evidence": m.group(7),
        })
    write_csv(path, rows, fieldnames=[
        "id", "pass", "area", "finding", "status", "confidence", "evidence"])
    return len(rows)


def rebuild_parser_exec(path):
    lines = _read_lines(path)[1:]
    rows = []
    # area,check (comma-free), then result(,) , STATUS , evidence(,)
    pat = re.compile(r"^([^,]+),([^,]+),(.*)," + _STATUS + r",(.*)$")
    for ln in lines:
        m = pat.match(ln)
        if not m:
            continue
        rows.append({
            "area": m.group(1), "check": m.group(2), "result": m.group(3),
            "status": m.group(4), "evidence": m.group(5),
        })
    write_csv(path, rows, fieldnames=["area", "check", "result", "status", "evidence"])
    return len(rows)


def main() -> int:
    f = os.path.join(_HERE, "audit_findings.csv")
    p = os.path.join(_HERE, "parser_executor_audit_summary.csv")
    nf = rebuild_findings(f)
    np_ = rebuild_parser_exec(p)
    print(f"rebuilt audit_findings.csv ({nf} rows), "
          f"parser_executor_audit_summary.csv ({np_} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
