"""NESTFUL contamination gate: zero overlap by question/trace hash + sample_id."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Set, Tuple

from .exec_bridge import question_hash, trace_hash

_HERE = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(V3_ROOT, "scripts", "lib"))
from paths import NESTFUL_DATASETS  # noqa: E402


def _coerce(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return v
    return v


def load_nestful_hashes() -> Tuple[Set[str], Set[str], Set[str]]:
    """(question_hashes, trace_hashes, sample_ids) over dev+test+full."""
    qs: Set[str] = set()
    ts: Set[str] = set()
    ids: Set[str] = set()
    found_any = False
    for path in NESTFUL_DATASETS.values():
        if not os.path.isfile(path):
            continue
        found_any = True
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                q = row.get("question") or row.get("input") or ""
                qs.add(question_hash(str(q)))
                calls = _coerce(row.get("gold_calls") or row.get("output")) or []
                if isinstance(calls, list) and calls:
                    try:
                        ts.add(trace_hash(calls))
                    except (KeyError, TypeError):
                        pass
                sid = row.get("sample_id")
                if sid:
                    ids.add(str(sid))
    if not found_any:
        raise FileNotFoundError(
            "No NESTFUL data found — the contamination gate cannot run, "
            "REFUSING to generate (would be unauditable).")
    return qs, ts, ids


class ContaminationChecker:
    def __init__(self) -> None:
        self.nest_q, self.nest_t, self.nest_ids = load_nestful_hashes()

    def check(self, question: str, gold_calls: List[Dict[str, Any]],
              sample_id: str) -> Tuple[bool, str]:
        if question_hash(question) in self.nest_q:
            return False, "question hash overlaps NESTFUL"
        try:
            if trace_hash(gold_calls) in self.nest_t:
                return False, "gold trace hash overlaps NESTFUL"
        except (KeyError, TypeError):
            pass
        if sample_id in self.nest_ids:
            return False, "sample_id overlaps NESTFUL"
        return True, ""
