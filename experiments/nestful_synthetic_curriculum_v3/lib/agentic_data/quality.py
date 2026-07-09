"""Acceptance policy: solver-gap thresholds + in-corpus dedup."""
from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

from ..nestful_like_generator import question_hash, trace_hash

# Preferred solver-gap acceptance (spec §6; Autodata's weak-fail/strong-pass)
STRONG_MIN = 0.80
WEAK_MAX = 0.50
GAP_MIN = 0.25


def solver_gap_verdict(weak: Dict[str, Any], strong: Optional[Dict[str, Any]]
                       ) -> Tuple[bool, Optional[str]]:
    """(accepted, rejection_reason). `strong` is None when skipped."""
    w = float(weak["score"])
    if strong is None:
        # weak already passed → strong run was skipped to save compute
        return False, "weak_solver_passed"
    s = float(strong["score"])
    if w > WEAK_MAX and s >= STRONG_MIN:
        return False, "too_easy_both_solvers_pass"
    if w > WEAK_MAX:
        return False, "weak_solver_passed"
    if s < STRONG_MIN and w <= WEAK_MAX and s < 0.5:
        return False, "too_hard_both_solvers_fail"
    if s < STRONG_MIN:
        return False, "strong_solver_failed"
    if s - w < GAP_MIN:
        return False, "weak_strong_gap_too_small"
    return True, None


class DedupIndex:
    """In-corpus dedup by question hash and gold-trace hash."""

    def __init__(self) -> None:
        self.q: Set[str] = set()
        self.t: Set[str] = set()

    def check_and_add(self, question: str, gold_calls) -> Optional[str]:
        qh = question_hash(question)
        th = trace_hash(gold_calls)
        if qh in self.q:
            return "duplicate_question"
        if th in self.t:
            return "duplicate_trace"
        self.q.add(qh)
        self.t.add(th)
        return None

    def remove(self, question: str, gold_calls) -> None:
        self.q.discard(question_hash(question))
        self.t.discard(trace_hash(gold_calls))
