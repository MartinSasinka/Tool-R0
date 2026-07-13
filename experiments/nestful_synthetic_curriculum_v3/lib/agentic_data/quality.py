"""Acceptance policy: solver-gap thresholds, strong-pass policy, in-corpus
dedup and diversity caps on accepted examples."""
from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from ..nestful_like_generator import question_hash, trace_hash
from .env_defaults import (
    DIVERSITY_ENFORCE_AFTER,
    DIVERSITY_MAX_SAME_FAILURE_TYPE,
    DIVERSITY_MAX_SAME_WEAK_SCORE,
    env_float,
    env_int,
)

# Preferred solver-gap acceptance (spec §6; Autodata's weak-fail/strong-pass).
# Override via env for fill-vs-quality tradeoffs:
#   SOLVER_WEAK_MAX=0.55   — allow slightly stronger weak failures
#   SOLVER_GAP_MIN=0.20    — allow smaller weak/strong separation
#   SOLVER_STRONG_MIN=0.80 — strong partial floor (exact_win policy still
#                            requires score 1.0 when STRONG_PASS_POLICY=exact_win)
def _solver_thresholds() -> Tuple[float, float, float]:
    env = os.environ
    return (
        float(env.get("SOLVER_WEAK_MAX", "0.50")),
        float(env.get("SOLVER_GAP_MIN", "0.25")),
        float(env.get("SOLVER_STRONG_MIN", "0.80")),
    )


STRONG_MIN = 0.80   # re-export default for tests/docs
WEAK_MAX = 0.50
GAP_MIN = 0.25

# STRONG_PASS_POLICY (explicit, spec 7D):
#   exact_win (default) — the strong solver must achieve a TRUE executable win
#     or solution-equivalent answer (score == 1.0). Partial strong solutions
#     never enter the filtered training set (only rejected/ logs).
#   threshold — legacy behavior (strong >= 0.80); mathematically equivalent
#     today because partial-prefix scores are capped below 0.8, but kept as an
#     explicit escape hatch.
def strong_pass_policy() -> str:
    p = os.environ.get("STRONG_PASS_POLICY", "exact_win")
    if p not in ("exact_win", "threshold"):
        raise ValueError(f"STRONG_PASS_POLICY={p!r} not in (exact_win, threshold)")
    return p


def solver_gap_verdict(weak: Dict[str, Any], strong: Optional[Dict[str, Any]]
                       ) -> Tuple[bool, Optional[str]]:
    """(accepted, rejection_reason). `strong` is None when skipped."""
    weak_max, gap_min, strong_min = _solver_thresholds()
    w = float(weak["score"])
    if strong is None:
        # weak already passed → strong run was skipped to save compute
        return False, "weak_solver_passed"
    s = float(strong["score"])
    if w > weak_max and s >= strong_min:
        return False, "too_easy_both_solvers_pass"
    if w > weak_max:
        return False, "weak_solver_passed"
    strong_needed = 0.999 if strong_pass_policy() == "exact_win" else strong_min
    if s < strong_needed and w <= weak_max and s < 0.5:
        return False, "too_hard_both_solvers_fail"
    if s < strong_needed:
        return False, "strong_solver_failed"
    if s - w < gap_min:
        return False, "weak_strong_gap_too_small"
    return True, None


class DiversityTracker:
    """Per-stage caps against a homogeneous accepted set (spec: the dataset
    must not be dominated by one weak-score bucket or one failure type).

    Enforcement starts only after `enforce_after` **new** accepted rows so tiny
    pilots and the warmup phase are never blocked.

    On resume (`resume_mode=True`), legacy rows are recorded in
    `seed_*` counters for reporting only — caps apply to NEW accepts this run,
    so a homogeneous partial corpus does not block the common 0.50 bucket
    forever while the generator hunts rarer failure modes.
    """

    def __init__(self, *, resume_mode: bool = False,
                 max_same_weak_score: Optional[float] = None,
                 max_same_failure_type: Optional[float] = None,
                 enforce_after: Optional[int] = None) -> None:
        self.resume_mode = resume_mode
        self.max_ws = env_float("DIVERSITY_MAX_SAME_WEAK_SCORE",
                               DIVERSITY_MAX_SAME_WEAK_SCORE) \
            if max_same_weak_score is None else max_same_weak_score
        self.max_ft = env_float("DIVERSITY_MAX_SAME_FAILURE_TYPE",
                                DIVERSITY_MAX_SAME_FAILURE_TYPE) \
            if max_same_failure_type is None else max_same_failure_type
        self.enforce_after = env_int("DIVERSITY_ENFORCE_AFTER",
                                     DIVERSITY_ENFORCE_AFTER) \
            if enforce_after is None else enforce_after
        # NEW rows this run (enforcement counters)
        self.ws: Counter = Counter()
        self.ft: Counter = Counter()
        self.n = 0
        # Legacy rows on resume (reference only — not used in verdict())
        self.seed_ws: Counter = Counter()
        self.seed_ft: Counter = Counter()
        self.n_seed = 0

    @staticmethod
    def _bucket(score: Any) -> str:
        try:
            return f"{float(score):.2f}"
        except (TypeError, ValueError):
            return "unknown"

    def seed_reference_from_rows(self, rows: List[Dict[str, Any]]) -> None:
        """Record legacy distribution for reporting; does NOT affect caps."""
        for row in rows:
            sg = row.get("solver_gap") or {}
            ws = self._bucket(sg.get("weak_score"))
            ft = str(sg.get("weak_status") or "unknown")
            self.seed_ws[ws] += 1
            self.seed_ft[ft] += 1
            self.n_seed += 1

    def seed_from_rows(self, rows: List[Dict[str, Any]]) -> None:
        """Fresh-run helper: seed rows count toward enforcement counters."""
        for row in rows:
            sg = row.get("solver_gap") or {}
            self.add(sg.get("weak_score"), sg.get("weak_status") or "unknown")

    def verdict(self, weak_score: Any, failure_type: str) -> Optional[str]:
        """Rejection reason if accepting this row would break a cap, else None."""
        if self.n < self.enforce_after:
            return None
        if (self.ws[self._bucket(weak_score)] + 1) / (self.n + 1) > self.max_ws:
            return "diversity_cap_weak_score"
        if (self.ft[failure_type] + 1) / (self.n + 1) > self.max_ft:
            return "diversity_cap_failure_type"
        return None

    def add(self, weak_score: Any, failure_type: str) -> None:
        self.ws[self._bucket(weak_score)] += 1
        self.ft[failure_type] += 1
        self.n += 1

    def stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "n_new": self.n,
            "weak_score_buckets_new": dict(self.ws),
            "failure_types_new": dict(self.ft),
            "weak_score_dominance_new": round(max(self.ws.values()) / self.n, 4)
            if self.n else None,
            "failure_type_dominance_new": round(max(self.ft.values()) / self.n, 4)
            if self.n else None,
            "caps": {"max_same_weak_score": self.max_ws,
                     "max_same_failure_type": self.max_ft,
                     "enforce_after": self.enforce_after,
                     "resume_mode": self.resume_mode},
        }
        if self.n_seed:
            out.update({
                "n_seed": self.n_seed,
                "weak_score_buckets_seed": dict(self.seed_ws),
                "failure_types_seed": dict(self.seed_ft),
                "weak_score_dominance_seed": round(
                    max(self.seed_ws.values()) / self.n_seed, 4),
                "failure_type_dominance_seed": round(
                    max(self.seed_ft.values()) / self.n_seed, 4),
            })
        return out


class DedupIndex:
    """In-corpus dedup by question hash and gold-trace hash."""

    def __init__(self) -> None:
        self.q: Set[str] = set()
        self.t: Set[str] = set()

    def seed_from_rows(self, rows: List[Dict[str, Any]]) -> int:
        """Pre-load hashes from existing accepted rows (resume). Returns count."""
        n = 0
        for row in rows:
            q = row.get("question") or ""
            gc = row.get("gold_calls")
            if not q or not isinstance(gc, list):
                continue
            self.q.add(question_hash(q))
            try:
                self.t.add(trace_hash(gc))
            except (KeyError, TypeError):
                pass
            n += 1
        return n

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
