"""Acceptance policy: solver-gap thresholds, strong-pass policy, in-corpus
dedup and diversity caps on accepted examples."""
from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from .exec_bridge import question_hash, trace_hash
from .env_defaults import (
    DIVERSITY_ENFORCE_AFTER,
    DIVERSITY_MAX_SAME_FAILURE_TYPE,
    DIVERSITY_MAX_SAME_WEAK_SCORE,
    ROLLOUT_FAILURE_ENFORCE_AFTER,
    ROLLOUT_FAILURE_MAX_SAME_TYPE,
    TIER_QUOTA_ENFORCE_AFTER,
    TIER_QUOTA_MAX_EASY_ANCHOR,
    TIER_QUOTA_MAX_PARTIAL_FRONTIER,
    TIER_QUOTA_MERGE_MAX_EASY_ANCHOR,
    TIER_QUOTA_MERGE_MAX_PARTIAL_FRONTIER,
    TIER_QUOTA_MERGE_MIN_FRONTIER,
    TIER_QUOTA_MIN_FRONTIER,
    TIER_QUOTA_STAGE2_MAX_EASY_ANCHOR,
    TIER_QUOTA_STAGE2_MAX_PARTIAL_FRONTIER,
    TIER_QUOTA_STAGE2_MIN_FRONTIER,
    TIER_QUOTA_STAGE3_MAX_EASY_ANCHOR,
    TIER_QUOTA_STAGE3_MAX_PARTIAL_FRONTIER,
    TIER_QUOTA_STAGE3_MIN_FRONTIER,
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

ACCEPTANCE_POLICIES = ("rollout_primary", "solver_gap")


def acceptance_policy() -> str:
    """How candidates are accepted after technical validation.

    ``rollout_primary`` (default when ``WEAK_SOLVER_BACKEND=local``):
      weak/strong solvers are cheap metadata only — they never veto a task
      solely because the weak model passed once. The 8-rollout GRPO-signal
      probe is the capability gate (≥2 reward levels, non-zero variance).

    ``solver_gap``: legacy Autodata weak-fail / strong-pass policy.
    """
    raw = os.environ.get("AGENTIC_ACCEPTANCE_POLICY", "").strip().lower()
    if raw in ACCEPTANCE_POLICIES:
        return raw
    backend = os.environ.get("WEAK_SOLVER_BACKEND", "local").strip().lower()
    return "rollout_primary" if backend == "local" else "solver_gap"


def solver_weak_max() -> float:
    return _solver_thresholds()[0]


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
    """(accepted, rejection_reason). `strong` is None when skipped.

    Under ``rollout_primary`` acceptance, always returns ``(True, None)`` —
    the multi-rollout probe decides usability for GRPO, not a one-shot weak
    pass."""
    if acceptance_policy() == "rollout_primary":
        return True, None
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


class TierQuotaTracker:
    """Cap accepted quality-tier mix (frontier / partial_frontier / easy_anchor).

    Quotas are stage-specific: Stage 2 targets more frontier anchors; Stage 3
    allows a higher partial-frontier share. Final global mix is enforced at
    merge time (see ``tier_quotas_for_merge``).
    """

    def __init__(self, *, stage: str, resume_mode: bool = False,
                 enforce_after: Optional[int] = None) -> None:
        self.stage = stage
        self.resume_mode = resume_mode
        self.enforce_after = env_int("TIER_QUOTA_ENFORCE_AFTER",
                                     TIER_QUOTA_ENFORCE_AFTER) \
            if enforce_after is None else enforce_after
        self.min_frontier, self.max_partial, self.max_easy = \
            tier_quotas_for_stage(stage)
        self.tiers: Counter = Counter()
        self.n = 0
        self.seed_tiers: Counter = Counter()
        self.n_seed = 0

    def seed_reference_from_rows(self, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            tier = ((row.get("quality") or {}).get("quality_tier")
                    or (row.get("rollout_signal") or {}).get("quality_tier")
                    or "unknown")
            self.seed_tiers[tier] += 1
            self.n_seed += 1

    def verdict(self, tier: str) -> Optional[str]:
        if self.n < self.enforce_after:
            return None
        tier = tier or "unknown"
        n_next = self.n + 1
        if tier == "easy_anchor" and (self.tiers["easy_anchor"] + 1) / n_next > self.max_easy:
            return "tier_quota_easy_anchor"
        if tier == "partial_frontier" and (
                self.tiers["partial_frontier"] + 1) / n_next > self.max_partial:
            return "tier_quota_partial_frontier"
        if tier != "frontier" and self.tiers["frontier"] / n_next < self.min_frontier:
            return "tier_quota_need_frontier"
        return None

    def add(self, tier: str) -> None:
        self.tiers[tier or "unknown"] += 1
        self.n += 1

    def stats(self) -> Dict[str, Any]:
        return {
            "n_new": self.n,
            "tiers_new": dict(self.tiers),
            "tier_shares_new": {k: round(v / self.n, 4) for k, v in self.tiers.items()}
            if self.n else {},
            "caps": {
                "stage": self.stage,
                "min_frontier": self.min_frontier,
                "max_partial_frontier": self.max_partial,
                "max_easy_anchor": self.max_easy,
                "enforce_after": self.enforce_after,
            },
            "n_seed": self.n_seed,
            "tiers_seed": dict(self.seed_tiers),
        }


class RolloutFailureTracker:
    """Cap dominance of one rollout failure type across accepted tasks."""

    def __init__(self, *, resume_mode: bool = False,
                 enforce_after: Optional[int] = None,
                 max_same_type: Optional[float] = None) -> None:
        self.resume_mode = resume_mode
        self.max_same = env_float("ROLLOUT_FAILURE_MAX_SAME_TYPE",
                                  ROLLOUT_FAILURE_MAX_SAME_TYPE) \
            if max_same_type is None else max_same_type
        self.enforce_after = env_int("ROLLOUT_FAILURE_ENFORCE_AFTER",
                                     ROLLOUT_FAILURE_ENFORCE_AFTER) \
            if enforce_after is None else enforce_after
        self.ft: Counter = Counter()
        self.n = 0
        self.seed_ft: Counter = Counter()
        self.n_seed = 0

    def seed_reference_from_rows(self, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            rs = row.get("rollout_signal") or {}
            dom = rs.get("dominant_rollout_failure")
            if dom:
                self.seed_ft[dom] += 1
                self.n_seed += 1

    def verdict(self, dominant_failure: str) -> Optional[str]:
        if self.n < self.enforce_after:
            return None
        dom = dominant_failure or "unknown"
        if (self.ft[dom] + 1) / (self.n + 1) > self.max_same:
            return "rollout_failure_type_cap"
        return None

    def add(self, dominant_failure: str) -> None:
        self.ft[dominant_failure or "unknown"] += 1
        self.n += 1

    def stats(self) -> Dict[str, Any]:
        return {
            "n_new": self.n,
            "failure_types_new": dict(self.ft),
            "dominance_new": round(max(self.ft.values()) / self.n, 4) if self.n else None,
            "cap": self.max_same,
            "enforce_after": self.enforce_after,
            "n_seed": self.n_seed,
            "failure_types_seed": dict(self.seed_ft),
        }


def tier_quotas_for_stage(stage: str) -> Tuple[float, float, float]:
    """(min_frontier, max_partial_frontier, max_easy_anchor) for a stage."""
    s = (stage or "").lower()
    if "stage2" in s:
        return (
            env_float("TIER_QUOTA_STAGE2_MIN_FRONTIER",
                      TIER_QUOTA_STAGE2_MIN_FRONTIER),
            env_float("TIER_QUOTA_STAGE2_MAX_PARTIAL_FRONTIER",
                      TIER_QUOTA_STAGE2_MAX_PARTIAL_FRONTIER),
            env_float("TIER_QUOTA_STAGE2_MAX_EASY_ANCHOR",
                      TIER_QUOTA_STAGE2_MAX_EASY_ANCHOR),
        )
    if "stage3" in s:
        return (
            env_float("TIER_QUOTA_STAGE3_MIN_FRONTIER",
                      TIER_QUOTA_STAGE3_MIN_FRONTIER),
            env_float("TIER_QUOTA_STAGE3_MAX_PARTIAL_FRONTIER",
                      TIER_QUOTA_STAGE3_MAX_PARTIAL_FRONTIER),
            env_float("TIER_QUOTA_STAGE3_MAX_EASY_ANCHOR",
                      TIER_QUOTA_STAGE3_MAX_EASY_ANCHOR),
        )
    return (
        env_float("TIER_QUOTA_MIN_FRONTIER", TIER_QUOTA_MIN_FRONTIER),
        env_float("TIER_QUOTA_MAX_PARTIAL_FRONTIER",
                  TIER_QUOTA_MAX_PARTIAL_FRONTIER),
        env_float("TIER_QUOTA_MAX_EASY_ANCHOR", TIER_QUOTA_MAX_EASY_ANCHOR),
    )


def tier_quotas_for_merge() -> Tuple[float, float, float]:
    """Global tier targets when building the final merged v5 dataset."""
    return (
        env_float("TIER_QUOTA_MERGE_MIN_FRONTIER", TIER_QUOTA_MERGE_MIN_FRONTIER),
        env_float("TIER_QUOTA_MERGE_MAX_PARTIAL_FRONTIER",
                  TIER_QUOTA_MERGE_MAX_PARTIAL_FRONTIER),
        env_float("TIER_QUOTA_MERGE_MAX_EASY_ANCHOR",
                  TIER_QUOTA_MERGE_MAX_EASY_ANCHOR),
    )
