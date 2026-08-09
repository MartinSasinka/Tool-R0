"""NESTFUL-profile-preserving GRPO sampling (profile1000 ± enrichment500).

Extends the Phase-O samplers with:
* pool awareness (PROFILE vs ENRICHMENT);
* distribution-preserving refill (same call bucket);
* cumulative largest-remainder quota accumulator;
* history-adaptive weights with all-correct / all-fail floors;
* three modes: uniform_profile, dynamic_profile, dynamic_profile_plus_enrichment.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import (
    ALL_CORRECT, ALL_FAIL_NO_PROGRESS, ALL_FAIL_WITH_PROCESS_VARIANCE,
    DEFAULT_CONFIG, EQUAL_PARTIAL, INVALID_GROUP, MIXED_BOTH, MIXED_PROCESS_ONLY,
    MIXED_TERMINAL, GroupObservation, HistoryEntry, PromptRef, PromptSampler,
    SamplerState, UniformPromptSampler, _std, _weighted_sample_without_replacement,
    classify_group, is_effective, prompt_refs_from_dataset,
)

# Spec aliases (user vocabulary ↔ Phase-O taxonomy)
MIXED_EFFECTIVE = "MIXED_EFFECTIVE"
ALL_FAIL_WITH_PROGRESS = "ALL_FAIL_WITH_PROGRESS"
LOW_VARIANCE = "LOW_VARIANCE"

NESTFUL_CALL_SHARES = {
    "2": 0.330, "3": 0.220, "4": 0.135, "5": 0.095, "6+": 0.220,
}
CALL_BUCKETS = ("2", "3", "4", "5", "6+")

NESTFUL_SAMPLER_DEFAULTS: Dict[str, Any] = {
    **DEFAULT_CONFIG,
    "sampler_mode": "dynamic_profile",
    "group_size": 8,
    "reward_variance_epsilon": 1e-6,
    "epsilon_reward_std": 1e-6,
    "epsilon_process_std": 1e-6,
    # Default matches gradient_accumulation_steps in reference Qwen3 MT-GRPO
    # configs (effective optimizer micro-batch). Override via config.
    "target_effective_groups": 4,
    "initial_oversample_factor": 1.5,
    "max_oversample_factor": 3.0,
    "max_refill_rounds": 5,
    "max_raw_groups_per_update_factor": 3.0,
    "profile_share": 0.80,
    "enrichment_share": 0.20,
    "allow_cross_pool_refill": False,
    "minimum_prompt_weight": 0.05,
    "minimum_initial_exposures": 1,
    "ema_alpha": 0.1,          # note: HistoryEntry uses decay = 1-alpha style via ema_decay
    "ema_decay": 0.9,          # 1 - 0.1
    "revisit_after_steps": 50,
    "all_correct_downweight_2": 0.25,
    "all_correct_downweight_3": 0.10,
    "all_fail_np_downweight_2": 0.25,
    "weight_frontier": 0.45,
    "weight_variance": 0.35,
    "weight_staleness": 0.15,
    "weight_coverage": 0.05,
    "drop_low_variance": True,
    "effective_group_rate_warn": 0.20,
    "invalid_group_rate_warn": 0.05,
    "profile_tv_warn": 0.10,
    "warn_window_steps": 50,
    # Online bootstrap (replaces offline pre-training probe)
    "bootstrap": {
        "enabled": True,
        "min_unique_profile_prompts_seen": 200,
        "min_observed_groups_per_call_bucket": 25,
        "prefer_unseen_prompts": True,
        "unseen_weight": 2.0,
        "seen_weight": 1.0,
    },
}


def map_group_class(phase_o_class: str, obs: GroupObservation,
                    eps: float) -> str:
    """Map Phase-O class names to the user-facing taxonomy."""
    if phase_o_class == INVALID_GROUP:
        return INVALID_GROUP
    if phase_o_class == ALL_CORRECT:
        return ALL_CORRECT
    if phase_o_class == ALL_FAIL_NO_PROGRESS:
        return ALL_FAIL_NO_PROGRESS
    if phase_o_class == ALL_FAIL_WITH_PROCESS_VARIANCE:
        return ALL_FAIL_WITH_PROGRESS
    if phase_o_class in (MIXED_TERMINAL, MIXED_BOTH, MIXED_PROCESS_ONLY):
        return MIXED_EFFECTIVE
    if phase_o_class == EQUAL_PARTIAL or obs.reward_std <= eps:
        return LOW_VARIANCE
    if obs.reward_std > eps:
        return MIXED_EFFECTIVE
    return LOW_VARIANCE


def is_effective_nestful(obs: GroupObservation, cfg: Mapping[str, Any]) -> bool:
    eps = float(cfg.get("reward_variance_epsilon")
                or cfg.get("epsilon_reward_std") or 1e-6)
    min_valid = int(cfg.get("min_valid_rollouts") or max(2, obs.group_size // 2))
    n_valid = int(sum(1 for f in (obs.parse_flags or [True] * obs.group_size) if f))
    if obs.group_class == INVALID_GROUP:
        return False
    if n_valid < min_valid and obs.parse_flags:
        return False
    user_cls = map_group_class(obs.group_class, obs, eps)
    if user_cls in (ALL_CORRECT, ALL_FAIL_NO_PROGRESS, INVALID_GROUP):
        return False
    if user_cls == LOW_VARIANCE and cfg.get("drop_low_variance", True):
        return False
    if user_cls in (MIXED_EFFECTIVE, ALL_FAIL_WITH_PROGRESS):
        return obs.reward_std > eps
    return is_effective(obs, {**DEFAULT_CONFIG, **dict(cfg)})


@dataclass
class QuotaAccumulator:
    """Cumulative deficit / largest-remainder scheduler for call buckets."""

    shares: Dict[str, float] = field(
        default_factory=lambda: dict(NESTFUL_CALL_SHARES))
    actual: Dict[str, int] = field(default_factory=lambda: {b: 0 for b in CALL_BUCKETS})
    total: int = 0

    def target_cumulative(self) -> Dict[str, float]:
        return {b: self.total * self.shares[b] for b in CALL_BUCKETS}

    def deficit(self) -> Dict[str, float]:
        tgt = self.target_cumulative()
        return {b: tgt[b] - self.actual.get(b, 0) for b in CALL_BUCKETS}

    def pick_bucket(self, available: Mapping[str, int]) -> str:
        defs = self.deficit()
        # prefer largest deficit among buckets that still have candidates
        order = sorted(
            (b for b in CALL_BUCKETS if available.get(b, 0) > 0),
            key=lambda b: (-defs[b], b))
        if not order:
            # fallback any with stock
            stock = [b for b, n in available.items() if n > 0]
            return stock[0] if stock else "2"
        return order[0]

    def observe(self, bucket: str) -> None:
        if bucket not in self.actual:
            self.actual[bucket] = 0
        self.actual[bucket] += 1
        self.total += 1

    def tv_distance(self) -> float:
        if self.total <= 0:
            return 0.0
        actual_share = {b: self.actual.get(b, 0) / self.total for b in CALL_BUCKETS}
        return 0.5 * sum(abs(self.shares[b] - actual_share[b]) for b in CALL_BUCKETS)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_effective": self.total,
            "shares_target": dict(self.shares),
            "actual_counts": dict(self.actual),
            "target_cumulative": self.target_cumulative(),
            "deficit": self.deficit(),
            "tv_distance": round(self.tv_distance(), 5),
        }


def attach_pool(refs: Sequence[PromptRef], pool: str) -> List[PromptRef]:
    out = []
    for r in refs:
        d = r.as_dict()
        d["pool"] = pool
        d["difficulty_signature"] = {**(r.difficulty_signature or {}), "pool": pool}
        rr = PromptRef(**d)
        if not rr.call_bucket:
            rr.call_bucket = "2"
        if not rr.query_mode:
            rr.query_mode = "DOMAIN_GROUNDED_IMPLICIT"
        out.append(rr)
    return out


def pool_of(ref: PromptRef) -> str:
    if getattr(ref, "pool", None):
        return str(ref.pool)
    return str((ref.difficulty_signature or {}).get("pool") or "PROFILE")


def load_profile_enrichment_refs(
        profile_path: str, enrichment_path: Optional[str] = None
) -> Tuple[List[PromptRef], List[PromptRef]]:
    import json
    from pathlib import Path

    def _rows(path: str) -> List[Dict[str, Any]]:
        p = Path(path)
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                if l.strip()]

    def _refs(rows: List[Dict[str, Any]], pool: str) -> List[PromptRef]:
        enriched = []
        for row in rows:
            calls = row.get("gold_calls") or []
            n = len(calls)
            bucket = str(n) if n <= 5 else "6+"
            declared = row.get("declared") or {}
            caps = (row.get("capability_families")
                    or declared.get("capability_families") or [])
            row = {
                **row,
                "task_id": row.get("task_id"),
                "call_bucket": bucket,
                "num_calls": n,
                "requested_query_mode": row.get("actual_query_mode")
                or row.get("requested_query_mode"),
                "pattern_family": declared.get("structural_pattern")
                or row.get("actual_primary_pattern"),
                "difficulty_band": row.get("difficulty_band") or "medium",
                "capability_families": list(caps),
            }
            enriched.append(row)
        return attach_pool(prompt_refs_from_dataset(enriched), pool)

    profile = _refs(_rows(profile_path), "PROFILE")
    enrichment: List[PromptRef] = []
    if enrichment_path:
        enrichment = _refs(_rows(enrichment_path), "ENRICHMENT")
    return profile, enrichment


class NestfulProfileSampler(PromptSampler):
    """History-adaptive sampler with pool + call-bucket awareness.

    Starts in *online bootstrap* (uniform-within-bucket, prefer unseen) and
    flips to history-adaptive weighting once enough PROFILE groups are seen —
    without a separate pre-training probe run.
    """

    name = "nestful_profile"

    def __init__(self, profile: Sequence[PromptRef],
                 enrichment: Optional[Sequence[PromptRef]] = None, *,
                 config: Optional[Dict[str, Any]] = None, seed: int = 0) -> None:
        raw = dict(config or {})
        boot_in = raw.pop("bootstrap", None)
        cfg = {**NESTFUL_SAMPLER_DEFAULTS, **raw}
        boot_def = dict(NESTFUL_SAMPLER_DEFAULTS["bootstrap"])
        if isinstance(boot_in, dict):
            boot_def.update(boot_in)
        elif isinstance(cfg.get("bootstrap"), dict):
            boot_def.update(cfg["bootstrap"])
        cfg["bootstrap"] = boot_def
        mode = str(cfg.get("sampler_mode") or "dynamic_profile")
        self.sampler_mode = mode
        self.profile = list(profile)
        self.enrichment = list(enrichment or [])
        if mode == "uniform_profile":
            prompts = list(self.profile)
        elif mode == "dynamic_profile":
            prompts = list(self.profile)
        else:
            prompts = list(self.profile) + list(self.enrichment)
        super().__init__(prompts, config=cfg, seed=seed)
        self.name = mode
        # EFFECTIVE-only quota (NESTFUL profile preservation target).
        self.profile_quota = QuotaAccumulator()
        # RAW candidate quota: advances on every planned/refill pick so bootstrap
        # does not stick on lexicographic bucket "2" while effective total==0.
        self.raw_quota = QuotaAccumulator()
        self.pool_actual = {"PROFILE": 0, "ENRICHMENT": 0}
        self.ext: Dict[str, Dict[str, Any]] = {}
        # Online bootstrap bookkeeping
        self.bootstrap_complete = (
            mode == "uniform_profile"
            or not bool(boot_def.get("enabled", True))
        )
        self.bootstrap_completed_at_step: Optional[int] = None
        self.unique_profile_prompts_seen: set = set()
        self.groups_observed_per_bucket: Dict[str, int] = {b: 0 for b in CALL_BUCKETS}
        self.bootstrap_groups: List[Dict[str, Any]] = []
        self._bootstrap_report_written = False

    def _boot_cfg(self) -> Dict[str, Any]:
        return dict(self.cfg.get("bootstrap") or NESTFUL_SAMPLER_DEFAULTS["bootstrap"])

    def in_bootstrap(self) -> bool:
        return (self.sampler_mode != "uniform_profile"
                and not self.bootstrap_complete
                and bool(self._boot_cfg().get("enabled", True)))

    def bootstrap_status(self) -> Dict[str, Any]:
        bc = self._boot_cfg()
        need_u = int(bc.get("min_unique_profile_prompts_seen", 200))
        need_b = int(bc.get("min_observed_groups_per_call_bucket", 25))
        per = dict(self.groups_observed_per_bucket)
        return {
            "bootstrap_complete": self.bootstrap_complete,
            "bootstrap_completed_at_step": self.bootstrap_completed_at_step,
            "unique_profile_prompts_seen": len(self.unique_profile_prompts_seen),
            "min_unique_profile_prompts_seen": need_u,
            "groups_observed_per_call_bucket": per,
            "min_observed_groups_per_call_bucket": need_b,
            "buckets_ready": {
                b: per.get(b, 0) >= need_b for b in CALL_BUCKETS
            },
            "all_buckets_ready": all(per.get(b, 0) >= need_b for b in CALL_BUCKETS),
            "unique_ready": len(self.unique_profile_prompts_seen) >= need_u,
        }

    def _maybe_complete_bootstrap(self, optimizer_step: int) -> bool:
        """Return True exactly once when bootstrap flips to history-adaptive."""
        if self.bootstrap_complete or self.sampler_mode == "uniform_profile":
            return False
        st = self.bootstrap_status()
        if st["unique_ready"] and st["all_buckets_ready"]:
            self.bootstrap_complete = True
            self.bootstrap_completed_at_step = int(optimizer_step)
            return True
        return False

    def _ext(self, pid: str) -> Dict[str, Any]:
        return self.ext.setdefault(pid, {
            "pool": "PROFILE",
            "call_bucket": "",
            "consecutive_all_correct": 0,
            "consecutive_all_fail_no_progress": 0,
            "consecutive_effective": 0,
            "total_effective_groups": 0,
            "total_dead_groups": 0,
            "effective_count": 0,
            "all_correct_count": 0,
            "all_fail_progress_count": 0,
            "all_fail_no_progress_count": 0,
            "low_variance_count": 0,
            "invalid_count": 0,
            "last_group_class": None,
            "current_sampling_weight": 1.0,
            "sampling_weight": 1.0,
        })

    def weight_components(self, prompt: PromptRef,
                          state: SamplerState) -> Dict[str, float]:
        if self.sampler_mode == "uniform_profile":
            return {"base_cell_weight": 1.0}

        cfg = self.cfg
        e = state.prompt.get(prompt.prompt_id)
        ext = self._ext(prompt.prompt_id)
        floor = float(cfg["minimum_prompt_weight"])

        # ── Online bootstrap: uniform-within-bucket, prefer unseen ──────────
        if self.in_bootstrap():
            bc = self._boot_cfg()
            unseen = (e is None or e.group_count == 0)
            if bc.get("prefer_unseen_prompts", True) and unseen:
                w = float(bc.get("unseen_weight", 2.0))
            else:
                w = float(bc.get("seen_weight", 1.0))
            return {"base_cell_weight": max(w, floor), "bootstrap": 1.0}

        # ── History-adaptive (post-bootstrap) ───────────────────────────────
        comps: Dict[str, float] = {"base_cell_weight": 1.0}
        if e is None or e.group_count < int(cfg.get("minimum_initial_exposures", 1)):
            comps["base_cell_weight"] = max(2.0, floor)
            comps["coverage_score"] = 1.0
            return comps

        p = float(e.ema_terminal_success)
        frontier = 4.0 * p * (1.0 - p)
        var = min(math.sqrt(max(e.ema_reward_variance, 0.0)) * 2.0, 1.0)
        age = max(state.global_step - e.last_sampled_step, 0)
        staleness = min(age / max(float(cfg.get("revisit_after_steps", 50)), 1.0), 1.0)
        coverage = 0.0

        score = (float(cfg["weight_frontier"]) * frontier
                 + float(cfg["weight_variance"]) * var
                 + float(cfg["weight_staleness"]) * staleness
                 + float(cfg["weight_coverage"]) * coverage)

        if ext["consecutive_all_correct"] >= 3:
            score *= float(cfg["all_correct_downweight_3"])
        elif ext["consecutive_all_correct"] >= 2:
            score *= float(cfg["all_correct_downweight_2"])
        if ext["consecutive_all_fail_no_progress"] >= 2:
            score *= float(cfg["all_fail_np_downweight_2"])

        comps["frontier_weight"] = frontier
        comps["variance_weight"] = var
        comps["staleness_weight"] = staleness
        comps["base_cell_weight"] = max(score, floor)
        return comps

    def _combine(self, comps: Dict[str, float]) -> float:
        if self.sampler_mode != "uniform_profile":
            floor = float(self.cfg["minimum_prompt_weight"])
            return max(float(comps.get("base_cell_weight", 1.0)), floor)
        return super()._combine(comps)

    def observe_group(self, observation: GroupObservation) -> bool:
        eps = float(self.cfg.get("reward_variance_epsilon") or 1e-6)
        observation.recompute()
        observation.group_class = classify_group(
            observation,
            eps_reward=float(self.cfg.get("epsilon_reward_std") or eps),
            eps_process=float(self.cfg.get("epsilon_process_std") or eps),
        )
        user_cls = map_group_class(observation.group_class, observation, eps)
        effective = is_effective_nestful(observation, self.cfg)

        self.state.observe(observation)
        ext = self._ext(observation.prompt_id)
        ref = self.by_id.get(observation.prompt_id)
        pool = pool_of(ref) if ref is not None else "PROFILE"
        bucket = observation.call_bucket or (ref.call_bucket if ref else "2")
        ext["pool"] = pool
        ext["call_bucket"] = bucket
        ext["last_group_class"] = user_cls

        # Per-class counters (never delete task from pool)
        if user_cls == ALL_CORRECT:
            ext["all_correct_count"] = int(ext.get("all_correct_count", 0)) + 1
            ext["consecutive_all_correct"] += 1
            ext["consecutive_all_fail_no_progress"] = 0
            ext["consecutive_effective"] = 0
            ext["total_dead_groups"] += 1
        elif user_cls == ALL_FAIL_NO_PROGRESS:
            ext["all_fail_no_progress_count"] = int(
                ext.get("all_fail_no_progress_count", 0)) + 1
            ext["consecutive_all_fail_no_progress"] += 1
            ext["consecutive_all_correct"] = 0
            ext["consecutive_effective"] = 0
            ext["total_dead_groups"] += 1
        elif user_cls == ALL_FAIL_WITH_PROGRESS:
            ext["all_fail_progress_count"] = int(
                ext.get("all_fail_progress_count", 0)) + 1
            ext["consecutive_all_correct"] = 0
            ext["consecutive_all_fail_no_progress"] = 0
            if effective:
                ext["consecutive_effective"] += 1
                ext["total_effective_groups"] += 1
                ext["effective_count"] = int(ext.get("effective_count", 0)) + 1
        elif user_cls == MIXED_EFFECTIVE:
            ext["consecutive_all_correct"] = 0
            ext["consecutive_all_fail_no_progress"] = 0
            if effective:
                ext["consecutive_effective"] += 1
                ext["total_effective_groups"] += 1
                ext["effective_count"] = int(ext.get("effective_count", 0)) + 1
        elif user_cls == LOW_VARIANCE:
            ext["low_variance_count"] = int(ext.get("low_variance_count", 0)) + 1
            ext["consecutive_all_correct"] = 0
            ext["total_dead_groups"] += 1
        elif user_cls == INVALID_GROUP:
            ext["invalid_count"] = int(ext.get("invalid_count", 0)) + 1
            ext["total_dead_groups"] += 1
        else:
            ext["total_dead_groups"] += 1

        # Bootstrap / coverage stats (all observed groups, not only effective)
        if pool == "PROFILE":
            self.unique_profile_prompts_seen.add(observation.prompt_id)
            self.groups_observed_per_bucket[bucket] = (
                self.groups_observed_per_bucket.get(bucket, 0) + 1)
            if not self.bootstrap_complete:
                self.bootstrap_groups.append({
                    "task_id": observation.prompt_id,
                    "call_bucket": bucket,
                    "group_class": user_cls,
                    "effective": effective,
                    "reward_mean": observation.reward_mean,
                    "reward_std": observation.reward_std,
                    "terminal_success_rate": observation.terminal_success_rate,
                })

        if effective:
            self.pool_actual[pool] = self.pool_actual.get(pool, 0) + 1
            if pool == "PROFILE":
                self.profile_quota.observe(bucket or "2")

        # Check bootstrap → history-adaptive transition (same run)
        flipped = self._maybe_complete_bootstrap(
            optimizer_step=int(observation.global_step))
        if flipped:
            print(
                f"[sampler] BOOTSTRAP_COMPLETE at optimizer/global_step="
                f"{self.bootstrap_completed_at_step} "
                f"unique_prompts={len(self.unique_profile_prompts_seen)} "
                f"groups_per_bucket={dict(self.groups_observed_per_bucket)}",
                flush=True,
            )
        return effective

    def sample_from_bucket(self, pool: str, bucket: str, n: int = 1) -> List[PromptRef]:
        source = self.profile if pool == "PROFILE" else self.enrichment
        if self.sampler_mode == "uniform_profile":
            source = self.profile
        pool_refs = [p for p in source if p.call_bucket == bucket]
        if not pool_refs:
            return []
        st = self.state
        weights = []
        for p in pool_refs:
            comps = self.weight_components(p, st)
            w = max(self._combine(comps), float(self.cfg["minimum_prompt_weight"]))
            p.weight_components = comps
            p.selection_weight = w
            self._ext(p.prompt_id)["current_sampling_weight"] = w
            self._ext(p.prompt_id)["sampling_weight"] = w
            weights.append(w)
        picks = _weighted_sample_without_replacement(
            pool_refs, weights, min(n, len(pool_refs)), self.rng)
        for p in picks:
            st.prompt_entry(p.prompt_id).n_sampled += 1
            st.epoch_use[p.prompt_id] = st.epoch_use.get(p.prompt_id, 0) + 1
        return picks

    def available_counts(self, pool: str) -> Dict[str, int]:
        source = self.profile if pool == "PROFILE" else self.enrichment
        c: Dict[str, int] = {b: 0 for b in CALL_BUCKETS}
        for p in source:
            c[p.call_bucket] = c.get(p.call_bucket, 0) + 1
        return c

    def pick_profile_bucket(self) -> str:
        """Pick next PROFILE call bucket.

        Prefer EFFECTIVE deficit once any effective group has been observed
        (so long-run EFFECTIVE shares track NESTFUL). While effective total is
        still 0, use RAW quota so bootstrap candidates are not stuck on "2".
        """
        avail = self.available_counts("PROFILE")
        if self.profile_quota.total > 0:
            return self.profile_quota.pick_bucket(avail)
        return self.raw_quota.pick_bucket(avail)

    def note_raw_candidate(self, bucket: str, pool: str = "PROFILE") -> None:
        """Record a RAW candidate allocation (independent of effective)."""
        if pool == "PROFILE":
            self.raw_quota.observe(bucket or "2")

    def bootstrap_report(self) -> Dict[str, Any]:
        rows = list(self.bootstrap_groups)
        n = len(rows)
        classes: Dict[str, int] = defaultdict(int)
        by_bucket: Dict[str, int] = defaultdict(int)
        eff = 0
        for r in rows:
            classes[r["group_class"]] += 1
            by_bucket[r["call_bucket"]] += 1
            if r["effective"]:
                eff += 1
        return {
            "n_groups": n,
            "n_unique_tasks": len({r["task_id"] for r in rows}),
            "effective_group_share": round(eff / n, 4) if n else 0.0,
            "dead_group_share": round(1.0 - (eff / n), 4) if n else 0.0,
            "all_correct_share": round(classes.get(ALL_CORRECT, 0) / n, 4) if n else 0.0,
            "all_fail_progress_share": round(
                classes.get(ALL_FAIL_WITH_PROGRESS, 0) / n, 4) if n else 0.0,
            "all_fail_no_progress_share": round(
                classes.get(ALL_FAIL_NO_PROGRESS, 0) / n, 4) if n else 0.0,
            "mixed_effective_share": round(
                classes.get(MIXED_EFFECTIVE, 0) / n, 4) if n else 0.0,
            "low_variance_share": round(
                classes.get(LOW_VARIANCE, 0) / n, 4) if n else 0.0,
            "invalid_share": round(classes.get(INVALID_GROUP, 0) / n, 4) if n else 0.0,
            "group_class_counts": dict(classes),
            "distribution_by_call_bucket": dict(by_bucket),
            "bootstrap_status": self.bootstrap_status(),
        }

    def state_dict(self) -> Dict[str, Any]:
        base = super().state_dict()
        base.update({
            "sampler_mode": self.sampler_mode,
            "profile_quota": self.profile_quota.as_dict(),
            "raw_quota": self.raw_quota.as_dict(),
            "pool_actual": dict(self.pool_actual),
            "ext": self.ext,
            "bootstrap_complete": self.bootstrap_complete,
            "bootstrap_completed_at_step": self.bootstrap_completed_at_step,
            "unique_profile_prompts_seen": sorted(self.unique_profile_prompts_seen),
            "groups_observed_per_bucket": dict(self.groups_observed_per_bucket),
            "bootstrap_groups": list(self.bootstrap_groups),
            "_bootstrap_report_written": self._bootstrap_report_written,
        })
        return base

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        pq = state.get("profile_quota") or {}
        self.profile_quota = QuotaAccumulator(
            shares=dict(pq.get("shares_target") or NESTFUL_CALL_SHARES),
            actual=dict(pq.get("actual_counts") or {b: 0 for b in CALL_BUCKETS}),
            total=int(pq.get("total_effective") or 0),
        )
        rq = state.get("raw_quota") or {}
        self.raw_quota = QuotaAccumulator(
            shares=dict(rq.get("shares_target") or NESTFUL_CALL_SHARES),
            actual=dict(rq.get("actual_counts") or {b: 0 for b in CALL_BUCKETS}),
            total=int(rq.get("total_effective") or 0),
        )
        self.pool_actual = dict(state.get("pool_actual") or {"PROFILE": 0, "ENRICHMENT": 0})
        self.ext = dict(state.get("ext") or {})
        if state.get("sampler_mode"):
            self.sampler_mode = str(state["sampler_mode"])
        self.bootstrap_complete = bool(state.get("bootstrap_complete", False))
        self.bootstrap_completed_at_step = state.get("bootstrap_completed_at_step")
        self.unique_profile_prompts_seen = set(
            state.get("unique_profile_prompts_seen") or [])
        self.groups_observed_per_bucket = {
            b: int((state.get("groups_observed_per_bucket") or {}).get(b, 0))
            for b in CALL_BUCKETS
        }
        self.bootstrap_groups = list(state.get("bootstrap_groups") or [])
        self._bootstrap_report_written = bool(
            state.get("_bootstrap_report_written", False))


def nestful_refill_batch(
        sampler: NestfulProfileSampler,
        score_group: Callable[[PromptRef, int], GroupObservation],
        *, global_step: int,
        target_effective: Optional[int] = None,
) -> Dict[str, Any]:
    """DAPO-style refill with pool + call-bucket quota preservation."""
    cfg = sampler.cfg
    target = int(target_effective or cfg["target_effective_groups"])
    over0 = float(cfg["initial_oversample_factor"])
    over_max = float(cfg["max_oversample_factor"])
    rounds_cap = int(cfg["max_refill_rounds"])
    mode = sampler.sampler_mode
    profile_share = float(cfg.get("profile_share", 0.80))
    allow_cross = bool(cfg.get("allow_cross_pool_refill", False))

    kept: List[GroupObservation] = []
    rejected: List[GroupObservation] = []
    rounds = 0
    n_candidates = 0
    class_counts: Dict[str, int] = defaultdict(int)

    def need_pool() -> str:
        if mode != "dynamic_profile_plus_enrichment":
            return "PROFILE"
        total = max(1, sampler.pool_actual.get("PROFILE", 0)
                    + sampler.pool_actual.get("ENRICHMENT", 0) + len(kept))
        # among kept this round + history
        prof = sampler.pool_actual.get("PROFILE", 0) + sum(
            1 for o in kept if pool_of(sampler.by_id.get(o.prompt_id, PromptRef("x")))
            == "PROFILE")
        # target profile share of (already effective + this batch target)
        target_prof = profile_share * (sampler.profile_quota.total + target)
        if prof < target_prof:
            return "PROFILE"
        return "ENRICHMENT"

    max_raw_factor = float(cfg.get("max_raw_groups_per_update_factor", 3.0))
    max_raw = max(target, int(math.ceil(target * max_raw_factor)))

    while len(kept) < target and rounds < rounds_cap and n_candidates < max_raw:
        rounds += 1
        factor = min(over0 * (1.0 + 0.25 * (rounds - 1)), over_max)
        remaining_raw = max_raw - n_candidates
        batch_n = max(1, min(
            remaining_raw,
            int(math.ceil((target - len(kept)) * factor))))
        picks: List[PromptRef] = []
        for _ in range(batch_n):
            pool = need_pool()
            if pool == "ENRICHMENT" and not sampler.enrichment:
                pool = "PROFILE"
            if pool == "PROFILE":
                bucket = sampler.pick_profile_bucket()
            else:
                avail = sampler.available_counts("ENRICHMENT")
                bucket = "6+" if avail.get("6+", 0) > 0 else (
                    max(avail, key=avail.get) if any(avail.values()) else "6+")
            got = sampler.sample_from_bucket(pool, bucket, 1)
            if got:
                sampler.note_raw_candidate(
                    got[0].call_bucket or bucket, pool=pool)
            if not got and pool == "ENRICHMENT" and allow_cross:
                got = sampler.sample_from_bucket("PROFILE", bucket, 1)
            if not got and pool == "PROFILE":
                for b in CALL_BUCKETS:
                    got = sampler.sample_from_bucket(pool, b, 1)
                    if got:
                        break
            picks.extend(got)
        n_candidates += len(picks)
        for prompt in picks:
            obs = score_group(prompt, global_step)
            if obs is None:
                continue
            obs.call_bucket = obs.call_bucket or prompt.call_bucket
            obs.query_mode = obs.query_mode or prompt.query_mode
            effective = sampler.observe_group(obs)
            eps = float(cfg.get("reward_variance_epsilon") or 1e-6)
            user_cls = map_group_class(obs.group_class, obs, eps)
            class_counts[user_cls] += 1
            (kept if effective else rejected).append(obs)
            if len(kept) >= target:
                break
            if n_candidates >= max_raw:
                break

    n_groups = max(len(kept) + len(rejected), 1)
    by_bucket = defaultdict(int)
    by_mode = defaultdict(int)
    for o in kept:
        by_bucket[o.call_bucket] += 1
        by_mode[o.query_mode] += 1

    return {
        "global_step": global_step,
        "sampler_mode": mode,
        "refill_rounds": rounds,
        "candidate_prompt_count": n_candidates,
        "accepted_effective_groups": len(kept),
        "rejected_groups": len(rejected),
        "target_effective_groups": target,
        "max_raw_groups": max_raw,
        "effective_deficit": max(0, target - len(kept)),
        "target_reached": len(kept) >= target,
        "hit_raw_cap": n_candidates >= max_raw and len(kept) < target,
        "effective_group_rate": round(len(kept) / n_groups, 4),
        "dead_group_rate": round(len(rejected) / n_groups, 4),
        "group_classes": dict(class_counts),
        "effective_groups_by_call_bucket": dict(by_bucket),
        "effective_groups_by_query_mode": dict(by_mode),
        "profile_quota": sampler.profile_quota.as_dict(),
        "pool_actual": dict(sampler.pool_actual),
        "profile_effective_groups": sum(
            1 for o in kept
            if pool_of(sampler.by_id.get(o.prompt_id, PromptRef("x"))) == "PROFILE"),
        "enrichment_effective_groups": sum(
            1 for o in kept
            if pool_of(sampler.by_id.get(o.prompt_id, PromptRef("x"))) == "ENRICHMENT"),
        "kept": kept,
        "rejected": rejected,
    }


# Register modes on the shared SAMPLERS map (lazy to avoid circular import issues)
def register() -> None:
    from . import SAMPLERS
    SAMPLERS["uniform_profile"] = NestfulProfileSampler
    SAMPLERS["dynamic_profile"] = NestfulProfileSampler
    SAMPLERS["dynamic_profile_plus_enrichment"] = NestfulProfileSampler
    SAMPLERS["nestful_profile"] = NestfulProfileSampler
