"""Stages 24-26: tiered selection, transfer-oriented split, nested subsets.

Order matters and is the opposite of Pilot4.2's. The heldout *keys* are chosen
first -- whole workflow families, whole normalized capability plans, whole query
intent families, one surface track -- and every task carrying a heldout key leaves
the training candidate pool before a single training task is picked. A split built
the other way round can only be repaired by deleting keys, which is how leakage
gets in.

Two rules are absolute:

* a tier is a hard quota. If it cannot be filled from the valid pool, selection
  fails, the deficit is reported per stratum, and ``TRAINING_READY`` stays false.
  A deficit is never covered from another tier.
* the reserve is cut last, from tasks no threshold in this module ever inspected.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

from . import (CALL_BUCKETS, HELDOUT_PARTS, HELDOUT_TARGET,
               LONG_HORIZON_CALL_TARGETS, LONG_HORIZON_DEEP_MIX,
               PROFILE_CALL_TARGETS, RESERVE_TARGET, TIER_CAPABILITY,
               TIER_CHALLENGE, TIER_LONG_HORIZON, TIER_PROFILE_CORE, TIERS,
               TIER_TARGETS)

SELECTION_TRAIN_TRACKS = ("A_NATIVE", "G_GENERAL_1")
#: the surface holdout is the whole third track: a task whose tools are renamed is
#: not a new task, so the only honest surface generalisation test is a track the
#: training pool never contains.
SURFACE_HOLDOUT_TRACK = "G_GENERAL_2"


@dataclass
class Task:
    """A selectable task: candidate metadata plus its validated query."""
    task_id: str
    row: Dict[str, Any]
    query: Dict[str, Any]
    verified: Dict[str, Any]

    @property
    def workflow_id(self) -> str:
        return self.row["workflow_id"]

    @property
    def plan_key(self) -> str:
        return self.row["normalized_capability_sequence"]

    @property
    def topology_key(self) -> str:
        return f"{self.row['actual_primary_pattern']}|{self.row['call_bucket']}"

    @property
    def template_key(self) -> str:
        return self.query["fingerprints"]["intent_fingerprint"]

    @property
    def capability_combo(self) -> str:
        return "+".join(self.row["capability_families"])

    @property
    def track(self) -> str:
        return self.row["surface_track"]

    @property
    def bucket(self) -> str:
        return self.row["call_bucket"]

    @property
    def call_count(self) -> int:
        return int(self.row["call_count"])

    @property
    def answer_type(self) -> str:
        return self.row["answer_type"]

    @property
    def coding(self) -> bool:
        return bool(self.row["coding_like"])

    @property
    def mode(self) -> str:
        return self.query["actual_mode"]

    @property
    def difficulty(self) -> str:
        return self.row["difficulty_band"]


def build_pool(out_dir: Path) -> List[Task]:
    """Join the three stage outputs into selectable tasks."""
    from .pipeline import SHORTLIST, VERIFIED, iter_jsonl, read_jsonl
    from .qstage import QUERY_VALID

    rows = {r["task_id"]: r for r in read_jsonl(out_dir / SHORTLIST)}
    ver = {r["task_id"]: r for r in iter_jsonl(out_dir / VERIFIED)
           if r.get("selectable")}
    pool: List[Task] = []
    for q in read_jsonl(out_dir / QUERY_VALID):
        tid = q["task_id"]
        if tid in rows and tid in ver:
            pool.append(Task(tid, rows[tid], q, ver[tid]))
    pool.sort(key=lambda t: t.task_id)
    return pool


# ── heldout construction ─────────────────────────────────────────────────
@dataclass
class HeldoutKeys:
    workflows: Set[str] = field(default_factory=set)
    plans: Set[str] = field(default_factory=set)
    topologies: Set[str] = field(default_factory=set)
    templates: Set[str] = field(default_factory=set)
    combos: Set[str] = field(default_factory=set)
    track: str = SURFACE_HOLDOUT_TRACK

    def blocks(self, task: Task) -> str:
        if task.workflow_id in self.workflows:
            return "workflow_family"
        if task.plan_key in self.plans:
            return "program_plan"
        if task.topology_key in self.topologies:
            return "actual_topology"
        if task.template_key in self.templates:
            return "query_template"
        if task.capability_combo in self.combos:
            return "capability_combination"
        if task.track == self.track:
            return "surface"
        return ""


#: No single holdout part may claim more than this share of the whole valid pool.
#: Each part reserves *whole* key groups, and on a small pool a part asking for 100
#: tasks can otherwise take a third of everything, leaving the tiers unfillable for
#: a reason that has nothing to do with data quality.
MAX_PART_POOL_SHARE = 0.07


def _greedy_keys(pool: Sequence[Task], key_fn, want: int, *,
                 exclude=lambda t: False, min_group: int = 1,
                 max_group: int = 10 ** 9, seed: int = 11) -> Tuple[Set[str], int]:
    """Pick whole key groups until ``want`` tasks are covered.

    Groups are taken small-first so a holdout of 100 tasks does not swallow a
    thousand-task family, which would starve the training pool of a capability it
    still has to learn.
    """
    want = min(want, max(1, int(MAX_PART_POOL_SHARE * len(pool))))
    groups: Dict[str, List[Task]] = {}
    for task in pool:
        if exclude(task):
            continue
        groups.setdefault(key_fn(task), []).append(task)

    # a group bigger than twice the part is never worth reserving: the surplus is
    # removed from training without being testable, which is pure loss
    ceiling = min(max_group, max(min_group, 2 * want))
    usable = [(k, v) for k, v in groups.items()
              if min_group <= len(v) <= ceiling]
    rng = random.Random(seed)
    rng.shuffle(usable)
    usable.sort(key=lambda kv: len(kv[1]))
    chosen: Set[str] = set()
    covered = 0
    for key, tasks in usable:
        if covered >= want:
            break
        chosen.add(key)
        covered += len(tasks)
    return chosen, covered


def plan_heldout(pool: Sequence[Task], *, seed: int = 991) -> HeldoutKeys:
    """Reserve the keys the heldout parts own. Nothing is selected yet."""
    keys = HeldoutKeys()
    not_surface = lambda t: t.track == SURFACE_HOLDOUT_TRACK        # noqa: E731

    keys.workflows, _ = _greedy_keys(
        pool, lambda t: t.workflow_id, HELDOUT_PARTS["workflow_family"],
        exclude=not_surface, min_group=3, seed=seed)
    blocked = lambda t: not_surface(t) or t.workflow_id in keys.workflows  # noqa: E731
    keys.plans, _ = _greedy_keys(
        pool, lambda t: t.plan_key, HELDOUT_PARTS["program_plan"],
        exclude=blocked, min_group=2, seed=seed + 1)
    blocked2 = lambda t: blocked(t) or t.plan_key in keys.plans        # noqa: E731
    keys.combos, _ = _greedy_keys(
        pool, lambda t: t.capability_combo,
        HELDOUT_PARTS["capability_combination"], exclude=blocked2,
        min_group=2, seed=seed + 2)
    blocked3 = lambda t: blocked2(t) or t.capability_combo in keys.combos  # noqa: E731
    keys.templates, _ = _greedy_keys(
        pool, lambda t: t.template_key, HELDOUT_PARTS["query_template"],
        exclude=blocked3, min_group=1, seed=seed + 3)
    blocked4 = lambda t: blocked3(t) or t.template_key in keys.templates  # noqa: E731
    # topologies are few and each is large, so this holdout takes the rarest ones
    keys.topologies, _ = _greedy_keys(
        pool, lambda t: t.topology_key, HELDOUT_PARTS["actual_topology"],
        exclude=blocked4, min_group=1, max_group=400, seed=seed + 4)
    return keys


def cut_heldout(pool: Sequence[Task], keys: HeldoutKeys,
                seed: int = 1234) -> Tuple[Dict[str, List[Task]], List[Task]]:
    """Split the pool into the seven heldout parts and the training candidates."""
    rng = random.Random(seed)
    parts: Dict[str, List[Task]] = {name: [] for name in HELDOUT_PARTS}
    train_candidates: List[Task] = []
    by_part: Dict[str, List[Task]] = {name: [] for name in HELDOUT_PARTS}
    for task in pool:
        reason = keys.blocks(task)
        if reason:
            by_part[reason].append(task)
        else:
            train_candidates.append(task)
    for name in ("workflow_family", "program_plan", "actual_topology",
                 "query_template", "capability_combination", "surface"):
        group = by_part[name]
        rng.shuffle(group)
        parts[name] = _stratified_take(group, HELDOUT_PARTS[name], seed=seed)
    # the standard profile holdout is drawn from the training-eligible pool and
    # then removed from it, so it measures in-distribution generalisation
    rng.shuffle(train_candidates)
    profile_want = min(HELDOUT_PARTS["standard_profile"],
                       max(1, int(MAX_PART_POOL_SHARE * len(pool))))
    profile_part = _profile_take(train_candidates, profile_want, seed=seed)
    taken = {t.task_id for t in profile_part}
    parts["standard_profile"] = profile_part
    train_candidates = [t for t in train_candidates if t.task_id not in taken]
    # an intent template used by the profile holdout must not stay in train either
    held_templates = {t.template_key for t in profile_part}
    train_candidates = [t for t in train_candidates
                        if t.template_key not in held_templates]
    return parts, train_candidates


def _stratified_take(group: Sequence[Task], want: int,
                     seed: int = 7) -> List[Task]:
    """Take ``want`` tasks spread over call buckets and answer types."""
    if len(group) <= want:
        return list(group)
    buckets: Dict[str, List[Task]] = {}
    for task in group:
        buckets.setdefault(f"{task.bucket}|{task.answer_type}", []).append(task)
    order = sorted(buckets)
    rng = random.Random(seed)
    for key in order:
        rng.shuffle(buckets[key])
    out: List[Task] = []
    i = 0
    while len(out) < want and any(buckets[k] for k in order):
        key = order[i % len(order)]
        if buckets[key]:
            out.append(buckets[key].pop())
        i += 1
    return out[:want]


def _profile_take(group: Sequence[Task], want: int, seed: int = 7) -> List[Task]:
    """Take ``want`` tasks in the PROFILE_CORE call-count proportions."""
    quota = {b: int(round(want * PROFILE_CALL_TARGETS[b][0])) for b in CALL_BUCKETS}
    by_bucket: Dict[str, List[Task]] = {b: [] for b in CALL_BUCKETS}
    for task in group:
        by_bucket.setdefault(task.bucket, []).append(task)
    rng = random.Random(seed)
    out: List[Task] = []
    for bucket in CALL_BUCKETS:
        pool = by_bucket.get(bucket, [])
        rng.shuffle(pool)
        out.extend(pool[:quota[bucket]])
    return out[:want]


# ── tier selection ───────────────────────────────────────────────────────
@dataclass
class TierResult:
    tier: str
    target: int
    tasks: List[Task]
    deficits: Dict[str, int]
    notes: List[str]

    @property
    def met(self) -> bool:
        return len(self.tasks) == self.target


def select_profile_core(pool: List[Task], target: int,
                        seed: int = 202607) -> TierResult:
    """Match the dev call-count distribution exactly; spread everything else."""
    quota = {b: int(round(target * PROFILE_CALL_TARGETS[b][0]))
             for b in CALL_BUCKETS}
    drift = target - sum(quota.values())
    quota["2"] += drift
    by_bucket: Dict[str, List[Task]] = {b: [] for b in CALL_BUCKETS}
    for task in pool:
        by_bucket.setdefault(task.bucket, []).append(task)
    rng = random.Random(seed)
    chosen: List[Task] = []
    deficits: Dict[str, int] = {}
    for bucket in CALL_BUCKETS:
        want = quota[bucket]
        have = by_bucket.get(bucket, [])
        rng.shuffle(have)
        picked = _spread(have, want, seed=seed)
        chosen.extend(picked)
        if len(picked) < want:
            deficits[f"call_bucket={bucket}"] = want - len(picked)
    return TierResult(TIER_PROFILE_CORE, target, chosen, deficits,
                      [f"quota={quota}"])


def select_long_horizon(pool: List[Task], target: int,
                        seed: int = 909) -> TierResult:
    quota = {b: int(round(target * s))
             for b, s in LONG_HORIZON_CALL_TARGETS.items()}
    quota["4"] += target - sum(quota.values())
    deep_target = quota["6+"]
    deep_quota = {n: int(round(deep_target * (lo + hi) / 2))
                  for n, (lo, hi) in LONG_HORIZON_DEEP_MIX.items()}
    rng = random.Random(seed)
    chosen: List[Task] = []
    deficits: Dict[str, int] = {}
    for bucket in ("4", "5"):
        have = [t for t in pool if t.bucket == bucket]
        rng.shuffle(have)
        picked = _spread(have, quota[bucket], seed=seed)
        chosen.extend(picked)
        if len(picked) < quota[bucket]:
            deficits[f"call_bucket={bucket}"] = quota[bucket] - len(picked)
    for n, want in sorted(deep_quota.items()):
        have = [t for t in pool if t.call_count == n]
        rng.shuffle(have)
        picked = _spread(have, want, seed=seed + n)
        chosen.extend(picked)
        if len(picked) < want:
            deficits[f"calls={n}"] = want - len(picked)
    return TierResult(TIER_LONG_HORIZON, target, chosen, deficits,
                      [f"quota={quota}", f"deep_quota={deep_quota}"])


def select_capability(pool: List[Task], target: int,
                      seed: int = 313) -> TierResult:
    """100 % coding/generic tasks, maximising distinct coding primitives."""
    coding = [t for t in pool if t.coding]
    rng = random.Random(seed)
    rng.shuffle(coding)
    # cover rare coding primitives first: the tier exists to widen actual usage
    freq: Dict[str, int] = {}
    for task in coding:
        for pid in task.row["primitives"]:
            freq[pid] = freq.get(pid, 0) + 1
    coding.sort(key=lambda t: min((freq[p] for p in t.row["primitives"]),
                                 default=10 ** 6))
    want_structured = int(round(target * 0.30))
    structured = [t for t in coding
                  if t.answer_type in ("string", "list", "object")]
    chosen: List[Task] = []
    seen: Set[str] = set()
    for task in structured[:want_structured]:
        chosen.append(task)
        seen.add(task.task_id)
    for task in coding:
        if len(chosen) >= target:
            break
        if task.task_id in seen:
            continue
        chosen.append(task)
        seen.add(task.task_id)
    deficits = {}
    if len(chosen) < target:
        deficits["coding_tasks"] = target - len(chosen)
    long_share = sum(1 for t in chosen if t.bucket == "6+") / max(1, len(chosen))
    five_plus = sum(1 for t in chosen if t.call_count >= 5) / max(1, len(chosen))
    notes = [f"6+_share={long_share:.3f}", f"5+_share={five_plus:.3f}"]
    return TierResult(TIER_CAPABILITY, target, chosen[:target], deficits, notes)


def select_challenge(pool: List[Task], target: int,
                     seed: int = 77) -> TierResult:
    """The hardest validated tasks: long, joined, non-numeric, many distractors."""
    def score(task: Task) -> Tuple[int, int, int]:
        feats = task.row["graph_features"]
        return (task.call_count + 2 * int(feats.get("n_join_nodes", 0) > 1)
                + int(feats.get("n_late_edges", 0) > 0)
                + int(task.answer_type in ("list", "object", "category")),
                int(task.verified.get("offered", {}).get("distractor_count", 0)),
                int(feats.get("depth", 0)))
    ranked = sorted(pool, key=score, reverse=True)
    chosen = _spread(ranked, target, seed=seed, preserve_order=True)
    deficits = {} if len(chosen) == target else {"challenge": target - len(chosen)}
    return TierResult(TIER_CHALLENGE, target, chosen, deficits, [])


def _spread(candidates: Sequence[Task], want: int, *, seed: int = 0,
            preserve_order: bool = False) -> List[Task]:
    """Round-robin over (workflow, pattern, answer type) so no stratum dominates."""
    if want <= 0 or not candidates:
        return []
    groups: Dict[str, List[Task]] = {}
    for task in candidates:
        key = (f"{task.workflow_id}|{task.row['actual_primary_pattern']}"
               f"|{task.answer_type}")
        groups.setdefault(key, []).append(task)
    order = sorted(groups)
    if not preserve_order:
        random.Random(seed).shuffle(order)
    out: List[Task] = []
    i = 0
    remaining = sum(len(v) for v in groups.values())
    while len(out) < want and remaining:
        key = order[i % len(order)]
        if groups[key]:
            out.append(groups[key].pop(0))
            remaining -= 1
        i += 1
    return out


def select_tiers(train_candidates: List[Task],
                 targets: Dict[str, int] | None = None,
                 seed: int = 20260731) -> Tuple[Dict[str, TierResult],
                                                List[Task]]:
    """Fill the four tiers from disjoint task sets, in priority order."""
    targets = targets or dict(TIER_TARGETS)
    available = list(train_candidates)
    results: Dict[str, TierResult] = {}
    # CHALLENGE and CAPABILITY are the scarce tiers, so they pick first; the
    # PROFILE_CORE distribution is the easiest to satisfy from what is left.
    for tier, fn in ((TIER_CHALLENGE, select_challenge),
                     (TIER_CAPABILITY, select_capability),
                     (TIER_LONG_HORIZON, select_long_horizon),
                     (TIER_PROFILE_CORE, select_profile_core)):
        res = fn(available, targets[tier], seed=seed + len(tier))
        taken = {t.task_id for t in res.tasks}
        available = [t for t in available if t.task_id not in taken]
        results[tier] = res
    return results, available


# ── nested subsets ───────────────────────────────────────────────────────
NESTED_STRATA = ("call_bucket", "actual_primary_pattern", "workflow_family",
                 "answer_type", "query_mode", "surface_track", "difficulty_band",
                 "selection_tier", "coding", "boolean_label")


def _strata_key(task: Task, tier: str) -> str:
    return "|".join([
        task.bucket, task.row["actual_primary_pattern"],
        task.workflow_id.split(".")[0], task.answer_type, task.mode,
        task.track, task.difficulty, tier,
        "coding" if task.coding else "plain",
        str(task.row.get("boolean_label")),
    ])


def nested_subsets(master: Sequence[Tuple[Task, str]],
                   sizes: Sequence[int], seed: int = 5) -> Dict[int, List[str]]:
    """Nested stratified subsets: each size is a superset of the previous one.

    Built by ranking every task once and taking prefixes, so nesting is automatic.
    The rank of the ``j``-th task of a stratum holding ``m`` tasks is
    ``(j + 0.5) / m``: a stratum with twice the members places twice as many tasks
    in any prefix, which is what keeps a 1000-task prefix proportional to the
    5000-task master rather than uniform over strata. Round-robin interleaving
    cannot do that -- with more strata than prefix slots it gives every stratum one
    task and flattens the distribution the profile was built to match.
    """
    groups: Dict[str, List[Tuple[Task, str]]] = {}
    for task, tier in master:
        groups.setdefault(_strata_key(task, tier), []).append((task, tier))
    rng = random.Random(seed)
    ranked: List[Tuple[float, str, str]] = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda pair: pair[0].task_id)
        # a per-stratum offset keeps two equally sized strata from always
        # contributing in the same order without breaking determinism
        jitter = rng.random()
        size = len(members)
        for j, (task, _tier) in enumerate(members):
            ranked.append(((j + jitter) / size, key, task.task_id))
    ranked.sort(key=lambda triple: (triple[0], triple[1], triple[2]))
    order = [task_id for _rank, _key, task_id in ranked]
    return {size: order[:size] for size in sorted(sizes)}


# ── overlap verification ─────────────────────────────────────────────────
def split_overlap_report(train: Sequence[Task],
                         parts: Dict[str, Sequence[Task]]) -> Dict[str, Any]:
    """Recompute every holdout rule from the selected tasks themselves."""
    t_ids = {t.task_id for t in train}
    t_workflows = {t.workflow_id for t in train}
    t_plans = {t.plan_key for t in train}
    t_topos = {t.topology_key for t in train}
    t_templates = {t.template_key for t in train}
    t_combos = {t.capability_combo for t in train}
    t_tracks = {t.track for t in train}
    t_instances = {t.row["workflow_instance_id"] for t in train}
    t_programs = {t.row["program_fingerprint"] for t in train}

    def overlap(tasks: Sequence[Task], attr, against: Set[str]) -> int:
        return len({attr(t) for t in tasks} & against)

    report: Dict[str, Any] = {
        "train_size": len(train),
        "instance_leakage": sum(
            1 for part in parts.values() for t in part
            if t.row["workflow_instance_id"] in t_instances),
        "program_leakage": sum(
            1 for part in parts.values() for t in part
            if t.row["program_fingerprint"] in t_programs),
        "task_id_leakage": sum(1 for part in parts.values() for t in part
                               if t.task_id in t_ids),
        "train_surface_tracks": sorted(t_tracks),
    }
    report["workflow_holdout_overlap"] = overlap(
        parts.get("workflow_family", []), lambda t: t.workflow_id, t_workflows)
    report["program_plan_holdout_overlap"] = overlap(
        parts.get("program_plan", []), lambda t: t.plan_key, t_plans)
    report["actual_topology_holdout_overlap"] = overlap(
        parts.get("actual_topology", []), lambda t: t.topology_key, t_topos)
    report["query_template_holdout_overlap"] = overlap(
        parts.get("query_template", []), lambda t: t.template_key, t_templates)
    report["capability_combination_holdout_overlap"] = overlap(
        parts.get("capability_combination", []),
        lambda t: t.capability_combo, t_combos)
    surface = parts.get("surface", [])
    report["surface_holdout_respects_config"] = bool(
        surface) and all(t.track == SURFACE_HOLDOUT_TRACK for t in surface) \
        and SURFACE_HOLDOUT_TRACK not in t_tracks
    report["standard_profile_template_overlap"] = overlap(
        parts.get("standard_profile", []), lambda t: t.template_key, t_templates)
    hard_rules = {
        "instance_leakage": 0, "program_leakage": 0, "task_id_leakage": 0,
        "workflow_holdout_overlap": 0, "program_plan_holdout_overlap": 0,
        "actual_topology_holdout_overlap": 0, "query_template_holdout_overlap": 0,
        "capability_combination_holdout_overlap": 0,
    }
    violations = [f"{k}={report[k]} (must be {v})"
                  for k, v in hard_rules.items() if report[k] != v]
    if not report["surface_holdout_respects_config"]:
        violations.append("surface holdout does not respect the configured track")
    report["passed"] = not violations
    report["violations"] = violations
    return report
