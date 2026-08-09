"""Adaptive prompt sampling for MT-GRPO (Phase O).

Interfaces and state only. Nothing here launches a rollout: the trainer calls
``sample_candidates`` before generating and ``observe_group`` after scoring, and
everything the sampler learns lives in ``state_dict`` so a checkpoint resume
restores the curriculum exactly.

The weight is an explicit, inspectable product of named components rather than
a learned score, so a training run can always answer "why was this prompt
sampled".
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "ttdf.sampler.v1"
SAMPLER_VERSION = "ttdf.sampler.2026.07"

# ── group taxonomy ────────────────────────────────────────────────────────
ALL_CORRECT = "ALL_CORRECT"
ALL_FAIL_NO_PROGRESS = "ALL_FAIL_NO_PROGRESS"
ALL_FAIL_WITH_PROCESS_VARIANCE = "ALL_FAIL_WITH_PROCESS_VARIANCE"
MIXED_TERMINAL = "MIXED_TERMINAL"
MIXED_PROCESS_ONLY = "MIXED_PROCESS_ONLY"
MIXED_BOTH = "MIXED_BOTH"
EQUAL_PARTIAL = "EQUAL_PARTIAL"
INVALID_GROUP = "INVALID_GROUP"

GROUP_CLASSES = [ALL_CORRECT, ALL_FAIL_NO_PROGRESS, ALL_FAIL_WITH_PROCESS_VARIANCE,
                 MIXED_TERMINAL, MIXED_PROCESS_ONLY, MIXED_BOTH, EQUAL_PARTIAL,
                 INVALID_GROUP]

# ── curriculum states ─────────────────────────────────────────────────────
LOCKED, PROBING, ACTIVE, MASTERED, TOO_HARD = \
    "LOCKED", "PROBING", "ACTIVE", "MASTERED", "TOO_HARD"
CURRICULUM_STATES = [LOCKED, PROBING, ACTIVE, MASTERED, TOO_HARD]


DEFAULT_CONFIG: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "sampler_version": SAMPLER_VERSION,
    "epsilon_reward_std": 1e-3,
    "epsilon_process_std": 1e-3,
    "ema_decay": 0.85,
    "frontier_center": 0.5,
    "frontier_sharpness": 6.0,
    "staleness_halflife": 200.0,
    "novelty_bonus": 0.5,
    "repeat_penalty": 0.35,
    "minimum_sampling_probability": 1e-4,
    "maximum_prompt_reuse_per_epoch": 4,
    "candidate_prompt_batch_size": 32,
    "target_effective_prompt_count": 16,
    "maximum_refill_rounds": 4,
    "minimum_cell_quota": 1,
    "weights": {
        "base_cell_weight": 1.0,
        "frontier_weight": 1.0,
        "variance_weight": 1.0,
        "staleness_weight": 0.5,
        "novelty_weight": 0.5,
        "repeat_penalty": 1.0,
    },
    "curriculum": {
        "enabled": False,
        "probe_group_count": 8,
        "unlock_effective_group_rate": 0.30,
        "activate_effective_group_rate": 0.40,
        "master_ema_success": 0.85,
        "master_max_variance": 0.02,
        "too_hard_all_fail_streak": 12,
        "prerequisite_master_share": 0.6,
    },
}


# ── data carriers ─────────────────────────────────────────────────────────
@dataclass
class PromptRef:
    prompt_id: str
    generation_cell: str = ""
    semantic_program_family: str = ""
    call_bucket: str = ""
    pattern_family: str = ""
    query_mode: str = ""
    capability_families: List[str] = field(default_factory=list)
    difficulty_band: str = ""
    difficulty_signature: Dict[str, Any] = field(default_factory=dict)
    weight_components: Dict[str, float] = field(default_factory=dict)
    selection_weight: float = 0.0
    pool: str = "PROFILE"  # PROFILE | ENRICHMENT

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroupObservation:
    global_step: int
    prompt_id: str
    generation_cell: str = ""
    semantic_program_family: str = ""
    difficulty_signature: Dict[str, Any] = field(default_factory=dict)
    group_size: int = 0
    terminal_rewards: List[float] = field(default_factory=list)
    process_rewards: List[float] = field(default_factory=list)
    total_rewards: List[float] = field(default_factory=list)
    parse_flags: List[bool] = field(default_factory=list)
    executable_flags: List[bool] = field(default_factory=list)
    call_bucket: str = ""
    pattern_family: str = ""
    query_mode: str = ""
    capability_families: List[str] = field(default_factory=list)
    difficulty_band: str = ""

    # derived
    reward_mean: float = 0.0
    reward_std: float = 0.0
    terminal_success_rate: float = 0.0
    process_std_within_terminal_class: float = 0.0
    parse_success_rate: float = 0.0
    executable_rate: float = 0.0
    group_class: str = INVALID_GROUP

    def __post_init__(self) -> None:
        if not self.total_rewards and (self.terminal_rewards or self.process_rewards):
            n = max(len(self.terminal_rewards), len(self.process_rewards))
            t = list(self.terminal_rewards) + [0.0] * (n - len(self.terminal_rewards))
            p = list(self.process_rewards) + [0.0] * (n - len(self.process_rewards))
            self.total_rewards = [a + b for a, b in zip(t, p)]
        if not self.group_size:
            self.group_size = len(self.total_rewards)
        self.recompute()

    def recompute(self, *, success_threshold: float = 0.5) -> None:
        self.reward_mean = _mean(self.total_rewards)
        self.reward_std = _std(self.total_rewards)
        wins = [1.0 if r >= success_threshold else 0.0 for r in self.terminal_rewards]
        self.terminal_success_rate = _mean(wins) if wins else 0.0
        self.parse_success_rate = _mean([1.0 if x else 0.0 for x in self.parse_flags]) \
            if self.parse_flags else 0.0
        self.executable_rate = _mean([1.0 if x else 0.0 for x in self.executable_flags]) \
            if self.executable_flags else 0.0
        self.process_std_within_terminal_class = _within_class_std(
            self.terminal_rewards, self.process_rewards, success_threshold)
        self.group_class = classify_group(self)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _within_class_std(terminal: Sequence[float], process: Sequence[float],
                      threshold: float) -> float:
    """Process spread *inside* each terminal outcome class.

    An all-fail group with different process rewards still carries gradient, so
    this is what separates ALL_FAIL_NO_PROGRESS from ALL_FAIL_WITH_PROCESS_VARIANCE.
    """
    if not process:
        return 0.0
    if not terminal:
        return _std(process)
    wins = [p for t, p in zip(terminal, process) if t >= threshold]
    losses = [p for t, p in zip(terminal, process) if t < threshold]
    stds = [_std(g) for g in (wins, losses) if len(g) >= 2]
    return max(stds) if stds else 0.0


def classify_group(obs: GroupObservation, *,
                   eps_reward: float = 1e-3,
                   eps_process: float = 1e-3,
                   success_threshold: float = 0.5) -> str:
    if obs.group_size < 2 or not obs.total_rewards:
        return INVALID_GROUP
    sr = obs.terminal_success_rate
    terminal_varies = 0.0 < sr < 1.0
    process_varies = obs.process_std_within_terminal_class > eps_process
    if sr >= 1.0 - 1e-9:
        return ALL_CORRECT
    if sr <= 1e-9:
        return (ALL_FAIL_WITH_PROCESS_VARIANCE if process_varies
                else ALL_FAIL_NO_PROGRESS)
    if terminal_varies and process_varies:
        return MIXED_BOTH
    if terminal_varies:
        return MIXED_TERMINAL
    if process_varies:
        return MIXED_PROCESS_ONLY
    if obs.reward_std <= eps_reward:
        return EQUAL_PARTIAL
    return MIXED_PROCESS_ONLY


def is_effective(obs: GroupObservation, cfg: Optional[Dict[str, Any]] = None) -> bool:
    """A group is kept when it can produce a non-degenerate advantage."""
    cfg = cfg or DEFAULT_CONFIG
    if obs.group_class == INVALID_GROUP:
        return False
    if obs.reward_std > cfg["epsilon_reward_std"]:
        return True
    if 0.0 < obs.terminal_success_rate < 1.0:
        return True
    return obs.process_std_within_terminal_class > cfg["epsilon_process_std"]


# ── history ───────────────────────────────────────────────────────────────
@dataclass
class HistoryEntry:
    n_sampled: int = 0
    n_rollouts: int = 0
    ema_terminal_success: float = 0.0
    ema_total_reward: float = 0.0
    ema_reward_variance: float = 0.0
    ema_executable_rate: float = 0.0
    all_correct_count: int = 0
    all_fail_count: int = 0
    effective_group_count: int = 0
    group_count: int = 0
    last_sampled_step: int = -1
    consecutive_rejections: int = 0
    consecutive_all_fail_no_progress: int = 0
    selection_weight: float = 0.0

    def update(self, obs: GroupObservation, *, decay: float, step: int,
               effective: bool) -> None:
        self.n_rollouts += obs.group_size
        self.group_count += 1
        self.last_sampled_step = step
        w = decay if self.group_count > 1 else 0.0
        self.ema_terminal_success = _ema(self.ema_terminal_success,
                                         obs.terminal_success_rate, w)
        self.ema_total_reward = _ema(self.ema_total_reward, obs.reward_mean, w)
        self.ema_reward_variance = _ema(self.ema_reward_variance,
                                        obs.reward_std ** 2, w)
        self.ema_executable_rate = _ema(self.ema_executable_rate,
                                        obs.executable_rate, w)
        if obs.group_class == ALL_CORRECT:
            self.all_correct_count += 1
        if obs.group_class in (ALL_FAIL_NO_PROGRESS, ALL_FAIL_WITH_PROCESS_VARIANCE):
            self.all_fail_count += 1
        # tracked apart from consecutive_rejections: an all-correct group is
        # also rejected, and a mastered cell must not be read as too hard
        if obs.group_class == ALL_FAIL_NO_PROGRESS:
            self.consecutive_all_fail_no_progress += 1
        else:
            self.consecutive_all_fail_no_progress = 0
        if effective:
            self.effective_group_count += 1
            self.consecutive_rejections = 0
        else:
            self.consecutive_rejections += 1

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ema(old: float, new: float, decay: float) -> float:
    return float(decay * old + (1.0 - decay) * new)


class SamplerState:
    """Everything a sampler remembers, per prompt and per aggregate axis."""

    AXES = ("generation_cell", "call_bucket", "pattern_family", "query_mode",
            "capability_family", "difficulty_band")

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = {**DEFAULT_CONFIG, **(cfg or {})}
        self.global_step = 0
        self.epoch = 0
        self.prompt: Dict[str, HistoryEntry] = {}
        self.axis: Dict[str, Dict[str, HistoryEntry]] = {a: {} for a in self.AXES}
        self.curriculum: Dict[str, str] = {}
        self.epoch_use: Dict[str, int] = {}
        self.n_observed = 0

    # -- history access
    def prompt_entry(self, prompt_id: str) -> HistoryEntry:
        return self.prompt.setdefault(prompt_id, HistoryEntry())

    def axis_entry(self, axis: str, key: str) -> HistoryEntry:
        return self.axis.setdefault(axis, {}).setdefault(key, HistoryEntry())

    def observe(self, obs: GroupObservation) -> bool:
        effective = is_effective(obs, self.cfg)
        decay = float(self.cfg["ema_decay"])
        self.global_step = max(self.global_step, obs.global_step)
        self.n_observed += 1
        self.prompt_entry(obs.prompt_id).update(
            obs, decay=decay, step=obs.global_step, effective=effective)
        axis_keys = {
            "generation_cell": [obs.generation_cell],
            "call_bucket": [obs.call_bucket],
            "pattern_family": [obs.pattern_family],
            "query_mode": [obs.query_mode],
            "capability_family": list(obs.capability_families),
            "difficulty_band": [obs.difficulty_band],
        }
        for axis, keys in axis_keys.items():
            for key in keys:
                if key:
                    self.axis_entry(axis, key).update(
                        obs, decay=decay, step=obs.global_step, effective=effective)
        return effective

    # -- serialisation
    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "sampler_version": SAMPLER_VERSION,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "n_observed": self.n_observed,
            "config": self.cfg,
            "prompt": {k: v.as_dict() for k, v in sorted(self.prompt.items())},
            "axis": {a: {k: v.as_dict() for k, v in sorted(d.items())}
                     for a, d in sorted(self.axis.items())},
            "curriculum": dict(sorted(self.curriculum.items())),
            "epoch_use": dict(sorted(self.epoch_use.items())),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.cfg = {**DEFAULT_CONFIG, **(state.get("config") or {})}
        self.global_step = int(state.get("global_step", 0))
        self.epoch = int(state.get("epoch", 0))
        self.n_observed = int(state.get("n_observed", 0))
        self.prompt = {k: HistoryEntry(**v)
                       for k, v in (state.get("prompt") or {}).items()}
        self.axis = {a: {k: HistoryEntry(**v) for k, v in d.items()}
                     for a, d in (state.get("axis") or {}).items()}
        for a in self.AXES:
            self.axis.setdefault(a, {})
        self.curriculum = dict(state.get("curriculum") or {})
        self.epoch_use = dict(state.get("epoch_use") or {})


# ── samplers ──────────────────────────────────────────────────────────────
class PromptSampler:
    """Base interface. Subclasses only override ``weight_components``."""

    name = "base"

    def __init__(self, prompts: Sequence[PromptRef], *,
                 config: Optional[Dict[str, Any]] = None, seed: int = 0) -> None:
        self.prompts = list(prompts)
        self.by_id = {p.prompt_id: p for p in self.prompts}
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.seed = seed
        self.rng = random.Random(seed)
        self.state = SamplerState(self.cfg)

    # -- API
    def sample_candidates(self, n: int,
                          state: Optional[SamplerState] = None) -> List[PromptRef]:
        st = state or self.state
        pool = [p for p in self.prompts if self._selectable(p, st)] or self.prompts
        weights = []
        floor = float(self.cfg["minimum_sampling_probability"])
        for p in pool:
            comps = self.weight_components(p, st)
            w = max(self._combine(comps), floor)
            p.weight_components = {k: round(v, 6) for k, v in comps.items()}
            p.selection_weight = w
            st.prompt_entry(p.prompt_id).selection_weight = w
            weights.append(w)
        picks = _weighted_sample_without_replacement(pool, weights, n, self.rng)
        for p in picks:
            entry = st.prompt_entry(p.prompt_id)
            entry.n_sampled += 1
            st.epoch_use[p.prompt_id] = st.epoch_use.get(p.prompt_id, 0) + 1
        return picks

    def observe_group(self, observation: GroupObservation) -> bool:
        effective = self.state.observe(observation)
        if self.cfg["curriculum"]["enabled"]:
            update_curriculum(self.state, self.cfg)
        return effective

    def state_dict(self) -> Dict[str, Any]:
        return {**self.state.state_dict(), "name": self.name, "seed": self.seed,
                "rng": self.rng.getstate()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.state.load_state_dict(state)
        rng_state = state.get("rng")
        if rng_state is not None:
            self.rng.setstate(_restore_rng_state(rng_state))

    # -- hooks
    def weight_components(self, prompt: PromptRef,
                          state: SamplerState) -> Dict[str, float]:
        return {"base_cell_weight": 1.0}

    def _combine(self, comps: Dict[str, float]) -> float:
        """base * prod(1 + w_k * component_k) * (1 - w_rep * repeat_penalty).

        Multiplicative so each component's contribution stays legible in the
        log: the stored components and the config weights reproduce the number.
        """
        w = self.cfg["weights"]
        base = comps.get("base_cell_weight", 1.0) * w.get("base_cell_weight", 1.0)
        total = max(base, 1e-6)
        for key, value in comps.items():
            if key in ("base_cell_weight", "repeat_penalty"):
                continue
            total *= max(1.0 + w.get(key, 1.0) * value, 1e-3)
        total *= max(1.0 - w.get("repeat_penalty", 1.0)
                     * comps.get("repeat_penalty", 0.0), 1e-3)
        return total

    def _selectable(self, prompt: PromptRef, state: SamplerState) -> bool:
        if state.epoch_use.get(prompt.prompt_id, 0) >= \
                self.cfg["maximum_prompt_reuse_per_epoch"]:
            return False
        if self.cfg["curriculum"]["enabled"]:
            status = state.curriculum.get(prompt.generation_cell, PROBING)
            if status in (LOCKED, MASTERED):
                return False
        return True


class UniformPromptSampler(PromptSampler):
    name = "uniform"


class DynamicEffectiveGroupSampler(PromptSampler):
    """Up-weights prompts whose recent groups actually carried gradient."""

    name = "dynamic_effective_group"

    def weight_components(self, prompt: PromptRef,
                          state: SamplerState) -> Dict[str, float]:
        e = state.prompt.get(prompt.prompt_id)
        if e is None or e.group_count == 0:
            return {"base_cell_weight": 1.0, "novelty_weight": 1.0}
        eff_rate = e.effective_group_count / max(e.group_count, 1)
        return {
            "base_cell_weight": 1.0,
            "variance_weight": eff_rate,
            "repeat_penalty": min(e.consecutive_rejections / 4.0, 1.0),
        }


class HistoryAdaptivePromptSampler(PromptSampler):
    """Frontier + variance + staleness + novelty, each stored separately."""

    name = "history_adaptive"

    def weight_components(self, prompt: PromptRef,
                          state: SamplerState) -> Dict[str, float]:
        cfg = self.cfg
        cell = state.axis["generation_cell"].get(prompt.generation_cell)
        e = state.prompt.get(prompt.prompt_id)
        comps: Dict[str, float] = {"base_cell_weight": 1.0}

        if cell and cell.group_count:
            cell_eff = cell.effective_group_count / max(cell.group_count, 1)
            comps["base_cell_weight"] = 0.5 + cell_eff
        if e is None or e.group_count == 0:
            comps["novelty_weight"] = float(cfg["novelty_bonus"])
            return comps

        # frontier: prompts near 50 % success carry the most signal
        d = e.ema_terminal_success - float(cfg["frontier_center"])
        comps["frontier_weight"] = math.exp(-float(cfg["frontier_sharpness"]) * d * d)
        comps["variance_weight"] = min(e.ema_reward_variance * 4.0, 1.0)
        age = max(state.global_step - e.last_sampled_step, 0)
        comps["staleness_weight"] = 1.0 - math.exp(
            -age / max(float(cfg["staleness_halflife"]), 1.0))
        comps["repeat_penalty"] = min(
            e.consecutive_rejections * float(cfg["repeat_penalty"]), 1.0)
        return comps


class CellCurriculumSampler(HistoryAdaptivePromptSampler):
    """History-adaptive weighting restricted to unlocked cells."""

    name = "cell_curriculum"

    def __init__(self, prompts: Sequence[PromptRef], *,
                 config: Optional[Dict[str, Any]] = None, seed: int = 0,
                 prerequisites: Optional[Dict[str, List[str]]] = None) -> None:
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        cfg["curriculum"] = {**DEFAULT_CONFIG["curriculum"],
                             **(cfg.get("curriculum") or {}), "enabled": True}
        super().__init__(prompts, config=cfg, seed=seed)
        self.prerequisites = dict(prerequisites or {})
        for p in self.prompts:
            if p.generation_cell not in self.state.curriculum:
                self.state.curriculum[p.generation_cell] = (
                    LOCKED if self.prerequisites.get(p.generation_cell) else PROBING)

    def observe_group(self, observation: GroupObservation) -> bool:
        effective = self.state.observe(observation)
        update_curriculum(self.state, self.cfg, self.prerequisites)
        return effective

    def easier_sibling(self, cell_id: str) -> Optional[str]:
        """TOO_HARD cells fall back to a sibling, never to a dropped capability."""
        target = None
        for p in self.prompts:
            if p.generation_cell == cell_id:
                target = p
                break
        if target is None:
            return None
        order = {"easy": 0, "medium": 1, "hard": 2}
        best: Optional[Tuple[int, str]] = None
        for p in self.prompts:
            if p.generation_cell == cell_id:
                continue
            if p.call_bucket != target.call_bucket:
                continue
            if not set(p.capability_families) & set(target.capability_families):
                continue
            rank = order.get(p.difficulty_band, 3)
            if rank < order.get(target.difficulty_band, 3):
                if best is None or rank < best[0]:
                    best = (rank, p.generation_cell)
        return best[1] if best else None


SAMPLERS: Dict[str, Any] = {
    "uniform": UniformPromptSampler,
    "dynamic": DynamicEffectiveGroupSampler,
    "dynamic_effective_group": DynamicEffectiveGroupSampler,
    "history_adaptive": HistoryAdaptivePromptSampler,
    "curriculum": CellCurriculumSampler,
    "cell_curriculum": CellCurriculumSampler,
}

# Nestful profile modes (lazy register — avoids import cycles at module load)
try:
    from .nestful_profile import register as _register_nestful
    _register_nestful()
except Exception:  # pragma: no cover
    pass


def update_curriculum(state: SamplerState, cfg: Dict[str, Any],
                      prerequisites: Optional[Dict[str, List[str]]] = None) -> None:
    c = {**DEFAULT_CONFIG["curriculum"], **(cfg.get("curriculum") or {})}
    prerequisites = prerequisites or {}
    for cell, entry in state.axis.get("generation_cell", {}).items():
        status = state.curriculum.get(cell, PROBING)
        eff_rate = entry.effective_group_count / max(entry.group_count, 1)

        if status == LOCKED:
            reqs = prerequisites.get(cell) or []
            mastered = sum(1 for r in reqs
                           if state.curriculum.get(r) == MASTERED)
            if not reqs or mastered / max(len(reqs), 1) >= c["prerequisite_master_share"]:
                status = PROBING
        elif status == PROBING:
            if entry.group_count >= c["probe_group_count"]:
                status = (ACTIVE if eff_rate >= c["unlock_effective_group_rate"]
                          else PROBING)
        if status == ACTIVE:
            if (entry.ema_terminal_success >= c["master_ema_success"]
                    and entry.ema_reward_variance <= c["master_max_variance"]):
                status = MASTERED
            elif entry.consecutive_all_fail_no_progress >= c["too_hard_all_fail_streak"]:
                status = TOO_HARD
        elif status == TOO_HARD and eff_rate >= c["activate_effective_group_rate"]:
            status = ACTIVE
        state.curriculum[cell] = status


# ── batch refill ──────────────────────────────────────────────────────────
def refill_batch(sampler: PromptSampler,
                 score_group: Callable[[PromptRef, int], GroupObservation],
                 *, global_step: int,
                 target_effective: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 max_rounds: Optional[int] = None) -> Dict[str, Any]:
    """Sample -> score -> keep informative groups -> refill.

    ``score_group`` is injected: in training it wraps the rollout worker, in the
    offline simulator it replays a recorded log. The sampler itself never
    generates anything.
    """
    cfg = sampler.cfg
    target = int(target_effective or cfg["target_effective_prompt_count"])
    size = int(batch_size or cfg["candidate_prompt_batch_size"])
    rounds_cap = int(max_rounds or cfg["maximum_refill_rounds"])

    kept: List[GroupObservation] = []
    rejected: List[GroupObservation] = []
    rounds = 0
    n_candidates = 0
    while len(kept) < target and rounds < rounds_cap:
        rounds += 1
        picks = sampler.sample_candidates(size)
        n_candidates += len(picks)
        for prompt in picks:
            obs = score_group(prompt, global_step)
            if obs is None:
                continue
            effective = sampler.observe_group(obs)
            (kept if effective else rejected).append(obs)
            if len(kept) >= target:
                break
    classes = {k: 0 for k in GROUP_CLASSES}
    for obs in kept + rejected:
        classes[obs.group_class] = classes.get(obs.group_class, 0) + 1
    n_groups = max(len(kept) + len(rejected), 1)
    return {
        "global_step": global_step,
        "refill_rounds": rounds,
        "candidate_prompt_count": n_candidates,
        "accepted_effective_groups": len(kept),
        "rejected_groups": len(rejected),
        "rejected_all_correct": classes.get(ALL_CORRECT, 0),
        "rejected_all_fail_no_progress": classes.get(ALL_FAIL_NO_PROGRESS, 0),
        "retained_all_fail_with_progress": sum(
            1 for o in kept if o.group_class == ALL_FAIL_WITH_PROCESS_VARIANCE),
        "dead_group_rate_before_filtering": round(
            (classes.get(ALL_CORRECT, 0) + classes.get(ALL_FAIL_NO_PROGRESS, 0)
             + classes.get(EQUAL_PARTIAL, 0)) / n_groups, 4),
        "effective_group_rate_after_filtering": round(len(kept) / n_groups, 4),
        "rollout_utilization": round(
            sum(o.group_size for o in kept) /
            max(sum(o.group_size for o in kept + rejected), 1), 4),
        "group_classes": classes,
        "target_reached": len(kept) >= target,
        "kept": kept,
        "rejected": rejected,
    }


def sampling_entropy(prompts: Sequence[PromptRef]) -> float:
    weights = [max(p.selection_weight, 0.0) for p in prompts]
    total = sum(weights)
    if total <= 0 or len(weights) <= 1:
        return 0.0
    h = -sum((w / total) * math.log(w / total) for w in weights if w > 0)
    return round(h / math.log(len(weights)), 6)


def _weighted_sample_without_replacement(pool: Sequence[PromptRef],
                                         weights: Sequence[float], n: int,
                                         rng: random.Random) -> List[PromptRef]:
    """Efraimidis-Spirakis: deterministic given the rng state."""
    if n >= len(pool):
        return list(pool)
    keyed = []
    for item, w in zip(pool, weights):
        u = rng.random() or 1e-12
        keyed.append((u ** (1.0 / max(w, 1e-12)), item))
    keyed.sort(key=lambda kv: kv[0], reverse=True)
    return [item for _k, item in keyed[:n]]


def _restore_rng_state(state: Any) -> Tuple[Any, ...]:
    """JSON round-trips tuples to lists; random.setstate needs tuples."""
    version, internal, gauss = state
    return (version, tuple(internal), gauss)


def prompt_refs_from_dataset(rows: Iterable[Dict[str, Any]]) -> List[PromptRef]:
    out: List[PromptRef] = []
    for row in rows:
        prov = row.get("provenance") or {}
        sig = row.get("difficulty_signature") or prov.get("difficulty_signature") or {}
        out.append(PromptRef(
            prompt_id=str(row.get("sample_id") or row.get("task_id")),
            generation_cell=str(row.get("generation_cell")
                                or prov.get("generation_cell_id") or ""),
            semantic_program_family=str(row.get("program_family_id")
                                        or prov.get("semantic_program_family") or ""),
            call_bucket=str(row.get("call_bucket")
                            or _bucket(int(row.get("num_calls") or
                                           len(row.get("gold_calls") or [])))),
            pattern_family=str(row.get("pattern_family")
                               or row.get("motif_type") or ""),
            query_mode=str(row.get("requested_query_mode")
                           or prov.get("query_mode") or ""),
            capability_families=list(row.get("capability_families") or []),
            difficulty_band=str(row.get("difficulty_band")
                                or prov.get("difficulty_band") or ""),
            difficulty_signature=sig))
    return out


def _bucket(n: int) -> str:
    if n <= 2:
        return "2"
    if n >= 6:
        return "6+"
    return str(n)
