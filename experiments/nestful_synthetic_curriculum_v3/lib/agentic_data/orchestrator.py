"""Data-scientist orchestrator: the Autodata / Agentic Self-Instruct loop.

Per batch (``CANDIDATES_PER_REQUEST`` candidates), in cost order (cheapest
gate first):
  registry-first skeleton OR challenger batch → schema gates → deterministic
  execution verifier → dedup + contamination → WEAK solver (metadata /
  optional prefilter; never hard-vetoes weak passes under rollout_primary) →
  optional STRONG solver → hold in best-of-N POOL → 8-rollout GRPO-signal
  probe on every pool survivor (primary acceptance gate when local Qwen is
  available) → rank by (signal, gap, novelty) → diversity caps → LLM judge
  → accept.
"""
from __future__ import annotations

import datetime
import json
import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .challenger import (challenger_messages, gen_mode, normalize_candidate,
                         parse_candidates, question_polish_messages,
                         repair_candidate)
from .quality import (DedupIndex, DiversityTracker, acceptance_policy,
                      solver_gap_verdict, solver_weak_max)
from .registry_first import (attach_polished_questions,
                             generate_registry_skeletons)
from .recipe import Recipe
from .rollout_signal import ROLLOUT_N, probe_rollout_signal
from .schema import (MOTIFS, STAGES, candidate_schema_errors, final_row,
                     looks_like_cot)
from .semantics import semantic_errors
from .multiturn_solver import (solver_gap_mode, solve_strong_multiturn,
                               solve_weak_multiturn)
from .rollout_signal import target_is_local
from .solvers import (best_of, parse_solver_output, score_prediction,
                      solver_messages, solver_params)
from .trace_validation import hard_trace_errors
from .verifier import deterministic_verify, judge_messages, judge_verdict
from .env_defaults import (
    BEST_OF_N_ENABLED,
    BEST_OF_N_MAX_ACCEPTS_PER_BATCH,
    BEST_OF_N_WEIGHT_GAP,
    BEST_OF_N_WEIGHT_NOVELTY,
    BEST_OF_N_WEIGHT_SIGNAL,
    CANDIDATES_PER_REQUEST as _CANDIDATES_PER_REQUEST_DEFAULT,
    MIN_ACCEPT_RATE,
    MIN_ACCEPT_RATE_RESUME,
    RESUME_MIN_ITERATIONS,
    WARMUP_BATCHES,
    WARMUP_BATCHES_RESUME,
    env_bool,
    env_float,
    env_int,
)
from .exec_bridge import TOOLS, tool_schema
from ..synthetic_gen_v5 import DiversityConfig, _UsageBalancer, _question_from_phrases

CANDIDATES_PER_REQUEST = env_int("CANDIDATES_PER_REQUEST",
                                 _CANDIDATES_PER_REQUEST_DEFAULT)
MAX_CONTAMINATION_STRIKES = 10


def best_of_n_enabled() -> bool:
    return env_bool("BEST_OF_N_ENABLED", BEST_OF_N_ENABLED)


def best_of_n_max_accepts() -> int:
    return max(1, env_int("BEST_OF_N_MAX_ACCEPTS_PER_BATCH",
                          BEST_OF_N_MAX_ACCEPTS_PER_BATCH))


def _composite_quality_score(*, gap: float, novelty: float,
                             signal_score: float) -> float:
    """Composite ranking score for Best-of-N candidate selection (spec:
    "vyber nejlepsiho kandidata podle gap / GRPO signalu / diverzity", not
    just the first candidate in the batch that clears every gate).

    * ``gap`` — strong-solver-score minus weak-solver-score, already in
      [0, 1] (a bigger separation means the task cleanly discriminates a
      capable solver from a weak one — the core Autodata acceptance signal);
    * ``novelty`` — mean inverse tool-usage frequency across this candidate's
      gold_calls, in (0, 1] (keeps the corpus from being dominated by a
      handful of tools even within otherwise-equal candidates);
    * ``signal_score`` — normalized GRPO reward-variance from the rollout
      probe, in [0, 1] (a stronger within-group reward spread is more useful
      training signal for GRPO).
    """
    w_gap = env_float("BEST_OF_N_WEIGHT_GAP", BEST_OF_N_WEIGHT_GAP)
    w_nov = env_float("BEST_OF_N_WEIGHT_NOVELTY", BEST_OF_N_WEIGHT_NOVELTY)
    w_sig = env_float("BEST_OF_N_WEIGHT_SIGNAL", BEST_OF_N_WEIGHT_SIGNAL)
    return (w_gap * max(0.0, min(1.0, gap))
            + w_nov * max(0.0, min(1.0, novelty))
            + w_sig * max(0.0, min(1.0, signal_score)))


def _signal_score(rollout_signal: Dict[str, Any]) -> float:
    """Normalize a rollout-probe summary to a [0, 1] ranking contribution."""
    if rollout_signal.get("skipped"):
        return 0.5   # neutral — no local weak backend to probe with
    variance = float(rollout_signal.get("reward_variance") or 0.0)
    # 0.25 ~= max variance for a 50/50 split between reward 0.0 and 1.0.
    return max(0.0, min(1.0, variance / 0.25))


def _tool_novelty(tool_names: List[str], usage: Counter) -> float:
    """Mean inverse-frequency of this candidate's tools against everything
    ALREADY accepted this stage — 1.0 for tools never used yet, decaying as a
    tool is reused, mirroring ``synthetic_gen_v5._UsageBalancer``'s sampling
    weight so best-of-N selection reinforces (not fights) generation-time
    diversity."""
    if not tool_names:
        return 1.0
    return sum(1.0 / (1.0 + usage.get(n, 0)) for n in tool_names) / len(tool_names)


def _accept_rate_stop_params(*, is_resume: bool) -> tuple[int, float, int]:
    """(warmup_batches, min_accept_rate, min_iterations_before_stop)."""
    default_warmup = WARMUP_BATCHES_RESUME if is_resume else WARMUP_BATCHES
    default_rate = MIN_ACCEPT_RATE_RESUME if is_resume else MIN_ACCEPT_RATE
    default_min_iters = RESUME_MIN_ITERATIONS if is_resume else default_warmup
    warmup = env_int("WARMUP_BATCHES", default_warmup)
    rate = env_float("MIN_ACCEPT_RATE", default_rate)
    min_iters = env_int("RESUME_MIN_ITERATIONS", default_min_iters)
    return warmup, rate, min_iters


class StageBudgetStop(RuntimeError):
    """Raised when a stop condition fires (budget, acceptance rate, strikes)."""


class StageWriter:
    """Crash-safe incremental writer: every NEW accepted row is appended to the
    stage file IMMEDIATELY (flush + fsync). Fresh runs truncate; resume runs
    append only new rows (existing rows are loaded separately)."""

    def __init__(self, path: str, *, append: bool = False,
                 existing_rows: int = 0) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mode = "a" if append else "w"
        self._fh = open(path, mode, encoding="utf-8")
        self.n_written = existing_rows       # total rows in file after close
        self.n_new = 0                       # rows written THIS run only

    def append(self, row: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except OSError:
            pass
        self.n_written += 1
        self.n_new += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _progress_enabled() -> bool:
    return os.environ.get("AGENTIC_PROGRESS_LOG", "1") != "0"


def _progress_log(msg: str) -> None:
    if _progress_enabled():
        print(msg, flush=True)


def _client_progress_snapshot(client) -> Dict[str, Any]:
    stats = client.stats.as_dict()
    local_weak = sum(
        r.get("local_requests", 0)
        for r in stats.get("by_role", {}).values())
    return {**stats, "local_weak_requests": local_weak}


def _write_progress_file(out_root: str, payload: Dict[str, Any]) -> None:
    if not _progress_enabled():
        return
    path = os.path.join(out_root, "RUN_PROGRESS.json")
    tmp = path + ".tmp"
    payload = {**payload, "updated_at": _now()}
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def _short_stage(stage: str) -> str:
    return stage.split("_", 1)[0]           # stage2_2call_... -> stage2


def motifs_for_stage(stage: str) -> Tuple[str, ...]:
    """Motifs that are structurally possible for this stage (fan_in needs
    >= 3 total calls: two independent producers + one consumer)."""
    from .schema import MOTIFS, MOTIFS_MIN_CALLS, STAGES as _STAGES
    _, hi = _STAGES[stage]
    return tuple(m for m in MOTIFS if MOTIFS_MIN_CALLS.get(m, 2) <= hi)


# NESTFUL-scale offered-tool-count ranges by gold call count (spec: Stage 2
# pilot rows over-offered 16-26 tools; NESTFUL's own 2-3 call tasks offer
# noticeably fewer). (lo, hi) is the BASE range; distractor_heavy adds a few.
_OFFERED_RANGE_BY_N_CALLS: Dict[int, Tuple[int, int]] = {
    2: (6, 11), 3: (8, 13), 4: (9, 15), 5: (10, 16), 6: (11, 17),
}


def _offered_schemas(rng: random.Random, used: List[str], motif: str,
                     n_gold_calls: int = 2) -> List[Dict[str, Any]]:
    """NESTFUL-like offered-tool menu (used + distractors), scaled to the
    task's own call count rather than a flat 10-26 range."""
    all_names = sorted(TOOLS.keys())
    used_set = set(used)
    lo, hi = _OFFERED_RANGE_BY_N_CALLS.get(
        n_gold_calls, _OFFERED_RANGE_BY_N_CALLS[max(_OFFERED_RANGE_BY_N_CALLS)])
    if motif == "distractor_heavy":
        lo, hi = lo + 3, hi + 4
    n_offered = rng.randrange(lo, hi + 1)
    domains_used = {TOOLS[n]["domain"] for n in used if n in TOOLS}
    same = [n for n in all_names if n not in used_set
            and TOOLS[n]["domain"] in domains_used]
    other = [n for n in all_names if n not in used_set
             and TOOLS[n]["domain"] not in domains_used]
    rng.shuffle(same)
    rng.shuffle(other)
    need = max(0, n_offered - len(used))
    # closely-related (same-domain) distractors dominate the menu, like a
    # real API surface — at least 2 when available, up to 2/3 of the slots
    n_same = min(len(same), max(2, (2 * need) // 3)) if need else 0
    offered = list(used) + same[:n_same] + other[:need - n_same]
    rng.shuffle(offered)
    return [tool_schema(n) for n in offered]


class Orchestrator:
    def __init__(self, *, client, models: Dict[str, str], out_root: str,
                 seed: int = 42, contamination_checker=None,
                 max_iterations_per_stage: int = 400,
                 run_judge: bool = True,
                 filtered_suffix: str = "") -> None:
        self.client = client
        self.models = models
        self.out_root = out_root
        self.seed = seed
        self.contamination = contamination_checker
        self.max_iterations = max_iterations_per_stage
        self.run_judge = run_judge
        # e.g. ".partial_salvaged" for offline cache-replay salvage runs
        self.filtered_suffix = filtered_suffix
        self.rejected: List[Dict[str, Any]] = []
        self.rejection_counter: Counter = Counter()
        self.solver_gap_log: List[Dict[str, Any]] = []
        self.stage_summaries: Dict[str, Any] = {}
        # accepted rows live on the ORCHESTRATOR (not a local variable), so an
        # early stop / exception can never lose them (root cause of the
        # "228 accepted / 0 written" overnight-run bug).
        self.accepted_by_stage: Dict[str, List[Dict[str, Any]]] = {}
        self.stage_paths: Dict[str, str] = {}
        self.diversity_by_stage: Dict[str, Any] = {}
        # tool -> count of ACCEPTED gold_calls this stage, used only to rank
        # best-of-N candidates by novelty (never gates acceptance itself).
        self.tool_usage_by_stage: Dict[str, Counter] = {}
        self._registry_balancer: Optional[_UsageBalancer] = None

    def _record_local_inference(self, role: str = "weak_solver", n: int = 1) -> None:
        """Count local HF weak-solver episodes (bypass OpenRouter client.chat)."""
        r = self.client.stats.by_role.setdefault(role, {
            "requests": 0, "cache_hits": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "spend_usd": 0.0, "local_requests": 0})
        r["local_requests"] = r.get("local_requests", 0) + n

    def _registry_balancer_for_stage(self, stage: str) -> _UsageBalancer:
        if self._registry_balancer is None:
            self._registry_balancer = _UsageBalancer(DiversityConfig())
        return self._registry_balancer

    def stage_file_path(self, stage: str, *, suffix: Optional[str] = None) -> str:
        from .schema import STAGE_FILES
        sfx = self.filtered_suffix if suffix is None else suffix
        base = STAGE_FILES[stage]
        if sfx:
            base = base.replace(".jsonl", f"{sfx}.jsonl")
        return os.path.join(self.out_root, "filtered", base)

    def find_resume_source(self, stage: str) -> Optional[str]:
        """Canonical filtered file first; fall back to *.partial_salvaged.jsonl."""
        canonical = self.stage_file_path(stage, suffix="")
        salvaged = self.stage_file_path(stage, suffix=".partial_salvaged")
        if os.path.isfile(canonical) and count_jsonl_rows(canonical) > 0:
            return canonical
        if os.path.isfile(salvaged) and count_jsonl_rows(salvaged) > 0:
            return salvaged
        return None

    # ------------------------------------------------------------ helpers
    def _reject(self, stage: str, cand: Dict[str, Any], reason: str,
                detail: Optional[str], recipe: Recipe,
                extra: Optional[Dict[str, Any]] = None) -> None:
        self.rejection_counter[reason] += 1
        rec = {
            "stage": stage,
            "reason": reason,
            "detail": (detail or "")[:400],
            "question": (cand.get("question") or "")[:400],
            "gold_calls": cand.get("gold_calls"),
            "recipe_version": recipe.version,
            "created_at": _now(),
        }
        if extra:
            rec.update(extra)
        self.rejected.append(rec)

    def _api_chat(self, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """client.chat; transient API/budget failures return None (skip)."""
        try:
            return self.client.chat(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ in ("OpenRouterError", "BudgetExceeded"):
                print(f"[orchestrator] {type(exc).__name__}: {exc} "
                      f"— skipping candidate", flush=True)
                return None
            raise

    def _solve_single_shot(self, role: str, question: str,
                           tools: List[Dict[str, Any]],
                           gold_calls, gold_obs, gold_answer, *, strong: bool,
                           seed: int) -> Dict[str, Any]:
        params = solver_params(strong)   # WEAK_SOLVER_MODE / STRONG_SOLVER_MODE
        scores: List[Dict[str, Any]] = []
        for a in range(params["attempts"]):
            resp = self._api_chat(
                role=role, model=self.models[role],
                messages=solver_messages(question, tools, strong=strong,
                                         mode=params["mode"]),
                temperature=params["temperature"],
                max_tokens=params["max_tokens"],
                json_mode=True, seed=seed + a)
            if resp is None:
                return {"score": 0.0, "status": "api_error", "n_calls": 0,
                        "solver_mode": "single_shot"}
            if resp.get("dry_run"):
                return {"score": 0.0, "status": "dry_run", "n_calls": 0,
                        "solver_mode": "single_shot"}
            parsed = resp["parsed"]
            calls = parse_solver_output(parsed)
            final = parsed.get("final_answer") if isinstance(parsed, dict) else None
            scored = score_prediction(calls, final, gold_calls,
                                      gold_obs, gold_answer)
            scored["solver_mode"] = "single_shot"
            scores.append(scored)
        return best_of(scores)

    def _solve(self, role: str, question: str, tools: List[Dict[str, Any]],
               gold_calls, gold_obs, gold_answer, *, strong: bool,
               seed: int, stage: str) -> Dict[str, Any]:
        if solver_gap_mode() == "multiturn":
            if strong:
                return solve_strong_multiturn(
                    self._api_chat, self.models[role],
                    question, tools, gold_calls, gold_obs, gold_answer,
                    stage=stage, seed=seed)
            if target_is_local():
                result = solve_weak_multiturn(
                    question, tools, gold_calls, gold_obs, gold_answer,
                    stage=stage, seed=seed)
                self._record_local_inference("weak_solver", 1)
                return result
        return self._solve_single_shot(
            role, question, tools, gold_calls, gold_obs, gold_answer,
            strong=strong, seed=seed)

    # ------------------------------------------------------------ main loop
    def generate_stage(self, stage: str, target: int,
                       *, resume: bool = False) -> List[Dict[str, Any]]:
        """Generate one stage up to `target` TOTAL accepted rows.

        With resume=True, loads existing rows from filtered/*.jsonl (or
        *.partial_salvaged.jsonl), seeds dedup, and generates only the
        remaining gap (target - len(existing)). New rows append to the
        canonical stage file."""
        assert stage in STAGES, f"unknown stage {stage}"
        out_path = self.stage_file_path(stage, suffix="")
        self.stage_paths[stage] = out_path
        existing_rows: List[Dict[str, Any]] = []
        resume_source: Optional[str] = None

        if resume:
            resume_source = self.find_resume_source(stage)
            if resume_source:
                existing_rows = load_jsonl_rows(resume_source)
                bad = [r for r in existing_rows
                       if r.get("stage") not in (stage, None)]
                if bad:
                    raise ValueError(
                        f"resume {stage}: {len(bad)} rows have wrong stage "
                        f"field in {resume_source}")
                # migrate partial_salvaged → canonical before appending
                if resume_source != out_path:
                    _atomic_write_jsonl(out_path, existing_rows)
                    print(f"[orchestrator] resume: migrated {len(existing_rows)} "
                          f"rows from {os.path.basename(resume_source)} -> "
                          f"{os.path.basename(out_path)}")
                print(f"[orchestrator] resume {stage}: {len(existing_rows)} "
                      f"existing, target {target}, need "
                      f"{max(0, target - len(existing_rows))} more")

        accepted: List[Dict[str, Any]] = list(existing_rows)
        self.accepted_by_stage[stage] = accepted
        n_existing = len(existing_rows)

        if n_existing >= target:
            print(f"[orchestrator] {stage}: already complete "
                  f"({n_existing}/{target}), skipping generation")
            self.stage_summaries[stage] = {
                "target": target, "accepted": n_existing,
                "resumed_from": n_existing, "accepted_new": 0,
                "status": "complete", "iterations": 0,
                "mean_rounds_per_accept": None,
                "recipe": None, "stage_file": out_path,
                "rows_written": n_existing, "resume_source": resume_source,
            }
            return accepted

        writer = StageWriter(out_path, append=resume and n_existing > 0,
                             existing_rows=n_existing)
        rng = random.Random(f"agentic|{stage}|{self.seed}")
        recipe = Recipe()
        iteration = 0
        try:
            self._stage_loop(stage=stage, target=target, accepted=accepted,
                             writer=writer, rng=rng, recipe=recipe,
                             n_existing=n_existing, is_resume=resume)
            iteration = self._last_iteration
        except BaseException:
            iteration = self._last_iteration
            raise
        finally:
            writer.close()
            status = "complete" if len(accepted) >= target else "partial"
            if iteration:
                api = _client_progress_snapshot(self.client)
                _progress_log(
                    f"[progress] STAGE {status.upper()} {stage}: "
                    f"{len(accepted)}/{target} accepted "
                    f"(+{writer.n_new} new, {iteration} batches, "
                    f"mean_rounds/accept={self._mean_rounds_per_accept}) | "
                    f"api={api['n_requests']} req ${api['spend_usd']:.4f}")
                _write_progress_file(self.out_root, {
                    "status": status,
                    "stage": stage,
                    "accepted": len(accepted),
                    "target": target,
                    "accepted_new_this_run": writer.n_new,
                    "iterations": iteration,
                    "mean_rounds_per_accept": self._mean_rounds_per_accept,
                    "client": api,
                })
            self.stage_summaries[stage] = {
                "target": target,
                "accepted": len(accepted),
                "resumed_from": n_existing,
                "accepted_new": writer.n_new,
                "status": "complete" if len(accepted) >= target else "partial",
                "iterations": iteration,
                "mean_rounds_per_accept": self._mean_rounds_per_accept,
                "recipe": recipe.as_dict(),
                "stage_file": out_path,
                "rows_written": writer.n_written,
                "resume_source": resume_source,
                "diversity": (self.diversity_by_stage[stage].stats()
                              if stage in self.diversity_by_stage else None),
            }
        return accepted

    def _stage_loop(self, *, stage: str, target: int,
                    accepted: List[Dict[str, Any]], writer: StageWriter,
                    rng: random.Random, recipe: Recipe,
                    n_existing: int = 0, is_resume: bool = False) -> None:
        dedup = DedupIndex()
        diversity = DiversityTracker(resume_mode=is_resume)
        tool_usage: Counter = Counter()
        if n_existing:
            dedup.seed_from_rows(accepted[:n_existing])
            # Reference-only: legacy rows inform reports but do NOT tighten caps.
            diversity.seed_reference_from_rows(accepted[:n_existing])
            for row in accepted[:n_existing]:
                for c in row.get("gold_calls") or []:
                    if isinstance(c, dict) and c.get("name"):
                        tool_usage[c["name"]] += 1
        self.diversity_by_stage[stage] = diversity
        self.tool_usage_by_stage[stage] = tool_usage
        warmup_batches, min_accept_rate, min_iters_before_stop = \
            _accept_rate_stop_params(is_resume=is_resume)
        iteration = 0
        contamination_strikes = 0
        # snapshot so the acceptance-rate stop only counts THIS stage
        rejections_at_stage_start = sum(self.rejection_counter.values())
        rounds_per_accept: List[int] = []
        rounds_since_accept = 0
        self._last_iteration = 0
        self._mean_rounds_per_accept = None
        _progress_log(f"[progress] {stage}: generating {n_existing}/{target} "
                      f"-> target {target}"
                      f"{f' (resume, need {target - n_existing} more)' if is_resume and n_existing else ''}")

        while len(accepted) < target:
            if iteration >= self.max_iterations:
                raise StageBudgetStop(
                    f"{stage}: iteration budget {self.max_iterations} exhausted "
                    f"at {len(accepted)}/{target} accepted")
            iteration += 1
            self._last_iteration = iteration
            rounds_since_accept += 1
            challenger_resp: Optional[Dict[str, Any]] = None
            stage_motifs = motifs_for_stage(stage)
            motif = stage_motifs[(iteration - 1) % len(stage_motifs)]

            if gen_mode() == "registry_first":
                balancer = self._registry_balancer_for_stage(stage)
                skeletons = generate_registry_skeletons(
                    stage, motif, CANDIDATES_PER_REQUEST, rng, balancer)
                if not skeletons:
                    self._reject(stage, {"question": ""}, "invalid_json",
                                 "registry-first: no executable skeletons",
                                 recipe)
                    continue
                if getattr(self.client, "backend", None) == "mock":
                    mock_cands = []
                    for sk in skeletons:
                        q = _question_from_phrases(
                            rng, sk["_phrases"], sk["_n_calls"])
                        mock_cands.append({
                            "question": q,
                            "tool_names": sk["tool_names"],
                            "gold_calls": sk["gold_calls"],
                            "motif_type": sk["motif_type"],
                            "answer_type": sk["answer_type"],
                            "rationale": "registry-first deterministic trace",
                            "_registry_first": True,
                        })
                    candidates = mock_cands
                    handler = getattr(self.client, "mock_handler", None)
                    if handler is not None:
                        for cand in candidates:
                            key = " ".join(cand["question"].lower().split())
                            handler.gold_by_question[key] = cand["gold_calls"]
                else:
                    resp = self._api_chat(
                        role="challenger", model=self.models["challenger"],
                        messages=question_polish_messages(
                            skeletons=skeletons,
                            feedback_block=recipe.feedback_block()),
                        temperature=0.85, max_tokens=1800, json_mode=True,
                        seed=self.seed * 1000 + iteration)
                    challenger_resp = resp
                    if resp is None:
                        continue
                    if resp.get("dry_run"):
                        print(f"[orchestrator] DRY RUN — stage {stage} stops here.")
                        return
                    candidates = attach_polished_questions(skeletons, resp["parsed"])
            else:
                resp = self._api_chat(
                    role="challenger", model=self.models["challenger"],
                    messages=challenger_messages(
                        stage=stage, motif=motif,
                        n_candidates=CANDIDATES_PER_REQUEST,
                        feedback_block=recipe.feedback_block(), rng=rng),
                    temperature=0.9, max_tokens=2400, json_mode=True,
                    seed=self.seed * 1000 + iteration)
                challenger_resp = resp
                if resp is None:
                    continue
                if resp.get("dry_run"):
                    print(f"[orchestrator] DRY RUN — stage {stage} stops here.")
                    return
                candidates = parse_candidates(resp["parsed"])
            if not candidates:
                snippet = ""
                if gen_mode() != "registry_first" and challenger_resp is not None:
                    snippet = challenger_resp.get("text", "")[:200]
                self._reject(stage, {"question": snippet},
                             "invalid_json", "challenger output unparseable",
                             recipe)
                continue

            batch_counter: Counter = Counter()
            batch_weak: List[float] = []
            batch_strong: List[float] = []
            # Best-of-N pool: candidates that cleared every FREE/cheap gate
            # plus the solver-gap policy, held back from acceptance until the
            # whole batch has been scored so the orchestrator can pick the
            # best one(s) instead of the first one encountered.
            pool: List[Dict[str, Any]] = []
            _progress_log(
                f"[progress] BATCH {iteration} start | candidates={len(candidates)} "
                f"| accepted {len(accepted)}/{target}")

            for cand in candidates:
                if len(accepted) >= target:
                    break
                cand = repair_candidate(normalize_candidate(cand))
                # 1. schema gates (free)
                errs = candidate_schema_errors(cand, stage)
                if errs:
                    self._reject(stage, cand, "invalid_schema", "; ".join(errs),
                                 recipe)
                    batch_counter["invalid_schema"] += 1
                    continue
                if looks_like_cot(cand.get("rationale")):
                    self._reject(stage, cand, "cot_leakage",
                                 "rationale too long (possible CoT)", recipe)
                    batch_counter["cot_leakage"] += 1
                    continue
                # 1b. hard trace structure (free, pre-execution): unique +
                # sequential labels, references only to prior calls' EXACT
                # output field, call-count consistency. See trace_validation
                # module docstring for the two real pilot rows that motivated
                # this ($var1 reused as the label of both calls).
                trace_errs = hard_trace_errors(cand, TOOLS, STAGES[stage])
                if trace_errs:
                    self._reject(stage, cand, "invalid_trace_labels",
                                 "; ".join(trace_errs), recipe)
                    batch_counter["invalid_trace_labels"] += 1
                    continue
                # 1c. semantic compatibility (free, pre-execution): reject
                # cross-family bindings like temperature -> money unless one
                # side is a generic/unit-agnostic slot (part, whole, value...).
                sem_errs = semantic_errors(cand.get("gold_calls") or [], TOOLS)
                if sem_errs:
                    self._reject(stage, cand, "semantic_incompatible_reference",
                                 "; ".join(sem_errs), recipe)
                    batch_counter["semantic_incompatible_reference"] += 1
                    continue
                # 2. deterministic executor — source of truth for gold
                v = deterministic_verify(cand)
                if not v["ok"]:
                    self._reject(stage, cand, v["reason"], v["detail"], recipe)
                    batch_counter[v["reason"]] += 1
                    continue
                observations, gold_answer = v["observations"], v["gold_answer"]
                question = cand["question"].strip()
                gold_calls = cand["gold_calls"]
                # 3. dedup + contamination. `sid` here is PROVISIONAL (only
                # used for the contamination sample_id-overlap check) — the
                # final sample_id is assigned at ACCEPT time below, once
                # best-of-N selection knows the candidate's real rank among
                # accepted rows.
                dup = dedup.check_and_add(question, gold_calls)
                if dup:
                    self._reject(stage, cand, dup, None, recipe)
                    batch_counter[dup] += 1
                    continue
                provisional_sid = (f"agentic_v5_{_short_stage(stage)}_pending_"
                                   f"{iteration:06d}_{len(pool)}")
                if self.contamination is not None:
                    ok, why = self.contamination.check(
                        question, gold_calls, provisional_sid)
                    if not ok:
                        contamination_strikes += 1
                        dedup.remove(question, gold_calls)
                        self._reject(stage, cand, "overlap_with_nestful", why,
                                     recipe)
                        batch_counter["overlap_with_nestful"] += 1
                        if contamination_strikes >= MAX_CONTAMINATION_STRIKES:
                            raise StageBudgetStop(
                                f"{stage}: contamination check failed "
                                f"{contamination_strikes}x — stopping for review")
                        continue
                # 4. offered tools (used + distractors), scaled to call count
                tools = _offered_schemas(rng, [c["name"] for c in gold_calls],
                                         cand["motif_type"], len(gold_calls))
                # 5. weak solver (skip strong when weak passes — saves compute)
                weak = self._solve("weak_solver", question, tools, gold_calls,
                                   observations, gold_answer, strong=False,
                                   seed=self.seed + iteration, stage=stage)
                batch_weak.append(weak["score"])
                strong = None
                weak_max = solver_weak_max()
                if weak["score"] <= weak_max:
                    strong = self._solve("strong_solver", question, tools,
                                         gold_calls, observations, gold_answer,
                                         strong=True,
                                         seed=self.seed + iteration, stage=stage)
                    batch_strong.append(strong["score"])
                gap_ok, gap_reason = solver_gap_verdict(weak, strong)
                gap_rec = {
                    "stage": stage,
                    "acceptance_policy": acceptance_policy(),
                    "solver_gap_mode": solver_gap_mode(),
                    "weak_status": weak["status"],
                    "weak_score": weak["score"],
                    "strong_status": strong["status"] if strong else "skipped",
                    "strong_score": strong["score"] if strong else None,
                    "gap": round(strong["score"] - weak["score"], 3)
                    if strong else None,
                    "motif": cand["motif_type"],
                    "n_gold_calls": len(gold_calls),
                    "weak_n_calls": weak.get("n_calls"),
                    "strong_n_calls": strong.get("n_calls") if strong else None,
                    "gap_passed": gap_ok,
                    "would_reject_solver_gap": gap_reason,
                    "accepted": False,
                }
                self.solver_gap_log.append(gap_rec)
                if acceptance_policy() == "solver_gap" and not gap_ok:
                    dedup.remove(question, gold_calls)
                    self._reject(stage, cand, gap_reason, None, recipe,
                                 extra={"solver_gap": gap_rec})
                    batch_counter[gap_reason] += 1
                    continue
                elif acceptance_policy() == "rollout_primary" and not gap_ok:
                    gap_rec["gap_passed"] = True
                # Held for rollout probe + best-of-N ranking — NOT accepted yet.
                pool.append({
                    "cand": cand, "question": question, "tools": tools,
                    "gold_calls": gold_calls, "observations": observations,
                    "gold_answer": gold_answer, "weak": weak, "strong": strong,
                    "gap_rec": gap_rec,
                })

            n_pool = len(pool)
            cheap_rejects = sum(batch_counter.values())
            _progress_log(
                f"[progress] BATCH {iteration} gates | pool={n_pool}/{len(candidates)} "
                f"| cheap_rejects={cheap_rejects}")

            # ---- rollout GRPO-signal probe on every pool survivor (primary
            # gate under rollout_primary; ranking input for best-of-N).
            rollout_pool: List[Dict[str, Any]] = []
            if n_pool:
                _progress_log(
                    f"[progress] BATCH {iteration} rollout | probing {n_pool} "
                    f"candidates x{ROLLOUT_N} (may take several min; "
                    f"[override]/[executor] lines are per-rollout noise)")
            for item in pool:
                rollout_signal = probe_rollout_signal(
                    item["question"], item["tools"], item["gold_calls"],
                    item["observations"], item["gold_answer"],
                    stage=stage, seed=self.seed + iteration)
                if target_is_local() and not rollout_signal.get("skipped"):
                    self._record_local_inference("weak_solver", ROLLOUT_N)
                item["rollout_signal"] = rollout_signal
                item["signal_score"] = _signal_score(rollout_signal)
                cand, gap_rec = item["cand"], item["gap_rec"]
                if rollout_signal.get("skipped"):
                    rollout_pool.append(item)
                    continue
                if not rollout_signal.get("grpo_signal_positive"):
                    dedup.remove(item["question"], item["gold_calls"])
                    gap_rec["final_status"] = "low_grpo_signal_prediction"
                    self._reject(stage, cand, "low_grpo_signal_prediction",
                                 f"policy={rollout_signal.get('reward_policy')}; "
                                 f"unique_rewards={rollout_signal.get('unique_rewards')}; "
                                 f"variance={rollout_signal.get('reward_variance')}; "
                                 f"rewards={rollout_signal.get('rewards')}; "
                                 f"weak_status={item['weak']['status']}", recipe,
                                 extra={"solver_gap": gap_rec,
                                        "rollout_signal": rollout_signal})
                    batch_counter["low_grpo_signal_prediction"] += 1
                    continue
                rollout_pool.append(item)
            pool = rollout_pool
            _progress_log(
                f"[progress] BATCH {iteration} rollout done | grpo_ok="
                f"{len(pool)}/{n_pool}")

            # ---- best-of-N: rank survivors by composite (signal, gap, novelty)
            tool_usage = self.tool_usage_by_stage[stage]
            for item in pool:
                item["novelty"] = _tool_novelty(
                    [c["name"] for c in item["gold_calls"]], tool_usage)
                item["gap_value"] = (item["gap_rec"]["gap"]
                                     if item["gap_rec"]["gap"] is not None else 0.0)
            ranked = sorted(
                pool,
                key=lambda it: _composite_quality_score(
                    gap=it["gap_value"], novelty=it["novelty"],
                    signal_score=it.get("signal_score", 0.5)),
                reverse=True) if best_of_n_enabled() else pool
            max_accepts = best_of_n_max_accepts() if best_of_n_enabled() else len(ranked)

            n_accepted_this_batch = 0
            for rank, item in enumerate(ranked):
                if len(accepted) >= target:
                    break
                cand, question = item["cand"], item["question"]
                gold_calls, tools = item["gold_calls"], item["tools"]
                observations, gold_answer = item["observations"], item["gold_answer"]
                weak, strong, gap_rec = item["weak"], item["strong"], item["gap_rec"]
                if n_accepted_this_batch >= max_accepts:
                    dedup.remove(question, gold_calls)
                    gap_rec["final_status"] = "best_of_n_not_selected"
                    self._reject(stage, cand, "best_of_n_not_selected",
                                 f"rank {rank + 1}/{len(ranked)} in batch "
                                 f"(novelty={item['novelty']:.3f}, "
                                 f"gap={item['gap_value']:.3f})", recipe,
                                 extra={"solver_gap": gap_rec})
                    batch_counter["best_of_n_not_selected"] += 1
                    continue
                # 5b. diversity caps (free) — the accepted set must not be
                # dominated by one weak-score bucket / one failure type
                div_reason = diversity.verdict(weak["score"], weak["status"])
                if div_reason:
                    dedup.remove(question, gold_calls)
                    gap_rec["final_status"] = div_reason
                    self._reject(stage, cand, div_reason,
                                 f"weak_score={weak['score']} "
                                 f"failure_type={weak['status']}", recipe,
                                 extra={"solver_gap": gap_rec})
                    batch_counter[div_reason] += 1
                    continue
                # Rollout probe already ran on the full pool above.
                rollout_signal = item.get("rollout_signal") or {"skipped": True}
                # 6. LLM style judge (secondary; cannot override execution)
                if self.run_judge:
                    jresp = self._api_chat(
                        role="judge", model=self.models["judge"],
                        messages=judge_messages(question, len(gold_calls)),
                        temperature=0.0, max_tokens=400, json_mode=True,
                        seed=self.seed)
                    if jresp is None:
                        dedup.remove(question, gold_calls)
                        gap_rec["final_status"] = "api_error"
                        self._reject(stage, cand, "api_error",
                                     "judge API call failed", recipe,
                                     extra={"solver_gap": gap_rec})
                        batch_counter["api_error"] += 1
                        continue
                    jv = judge_verdict(jresp["parsed"])
                    if not jv["ok"]:
                        dedup.remove(question, gold_calls)
                        gap_rec["final_status"] = f"judge_rejected:{jv['reason']}"
                        self._reject(stage, cand, jv["reason"], jv["detail"],
                                     recipe)
                        batch_counter[jv["reason"]] += 1
                        continue
                # 7. accept — final sample_id assigned NOW (deterministic on
                # accepted-count, unaffected by how many candidates in the
                # batch lost the best-of-N ranking).
                sid = (f"agentic_v5_{_short_stage(stage)}_"
                       f"{len(accepted) + 1:06d}")
                row = final_row(
                    sample_id=sid, question=question, tools=tools,
                    gold_calls=gold_calls, observations=observations,
                    gold_answer=gold_answer, stage=stage,
                    motif_type=cand["motif_type"],
                    answer_type=("boolean" if isinstance(gold_answer, bool)
                                 else "scalar" if isinstance(gold_answer, (int, float))
                                 else "string" if isinstance(gold_answer, str)
                                 else "list"),
                    generation_seed=self.seed, models=self.models,
                    solver_gap={
                        "weak_status": gap_rec.get("weak_status", weak["status"]),
                        "strong_status": gap_rec.get("strong_status",
                                                     (strong or {}).get("status", "skipped")),
                        "weak_score": gap_rec.get("weak_score", weak["score"]),
                        "strong_score": gap_rec.get("strong_score"),
                        "gap": gap_rec.get("gap"),
                    },
                    rollout_signal=rollout_signal,
                    provenance={
                        "recipe_version": recipe.version,
                        "iteration": iteration,
                        "prompt_hash": None,
                        "raw_response_path": _relpath_or_none(
                            (challenger_resp or {}).get("raw_path"), self.out_root),
                        "created_at": _now(),
                        "tool_schema_source_policy": "aggregate_style_only",
                        "best_of_n": {
                            "enabled": best_of_n_enabled(),
                            "batch_size": len(candidates),
                            "pool_size": len(pool),
                            "rank_in_batch": rank + 1,
                            "novelty": round(item["novelty"], 4),
                            "gap": round(item["gap_value"], 4),
                            "signal_score": round(item.get("signal_score", 0.5), 4),
                        },
                    })
                gap_rec["accepted"] = True
                gap_rec["final_status"] = "accepted"
                accepted.append(row)
                writer.append(row)            # crash-safe: persisted NOW
                diversity.add(weak["score"], weak["status"])
                for c in gold_calls:
                    if isinstance(c, dict) and c.get("name"):
                        tool_usage[c["name"]] += 1
                n_accepted_this_batch += 1
                rounds_per_accept.append(rounds_since_accept)
                rounds_since_accept = 0
                if rounds_per_accept:
                    self._mean_rounds_per_accept = round(
                        sum(rounds_per_accept) / len(rounds_per_accept), 2)
                _progress_log(
                    f"[progress] ACCEPT {len(accepted)}/{target} | "
                    f"weak={weak['score']} {weak['status']} | "
                    f"strong={(strong or {}).get('score', 'skipped')} "
                    f"{(strong or {}).get('status', 'skipped')} | "
                    f"motif={cand['motif_type']} | best_of_n_rank="
                    f"{rank + 1}/{len(ranked)} | iter={iteration} | id={sid}")

            # ---- batch-level analysis → recipe revision (data scientist) ----
            batch_stats = {
                "iteration": iteration,
                "accepted_total": len(accepted),
                "weak_mean": round(sum(batch_weak) / len(batch_weak), 3)
                if batch_weak else None,
                "strong_mean": round(sum(batch_strong) / len(batch_strong), 3)
                if batch_strong else None,
            }
            recipe.update_from_batch(batch_counter, batch_stats)

            n_rej_stage = (sum(self.rejection_counter.values())
                           - rejections_at_stage_start)
            n_new_acc = len(accepted) - n_existing
            accept_rate = n_new_acc / max(1, n_new_acc + n_rej_stage)
            top_rejects = batch_counter.most_common(3)
            reject_summary = ", ".join(f"{r}:{c}" for r, c in top_rejects) or "none"
            api = _client_progress_snapshot(self.client)
            _progress_log(
                f"[progress] BATCH {iteration} | accepted {len(accepted)}/{target} "
                f"| new={n_new_acc} rejected={n_rej_stage} "
                f"| rate={accept_rate:.3f} | batch_rejects: {reject_summary} "
                f"| api={api['n_requests']} req ${api['spend_usd']:.4f} "
                f"| local_weak={api['local_weak_requests']}")
            _write_progress_file(self.out_root, {
                "status": "running",
                "stage": stage,
                "accepted": len(accepted),
                "target": target,
                "accepted_new_this_run": n_new_acc,
                "rejected_this_run": n_rej_stage,
                "accept_rate": round(accept_rate, 4),
                "iteration": iteration,
                "max_iterations": self.max_iterations,
                "batch_rejects": dict(batch_counter),
                "top_reject_reasons": dict(self.rejection_counter),
                "mean_rounds_per_accept": self._mean_rounds_per_accept,
                "recipe_version": recipe.version,
                "client": api,
            })

            # ---- stop condition: acceptance rate too low after warmup ------
            if iteration >= warmup_batches:
                if is_resume and iteration < min_iters_before_stop:
                    continue
                n_rej = sum(self.rejection_counter.values()) - rejections_at_stage_start
                n_new_acc = len(accepted) - n_existing
                rate = n_new_acc / max(1, n_new_acc + n_rej)
                if rate < min_accept_rate:
                    raise StageBudgetStop(
                        f"{stage}: acceptance rate {rate:.3f} < {min_accept_rate} "
                        f"after {iteration} batches ({n_new_acc} new accepted / "
                        f"{n_rej} rejected this run"
                        f"{'; resume patience exhausted' if is_resume else ''}"
                        f" — revise the recipe manually")


def _relpath_or_none(path: Optional[str], root: str) -> Optional[str]:
    if not path:
        return None
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return path


def load_jsonl_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _atomic_write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write rows to `path` via .tmp + flush + fsync + atomic os.replace."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def count_jsonl_rows(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def write_outputs(out_root: str, accepted_by_stage: Dict[str, List[Dict[str, Any]]],
                  orch: Orchestrator) -> Dict[str, str]:
    """Finalize filtered/*.jsonl (atomic rewrite of the incrementally-written
    files) + rejected/*. Returns stage → path map. Fails loudly if the row
    count on disk disagrees with the accepted rows in memory."""
    filtered_dir = os.path.join(out_root, "filtered")
    rejected_dir = os.path.join(out_root, "rejected")
    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    for stage, rows in accepted_by_stage.items():
        path = orch.stage_paths.get(stage) or orch.stage_file_path(stage)
        _atomic_write_jsonl(path, rows)
        n_disk = count_jsonl_rows(path)
        if n_disk != len(rows):
            raise RuntimeError(
                f"COUNT MISMATCH after write: {stage} memory={len(rows)} "
                f"disk={n_disk} ({path})")
        paths[stage] = path
    _atomic_write_jsonl(os.path.join(rejected_dir, "rejected_examples.jsonl"),
                        orch.rejected)
    tmp = os.path.join(rejected_dir, "rejection_reasons.csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write("reason,count\n")
        for reason, count in orch.rejection_counter.most_common():
            fh.write(f"{reason},{count}\n")
    os.replace(tmp, os.path.join(rejected_dir, "rejection_reasons.csv"))
    return paths
