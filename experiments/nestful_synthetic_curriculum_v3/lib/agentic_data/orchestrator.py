"""Data-scientist orchestrator: the Autodata / Agentic Self-Instruct loop.

Per stage, in cost order (cheapest gate first):
  challenger batch → schema gates → deterministic execution verifier →
  dedup + contamination → WEAK solver (skip strong if weak passes, saving
  compute like the paper) → STRONG solver (best-of-N) → solver-gap policy →
  LLM style judge → accept.
After every batch the orchestrator aggregates rejection reasons and revises
the challenger recipe (batch-level analysis + prompt update).
"""
from __future__ import annotations

import datetime
import json
import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional

from .challenger import challenger_messages, normalize_candidate, parse_candidates
from .quality import DedupIndex, solver_gap_verdict
from .recipe import Recipe
from .schema import (MOTIFS, STAGES, candidate_schema_errors, final_row,
                     looks_like_cot)
from .solvers import (STRONG_ATTEMPTS, STRONG_MAX_TOKENS, WEAK_MAX_TOKENS,
                      best_of, parse_solver_output, score_prediction,
                      solver_messages)
from .verifier import deterministic_verify, judge_messages, judge_verdict
from ..nestful_like_generator import TOOLS, tool_schema

CANDIDATES_PER_REQUEST = 5
MIN_ACCEPT_RATE = 0.02          # abort stage when below this after warmup
WARMUP_BATCHES = 5
MAX_CONTAMINATION_STRIKES = 10


class StageBudgetStop(RuntimeError):
    """Raised when a stop condition fires (budget, acceptance rate, strikes)."""


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _short_stage(stage: str) -> str:
    return stage.split("_", 1)[0]           # stage2_2call_... -> stage2


def _offered_schemas(rng: random.Random, used: List[str], motif: str
                     ) -> List[Dict[str, Any]]:
    """NESTFUL-like offered-tool menu (used + distractors), like det. v4."""
    all_names = sorted(TOOLS.keys())
    used_set = set(used)
    n_offered = rng.randrange(16, 26) if motif == "distractor_heavy" \
        else rng.randrange(10, 20)
    domains_used = {TOOLS[n]["domain"] for n in used if n in TOOLS}
    same = [n for n in all_names if n not in used_set
            and TOOLS[n]["domain"] in domains_used]
    other = [n for n in all_names if n not in used_set
             and TOOLS[n]["domain"] not in domains_used]
    rng.shuffle(same)
    rng.shuffle(other)
    need = max(0, n_offered - len(used))
    n_same = min(len(same), max(1, need // 2))
    offered = list(used) + same[:n_same] + other[:need - n_same]
    rng.shuffle(offered)
    return [tool_schema(n) for n in offered]


class Orchestrator:
    def __init__(self, *, client, models: Dict[str, str], out_root: str,
                 seed: int = 42, contamination_checker=None,
                 max_iterations_per_stage: int = 400,
                 run_judge: bool = True) -> None:
        self.client = client
        self.models = models
        self.out_root = out_root
        self.seed = seed
        self.contamination = contamination_checker
        self.max_iterations = max_iterations_per_stage
        self.run_judge = run_judge
        self.rejected: List[Dict[str, Any]] = []
        self.rejection_counter: Counter = Counter()
        self.solver_gap_log: List[Dict[str, Any]] = []
        self.stage_summaries: Dict[str, Any] = {}

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

    def _solve(self, role: str, question: str, tools: List[Dict[str, Any]],
               gold_calls, gold_obs, gold_answer, *, strong: bool,
               seed: int) -> Dict[str, Any]:
        attempts = STRONG_ATTEMPTS if strong else 1
        scores: List[Dict[str, Any]] = []
        for a in range(attempts):
            resp = self.client.chat(
                role=role, model=self.models[role],
                messages=solver_messages(question, tools, strong=strong),
                temperature=0.7 if strong else 0.2,
                max_tokens=STRONG_MAX_TOKENS if strong else WEAK_MAX_TOKENS,
                json_mode=True, seed=seed + a)
            if resp.get("dry_run"):
                return {"score": 0.0, "status": "dry_run", "n_calls": 0}
            parsed = resp["parsed"]
            calls = parse_solver_output(parsed)
            final = parsed.get("final_answer") if isinstance(parsed, dict) else None
            scores.append(score_prediction(calls, final, gold_calls,
                                           gold_obs, gold_answer))
        return best_of(scores)

    # ------------------------------------------------------------ main loop
    def generate_stage(self, stage: str, target: int) -> List[Dict[str, Any]]:
        assert stage in STAGES, f"unknown stage {stage}"
        rng = random.Random(f"agentic|{stage}|{self.seed}")
        recipe = Recipe()
        dedup = DedupIndex()
        accepted: List[Dict[str, Any]] = []
        iteration = 0
        contamination_strikes = 0
        # snapshot so the acceptance-rate stop only counts THIS stage
        rejections_at_stage_start = sum(self.rejection_counter.values())
        rounds_per_accept: List[int] = []
        rounds_since_accept = 0

        while len(accepted) < target:
            if iteration >= self.max_iterations:
                raise StageBudgetStop(
                    f"{stage}: iteration budget {self.max_iterations} exhausted "
                    f"at {len(accepted)}/{target} accepted")
            iteration += 1
            rounds_since_accept += 1
            motif = MOTIFS[(iteration - 1) % len(MOTIFS)]

            resp = self.client.chat(
                role="challenger", model=self.models["challenger"],
                messages=challenger_messages(
                    stage=stage, motif=motif,
                    n_candidates=CANDIDATES_PER_REQUEST,
                    feedback_block=recipe.feedback_block(), rng=rng),
                temperature=0.9, max_tokens=2400, json_mode=True,
                seed=self.seed * 1000 + iteration)
            if resp.get("dry_run"):
                print(f"[orchestrator] DRY RUN — stage {stage} stops here.")
                return []
            candidates = parse_candidates(resp["parsed"])
            if not candidates:
                self._reject(stage, {"question": resp["text"][:200]},
                             "invalid_json", "challenger output unparseable",
                             recipe)
                continue

            batch_counter: Counter = Counter()
            batch_weak: List[float] = []
            batch_strong: List[float] = []

            for cand in candidates:
                if len(accepted) >= target:
                    break
                cand = normalize_candidate(cand)
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
                # 2. deterministic executor — source of truth for gold
                v = deterministic_verify(cand)
                if not v["ok"]:
                    self._reject(stage, cand, v["reason"], v["detail"], recipe)
                    batch_counter[v["reason"]] += 1
                    continue
                observations, gold_answer = v["observations"], v["gold_answer"]
                question = cand["question"].strip()
                gold_calls = cand["gold_calls"]
                # 3. dedup + contamination
                dup = dedup.check_and_add(question, gold_calls)
                if dup:
                    self._reject(stage, cand, dup, None, recipe)
                    batch_counter[dup] += 1
                    continue
                sid = (f"agentic_v4_{_short_stage(stage)}_"
                       f"{len(accepted) + 1:06d}")
                if self.contamination is not None:
                    ok, why = self.contamination.check(question, gold_calls, sid)
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
                # 4. offered tools (used + distractors)
                tools = _offered_schemas(rng, [c["name"] for c in gold_calls],
                                         cand["motif_type"])
                # 5. weak solver (skip strong when weak passes — saves compute)
                weak = self._solve("weak_solver", question, tools, gold_calls,
                                   observations, gold_answer, strong=False,
                                   seed=self.seed + iteration)
                batch_weak.append(weak["score"])
                strong = None
                if weak["score"] <= 0.5:
                    strong = self._solve("strong_solver", question, tools,
                                         gold_calls, observations, gold_answer,
                                         strong=True, seed=self.seed + iteration)
                    batch_strong.append(strong["score"])
                gap_ok, gap_reason = solver_gap_verdict(weak, strong)
                gap_rec = {
                    "stage": stage, "weak_status": weak["status"],
                    "weak_score": weak["score"],
                    "strong_status": strong["status"] if strong else "skipped",
                    "strong_score": strong["score"] if strong else None,
                    "gap": round(strong["score"] - weak["score"], 3)
                    if strong else None,
                    "accepted": gap_ok,
                }
                self.solver_gap_log.append(gap_rec)
                if not gap_ok:
                    dedup.remove(question, gold_calls)
                    self._reject(stage, cand, gap_reason, None, recipe,
                                 extra={"solver_gap": gap_rec})
                    batch_counter[gap_reason] += 1
                    continue
                # 6. LLM style judge (secondary; cannot override execution)
                if self.run_judge:
                    jresp = self.client.chat(
                        role="judge", model=self.models["judge"],
                        messages=judge_messages(question, len(gold_calls)),
                        temperature=0.0, max_tokens=400, json_mode=True,
                        seed=self.seed)
                    jv = judge_verdict(jresp["parsed"])
                    if not jv["ok"]:
                        dedup.remove(question, gold_calls)
                        self._reject(stage, cand, jv["reason"], jv["detail"],
                                     recipe)
                        batch_counter[jv["reason"]] += 1
                        continue
                # 7. accept
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
                        "weak_status": weak["status"],
                        "strong_status": strong["status"],
                        "weak_score": weak["score"],
                        "strong_score": strong["score"],
                        "gap": round(strong["score"] - weak["score"], 3),
                    },
                    provenance={
                        "recipe_version": recipe.version,
                        "iteration": iteration,
                        "prompt_hash": None,
                        "raw_response_path": _relpath_or_none(
                            resp.get("raw_path"), self.out_root),
                        "created_at": _now(),
                        "tool_schema_source_policy": "aggregate_style_only",
                    })
                accepted.append(row)
                rounds_per_accept.append(rounds_since_accept)
                rounds_since_accept = 0

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

            # ---- stop condition: acceptance rate too low after warmup ------
            if iteration >= WARMUP_BATCHES:
                n_rej = sum(self.rejection_counter.values()) - rejections_at_stage_start
                n_acc = len(accepted)
                rate = n_acc / max(1, n_acc + n_rej)
                if rate < MIN_ACCEPT_RATE:
                    raise StageBudgetStop(
                        f"{stage}: acceptance rate {rate:.3f} < {MIN_ACCEPT_RATE} "
                        f"after {iteration} batches ({n_acc} accepted / {n_rej} "
                        "rejected) — revise the recipe manually")

        self.stage_summaries[stage] = {
            "target": target, "accepted": len(accepted),
            "iterations": iteration,
            "mean_rounds_per_accept": round(
                sum(rounds_per_accept) / len(rounds_per_accept), 2)
            if rounds_per_accept else None,
            "recipe": recipe.as_dict(),
        }
        return accepted


def _relpath_or_none(path: Optional[str], root: str) -> Optional[str]:
    if not path:
        return None
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return path


def write_outputs(out_root: str, accepted_by_stage: Dict[str, List[Dict[str, Any]]],
                  orch: Orchestrator) -> Dict[str, str]:
    """Write filtered/*.jsonl, rejected/*, and return stage → path map."""
    from .schema import STAGE_FILES
    filtered_dir = os.path.join(out_root, "filtered")
    rejected_dir = os.path.join(out_root, "rejected")
    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    for stage, rows in accepted_by_stage.items():
        path = os.path.join(filtered_dir, STAGE_FILES[stage])
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths[stage] = path
    with open(os.path.join(rejected_dir, "rejected_examples.jsonl"), "w",
              encoding="utf-8") as fh:
        for rec in orch.rejected:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(rejected_dir, "rejection_reasons.csv"), "w",
              encoding="utf-8", newline="") as fh:
        fh.write("reason,count\n")
        for reason, count in orch.rejection_counter.most_common():
            fh.write(f"{reason},{count}\n")
    return paths
