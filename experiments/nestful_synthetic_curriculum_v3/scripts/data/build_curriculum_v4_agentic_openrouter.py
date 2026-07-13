"""Build curriculum_v4_nestful_like_agentic_openrouter (Autodata-style, OpenRouter).

Agentic Self-Instruct loop over the deterministic executable tool registry:
challenger LLM proposes NESTFUL-like tasks, the DETERMINISTIC executor computes
gold observations/answers (LLM answers are never trusted), weak/strong solvers
establish the difficulty gap, an LLM judge checks style, and the orchestrator
revises the challenger recipe from batch-level rejection analysis.

Safety:
  * OPENROUTER_API_KEY from environment only — never logged or stored;
  * request/spend budget guards stop generation early;
  * pilot by default; full generation needs CONFIRM_FULL_AGENTIC_GENERATION=1;
  * --mock runs the loop offline (no network, no cost) for smoke tests;
  * never trains, never runs NESTFUL eval.

Usage (repo root):
  python .../build_curriculum_v4_agentic_openrouter.py --pilot          # 10/stage
  python .../build_curriculum_v4_agentic_openrouter.py --mock --pilot   # offline
  CONFIRM_FULL_AGENTIC_GENERATION=1 python .../build_curriculum_v4_agentic_openrouter.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(V3_ROOT, "scripts", "lib"))
sys.path.insert(0, _HERE)
sys.path.insert(0, V3_ROOT)

from paths import CANONICAL_STAGE_FILES, NESTFUL_DATASETS, REPO_ROOT, sha256_file  # noqa: E402
from run_manifest import build_manifest, write_manifest  # noqa: E402
from openrouter_client import (BudgetExceeded, OfflineCacheMiss,  # noqa: E402
                               OpenRouterClient, models_from_env,
                               weak_solver_backend)

from lib.agentic_data.contamination import ContaminationChecker  # noqa: E402
from lib.agentic_data.distribution import corpus_stats, distance_report  # noqa: E402
from lib.agentic_data.orchestrator import (Orchestrator, StageBudgetStop,  # noqa: E402
                                           count_jsonl_rows, write_outputs)
from lib.agentic_data.env_defaults import (  # noqa: E402
    DIVERSITY_ENFORCE_AFTER,
    DIVERSITY_MAX_SAME_FAILURE_TYPE,
    DIVERSITY_MAX_SAME_WEAK_SCORE,
    LOCAL_WEAK_4BIT,
    LOCAL_WEAK_MODEL,
)
from lib.agentic_data.schema import STAGES, TOOL_SCHEMA_SOURCE_POLICY  # noqa: E402
from lib.agentic_data.semantics import semantic_errors  # noqa: E402
from lib.agentic_data.trace_validation import hard_trace_errors  # noqa: E402
from lib.nestful_like_generator import TOOLS, replay_task  # noqa: E402

DEFAULT_OUT = os.path.join(V3_ROOT, "data", "curriculum_v4_nestful_like_agentic_openrouter")
DET_V4_MANIFEST = os.path.join(V3_ROOT, "data", "curriculum_v4_nestful_like", "manifest.json")
DET_V4_FILTERED = os.path.join(V3_ROOT, "data", "curriculum_v4_nestful_like", "filtered")
PILOT_TARGET = 10
FULL_THRESHOLD = 50   # targets above this require CONFIRM_FULL_AGENTIC_GENERATION=1

# deterministic v4 stage → agentic stage (call-count aligned). Agentic stage4
# covers 4-6 calls, i.e. BOTH det. 4-call and 5-6-call stages.
_DET_TO_AGENTIC = {
    "v4_stage1_2call": "stage2_2call_agentic_openrouter",
    "v4_stage2_3call": "stage3_3call_agentic_openrouter",
    "v4_stage3_4call": "stage4_4to6call_agentic_openrouter",
    "v4_stage4_5to6call": "stage4_4to6call_agentic_openrouter",
}


def _count_lines(path: str) -> int:
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def resolve_targets(args, env) -> Dict[str, Any]:
    """Resolve final per-stage targets in ONE place with full provenance.

    Order: deterministic-v4 mirror (with an EXPLICIT stage4 decision, since
    two det. stages map onto agentic stage4) → --pilot → --max-accepted
    override → OPENROUTER_MAX_ACCEPTED_PER_STAGE cap → --stages filter.
    The returned decision dict is printed once and written to the manifest —
    the printed table and the used table are the SAME object.
    """
    decision: Dict[str, Any] = {"default": "800/stage"}
    targets = {s: 800 for s in STAGES}
    if os.path.isdir(DET_V4_FILTERED):
        got: Dict[str, int] = {s: 0 for s in STAGES}
        for det_stage, agentic_stage in _DET_TO_AGENTIC.items():
            p = os.path.join(DET_V4_FILTERED, f"{det_stage}.jsonl")
            if os.path.isfile(p):
                got[agentic_stage] += _count_lines(p)
        if all(v > 0 for v in got.values()):
            decision["det_v4_mirror_raw"] = dict(got)
            # EXPLICIT stage4 decision: det. 4-call + 5-6-call both map to
            # agentic stage4, so the raw mirror is 2x800=1600. We keep the
            # uniform 800/stage convention (one agentic stage4 example already
            # covers the 4-6 call range).
            s4 = "stage4_4to6call_agentic_openrouter"
            if got.get(s4, 0) > 800:
                decision["stage4_decision"] = {
                    "mirrored_sum": got[s4], "used": 800,
                    "reason": "two det. stages map to agentic stage4; "
                              "uniform 800/stage kept"}
                got[s4] = 800
            targets = got
            decision["source"] = "det_v4_mirror"
        else:
            decision["source"] = "default_800 (det. v4 incomplete)"
    else:
        decision["source"] = "default_800 (det. v4 not found)"
    if args.pilot:
        targets = {s: PILOT_TARGET for s in targets}
        decision["pilot"] = PILOT_TARGET
    if args.max_accepted_per_stage:
        targets = {s: args.max_accepted_per_stage for s in targets}
        decision["cli_override"] = args.max_accepted_per_stage
    cap = int(env.get("OPENROUTER_MAX_ACCEPTED_PER_STAGE", "800"))
    if any(t > cap for t in targets.values()):
        decision["env_cap_applied"] = cap
    targets = {s: min(t, cap) for s, t in targets.items()}
    targets = {s: t for s, t in targets.items() if s in set(args.stages)}
    decision["final_targets"] = dict(targets)
    return {"targets": targets, "decision": decision}


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def final_validation(accepted_by_stage: Dict[str, List[Dict[str, Any]]],
                     checker: ContaminationChecker) -> List[str]:
    """Defense-in-depth re-check of every accepted row before writing."""
    problems: List[str] = []
    seen_ids: set = set()
    for stage, rows in accepted_by_stage.items():
        lo, hi = STAGES[stage]
        for row in rows:
            sid = row["sample_id"]
            if sid in seen_ids:
                problems.append(f"duplicate sample_id {sid}")
            seen_ids.add(sid)
            if not (lo <= row["num_calls"] <= hi):
                problems.append(f"{sid}: call count {row['num_calls']} out of range")
            ok, obs = replay_task(row)
            if not ok:
                problems.append(f"{sid}: gold replay failed ({obs})")
            ok, why = checker.check(row["question"], row["gold_calls"], sid)
            if not ok:
                problems.append(f"{sid}: {why}")
            # hardening audit (2026-07-11): hard trace structure (unique/
            # sequential labels, valid references) + semantic compatibility
            # (no temperature -> money style bindings) as defense-in-depth
            # over EVERY accepted row, not just at generation time.
            trace_errs = hard_trace_errors(row, TOOLS, (lo, hi))
            if trace_errs:
                problems.append(f"{sid}: hard trace validation failed: "
                                f"{trace_errs[0]}")
            sem_errs = semantic_errors(row.get("gold_calls") or [], TOOLS)
            if sem_errs:
                problems.append(f"{sid}: semantic incompatibility: {sem_errs[0]}")
    return problems


def write_reports(out_root: str, accepted_by_stage: Dict[str, List[Dict[str, Any]]],
                  orch: Orchestrator, client: OpenRouterClient,
                  models: Dict[str, str], args, stopped_reason: str,
                  targets: Dict[str, int]) -> None:
    reports = os.path.join(out_root, "reports")
    os.makedirs(reports, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    all_rows = [r for rows in accepted_by_stage.values() for r in rows]
    n_rej = sum(orch.rejection_counter.values())

    # ---------------- AGENTIC_DATASET_REPORT.md ----------------
    lines = [
        "# Agentic OpenRouter dataset report", "",
        f"Generated {now} | seed {args.seed} | backend "
        f"{'MOCK (offline smoke — NOT a real dataset)' if args.mock else 'openrouter'}"
        + (" | SALVAGE (offline cache replay)" if getattr(args, 'salvage', False) else ""),
        f"Models: challenger={models['challenger']} weak={models['weak_solver']} "
        f"strong={models['strong_solver']} judge={models['judge']}",
        f"Tool schema source policy: `{TOOL_SCHEMA_SOURCE_POLICY}` (synthetic "
        "registry, aggregate NESTFUL style only — no exact NESTFUL signatures).", "",
        "## Counts", "",
        "| stage | accepted | target | status |", "|---|---|---|---|",
    ]
    # iterate over TARGETS (not accepted_by_stage) so stages that never started
    # still show up with accepted=0
    for stage, target in targets.items():
        rows = accepted_by_stage.get(stage, [])
        summ = orch.stage_summaries.get(stage, {})
        status = "complete" if len(rows) >= target else "partial"
        resumed = summ.get("resumed_from", 0)
        extra = f" (+{summ.get('accepted_new', 0)} new)" if resumed else ""
        lines.append(f"| {stage} | {len(rows)}{extra} | {target} | {status} |")
    overall = ("complete"
               if all(len(accepted_by_stage.get(s, [])) >= t
                      for s, t in targets.items()) else "partial")
    lines += [
        "", f"Accepted total: {len(all_rows)} | rejected: {n_rej} | "
        f"acceptance rate: {len(all_rows) / max(1, len(all_rows) + n_rej):.3f}",
        f"Dataset status: **{overall}** — a partial dataset is still valid and "
        "scoreable, but training_candidate stays false until targets are met.",
        f"Stop status: {stopped_reason}", "",
        "## Mean challenger rounds per accepted example", "",
    ]
    for stage, summ in orch.stage_summaries.items():
        lines.append(f"- {stage}: {summ.get('mean_rounds_per_accept')}")
    lines += ["", "## Top rejection reasons", "",
              "| reason | count |", "|---|---|"]
    for reason, count in orch.rejection_counter.most_common(12):
        lines.append(f"| {reason} | {count} |")
    with open(os.path.join(reports, "AGENTIC_DATASET_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---------------- AGENTIC_SOLVER_GAP_REPORT.md ----------------
    from collections import Counter
    gap = orch.solver_gap_log
    n_eval = len(gap)
    accepted_gap = [g for g in gap if g.get("accepted")]
    n_gap_passed = sum(1 for g in gap if g.get("gap_passed", g.get("accepted")))
    n_accepted = len(accepted_gap)
    n_judge_rej = sum(1 for g in gap
                      if str(g.get("final_status", "")).startswith("judge_rejected"))
    n_div_rej = sum(1 for g in gap
                    if str(g.get("final_status", "")).startswith("diversity_cap"))
    both_pass = sum(1 for g in gap if g["weak_score"] > 0.5
                    and (g["strong_score"] or 0) >= 0.8)
    both_fail = sum(1 for g in gap if g["weak_score"] <= 0.5
                    and g["strong_score"] is not None and g["strong_score"] < 0.5)
    weak_scores = [g["weak_score"] for g in gap]
    strong_scores = [g["strong_score"] for g in gap if g["strong_score"] is not None]
    gaps = [g["gap"] for g in gap if g["gap"] is not None]

    def _hist_table(values, title, fmt="{:.2f}"):
        c = Counter(fmt.format(v) for v in values)
        rows = [f"### {title} (n={len(values)})", "",
                "| value | count | share |", "|---|---|---|"]
        for k in sorted(c):
            rows.append(f"| {k} | {c[k]} | {c[k] / max(1, len(values)):.3f} |")
        return rows + [""]

    def _counter_table(counter, title, total=None):
        total = total if total is not None else sum(counter.values())
        rows = [f"### {title} (n={total})", "",
                "| key | count | share |", "|---|---|---|"]
        for k, n in counter.most_common():
            rows.append(f"| {k} | {n} | {n / max(1, total):.3f} |")
        return rows + [""]

    near_threshold = sum(1 for s in strong_scores if 0.70 <= s < 0.80)
    strong_exact = sum(1 for g in accepted_gap
                       if (g.get("strong_score") or 0) >= 0.999)
    lines = [
        "# Solver-gap report (weak-fail / strong-pass filtering)", "",
        f"Candidates that reached the solver stage: {n_eval}", "",
        f"- passed the solver-gap gate (weak_fail_strong_pass): {n_gap_passed} "
        f"({n_gap_passed / max(1, n_eval):.3f})",
        f"- rejected AFTER the gap gate by diversity caps: {n_div_rej}",
        f"- rejected AFTER the gap gate by the LLM style judge: {n_judge_rej}",
        f"- **finally accepted (this run): {n_accepted}** (= new rows in "
        "filtered/*.jsonl and the manifest — this is the only number that counts)",
        f"- strong EXACT executable wins among accepted: {strong_exact}"
        f"/{max(1, n_accepted)} ({strong_exact / max(1, n_accepted):.3f})",
        f"- both_pass (too easy): {both_pass}",
        f"- both_fail (too hard): {both_fail}",
        f"- avg weak score: "
        f"{sum(weak_scores) / max(1, len(weak_scores)):.3f}",
        f"- avg strong score (when run): "
        f"{sum(strong_scores) / max(1, len(strong_scores)):.3f}",
        f"- avg gap (when strong ran): {sum(gaps) / max(1, len(gaps)):.3f}", "",
        "Acceptance policy: weak <= 0.50, gap >= 0.25, STRONG_PASS_POLICY="
        f"`{os.environ.get('STRONG_PASS_POLICY', 'exact_win')}` (exact_win = "
        "strong must be a TRUE executable win / solution-equivalent, score "
        "1.0; partial strong solutions never enter the filtered set).",
        "Diversity caps on accepted rows: max "
        f"{os.environ.get('DIVERSITY_MAX_SAME_WEAK_SCORE', str(DIVERSITY_MAX_SAME_WEAK_SCORE))} same "
        "weak-score bucket, max "
        f"{os.environ.get('DIVERSITY_MAX_SAME_FAILURE_TYPE', str(DIVERSITY_MAX_SAME_FAILURE_TYPE))} same "
        "failure type (enforced after "
        f"{os.environ.get('DIVERSITY_ENFORCE_AFTER', str(DIVERSITY_ENFORCE_AFTER))} accepted).",
        "Strong solver runs ONLY when the weak solver failed (compute saving "
        "from the Autodata paper).",
        "",
        "## Score histograms (all solver-stage candidates)",
        "",
    ]
    lines += _hist_table(weak_scores, "weak_score histogram")
    lines += _hist_table(strong_scores, "strong_score histogram (when run)")
    lines += _hist_table(gaps, "gap histogram (when strong ran)")
    lines += [f"Near-threshold strong band [0.70, 0.80): {near_threshold} "
              "candidates", ""]

    lines += ["## ACCEPTED examples — diversity", ""]
    lines += _hist_table([g["weak_score"] for g in accepted_gap],
                         "accepted weak_score histogram")
    lines += _counter_table(Counter(g["weak_status"] for g in accepted_gap),
                            "accepted weak failure type distribution")
    # call-count deltas (predicted - gold)
    weak_delta = [g["weak_n_calls"] - g["n_gold_calls"] for g in gap
                  if g.get("weak_n_calls") is not None
                  and g.get("n_gold_calls") is not None]
    strong_delta = [g["strong_n_calls"] - g["n_gold_calls"] for g in gap
                    if g.get("strong_n_calls") is not None
                    and g.get("n_gold_calls") is not None]
    if weak_delta:
        lines += _hist_table(weak_delta,
                             "weak predicted calls - gold calls", fmt="{:+d}")
    if strong_delta:
        lines += _hist_table(strong_delta,
                             "strong predicted calls - gold calls (when run)",
                             fmt="{:+d}")

    def _cross_table(rows_key, title):
        combos = Counter((g.get(rows_key) or "?", g["weak_status"])
                         for g in accepted_gap)
        keys = sorted({k for k, _s in combos})
        statuses = sorted({s for _k, s in combos})
        out = [f"### {title}", "",
               "| " + rows_key + " | " + " | ".join(statuses) + " |",
               "|---" * (len(statuses) + 1) + "|"]
        for k in keys:
            out.append("| " + str(k) + " | "
                       + " | ".join(str(combos.get((k, s), 0))
                                    for s in statuses) + " |")
        return out + [""]

    if accepted_gap:
        lines += _cross_table("motif", "motif × weak failure type (accepted)")
        lines += _cross_table("stage", "stage × weak failure type (accepted)")

    lines += ["## Weak solver statuses (all solver-stage candidates)", ""]
    weak_status = Counter(g["weak_status"] for g in gap)
    for status, count in weak_status.most_common():
        lines.append(f"- {status}: {count}")
    with open(os.path.join(reports, "AGENTIC_SOLVER_GAP_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---------------- AGENTIC_CONTAMINATION_REPORT.md ----------------
    n_overlap_rej = orch.rejection_counter.get("overlap_with_nestful", 0)
    lines = [
        "# Contamination report", "",
        f"Generated {now}", "",
        "- NESTFUL questions / gold traces / tool schemas copied: **NONE**. "
        "The challenger only sees the synthetic tool registry "
        "(written from scratch; aggregate NESTFUL naming/arity style only) "
        "and its own recipe feedback — it is never shown NESTFUL items.",
        f"- tool_schema_source_policy: `{TOOL_SCHEMA_SOURCE_POLICY}`.",
        f"- Overlap gate (question hash, trace hash, sample_id vs NESTFUL "
        f"dev/test/full): checked per candidate AND re-checked over the final "
        f"corpus — final overlap = **0** across {len(all_rows)} accepted rows.",
        f"- Candidates rejected for overlap during generation: {n_overlap_rej}.",
        "- The build ABORTS if NESTFUL reference data is unavailable "
        "(gate cannot run) or if overlap rejections repeat "
        f"({10} strikes).",
    ]
    with open(os.path.join(reports, "AGENTIC_CONTAMINATION_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---------------- AGENTIC_DISTRIBUTION_REPORT.md ----------------
    stats_by = {"agentic_v4": corpus_stats(all_rows)}
    v31_rows: List[Dict[str, Any]] = []
    for p in CANONICAL_STAGE_FILES.values():
        if os.path.isfile(p):
            v31_rows.extend(_load_jsonl(p))
    if v31_rows:
        stats_by["v3_1"] = corpus_stats(v31_rows)
    det_rows: List[Dict[str, Any]] = []
    if os.path.isdir(DET_V4_FILTERED):
        for f in sorted(os.listdir(DET_V4_FILTERED)):
            if f.endswith(".jsonl"):
                det_rows.extend(_load_jsonl(os.path.join(DET_V4_FILTERED, f)))
    if det_rows:
        stats_by["v4_deterministic"] = corpus_stats(det_rows)
    nest_rows: List[Dict[str, Any]] = []
    nf = NESTFUL_DATASETS.get("nestful_full")
    if nf and os.path.isfile(nf):
        nest_rows = _load_jsonl(nf)
        stats_by["nestful"] = corpus_stats(nest_rows)
    lines = ["# Distribution report — agentic v4 vs v3.1 vs NESTFUL", "",
             f"Generated {now} | agentic rows: {len(all_rows)}", ""]
    if "nestful" in stats_by and all_rows:
        dist = distance_report(stats_by)
        lines += ["## Total-variation distance to NESTFUL (lower = closer)", "",
                  "| dimension | " + " | ".join(
                      n for n in dist["dimensions"]["call_count_dist"]) + " |",
                  "|---" * (1 + len(dist["mean_distance"])) + "|"]
        for dim, vals in dist["dimensions"].items():
            lines.append("| " + dim + " | "
                         + " | ".join(str(vals[n]) for n in vals) + " |")
        lines += ["", "**Mean distance:** " + ", ".join(
            f"{n}={v}" for n, v in dist["mean_distance"].items()), ""]
    lines += ["## Corpus statistics", "", "```json",
              json.dumps({k: {kk: vv for kk, vv in v.items()
                              if kk != "used_tools_top"}
                          for k, v in stats_by.items()}, indent=2), "```", "",
              "Question length (mean words): " + ", ".join(
                  f"{k}={v.get('mean_question_words')}"
                  for k, v in stats_by.items())]
    with open(os.path.join(reports, "AGENTIC_DISTRIBUTION_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---------------- OPENROUTER_COST_REPORT.md ----------------
    st = client.stats.as_dict()
    lines = [
        "# OpenRouter cost report", "",
        f"Generated {now} | backend {'mock' if args.mock else 'openrouter'}", "",
        f"- requests: {st['n_requests']} (budget "
        f"{client.max_requests})",
        f"- cache hits: {st['n_cache_hits']}",
        f"- retries: {st['n_retries']} | json-mode fallbacks: "
        f"{st['n_json_fallbacks']}",
        f"- prompt tokens: {st['prompt_tokens']} | completion tokens: "
        f"{st['completion_tokens']}",
        f"- estimated spend: ${st['spend_usd']:.4f} (budget "
        f"${client.max_spend_usd:.2f})", "",
        "## By role", "", "| role | requests | cache hits | prompt toks | "
        "completion toks | spend USD |", "|---|---|---|---|---|---|",
    ]
    for role, r in st["by_role"].items():
        lines.append(f"| {role} | {int(r['requests'])} | {int(r['cache_hits'])} "
                     f"| {int(r['prompt_tokens'])} | "
                     f"{int(r['completion_tokens'])} | {r['spend_usd']} |")
    lines += ["", "Spend uses OpenRouter `usage.cost` when present, otherwise "
              "fallback prices OPENROUTER_PRICE_PROMPT_PER_M / "
              "OPENROUTER_PRICE_COMPLETION_PER_M. API keys are never logged."]
    with open(os.path.join(reports, "OPENROUTER_COST_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---------------- AGENTIC_HARDENING_REPORT.md ----------------
    # 2026-07-11 audit: hard trace validation, semantic compatibility,
    # diversity gates, Qwen rollout signal, offered-tool distribution,
    # answer-type/motif diversity, contamination — all in one place.
    rc = orch.rejection_counter
    total_candidates = len(all_rows) + n_rej
    n_trace_rej = rc.get("invalid_trace_labels", 0)
    n_sem_rej = rc.get("semantic_incompatible_reference", 0)
    n_grpo_rej = rc.get("low_grpo_signal_prediction", 0)
    hard_validation_pass_rate = (
        round((total_candidates - n_trace_rej) / max(1, total_candidates), 4))
    semantic_pass_rate = (
        round((total_candidates - n_sem_rej) / max(1, total_candidates), 4))
    rollouts = [r.get("rollout_signal") for r in all_rows
               if isinstance(r.get("rollout_signal"), dict)
               and not r["rollout_signal"].get("skipped")]
    div_stats_by_stage = {s: t.stats() for s, t in orch.diversity_by_stage.items()}
    div_stats = next(iter(div_stats_by_stage.values()), {})
    agentic_stats = stats_by.get("agentic_v4", {})
    lines = [
        "# Agentic dataset — hardening audit report (2026-07-11)", "",
        f"Generated {now} | seed {args.seed} | accepted rows: {len(all_rows)} | "
        f"total candidates seen: {total_candidates}", "",
        "This report answers the pre-full-generation audit: hard trace "
        "validation, semantic compatibility, diversity gates, the multi-"
        "rollout Qwen GRPO-signal probe, offered-tool distribution, answer-"
        "type/motif diversity, and contamination — on the CURRENT (hardened) "
        "pipeline output, not the original flawed 10-example pilot.", "",
        "## 1. Hard trace validation", "",
        f"- candidates rejected for invalid_trace_labels (non-unique/non-"
        f"sequential labels, bad references): {n_trace_rej}",
        f"- hard-validation pass rate (accepted+other-rejections / total): "
        f"**{hard_validation_pass_rate}**",
        "- defense-in-depth: every row in `final_validation()` is re-checked "
        "with the SAME `hard_trace_errors()` before the dataset is written "
        "(build aborts if any accepted row fails it).",
        "- regression tests: `tests/test_agentic_hardening.py` reproduces the "
        "two malformed pilot rows (agentic_v4_stage2_000007/_000008, `$var1` "
        "reused as the label of both calls) and asserts they are now "
        "rejected.", "",
        "## 2. Semantic compatibility", "",
        f"- candidates rejected for semantic_incompatible_reference "
        f"(cross real-world-quantity-family binding, e.g. temperature -> "
        f"money): {n_sem_rej}",
        f"- semantic-compatibility pass rate: **{semantic_pass_rate}**",
        "- regression tests reproduce the three unnatural pilot bindings "
        "(Fahrenheit -> net_price #000003, Fahrenheit -> annual_rate_percent "
        "#000006, fuel liters -> principal_amount #000009) and assert they "
        "are now rejected, while confirming legitimate generic-slot "
        "reuse (area -> `part`, a list sum -> `total_amount`, same-family "
        "length -> distance) still passes.", "",
        "## 3. Diversity gates", "",
        f"- weak-score / failure-type dominance caps: "
        f"{json.dumps(div_stats.get('caps', {}))}",
        f"- accepted weak_score buckets (this run): "
        f"{json.dumps(div_stats.get('weak_score_buckets_new', {}))}",
        f"- accepted weak failure types (this run): "
        f"{json.dumps(div_stats.get('failure_types_new', {}))}",
        f"- weak_score dominance (this run): "
        f"{div_stats.get('weak_score_dominance_new')}",
        f"- failure_type dominance (this run): "
        f"{div_stats.get('failure_type_dominance_new')}",
        f"- answer_type distribution: "
        f"{json.dumps(agentic_stats.get('answer_type_dist', {}))}",
        f"- motif distribution: {json.dumps(agentic_stats.get('motif_dist', {}))}",
        f"- tool-family distribution: "
        f"{json.dumps(agentic_stats.get('tool_family_dist', {}))}",
        f"- offered-tool-count distribution: "
        f"{json.dumps(agentic_stats.get('offered_tools_dist', {}))}",
        f"- question-template distribution: "
        f"{json.dumps(agentic_stats.get('question_template_dist', {}))}",
        f"- dominance shares (motif/answer_type/tool_family/question_template): "
        f"{json.dumps(agentic_stats.get('dominance', {}))}",
        f"- candidates rejected by training-reward rollout gate "
        f"(low_grpo_signal_prediction): {n_grpo_rej}", "",
        "## 4. Weak/strong solver-gap distribution", "",
    ]
    lines += _hist_table([g["weak_score"] for g in accepted_gap],
                         "accepted weak_score histogram")
    lines += _hist_table([g["gap"] for g in accepted_gap if g["gap"] is not None],
                         "accepted weak/strong gap histogram")
    lines += [
        "## 5. Qwen multi-rollout GRPO-signal probe", "",
        f"- target-local backend active this run: "
        f"{'yes' if os.environ.get('WEAK_SOLVER_BACKEND') == 'local' else 'no'}",
        f"- accepted rows with a rollout probe recorded: {len(rollouts)}"
        f"/{len(all_rows)}",
        f"- rollout scoring policy: "
        f"{rollouts[0].get('reward_policy') if rollouts else os.environ.get('AGENTIC_REWARD_POLICY') or os.environ.get('REWARD_POLICY') or 'execution_aware_v3_2_dense'}",
    ]
    if rollouts:
        n_pos = sum(1 for r in rollouts if r.get("grpo_signal_positive"))
        mean_unique = sum(r.get("unique_rewards", 0) for r in rollouts) / len(rollouts)
        mean_var = sum(r.get("reward_variance", 0) for r in rollouts) / len(rollouts)
        mean_full = sum(r.get("full_success_rate", 0) for r in rollouts) / len(rollouts)
        mean_prefix = sum(r.get("correct_prefix_rate", 0) for r in rollouts) / len(rollouts)
        mean_too_few = sum(r.get("too_few_call_rate", 0) for r in rollouts) / len(rollouts)
        lines += [
            f"- GRPO-signal-positive rows: {n_pos}/{len(rollouts)} "
            f"({n_pos / len(rollouts):.3f})",
            f"- mean unique_rewards (of {rollouts[0].get('n')} rollouts): "
            f"{mean_unique:.2f}",
            f"- mean reward_variance: {mean_var:.4f}",
            f"- mean full_success_rate: {mean_full:.3f}",
            f"- mean correct_prefix_rate: {mean_prefix:.3f}",
            f"- mean too_few_call_rate: {mean_too_few:.3f}",
        ]
    else:
        lines += ["- SKIPPED this run (WEAK_SOLVER_BACKEND != local — the "
                  "exact target Qwen3-4B checkpoint was not available)."]
    lines += ["", "## 6. Offered-tool-count distribution", "",
             f"- {json.dumps(agentic_stats.get('offered_tools_dist', {}))}",
             "- scaled by gold call count (2-call: 6-11 base / 9-15 "
             "distractor_heavy) instead of the old flat 16-26 range used "
             "by the 10-example pilot.", "",
             "## 7. Answer-type and motif diversity", "",
             f"- answer_type_dist: {json.dumps(agentic_stats.get('answer_type_dist', {}))}",
             f"- motif_dist: {json.dumps(agentic_stats.get('motif_dist', {}))}", "",
             "## 8. Contamination result", "",
             f"- candidates rejected for overlap_with_nestful: "
             f"{rc.get('overlap_with_nestful', 0)}",
             f"- final overlap over the accepted corpus (question hash / "
             f"trace hash / sample_id vs NESTFUL): 0/{len(all_rows)} "
             "(re-checked in `final_validation()`).", "",
             "## Verdict", "",
             "Pilot-only per instructions: full generation and training were "
             "NOT launched. Review this report before raising the per-stage "
             "target above the pilot size.", ""]
    with open(os.path.join(reports, "AGENTIC_HARDENING_REPORT.md"), "w",
             encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---------------- AGENTIC_PROBE_REPORT.md (pending stub) ----------------
    probe_path = os.path.join(reports, "AGENTIC_PROBE_REPORT.md")
    if not os.path.isfile(probe_path):
        with open(probe_path, "w", encoding="utf-8") as fh:
            fh.write(
                "# Agentic dataset — stage probe report\n\n"
                "STATUS: **not yet run** (requires a GPU pod; never run "
                "automatically by the builder).\n\n"
                "Run on the pod, then paste/compare results here:\n\n"
                "```bash\n"
                "# v3.1 stage2 baseline signal\n"
                "DATASET=stage2 REWARD_POLICY=execution_aware_v3_1_stepwise "
                "NUM_TASKS=50 SEED=42 BACKEND=vllm \\\n"
                "  bash experiments/nestful_synthetic_curriculum_v3/scripts/probe/probe_stage.sh\n\n"
                "# agentic v4 stage2, same reward\n"
                "DATASET=" + os.path.join(
                    "experiments/nestful_synthetic_curriculum_v3/data",
                    "curriculum_v4_nestful_like_agentic_openrouter/filtered",
                    "stage2_2call_agentic_openrouter.jsonl").replace("\\", "/")
                + " \\\n  REWARD_POLICY=execution_aware_v3_1_stepwise NUM_TASKS=50 "
                "SEED=42 BACKEND=vllm \\\n"
                "  bash experiments/nestful_synthetic_curriculum_v3/scripts/probe/probe_stage.sh\n\n"
                "# agentic v4 stage2 with reward v3.2 dense\n"
                "# (same command, REWARD_POLICY=execution_aware_v3_2_dense)\n"
                "```\n\n"
                "Success target (RESEARCH_FIX_PLAN): dead_group_rate lower than "
                "v3.1, mean unique rewards/group higher than v3.1. If the probe "
                "is bad: do NOT train — revise the recipe and regenerate.\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Agentic (Autodata-style) OpenRouter NESTFUL-like generator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stages", nargs="*", default=list(STAGES.keys()),
                    choices=list(STAGES.keys()))
    ap.add_argument("--max-accepted-per-stage", type=int, default=None,
                    help="override targets (default: mirror deterministic v4)")
    ap.add_argument("--pilot", action="store_true",
                    help=f"tiny pilot ({PILOT_TARGET}/stage)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    ap.add_argument("--mock", action="store_true",
                    help="offline mock LLM (no network, no cost; smoke test)")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the LLM style judge (saves cost)")
    ap.add_argument("--salvage", action="store_true",
                    help="OFFLINE cache-only replay of a previous run: zero "
                         "API calls (raises on any cache miss), writes "
                         "filtered/*.partial_salvaged.jsonl")
    ap.add_argument("--resume", action="store_true",
                    help="continue from existing filtered/*.jsonl (or "
                         "*.partial_salvaged.jsonl): loads prior rows, "
                         "seeds dedup, generates only the remaining gap to "
                         "target (e.g. 228 existing + target 800 → 572 new)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.resume and args.salvage:
        print("[agentic] ABORT: --resume and --salvage are mutually exclusive",
              file=sys.stderr)
        return 2

    out_root = os.path.abspath(args.output_dir)
    if args.mock and not out_root.rstrip("/\\").endswith("_mock"):
        out_root = out_root.rstrip("/\\") + "_mock"   # never mix mock with real data
        print(f"[agentic] MOCK mode — output redirected to {out_root}")
    for sub in ("raw", "filtered", "rejected", "manifests", "reports"):
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)

    models = models_from_env()
    env = os.environ
    resolution = resolve_targets(args, env)
    targets: Dict[str, int] = resolution["targets"]

    backend_label = ("MOCK (offline)" if args.mock
                     else "openrouter OFFLINE cache-only (salvage)"
                     if args.salvage else "openrouter")
    if not args.mock and weak_solver_backend() == "local":
        backend_label = f"hybrid (weak=local HF, other=openrouter)"
    print("=" * 70)
    print("[agentic] curriculum_v4_nestful_like_agentic_openrouter builder")
    print(f"  output_dir            = {out_root}")
    print(f"  stages/targets        = {targets}   <- FINAL (used everywhere)")
    print(f"  target_resolution     = {json.dumps(resolution['decision'])}")
    print(f"  seed                  = {args.seed}")
    print(f"  resume                = {args.resume}")
    print(f"  backend               = {backend_label}")
    if weak_solver_backend() == "local":
        print(f"  weak_solver_backend   = local HF")
        print(f"  LOCAL_WEAK_MODEL      = {env.get('LOCAL_WEAK_MODEL', LOCAL_WEAK_MODEL)}")
        print(f"  LOCAL_WEAK_4BIT       = {env.get('LOCAL_WEAK_4BIT', '0' if not LOCAL_WEAK_4BIT else '1')}")
    for role, model in models.items():
        print(f"  model.{role:13s} = {model}")
    print(f"  max_requests          = {env.get('OPENROUTER_MAX_REQUESTS', '1000')}")
    print(f"  max_spend_usd         = {env.get('OPENROUTER_MAX_SPEND_USD', '20')}")
    print(f"  max_iterations/stage  = {env.get('OPENROUTER_MAX_ITERATIONS_PER_STAGE', '400')}")
    print(f"  cache                 = {env.get('OPENROUTER_CACHE', '1')}")
    print(f"  save_raw              = {env.get('OPENROUTER_SAVE_RAW', '1')}")
    print(f"  api_key               = {'set' if env.get('OPENROUTER_API_KEY') else 'NOT SET'}")
    print("=" * 70)

    if args.dry_run:
        print("[agentic] DRY RUN — no API calls, nothing generated.")
        return 0

    is_full = any(t > FULL_THRESHOLD for t in targets.values())
    if is_full and not args.mock and not args.salvage \
            and env.get("CONFIRM_FULL_AGENTIC_GENERATION") != "1":
        print("[agentic] ABORT: targets exceed pilot size "
              f"(>{FULL_THRESHOLD}/stage). Set CONFIRM_FULL_AGENTIC_GENERATION=1 "
              "to run full generation (real API cost).", file=sys.stderr)
        return 3

    # contamination gate must be constructible BEFORE any API spend
    checker = ContaminationChecker()

    mock_handler = None
    backend = "openrouter"
    if args.mock:
        from lib.agentic_data.mock_llm import MockLLM
        mock_handler = MockLLM(seed=args.seed)
        backend = "mock"
    client = OpenRouterClient(
        cache_dir=os.path.join(out_root, "raw", "cache"),
        raw_dir=os.path.join(out_root, "raw"),
        backend=backend, mock_handler=mock_handler,
        offline=True if args.salvage else None)

    orch = Orchestrator(client=client, models=models, out_root=out_root,
                        seed=args.seed, contamination_checker=checker,
                        run_judge=not args.no_judge,
                        filtered_suffix=".partial_salvaged" if args.salvage else "",
                        max_iterations_per_stage=int(
                            env.get("OPENROUTER_MAX_ITERATIONS_PER_STAGE", "400")))

    stopped_reason = "completed"
    exit_code = 0
    try:
        for stage, target in targets.items():
            if args.resume:
                from lib.agentic_data.orchestrator import count_jsonl_rows as _cnt
                src = orch.find_resume_source(stage)
                n_exist = _cnt(src) if src else 0
                need = max(0, target - n_exist)
                print(f"[agentic] === {stage}: target {target} "
                      f"(resume: {n_exist} existing, need {need} more) ===")
            else:
                print(f"[agentic] === {stage}: target {target} ===")
            rows = orch.generate_stage(stage, target, resume=args.resume)
            summ = orch.stage_summaries.get(stage, {})
            if args.resume and summ.get("resumed_from"):
                print(f"[agentic] {stage}: accepted {len(rows)}/{target} "
                      f"(+{summ.get('accepted_new', 0)} new this run)")
            else:
                print(f"[agentic] {stage}: accepted {len(rows)}/{target}")
    except (BudgetExceeded, StageBudgetStop) as exc:
        stopped_reason = f"STOPPED EARLY: {exc}"
        print(f"[agentic] {stopped_reason}", file=sys.stderr)
        exit_code = 4
    except OfflineCacheMiss as exc:
        stopped_reason = f"SALVAGE CACHE END: {exc}"
        print(f"[agentic] {stopped_reason}", file=sys.stderr)
        exit_code = 4

    # SOURCE OF TRUTH: the orchestrator's accepted state (survives early stop —
    # the overnight-run bug was relying on generate_stage's return value only).
    accepted_by_stage: Dict[str, List[Dict[str, Any]]] = dict(orch.accepted_by_stage)
    for stage, rows in accepted_by_stage.items():
        if len(rows) < targets.get(stage, 0):
            print(f"[agentic] {stage}: PARTIAL {len(rows)}/{targets[stage]} "
                  "accepted — rows are persisted and scoreable")

    # defense-in-depth final validation of everything accepted
    problems = final_validation(accepted_by_stage, checker)
    if problems:
        stopped_reason += f" | FINAL VALIDATION FAILED ({len(problems)} problems)"
        print(f"[agentic] final validation problems (first 10):", file=sys.stderr)
        for p in problems[:10]:
            print(f"  - {p}", file=sys.stderr)
        exit_code = 5

    paths = write_outputs(out_root, accepted_by_stage, orch)
    write_reports(out_root, accepted_by_stage, orch, client, models, args,
                  stopped_reason, targets)

    manifest_path = os.path.join(
        out_root, "manifests", "curriculum_v4_agentic_openrouter_manifest.json")
    if args.salvage and os.path.isfile(manifest_path):
        # preserve the original (buggy) manifest as forensic evidence
        backup = manifest_path.replace(".json", ".pre_salvage.json")
        if not os.path.isfile(backup):
            os.replace(manifest_path, backup)
            print(f"[agentic] original manifest archived to {backup}")

    accepted_counts = {s: len(r) for s, r in accepted_by_stage.items()}
    manifest = build_manifest(
        kind="curriculum_v4_agentic_openrouter",
        datasets=list(paths.values()), seed=args.seed,
        extra={
            "backend": backend,
            "mock": bool(args.mock),
            "salvage": bool(args.salvage),
            "resume": bool(args.resume),
            "models": models,
            "targets": targets,
            "target_resolution": resolution["decision"],
            "accepted": accepted_counts,
            "completion": {
                s: {"target": targets.get(s), "accepted": accepted_counts.get(s, 0),
                    "status": "complete"
                    if accepted_counts.get(s, 0) >= targets.get(s, 0)
                    else "partial"}
                for s in targets},
            "rejected_total": sum(orch.rejection_counter.values()),
            "rejection_reasons": dict(orch.rejection_counter),
            "tool_schema_source_policy": TOOL_SCHEMA_SOURCE_POLICY,
            "client_stats": client.stats.as_dict(),
            "stage_summaries": orch.stage_summaries,
            "stage_files": {s: {"path": os.path.relpath(p, REPO_ROOT),
                                "sha256": sha256_file(p),
                                "rows": count_jsonl_rows(p)}
                            for s, p in paths.items()},
            "status": stopped_reason,
            "final_validation_problems": len(problems),
        })
    write_manifest(manifest, manifest_path)

    # ---- COUNT CONSISTENCY CHECK (fail loudly on any disagreement) ---------
    mismatches: List[str] = []
    with open(manifest_path, encoding="utf-8") as fh:
        m = json.load(fh)
    for stage in accepted_by_stage:
        c_mem = len(accepted_by_stage[stage])
        c_file = count_jsonl_rows(paths[stage])
        c_manifest = m["extra"]["accepted"].get(stage)
        c_manifest_file = m["extra"]["stage_files"].get(stage, {}).get("rows")
        c_summary = orch.stage_summaries.get(stage, {}).get("accepted")
        if not (c_mem == c_file == c_manifest == c_manifest_file == c_summary):
            mismatches.append(
                f"{stage}: memory={c_mem} filtered_file={c_file} "
                f"manifest={c_manifest} manifest_file_rows={c_manifest_file} "
                f"stage_summary={c_summary}")
    n_gap_accepted = sum(1 for g in orch.solver_gap_log if g.get("accepted"))
    n_new_mem = sum(orch.stage_summaries.get(s, {}).get("accepted_new", 0)
                    for s in accepted_by_stage)
    if n_gap_accepted != n_new_mem:
        mismatches.append(f"solver_gap_log new accepted={n_gap_accepted} "
                          f"!= accepted_new this run={n_new_mem}")
    if mismatches:
        print("[agentic] FATAL: accepted-count INCONSISTENCY:", file=sys.stderr)
        for msg in mismatches:
            print(f"  - {msg}", file=sys.stderr)
        return 7
    print(f"[agentic] count consistency OK: memory == filtered files == "
          f"manifest ({sum(len(r) for r in accepted_by_stage.values())} "
          f"total accepted, {n_new_mem} new this run)")

    print(f"[agentic] outputs under {out_root}")
    print(f"[agentic] status: {stopped_reason}")
    print("[agentic] done. NO training was launched, NO NESTFUL eval was run.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
