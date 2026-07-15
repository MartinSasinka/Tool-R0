"""Build curriculum_v5_agentic_synthetic (Autodata-style, OpenRouter, v5 registry).

Agentic Self-Instruct loop over the VERSIONED EXECUTABLE v5 tool registry
(``lib/synthetic_tools.py``, ~163 tools): a challenger LLM proposes
NESTFUL-like tasks, the REAL trainer executor (``executor.py``
``ToolExecutor(mode="synthetic")``, via ``lib/agentic_data/exec_bridge.py``)
computes gold observations/answers (LLM answers are never trusted), weak/
strong solvers establish the difficulty gap (scored by executing THEIR
predicted calls through the SAME real executor — a wrong value never
receives the gold result), a multi-rollout GRPO-signal probe measures
training signal against the exact target Qwen weak-solver setup, an LLM
judge checks style, and the orchestrator SELECTS the best candidate out of
each batch (best-of-N — see ``orchestrator._composite_quality_score``)
instead of accepting every candidate that clears the gates in generation
order.

Differences from the legacy ``build_curriculum_v4_agentic_openrouter.py``:
  * tool registry: ``lib/synthetic_tools.py`` (v5, ~163 tools) instead of
    ``lib/nestful_like_generator.py`` (v4, ~34 tools);
  * execution: real ``executor.mode=synthetic`` everywhere (challenger
    verify, weak/strong solvers, rollout probe) instead of gold_replay-only;
  * candidate selection: best-of-N ranking per batch instead of first-pass;
  * output tree uses the ``curriculum_v5_agentic`` naming so
    ``scripts/lib/paths.py`` classifies it as a v5 (real-executor) dataset,
    never as a legacy v4 (gold_replay-only) one.

Safety:
  * OPENROUTER_API_KEY from environment only — never logged or stored;
  * request/spend budget guards stop generation early;
  * pilot by default; full generation needs CONFIRM_FULL_AGENTIC_GENERATION=1;
  * --mock runs the loop offline (no network, no cost) for smoke tests;
  * never trains, never runs NESTFUL eval.

Usage (repo root):
  python .../build_curriculum_v5_agentic_openrouter.py --pilot          # 10/stage
  python .../build_curriculum_v5_agentic_openrouter.py --mock --pilot   # offline
  CONFIRM_FULL_AGENTIC_GENERATION=1 python .../build_curriculum_v5_agentic_openrouter.py
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

from paths import NESTFUL_DATASETS, REPO_ROOT, sha256_file  # noqa: E402
from run_manifest import build_manifest, write_manifest  # noqa: E402
from openrouter_client import (BudgetExceeded, OfflineCacheMiss,  # noqa: E402
                               OpenRouterClient, models_from_env,
                               weak_solver_backend)

from lib.agentic_data.contamination import ContaminationChecker  # noqa: E402
from lib.agentic_data.distribution import corpus_stats, distance_report  # noqa: E402
from lib.agentic_data.orchestrator import (Orchestrator, StageBudgetStop,  # noqa: E402
                                           best_of_n_enabled,
                                           best_of_n_max_accepts,
                                           count_jsonl_rows, write_outputs)
from lib.agentic_data.env_defaults import (  # noqa: E402
    DIVERSITY_ENFORCE_AFTER,
    DIVERSITY_MAX_SAME_FAILURE_TYPE,
    DIVERSITY_MAX_SAME_WEAK_SCORE,
    LOCAL_WEAK_4BIT,
    LOCAL_WEAK_MODEL,
)
from lib.agentic_data.exec_bridge import (  # noqa: E402
    REGISTRY_SOURCE, REGISTRY_VERSION, TOOLS, registry_hash, replay_task,
)
from lib.agentic_data.schema import STAGES, TOOL_SCHEMA_SOURCE_POLICY  # noqa: E402
from lib.agentic_data.semantics import semantic_errors  # noqa: E402
from lib.agentic_data.trace_validation import hard_trace_errors  # noqa: E402

# NOTE: this directory name is load-bearing — scripts/lib/paths.py's
# `is_v5_agentic_synthetic_dataset_path()` matches "curriculum_v5_agentic" so
# probe_stage.py (and any future consumer) auto-selects
# executor.mode=synthetic for this tree, never the legacy gold_replay guard.
DEFAULT_OUT = os.path.join(V3_ROOT, "data", "curriculum_v5_agentic_synthetic")
PILOT_TARGET = 10
DEFAULT_TARGET_PER_STAGE = 800
FULL_THRESHOLD = 50   # targets above this require CONFIRM_FULL_AGENTIC_GENERATION=1


def resolve_targets(args, env) -> Dict[str, Any]:
    """Resolve final per-stage targets in ONE place with full provenance.

    Order: flat default (800/stage) → --pilot → --max-accepted override →
    OPENROUTER_MAX_ACCEPTED_PER_STAGE cap → --stages filter. The returned
    decision dict is printed once and written to the manifest — the printed
    table and the used table are the SAME object."""
    decision: Dict[str, Any] = {"default": f"{DEFAULT_TARGET_PER_STAGE}/stage"}
    targets = {s: DEFAULT_TARGET_PER_STAGE for s in STAGES}
    if args.pilot:
        targets = {s: PILOT_TARGET for s in targets}
        decision["pilot"] = PILOT_TARGET
    if args.max_accepted_per_stage:
        targets = {s: args.max_accepted_per_stage for s in targets}
        decision["cli_override"] = args.max_accepted_per_stage
    cap = int(env.get("OPENROUTER_MAX_ACCEPTED_PER_STAGE", str(DEFAULT_TARGET_PER_STAGE)))
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
    """Defense-in-depth re-check of every accepted row before writing —
    replays each gold trace through the REAL trainer executor
    (``executor.mode="synthetic"``), not gold_replay."""
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
                problems.append(f"{sid}: real-executor replay failed ({obs})")
            ok, why = checker.check(row["question"], row["gold_calls"], sid)
            if not ok:
                problems.append(f"{sid}: {why}")
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

    # ---------------- AGENTIC_V5_DATASET_REPORT.md ----------------
    lines = [
        "# Agentic v5 OpenRouter dataset report", "",
        f"Generated {now} | seed {args.seed} | backend "
        f"{'MOCK (offline smoke — NOT a real dataset)' if args.mock else 'openrouter'}"
        + (" | SALVAGE (offline cache replay)" if getattr(args, 'salvage', False) else ""),
        f"Registry: {REGISTRY_SOURCE} version={REGISTRY_VERSION} "
        f"hash={registry_hash()[:16]} | {len(TOOLS)} tools",
        f"Executor: executor.mode=synthetic (REAL execution) everywhere — "
        f"challenger verify, weak/strong solvers, rollout probe.",
        f"Models: challenger={models['challenger']} weak={models['weak_solver']} "
        f"strong={models['strong_solver']} judge={models['judge']}",
        f"Best-of-N: enabled={best_of_n_enabled()} "
        f"max_accepts_per_batch={best_of_n_max_accepts()}",
        f"Tool schema source policy: `{TOOL_SCHEMA_SOURCE_POLICY}` (synthetic "
        "registry, aggregate NESTFUL style only — no exact NESTFUL signatures).", "",
        "## Counts", "",
        "| stage | accepted | target | status |", "|---|---|---|---|",
    ]
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
    for reason, count in orch.rejection_counter.most_common(15):
        lines.append(f"| {reason} | {count} |")
    n_bon_rej = orch.rejection_counter.get("best_of_n_not_selected", 0)
    lines += ["", "## Best-of-N candidate selection", "",
              f"- candidates that lost the batch ranking "
              f"(best_of_n_not_selected): {n_bon_rej}",
              f"- max accepted per batch: {best_of_n_max_accepts()}"]
    with open(os.path.join(reports, "AGENTIC_V5_DATASET_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---------------- AGENTIC_V5_SOLVER_GAP_REPORT.md ----------------
    from collections import Counter
    gap = orch.solver_gap_log
    n_eval = len(gap)
    accepted_gap = [g for g in gap if g.get("accepted")]
    n_gap_passed = sum(1 for g in gap if g.get("gap_passed", g.get("accepted")))
    n_accepted = len(accepted_gap)
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

    lines = [
        "# Solver-gap report (weak-fail / strong-pass filtering, real executor)", "",
        f"Candidates that reached the solver stage: {n_eval}",
        f"- weak/strong solved with real ``executor.mode=synthetic`` "
        f"execution — a wrong predicted value never receives the gold result.",
        "",
        f"- passed the solver-gap gate (weak_fail_strong_pass): {n_gap_passed} "
        f"({n_gap_passed / max(1, n_eval):.3f})",
        f"- **finally accepted (this run): {n_accepted}** (after best-of-N "
        "selection, diversity caps, the rollout probe and the judge)",
        f"- avg weak score: {sum(weak_scores) / max(1, len(weak_scores)):.3f}",
        f"- avg strong score (when run): "
        f"{sum(strong_scores) / max(1, len(strong_scores)):.3f}",
        f"- avg gap (when strong ran): {sum(gaps) / max(1, len(gaps)):.3f}", "",
        "## Score histograms (all solver-stage candidates)", "",
    ]
    lines += _hist_table(weak_scores, "weak_score histogram")
    lines += _hist_table(strong_scores, "strong_score histogram (when run)")
    lines += _hist_table(gaps, "gap histogram (when strong ran)")
    with open(os.path.join(reports, "AGENTIC_V5_SOLVER_GAP_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---------------- AGENTIC_V5_DISTRIBUTION_REPORT.md ----------------
    stats_by = {"agentic_v5": corpus_stats(all_rows)}
    nest_rows: List[Dict[str, Any]] = []
    nf = NESTFUL_DATASETS.get("nestful_full")
    if nf and os.path.isfile(nf):
        nest_rows = _load_jsonl(nf)
        stats_by["nestful"] = corpus_stats(nest_rows)
    lines = ["# Distribution report — agentic v5 vs NESTFUL", "",
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
                          for k, v in stats_by.items()}, indent=2), "```"]
    with open(os.path.join(reports, "AGENTIC_V5_DISTRIBUTION_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---------------- OPENROUTER_COST_REPORT.md ----------------
    st = client.stats.as_dict()
    lines = [
        "# OpenRouter cost report", "",
        f"Generated {now} | backend {'mock' if args.mock else 'openrouter'}", "",
        f"- requests: {st['n_requests']} (budget {client.max_requests})",
        f"- cache hits: {st['n_cache_hits']}",
        f"- estimated spend: ${st['spend_usd']:.4f} (budget "
        f"${client.max_spend_usd:.2f})", "",
        "## By role", "", "| role | requests | cache hits | spend USD |",
        "|---|---|---|---|",
    ]
    for role, r in st["by_role"].items():
        lines.append(f"| {role} | {int(r['requests'])} | {int(r['cache_hits'])} "
                     f"| {r['spend_usd']} |")
    with open(os.path.join(reports, "OPENROUTER_COST_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---------------- AGENTIC_V5_PROBE_REPORT.md (pending stub) ----------------
    probe_path = os.path.join(reports, "AGENTIC_V5_PROBE_REPORT.md")
    if not os.path.isfile(probe_path):
        with open(probe_path, "w", encoding="utf-8") as fh:
            fh.write(
                "# Agentic v5 dataset — stage probe report\n\n"
                "STATUS: **not yet run** (requires a GPU pod; never run "
                "automatically by the builder).\n\n"
                "Run on the pod, then paste/compare results here:\n\n"
                "```bash\n"
                "DATASET=" + os.path.join(
                    "experiments/nestful_synthetic_curriculum_v3/data",
                    "curriculum_v5_agentic_synthetic/filtered",
                    "stage2_2call_agentic_openrouter.jsonl").replace("\\", "/")
                + " \\\n  REWARD_POLICY=execution_aware_v3_2_dense NUM_TASKS=50 "
                "SEED=42 BACKEND=vllm \\\n"
                "  bash experiments/nestful_synthetic_curriculum_v3/scripts/probe/probe_stage.sh\n"
                "```\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Agentic (Autodata-style) OpenRouter v5 synthetic-tools "
                    "generator — real executor.mode=synthetic + best-of-N.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stages", nargs="*", default=list(STAGES.keys()),
                    choices=list(STAGES.keys()))
    ap.add_argument("--max-accepted-per-stage", type=int, default=None,
                    help=f"override targets (default: {DEFAULT_TARGET_PER_STAGE}/stage)")
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
                         "seeds dedup, generates only the remaining gap")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.resume and args.salvage:
        print("[agentic-v5] ABORT: --resume and --salvage are mutually exclusive",
              file=sys.stderr)
        return 2

    out_root = os.path.abspath(args.output_dir)
    if args.mock and not out_root.rstrip("/\\").endswith("_mock"):
        out_root = out_root.rstrip("/\\") + "_mock"   # never mix mock with real data
        print(f"[agentic-v5] MOCK mode — output redirected to {out_root}")
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
        backend_label = "hybrid (weak=local HF, other=openrouter)"
    print("=" * 70)
    print("[agentic-v5] curriculum_v5_agentic_synthetic builder")
    print(f"  registry              = {REGISTRY_SOURCE} v{REGISTRY_VERSION} "
          f"hash={registry_hash()[:16]} ({len(TOOLS)} tools)")
    print(f"  executor.mode         = synthetic (REAL execution, everywhere)")
    print(f"  best_of_n             = enabled={best_of_n_enabled()} "
          f"max_accepts_per_batch={best_of_n_max_accepts()}")
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
        print("[agentic-v5] DRY RUN — no API calls, nothing generated.")
        return 0

    is_full = any(t > FULL_THRESHOLD for t in targets.values())
    if is_full and not args.mock and not args.salvage \
            and env.get("CONFIRM_FULL_AGENTIC_GENERATION") != "1":
        print("[agentic-v5] ABORT: targets exceed pilot size "
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
                src = orch.find_resume_source(stage)
                n_exist = count_jsonl_rows(src) if src else 0
                need = max(0, target - n_exist)
                print(f"[agentic-v5] === {stage}: target {target} "
                      f"(resume: {n_exist} existing, need {need} more) ===")
            else:
                print(f"[agentic-v5] === {stage}: target {target} ===")
            rows = orch.generate_stage(stage, target, resume=args.resume)
            summ = orch.stage_summaries.get(stage, {})
            if args.resume and summ.get("resumed_from"):
                print(f"[agentic-v5] {stage}: accepted {len(rows)}/{target} "
                      f"(+{summ.get('accepted_new', 0)} new this run)")
            else:
                print(f"[agentic-v5] {stage}: accepted {len(rows)}/{target}")
    except (BudgetExceeded, StageBudgetStop) as exc:
        stopped_reason = f"STOPPED EARLY: {exc}"
        print(f"[agentic-v5] {stopped_reason}", file=sys.stderr)
        exit_code = 4
    except OfflineCacheMiss as exc:
        stopped_reason = f"SALVAGE CACHE END: {exc}"
        print(f"[agentic-v5] {stopped_reason}", file=sys.stderr)
        exit_code = 4

    accepted_by_stage: Dict[str, List[Dict[str, Any]]] = dict(orch.accepted_by_stage)
    for stage, rows in accepted_by_stage.items():
        if len(rows) < targets.get(stage, 0):
            print(f"[agentic-v5] {stage}: PARTIAL {len(rows)}/{targets[stage]} "
                  "accepted — rows are persisted and scoreable")

    problems = final_validation(accepted_by_stage, checker)
    if problems:
        stopped_reason += f" | FINAL VALIDATION FAILED ({len(problems)} problems)"
        print("[agentic-v5] final validation problems (first 10):", file=sys.stderr)
        for p in problems[:10]:
            print(f"  - {p}", file=sys.stderr)
        exit_code = 5

    paths = write_outputs(out_root, accepted_by_stage, orch)
    write_reports(out_root, accepted_by_stage, orch, client, models, args,
                  stopped_reason, targets)

    accepted_counts = {s: len(r) for s, r in accepted_by_stage.items()}
    manifest_path = os.path.join(
        out_root, "manifests", "curriculum_v5_agentic_openrouter_manifest.json")
    manifest = build_manifest(
        kind="curriculum_v5_agentic_openrouter",
        datasets=list(paths.values()), seed=args.seed,
        extra={
            "backend": backend,
            "mock": bool(args.mock),
            "salvage": bool(args.salvage),
            "resume": bool(args.resume),
            "models": models,
            "registry_source": REGISTRY_SOURCE,
            "registry_version": REGISTRY_VERSION,
            "registry_hash": registry_hash(),
            "executor_mode": "synthetic",
            "best_of_n_enabled": best_of_n_enabled(),
            "best_of_n_max_accepts_per_batch": best_of_n_max_accepts(),
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
    if mismatches:
        print("[agentic-v5] FATAL: accepted-count INCONSISTENCY:", file=sys.stderr)
        for msg in mismatches:
            print(f"  - {msg}", file=sys.stderr)
        return 7
    print(f"[agentic-v5] count consistency OK: memory == filtered files == "
          f"manifest ({sum(len(r) for r in accepted_by_stage.values())} "
          f"total accepted)")

    print(f"[agentic-v5] outputs under {out_root}")
    print(f"[agentic-v5] status: {stopped_reason}")
    print("[agentic-v5] done. NO training was launched, NO NESTFUL eval was run.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
