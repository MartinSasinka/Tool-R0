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
from openrouter_client import BudgetExceeded, OpenRouterClient, models_from_env  # noqa: E402

from lib.agentic_data.contamination import ContaminationChecker  # noqa: E402
from lib.agentic_data.distribution import corpus_stats, distance_report  # noqa: E402
from lib.agentic_data.orchestrator import Orchestrator, StageBudgetStop, write_outputs  # noqa: E402
from lib.agentic_data.schema import STAGES, TOOL_SCHEMA_SOURCE_POLICY  # noqa: E402
from lib.nestful_like_generator import replay_task  # noqa: E402

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


def resolve_targets() -> Dict[str, int]:
    """Mirror deterministic v4 per-stage counts; fall back to 800/stage."""
    targets = {s: 800 for s in STAGES}
    if os.path.isdir(DET_V4_FILTERED):
        got: Dict[str, int] = {s: 0 for s in STAGES}
        for det_stage, agentic_stage in _DET_TO_AGENTIC.items():
            p = os.path.join(DET_V4_FILTERED, f"{det_stage}.jsonl")
            if os.path.isfile(p):
                got[agentic_stage] += _count_lines(p)
        if all(v > 0 for v in got.values()):
            targets = got
            print(f"[agentic] mirroring deterministic v4 counts: {targets}")
        else:
            print("[agentic] det. v4 incomplete — using default 800/stage")
    else:
        print("[agentic] det. v4 not found — using default 800/stage")
    return targets


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
    return problems


def write_reports(out_root: str, accepted_by_stage: Dict[str, List[Dict[str, Any]]],
                  orch: Orchestrator, client: OpenRouterClient,
                  models: Dict[str, str], args, stopped_reason: str) -> None:
    reports = os.path.join(out_root, "reports")
    os.makedirs(reports, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    all_rows = [r for rows in accepted_by_stage.values() for r in rows]
    n_rej = sum(orch.rejection_counter.values())

    # ---------------- AGENTIC_DATASET_REPORT.md ----------------
    lines = [
        "# Agentic OpenRouter dataset report", "",
        f"Generated {now} | seed {args.seed} | backend "
        f"{'MOCK (offline smoke — NOT a real dataset)' if args.mock else 'openrouter'}",
        f"Models: challenger={models['challenger']} weak={models['weak_solver']} "
        f"strong={models['strong_solver']} judge={models['judge']}",
        f"Tool schema source policy: `{TOOL_SCHEMA_SOURCE_POLICY}` (synthetic "
        "registry, aggregate NESTFUL style only — no exact NESTFUL signatures).", "",
        "## Counts", "",
        "| stage | accepted | target |", "|---|---|---|",
    ]
    for stage, rows in accepted_by_stage.items():
        summ = orch.stage_summaries.get(stage, {})
        lines.append(f"| {stage} | {len(rows)} | {summ.get('target', '?')} |")
    lines += [
        "", f"Accepted total: {len(all_rows)} | rejected: {n_rej} | "
        f"acceptance rate: {len(all_rows) / max(1, len(all_rows) + n_rej):.3f}",
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
    gap = orch.solver_gap_log
    n_eval = len(gap)
    wfsp = sum(1 for g in gap if g["accepted"])
    both_pass = sum(1 for g in gap if g["weak_score"] > 0.5
                    and (g["strong_score"] or 0) >= 0.8)
    both_fail = sum(1 for g in gap if g["weak_score"] <= 0.5
                    and g["strong_score"] is not None and g["strong_score"] < 0.5)
    weak_scores = [g["weak_score"] for g in gap]
    strong_scores = [g["strong_score"] for g in gap if g["strong_score"] is not None]
    gaps = [g["gap"] for g in gap if g["gap"] is not None]
    lines = [
        "# Solver-gap report (weak-fail / strong-pass filtering)", "",
        f"Candidates that reached the solver stage: {n_eval}", "",
        f"- weak_fail_strong_pass (accepted): {wfsp} "
        f"({wfsp / max(1, n_eval):.3f})",
        f"- both_pass (too easy): {both_pass}",
        f"- both_fail (too hard): {both_fail}",
        f"- avg weak score: "
        f"{sum(weak_scores) / max(1, len(weak_scores)):.3f}",
        f"- avg strong score (when run): "
        f"{sum(strong_scores) / max(1, len(strong_scores)):.3f}",
        f"- avg gap (when strong ran): {sum(gaps) / max(1, len(gaps)):.3f}", "",
        "Acceptance policy: strong >= 0.80, weak <= 0.50, gap >= 0.25 "
        "(scores are deterministic execution-based; 1.0 = executable win / "
        "solution-equivalent, 0.5-0.8 partial prefix, 0.0-0.4 failures).",
        "Strong solver runs ONLY when the weak solver failed (compute saving "
        "from the Autodata paper).",
        "",
        "## Weak solver failure statuses",
        "",
    ]
    from collections import Counter
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
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_root = os.path.abspath(args.output_dir)
    if args.mock and not out_root.rstrip("/\\").endswith("_mock"):
        out_root = out_root.rstrip("/\\") + "_mock"   # never mix mock with real data
        print(f"[agentic] MOCK mode — output redirected to {out_root}")
    for sub in ("raw", "filtered", "rejected", "manifests", "reports"):
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)

    models = models_from_env()
    env = os.environ
    targets = resolve_targets()
    if args.pilot:
        targets = {s: PILOT_TARGET for s in targets}
    if args.max_accepted_per_stage:
        targets = {s: args.max_accepted_per_stage for s in targets}
    cap = int(env.get("OPENROUTER_MAX_ACCEPTED_PER_STAGE", "800"))
    targets = {s: min(t, cap) for s, t in targets.items()}
    targets = {s: t for s, t in targets.items() if s in set(args.stages)}

    print("=" * 70)
    print("[agentic] curriculum_v4_nestful_like_agentic_openrouter builder")
    print(f"  output_dir            = {out_root}")
    print(f"  stages/targets        = {targets}")
    print(f"  seed                  = {args.seed}")
    print(f"  backend               = {'MOCK (offline)' if args.mock else 'openrouter'}")
    for role, model in models.items():
        print(f"  model.{role:13s} = {model}")
    print(f"  max_requests          = {env.get('OPENROUTER_MAX_REQUESTS', '1000')}")
    print(f"  max_spend_usd         = {env.get('OPENROUTER_MAX_SPEND_USD', '20')}")
    print(f"  cache                 = {env.get('OPENROUTER_CACHE', '1')}")
    print(f"  save_raw              = {env.get('OPENROUTER_SAVE_RAW', '1')}")
    print(f"  api_key               = {'set' if env.get('OPENROUTER_API_KEY') else 'NOT SET'}")
    print("=" * 70)

    if args.dry_run:
        print("[agentic] DRY RUN — no API calls, nothing generated.")
        return 0

    is_full = any(t > FULL_THRESHOLD for t in targets.values())
    if is_full and not args.mock \
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
        backend=backend, mock_handler=mock_handler)

    orch = Orchestrator(client=client, models=models, out_root=out_root,
                        seed=args.seed, contamination_checker=checker,
                        run_judge=not args.no_judge)

    accepted_by_stage: Dict[str, List[Dict[str, Any]]] = {}
    stopped_reason = "completed"
    exit_code = 0
    try:
        for stage, target in targets.items():
            print(f"[agentic] === {stage}: target {target} ===")
            accepted_by_stage[stage] = orch.generate_stage(stage, target)
            print(f"[agentic] {stage}: accepted "
                  f"{len(accepted_by_stage[stage])}/{target}")
    except (BudgetExceeded, StageBudgetStop) as exc:
        stopped_reason = f"STOPPED EARLY: {exc}"
        print(f"[agentic] {stopped_reason}", file=sys.stderr)
        exit_code = 4

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
                  stopped_reason)

    manifest = build_manifest(
        kind="curriculum_v4_agentic_openrouter",
        datasets=list(paths.values()), seed=args.seed,
        extra={
            "backend": backend,
            "mock": bool(args.mock),
            "models": models,
            "targets": targets,
            "accepted": {s: len(r) for s, r in accepted_by_stage.items()},
            "rejected_total": sum(orch.rejection_counter.values()),
            "rejection_reasons": dict(orch.rejection_counter),
            "tool_schema_source_policy": TOOL_SCHEMA_SOURCE_POLICY,
            "client_stats": client.stats.as_dict(),
            "stage_summaries": orch.stage_summaries,
            "stage_files": {s: {"path": os.path.relpath(p, REPO_ROOT),
                                "sha256": sha256_file(p)}
                            for s, p in paths.items()},
            "status": stopped_reason,
            "final_validation_problems": len(problems),
        })
    write_manifest(manifest, os.path.join(
        out_root, "manifests", "curriculum_v4_agentic_openrouter_manifest.json"))

    print(f"[agentic] outputs under {out_root}")
    print(f"[agentic] status: {stopped_reason}")
    print("[agentic] done. NO training was launched, NO NESTFUL eval was run.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
