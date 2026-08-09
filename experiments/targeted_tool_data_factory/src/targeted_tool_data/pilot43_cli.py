"""CLI for the Pilot4.3 build steps.

Failure mode prevented: analysis scripts that only ever run from an ad-hoc
shell invocation and therefore never leave reproducible artefacts. Every step
below is reachable as ``python -m targeted_tool_data.cli <command>`` and writes
into ``outputs/``.

Windows PowerShell::

    $env:PYTHONPATH="src"; python -m targeted_tool_data.cli audit-pilot42-final
    $env:PYTHONPATH="src"; python -m targeted_tool_data.cli build-target-profile-v3

The commands are ordered like the pipeline, and each one refuses to run when its
predecessor's artifact is missing rather than producing an empty file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

MODULE_ROOT = Path(__file__).resolve().parents[2]
# the audit package lives at the factory root, outside the installed package
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

PILOT43_COMMANDS = [
    "audit-pilot42-final",
    "build-workflow-registry-v3",
    "validate-primitive-registry-v3",
    "build-target-profile-v3",
    "generate-pilot43-semantic",
    "validate-pilot43-semantic",
    "shortlist-pilot43",
    "run-pilot43-v4",
    "resume-audit-pilot43",
    "freeze-pilot43-selectable",
    "allocate-pilot43-render",
    "render-pilot43-openrouter",
    "render-pilot43-deterministic",
    "validate-pilot43-queries",
    "select-pilot43",
    "build-pilot43-nested-subsets",
    "gate-pilot43",
    "independent-audit-pilot43",
    "prepare-human-audit-pilot43",
    "import-human-audit-pilot43",
    "probe-pilot43-grpo-signal",
    "compare-pilot42-pilot43",
    "freeze-pilot43",
    "report-pilot43",
]

DEFAULT_PILOT42_EXPORT = MODULE_ROOT / "outputs" / "pilot4_2_workflow_grounded_v2"
DEFAULT_AUDIT_DIR = MODULE_ROOT / "outputs" / "pilot42_final_audit"
DEFAULT_FINAL_DIR = MODULE_ROOT / "outputs" / "pilot4_3_nestful_final"


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="targeted-data pilot43")
    ap.add_argument("command", choices=PILOT43_COMMANDS)
    ap.add_argument("--export-dir", default=None,
                    help="frozen Pilot4.2 export; read-only")
    ap.add_argument("--audit-dir", default=None,
                    help="where the Pilot4.2 per-task ledger and defect report go")
    ap.add_argument("--output-dir", default=None,
                    help="pilot4_3_nestful_final directory")
    ap.add_argument("--profile-source", default=None,
                    help="frozen dev-200 aggregate profile (target_profile_v2.json)")
    ap.add_argument("--dev-rows", default=None,
                    help="raw NESTFUL dev split used to recompute the patterns")
    ap.add_argument("--run-id", default=None,
                    help="must equal the output directory name")
    ap.add_argument("--candidate-target", type=int, default=58000)
    ap.add_argument("--shortlist-target", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--workers", default="auto")
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--all-answer-types", action="store_true",
                    help="kept for the documented invocation; V4 always covers all")
    ap.add_argument("--counterfactual-instances", type=int, default=5)
    ap.add_argument("--stage", choices=["smoke", "pilot", "full"], default="smoke")
    # rendering is network-bound, so its concurrency is unrelated to --workers
    ap.add_argument("--llm-workers", type=int, default=12)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--profile-core", type=int, default=3000)
    ap.add_argument("--long-horizon", type=int, default=1200)
    ap.add_argument("--capability-enrichment", type=int, default=600)
    ap.add_argument("--challenge", type=int, default=200)
    ap.add_argument("--heldout", type=int, default=1000)
    ap.add_argument("--reserve", type=int, default=1000)
    ap.add_argument("--sample-size", type=int, default=2000)
    ap.add_argument("--initial-rollouts", type=int, default=4)
    ap.add_argument("--max-rollouts", type=int, default=8)
    ap.add_argument("--human-sample", type=int, default=400)
    ap.add_argument("--ratings", default=None,
                    help="filled-in human_audit_import_template.csv")
    ap.add_argument("--provider", default="",
                    help="probe backend id, e.g. openai_compatible_local")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--model", default="")
    return ap


def _out(args: argparse.Namespace) -> Path:
    path = Path(args.output_dir or DEFAULT_FINAL_DIR)
    if args.run_id and path.name != args.run_id:
        raise SystemExit(f"--run-id {args.run_id!r} does not match output "
                         f"directory {path.name!r}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _workers(value: str) -> int:
    if str(value).strip() in ("auto", ""):
        return max(1, (os.cpu_count() or 2) - 1)
    return int(value)


def _emit(payload: Any) -> int:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


def _need(path: Path, what: str) -> None:
    if not path.exists():
        raise SystemExit(f"missing {path} -- run {what} first")


# ── Pilot4.2 audit and registries ────────────────────────────────────────
def _audit_pilot42_final(args: argparse.Namespace) -> int:
    from analysis import pilot42_final_audit as audit

    argv = [
        "--export-dir", str(args.export_dir or DEFAULT_PILOT42_EXPORT),
        "--audit-dir", str(args.audit_dir or DEFAULT_AUDIT_DIR),
        "--final-dir", str(args.output_dir or DEFAULT_FINAL_DIR),
    ]
    return audit.main(argv)


def _build_target_profile_v3(args: argparse.Namespace) -> int:
    from .pilot43 import profile as prof

    source = Path(args.profile_source) if args.profile_source else (
        MODULE_ROOT / prof.DEFAULT_SOURCE)
    dev_rows = Path(args.dev_rows) if args.dev_rows else prof.DEFAULT_DEV_ROWS
    out_dir = _out(args)
    target = out_dir / "target_profile_v3.json"
    built = prof.build_profile_v3(source_path=source, write_to=target,
                                  dev_rows_path=dev_rows)
    structural = built["structural_patterns"]
    return _emit({
        "path": str(target),
        "n_rows": built["n_rows"],
        "profile_hash": built["profile_hash"],
        "pattern_classifier": structural["classifier"],
        "classifier_mismatch": structural["classifier_mismatch"],
        "raw_dev_programs_found": structural["raw_dev_programs_found"],
        "sources": [s["path"] for s in built["sources"]],
    })


def _build_workflow_registry(args: argparse.Namespace) -> int:
    from .pilot43 import export as exp
    from .pilot43.blueprints import all_blueprints, assert_full_registry, registry_hash

    assert_full_registry()
    out_dir = _out(args)
    exp.write_registries(out_dir)
    blueprints = all_blueprints()
    domains: Dict[str, int] = {}
    plans = 0
    for bp in blueprints:
        domains[bp.domain] = domains.get(bp.domain, 0) + 1
        plans += len(bp.plans)
    return _emit({
        "path": str(out_dir / "workflow_registry_v3.json"),
        "workflow_families": len(blueprints),
        "capability_plan_variants": plans,
        "domains": len(domains),
        "per_domain": dict(sorted(domains.items())),
        "registry_hash": registry_hash(),
    })


def _validate_primitive_registry(args: argparse.Namespace) -> int:
    from .pilot43.ops import (CODING_FAMILIES, build_ops, registry_hash,
                              validate_ops)
    from .pilot43 import export as exp

    ops = build_ops()
    problems = validate_ops(ops)
    out_dir = _out(args)
    exp.write_registries(out_dir)
    families = {op.family for op in ops.values()}
    return _emit({
        "path": str(out_dir / "primitive_registry_v3.json"),
        "primitives": len(ops),
        "capability_families": len(families),
        "coding_primitives": sum(1 for op in ops.values() if op.coding_like),
        "coding_families": len([f for f in families if f in CODING_FAMILIES]),
        "registry_hash": registry_hash(ops),
        "problems": problems,
        "valid": not problems,
    })


# ── generation ───────────────────────────────────────────────────────────
def _generate(args: argparse.Namespace) -> int:
    from .pilot43 import pipeline as pipe

    out_dir = _out(args)
    return _emit(pipe.generate(out_dir, target=args.candidate_target,
                               seed=args.seed, workers=_workers(args.workers)))


def _validate_semantic(args: argparse.Namespace) -> int:
    from .pilot43 import pipeline as pipe

    out_dir = _out(args)
    _need(out_dir / pipe.CANDIDATES, "generate-pilot43-semantic")
    return _emit(pipe.validate_pool(out_dir))


def _shortlist(args: argparse.Namespace) -> int:
    from .pilot43 import pipeline as pipe
    from .pilot43 import profile as prof

    out_dir = _out(args)
    _need(out_dir / pipe.HARD_VALID, "validate-pilot43-semantic")
    return _emit(pipe.shortlist(out_dir, target=args.shortlist_target,
                                seed=args.seed + 1,
                                profile=prof.build_profile_v3()))


def _run_v4(args: argparse.Namespace) -> int:
    """V4, node necessity, distractors and the offered set: one verified pass."""
    from .pilot43 import pipeline as pipe

    out_dir = _out(args)
    _need(out_dir / pipe.SHORTLIST, "shortlist-pilot43")
    report = pipe.verify(out_dir, workers=_workers(args.workers),
                         chunk=args.chunk, limit=args.limit,
                         resume=args.resume)
    artifacts = pipe.write_verification_artifacts(out_dir)
    return _emit({**report, "artifacts": artifacts})


# ── resume / freeze / allocate ───────────────────────────────────────────
def _resume_audit(args: argparse.Namespace) -> int:
    import subprocess

    out_dir = _out(args)
    script = (Path(__file__).resolve().parents[2]
              / "scripts" / "pilot43_resume_audit.py")
    proc = subprocess.run([sys.executable, str(script), "--out-dir", str(out_dir)],
                          check=False)
    return int(proc.returncode)


def _freeze_selectable(args: argparse.Namespace) -> int:
    from .pilot43 import resume as res

    out_dir = _out(args)
    _need(out_dir / "verified_candidates.jsonl", "run-pilot43-v4")
    _need(out_dir / "query_render_shortlist.jsonl", "shortlist-pilot43")
    manifest = res.freeze_selectable(out_dir)
    reused = res.export_reused_llm(out_dir)
    return _emit({"manifest": manifest, "reused": reused})


def _allocate_render(args: argparse.Namespace) -> int:
    from .pilot43 import resume as res

    out_dir = _out(args)
    _need(out_dir / res.SELECTABLE_FINAL, "freeze-pilot43-selectable")
    if not (out_dir / res.REUSED_LLM).exists():
        res.export_reused_llm(out_dir)
    plan = res.build_render_allocation(out_dir, seed=args.seed)
    return _emit({k: v for k, v in plan.items()
                  if k not in ("allocated_llm_task_ids",
                               "allocated_deterministic_task_ids")})


# ── query rendering ──────────────────────────────────────────────────────
def _render_openrouter(args: argparse.Namespace) -> int:
    from .pilot43 import orrun, profile as prof, qstage, resume as res
    from .pilot43.orclient import (OpenRouterClient, get_api_key,
                                   load_openrouter_config)
    from .pilot43.pipeline import iter_jsonl

    out_dir = _out(args)
    _need(out_dir / "verified_candidates.jsonl", "run-pilot43-v4")
    if not get_api_key():
        return _emit({"stage": args.stage, "executed": False,
                      "reason": "no OPENROUTER_API_KEY in environment or .env",
                      "consequence": "LLM_VALIDATED stays false; implicit-mode "
                                     "queries fall back to deterministic renderers"})
    cfg = load_openrouter_config()
    client = OpenRouterClient(cfg, out_dir)
    tasks = qstage.build_render_tasks(out_dir, profile=prof.build_profile_v3(),
                                     seed=args.seed, limit=args.limit)
    # When a render allocation exists, only spend on the planned LLM subset.
    alloc_path = out_dir / "render_allocation_llm.jsonl"
    if alloc_path.exists() and args.stage == "full":
        allowed = {r["task_id"] for r in iter_jsonl(alloc_path)}
        # also keep anything already reusable so resume accounting stays honest
        if (out_dir / res.REUSED_LLM).exists():
            allowed |= {r["task_id"] for r in iter_jsonl(out_dir / res.REUSED_LLM)}
        before = len(tasks)
        tasks = [t for t in tasks if t["task_id"] in allowed]
        filtered = before - len(tasks)
    else:
        filtered = 0
    report = orrun.run_stage(args.stage, tasks, client, out_dir, seed=args.seed,
                             workers=max(1, int(args.llm_workers)),
                             progress_every=args.progress_every)
    client.close()
    return _emit({**report, "tasks_available": len(tasks),
                  "allocation_filtered": filtered})


def _render_deterministic(args: argparse.Namespace) -> int:
    from .pilot43 import profile as prof, qstage

    out_dir = _out(args)
    _need(out_dir / "verified_candidates.jsonl", "run-pilot43-v4")
    return _emit(qstage.render_pool(out_dir, profile=prof.build_profile_v3(),
                                    seed=args.seed, limit=args.limit,
                                    resume=args.resume))


def _validate_queries(args: argparse.Namespace) -> int:
    from .pilot43 import profile as prof, qstage

    out_dir = _out(args)
    _need(out_dir / qstage.DETERMINISTIC, "render-pilot43-deterministic")
    rows = qstage.selectable_rows(out_dir)
    report = qstage.finalise_pool(
        out_dir, expected=len(rows), rendered_now=0,
        mode_targets=qstage.mode_targets_from(prof.build_profile_v3()))
    (out_dir / "query_quality_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "template_diversity_report.json").write_text(
        json.dumps({"diversity": report["diversity"],
                    "gates": report["diversity_gates"]}, indent=1,
                   ensure_ascii=False), encoding="utf-8")
    return _emit(report)


# ── selection, export, gates ─────────────────────────────────────────────
def _select(args: argparse.Namespace) -> int:
    from .pilot43 import export as exp
    from .pilot43.pipeline import read_jsonl
    from .pilot43.qstage import QUERY_VALID

    out_dir = _out(args)
    _need(out_dir / QUERY_VALID, "validate-pilot43-queries")
    targets = {"PROFILE_CORE": args.profile_core,
               "LONG_HORIZON_ENRICHMENT": args.long_horizon,
               "CAPABILITY_ENRICHMENT": args.capability_enrichment,
               "CHALLENGE": args.challenge}
    report = exp.export_dataset(out_dir, seed=args.seed, targets=targets)
    master = read_jsonl(out_dir / exp.MASTER_FILE)
    exp.write_metric_tables(out_dir, master)
    (out_dir / "generation_cells_v3.json").write_text(
        json.dumps(exp.generation_cells(master), indent=1, ensure_ascii=False),
        encoding="utf-8")
    _write_solution_equivalence(out_dir, master)
    return _emit({k: v for k, v in report.items() if k != "files"})


def _write_solution_equivalence(out_dir: Path, master: List[Dict[str, Any]]) -> None:
    from collections import Counter

    classes: Counter = Counter()
    minimal_equal = 0
    for row in master:
        spec = row["verifier"]
        for name in spec["accepted_solution_classes"]:
            classes[name] += 1
        minimal_equal += int(spec["minimal_valid_call_count"]
                             == spec["gold_call_count"])
    payload = {
        "run_id": row["run_id"] if master else "",
        "n_tasks": len(master),
        "accepted_solution_classes": dict(classes),
        "tasks_where_gold_is_minimal": minimal_equal,
        "tasks_requiring_strict_trace": sum(
            1 for r in master if r["verifier"]["strict_trace_required"]),
        "note": ("Correctness is terminal-state plus answer-grounding. Strict trace "
                 "equality is required only where no alternative binding was found "
                 "for any node."),
    }
    (out_dir / "solution_equivalence_report.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")


def _nested_subsets(args: argparse.Namespace) -> int:
    """Re-check the nested subsets that ``select-pilot43`` wrote."""
    from .pilot43 import export as exp
    from .pilot43.pipeline import read_jsonl

    out_dir = _out(args)
    _need(out_dir / exp.MASTER_FILE, "select-pilot43")
    ids = {name: [r["task_id"] for r in read_jsonl(out_dir / name)]
           for name in ("train_mix_1000.jsonl", "train_mix_2000.jsonl",
                        "train_mix_3000.jsonl", exp.MASTER_FILE,
                        "train_profile_1000.jsonl", "train_profile_2000.jsonl",
                        "train_profile_3000.jsonl")
           if (out_dir / name).exists()}
    def nested(a: str, b: str) -> bool:
        return set(ids.get(a, [])).issubset(set(ids.get(b, [])))
    checks = {
        "mix_1000_in_2000": nested("train_mix_1000.jsonl", "train_mix_2000.jsonl"),
        "mix_2000_in_3000": nested("train_mix_2000.jsonl", "train_mix_3000.jsonl"),
        "mix_3000_in_master": nested("train_mix_3000.jsonl", exp.MASTER_FILE),
        "profile_1000_in_2000": nested("train_profile_1000.jsonl",
                                       "train_profile_2000.jsonl"),
        "profile_2000_in_3000": nested("train_profile_2000.jsonl",
                                       "train_profile_3000.jsonl"),
    }
    return _emit({"sizes": {k: len(v) for k, v in ids.items()},
                  "nesting": checks, "valid": all(checks.values())})


def _gate(args: argparse.Namespace) -> int:
    from .pilot43 import export as exp, gates

    out_dir = _out(args)
    _need(out_dir / exp.MASTER_FILE, "select-pilot43")
    payload = gates.run(out_dir)
    return _emit({"AUTOMATED_GATES_PASSED": payload["AUTOMATED_GATES_PASSED"],
                  "n_checks": payload["n_checks"],
                  "n_passed": payload["n_passed"],
                  "blocking_failures": payload["blocking_failures"],
                  "readiness": payload["readiness"]["statuses"]})


#: The audit is spec-driven so it can stay ignorant of the producer: it is told
#: where the declared values sit in the record and recomputes everything else from
#: the content. Writing the spec next to the report keeps the audit reproducible.
AUDIT_SPEC: Dict[str, Any] = {
    "run_label": "pilot4_3_nestful_final",
    "files": {
        "train": "train_master_5000.jsonl",
        "profile_core": "train_profile_core_3000.jsonl",
        "long_horizon": "train_long_horizon_1200.jsonl",
        "capability_enrichment": "train_capability_enrichment_600.jsonl",
        "challenge": "train_challenge_200.jsonl",
        "heldout": "heldout_all.jsonl",
        "reserve": "reserve_1000.jsonl",
    },
    "train_split": "train",
    "expected_counts": {"train": 5000, "profile_core": 3000,
                        "long_horizon": 1200, "capability_enrichment": 600,
                        "challenge": 200, "reserve": 1000},
    "declared_paths": {
        "call_count": "call_count",
        "structural_pattern": "declared.structural_pattern",
        "actual_structural_pattern": "declared.structural_pattern",
        "answer_type": "answer_type",
        "workflow_id": "workflow_id",
        "cell_tier": "cell_tier",
        "query_mode": "actual_query_mode",
    },
    "validation_paths": {
        "v4": "validation.v4",
        "critic": "validation.critic",
        "node_necessity": "validation.node_necessity",
    },
    "overlap_keys": ["workflow_id", "program_fingerprint",
                     "declared.normalized_capability_sequence",
                     "query_fingerprints.intent_fingerprint"],
    "overlap_against": ["heldout"],
    "dedupe_key": "task_id",
    "node_value_kinds": {"mode": "from_calls", "path": "observation"},
    "surface_map": {"source": "record_tools",
                    "semantic_id_key": "primitive_id",
                    "primitive_registry": "primitive_registry_v3.json"},
    "csv_name": "independent_audit_per_task.csv",
    "report_prefix": "PILOT43_INDEPENDENT_AUDIT",
    "text_key": "question",
}


def _independent_audit(args: argparse.Namespace) -> int:
    from analysis.pilot43_independent_audit.audit import audit_export

    out_dir = _out(args)
    _need(out_dir / "train_master_5000.jsonl", "select-pilot43")
    spec = dict(AUDIT_SPEC)
    (out_dir / "independent_audit_spec.json").write_text(
        json.dumps(spec, indent=1), encoding="utf-8")
    result = audit_export(out_dir, spec)
    return _emit({"verdict": result["verdict"],
                  "INDEPENDENT_AUDIT_PASSED": result["INDEPENDENT_AUDIT_PASSED"],
                  "n_records_audited": result["n_records_audited"],
                  "deficits": result["deficits"],
                  "disagreements": result["disagreements"]})


# ── human audit, probe ───────────────────────────────────────────────────
def _prepare_human(args: argparse.Namespace) -> int:
    from .pilot43 import human

    out_dir = _out(args)
    _need(out_dir / "train_master_5000.jsonl", "select-pilot43")
    stats = human.prepare(out_dir, size=args.human_sample, seed=args.seed + 7)
    if not (out_dir / human.RESULTS_FILE).exists():
        human.pending_notice(out_dir)
    return _emit(stats)


def _import_human(args: argparse.Namespace) -> int:
    from .pilot43 import human

    out_dir = _out(args)
    if not args.ratings:
        raise SystemExit("--ratings <csv> is required")
    path = Path(args.ratings)
    if not path.is_file():
        raise SystemExit(f"{path} not found")
    result = human.import_results(out_dir, path)
    return _emit({k: v for k, v in result.items() if k != "by_stratum"})


def _probe(args: argparse.Namespace) -> int:
    from .pilot43 import probe

    out_dir = _out(args)
    _need(out_dir / "train_master_5000.jsonl", "select-pilot43")
    sampler = None
    reason = "no --provider given"
    model = args.model or probe.DEFAULT_MODEL
    if args.provider:
        try:
            from .providers import make_provider

            backend = make_provider({"provider": args.provider,
                                     "base_url": args.base_url,
                                     "model": model})
            sampler = probe.provider_sampler(backend)
        except Exception as exc:                              # noqa: BLE001
            sampler = None
            reason = f"{type(exc).__name__}: {exc}"
    report = probe.run(out_dir, sampler=sampler, sample_size=args.sample_size,
                       initial_rollouts=args.initial_rollouts,
                       max_rollouts=args.max_rollouts, model=model,
                       provider_id=args.provider,
                       unavailable_reason=reason)
    return _emit({k: v for k, v in report.items() if k != "per_cell"})


# ── comparison, freeze, report ───────────────────────────────────────────
def _compare(args: argparse.Namespace) -> int:
    from .pilot43 import reports

    out_dir = _out(args)
    audit_dir = Path(args.audit_dir or DEFAULT_AUDIT_DIR)
    candidates = [audit_dir, out_dir]
    source = next((d for d in candidates
                   if (d / reports.P42_SUMMARY_FILE).is_file()
                   or (d / reports.P42_ROOT_CAUSE_FILE).is_file()), None)
    return _emit(reports.compare(out_dir, source))


def _freeze(args: argparse.Namespace) -> int:
    from .pilot43 import freeze

    out_dir = _out(args)
    _need(out_dir / "train_master_5000.jsonl", "select-pilot43")
    payload = freeze.build(out_dir, cli_args=sys.argv[1:],
                           seeds={"generate": args.seed,
                                  "shortlist": args.seed + 1,
                                  "select": args.seed,
                                  "render": 4242})
    return _emit({"input_hashes": len(payload["input_hashes"]),
                  "artifact_hashes": len(payload["artifact_hashes"]),
                  "git": payload["git"],
                  "workflow_registry_hash": payload["workflow_registry_hash"],
                  "primitive_registry_hash": payload["primitive_registry_hash"],
                  "source_snapshot": {
                      k: v for k, v in payload["source_snapshot"].items()
                      if k != "changed_files"}})


def _report(args: argparse.Namespace) -> int:
    from .pilot43 import reports

    out_dir = _out(args)
    payload = reports.implementation_report(out_dir)
    return _emit({"statuses": payload["statuses"],
                  "sizes": payload["sizes"],
                  "TRAINING_READY": payload["TRAINING_READY"],
                  "n_blocking_failures": len(payload["blocking_failures"])})


HANDLERS = {
    "audit-pilot42-final": _audit_pilot42_final,
    "build-workflow-registry-v3": _build_workflow_registry,
    "validate-primitive-registry-v3": _validate_primitive_registry,
    "build-target-profile-v3": _build_target_profile_v3,
    "generate-pilot43-semantic": _generate,
    "validate-pilot43-semantic": _validate_semantic,
    "shortlist-pilot43": _shortlist,
    "run-pilot43-v4": _run_v4,
    "resume-audit-pilot43": _resume_audit,
    "freeze-pilot43-selectable": _freeze_selectable,
    "allocate-pilot43-render": _allocate_render,
    "render-pilot43-openrouter": _render_openrouter,
    "render-pilot43-deterministic": _render_deterministic,
    "validate-pilot43-queries": _validate_queries,
    "select-pilot43": _select,
    "build-pilot43-nested-subsets": _nested_subsets,
    "gate-pilot43": _gate,
    "independent-audit-pilot43": _independent_audit,
    "prepare-human-audit-pilot43": _prepare_human,
    "import-human-audit-pilot43": _import_human,
    "probe-pilot43-grpo-signal": _probe,
    "compare-pilot42-pilot43": _compare,
    "freeze-pilot43": _freeze,
    "report-pilot43": _report,
}


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    handler = HANDLERS.get(args.command)
    if handler is None:                        # pragma: no cover - argparse guards
        raise ValueError(args.command)
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
