"""Pilot4 / observability CLI.

Dispatched from :mod:`targeted_tool_data.cli` so the package keeps a single
entry point:

    python -m targeted_tool_data.cli audit-provenance --output-dir ...
    python -m targeted_tool_data.cli audit-query-realism --profile-safe
    python -m targeted_tool_data.cli build-profile-v2 --nestful-dev ...
    python -m targeted_tool_data.cli capability-audit
    python -m targeted_tool_data.cli generate-pilot4 --config ...
    python -m targeted_tool_data.cli compare-datasets --baseline ... --candidate ...
    python -m targeted_tool_data.cli simulate-sampler --rollout-log ...
    python -m targeted_tool_data.cli implementation-report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PILOT4_COMMANDS = [
    "audit-provenance", "audit-query-realism", "build-profile-v2",
    "capability-audit", "generate-pilot4", "compare-datasets",
    "simulate-sampler", "implementation-report",
]

REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEV = (REPO_ROOT / "experiments" / "nestful_mtgrpo_minimal" /
               "data" / "splits" / "nestful_dev.jsonl")


def _load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


# ── Phase A ───────────────────────────────────────────────────────────────
def cmd_audit_provenance(args: argparse.Namespace, argv: Sequence[str]) -> int:
    from .provenance import run_provenance_audit

    out_dir = Path(args.output_dir or MODULE_ROOT / "reports" / "pilot3_provenance")
    result = run_provenance_audit(
        REPO_ROOT, out_dir,
        subset=Path(args.run_subset) if args.run_subset else None,
        parents=[Path(p) for p in args.parent_train] if args.parent_train else None,
        search_git_history=not args.no_git_history,
        cli_args=argv)
    audit = result.get("audit") or result
    print(f"[provenance] status={audit.get('status')} "
          f"canonical={audit.get('n_canonical_matched')}/"
          f"{audit.get('n_subset_rows')} -> {out_dir}")
    return 0


# ── Phase B ───────────────────────────────────────────────────────────────
# The parent the D1 checkpoint actually trained on is the commit-e83f57d
# revision of the pilot3 export, recovered by the provenance audit; the working
# tree copy was later regenerated. Both are audited so the difference is visible.
_PROFILE_SAFE_SETS = [
    ("d1_train_subset_300",
     "outputs/runpod_pilot3/train_nestful500_from_zip/train_nestful500/"
     "train_subset_300.jsonl", True),
    ("pilot3_train_600_as_trained",
     "reports/pilot3_provenance/_git_revisions/train_grpo_pilot3@e83f57de.jsonl", True),
    ("pilot3_train_600_worktree",
     "outputs/selected/export_pilot3/train_grpo_pilot3.jsonl", True),
    ("pilot3_heldout_200",
     "outputs/selected/export_pilot3/heldout_grpo_pilot3.jsonl", True),
    ("pilot3_reserve_200",
     "outputs/selected/export_pilot3/reserve_grpo_pilot3.jsonl", True),
]


def _resolve(rel: str) -> Optional[Path]:
    for base in (MODULE_ROOT, REPO_ROOT):
        p = base / rel
        if p.exists():
            return p
    return None


# diagnostic-500 has moved between bundles; it is read for exploratory reports
# only and never for generation quotas, so all known copies are acceptable
_DIAGNOSTIC_CANDIDATES = [
    "outputs/runpod_pilot3/diagnostic_500.jsonl",
    "runpod_bundle_pilot3/data/nestful_diagnostic_500.jsonl",
    "runpod_bundle_pilot2/data/nestful_diagnostic_500.jsonl",
]


def _resolve_diagnostic(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for rel in _DIAGNOSTIC_CANDIDATES:
        p = _resolve(rel)
        if p is not None:
            return p
    return None


def cmd_audit_query_realism(args: argparse.Namespace, argv: Sequence[str]) -> int:
    from . import query_realism as qr
    from .repro import stamp, write_csv, write_json, write_text

    out_dir = Path(args.output_dir or MODULE_ROOT / "reports" / "query_realism")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_task: List[Dict[str, Any]] = []
    aggregates: List[Dict[str, Any]] = []
    inputs: List[Path] = []

    for label, rel, keep_text in _PROFILE_SAFE_SETS:
        path = Path(rel) if Path(rel).exists() else _resolve(rel)
        if path is None:
            print(f"[query-realism] missing {rel} — skipped")
            continue
        inputs.append(path)
        rows, agg = qr.audit_dataset(_load_jsonl(path), label, keep_text=keep_text)
        per_task += rows
        aggregates.append(agg)

    dev = Path(args.nestful_dev) if args.nestful_dev else DEFAULT_DEV
    if dev.exists():
        inputs.append(dev)
        rows, agg = qr.audit_dataset(_load_jsonl(dev), "nestful_dev_200",
                                     keep_text=False)
        per_task += rows
        aggregates.append(agg)

    fields = ["dataset", "task_ref", "call_count", "call_bucket", "query_mode",
              "confidence", "n_gold_operations", "n_exactly_named_operations",
              "n_lexically_cued_operations", "n_implicit_operations",
              "exact_operation_coverage", "lexical_operation_coverage",
              "implicit_operation_rate", "sequence_leakage", "lcs_ratio",
              "kendall_agreement", "exact_ordered_operation_coverage",
              "adjacent_step_mappable_share", "procedural_cue_count",
              "step_number_count", "explicit_intermediate_reference_count"]
    write_csv(out_dir / "QUERY_REALISM_PER_TASK.csv", per_task, fields)
    write_json(out_dir / "QUERY_REALISM_PROFILE.json", {
        "schema_version": qr.SCHEMA_VERSION,
        "lexicon_version": qr.LEXICON_VERSION,
        "mode": "PROFILE_SAFE",
        "datasets": aggregates,
        "provenance": stamp(REPO_ROOT, schema_version=qr.SCHEMA_VERSION,
                            cli_args=argv, input_paths=inputs),
    })
    write_text(out_dir / "QUERY_REALISM_AUDIT_PROFILE_SAFE.md",
               qr.markdown_report(
                   "QUERY_REALISM_AUDIT_PROFILE_SAFE", aggregates,
                   notes=["Synthetic pilot3 sets keep their text; the NESTFUL "
                          "dev profile stores aggregates and hashed ids only."]))
    write_text(out_dir / "PLAN_LEAK_EXAMPLES.md",
               qr.examples_markdown(qr.select_examples(per_task)))

    # diagnostic-500 stays in a separate, exploratory report
    diag = _resolve_diagnostic(args.diagnostic)
    if diag is not None:
        rows, agg = qr.audit_dataset(_load_jsonl(diag), "diagnostic_500",
                                     keep_text=False)
        write_text(out_dir / "QUERY_REALISM_AUDIT_DIAGNOSTIC_EXPLORATORY.md",
                   qr.markdown_report(
                       "QUERY_REALISM_AUDIT_DIAGNOSTIC_EXPLORATORY", [agg],
                       notes=["EXPLORATORY ONLY. Never a generation target.",
                              "Aggregates and hashed ids only; no benchmark text.",
                              f"Source: `{diag.name}` ({len(rows)} rows)."]))
    else:
        write_text(out_dir / "QUERY_REALISM_AUDIT_DIAGNOSTIC_EXPLORATORY.md",
                   "# QUERY_REALISM_AUDIT_DIAGNOSTIC_EXPLORATORY\n\n"
                   "diagnostic-500 was not found on disk; no exploratory "
                   "statistics were produced.\n")
    print(f"[query-realism] {len(per_task)} tasks over {len(aggregates)} datasets "
          f"-> {out_dir}")
    return 0


# ── Phase C ───────────────────────────────────────────────────────────────
def cmd_build_profile_v2(args: argparse.Namespace, argv: Sequence[str]) -> int:
    from .profile_v2 import (build_profile_v2, derive_topology_constraints,
                             markdown_report)
    from .repro import stamp, write_json, write_text

    dev = Path(args.nestful_dev) if args.nestful_dev else DEFAULT_DEV
    out = Path(args.output or MODULE_ROOT / "outputs" / "profiles" /
               "target_profile_v2.json")
    rows = _load_jsonl(dev)
    profile = build_profile_v2(rows, source="nestful_dev_200", mode="PROFILE_SAFE")
    profile["provenance"] = stamp(REPO_ROOT, schema_version=profile["schema_version"],
                                  cli_args=argv, input_paths=[dev])
    write_json(out, profile)
    write_json(out.parent / "topology_constraints_v2.json",
               derive_topology_constraints(profile))
    write_text(MODULE_ROOT / "reports" / "profile_v2" / "TARGET_PROFILE_V2.md",
               markdown_report(profile))
    print(f"[profile-v2] {len(rows)} dev rows -> {out}")
    return 0


# ── Phase D ───────────────────────────────────────────────────────────────
def cmd_capability_audit(args: argparse.Namespace, argv: Sequence[str]) -> int:
    from collections import Counter

    from . import capability as cap
    from . import query_realism as qr
    from .repro import stamp, write_csv, write_json, write_text

    out_dir = Path(args.output_dir or MODULE_ROOT / "reports" / "capability")
    registry = cap.build_registry()
    cov = cap.coverage(registry)
    errs = cap.validate(registry)

    write_json(out_dir / "CAPABILITY_REGISTRY.json", {
        "schema_version": cap.SCHEMA_VERSION,
        "families": cap.CAPABILITY_FAMILIES,
        "coverage": cov,
        "validation_errors": errs,
        "primitives": registry,
        "provenance": stamp(REPO_ROOT, schema_version=cap.SCHEMA_VERSION,
                            cli_args=argv),
    })
    write_text(out_dir / "CAPABILITY_REGISTRY_AUDIT.md",
               cap.markdown_report(registry, cov, errs))

    # PROFILE_SAFE demand: what the dev-200 profile actually asks for
    dev = Path(args.nestful_dev) if args.nestful_dev else DEFAULT_DEV
    demand: Counter = Counter()
    if dev.exists():
        idx = qr.surface_name_index()
        for row in _load_jsonl(dev):
            for call in row.get("output") or []:
                sid = idx.get(str((call or {}).get("name") or ""))
                demand[cap.family_of(sid) if sid else "unmapped"] += 1
    write_csv(out_dir / "CAPABILITY_GAPS_PROFILE_SAFE.csv",
              cap.gap_rows(registry, dict(demand)))

    diag = _resolve_diagnostic(args.diagnostic)
    diag_demand: Counter = Counter()
    if diag is not None:
        idx = qr.surface_name_index()
        for row in _load_jsonl(diag):
            for call in row.get("output") or []:
                sid = idx.get(str((call or {}).get("name") or ""))
                diag_demand[cap.family_of(sid) if sid else "unmapped"] += 1
    write_csv(out_dir / "CAPABILITY_GAPS_DIAGNOSTIC_EXPLORATORY.csv",
              cap.gap_rows(registry, dict(diag_demand)))
    print(f"[capability] {cov['n_primitives']} primitives, "
          f"{cov['n_families_populated']}/{cov['n_families_declared']} families "
          f"-> {out_dir}")
    return 0


# ── Phase M ───────────────────────────────────────────────────────────────
def cmd_generate_pilot4(args: argparse.Namespace, argv: Sequence[str]) -> int:
    from .pilot4.pipeline import DEFAULT_CONFIG, run_pipeline

    cfg: Dict[str, Any] = dict(DEFAULT_CONFIG)
    if args.config:
        cfg.update(_read_config(Path(args.config)))
    for key, val in (("candidate_target", args.candidates),
                     ("selected_total", args.selected), ("seed", args.seed)):
        if val is not None:
            cfg[key] = val
    if args.train or args.heldout or args.reserve:
        cfg["splits"] = {"train": args.train or cfg["splits"]["train"],
                         "heldout": args.heldout or cfg["splits"]["heldout"],
                         "reserve": args.reserve or cfg["splits"]["reserve"]}
    if args.run_v4:
        cfg["run_v4_minimal_path"] = True

    out_dir = Path(args.output_dir or MODULE_ROOT / "outputs" / cfg["run_id"])
    dev = Path(args.nestful_dev) if args.nestful_dev else DEFAULT_DEV
    result = run_pipeline(REPO_ROOT, out_dir, dev_path=dev, config=cfg,
                          cli_args=argv)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("validation_report", "selection_report",
                                   "freeze_manifest")}, indent=2))
    return 0


def _read_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text) or {}
    return json.loads(text)


# ── Phase N ───────────────────────────────────────────────────────────────
def cmd_compare_datasets(args: argparse.Namespace, argv: Sequence[str]) -> int:
    from .pilot4.compare import run_comparison

    out_dir = Path(args.output_dir or MODULE_ROOT / "reports" / "pilot3_vs_pilot4")
    result = run_comparison(REPO_ROOT, out_dir, baseline=args.baseline,
                            candidate=args.candidate,
                            nestful_dev=Path(args.nestful_dev) if args.nestful_dev
                            else DEFAULT_DEV,
                            cli_args=argv)
    print(f"[compare] {result['n_metrics']} metrics -> {out_dir}")
    return 0


# ── Phase O ───────────────────────────────────────────────────────────────
def cmd_simulate_sampler(args: argparse.Namespace, argv: Sequence[str]) -> int:
    from .sampling.simulate import run_simulation

    out_dir = Path(args.output_dir or MODULE_ROOT / "reports" / "sampler_simulation")
    result = run_simulation(
        REPO_ROOT, out_dir,
        rollout_log=Path(args.rollout_log) if args.rollout_log else None,
        dataset=Path(args.dataset) if args.dataset else None,
        samplers=args.sampler.split(",") if args.sampler else None,
        steps=args.steps, seed=args.seed or 0, cli_args=argv)
    print(f"[simulate-sampler] {result['n_samplers']} samplers, "
          f"{result['n_steps']} steps -> {out_dir}")
    return 0


# ── Phase 24: final report ────────────────────────────────────────────────
def cmd_implementation_report(args: argparse.Namespace,
                              argv: Sequence[str]) -> int:
    from .pilot4.report import build_report

    out_dir = Path(args.output_dir or MODULE_ROOT / "reports")
    result = build_report(REPO_ROOT, out_dir, run_id=args.run_id,
                          module_root=MODULE_ROOT, cli_args=argv)
    print(f"[report] provenance={result['provenance_status']} "
          f"metrics={result['n_metrics']} "
          f"tests={result['n_test_functions']} -> {out_dir}")
    return 0


# ── dispatch ──────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="targeted-data (pilot4)")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("audit-provenance")
    p.add_argument("--run-subset", default=None)
    p.add_argument("--parent-train", action="append", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-git-history", action="store_true")
    p.set_defaults(fn=cmd_audit_provenance)

    p = sub.add_parser("audit-query-realism")
    p.add_argument("--profile-safe", action="store_true", default=True)
    p.add_argument("--nestful-dev", default=None)
    p.add_argument("--diagnostic", default=None)
    p.add_argument("--output-dir", default=None)
    p.set_defaults(fn=cmd_audit_query_realism)

    p = sub.add_parser("build-profile-v2")
    p.add_argument("--nestful-dev", default=None)
    p.add_argument("--output", default=None)
    p.set_defaults(fn=cmd_build_profile_v2)

    p = sub.add_parser("capability-audit")
    p.add_argument("--nestful-dev", default=None)
    p.add_argument("--diagnostic", default=None)
    p.add_argument("--output-dir", default=None)
    p.set_defaults(fn=cmd_capability_audit)

    p = sub.add_parser("generate-pilot4")
    p.add_argument("--config", default=None)
    p.add_argument("--candidates", type=int, default=None)
    p.add_argument("--selected", type=int, default=None)
    p.add_argument("--train", type=int, default=None)
    p.add_argument("--heldout", type=int, default=None)
    p.add_argument("--reserve", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--nestful-dev", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--run-v4", action="store_true",
                   help="enable the bounded minimal-path search (slow)")
    p.set_defaults(fn=cmd_generate_pilot4)

    p = sub.add_parser("compare-datasets")
    p.add_argument("--baseline", default="pilot3")
    p.add_argument("--candidate", default="pilot4_profile_safe")
    p.add_argument("--nestful-dev", default=None)
    p.add_argument("--output-dir", default=None)
    p.set_defaults(fn=cmd_compare_datasets)

    p = sub.add_parser("simulate-sampler")
    p.add_argument("--rollout-log", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--sampler", default=None,
                   help="comma separated: uniform,dynamic,history_adaptive,curriculum")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None)
    p.set_defaults(fn=cmd_simulate_sampler)

    p = sub.add_parser("implementation-report")
    p.add_argument("--run-id", default="pilot4_profile_safe")
    p.add_argument("--output-dir", default=None)
    p.set_defaults(fn=cmd_implementation_report)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    args = build_parser().parse_args(argv)
    return int(args.fn(args, argv) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
