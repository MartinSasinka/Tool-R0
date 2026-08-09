"""CLI commands for Pilot4.1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .pilot41 import RUN_ID
from .pilot41.graph_leak import run_pilot4_language_audit
from .pilot41.pipeline import (DEFAULT_CONFIG, finalize_dataset,
                               run_deterministic_phase, run_openrouter_stage)
from .pilot41.workflows import export_registry

REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_ROOT = Path(__file__).resolve().parents[2]

PILOT41_COMMANDS = [
    "audit-pilot4-language",
    "build-workflow-registry",
    "generate-semantic-pilot41",
    "select-render-shortlist",
    "render-queries-openrouter",
    "validate-queries",
    "select-pilot41",
    "run-v4-selected",
    "audit-pilot41",
    "freeze-pilot41",
    "implementation-report-pilot41",
]


def _out(args: argparse.Namespace) -> Path:
    return Path(args.output_dir or MODULE_ROOT / "outputs" / RUN_ID)


def cmd_audit_pilot4_language(args: argparse.Namespace, argv: Sequence[str]) -> int:
    out = Path(args.output_dir or MODULE_ROOT / "reports" / "pilot4_language_audit")
    summary = run_pilot4_language_audit(REPO_ROOT, out, cli_args=argv)
    print(f"[audit-pilot4-language] train_stages_related="
          f"{summary['train']['stages_related_phrase_rate']} "
          f"high_or_complete={summary['train']['high_or_complete_rate']} -> {out}")
    return 0


def cmd_build_workflow_registry(args: argparse.Namespace, argv: Sequence[str]) -> int:
    out = Path(args.output or _out(args) / "workflow_registry.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = export_registry(out)
    print(f"[build-workflow-registry] {payload['n_workflows']} workflows "
          f"hash={payload['registry_hash']} -> {out}")
    return 0


def cmd_generate_semantic(args: argparse.Namespace, argv: Sequence[str]) -> int:
    cfg = dict(DEFAULT_CONFIG)
    if args.candidates:
        cfg["candidate_target"] = args.candidates
    if args.seed is not None:
        cfg["seed"] = args.seed
    result = run_deterministic_phase(REPO_ROOT, _out(args), config=cfg, cli_args=argv)
    print(json.dumps({k: result[k] for k in result if k != "provenance"}, indent=2))
    return 0


def cmd_select_shortlist(args: argparse.Namespace, argv: Sequence[str]) -> int:
    # shortlist already written by generate; allow re-run from candidates
    from .pilot41.generate import select_render_shortlist
    from .repro import write_jsonl

    out = _out(args)
    path = out / "semantic_validated.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    short = select_render_shortlist(rows, target=args.target or 2000,
                                    seed=args.seed or 20260731)
    write_jsonl(out / "llm_render_shortlist.jsonl", short)
    write_jsonl(out / "query_validated.jsonl", short)
    print(f"[select-render-shortlist] {len(short)} -> {out}")
    return 0


def cmd_render_openrouter(args: argparse.Namespace, argv: Sequence[str]) -> int:
    cfg = dict(DEFAULT_CONFIG)
    if args.config:
        from .pilot41.openrouter import load_openrouter_config
        cfg["openrouter_config"] = args.config
    stage = args.stage or "smoke"
    n = {"smoke": cfg["smoke_n"], "pilot": cfg["pilot_n"],
         "full": args.n or 2200}.get(stage, args.n or 25)
    if args.n:
        n = args.n
    if args.replay:
        cfg["llm_mode"] = "REPLAY_EXISTING_LLM_OUTPUTS"
    summary = run_openrouter_stage(REPO_ROOT, _out(args), n=n, stage_name=stage,
                                   config=cfg, critic_all=not args.critic_sparse)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("llm_status") in ("ok", "not_run") else 2


def cmd_validate_queries(args: argparse.Namespace, argv: Sequence[str]) -> int:
    from .pilot41.validators import validate_query_record, v13_template_diversity
    from .repro import write_json

    out = _out(args)
    path = out / (args.input or "query_validated.jsonl")
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    reports = [validate_query_record(r) for r in rows]
    passed = sum(1 for r in reports if r["passed"])
    v13 = v13_template_diversity(rows)
    write_json(out / "validation_report.json", {
        "n": len(rows), "n_passed": passed,
        "pass_rate": round(passed / max(len(rows), 1), 4),
        "V13": v13,
    })
    print(f"[validate-queries] {passed}/{len(rows)} passed; "
          f"V13_passed={v13['passed']}")
    return 0


def cmd_select_pilot41(args: argparse.Namespace, argv: Sequence[str]) -> int:
    cfg = dict(DEFAULT_CONFIG)
    cfg["splits"] = {
        "train": args.train or 1000,
        "heldout": args.heldout or 250,
        "reserve": args.reserve or 250,
    }
    cfg["selected_total"] = sum(cfg["splits"].values())
    result = finalize_dataset(REPO_ROOT, _out(args), config=cfg,
                              prefer_llm=not args.deterministic_only,
                              cli_args=argv)
    print(json.dumps(result, indent=2))
    return 0


def cmd_run_v4(args: argparse.Namespace, argv: Sequence[str]) -> int:
    from .pilot41.pipeline import run_v4_selected
    from .repro import write_json

    out = _out(args)
    rows = []
    with (out / "canonical.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    result = run_v4_selected(rows, out, workers=0)
    write_json(out / "v4_report.json", result["summary"])
    print(json.dumps(result["summary"], indent=2))
    return 0


def cmd_audit_pilot41(args: argparse.Namespace, argv: Sequence[str]) -> int:
    from .pilot41.audit import run_pilot41_audit
    out = Path(args.output_dir or MODULE_ROOT / "reports" / "pilot4_vs_pilot41")
    summary = run_pilot41_audit(REPO_ROOT, out, cli_args=argv)
    print(f"[audit-pilot41] metrics={summary.get('n_metrics')} -> {out}")
    return 0


def cmd_freeze(args: argparse.Namespace, argv: Sequence[str]) -> int:
    # freeze is part of finalize; re-run finalize if needed
    return cmd_select_pilot41(args, argv)


def cmd_report(args: argparse.Namespace, argv: Sequence[str]) -> int:
    from .pilot41.report import build_report
    out = Path(args.output_dir or MODULE_ROOT / "reports")
    result = build_report(REPO_ROOT, out, cli_args=argv)
    print(f"[report-pilot41] -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="targeted-data (pilot41)")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("audit-pilot4-language")
    p.add_argument("--output-dir", default=None)
    p.set_defaults(fn=cmd_audit_pilot4_language)

    p = sub.add_parser("build-workflow-registry")
    p.add_argument("--output", default=None)
    p.add_argument("--output-dir", default=None)
    p.set_defaults(fn=cmd_build_workflow_registry)

    p = sub.add_parser("generate-semantic-pilot41")
    p.add_argument("--candidate-target", dest="candidates", type=int, default=None)
    p.add_argument("--candidates", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.set_defaults(fn=cmd_generate_semantic)

    p = sub.add_parser("select-render-shortlist")
    p.add_argument("--target", type=int, default=2000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.set_defaults(fn=cmd_select_shortlist)

    p = sub.add_parser("render-queries-openrouter")
    p.add_argument("--config", default=None)
    p.add_argument("--stage", choices=["smoke", "pilot", "full"], default="smoke")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--replay", action="store_true")
    p.add_argument("--critic-sparse", action="store_true")
    p.add_argument("--output-dir", default=None)
    p.set_defaults(fn=cmd_render_openrouter)

    p = sub.add_parser("validate-queries")
    p.add_argument("--input", default=None)
    p.add_argument("--validators", default="V9,V10,V11,V12,V13")
    p.add_argument("--output-dir", default=None)
    p.set_defaults(fn=cmd_validate_queries)

    p = sub.add_parser("select-pilot41")
    p.add_argument("--train", type=int, default=1000)
    p.add_argument("--heldout", type=int, default=250)
    p.add_argument("--reserve", type=int, default=250)
    p.add_argument("--deterministic-only", action="store_true")
    p.add_argument("--output-dir", default=None)
    p.set_defaults(fn=cmd_select_pilot41)

    p = sub.add_parser("run-v4-selected")
    p.add_argument("--workers", default="auto")
    p.add_argument("--output-dir", default=None)
    p.set_defaults(fn=cmd_run_v4)

    p = sub.add_parser("audit-pilot41")
    p.add_argument("--output-dir", default=None)
    p.set_defaults(fn=cmd_audit_pilot41)

    p = sub.add_parser("freeze-pilot41")
    p.add_argument("--train", type=int, default=1000)
    p.add_argument("--heldout", type=int, default=250)
    p.add_argument("--reserve", type=int, default=250)
    p.add_argument("--deterministic-only", action="store_true")
    p.add_argument("--output-dir", default=None)
    p.set_defaults(fn=cmd_freeze)

    p = sub.add_parser("implementation-report-pilot41")
    p.add_argument("--output-dir", default=None)
    p.set_defaults(fn=cmd_report)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    args = build_parser().parse_args(argv)
    return int(args.fn(args, argv) or 0)
