"""CLI for the Pilot4.2 workflow-grounded pipeline."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional

from .pilot42 import RUN_ID
from .pilot42.pipeline import (finalize, gate_phase, generate_phase,
                               render_openrouter_stage, run_all)
from .pilot42.primitives_v2 import export_registry as export_primitives
from .pilot42.workflows_v2 import export_registry as export_workflows

MODULE_ROOT = Path(__file__).resolve().parents[2]
PILOT42_COMMANDS = [
    "audit-pilot41-root-cause",
    "build-workflow-registry-v2",
    "validate-primitive-registry-v2",
    "generate-pilot42-semantic",
    "validate-pilot42-semantic",
    "select-query-render-shortlist",
    "gate-pilot42",
    "render-pilot42-openrouter",
    "validate-pilot42-queries",
    "run-pilot42-v4",
    "select-pilot42",
    "build-nested-subsets",
    "audit-pilot42",
    "compare-pilot41-pilot42",
    "prepare-human-audit-pilot42",
    "freeze-pilot42",
    "run-pilot42",
]


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="targeted-data pilot42")
    ap.add_argument("command", choices=PILOT42_COMMANDS)
    ap.add_argument("--output-dir")
    ap.add_argument("--candidate-target", type=int, default=20_000)
    ap.add_argument("--selected-target", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--new-run-suffix")
    ap.add_argument("--stage", choices=["smoke", "pilot", "full"], default="smoke")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--config", default=None)
    return ap


def _out(args: argparse.Namespace, *, create: bool = True) -> Path:
    path = Path(args.output_dir or MODULE_ROOT / "outputs" / RUN_ID)
    if args.new_run_suffix:
        path = path.with_name(f"{path.name}_{args.new_run_suffix}")
    if create:
        if path.exists() and any(path.iterdir()) and not args.resume:
            raise FileExistsError(
                f"refusing to overwrite {path}; use --resume or --new-run-suffix")
        path.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit-pilot41-root-cause":
        source = (MODULE_ROOT / "reports" / "pilot41_root_cause"
                  / "PILOT41_ROOT_CAUSE_AUDIT.md")
        target_dir = Path(args.output_dir or MODULE_ROOT / "reports" / "pilot41_root_cause")
        target_dir.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            shutil.copy2(source, target_dir / source.name)
            js = source.with_suffix(".json")
            if js.is_file():
                shutil.copy2(js, target_dir / js.name)
        print(f"[audit-pilot41-root-cause] -> {target_dir}")
        return 0
    if args.command == "run-pilot42":
        result = run_all(MODULE_ROOT, Path(args.output_dir) if args.output_dir else None,
                         candidate_target=args.candidate_target,
                         selected_target=args.selected_target, seed=args.seed,
                         resume=args.resume, new_run_suffix=args.new_run_suffix)
        print(result)
        return 0

    out = _out(args, create=args.command not in (
        "compare-pilot41-pilot42", "audit-pilot42"))
    if args.command == "build-workflow-registry-v2":
        result = export_workflows(out / "workflow_registry.json")
    elif args.command == "validate-primitive-registry-v2":
        result = export_primitives(out / "primitive_registry.json")
    elif args.command in ("generate-pilot42-semantic", "validate-pilot42-semantic"):
        result = generate_phase(out, args.candidate_target, args.seed)
    elif args.command == "select-query-render-shortlist":
        from .repro import write_jsonl
        rows = []
        with (out / "semantic_validated.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        target = args.n or 6000
        write_jsonl(out / "query_render_shortlist.jsonl", rows[:target])
        result = {"shortlist": min(target, len(rows))}
    elif args.command in ("gate-pilot42", "run-pilot42-v4", "validate-pilot42-queries"):
        result = gate_phase(out)
    elif args.command == "render-pilot42-openrouter":
        cfg = Path(args.config) if args.config else None
        result = render_openrouter_stage(out, stage=args.stage, n=args.n,
                                         config_path=cfg)
    elif args.command in ("select-pilot42", "freeze-pilot42",
                          "build-nested-subsets", "prepare-human-audit-pilot42"):
        result = finalize(out, args.selected_target, args.seed)
    elif args.command in ("audit-pilot42", "compare-pilot41-pilot42"):
        from .pilot42.audit_compare import write_comparison
        target = Path(args.output_dir or out)
        result = {"path": str(write_comparison(
            target / "PILOT41_VS_PILOT42_AUDIT.md",
            {"note": "see freeze_manifest and data quality report"}))}
    else:  # pragma: no cover
        raise ValueError(args.command)
    print(result if not isinstance(result, dict)
          else json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
