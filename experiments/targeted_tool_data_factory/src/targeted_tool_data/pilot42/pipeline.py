"""Pilot4.2 phase orchestration and guarded artifact writing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..repro import sha256_file, stamp, write_csv, write_json, write_jsonl
from . import RUN_ID, SCHEMA_VERSION
from .audit_compare import write_comparison
from .cells import build_cells, cells_summary
from .distractors import attach_distractors
from .generate import generate_semantic_pool
from .human_audit import write_human_audit
from .primitives_v2 import export_registry as export_primitives
from .query_validators import validate_query
from .report import write_reports
from .select import select_records
from .split import split_records
from .subsets import assert_nested, nested_stratified_subsets
from .v4_gate import evaluate_v4
from .validate_semantic import validate_record
from .workflows_v2 import export_registry as export_workflows

DEFAULT_SEED = 20260731


def render_openrouter_stage(out: Path, *, stage: str = "smoke", n: int | None = None,
                            config_path: Path | None = None) -> Dict[str, Any]:
    """Staged OpenRouter writer + 100% critic on hard-gated shortlist rows."""
    from .openrouter import OpenRouterSession, load_openrouter_config
    from ..paraphrase import BudgetExceeded

    n = n or {"smoke": 30, "pilot": 150, "full": 6000}.get(stage, 30)
    cfg_path = config_path or (
        Path(__file__).resolve().parents[3] / "configs" / "pilot4_2_openrouter.yaml")
    or_cfg = load_openrouter_config(cfg_path if cfg_path.is_file() else None)
    session = OpenRouterSession(
        cfg=or_cfg,
        log_path=out / "openrouter_requests.jsonl",
        usage_path=out / "openrouter_usage_summary.json",
        failures_path=out / "openrouter_failures.jsonl",
        mode="GENERATE_NEW_LLM_OUTPUTS",
    )
    if not session.available:
        write_json(out / f"openrouter_{stage}_summary.json", {
            "stage": stage, "llm_status": "not_run",
            "reason": "OPENROUTER_API_KEY missing"})
        return {"stage": stage, "llm_status": "not_run", "n": 0}

    pool = _read_jsonl(out / "hard_gated_pool.jsonl") or _read_jsonl(
        out / "query_render_shortlist.jsonl")
    rows = pool[:n]
    rendered, reject, errors = [], 0, 0
    try:
        for rec in rows:
            try:
                contract = rec.get("query_contract") or {}
                writer_contract = {
                    **contract,
                    "constants": [f.get("value") for f in (contract.get("facts") or [])],
                    "units": [f.get("unit") for f in (contract.get("facts") or [])
                              if f.get("unit")],
                    "entities": [contract.get("entity")],
                    "target_variable": {"role": contract.get("target_role")},
                    "task_id": rec.get("task_id"),
                }
                writer = session.write_query(writer_contract)
                query = writer["raw_response"]["query"]
                cand = dict(rec)
                cand["question"] = query
                cand["query"] = query
                cand["query_source"] = "openrouter_writer"
                cand["query_validation"] = validate_query(cand)
                critics = session.critique_twice_if_needed(
                    cand, writer_contract, query, cand["query_validation"])
                cand["llm_critic"] = critics[0]["raw_response"]
                cand["llm_critics"] = [c["raw_response"] for c in critics]
                v0 = str(critics[0]["raw_response"].get("verdict") or "").upper()
                v1 = (str(critics[1]["raw_response"].get("verdict") or "").upper()
                      if len(critics) > 1 else "PASS")
                if (cand["query_validation"]["passed"] and v0 == "PASS"
                        and v1 in ("PASS", "")):
                    rendered.append(cand)
                else:
                    reject += 1
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001
                errors += 1
                reject += 1
                with (out / "openrouter_failures.jsonl").open(
                        "a", encoding="utf-8", newline="\n") as fh:
                    fh.write(json.dumps({
                        "stage": stage, "task_id": rec.get("task_id"),
                        "error": str(exc)}) + "\n")
    except BudgetExceeded as exc:
        write_json(out / f"openrouter_{stage}_partial.json", {"error": str(exc)})

    write_jsonl(out / f"llm_rendered_{stage}.jsonl", rendered)
    # merge into llm_rendered.jsonl (union by task_id)
    merged: Dict[str, Dict[str, Any]] = {}
    for name in ("smoke", "pilot", "full"):
        for r in _read_jsonl(out / f"llm_rendered_{name}.jsonl"):
            merged[r["task_id"]] = r
    write_jsonl(out / "llm_rendered.jsonl", list(merged.values()))
    # prefer LLM in hard_gated pool
    by_id = {r["task_id"]: r for r in _read_jsonl(out / "hard_gated_pool.jsonl")}
    for r in merged.values():
        by_id[r["task_id"]] = r
    write_jsonl(out / "hard_gated_pool.jsonl", list(by_id.values()))
    summary = {
        "stage": stage, "n_input": len(rows), "n_passed": len(rendered),
        "n_reject": reject, "n_errors": errors,
        "pass_rate": round(len(rendered) / max(len(rows), 1), 4),
        "budget": session.budget.as_dict(),
        "llm_status": "ok" if rendered else "failed_or_empty",
        "critic_coverage": 1.0 if rendered else 0.0,
    }
    write_json(out / f"openrouter_{stage}_summary.json", summary)
    write_json(out / "openrouter_model_snapshot.json", {
        "writer_model": or_cfg.get("writer_model"),
        "critic_model": or_cfg.get("critic_model"),
        "audit_model": or_cfg.get("audit_model"),
    })
    return summary


def prepare_output(factory_root: Path, output_dir: Path | None = None, *,
                   resume: bool = False, new_run_suffix: str | None = None) -> Path:
    out = Path(output_dir or factory_root / "outputs" / RUN_ID)
    if new_run_suffix:
        out = out.with_name(f"{out.name}_{new_run_suffix}")
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError(
            f"refusing to overwrite existing run {out}; use --resume or --new-run-suffix")
    out.mkdir(parents=True, exist_ok=True)
    return out


def generate_phase(out: Path, candidate_target: int = 20_000,
                   seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    export_workflows(out / "workflow_registry.json")
    export_primitives(out / "primitive_registry.json")
    cells = build_cells()
    write_json(out / "generation_cells.json", {
        "schema_version": SCHEMA_VERSION, "cells": [c.as_dict() for c in cells],
        "summary": cells_summary(cells)})
    rows = generate_semantic_pool(cells, candidate_target, seed)
    write_jsonl(out / "semantic_candidates.jsonl", rows)
    valid, rejected = [], []
    for row in rows:
        report = validate_record(row)
        row["semantic_validation"] = report
        (valid if report["passed"] else rejected).append(row)
    write_jsonl(out / "semantic_validated.jsonl", valid)
    write_jsonl(out / "semantic_rejected.jsonl", rejected)
    # shortlist for query/LLM path from hard-valid pool
    shortlist = valid[: min(6000, len(valid))]
    write_jsonl(out / "query_render_shortlist.jsonl", shortlist)
    write_json(out / "generation_summary.json", {
        "generated": len(rows), "hard_validated": len(valid),
        "rejected": len(rejected), "shortlist": len(shortlist)})
    return {"generated": len(rows), "hard_validated": len(valid),
            "rejected": len(rejected), "shortlist": len(shortlist)}


def gate_phase(out: Path) -> Dict[str, Any]:
    rows = _read_jsonl(out / "semantic_validated.jsonl")
    gated, rejected = [], []
    reasons: Dict[str, int] = {}
    for row in rows:
        row = attach_distractors(row)
        row["query_validation"] = validate_query(row)
        row["v4_gate"] = evaluate_v4(row)
        ok_q = row["query_validation"]["passed"]
        ok_v4 = row["v4_gate"]["passed"]
        if ok_q and ok_v4:
            gated.append(row)
        else:
            rejected.append(row)
            if not ok_q:
                reasons["query_fail"] = reasons.get("query_fail", 0) + 1
            if not ok_v4:
                key = ("v4_shortcut" if row["v4_gate"].get("has_shortcut")
                       else "v4_unresolved")
                reasons[key] = reasons.get(key, 0) + 1
    write_jsonl(out / "hard_gated_pool.jsonl", gated)
    write_jsonl(out / "hard_gate_rejected.jsonl", rejected)
    write_jsonl(out / "query_validated.jsonl", gated)
    summary = {"hard_gated": len(gated), "rejected": len(rejected),
               "reject_reasons": reasons}
    write_json(out / "validation_report.json", summary)
    return summary


def _export_nestful(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{
        "id": r["task_id"], "input": r["question"], "tools": r.get("tools") or [],
        "output": r.get("gold_calls") or [], "gold_answer": r.get("gold_answer"),
    } for r in rows]


def _export_grpo(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        out.append({
            "task_id": r["task_id"], "question": r["question"],
            "tools": r.get("tools") or [], "gold_calls": r.get("gold_calls") or [],
            "gold_answer": r.get("gold_answer"),
            "workflow_id": r.get("workflow_id"),
            "call_count": r.get("call_count"),
            "requested_query_mode": r.get("requested_query_mode"),
            "cell_tier": r.get("cell_tier"),
            "was_generated_from_workflow": r.get("was_generated_from_workflow"),
        })
    return out


def finalize(out: Path, selected_target: int = 4000,
             seed: int = DEFAULT_SEED,
             repo_root: Optional[Path] = None) -> Dict[str, Any]:
    rows = _read_jsonl(out / "hard_gated_pool.jsonl")
    if any(not (r.get("semantic_validation", {}).get("passed")
                and r.get("query_validation", {}).get("passed")
                and r.get("v4_gate", {}).get("passed")) for r in rows):
        raise RuntimeError("selection input contains a row that failed a hard gate")
    selected, selection = select_records(rows, selected_target, seed)
    write_jsonl(out / "selected.jsonl", selected)
    write_json(out / "selection_report.json", selection)

    # Nested subsets from train portion after split — first split, then nest train
    splits, leakage = split_records(
        selected, {"train_master": 3000, "heldout": 500, "reserve": 500}, seed)
    # tolerate key name train vs train_master
    if "train_master" not in splits and "train" in splits:
        splits["train_master"] = splits.pop("train")
    train = splits.get("train_master") or []
    heldout = splits.get("heldout") or []
    reserve = splits.get("reserve") or []

    write_jsonl(out / "train_master_3000.jsonl", _export_grpo(train))
    write_jsonl(out / "heldout_500.jsonl", _export_grpo(heldout))
    write_jsonl(out / "reserve_500.jsonl", _export_grpo(reserve))

    subsets = nested_stratified_subsets(train, sizes=(500, 1000, 2000, 3000), seed=seed)
    assert assert_nested(subsets), "nested subset inclusion failed"
    write_jsonl(out / "train_core_500.jsonl", _export_grpo(subsets[500]))
    write_jsonl(out / "train_core_1000.jsonl", _export_grpo(subsets[1000]))
    write_jsonl(out / "train_core_2000.jsonl", _export_grpo(subsets[2000]))
    # master equals full train
    write_jsonl(out / "train_master_3000.jsonl", _export_grpo(train))

    canonical = []
    for name, part in (("train_master", train), ("heldout", heldout),
                       ("reserve", reserve)):
        for r in part:
            rr = dict(r)
            rr["split"] = name
            canonical.append(rr)
    write_jsonl(out / "canonical.jsonl", canonical)
    write_jsonl(out / "nestful_compat_train_master.jsonl", _export_nestful(train))
    write_jsonl(out / "nestful_compat_train_core_2000.jsonl",
                _export_nestful(subsets[2000]))
    write_jsonl(out / "nestful_compat_train_core_1000.jsonl",
                _export_nestful(subsets[1000]))
    write_jsonl(out / "nestful_compat_train_core_500.jsonl",
                _export_nestful(subsets[500]))
    write_jsonl(out / "nestful_compat_heldout.jsonl", _export_nestful(heldout))
    write_jsonl(out / "nestful_compat_reserve.jsonl", _export_nestful(reserve))

    write_json(out / "split_manifest.json", leakage)
    write_human_audit(selected, out, sample_size=300, seed=seed)

    # per-task ledger
    ledger = []
    for r in selected:
        ledger.append({
            "sample_id": r["task_id"], "workflow_id": r.get("workflow_id"),
            "all_hard_gates_passed": True,
            "workflow_program_alignment": (r.get("semantic_validation") or {})
            .get("layers", {}).get("V_WORKFLOW_PROGRAM_QUERY_ALIGNMENT"),
            "v4_minimal_path": r.get("v4_gate"),
            "query_fact_validation": r.get("query_validation"),
            "llm_critic": r.get("llm_critic"),
            "warnings": [],
        })
    write_jsonl(out / "per_task_validation_ledger.jsonl", ledger)
    write_csv(out / "per_task_validation_ledger.csv", [
        {"sample_id": x["sample_id"], "workflow_id": x["workflow_id"],
         "all_hard_gates_passed": x["all_hard_gates_passed"]}
        for x in ledger])

    v4_shortcuts = sum(1 for r in selected if (r.get("v4_gate") or {}).get("has_shortcut"))
    v4_unresolved = sum(1 for r in selected if (r.get("v4_gate") or {}).get("unresolved"))
    write_json(out / "v4_report.json", {
        "n": len(selected), "shortcut_rate": 0.0 if not selected else
        round(v4_shortcuts / len(selected), 4),
        "unresolved_rate": 0.0 if not selected else
        round(v4_unresolved / len(selected), 4),
        "n_shortcuts": v4_shortcuts, "n_unresolved": v4_unresolved,
    })

    metrics = {
        "semantic_candidates": _count(out / "semantic_candidates.jsonl"),
        "hard_validated": _count(out / "semantic_validated.jsonl"),
        "v4_safe": len(rows),
        "selected": len(selected),
        "train_master": len(train), "heldout": len(heldout), "reserve": len(reserve),
        "selection_all_hard_constraints_met":
            selection["selection_all_hard_constraints_met"],
        "v4_shortcuts_selected": v4_shortcuts,
        "v4_unresolved_selected": v4_unresolved,
    }
    write_reports(out, metrics, selected)
    write_comparison(out / "PILOT41_VS_PILOT42_AUDIT.md",
                     {"selected": len(selected), **metrics})
    write_csv(out / "PILOT41_VS_PILOT42_METRICS.csv", [
        {"metric": k, "value": v} for k, v in metrics.items()
    ])

    if len(selected) < selected_target:
        write_json(out / "deficit_report.json", {
            "schema_version": SCHEMA_VERSION,
            "requested": selected_target, "available": len(selected),
            "deficit": selected_target - len(selected),
            "rules_weakened": False,
            "eligible_pool": selection.get("eligible_pool"),
            "note": "Hard validators were not weakened; partial dataset frozen.",
        })

    automated = (
        selection["selection_all_hard_constraints_met"]
        and leakage.get("leak_free", False)
        and v4_shortcuts == 0 and v4_unresolved == 0
        and len(selected) == selected_target
        and len(train) == 3000 and len(heldout) == 500 and len(reserve) == 500
    )
    files = {}
    for p in sorted(out.glob("*")):
        if p.is_file() and p.name != "MANIFEST.sha256.json":
            files[p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    write_json(out / "MANIFEST.sha256.json", {"files": files})
    provenance = stamp(repo_root or out.parents[2], schema_version=SCHEMA_VERSION,
                       seeds={"seed": seed}, config={"selected_target": selected_target}) \
        if repo_root or True else {}
    try:
        provenance = stamp(
            Path(__file__).resolve().parents[4],
            schema_version=SCHEMA_VERSION,
            seeds={"seed": seed},
            config={"selected_target": selected_target})
    except Exception:  # noqa: BLE001
        provenance = {"schema_version": SCHEMA_VERSION}

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "frozen": True,
        "AUTOMATED_GATES_PASSED": automated,
        "TRAINING_READY": False,
        "HUMAN_REVIEW_PENDING": True,
        "LLM_VALIDATED": False,
        "NOT_TESTED_BY_MODEL_PROBE": True,
        "NOT_TESTED_BY_TRAINING": True,
        "NOT_TESTED_BY_NESTFUL": True,
        "selection_all_hard_constraints_met":
            selection["selection_all_hard_constraints_met"],
        "counts": metrics,
        "split_manifest": leakage,
        "provenance": provenance,
    }
    write_json(out / "freeze_manifest.json", manifest)
    return manifest


def run_all(factory_root: Path, output_dir: Path | None = None, *,
            candidate_target: int = 20_000, selected_target: int = 4000,
            seed: int = DEFAULT_SEED, resume: bool = False,
            new_run_suffix: str | None = None) -> Dict[str, Any]:
    out = prepare_output(factory_root, output_dir, resume=resume,
                         new_run_suffix=new_run_suffix)
    generation = generate_phase(out, candidate_target, seed)
    gates = gate_phase(out)
    manifest = finalize(out, selected_target, seed, repo_root=factory_root)
    return {"output_dir": str(out), "generation": generation,
            "gates": gates, "manifest": manifest}


def _count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
