"""Pilot4.1 pipeline orchestration (deterministic first, then staged LLM)."""
from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..paraphrase import BudgetExceeded
from ..pilot4.validate import v4_minimal_path
from ..repro import sha256_file, stamp, write_csv, write_json, write_jsonl, write_text
from ..schemas import TaskRecord
from . import RUN_ID, SCHEMA_VERSION
from .cells import build_cells, cells_summary
from .generate import generate_semantic_pool, select_render_shortlist
from .graph_leak import analyze_graph_leak, audit_dataset
from .openrouter import OpenRouterSession, load_openrouter_config
from .query_render import query_template_fingerprint
from .select import select_records, split_records
from .validators import (validate_query_record, v12_llm_semantic_alignment,
                         v13_template_diversity)
from .workflows import export_registry

DEFAULT_CONFIG: Dict[str, Any] = {
    "run_id": RUN_ID,
    "mode": "PROFILE_SAFE",
    "seed": 20260731,
    "candidate_target": 10000,
    "shortlist_target": 2000,
    "selected_total": 1500,
    "splits": {"train": 1000, "heldout": 250, "reserve": 250},
    "n_core_cells": 60,
    "paired_variant_rate": 0.30,
    "llm_mode": "GENERATE_NEW_LLM_OUTPUTS",
    "openrouter_config": "configs/pilot4_1_openrouter.yaml",
    "smoke_n": 25,
    "pilot_n": 100,
}


def _export_grpo(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        out.append({
            "sample_id": r["task_id"],
            "question": r["question"],
            "tools": r["tools"],
            "gold_calls": r["gold_calls"],
            "gold_answer": r["gold_answer"],
            "observations": r.get("oracle_observations"),
            "num_calls": r["call_count"],
            "answer_type": type(r["gold_answer"]).__name__,
            "source": "pilot41",
            "stage": r.get("split") or "train",
            "generation_seed": r.get("generation_seed"),
            "provenance": {
                "workflow_id": r.get("workflow_id"),
                "query_mode": r.get("requested_query_mode"),
                "query_source": r.get("query_source"),
            },
        })
    return out


def run_deterministic_phase(repo_root: Path, out_dir: Path, *,
                            config: Optional[Dict[str, Any]] = None,
                            cli_args: Optional[Sequence[str]] = None
                            ) -> Dict[str, Any]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])

    wf_payload = export_registry(out_dir / "workflow_registry.json")
    cells = build_cells(train_n=cfg["splits"]["train"],
                        n_core_cells=int(cfg["n_core_cells"]))
    write_json(out_dir / "generation_cells.json", {
        "summary": cells_summary(cells),
        "cells": [c.as_dict() for c in cells],
    })

    print(f"[pilot41] generating ~{cfg['candidate_target']} semantic candidates…")
    candidates = generate_semantic_pool(
        cells, candidate_target=int(cfg["candidate_target"]), seed=seed)
    write_jsonl(out_dir / "semantic_candidates.jsonl", candidates)
    validated = [c for c in candidates
                 if (c.get("query_validation") or {}).get("passed")]
    write_jsonl(out_dir / "semantic_validated.jsonl", validated)
    shortlist = select_render_shortlist(
        validated, target=int(cfg["shortlist_target"]), seed=seed)
    write_jsonl(out_dir / "llm_render_shortlist.jsonl", shortlist)

    # deterministic queries already on shortlist → query_validated baseline
    write_jsonl(out_dir / "query_validated.jsonl", shortlist)
    leak = audit_dataset(shortlist, label="pilot41_shortlist_deterministic")
    write_json(out_dir / "query_realism_report.json", {
        k: v for k, v in leak.items() if k != "per_task"
    })
    write_json(out_dir / "semantic_coherence_report.json", {
        "mean_edge_rejection_rate": round(
            sum((c.get("semantic_edge_report") or {}).get("rejection_rate", 0)
                for c in validated) / max(len(validated), 1), 4),
        "n_validated": len(validated),
        "n_shortlist": len(shortlist),
    })

    return {
        "out_dir": str(out_dir),
        "n_candidates": len(candidates),
        "n_validated": len(validated),
        "n_shortlist": len(shortlist),
        "workflow_hash": wf_payload["registry_hash"],
        "stages_related_rate": leak["stages_related_phrase_rate"],
        "high_or_complete_graph_leak_rate": leak["high_or_complete_rate"],
        "provenance": stamp(repo_root, schema_version=SCHEMA_VERSION,
                            cli_args=cli_args, seeds={"seed": seed},
                            config=cfg),
    }


def _apply_writer_result(rec: Dict[str, Any], writer_row: Dict[str, Any],
                         critic_row: Optional[Dict[str, Any]] = None
                         ) -> Dict[str, Any]:
    parsed = writer_row["raw_response"]
    out = dict(rec)
    out["question"] = parsed["query"]
    out["query_source"] = "openrouter_writer"
    out["query_template_family"] = query_template_fingerprint(parsed["query"])
    out["llm_writer"] = {
        "request_id": writer_row["request_id"],
        "model": writer_row["model"],
        "provider": writer_row["provider"],
        "response_hash": writer_row["response_hash"],
        "cost_usd": writer_row["actual_cost_usd"],
        "raw": parsed,
    }
    if critic_row:
        out["llm_critic"] = critic_row["raw_response"]
        out["llm_critic_meta"] = {
            "request_id": critic_row["request_id"],
            "model": critic_row["model"],
            "cost_usd": critic_row["actual_cost_usd"],
        }
    out["query_validation"] = validate_query_record(out, run_v12=bool(critic_row))
    out["graph_leak"] = analyze_graph_leak(out)
    return out


def run_openrouter_stage(repo_root: Path, out_dir: Path, *,
                         n: int,
                         stage_name: str,
                         config: Optional[Dict[str, Any]] = None,
                         critic_all: bool = True,
                         ) -> Dict[str, Any]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    shortlist_path = out_dir / "llm_render_shortlist.jsonl"
    rows = []
    with shortlist_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    rows = rows[:n]
    # reuse passes from earlier stages to avoid re-spend
    prior_pass: Dict[str, Dict[str, Any]] = {}
    for prior_name in ("smoke", "pilot", "full"):
        if prior_name == stage_name:
            continue
        pp = out_dir / f"llm_rendered_{prior_name}.jsonl"
        if not pp.is_file():
            continue
        with pp.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                prior_pass[r["task_id"]] = r
    or_cfg_path = Path(cfg["openrouter_config"])
    if not or_cfg_path.is_absolute():
        or_cfg_path = (repo_root / "experiments" / "targeted_tool_data_factory"
                       / or_cfg_path)
        if not or_cfg_path.is_file():
            alt = Path(__file__).resolve().parents[3] / cfg["openrouter_config"]
            if alt.is_file():
                or_cfg_path = alt
    or_cfg = load_openrouter_config(or_cfg_path if or_cfg_path.is_file() else None)
    session = OpenRouterSession(
        cfg=or_cfg,
        log_path=out_dir / "openrouter_requests.jsonl",
        usage_path=out_dir / "openrouter_usage_summary.json",
        failures_path=out_dir / "openrouter_failures.jsonl",
        mode=str(cfg.get("llm_mode") or "GENERATE_NEW_LLM_OUTPUTS"),
    )
    if not session.available and session.mode != "REPLAY_EXISTING_LLM_OUTPUTS":
        return {"stage": stage_name, "llm_status": "not_run",
                "reason": "OPENROUTER_API_KEY missing", "n": 0}

    rendered: List[Dict[str, Any]] = []
    stats = Counter()
    try:
        for i, rec in enumerate(rows):
            tid = rec.get("task_id")
            if tid in prior_pass:
                rendered.append(prior_pass[tid])
                stats["pass"] += 1
                stats["reused_prior"] += 1
                continue
            try:
                contract = rec["semantic_contract"]
                writer_row = session.write_query(contract)
                candidate = _apply_writer_result(rec, writer_row)
                det = candidate["query_validation"]
                # sparse: critic on failures + ~10% audit sample
                need_critic = (critic_all or (not det["passed"])
                               or ((i % 10) == 0))
                critic_row = None
                if need_critic:
                    critic_row = session.critique(
                        contract, candidate["question"], det,
                        use_audit_model=(i % 10 == 0))
                    candidate = _apply_writer_result(rec, writer_row, critic_row)
                # rewrite loop — only on REWRITE or deterministic fail
                tries = 0
                while tries < 2:
                    verdict = ""
                    if critic_row:
                        verdict = str(
                            critic_row["raw_response"].get("verdict") or "").upper()
                    if verdict == "REJECT":
                        break
                    if candidate["query_validation"]["passed"] and verdict in (
                            "", "PASS"):
                        break
                    if verdict not in ("REWRITE", "") and candidate[
                            "query_validation"]["passed"]:
                        break
                    tries += 1
                    reasons = list(
                        (candidate["query_validation"].get("layers") or {})
                        .get("V11_QUERY_MODE_COMPLIANCE", {}).get("warnings") or [])
                    for layer in (candidate["query_validation"].get("layers")
                                  or {}).values():
                        reasons += list((layer or {}).get("warnings") or [])
                    if critic_row:
                        reasons += list(
                            critic_row["raw_response"].get("failure_reasons") or [])
                    writer_row = session.write_query(
                        contract, rewrite_of=candidate["question"],
                        failure_reasons=list(dict.fromkeys(reasons))[:16])
                    det2 = validate_query_record(
                        {**rec, "question": writer_row["raw_response"]["query"]},
                        run_v12=False)
                    critic_row = session.critique(
                        contract, writer_row["raw_response"]["query"], det2,
                        use_audit_model=True)
                    candidate = _apply_writer_result(rec, writer_row, critic_row)
                if candidate["query_validation"]["passed"] and (
                        not critic_row
                        or str(critic_row["raw_response"].get("verdict")
                               or "").upper() == "PASS"):
                    stats["pass"] += 1
                    rendered.append(candidate)
                else:
                    stats["reject"] += 1
                    stats["REJECT_QUERY_RENDER"] += 1
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001
                stats["error"] += 1
                stats["REJECT_QUERY_RENDER"] += 1
                with (out_dir / "openrouter_failures.jsonl").open(
                        "a", encoding="utf-8", newline="\n") as fh:
                    fh.write(json.dumps({
                        "stage": stage_name,
                        "task_id": rec.get("task_id"),
                        "error": str(exc),
                    }, ensure_ascii=False) + "\n")
    except BudgetExceeded as exc:
        stats["budget_stop"] = 1
        write_json(out_dir / f"openrouter_{stage_name}_partial.json", {
            "error": str(exc), "stats": dict(stats), "n_rendered": len(rendered),
        })

    out_path = out_dir / f"llm_rendered_{stage_name}.jsonl"
    write_jsonl(out_path, rendered)
    leak = audit_dataset(rendered, label=f"pilot41_{stage_name}") if rendered else {}
    summary = {
        "stage": stage_name,
        "n_input": len(rows),
        "n_passed": len(rendered),
        "pass_rate": round(len(rendered) / max(len(rows), 1), 4),
        "stats": dict(stats),
        "graph_leak": {k: v for k, v in leak.items() if k != "per_task"},
        "budget": session.budget.as_dict(),
        "llm_status": "ok" if rendered else "failed_or_empty",
    }
    write_json(out_dir / f"openrouter_{stage_name}_summary.json", summary)
    return summary


def finalize_dataset(repo_root: Path, out_dir: Path, *,
                     config: Optional[Dict[str, Any]] = None,
                     prefer_llm: bool = True,
                     cli_args: Optional[Sequence[str]] = None
                     ) -> Dict[str, Any]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    # merge best available rendered set
    pool: List[Dict[str, Any]] = []
    for name in ("llm_rendered_full.jsonl", "llm_rendered_pilot.jsonl",
                 "llm_rendered_smoke.jsonl", "query_validated.jsonl"):
        p = out_dir / name
        if p.is_file():
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        pool.append(json.loads(line))
            if prefer_llm and name.startswith("llm_rendered") and len(pool) >= 500:
                break
    # dedupe by task_id preferring LLM
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in pool:
        prev = by_id.get(r["task_id"])
        if prev is None or (r.get("query_source") == "openrouter_writer"
                            and prev.get("query_source") != "openrouter_writer"):
            by_id[r["task_id"]] = r
    # also include remaining validated semantics for fill
    sem_path = out_dir / "semantic_validated.jsonl"
    if sem_path.is_file() and len(by_id) < int(cfg["selected_total"]) * 2:
        with sem_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                by_id.setdefault(r["task_id"], r)
    candidates = list(by_id.values())
    cells = [type("C", (), c) for c in
             (json.loads((out_dir / "generation_cells.json").read_text(
                 encoding="utf-8")).get("cells") or [])]
    # rebuild Cell41-like objects
    from .cells import Cell41
    cell_objs = [
        Cell41(**{k: c[k] for k in Cell41.__dataclass_fields__ if k in c})
        for c in json.loads(
            (out_dir / "generation_cells.json").read_text(encoding="utf-8")
        )["cells"]
    ]

    selected, sel_report = select_records(
        candidates, cell_objs, n_selected=int(cfg["selected_total"]),
        train_n=int(cfg["splits"]["train"]), seed=int(cfg["seed"]))
    write_json(out_dir / "selection_report.json", sel_report)
    write_jsonl(out_dir / "selected.jsonl", selected)

    splits, split_manifest = split_records(
        selected, cfg["splits"], seed=int(cfg["seed"]))
    write_json(out_dir / "split_manifest.json", split_manifest)
    for name, rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", _export_grpo(rows))
    canonical = []
    for name, rows in splits.items():
        for r in rows:
            r = dict(r)
            r["split"] = name
            canonical.append(r)
    write_jsonl(out_dir / "canonical.jsonl", canonical)
    write_jsonl(out_dir / "nestful_compat.jsonl", [
        {"id": r["task_id"], "input": r["question"],
         "tools": r["tools"], "output": r["gold_calls"],
         "gold_answer": r["gold_answer"]} for r in canonical
    ])

    # V4 on selected
    v4_rows = run_v4_selected(canonical, out_dir)
    write_json(out_dir / "v4_report.json", v4_rows["summary"])

    # human audit sample
    audit_csv = build_human_audit_sample(canonical)
    write_csv(out_dir / "human_audit_sample.csv", audit_csv)

    # hashes
    files = {}
    for p in sorted(out_dir.glob("*")):
        if p.is_file() and p.name != "MANIFEST.sha256.json":
            files[p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    write_json(out_dir / "MANIFEST.sha256.json", {"files": files})
    ordered = [r["task_id"] for r in canonical]
    usage = {}
    if (out_dir / "openrouter_usage_summary.json").is_file():
        usage = json.loads(
            (out_dir / "openrouter_usage_summary.json").read_text(encoding="utf-8"))
    llm_n = sum(1 for r in selected
                if r.get("query_source") == "openrouter_writer")
    llm_status = "ok" if llm_n else (
        "not_run" if not usage else "failed_or_empty")
    freeze = {
        "schema_version": SCHEMA_VERSION,
        "run_id": cfg["run_id"],
        "mode": cfg["mode"],
        "frozen": True,
        "llm_status": llm_status,
        "llm_mode_default_replay": "REPLAY_EXISTING_LLM_OUTPUTS",
        "counts": {
            "candidates": _count_lines(out_dir / "semantic_candidates.jsonl"),
            "validated": _count_lines(out_dir / "semantic_validated.jsonl"),
            "shortlist": _count_lines(out_dir / "llm_render_shortlist.jsonl"),
            "selected": len(selected),
            "selected_llm_queries": llm_n,
            **{k: len(v) for k, v in splits.items()},
        },
        "ordered_sample_ids_hash": short_hash_ids(ordered),
        "split_manifest": split_manifest,
        "selection_all_hard_constraints_met": sel_report.get(
            "all_hard_constraints_met"),
        "v4_shortcut_rate": v4_rows["summary"].get("shortcut_rate"),
        "openrouter_usage": usage,
        "provenance": stamp(repo_root, schema_version=SCHEMA_VERSION,
                            cli_args=cli_args, seeds={"seed": int(cfg["seed"])},
                            config=cfg),
    }
    write_json(out_dir / "freeze_manifest.json", freeze)
    return {"out_dir": str(out_dir), "counts": freeze["counts"],
            "leak_free": split_manifest.get("leak_free"),
            "selection": {k: sel_report[k] for k in (
                "n_selected", "all_hard_constraints_met",
                "n_singleton_core_cells")}}


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.open("r", encoding="utf-8") if line.strip())


def short_hash_ids(ids: Sequence[str]) -> str:
    from ..util import short_hash
    return short_hash(list(ids))


def run_v4_selected(rows: Sequence[Dict[str, Any]], out_dir: Path,
                    *, workers: int = 0) -> Dict[str, Any]:
    """Run V4 minimal-path on selected rows; cache by semantic_program_id."""
    cache_path = out_dir / "v4_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    results = []
    shortcuts = 0
    unresolved = 0
    for rec in rows:
        key = rec.get("semantic_program_id") or rec["task_id"]
        if key in cache:
            row = cache[key]
        else:
            row = _v4_one(rec)
            cache[key] = row
        row = dict(row)
        row["task_id"] = rec["task_id"]
        results.append(row)
        if row.get("has_shortcut"):
            shortcuts += 1
        if not row.get("search_complete"):
            unresolved += 1
    write_json(cache_path, cache)
    write_jsonl(out_dir / "v4_per_task.jsonl", results)
    n = len(results) or 1
    summary = {
        "n": len(results),
        "shortcut_rate": round(shortcuts / n, 4),
        "unresolved_rate": round(unresolved / n, 4),
        "n_shortcuts": shortcuts,
        "n_unresolved": unresolved,
    }
    return {"summary": summary, "rows": results}


def _v4_one(rec: Dict[str, Any]) -> Dict[str, Any]:
    """V4 via pilot4 bounded minimal-path search + constant-equality probe."""
    gold_n = int(rec.get("call_count") or len(rec.get("gold_calls") or []))
    consts = list(rec.get("constants") or [])
    ans = rec.get("gold_answer")
    const_shortcut = ans in consts and gold_n >= 2
    if not isinstance(ans, (int, float)) or isinstance(ans, bool):
        return {
            "gold_call_count": gold_n,
            "minimal_valid_call_count": gold_n,
            "has_shortcut": False,
            "alternative_minimal_paths": 0,
            "search_complete": True,
            "search_budget_exhausted": False,
            "method": "skipped_non_numeric",
            "safe_for_core_train": True,
        }
    try:
        errs, meta = v4_minimal_path(rec, max_depth=2, max_evals=4000)
        searched = bool(meta.get("searched"))
        exhausted = bool(meta.get("exhausted"))
        shortcut_depth = meta.get("shortcut_depth")
        path_shortcut = bool(errs) or shortcut_depth is not None
        has_shortcut = const_shortcut or path_shortcut
        complete = (searched and not exhausted) or const_shortcut
        return {
            "gold_call_count": gold_n,
            "minimal_valid_call_count": (
                int(shortcut_depth) if shortcut_depth is not None
                else (1 if const_shortcut else gold_n)),
            "has_shortcut": has_shortcut,
            "alternative_minimal_paths": int(has_shortcut),
            "search_complete": complete,
            "search_budget_exhausted": exhausted,
            "method": "pilot4_v4_minimal_path",
            "meta": meta,
            "errors": list(errs or []),
            "safe_for_core_train": complete and not has_shortcut,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "gold_call_count": gold_n,
            "minimal_valid_call_count": 1 if const_shortcut else gold_n,
            "has_shortcut": const_shortcut,
            "alternative_minimal_paths": int(const_shortcut),
            "search_complete": False,
            "search_budget_exhausted": True,
            "method": "constant_equality_probe_fallback",
            "error": str(exc),
            "safe_for_core_train": False,
        }


def build_human_audit_sample(rows: Sequence[Dict[str, Any]],
                             n: int = 150) -> List[Dict[str, Any]]:
    by_pattern: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_pattern.setdefault(r.get("pattern_family") or "UNK", []).append(r)
    sample = []
    # 10 per pattern family
    for pattern, items in sorted(by_pattern.items()):
        for rec in items[:10]:
            sample.append(_audit_row(rec))
            if len(sample) >= n:
                return sample[:n]
    # pad
    for rec in rows:
        if len(sample) >= n:
            break
        if all(s["task_id"] != rec["task_id"] for s in sample):
            sample.append(_audit_row(rec))
    return sample[:n]


def _audit_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": rec["task_id"],
        "split": rec.get("split"),
        "pattern_family": rec.get("pattern_family"),
        "query_mode": rec.get("requested_query_mode"),
        "surface_track": rec.get("surface_track"),
        "difficulty_band": rec.get("difficulty_band"),
        "workflow_id": rec.get("workflow_id"),
        "question": rec.get("question"),
        "Does the query sound like a plausible user request?": "",
        "Does the query disclose the computation plan?": "",
        "Are all facts necessary?": "",
        "Is the workflow semantically coherent?": "",
        "Could the question be answered without the gold program?": "",
        "Are units and entities natural?": "",
        "Is the query ambiguous?": "",
    }
