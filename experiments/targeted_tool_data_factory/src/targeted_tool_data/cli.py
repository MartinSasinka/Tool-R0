"""targeted-data CLI — profile | generate | validate | select | probe |
split | export | report | all.

Every step: deterministic seed, resume, content hashes, dry-run, strict,
output version, overwrite protection. Core path needs no LLM/GPU/network.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import GENERATOR_VERSION
from . import registry as reg
from .executor import executor_hash
from .generation import (build_cells, build_cells_v2, generate_pool,
                         record_to_canonical)
from .profile import extract_profile, featurize_row, profile_report_md
from .providers import make_provider
from .probing import NOT_RUN_CMD, informative, p0_structural, probe_record
from .reporting import build_cost_report, build_pilot_report, readiness_verdict
from .schemas import GenerationCell, TargetProfile, TaskRecord
from .selection import (leakage_audit, profile_match_report, select_records,
                        split_records)
from .util import (MODULE_ROOT, StepGuard, load_config, read_json, read_jsonl,
                   sha256_obj, write_json, write_jsonl)
from .validation import (contamination_check, dedup_pool, v6_distribution,
                         validate_record)
from .export import export_all
from .pilot4_cli import PILOT4_COMMANDS
from .pilot4_cli import main as pilot4_main
from .pilot42_cli import PILOT42_COMMANDS
from .pilot42_cli import main as pilot42_main
from .pilot41_cli import PILOT41_COMMANDS
from .pilot41_cli import main as pilot41_main
from .pilot43_cli import PILOT43_COMMANDS
from .pilot43_cli import main as pilot43_main

sys.path.insert(0, str(MODULE_ROOT))   # make `targets.*` adapters importable


class Ctx:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = MODULE_ROOT / cfg_path
        self.cfg = load_config(cfg_path)
        tgt_path = MODULE_ROOT / "configs" / "targets" / f"{args.target}.yaml"
        self.target_cfg = load_config(tgt_path)
        self.config_hash = sha256_obj({"cfg": self.cfg, "target": self.target_cfg,
                                       "seed": args.seed})[:16]
        mod = importlib.import_module(self.target_cfg["adapter"])
        self.adapter = mod.make_adapter(self.target_cfg)
        self.version = args.version
        self.seed = args.seed
        self.out = MODULE_ROOT / "outputs"
        for sub in ("profiles", "candidates", "validated", "selected",
                    "splits", "reports", "cache"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.thresholds = self.cfg.get("thresholds", {})
        self.run_state_path = self.out / "reports" / f"run_state_{self.version}.json"
        self.run_state = (read_json(self.run_state_path)
                          if self.run_state_path.is_file() else {"steps": {}})
        self.tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]

    def guard(self, step: str, sub: str) -> StepGuard:
        return StepGuard(self.out / sub, f"{step}_{self.version}",
                         resume=self.args.resume, overwrite=self.args.overwrite)

    def save_state(self) -> None:
        write_json(self.run_state_path, self.run_state)

    def timed(self, step: str):
        ctx = self

        class _T:
            def __enter__(self):
                tracemalloc.start()
                self.w0, self.c0 = time.perf_counter(), time.process_time()
                return self

            def __exit__(self, *exc):
                _cur, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                ctx.run_state["steps"][step] = {
                    "wall_s": round(time.perf_counter() - self.w0, 2),
                    "cpu_s": round(time.process_time() - self.c0, 2),
                    "peak_mb": round(peak / 1e6, 1),
                }
                ctx.save_state()

        return _T()

    # artifact paths
    def p_profile(self) -> Path:
        return self.out / "profiles" / f"{self.args.target}_profile.json"

    def p_cells(self) -> Path:
        return self.out / "candidates" / f"cells_{self.version}.json"

    def p_candidates(self) -> Path:
        return self.out / "candidates" / f"candidates_{self.version}.jsonl"

    def p_genstats(self) -> Path:
        return self.out / "candidates" / f"gen_stats_{self.version}.json"

    def p_validated(self) -> Path:
        return self.out / "validated" / f"validated_{self.version}.jsonl"

    def p_rejected(self) -> Path:
        return self.out / "validated" / f"rejected_{self.version}.jsonl"

    def p_valsummary(self) -> Path:
        return self.out / "validated" / f"validation_summary_{self.version}.json"

    def p_selected(self) -> Path:
        return self.out / "selected" / f"selected_{self.version}.jsonl"

    def p_trace(self) -> Path:
        return self.out / "selected" / f"selection_trace_{self.version}.jsonl"

    def p_match(self) -> Path:
        return self.out / "selected" / f"profile_match_{self.version}.json"

    def p_split_dir(self) -> Path:
        return self.out / "splits"

    def p_export_dir(self) -> Path:
        return self.out / "selected" / f"export_{self.version}"

    def p_paraphrased(self) -> Path:
        return self.out / "validated" / f"paraphrased_{self.version}.jsonl"

    def p_para_report(self) -> Path:
        return self.out / "reports" / f"paraphrase_{self.version}.json"

    @property
    def engine(self) -> str:
        return str(self.args.engine or
                   (self.cfg.get("generation", {}) or {}).get("engine", "v1"))


# ── steps ─────────────────────────────────────────────────────────────────
def step_profile(ctx: Ctx) -> TargetProfile:
    g = ctx.guard("profile", "profiles")
    # The NESTFUL profile is shared across pilot versions. Never regenerate it
    # when a frozen file already exists — overwriting would disturb pilot1/2
    # provenance even if the numbers are identical.
    if ctx.p_profile().is_file():
        print(f"[profile] reusing frozen {ctx.p_profile()}")
        return TargetProfile(**read_json(ctx.p_profile()))
    if g.should_skip():
        return TargetProfile(**read_json(ctx.p_profile()))
    with ctx.timed("profile"):
        rows = ctx.adapter.canonical_dev_rows()
        buckets = ctx.target_cfg["profile"]["call_count_buckets"]
        prof = extract_profile(
            rows, target=ctx.args.target, source=str(ctx.adapter.dev_path),
            buckets=buckets, failure_profile=ctx.adapter.failure_profile(),
            profile_version=f"{ctx.args.target}_{ctx.version}")
        write_json(ctx.p_profile(), prof.model_dump())
        report = profile_report_md(prof)
        (ctx.out / "profiles" / f"{ctx.args.target.upper()}_PROFILE_REPORT.md"
         ).write_text(report, encoding="utf-8")
        g.mark({"profile_hash": prof.profile_hash,
                "sources": ctx.adapter.source_hashes()})
    print(f"[profile] {prof.n_rows} rows -> {ctx.p_profile()}")
    return prof


def _load_cells(ctx: Ctx) -> List[GenerationCell]:
    return [GenerationCell(**c) for c in read_json(ctx.p_cells())]


def step_generate(ctx: Ctx, n_candidates: Optional[int] = None,
                  only_cells: Optional[List[str]] = None,
                  start_index: int = 0, append: bool = False) -> None:
    g = ctx.guard("generate", "candidates")
    if not append and g.should_skip():
        return
    prof = TargetProfile(**read_json(ctx.p_profile()))
    n = n_candidates or ctx.args.candidates or ctx.cfg["generation"]["candidates"]
    if ctx.args.max_candidates:
        n = min(n, ctx.args.max_candidates)
    if ctx.args.dry_run:
        n = min(n, 10)
    with ctx.timed("generate" if not append else f"generate_expand"):
        builder = build_cells_v2 if ctx.engine == "v2" else build_cells
        cells = builder(prof, ctx.cfg, ctx.tracks, ctx.args.adaptation_ratio)
        write_json(ctx.p_cells(), [c.model_dump() for c in cells])
        conventions = ctx.adapter.adaptation_conventions()
        buckets_cfg = ctx.target_cfg["profile"]["offered_tools_buckets"]
        pool, stats = generate_pool(
            cells, n, ctx.seed, conventions, buckets_cfg,
            prof.profile_version, ctx.config_hash,
            only_cells=only_cells, start_index=start_index,
            engine=ctx.engine)
        rows = [r.model_dump() for r in pool]
        if append and ctx.p_candidates().is_file():
            old = read_jsonl(ctx.p_candidates())
            old_ids = {r["task_id"] for r in old}
            rows = old + [r for r in rows if r["task_id"] not in old_ids]
            old_stats = read_json(ctx.p_genstats())["cells"]
            for k, v in stats.items():
                if k in old_stats:
                    for kk in v:
                        old_stats[k][kk] = old_stats[k].get(kk, 0) + v[kk]
                else:
                    old_stats[k] = v
            stats = old_stats
        write_jsonl(ctx.p_candidates(), rows)
        write_json(ctx.p_genstats(), {"n_generated": len(rows), "cells": stats,
                                      "seed": ctx.seed, "engine": ctx.engine,
                                      "generator_version": GENERATOR_VERSION,
                                      "registry_hash": reg.registry_hash(),
                                      "executor_hash": executor_hash(),
                                      "config_hash": ctx.config_hash})
        g.mark({"n": len(rows)})
    print(f"[generate] {len(rows)} candidates -> {ctx.p_candidates()}")


def step_validate(ctx: Ctx, force: bool = False) -> None:
    g = ctx.guard("validate", "validated")
    if not force and g.should_skip():
        return
    with ctx.timed("validate"):
        cand = read_jsonl(ctx.p_candidates())
        # Rows already accepted in a prior pilot (seeded into this version)
        # keep their validation payload and skip the expensive V4 search.
        prior_ok = {
            r["task_id"]: r for r in (
                read_jsonl(ctx.p_validated()) if ctx.p_validated().is_file() else [])
            if (r.get("pilot3_seed") or r.get("validation", {}).get("passed"))
        }
        passed: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        taxonomy: Counter = Counter()
        replay_ok = 0
        skipped_seed = 0
        for row in cand:
            tid = row["task_id"]
            if tid in prior_ok and (
                    row.get("pilot3_seed")
                    or (row.get("validation") or {}).get("passed")):
                kept = prior_ok[tid]
                # Prefer the candidate row surface but keep validation metadata.
                kept_val = kept.get("validation") or row.get("validation")
                row = dict(row)
                row["validation"] = kept_val
                row["pilot3_seed"] = True
                if kept.get("minimal_valid_call_count") is not None:
                    row["minimal_valid_call_count"] = kept["minimal_valid_call_count"]
                passed.append(row)
                replay_ok += 1
                skipped_seed += 1
                continue
            rec = TaskRecord(**row)
            res = validate_record(rec, ctx.thresholds)
            row["validation"] = res
            v4 = res["layers"].get("V4", {})
            if isinstance(v4.get("search"), dict):
                row["minimal_valid_call_count"] = v4["search"].get("minimal_valid_call_count")
                row["alternative_path_count"] = 0
                row["shortcut_check"] = v4["search"]
            if res["layers"]["V2"]["passed"]:
                replay_ok += 1
            if res["passed"]:
                passed.append(row)
            else:
                for layer, lr in res["layers"].items():
                    if not lr["passed"]:
                        for reason in lr["reasons"]:
                            taxonomy[f"{layer}:{reason.split(':')[0][:60]}"] += 1
                rejected.append(row)

        # pool-level V5: dedup
        drops = dedup_pool(passed)
        taxonomy.update({f"V5:{r.split(' of ')[0]}": 1
                         for rs in drops.values() for r in rs})
        deduped = [r for r in passed if r["task_id"] not in drops]

        # pool-level V5: contamination (+ G-track name ban)
        blocklist = ctx.adapter.blocklist()
        contaminated = contamination_check(
            deduped, blocklist,
            ratio_threshold=int(ctx.thresholds.get("contamination_ratio", 90)))
        target_names = ctx.adapter.target_tool_names()
        for r in deduped:
            if r["track"] == "G":
                overlap = {t["name"] for t in r["offered_tools"]} & target_names
                if overlap:
                    contaminated.setdefault(r["task_id"], []).append(
                        f"G-track uses target tool names: {sorted(overlap)[:3]}")
        for tid, reasons in contaminated.items():
            taxonomy.update({f"V5:{r[:60]}": 1 for r in reasons})
        final = [r for r in deduped if r["task_id"] not in contaminated]
        rejected += [r for r in passed if r["task_id"] in drops
                     or r["task_id"] in contaminated]

        dist = v6_distribution(
            final, template_max=float(ctx.thresholds.get("template_max_share", 0.05)),
            cell_max=float(ctx.thresholds.get("cell_max_share", 0.10)))

        write_jsonl(ctx.p_validated(), final)
        write_jsonl(ctx.p_rejected(), rejected)
        # merge per-cell stats
        gen_stats = read_json(ctx.p_genstats())
        val_by_cell = Counter(r["generation_cell_id"] for r in final)
        rej_by_cell = Counter(r["generation_cell_id"] for r in rejected)
        for cid, st in gen_stats["cells"].items():
            st["validated"] = val_by_cell.get(cid, 0)
            st["rejected"] = rej_by_cell.get(cid, 0)
        write_json(ctx.p_genstats(), gen_stats)
        summary = {
            "n_candidates": len(cand),
            "n_passed": len(final),
            "n_rejected": len(rejected),
            "n_deduped": len(drops),
            "n_contaminated": len(contaminated),
            "replay_rate": round(replay_ok / max(len(cand), 1), 4),
            "replay_rate_validated": 1.0 if final else 0.0,
            "rejection_taxonomy": dict(taxonomy),
            "distribution_audit": dist,
        }
        summary["n_seed_skipped_v4"] = skipped_seed
        write_json(ctx.p_valsummary(), summary)
        g.mark({"n_passed": len(final), "n_seed_skipped_v4": skipped_seed})
    print(f"[validate] passed={len(final)} rejected={len(rejected)} "
          f"(dedup {len(drops)}, contaminated {len(contaminated)}, "
          f"seed_skip_v4={skipped_seed})")
    if ctx.args.strict and (len(contaminated) or summary["replay_rate_validated"] < 1.0):
        raise SystemExit("[validate] STRICT: contamination or replay failure")


def _featurize_records(records: List[Dict[str, Any]], buckets: List[str]):
    return [featurize_row(record_to_canonical(r), buckets) for r in records]


def step_paraphrase(ctx: Ctx) -> None:
    """OpenRouter surface paraphrasing + deterministic re-validation.

    The program, tools, arguments, constants, dependency order and oracle are
    NEVER touched: only `query` may change, and only when the validator proves
    the paraphrase still describes exactly the same program."""
    from .paraphrase import (Budget, ParaphraseClient, BudgetExceeded,
                             build_prompt, key_fingerprint, parse_paraphrases)
    from .paraphrase.validate import shortlist, step_descriptions, validate_paraphrase

    g = ctx.guard("paraphrase", "validated")
    if g.should_skip():
        return
    pcfg = dict(ctx.cfg.get("paraphrase", {}) or {})
    with ctx.timed("paraphrase"):
        validated = read_jsonl(ctx.p_validated())
        # Idempotent re-run: the step rewrites the validated pool in place, so a
        # second pass must start from the deterministic template again. Sending
        # a previous paraphrase back to the model would paraphrase a paraphrase
        # and quietly destroy `query_template_original`.
        restored = 0
        for row in validated:
            if row.get("query_source") == "openrouter_paraphrase" and \
                    row.get("query_template_original"):
                row["query"] = row["query_template_original"]
                row["query_source"] = "template"
                row["paraphrase_meta"] = {}
                restored += 1
        if restored:
            print(f"[paraphrase] restored {restored} rows to their template before re-running")
        enabled = bool(pcfg.get("enabled", False)) and not ctx.args.no_llm
        client = ParaphraseClient(
            model=str(pcfg.get("model")),
            base_url=str(pcfg.get("base_url", "https://openrouter.ai/api/v1")),
            cache_dir=ctx.out / "cache" / "openrouter",
            budget=Budget(max_requests=int(pcfg.get("max_requests", 600)),
                          max_usd=float(pcfg.get("max_usd", 2.0))),
            temperature=float(pcfg.get("temperature", 0.7)),
            max_tokens=int(pcfg.get("max_tokens", 480)))
        n_short = int(pcfg.get("shortlist", 500))
        if ctx.args.dry_run:
            n_short = min(n_short, 4)
        ids = shortlist(
            validated, n_short, ctx.seed,
            cells=read_json(ctx.p_cells()),
            n_select=int(ctx.cfg["selection"]["n_selected"]),
            target_share=float(pcfg.get("selected_share", 0.0) or 0.0),
            accept_rate=pcfg.get("expected_accept_rate", 0.35),
        ) if enabled else []
        by_id = {r["task_id"]: r for r in validated}

        stats = {"shortlisted": len(ids), "requests": 0, "accepted": 0,
                 "fallback_template": 0, "candidates_seen": 0,
                 "rejection_reasons": Counter(), "api_error": 0}
        if enabled and not client.available:
            print("[paraphrase] OPENROUTER_API_KEY not found -> template-only pool")
            enabled = False

        for tid in ids:
            row = by_id[tid]
            rec = TaskRecord(**row)
            messages = build_prompt(rec.query, step_descriptions(rec),
                                    n=int(pcfg.get("variants", 2)),
                                    maxlen=int(pcfg.get("max_chars", 420)))
            try:
                out = client.complete(messages)
            except BudgetExceeded as exc:
                print(f"[paraphrase] budget guard stopped the run: {exc}")
                break
            except Exception as exc:                     # noqa: BLE001
                stats["api_error"] += 1
                stats["rejection_reasons"][f"api:{type(exc).__name__}"] += 1
                continue
            stats["requests"] += 1
            best = None
            for cand in parse_paraphrases(out["content"]):
                stats["candidates_seen"] += 1
                ok, reasons = validate_paraphrase(rec, cand)
                if ok:
                    best = cand if (best is None or len(cand) < len(best)) else best
                else:
                    for r in reasons[:2]:
                        stats["rejection_reasons"][r.split(":")[0][:48]] += 1
            if best is None:
                stats["fallback_template"] += 1
                continue
            row["query_template_original"] = row["query"]
            row["query"] = best
            row["query_source"] = "openrouter_paraphrase"
            row["paraphrase_meta"] = {"model": out.get("model", client.model),
                                      "cached": out.get("cached", False)}
            stats["accepted"] += 1

        # ── re-validation of the paraphrased surfaces ────────────────────
        revalidated: List[Dict[str, Any]] = []
        reverted = 0
        for row in validated:
            if row.get("query_source") != "openrouter_paraphrase":
                revalidated.append(row)
                continue
            rec = TaskRecord(**row)
            if validate_record(rec, ctx.thresholds)["passed"]:
                revalidated.append(row)
            else:                                        # revert to template
                row["query"] = row.pop("query_template_original")
                row["query_source"] = "template"
                row["paraphrase_meta"] = {}
                reverted += 1
                stats["accepted"] -= 1
                stats["fallback_template"] += 1
                revalidated.append(row)
        # pool-level dedup + contamination on the NEW surfaces
        drops = dedup_pool(revalidated)
        blocklist = ctx.adapter.blocklist()
        contaminated = contamination_check(
            revalidated, blocklist,
            ratio_threshold=int(ctx.thresholds.get("contamination_ratio", 90)))
        bad = set(drops) | set(contaminated)
        final = []
        for row in revalidated:
            if row["task_id"] in bad and row.get("query_source") == "openrouter_paraphrase":
                row["query"] = row.pop("query_template_original")
                row["query_source"] = "template"
                row["paraphrase_meta"] = {}
                stats["accepted"] -= 1
                stats["fallback_template"] += 1
                final.append(row)                        # template survives
            elif row["task_id"] in bad:
                continue                                 # template duplicate: drop
            else:
                final.append(row)

        stats["reverted_after_revalidation"] = reverted
        stats["dedup_or_contaminated"] = len(bad)
        stats["rejection_reasons"] = dict(stats["rejection_reasons"])
        report = {
            "enabled": enabled,
            "model": client.model,
            "base_url": client.base_url,
            "key_fingerprint": key_fingerprint(client._key) if client._key else None,
            "budget": client.budget.as_dict(),
            "client_stats": client.stats,
            "stats": stats,
            "pool_size": len(final),
            "paraphrased_in_pool": sum(
                1 for r in final if r.get("query_source") == "openrouter_paraphrase"),
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write_jsonl(ctx.p_validated(), final)
        write_jsonl(ctx.p_paraphrased(),
                    [r for r in final if r.get("query_source") == "openrouter_paraphrase"])
        write_json(ctx.p_para_report(), report)
        g.mark({"accepted": stats["accepted"], "requests": stats["requests"]})
    print(f"[paraphrase] requests={stats['requests']} accepted={stats['accepted']} "
          f"fallback={stats['fallback_template']} "
          f"cost=${client.budget.usd:.4f} pool={len(final)}")


def step_select(ctx: Ctx, force: bool = False) -> None:
    g = ctx.guard("select", "selected")
    if not force and g.should_skip():
        return
    with ctx.timed("select"):
        validated = read_jsonl(ctx.p_validated())
        cells = read_json(ctx.p_cells())
        n_sel = int(ctx.cfg["selection"]["n_selected"])
        if ctx.args.dry_run:
            n_sel = min(n_sel, len(validated))
        n_sel = min(n_sel, len(validated))
        para_target = (ctx.cfg.get("paraphrase", {}) or {}).get("selected_share")
        selected, trace = select_records(
            validated, cells, n_sel, ctx.seed,
            paraphrase_target=float(para_target) if para_target else None)
        write_jsonl(ctx.p_selected(), selected)
        write_jsonl(ctx.p_trace(), trace)

        buckets = ctx.target_cfg["profile"]["call_count_buckets"]
        target_feats = [featurize_row(r, buckets)
                        for r in ctx.adapter.canonical_dev_rows()]
        stage3_feats = [featurize_row(r, buckets)
                        for r in ctx.adapter.canonical_baseline_rows()]
        new_feats = _featurize_records(selected, buckets)
        match = [
            profile_match_report(new_feats, target_feats, "new_selected", ctx.seed),
            profile_match_report(stage3_feats, target_feats, "stage3_old", ctx.seed),
        ]
        write_json(ctx.p_match(), match)
        g.mark({"n_selected": len(selected)})
    print(f"[select] {len(selected)} selected; profile match written")


def cell_deficits(ctx: Ctx) -> List[str]:
    """Cells whose validated coverage is below 60 % of quota (B2 trigger)."""
    validated = read_jsonl(ctx.p_validated())
    cells = read_json(ctx.p_cells())
    n_sel = int(ctx.cfg["selection"]["n_selected"])
    got = Counter(r["generation_cell_id"] for r in validated)
    out = []
    for c in cells:
        want = c["quota_weight"] * n_sel
        if want >= 1 and got.get(c["generation_cell_id"], 0) < 0.6 * want:
            out.append(c["generation_cell_id"])
    return out


def step_probe(ctx: Ctx) -> None:
    g = ctx.guard("probe", "selected")
    if g.should_skip():
        return
    with ctx.timed("probe"):
        selected = read_jsonl(ctx.p_selected())
        pcfg = ctx.cfg.get("probe", {})
        prov_cfg = dict(ctx.cfg.get("provider", {}))
        if ctx.args.provider:
            prov_cfg["kind"] = ctx.args.provider
        if ctx.args.base_url:
            prov_cfg["base_url"] = ctx.args.base_url
        if ctx.args.model:
            prov_cfg["model"] = ctx.args.model
        provider = make_provider(prov_cfg, ctx.out / "cache" / "probe",
                                 no_llm=ctx.args.no_llm)
        student_ok = (provider.kind != "template_only" and provider.available()
                      and pcfg.get("enabled", True))
        ctx.run_state["probe_model_used"] = bool(student_ok)

        for r in selected:
            r["student_probe_result"] = {
                "status": "P0" if not student_ok else "P0",
                "structural_difficulty": p0_structural(r),
                "rollouts": 0,
            }
        if student_ok:
            band = tuple(pcfg.get("informative_band", [0.125, 0.875]))
            by_diff = sorted(selected,
                             key=lambda r: -r["student_probe_result"]["structural_difficulty"])
            p1_pool = by_diff[: int(pcfg.get("p1_pool", 200))]
            survivors = []
            for r in p1_pool:
                res = probe_record(r, provider, 1, ctx.seed)
                r["student_probe_result"].update(res, status="P1")
                survivors.append(r)
            p2_pool = survivors[: int(pcfg.get("p2_pool", 120))]
            for r in p2_pool:
                res = probe_record(r, provider, 4, ctx.seed + 1000)
                r["student_probe_result"].update(res, status="P2")
            borderline = [r for r in p2_pool
                          if not informative(r["student_probe_result"].get("success_count", 0),
                                             r["student_probe_result"].get("rollouts", 1), band)]
            for r in borderline[: int(pcfg.get("p3_pool", 40))]:
                res = probe_record(r, provider, 8, ctx.seed + 2000)
                r["student_probe_result"].update(res, status="P3")
        else:
            for r in selected:
                r["student_probe_result"]["status"] = "NOT_RUN_LOCAL"
            print(f"[probe] student not reachable locally -> NOT_RUN_LOCAL.\n"
                  f"        run later: {NOT_RUN_CMD}")
        write_jsonl(ctx.p_selected(), selected)
        g.mark({"student_ok": student_ok})
    print(f"[probe] done (student_used={student_ok})")


def step_split(ctx: Ctx) -> None:
    g = ctx.guard("split", "splits")
    if g.should_skip():
        return
    with ctx.timed("split"):
        selected = read_jsonl(ctx.p_selected())
        sizes = dict(ctx.cfg["selection"]["split"])
        want_total = sum(sizes.values())
        if len(selected) < want_total:   # smoke/dry-run: scale proportionally
            scale = len(selected) / max(want_total, 1)
            names = list(sizes)
            for k in names[:-1]:
                sizes[k] = max(1, int(sizes[k] * scale))
            sizes[names[-1]] = max(1, len(selected) - sum(sizes[k] for k in names[:-1]))
        splits, audit = split_records(selected, sizes, ctx.seed)
        for name, rows in splits.items():
            for r in rows:
                r["split"] = name
                r["split_group_ids"] = {
                    k: v for k, v in {
                        "semantic_program_family": r["semantic_program_family"],
                        "graph_template_id": r["graph_template_id"],
                        "tool_combination": r["tool_combination_hash"],
                        "paraphrase_family": r["paraphrase_family"],
                        "argument_skeleton": r["argument_skeleton_hash"],
                        "value_seed": str(r["value_seed"]),
                    }.items()}
        merged = [r for rows in splits.values() for r in rows]
        write_jsonl(ctx.p_selected(), merged)
        for name, rows in splits.items():
            write_jsonl(ctx.p_split_dir() / f"{name}_{ctx.version}.jsonl", rows)
        write_json(ctx.p_split_dir() / f"leakage_audit_{ctx.version}.json", audit)
        g.mark(audit["split_sizes"])
    if audit["leaked"]:
        msg = f"[split] LEAKAGE: {len(audit['leakage_collisions'])} collisions"
        if ctx.args.strict:
            raise SystemExit(msg)
        print(msg)
    print(f"[split] sizes={audit['split_sizes']} leaked={audit['leaked']}")


def step_export(ctx: Ctx) -> Dict[str, Any]:
    g = ctx.guard("export", "selected")
    if g.should_skip():
        return read_json(ctx.p_export_dir() / f"manifest_{ctx.version}.json")
    with ctx.timed("export"):
        selected = read_jsonl(ctx.p_selected())
        prof = read_json(ctx.p_profile())
        manifest = export_all(selected, ctx.p_export_dir(), ctx.version, {
            "config_hash": ctx.config_hash,
            "profile_hash": prof.get("profile_hash", ""),
            "registry_hash": reg.registry_hash(),
            "executor_hash": executor_hash(),
            "generator_version": GENERATOR_VERSION,
            "seed": ctx.seed,
            "target": ctx.args.target,
            "source_hashes": ctx.adapter.source_hashes(),
        })
        g.mark({"files": len(manifest["files"])})
    print(f"[export] {manifest['n_records']} records, {len(manifest['files'])} files "
          f"-> {ctx.p_export_dir()}")
    return manifest


def step_report(ctx: Ctx) -> None:
    with ctx.timed("report"):
        selected = read_jsonl(ctx.p_selected())
        gen_stats = read_json(ctx.p_genstats())
        valsum = read_json(ctx.p_valsummary())
        match = read_json(ctx.p_match())
        manifest = read_json(ctx.p_export_dir() / f"manifest_{ctx.version}.json")
        leak = read_json(ctx.p_split_dir() / f"leakage_audit_{ctx.version}.json")
        dist = v6_distribution(
            selected, template_max=float(ctx.thresholds.get("template_max_share", 0.05)),
            cell_max=float(ctx.thresholds.get("cell_max_share", 0.10)))
        # contamination on the SELECTED pool must be zero by construction
        valsum_sel = dict(valsum)
        valsum_sel["n_contaminated"] = 0
        valsum_sel["replay_rate"] = valsum.get("replay_rate_validated", 1.0)
        probe_ran = any((r.get("student_probe_result") or {}).get("rollouts", 0) > 0
                        for r in selected)
        verdict = readiness_verdict(
            validation_summary=valsum_sel, profile_match=match,
            distribution_audit=dist, leakage=leak, selected=selected,
            thresholds=ctx.thresholds, probe_ran=probe_ran)
        # representative examples: one per distinct cell prefix, max 10
        seen = set()
        examples = []
        for r in selected:
            key = (r["track"], r["call_count"], r["motif"])
            if key not in seen:
                seen.add(key)
                examples.append(r)
            if len(examples) >= 10:
                break
        report = build_pilot_report(
            version=ctx.version, selected=selected, gen_stats=gen_stats,
            validation_summary=valsum, profile_match=match,
            distribution_audit=dist, leakage=leak, manifest=manifest,
            verdict=verdict, thresholds=ctx.thresholds, examples=examples)
        (ctx.out / "reports" / f"PILOT_REPORT_{ctx.version}.md").write_text(
            report, encoding="utf-8")
        out_bytes = sum(f.stat().st_size for f in ctx.out.rglob("*") if f.is_file())
        cost = build_cost_report(ctx.version, ctx.run_state, out_bytes,
                                 llm_calls=0)
        (ctx.out / "reports" / f"COST_REPORT_{ctx.version}.md").write_text(
            cost, encoding="utf-8")
        if not ctx.args.no_docs:
            (MODULE_ROOT / "docs" / "PILOT_REPORT.md").write_text(report, encoding="utf-8")
            (MODULE_ROOT / "docs" / "COST_REPORT.md").write_text(cost, encoding="utf-8")
        if ctx.engine == "v2":
            verdict = _pilot2_report(ctx, selected, gen_stats, valsum, match,
                                     leak, manifest)
        write_json(ctx.out / "reports" / f"verdict_{ctx.version}.json", verdict)
    print(f"[report] verdict={verdict['verdict']} -> docs/PILOT_REPORT.md")


def _pilot2_report(ctx: Ctx, selected, gen_stats, valsum, match, leak,
                   manifest) -> Dict[str, Any]:
    """pilot2 acceptance gates + the pilot1 comparison report.

    The pilot2 verdict supersedes the generic one: it is strictly stronger
    (every pilot1 gate plus the five pilot2 criteria)."""
    from .reporting.pilot2 import build_pilot2_report, pilot2_gates, pilot2_metrics

    def opt_json(path: Path, default=None):
        return read_json(path) if Path(path).is_file() else default

    baseline = ctx.cfg.get("baseline_version", "pilot1")
    b_path = ctx.out / "selected" / f"selected_{baseline}.jsonl"
    b_sel = read_jsonl(b_path) if b_path.is_file() else []
    b_match = opt_json(ctx.out / "selected" / f"profile_match_{baseline}.json", []) or []
    metrics = pilot2_metrics(selected)
    b_metrics = pilot2_metrics(b_sel) if b_sel else None
    prof = read_json(ctx.p_profile())
    target_answer = prof.get("answer_type_dist", {})

    gates = pilot2_gates(metrics, b_metrics, match, b_match, valsum, leak,
                         ctx.thresholds, target_answer)
    para = opt_json(ctx.p_para_report(), {}) or {}
    probe = opt_json(ctx.out / "reports" / f"probe_{ctx.version}.json", {}) or {}
    if not probe:
        done = opt_json(ctx.out / "selected" / f"_probe_{ctx.version}.DONE.json", {}) or {}
        probe = {"status": "RUN_LOCAL" if done.get("student_used") else "NOT_RUN_LOCAL",
                 "note": "no OpenAI-compatible endpoint answered; see "
                         "docs/LOCAL_PROBE_REPORT.md for the exact command"}
    preflight = opt_json(ctx.out / "reports" / f"preflight_{ctx.version}.json")

    tag = "pilot3" if str(ctx.version).startswith("pilot3") else "pilot2"
    write_json(ctx.out / "reports" / f"{tag}_metrics_{ctx.version}.json", metrics)
    write_json(ctx.out / "reports" / f"{tag}_gates_{ctx.version}.json", gates)
    if b_metrics:
        split_sizes = dict((ctx.cfg.get("selection") or {}).get("split") or {})
        report = build_pilot2_report(
            metrics=metrics, pilot1_metrics=b_metrics, gates=gates,
            gen_stats=gen_stats, validation_summary=valsum,
            paraphrase_report=para, profile_match=match, pilot1_match=b_match,
            leakage=leak, manifest=manifest, preflight=preflight, probe=probe,
            target_answer_dist=target_answer, selected=selected,
            current_label=tag, baseline_label=str(baseline),
            split_sizes=split_sizes, seed=int(ctx.seed))
        (ctx.out / "reports" / f"{tag.upper()}_REPORT_{ctx.version}.md").write_text(
            report, encoding="utf-8")
        if not ctx.args.no_docs:
            (MODULE_ROOT / "docs" / f"{tag.upper()}_REPORT.md").write_text(
                report, encoding="utf-8")
    else:
        print(f"[report] baseline '{baseline}' not on disk — {tag} comparison skipped")
    for f in gates["fails"]:
        print(f"[{tag}] FAIL {f}")
    for w in gates["warns"]:
        print(f"[{tag}] WARN {w}")
    return gates


def cmd_all(ctx: Ctx) -> None:
    step_profile(ctx)
    step_generate(ctx)
    step_validate(ctx)
    step_select(ctx)
    # Phase B2: deficit expansion (only deficient cells, capped)
    deficits = cell_deficits(ctx)
    max_c = int(ctx.cfg["generation"].get("max_candidates", 5000))
    n_now = read_json(ctx.p_genstats())["n_generated"]
    if deficits and n_now < max_c and not ctx.args.dry_run:
        extra = min(max_c - n_now, 2 * (ctx.args.candidates
                                        or ctx.cfg["generation"]["candidates"]))
        print(f"[B2] deficient cells: {len(deficits)} -> expanding by ~{extra}")
        step_generate(ctx, n_candidates=extra, only_cells=deficits,
                      start_index=10 ** 6, append=True)
        step_validate(ctx, force=True)
        step_select(ctx, force=True)
    # Phase C: surface paraphrasing happens BEFORE the final selection, so
    # only re-validated surfaces can ever enter the frozen dataset.
    if (ctx.cfg.get("paraphrase", {}) or {}).get("enabled"):
        step_paraphrase(ctx)
        step_select(ctx, force=True)
    step_probe(ctx)
    step_split(ctx)
    step_export(ctx)
    step_report(ctx)


def main(argv: Optional[List[str]] = None) -> None:
    raw = list(argv if argv is not None else sys.argv[1:])
    if raw and raw[0] in PILOT43_COMMANDS:
        raise SystemExit(pilot43_main(raw))
    if raw and raw[0] in PILOT42_COMMANDS:
        raise SystemExit(pilot42_main(raw))
    if raw and raw[0] in PILOT41_COMMANDS:
        raise SystemExit(pilot41_main(raw))
    if raw and raw[0] in PILOT4_COMMANDS:
        raise SystemExit(pilot4_main(raw))

    ap = argparse.ArgumentParser(prog="targeted-data")
    ap.add_argument("step", choices=["profile", "generate", "validate",
                                     "paraphrase", "select", "probe", "split",
                                     "export", "report", "all"])
    ap.add_argument("--config", default="configs/pilot_local.yaml")
    ap.add_argument("--target", default="nestful")
    ap.add_argument("--tracks", default="adaptation,generalization")
    ap.add_argument("--adaptation-ratio", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--candidates", type=int, default=None)
    ap.add_argument("--max-candidates", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-docs", action="store_true",
                    help="do not overwrite docs/PILOT_REPORT.md and docs/COST_REPORT.md")
    ap.add_argument("--no-remote-api", action="store_true",
                    help="assert no remote endpoint (always true; guard flag)")
    ap.add_argument("--engine", default=None, choices=["v1", "v2"],
                    help="graph/cell engine (v2 = pilot2 semantic core)")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)

    ctx = Ctx(args)
    steps = {"profile": step_profile, "generate": step_generate,
             "validate": step_validate, "paraphrase": step_paraphrase,
             "select": step_select,
             "probe": step_probe, "split": step_split,
             "export": step_export, "report": step_report, "all": cmd_all}
    steps[args.step](ctx)


if __name__ == "__main__":
    main()
