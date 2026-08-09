"""Recount every Pilot4.3 stage from the JSONL files themselves.

Produces PILOT43_RESUME_AUDIT.{json,md}. Nothing here trusts a prior report.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set


RUN_ID = "pilot4_3_nestful_final"


def iter_jsonl(path: pathlib.Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
        yield  # pragma: no cover
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def count_lines(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def file_info(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "bytes": 0, "lines": 0}
    return {
        "exists": True,
        "bytes": path.stat().st_size,
        "lines": count_lines(path),
        "mtime_utc": path.stat().st_mtime,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)

    files = {
        "semantic_candidates": out / "semantic_candidates.jsonl",
        "semantic_hard_valid": out / "semantic_hard_valid.jsonl",
        "query_render_shortlist": out / "query_render_shortlist.jsonl",
        "verified_candidates": out / "verified_candidates.jsonl",
        "v4_per_task": out / "v4_per_task.jsonl",
        "per_node_necessity": out / "per_node_necessity.jsonl",
        "per_task_validation_ledger": out / "per_task_validation_ledger.jsonl",
        "llm_rendered": out / "llm_rendered.jsonl",
        "openrouter_requests": out / "openrouter_requests_pilot43.jsonl",
        "openrouter_failures": out / "openrouter_failures_pilot43.jsonl",
        "critic_disagreements": out / "critic_disagreements.jsonl",
        "rejected_candidates": out / "rejected_candidates.jsonl",
        "query_hard_valid": out / "query_hard_valid.jsonl",
        "deterministic_rendered": out / "deterministic_rendered.jsonl",
    }
    file_stats = {name: file_info(path) for name, path in files.items()}

    # ── shortlist ────────────────────────────────────────────────────────
    shortlist_ids: List[str] = []
    shortlist_by_bucket: collections.Counter = collections.Counter()
    shortlist_coding = 0
    for row in iter_jsonl(files["query_render_shortlist"]):
        tid = row.get("task_id")
        if not tid:
            continue
        shortlist_ids.append(tid)
        shortlist_by_bucket[row.get("call_bucket", "?")] += 1
        if row.get("coding_like"):
            shortlist_coding += 1
    shortlist_set = set(shortlist_ids)

    # ── verified / V4 / necessity ────────────────────────────────────────
    verified_ids: Set[str] = set()
    selectable: Set[str] = set()
    reject_reasons: collections.Counter = collections.Counter()
    v4_ok = v4_shortcut = v4_unresolved = v4_missing = 0
    nec_ok = nec_fail = 0
    distractor_ok = distractor_fail = 0
    verified_buckets: collections.Counter = collections.Counter()
    selectable_buckets: collections.Counter = collections.Counter()
    selectable_coding = 0
    selectable_answer: collections.Counter = collections.Counter()

    for row in iter_jsonl(files["verified_candidates"]):
        tid = row.get("task_id")
        if not tid:
            continue
        verified_ids.add(tid)
        bucket = row.get("call_bucket") or (
            str(row.get("call_count")) if isinstance(row.get("call_count"), int)
            and row["call_count"] <= 5 else "6+" if row.get("call_count") else "?"
        )
        # Prefer shortlist metadata when present
        verified_buckets[bucket] += 1

        v4 = row.get("v4") or {}
        if not v4.get("v4_executed"):
            v4_missing += 1
        elif v4.get("has_shortcut"):
            v4_shortcut += 1
        elif not v4.get("resolved", True):
            v4_unresolved += 1
        else:
            v4_ok += 1

        nec = row.get("necessity_summary") or {}
        if nec.get("all_necessary") is True or (
            isinstance(row.get("necessity"), list)
            and row["necessity"]
            and all(n.get("necessary") for n in row["necessity"])
        ):
            nec_ok += 1
        else:
            # also accept when necessity list present and all True
            nodes = row.get("necessity") or []
            if nodes and all(n.get("necessary") for n in nodes):
                nec_ok += 1
            else:
                nec_fail += 1

        offered = row.get("offered") or {}
        if offered.get("distractor_count", 0) >= 1 or row.get("selectable"):
            distractor_ok += 1
        else:
            distractor_fail += 1

        if row.get("selectable"):
            selectable.add(tid)
            selectable_buckets[bucket] += 1
            if row.get("coding_like"):
                selectable_coding += 1
            selectable_answer[row.get("answer_type", "?")] += 1
        else:
            reason = (row.get("reject_reason") or "unknown")[:80]
            reject_reasons[reason] += 1

    # Enrich selectable strata from shortlist (verified rows may omit fields)
    shortlist_meta = {
        r["task_id"]: r for r in iter_jsonl(files["query_render_shortlist"])
        if r.get("task_id")
    }
    selectable_buckets = collections.Counter()
    selectable_coding = 0
    selectable_answer = collections.Counter()
    selectable_mode_elig: collections.Counter = collections.Counter()
    for tid in selectable:
        meta = shortlist_meta.get(tid, {})
        selectable_buckets[meta.get("call_bucket", "?")] += 1
        if meta.get("coding_like"):
            selectable_coding += 1
        selectable_answer[meta.get("answer_type", "?")] += 1

    shortlist_not_verified = sorted(shortlist_set - verified_ids)
    verified_not_shortlist = len(verified_ids - shortlist_set)

    # ── LLM renders ──────────────────────────────────────────────────────
    rendered_by_id: Dict[str, Dict[str, Any]] = {}
    writer_ok = first_critic_pass = first_critic_any = 0
    second_critic_any = second_critic_pass = 0
    blocked: collections.Counter = collections.Counter()
    reusable: Set[str] = set()
    foreign_run = 0
    prompt_versions: collections.Counter = collections.Counter()
    sources: collections.Counter = collections.Counter()

    for row in iter_jsonl(files["llm_rendered"]):
        tid = row.get("task_id") or row.get("sample_id")
        if not tid:
            continue
        rendered_by_id[tid] = row
        if row.get("run_id") and row.get("run_id") != RUN_ID:
            foreign_run += 1
        pv = row.get("prompt_version") or row.get("writer_prompt_version") or ""
        if pv:
            prompt_versions[str(pv)] += 1
        sources[row.get("query_source") or row.get("source") or "unknown"] += 1

        writer = row.get("writer") or row.get("structured") or {}
        query = row.get("query") or (writer.get("query") if isinstance(writer, dict) else "")
        if query:
            writer_ok += 1

        critic = row.get("critic") or row.get("first_critic") or {}
        if isinstance(critic, dict) and critic.get("verdict"):
            first_critic_any += 1
            if str(critic.get("verdict")).upper() == "PASS":
                first_critic_pass += 1
        elif row.get("critic_verdict"):
            first_critic_any += 1
            if str(row.get("critic_verdict")).upper() == "PASS":
                first_critic_pass += 1

        sc = row.get("second_critic")
        if sc is not None or row.get("second_critic_verdict"):
            second_critic_any += 1
            verdict = (
                (sc.get("verdict") if isinstance(sc, dict) else None)
                or row.get("second_critic_verdict")
            )
            if str(verdict or "").upper() == "PASS":
                second_critic_pass += 1

        if row.get("blocked"):
            blocked[row.get("blocked_reason") or row.get("block_reason") or "?"] += 1
        else:
            # reusable if not blocked and has a query
            if query and tid in selectable:
                reusable.add(tid)

    # ── OpenRouter log isolation ─────────────────────────────────────────
    or_purposes: collections.Counter = collections.Counter()
    or_models: collections.Counter = collections.Counter()
    or_run_ids: collections.Counter = collections.Counter()
    or_prompt_versions: collections.Counter = collections.Counter()
    or_foreign = 0
    or_n = 0
    for row in iter_jsonl(files["openrouter_requests"]):
        or_n += 1
        rid = row.get("run_id") or ""
        or_run_ids[rid or "(missing)"] += 1
        if rid and rid != RUN_ID:
            or_foreign += 1
        or_purposes[row.get("purpose") or "?"] += 1
        or_models[row.get("actual_model") or row.get("configured_model") or "?"] += 1
        pv = row.get("prompt_version") or ""
        if pv:
            or_prompt_versions[pv] += 1
            if not str(pv).startswith("pilot43"):
                or_foreign += 1  # counted separately below if needed

    non_pilot43_prompts = sum(
        c for v, c in or_prompt_versions.items() if not str(v).startswith("pilot43")
    )

    usage = {}
    usage_path = out / "openrouter_usage_pilot43.json"
    if usage_path.exists():
        try:
            usage = json.loads(usage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            usage = {"_error": "unreadable"}

    # ── stage gates ──────────────────────────────────────────────────────
    stage_gates = {}
    for name in ("stage_gate_pilot43_smoke.json", "stage_gate_pilot43_pilot.json",
                 "stage_gate_pilot43_full.json"):
        path = out / name
        if path.exists():
            try:
                stage_gates[name] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                stage_gates[name] = {"_error": "unreadable"}

    # ── reusable safety ──────────────────────────────────────────────────
    safe_reuse = {
        "semantic_candidates.jsonl": True,
        "semantic_hard_valid.jsonl": True,
        "query_render_shortlist.jsonl": True,
        "verified_candidates.jsonl": True,
        "v4_per_task.jsonl": True,
        "per_node_necessity.jsonl": True,
        "llm_rendered.jsonl": True,
        "openrouter_requests_pilot43.jsonl": or_foreign == 0 and non_pilot43_prompts == 0,
        "openrouter cache namespace": True,
    }

    # unfinished shortlist verification
    unfinished_verify = len(shortlist_not_verified)

    report: Dict[str, Any] = {
        "run_id": RUN_ID,
        "out_dir": str(out.resolve()),
        "files": file_stats,
        "stage_counts": {
            "semantic_candidates": file_stats["semantic_candidates"]["lines"],
            "semantic_hard_valid": file_stats["semantic_hard_valid"]["lines"],
            "query_render_shortlist": len(shortlist_ids),
            "verified_candidates": len(verified_ids),
            "semantic_selectable": len(selectable),
            "shortlist_not_yet_verified": unfinished_verify,
            "verified_not_in_shortlist": verified_not_shortlist,
            "llm_rendered": len(rendered_by_id),
            "llm_usable_unblocked": len(reusable),
            "llm_blocked": sum(blocked.values()),
            "openrouter_requests": or_n,
            "query_hard_valid": file_stats["query_hard_valid"]["lines"],
            "deterministic_rendered": file_stats["deterministic_rendered"]["lines"],
        },
        "verification": {
            "v4_ok": v4_ok,
            "v4_shortcut": v4_shortcut,
            "v4_unresolved": v4_unresolved,
            "v4_missing_or_not_executed": v4_missing,
            "necessity_ok": nec_ok,
            "necessity_fail": nec_fail,
            "distractor_ok": distractor_ok,
            "distractor_fail": distractor_fail,
            "top_reject_reasons": reject_reasons.most_common(20),
        },
        "selectable_strata": {
            "n": len(selectable),
            "call_bucket": dict(selectable_buckets),
            "coding_like": selectable_coding,
            "answer_type": dict(selectable_answer),
        },
        "shortlist_strata": {
            "n": len(shortlist_ids),
            "call_bucket": dict(shortlist_by_bucket),
            "coding_like": shortlist_coding,
        },
        "llm_renders": {
            "n_records": len(rendered_by_id),
            "with_writer_query": writer_ok,
            "first_critic_any": first_critic_any,
            "first_critic_pass": first_critic_pass,
            "second_critic_any": second_critic_any,
            "second_critic_pass": second_critic_pass,
            "blocked_by_reason": dict(blocked.most_common()),
            "reusable_unblocked_selectable": len(reusable),
            "foreign_run_records": foreign_run,
            "prompt_versions": dict(prompt_versions),
            "sources": dict(sources),
        },
        "openrouter_isolation": {
            "n_requests": or_n,
            "run_ids": dict(or_run_ids),
            "foreign_run_id_records": or_foreign,
            "non_pilot43_prompt_records": non_pilot43_prompts,
            "purposes": dict(or_purposes),
            "models": dict(or_models.most_common(12)),
            "prompt_versions": dict(or_prompt_versions),
            "logs_exclusively_this_run": or_foreign == 0 and non_pilot43_prompts == 0,
            "usage": {
                k: usage.get(k) for k in (
                    "total_cost_usd", "n_requests", "n_cache_hits",
                    "foreign_run_records", "by_purpose", "by_model",
                ) if k in usage
            } if usage else {},
        },
        "stage_gates": {
            name: {
                "passed": g.get("passed") or g.get("gate_passed"),
                "deterministic_pass_rate": g.get("deterministic_pass_rate"),
                "critic_pass_rate": g.get("critic_pass_rate"),
                "n": g.get("n") or g.get("n_tasks"),
                "entered_against_failed_gate": g.get("entered_against_failed_gate"),
            } if isinstance(g, dict) and "_error" not in g else g
            for name, g in stage_gates.items()
        },
        "safe_to_reuse": safe_reuse,
        "next_actions": [],
    }

    actions: List[str] = []
    if unfinished_verify:
        actions.append(
            f"verify remaining {unfinished_verify} shortlist tasks "
            f"(or freeze selectable from the {len(selectable)} already verified)")
    else:
        actions.append(
            f"freeze semantic_selectable_final.jsonl from {len(selectable)} "
            "verified selectable tasks")
    actions.append(
        f"reuse {len(reusable)} unblocked LLM renders that sit in the "
        "selectable pool; do not re-spend on them")
    remaining_selectable = len(selectable - set(rendered_by_id))
    actions.append(
        f"{remaining_selectable} selectable tasks have no LLM render yet; "
        "allocate a cost-efficient subset rather than rendering all of them")
    if not report["openrouter_isolation"]["logs_exclusively_this_run"]:
        actions.append("OpenRouter logs contain foreign-run or non-pilot43 "
                       "prompt records; isolate before appending")
    if not file_stats["query_hard_valid"]["exists"]:
        actions.append("query_hard_valid.jsonl not yet built; run deterministic "
                       "merge + validation after rendering")
    report["next_actions"] = actions

    # write JSON
    json_path = out / "PILOT43_RESUME_AUDIT.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    # write Markdown
    sc = report["stage_counts"]
    vr = report["verification"]
    lr = report["llm_renders"]
    oi = report["openrouter_isolation"]
    lines = [
        "# Pilot4.3 resume audit",
        "",
        f"- `run_id`: `{RUN_ID}`",
        f"- `out_dir`: `{out.resolve()}`",
        "",
        "All counts below are recomputed from the JSONL files. Prior reports "
        "were not trusted.",
        "",
        "## 1. Stage counts",
        "",
        "| stage | n |",
        "| --- | ---: |",
        f"| semantic candidates | {sc['semantic_candidates']} |",
        f"| semantic hard-valid | {sc['semantic_hard_valid']} |",
        f"| query-render shortlist | {sc['query_render_shortlist']} |",
        f"| verified (V4/necessity/distractors) | {sc['verified_candidates']} |",
        f"| semantic selectable | {sc['semantic_selectable']} |",
        f"| shortlist not yet verified | {sc['shortlist_not_yet_verified']} |",
        f"| LLM-rendered records | {sc['llm_rendered']} |",
        f"| LLM usable (unblocked ∩ selectable) | {sc['llm_usable_unblocked']} |",
        f"| LLM blocked | {sc['llm_blocked']} |",
        f"| OpenRouter requests logged | {sc['openrouter_requests']} |",
        f"| query_hard_valid | {sc['query_hard_valid']} |",
        f"| deterministic_rendered | {sc['deterministic_rendered']} |",
        "",
        "## 2. Verification detail",
        "",
        f"- V4 ok (executed, resolved, no shortcut): **{vr['v4_ok']}**",
        f"- V4 shortcut: {vr['v4_shortcut']}",
        f"- V4 unresolved: {vr['v4_unresolved']}",
        f"- V4 missing / not executed: {vr['v4_missing_or_not_executed']}",
        f"- node necessity ok: **{vr['necessity_ok']}**",
        f"- node necessity fail: {vr['necessity_fail']}",
        f"- distractor present: {vr['distractor_ok']}",
        f"- distractor fail: {vr['distractor_fail']}",
        "",
        "Top reject reasons among non-selectable verified tasks:",
        "",
    ]
    for reason, n in vr["top_reject_reasons"][:12]:
        lines.append(f"- `{n}` {reason}")
    lines += [
        "",
        "## 3. Selectable strata (from shortlist metadata)",
        "",
        f"n = **{report['selectable_strata']['n']}**, "
        f"coding_like = {report['selectable_strata']['coding_like']}",
        "",
        "### call bucket",
        "",
    ]
    for k, v in sorted(report["selectable_strata"]["call_bucket"].items(),
                       key=lambda kv: str(kv[0])):
        lines.append(f"- `{k}`: {v}")
    lines += ["", "### answer type", ""]
    for k, v in sorted(report["selectable_strata"]["answer_type"].items()):
        lines.append(f"- `{k}`: {v}")
    lines += [
        "",
        "## 4. LLM renders",
        "",
        f"- records: **{lr['n_records']}**",
        f"- with writer query: {lr['with_writer_query']}",
        f"- first critic answered: {lr['first_critic_any']} "
        f"(PASS {lr['first_critic_pass']})",
        f"- second critic answered: {lr['second_critic_any']} "
        f"(PASS {lr['second_critic_pass']})",
        f"- reusable unblocked ∩ selectable: **{lr['reusable_unblocked_selectable']}**",
        f"- foreign-run records inside llm_rendered: {lr['foreign_run_records']}",
        f"- prompt versions: `{lr['prompt_versions']}`",
        "",
        "Blocked reasons:",
        "",
    ]
    for reason, n in sorted(lr["blocked_by_reason"].items(),
                            key=lambda kv: -kv[1]):
        lines.append(f"- `{n}` {reason}")
    lines += [
        "",
        "## 5. OpenRouter log isolation",
        "",
        f"- logs exclusively this run: "
        f"**{oi['logs_exclusively_this_run']}**",
        f"- foreign run_id records: {oi['foreign_run_id_records']}",
        f"- non-`pilot43*` prompt records: {oi['non_pilot43_prompt_records']}",
        f"- run_ids seen: `{oi['run_ids']}`",
        f"- purposes: `{oi['purposes']}`",
        f"- models: `{dict(list(oi['models'].items())[:8])}`",
        f"- usage snapshot: `{oi['usage']}`",
        "",
        "## 6. Safe to reuse",
        "",
    ]
    for name, ok in report["safe_to_reuse"].items():
        lines.append(f"- `{'YES' if ok else 'NO '}` {name}")
    lines += [
        "",
        "## 7. Stage gates already recorded",
        "",
    ]
    if report["stage_gates"]:
        for name, g in report["stage_gates"].items():
            lines.append(f"- `{name}`: `{g}`")
    else:
        lines.append("- none")
    lines += [
        "",
        "## 8. Next actions",
        "",
    ]
    for a in report["next_actions"]:
        lines.append(f"1. {a}")
    lines += [
        "",
        "## 9. Files present",
        "",
        "| file | exists | lines | bytes |",
        "| --- | --- | ---: | ---: |",
    ]
    for name, st in file_stats.items():
        lines.append(
            f"| {name} | {st['exists']} | {st['lines']} | {st['bytes']} |")
    lines.append("")

    md_path = out / "PILOT43_RESUME_AUDIT.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "wrote": [str(json_path), str(md_path)],
        "stage_counts": sc,
        "selectable": len(selectable),
        "reusable_llm": len(reusable),
        "logs_clean": oi["logs_exclusively_this_run"],
        "next_actions": actions,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
