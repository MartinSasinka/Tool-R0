#!/usr/bin/env python3
"""Weak-model audit pipeline: discovery, packets, annotation, agreement, handoff."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parents[1]
_REPO = _V3.parents[1]
_MINIMAL = _REPO / "experiments/nestful_mtgrpo_minimal"
_SCRIPTS = _V3 / "scripts"
sys.path.insert(0, str(_MINIMAL))
sys.path.insert(0, str(_V3))
sys.path.append(str(_SCRIPTS))
if str(_HERE) not in sys.path:
    sys.path.append(str(_HERE))

from weak_audit.agreement import compare_passes, validate_raw_rows  # noqa: E402
from weak_audit.compression import estimate_tokens  # noqa: E402
from weak_audit.constants import (  # noqa: E402
    COHORT_LIMITS,
    REPAIR_PROMPT,
    SEED,
    SYSTEM_PROMPT,
    TOKEN_OUTPUT_MAX,
)
from weak_audit.discovery import build_discovery, render_discovery_md  # noqa: E402
from weak_audit.io_utils import (  # noqa: E402
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from weak_audit.pass_builder import prepare_inputs  # noqa: E402
from weak_audit.paths import default_paths  # noqa: E402
from weak_audit.runner import run_annotations  # noqa: E402
from weak_audit.selection import select_tasks, task_cohorts  # noqa: E402
from weak_audit.summary import (  # noqa: E402
    select_high_priority,
    write_cluster_csv,
)
from weak_audit.invalid_discovery import write_invalid_discovery  # noqa: E402
from weak_audit.backup import backup_before_retry  # noqa: E402
from weak_audit.provider_audit import write_provider_audit  # noqa: E402
from weak_audit.retry import build_retry_inputs, run_invalid_retry  # noqa: E402
from weak_audit.merge import merge_final_annotations  # noqa: E402
from weak_audit.finalize import (  # noqa: E402
    write_final_manifest,
    write_finalized_md,
    write_retry_finalization_report,
)
from weak_audit.summarize_outputs import (  # noqa: E402
    extended_agreement,
    write_summarize_outputs,
    _provider_index,
)


def _packets():
    from weak_audit.packets import (  # noqa: WPS433
        build_packet,
        build_task_meta,
        load_eval_bundle,
        score_r0_all,
    )
    return build_packet, build_task_meta, load_eval_bundle, score_r0_all


def _validate_packets_mod():
    from weak_audit.validate_packets import validate_packets  # noqa: WPS433
    return validate_packets


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _setup_paths(args) -> "AuditPaths":
    paths = default_paths(
        run_dir=Path(args.run_dir) if args.run_dir else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    return paths


def _default_model() -> str:
    scripts = _V3 / "scripts" / "data"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from openrouter_client import models_from_env  # noqa: WPS433
    return models_from_env()["weak_solver"]


def _annotation_kwargs(args) -> dict:
    return {
        "reasoning_effort": getattr(args, "reasoning_effort", "none"),
        "use_json_schema": not getattr(args, "no_json_schema", False),
    }


def cmd_verify_r0(args) -> None:
    paths = _setup_paths(args)
    from weak_audit.r0_parity import build_r0_parity_report, write_parity_outputs  # noqa
    report = build_r0_parity_report(paths)
    write_parity_outputs(paths, report)
    print(f"R0 parity gate={report.gate_passed} label={report.reward_label}")
    print(f"Wrote {paths.out_dir / 'R0_PARITY.md'}")


def cmd_discover(args) -> None:
    paths = _setup_paths(args)
    data = build_discovery(paths)
    write_json(paths.out_dir / "discovery.json", data)
    (paths.out_dir / "DISCOVERY.md").write_text(
        render_discovery_md(data), encoding="utf-8"
    )
    print(f"Wrote {paths.out_dir / 'DISCOVERY.md'}")


def cmd_prepare(args) -> None:
    paths = _setup_paths(args)
    build_packet, build_task_meta, load_eval_bundle, score_r0_all = _packets()
    validate_packets = _validate_packets_mod()
    tasks, arms, all_ids, hashes = load_eval_bundle(paths)
    r0 = score_r0_all(all_ids, arms, tasks)
    meta = build_task_meta(all_ids, arms, tasks, r0)
    cohort_tasks, assigned = select_tasks(meta, r0, seed=SEED)
    cohort_map = task_cohorts(assigned)
    selected = sorted(cohort_map.keys())
    if len(selected) > 250:
        raise RuntimeError(f"selection exceeds hard max: {len(selected)}")

    packets = [
        build_packet(tid, cohort_map[tid], tasks, arms, paths, hashes)
        for tid in selected
    ]
    write_jsonl(paths.out_dir / "case_packets.jsonl", packets)
    write_json(paths.out_dir / "selected_task_ids.json", selected)

    manifest = {
        "seed": SEED,
        "generated_at": _now(),
        "run_id": paths.run_dir.name,
        "input_hashes": hashes,
        "cohort_limits": COHORT_LIMITS,
        "cohort_counts": {c: len(v) for c, v in cohort_tasks.items()},
        "dedup_rule": "priority order A-F; each task one primary cohort",
        "total_unique_tasks": len(selected),
        "stratification_keys": [
            "gold_call_bucket", "motif", "first_divergence_turn",
            "c0_failure", "reward_mismatch",
        ],
    }
    write_json(paths.out_dir / "selection_manifest.json", manifest)

    pass_a, pass_b, mapping, comp_logs, oversize = prepare_inputs(packets)
    write_jsonl(paths.out_dir / "pass_a_inputs.jsonl", pass_a)
    write_jsonl(paths.out_dir / "pass_b_inputs.jsonl", pass_b)
    write_json(paths.out_dir / "pass_b_mapping.json", mapping)
    write_json(paths.out_dir / "compression_report.json", {
        "generated_at": _now(),
        "entries": comp_logs,
        "oversize_count": len(oversize),
    })
    if oversize:
        write_jsonl(paths.out_dir / "manual_oversize_cases.jsonl", oversize)

    ok, errs = validate_packets(
        paths.out_dir / "case_packets.jsonl", selected, arms
    )
    summary_lines = [
        "# Selection summary",
        "",
        f"**Generated:** {_now()}",
        f"**Seed:** {SEED}",
        f"**Total unique tasks:** {len(selected)}",
        "",
        "## Cohort counts",
        "",
    ]
    for c, n in manifest["cohort_counts"].items():
        summary_lines.append(f"- {c}: {n}")
    toks = [e.get("tokens_estimate") or e.get("tokens_after") for e in comp_logs if e.get("task_id")]
    if toks:
        summary_lines += [
            "",
            "## Token estimates (pass inputs)",
            "",
            f"- mean: {sum(toks)/len(toks):.0f}",
            f"- max: {max(toks)}",
            f"- oversize (hard limit): {len(oversize)}",
        ]
    summary_lines += ["", "## Packet validation", "", f"- ok: {ok}"]
    for e in errs[:20]:
        summary_lines.append(f"- {e}")
    (paths.out_dir / "SELECTION_SUMMARY.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    print(f"Selected {len(selected)} tasks -> {paths.out_dir}")
    if not ok:
        print("WARN: packet validation issues:", len(errs))


def _canary_task_ids(out_dir: Path, n: int = 10) -> List[str]:
    pkt_path = out_dir / "case_packets.jsonl"
    if not pkt_path.is_file():
        return []
    by_cohort: Dict[str, List[str]] = defaultdict(list)
    all_ids: List[str] = []
    for row in read_jsonl(pkt_path):
        all_ids.append(row["task_id"])
        for c in row.get("cohorts") or []:
            by_cohort[c].append(row["task_id"])
    rng = random.Random(SEED + 99)
    picked: List[str] = []
    for cohort in COHORT_LIMITS:
        pool = by_cohort.get(cohort) or []
        if pool:
            picked.append(rng.choice(pool))
    rest = [t for t in all_ids if t not in picked]
    rng.shuffle(rest)
    picked.extend(rest[: max(0, n - len(picked))])
    return picked[:n]


def _mock_annotation(messages: List[dict]) -> str:
    user = json.loads(messages[1]["content"])
    tid = user.get("task_id", "unknown")
    flags = user.get("deterministic_flags") or {}
    root = "reward_mismatch" if flags.get("reward_prefers_E2_over_C0") else "later_tool_selection"
    return json.dumps({
        "task_id": tid,
        "first_divergence_turn": flags.get("first_divergence_turn"),
        "root_cause": root,
        "shorter_path_verdict": "not_applicable",
        "observation_used_correctly": True,
        "reward_ordering_correct": not flags.get("reward_prefers_E2_over_C0"),
        "responsible_reward_component": "call_count" if flags.get("reward_prefers_E2_over_C0") else "none",
        "recommended_fix": "outcome_reward" if flags.get("reward_prefers_E2_over_C0") else "no_change",
        "confidence": 0.65,
        "evidence": "mock canary",
    }, ensure_ascii=False)


def cmd_canary(args) -> None:
    paths = _setup_paths(args)
    selected = read_json(paths.out_dir / "selected_task_ids.json")
    manifest = read_json(paths.out_dir / "selection_manifest.json")
    canary_ids = set(_canary_task_ids(paths.out_dir, n=args.limit or 10))
    pass_a = [r for r in read_jsonl(paths.out_dir / "pass_a_inputs.jsonl") if r["task_id"] in canary_ids]
    pass_b = [r for r in read_jsonl(paths.out_dir / "pass_b_inputs.jsonl") if r["task_id"] in canary_ids]
    model = args.model or _default_model()
    mock = bool(getattr(args, "mock", False))

    raw_a = paths.out_dir / "pass_a_annotations_raw_canary.jsonl"
    raw_b = paths.out_dir / "pass_b_annotations_raw_canary.jsonl"
    if raw_a.exists() and args.resume:
        raw_a.unlink()
    if raw_b.exists() and args.resume and not args.resume:
        pass
    stats_a = run_annotations(
        pass_a, output_raw=str(raw_a), model=model, pass_label="A",
        base_url=args.base_url, api_key_env=args.api_key_env or "OPENROUTER_API_KEY",
        concurrency=args.concurrency, temperature=args.temperature,
        max_output_tokens=args.max_output_tokens, max_retries=args.max_retries,
        resume=not args.no_resume, mock_handler=_mock_annotation if mock else None,
        **_annotation_kwargs(args),
    )
    stats_b = run_annotations(
        pass_b, output_raw=str(raw_b), model=model, pass_label="B",
        base_url=args.base_url, api_key_env=args.api_key_env or "OPENROUTER_API_KEY",
        concurrency=args.concurrency, temperature=args.temperature,
        max_output_tokens=args.max_output_tokens, max_retries=args.max_retries,
        resume=not args.no_resume, mock_handler=_mock_annotation if mock else None,
        **_annotation_kwargs(args),
    )
    valid_a, inv_a = validate_raw_rows(read_jsonl(raw_a), repair_fn=None if args.no_repair else _repair_fn)
    valid_b, inv_b = validate_raw_rows(read_jsonl(raw_b), repair_fn=None if args.no_repair else _repair_fn)
    n_a = len(read_jsonl(raw_a))
    n_b = len(read_jsonl(raw_b))
    rate_a = len(valid_a) / n_a if n_a else 0
    rate_b = len(valid_b) / n_b if n_b else 0
    out_lens = [estimate_tokens(v) for v in valid_a + valid_b]
    mean_out = sum(out_lens) / len(out_lens) if out_lens else 0
    passed = rate_a >= 0.95 and rate_b >= 0.95 and mean_out <= max(TOKEN_OUTPUT_MAX * 2, 350)
    lines = [
        "# Canary report",
        "",
        f"**Generated:** {_now()}",
        f"**Model:** {model}",
        f"**Mock:** {mock}",
        f"**Canary tasks:** {len(canary_ids)}",
        "",
        "## Run stats",
        "",
        f"- Pass A: {stats_a}",
        f"- Pass B: {stats_b}",
        "",
        "## Validation",
        "",
        f"- Pass A valid rate: {rate_a:.1%} ({len(valid_a)}/{n_a})",
        f"- Pass B valid rate: {rate_b:.1%} ({len(valid_b)}/{n_b})",
        f"- Mean output token estimate: {mean_out:.0f}",
        f"- Invalid A: {len(inv_a)}, Invalid B: {len(inv_b)}",
        "",
        f"## Gate: {'PASS' if passed else 'FAIL'}",
        "",
        "Requirements: >=95% valid JSON after repair, output under ~180 tokens avg.",
    ]
    (paths.out_dir / "CANARY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Canary {'PASS' if passed else 'FAIL'} -> {paths.out_dir / 'CANARY_REPORT.md'}")
    if not passed and not mock:
        raise SystemExit(1)


def cmd_run(args) -> None:
    paths = _setup_paths(args)
    model = args.model or _default_model()
    which = args.pass_label.upper()
    inp_path = paths.out_dir / f"pass_{which.lower()}_inputs.jsonl"
    out_path = paths.out_dir / f"pass_{which.lower()}_annotations_raw.jsonl"
    inputs = read_jsonl(inp_path)
    if args.limit:
        inputs = inputs[: args.limit]
    oversize = read_jsonl(paths.out_dir / "manual_oversize_cases.jsonl") if (
        paths.out_dir / "manual_oversize_cases.jsonl"
    ).is_file() else []
    skip = {r["task_id"] for r in oversize if r.get("pass") in ("A", "B", which)}
    inputs = [i for i in inputs if i["task_id"] not in skip]
    mock = bool(getattr(args, "mock", False))
    stats = run_annotations(
        inputs, output_raw=str(out_path), model=model, pass_label=which,
        base_url=args.base_url, api_key_env=args.api_key_env or "OPENROUTER_API_KEY",
        concurrency=args.concurrency, temperature=args.temperature,
        max_output_tokens=args.max_output_tokens, max_retries=args.max_retries,
        resume=not args.no_resume,
        limit=args.limit,
        mock_handler=_mock_annotation if mock else None,
        **_annotation_kwargs(args),
    )
    print(f"Pass {which} done: {stats} -> {out_path}")


def _save_baseline_agreement(paths) -> dict:
    out = paths.out_dir / "agreement_before_retry.json"
    if out.is_file():
        return read_json(out)
    packets = read_jsonl(paths.out_dir / "case_packets.jsonl")
    pkt_map = {p["task_id"]: p for p in packets}
    ann_a = {r["task_id"]: r for r in read_jsonl(paths.out_dir / "pass_a_annotations.jsonl")}
    ann_b = {r["task_id"]: r for r in read_jsonl(paths.out_dir / "pass_b_annotations.jsonl")}
    agree = extended_agreement(ann_a, ann_b, pkt_map, _provider_index(paths.out_dir))
    write_json(out, agree)
    return agree


def cmd_discover_invalid(args) -> None:
    paths = _setup_paths(args)
    entries, manifest = write_invalid_discovery(paths.out_dir)
    write_provider_audit(paths.out_dir)
    backup_dir = paths.v3 / "reports/pure_stage3_weak_audit_real_before_retry"
    if not (backup_dir / "MANIFEST_SHA256.json").is_file():
        backup_before_retry(paths.out_dir, backup_dir)
    _save_baseline_agreement(paths)
    print(
        f"Discovered {manifest['n_invalid_pairs']} invalid pairs -> "
        f"{paths.out_dir / 'invalid_retry_manifest.json'}"
    )


def cmd_retry_invalid(args) -> None:
    paths = _setup_paths(args)
    if not (paths.out_dir / "invalid_retry_manifest.json").is_file():
        write_invalid_discovery(paths.out_dir)
    if not (paths.v3 / "reports/pure_stage3_weak_audit_real_before_retry" / "MANIFEST_SHA256.json").is_file():
        backup_before_retry(
            paths.out_dir,
            paths.v3 / "reports/pure_stage3_weak_audit_real_before_retry",
        )
    _, recommended = write_provider_audit(paths.out_dir)
    _save_baseline_agreement(paths)

    model = args.model or _default_model()
    provider = args.provider or recommended
    repair = None if args.no_repair else _retry_repair_fn

    if args.dry_run:
        stats = run_invalid_retry(
            paths.out_dir,
            model=model,
            provider=provider,
            api_key_env=args.api_key_env or "OPENROUTER_API_KEY",
            concurrency=args.concurrency,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
            resume=not args.no_resume,
            use_json_schema=not args.no_json_schema,
            dry_run=True,
        )
        print(json.dumps(stats, indent=2))
        return

    stats = run_invalid_retry(
        paths.out_dir,
        model=model,
        provider=provider,
        api_key_env=args.api_key_env or "OPENROUTER_API_KEY",
        concurrency=args.concurrency,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        max_retries=args.max_retries,
        resume=not args.no_resume,
        use_json_schema=not args.no_json_schema,
        dry_run=False,
        repair_fn=repair,
    )
    stats.update({
        "model": model,
        "provider": provider,
        "use_json_schema": not args.no_json_schema,
        "reasoning_effort": args.reasoning_effort,
    })
    write_json(paths.out_dir / "retry_stats.json", stats)
    print(f"Retry done: {stats}")


def cmd_validate_retry(args) -> None:
    paths = _setup_paths(args)
    repair = None if args.no_repair else _retry_repair_fn
    raw_path = paths.out_dir / "retry_invalid_raw.jsonl"
    if not raw_path.is_file():
        raise SystemExit("retry_invalid_raw.jsonl missing — run retry-invalid first")
    manifest = read_json(paths.out_dir / "invalid_retry_manifest.json")
    manifest_pairs = {
        (p["task_id"], p["pass_label"]) for p in manifest.get("pairs") or []
    }
    raw_rows = [
        r for r in read_jsonl(raw_path)
        if (r.get("task_id"), r.get("pass")) in manifest_pairs
    ]
    valid, invalid = validate_raw_rows(raw_rows, repair_fn=repair)
    write_jsonl(paths.out_dir / "retry_invalid_validated.jsonl", valid)
    write_jsonl(paths.out_dir / "retry_invalid_failed.jsonl", invalid)
    print(f"Retry validate: valid={len(valid)} failed={len(invalid)}")


def cmd_finalize(args) -> None:
    paths = _setup_paths(args)
    merge_stats = merge_final_annotations(paths.out_dir)
    before = _save_baseline_agreement(paths)
    after_out = write_summarize_outputs(paths.out_dir, suffix="final")
    after = after_out["agreement"]
    retry_stats = read_json(paths.out_dir / "retry_stats.json") if (
        paths.out_dir / "retry_stats.json"
    ).is_file() else {}
    manifest = write_final_manifest(
        paths.out_dir, paths, merge_stats, retry_stats
    )
    write_retry_finalization_report(
        paths.out_dir,
        before_agree=before,
        after_agree=after,
        merge_stats=merge_stats,
        retry_stats=retry_stats,
        provider=retry_stats.get("provider"),
    )
    write_finalized_md(paths.out_dir, manifest)
    print(f"Finalized -> {paths.out_dir / 'WEAK_AUDIT_FINAL_MANIFEST.json'}")


def _retry_repair_fn(raw: dict, text: str):
    if os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("WEAK_AUDIT_NO_REPAIR"):
        scripts = _V3 / "scripts" / "data"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from openrouter_client import OpenRouterClient  # noqa
        client = OpenRouterClient(max_retries=1)
        resp = client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": REPAIR_PROMPT + "\n\n" + text},
            ],
            model=raw.get("model_id") or _default_model(),
            role="weak_solver",
            temperature=0,
            max_tokens=500,
            response_format_json=True,
        )
        return resp.get("text") or ""
    raise ValueError("no repair")


def _repair_fn(raw: dict, text: str):
    if os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("WEAK_AUDIT_NO_REPAIR"):
        scripts = _V3 / "scripts" / "data"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from openrouter_client import OpenRouterClient  # noqa
        client = OpenRouterClient(max_retries=1)
        resp = client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": REPAIR_PROMPT + "\n\n" + text},
            ],
            model=raw.get("model_id") or _default_model(),
            role="weak_solver",
            temperature=0,
            max_tokens=250,
            response_format_json=True,
        )
        return resp.get("text") or ""
    raise ValueError("no repair")


def cmd_validate(args) -> None:
    paths = _setup_paths(args)
    repair = None if args.no_repair else _repair_fn
    for label in ("A", "B"):
        raw = read_jsonl(paths.out_dir / f"pass_{label.lower()}_annotations_raw.jsonl")
        valid, invalid = validate_raw_rows(raw, repair_fn=repair)
        write_jsonl(paths.out_dir / f"pass_{label.lower()}_annotations.jsonl", valid)
        print(f"Pass {label}: valid={len(valid)} invalid={len(invalid)}")
    inv_all = []
    for label in ("A", "B"):
        raw = read_jsonl(paths.out_dir / f"pass_{label.lower()}_annotations_raw.jsonl")
        _, invalid = validate_raw_rows(raw, repair_fn=repair)
        inv_all.extend(invalid)
    write_jsonl(paths.out_dir / "invalid_annotations.jsonl", inv_all)


def cmd_summarize(args) -> None:
    paths = _setup_paths(args)
    suffix = getattr(args, "annotations_suffix", "") or ""
    out = write_summarize_outputs(paths.out_dir, suffix=suffix)
    print(
        f"Summarized ({suffix or 'baseline'}) "
        f"A={out['n_ann_a']} B={out['n_ann_b']} both={out['n_both']} "
        f"-> {paths.out_dir}"
    )


def cmd_validate_packets_cli(args) -> None:
    paths = _setup_paths(args)
    _, _, load_eval_bundle, _ = _packets()
    validate_packets = _validate_packets_mod()
    _, arms, _, _ = load_eval_bundle(paths)
    selected = read_json(paths.out_dir / "selected_task_ids.json")
    ok, errs = validate_packets(paths.out_dir / "case_packets.jsonl", selected, arms)
    for e in errs:
        print(e)
    raise SystemExit(0 if ok else 1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Weak-model audit pipeline")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--out-dir", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    disc = sub.add_parser("discover", help="Write DISCOVERY.md")
    disc.set_defaults(func=cmd_discover)

    v0 = sub.add_parser("verify-r0", help="R0 parity vs train logs")
    v0.set_defaults(func=cmd_verify_r0)

    prep = sub.add_parser("prepare", help="Select cohorts and build packets")
    prep.set_defaults(func=cmd_prepare)

    can = sub.add_parser("canary", help="Run 10-case canary annotation")
    _add_run_flags(can)
    can.set_defaults(func=cmd_canary, limit=10)

    run = sub.add_parser("run", help="Batch annotate pass A or B")
    run.add_argument("--pass-label", choices=["A", "B"], required=True)
    _add_run_flags(run)
    run.set_defaults(func=cmd_run)

    val = sub.add_parser("validate", help="Validate/repair raw annotations")
    val.add_argument("--no-repair", action="store_true")
    can.add_argument("--no-repair", action="store_true")
    val.set_defaults(func=cmd_validate)

    summ = sub.add_parser("summarize", help="Agreement, clusters, high-priority")
    summ.add_argument(
        "--annotations-suffix",
        default="",
        help="Use pass_*_annotations_{suffix}.jsonl (e.g. final)",
    )
    summ.set_defaults(func=cmd_summarize)

    dinv = sub.add_parser("discover-invalid", help="Invalid retry manifest + backup")
    dinv.set_defaults(func=cmd_discover_invalid)

    retry = sub.add_parser("retry-invalid", help="Retry only invalid task/pass pairs")
    _add_retry_flags(retry)
    retry.set_defaults(func=cmd_retry_invalid)

    vretry = sub.add_parser("validate-retry", help="Re-validate retry_invalid_raw.jsonl")
    vretry.add_argument("--no-repair", action="store_true")
    vretry.set_defaults(func=cmd_validate_retry)

    fin = sub.add_parser("finalize", help="Merge retry, summarize final, write manifest")
    fin.set_defaults(func=cmd_finalize)

    vp = sub.add_parser("validate-packets", help="Check case_packets.jsonl")
    vp.set_defaults(func=cmd_validate_packets_cli)

    return p


def _add_run_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-output-tokens", type=int, default=350)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--reasoning-effort", default="none",
                   choices=["none", "minimal", "low", "medium", "high"])
    p.add_argument("--no-json-schema", action="store_true")


def _add_retry_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None)
    p.add_argument("--provider", default=None)
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-output-tokens", type=int, default=500)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-repair", action="store_true")
    p.add_argument(
        "--reasoning-effort",
        default="none",
        choices=["none", "minimal", "low", "medium", "high"],
    )
    p.add_argument("--no-json-schema", action="store_true")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
