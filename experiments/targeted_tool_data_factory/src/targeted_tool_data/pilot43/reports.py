"""Final reports: implementation report, and the Pilot4.2 vs Pilot4.3 comparison.

Every number quoted here is read from an artifact that was itself computed from the
exported records; nothing is recomputed from an in-memory object that only exists
during a build. That is deliberate: a report which can only be produced by the same
process that produced the dataset cannot be checked afterwards.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import RUN_ID, SCHEMA_VERSION, TRAINING_READINESS_KEYS
from .blueprints import all_blueprints
from .ops import build_ops

IMPL_MD = "PILOT43_IMPLEMENTATION_REPORT.md"
IMPL_JSON = "PILOT43_IMPLEMENTATION_REPORT.json"
COMPARE_MD = "PILOT42_VS_PILOT43_AUDIT.md"
COMPARE_CSV = "PILOT42_VS_PILOT43_METRICS.csv"

#: The one interpretation the reports are allowed to make (spec 39).
CLAIM = ("Pilot4.3 meets the defined offline, human-review and model-relative "
         "criteria for starting a controlled GRPO experiment. It does not follow "
         "that Pilot4.3 improves NESTFUL; only a backend-matched NESTFUL "
         "evaluation after training can establish transfer.")


def _json(out_dir: Path, name: str) -> Dict[str, Any]:
    path = out_dir / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _pct(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value) * 100:.2f} %"
    return str(value)


def implementation_report(out_dir: Path, *, stages: Optional[Dict[str, Any]] = None
                          ) -> Dict[str, Any]:
    quality = _json(out_dir, "PILOT43_DATA_QUALITY_REPORT.json")
    selection = _json(out_dir, "selection_report.json")
    audit = _json(out_dir, "PILOT43_INDEPENDENT_AUDIT.json")
    human = _json(out_dir, "human_audit_results.json")
    probe = _json(out_dir, "model_probe_report.json")
    freeze = _json(out_dir, "freeze_manifest.json")
    cells = _json(out_dir, "generation_cells_v3.json")
    profile = _json(out_dir, "target_profile_v3.json")
    or_usage = _json(out_dir, "openrouter_usage_pilot43.json")
    p42 = _json(out_dir, "PILOT42_VS_PILOT43_METRICS.json")

    readiness = (quality.get("readiness") or {}).get("statuses") or {
        k: False for k in TRAINING_READINESS_KEYS}
    evidence = (quality.get("readiness") or {}).get("evidence") or {}
    metrics = (quality.get("metrics") or {}).get("train_master") or {}
    tier_metrics = (quality.get("metrics") or {}).get("tiers") or {}

    payload = {
        "run_id": RUN_ID,
        "schema_version": SCHEMA_VERSION,
        "dataset_version": "Pilot4.3",
        "interpretation": CLAIM,
        "statuses": readiness,
        "status_evidence": evidence,
        "blocking_failures": quality.get("blocking_failures", []),
        "advisory_failures": quality.get("advisory_failures", []),
        "sizes": {
            "tiers": {k: v.get("selected") for k, v in
                      (selection.get("tiers") or {}).items()},
            "train_master": selection.get("train_master"),
            "heldout": selection.get("heldout"),
            "heldout_total": selection.get("heldout_total"),
            "reserve": selection.get("reserve"),
            "selected_total": selection.get("selected_total"),
        },
        "registry": {
            "workflow_families": len(all_blueprints()),
            "primitives_registered": len(build_ops()),
            "primitives_used": metrics.get("actual_primitives_used"),
            "capability_families_used": metrics.get("actual_capability_families"),
            "coding_families_used": metrics.get("coding_capability_families"),
        },
        "train_master_metrics": metrics,
        "tier_metrics": tier_metrics,
        "generation_cells": {k: v for k, v in cells.items() if k != "cells"},
        "split": selection.get("split_overlap"),
        "independent_audit": {k: v for k, v in audit.items()
                              if not k.startswith("_")
                              and k in ("verdict", "INDEPENDENT_AUDIT_PASSED",
                                        "deficits", "counts",
                                        "n_records_audited", "n_unique_tasks",
                                        "disagreements", "independence")},
        "human_audit": {k: v for k, v in human.items() if k != "by_stratum"},
        "model_probe": {k: v for k, v in probe.items() if k != "per_cell"},
        "openrouter": {
            "usage": {k: v for k, v in or_usage.items()
                      if k in ("totals", "run_id", "foreign_run_records",
                               "config")},
            "stage_gates": {stage: _json(out_dir,
                                         f"stage_gate_pilot43_{stage}.json")
                            for stage in ("smoke", "pilot", "full")},
        },
        "target_profile": {k: v for k, v in profile.items()
                           if k in ("n_rows", "profile_hash", "sources")},
        "reproducibility": {k: v for k, v in freeze.items()
                            if k not in ("artifact_hashes", "input_hashes",
                                         "source_snapshot")},
        "n_input_hashes": len(freeze.get("input_hashes") or {}),
        "n_artifact_hashes": len(freeze.get("artifact_hashes") or {}),
        "stages": stages or {},
        "pilot42_comparison": p42,
        "TRAINING_READY": bool(readiness.get("TRAINING_READY")),
    }
    (out_dir / IMPL_JSON).write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / IMPL_MD).write_text(_markdown(out_dir, payload), encoding="utf-8")
    return payload


# ── markdown ─────────────────────────────────────────────────────────────
def _status_table(statuses: Dict[str, Any],
                  evidence: Dict[str, Any]) -> List[str]:
    rows = ["| status | value | evidence |", "| --- | --- | --- |"]
    for key in TRAINING_READINESS_KEYS:
        paths = ", ".join(f"`{p}`" for p in (evidence.get(key) or []))
        rows.append(f"| `{key}` | **{bool(statuses.get(key))}** | {paths} |")
    return rows


def _dist_table(title: str, dist: Dict[str, Any]) -> List[str]:
    if not dist:
        return []
    out = [f"{title}", "", "| key | share |", "| --- | --- |"]
    out += [f"| `{k}` | {_pct(v)} |" for k, v in dist.items()]
    out.append("")
    return out


def _markdown(out_dir: Path, p: Dict[str, Any]) -> str:
    m = p["train_master_metrics"]
    tiers = p["tier_metrics"]
    core = tiers.get("PROFILE_CORE") or {}
    lh = tiers.get("LONG_HORIZON_ENRICHMENT") or {}
    cap = tiers.get("CAPABILITY_ENRICHMENT") or {}
    fails = p["blocking_failures"]
    lines: List[str] = []
    add = lines.append

    add("# Pilot4.3 implementation report")
    add("")
    add(f"- `dataset_version`: Pilot4.3")
    add(f"- `run_id`: `{RUN_ID}`")
    add(f"- `schema_version`: `{SCHEMA_VERSION}`")
    add("")
    add("## 1. Executive summary")
    add("")
    lines.extend(_status_table(p["statuses"], p["status_evidence"]))
    add("")
    add(f"`TRAINING_READY` is **{p['TRAINING_READY']}**. "
        f"{len(fails)} blocking acceptance checks are unmet.")
    add("")
    add(f"> {CLAIM}")
    add("")
    if fails:
        add("Unmet blocking criteria:")
        add("")
        add("| check | requirement | observed |")
        add("| --- | --- | --- |")
        for c in fails:
            add(f"| `{c['id']}` | {c['requirement']} | `{c['observed']}` |")
        add("")

    add("## 2. Pilot4.2 root causes")
    add("")
    add("See `PILOT42_VS_PILOT43_AUDIT.md` for the recomputed Pilot4.2 numbers and "
        "the mechanism Pilot4.3 changed for each of the seventeen defects.")
    add("")

    add("## 3. Dataset composition")
    add("")
    add("| split | tasks |")
    add("| --- | --- |")
    for tier, count in (p["sizes"]["tiers"] or {}).items():
        add(f"| {tier} | {count} |")
    add(f"| TRAIN_MASTER | {p['sizes']['train_master']} |")
    for part, count in (p["sizes"]["heldout"] or {}).items():
        add(f"| heldout/{part} | {count} |")
    add(f"| HELDOUT total | {p['sizes']['heldout_total']} |")
    add(f"| RESERVE | {p['sizes']['reserve']} |")
    add(f"| SELECTED total | {p['sizes']['selected_total']} |")
    add("")

    add("## 4. Registry and coverage")
    add("")
    reg = p["registry"]
    add(f"- workflow families: {reg['workflow_families']}")
    add(f"- primitives registered: {reg['primitives_registered']}")
    add(f"- primitives actually used in gold calls: {reg['primitives_used']}")
    add(f"- capability families used: {reg['capability_families_used']}")
    add(f"- generic/coding families used: {reg['coding_families_used']}")
    add(f"- generic/coding task share: {_pct(m.get('coding_task_share'))}")
    add(f"- generic/coding gold-call share: {_pct(m.get('coding_call_share'))}")
    add(f"- max exact primitive-sequence share: "
        f"{_pct(m.get('max_exact_sequence_share'))}")
    add(f"- top-10 exact sequence share: "
        f"{_pct(m.get('top10_exact_sequence_share'))}")
    add("")

    add("## 5. Call-count distribution")
    add("")
    add("| calls | PROFILE_CORE | TRAIN_MASTER | LONG_HORIZON |")
    add("| --- | --- | --- | --- |")
    for bucket in ("2", "3", "4", "5", "6+"):
        add(f"| {bucket} | "
            f"{_pct((core.get('call_bucket_distribution') or {}).get(bucket))} | "
            f"{_pct((m.get('call_bucket_distribution') or {}).get(bucket))} | "
            f"{_pct((lh.get('call_bucket_distribution') or {}).get(bucket))} |")
    add("")
    add(f"PROFILE_CORE 6+ share {_pct(core.get('six_plus_share'))}; "
        f"TRAIN_MASTER 6+ share {_pct(m.get('six_plus_share'))} (enrichment, not "
        f"a profile match); LONG_HORIZON 6+ share {_pct(lh.get('six_plus_share'))}.")
    add("")
    lines.extend(_dist_table("### Exact call counts in TRAIN_MASTER",
                             m.get("call_count_distribution") or {}))

    add("## 6. Structure")
    add("")
    add(f"- distinct actual pattern families: {m.get('distinct_patterns')}")
    add(f"- distinct pattern families among 6+ tasks: "
        f"{m.get('distinct_patterns_6plus')}")
    add("")
    lines.extend(_dist_table("### Actual pattern distribution",
                             m.get("pattern_distribution") or {}))

    add("## 7. Answers and boolean balance")
    add("")
    lines.extend(_dist_table("### Answer types",
                             m.get("answer_type_distribution") or {}))
    add(f"- boolean tasks: {m.get('boolean_count')}, True share "
        f"{_pct(m.get('boolean_true_share'))}")
    add(f"- string/list/object share in CAPABILITY_ENRICHMENT: "
        f"{_pct(cap.get('structured_answer_share'))}")
    add("")

    add("## 8. Queries")
    add("")
    lines.extend(_dist_table("### Actual (classified) query modes",
                             m.get("query_mode_distribution") or {}))
    div = m.get("diversity") or {}
    add(f"- exact duplicate rate: {div.get('exact_duplicate_rate')}")
    add(f"- distinct lexical skeletons: {div.get('distinct_skeletons')}")
    add(f"- distinct intent templates: {div.get('distinct_intent_templates')}")
    add(f"- max intent-template share: {_pct(div.get('max_intent_share'))}")
    add(f"- LLM-written share: {_pct(m.get('llm_query_share'))}")
    add(f"- first-critic coverage of LLM queries: {_pct(m.get('critic_coverage'))}")
    add("")

    add("## 9. V4 and node necessity")
    add("")
    add(f"- V4 coverage: {_pct(m.get('v4_coverage'))}, skipped "
        f"{m.get('v4_skipped')}, shortcuts {m.get('v4_shortcuts')}, unresolved "
        f"{m.get('v4_unresolved')}")
    add(f"- node necessity coverage: {_pct(m.get('node_necessity_coverage'))}, "
        f"nodes checked {m.get('nodes_checked')}, unnecessary "
        f"{m.get('unnecessary_gold_nodes')}")
    add("")

    add("## 10. Split integrity")
    add("")
    for key, value in (p.get("split") or {}).items():
        if key in ("violations", "train_surface_tracks"):
            continue
        add(f"- `{key}`: {value}")
    add("")

    add("## 11. Independent audit")
    add("")
    ia = p.get("independent_audit") or {}
    if ia:
        add(f"- verdict: **{ia.get('verdict')}**, "
            f"`INDEPENDENT_AUDIT_PASSED={ia.get('INDEPENDENT_AUDIT_PASSED')}`")
        add(f"- records audited: {ia.get('n_records_audited')}, unique tasks: "
            f"{ia.get('n_unique_tasks')}")
        for item in (ia.get("deficits") or [])[:20]:
            add(f"- deficit: {item}")
    else:
        add("Not run.")
    add("")

    add("## 12. Human audit")
    add("")
    ha = p.get("human_audit") or {}
    if ha.get("n_tasks_rated"):
        add(f"- tasks rated: {ha['n_tasks_rated']} by "
            f"{len(ha.get('reviewers') or {})} reviewers")
        add(f"- thresholds met: **{ha.get('thresholds_met')}**")
        for key, value in (ha.get("observed") or {}).items():
            add(f"- `{key}`: {value}")
    else:
        add(f"Package prepared, ratings not imported: "
            f"`{ha.get('reason', 'no ratings')}`. `HUMAN_VALIDATED=false`.")
        add("")
        add(f"To complete: `{ha.get('next_command', '')}`")
    add("")

    add("## 13. Model-relative GRPO-signal probe")
    add("")
    mp = p.get("model_probe") or {}
    if mp.get("executed"):
        obs = mp.get("observed") or {}
        add(f"- groups: {obs.get('n_groups')}")
        add(f"- effective-group rate: {_pct(obs.get('effective_group_rate'))}")
        add(f"- dead-group rate: {_pct(obs.get('dead_group_rate'))}")
        add(f"- thresholds met: **{mp.get('thresholds_met')}**")
    else:
        add(f"Not executed: `{mp.get('reason', 'no inference backend')}`. "
            f"`GRPO_SIGNAL_READY=false`.")
        add("")
        add(f"To complete: `{mp.get('next_command', '')}`")
    add("")

    add("## 14. Reproducibility")
    add("")
    repro = p.get("reproducibility") or {}
    git = repro.get("git") or {}
    add(f"- commit: `{git.get('commit', '')[:12]}` on `{git.get('branch')}`, "
        f"dirty: {git.get('dirty')}")
    add(f"- input hashes: {p['n_input_hashes']}")
    add(f"- artifact hashes: {p['n_artifact_hashes']}")
    add(f"- workflow registry hash: `{repro.get('workflow_registry_hash', '')[:16]}`")
    add(f"- primitive registry hash: `{repro.get('primitive_registry_hash', '')[:16]}`")
    add(f"- ordered sample ids hash: "
        f"`{str(repro.get('ordered_sample_ids_sha256', ''))[:16]}` "
        f"({repro.get('n_ordered_sample_ids')} ids)")
    snapshot = _json(out_dir, "SOURCE_TREE_MANIFEST.json")
    if snapshot:
        add(f"- {snapshot.get('reproduction_note')}")
    add("")

    add("## 15. Known limitations")
    add("")
    for item in _limitations(p):
        add(f"- {item}")
    add("")

    add("## 16. Reproduction commands")
    add("")
    add("```powershell")
    for cmd in REPRODUCTION_COMMANDS:
        add(cmd)
    add("```")
    return "\n".join(lines) + "\n"


def _limitations(p: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    statuses = p["statuses"]
    if not statuses.get("LLM_VALIDATED"):
        out.append("`LLM_VALIDATED=false`: not every selected implicit-mode query "
                   "was written and criticised by the OpenRouter layer, so the "
                   "affected records carry `query_source=deterministic_fallback`.")
    if not statuses.get("HUMAN_VALIDATED"):
        out.append("`HUMAN_VALIDATED=false`: the audit package exists but no "
                   "reviewer ratings have been imported.")
    if not statuses.get("GRPO_SIGNAL_READY"):
        out.append("`GRPO_SIGNAL_READY=false`: the model-relative probe needs a "
                   "reachable base-model backend.")
    if p["blocking_failures"]:
        out.append(f"{len(p['blocking_failures'])} blocking acceptance checks are "
                   "unmet; see section 1.")
    out.append("Structural, capability and query metrics are properties of this "
               "synthetic corpus; they do not predict NESTFUL performance.")
    out.append("The V4 shortcut search is complete only up to its configured depth "
               "and time budget; each task records the depth actually reached.")
    return out


REPRODUCTION_COMMANDS = (
    '$env:PYTHONPATH="src"',
    "python -m targeted_tool_data.cli audit-pilot42-final",
    "python -m targeted_tool_data.cli build-target-profile-v3",
    "python -m targeted_tool_data.cli build-workflow-registry-v3",
    "python -m targeted_tool_data.cli validate-primitive-registry-v3",
    "python -m targeted_tool_data.cli generate-pilot43-semantic "
    "--candidate-target 58000 --run-id pilot4_3_nestful_final",
    "python -m targeted_tool_data.cli validate-pilot43-semantic",
    "python -m targeted_tool_data.cli run-pilot43-v4 --all-answer-types "
    "--counterfactual-instances 5 --workers auto --resume",
    "python -m targeted_tool_data.cli render-pilot43-openrouter --stage smoke",
    "python -m targeted_tool_data.cli render-pilot43-openrouter --stage pilot",
    "python -m targeted_tool_data.cli render-pilot43-openrouter --stage full",
    "python -m targeted_tool_data.cli validate-pilot43-queries",
    "python -m targeted_tool_data.cli select-pilot43 --profile-core 3000 "
    "--long-horizon 1200 --capability-enrichment 600 --challenge 200 "
    "--heldout 1000 --reserve 1000",
    "python -m targeted_tool_data.cli build-pilot43-nested-subsets",
    "python -m targeted_tool_data.cli independent-audit-pilot43",
    "python -m targeted_tool_data.cli prepare-human-audit-pilot43",
    "python -m targeted_tool_data.cli import-human-audit-pilot43 "
    "--ratings outputs/pilot4_3_nestful_final/human_audit_ratings.csv",
    "python -m targeted_tool_data.cli probe-pilot43-grpo-signal "
    "--sample-size 2000 --initial-rollouts 4 --max-rollouts 8",
    "python -m targeted_tool_data.cli compare-pilot42-pilot43",
    "python -m targeted_tool_data.cli freeze-pilot43",
)


# ── Pilot4.2 vs Pilot4.3 ─────────────────────────────────────────────────
#: (row label, pilot42 audit key, pilot43 metric key, direction)
COMPARISON_ROWS = (
    ("tasks", "n", "n", ""),
    ("6+ call share", "six_plus_share", "six_plus_share", "up"),
    ("actual primitives used", "actual_primitives_used",
     "actual_primitives_used", "up"),
    ("actual capability families", "actual_capability_families",
     "actual_capability_families", "up"),
    ("generic/coding task share", "coding_task_share", "coding_task_share", "up"),
    ("generic/coding gold-call share", "coding_call_share", "coding_call_share",
     "up"),
    ("declared pattern == actual pattern", "pattern_match_rate",
     "pattern_match_rate", "up"),
    ("max exact primitive-sequence share", "max_exact_sequence_share",
     "max_exact_sequence_share", "down"),
    ("boolean True share", "boolean_true_share", "boolean_true_share", "band"),
    ("graph-explicit query share", "graph_explicit_share",
     "graph_explicit_share", "down"),
    ("exact query duplicate rate", "exact_duplicate_rate",
     "exact_duplicate_rate", "down"),
    ("V4 coverage", "v4_coverage", "v4_coverage", "up"),
    ("V4 skipped", "v4_skipped", "v4_skipped", "down"),
    ("node necessity coverage", "node_necessity_coverage",
     "node_necessity_coverage", "up"),
    ("distinct intent templates", "distinct_intent_templates",
     "distinct_intent_templates", "up"),
)

DEFECTS = (
    ("no real 6+ call tasks",
     "plans declare 6-10 node capability graphs and the exported call count is "
     "recomputed from gold_calls at every gate"),
    ("declared pattern did not match the DAG",
     "patterns are classified from edges reconstructed out of the built program; a "
     "mismatch rejects the candidate"),
    ("low real capability and primitive diversity",
     "coverage is counted from gold calls, with per-sequence concentration caps in "
     "the validate stage"),
    ("cosmetic generic/coding labels",
     "generic/coding primitives operate on strings, lists, dicts, paths, URLs and "
     "dates; the coding share is measured per call"),
    ("OpenRouter rendering/critics never really ran",
     "staged smoke/pilot/full runner with machine-readable gates, and a "
     "deterministic fallback that marks itself as such"),
    ("over-explicit, repetitive queries",
     "query contracts plus three-level fingerprint diversity gates and an "
     "independent mode classifier"),
    ("V4 skipped for non-numeric answers",
     "V4 runs for every answer type, with counterfactual confirmation"),
    ("thin node necessity evidence",
     "per-node deletion, edge deletion and alternative-binding evidence per task"),
    ("workflow/program/query leakage between splits",
     "heldout keys are chosen first and whole key groups leave the training pool"),
    ("unbalanced booleans",
     "predicate calibration against the oracle value, reported per workflow"),
    ("unrealistic values",
     "per-workflow value generators with domain constraints and realism gates"),
    ("tier quotas only reported",
     "quotas are hard: an unfillable tier fails selection"),
    ("registry size quoted as coverage",
     "registry size and used-primitive count are separate numbers"),
    ("metrics from metadata labels",
     "all metrics are computed from the exported records"),
    ("mixed OpenRouter logs",
     "run-scoped log files, and a foreign run_id refuses the client"),
    ("empty input hashes",
     "freeze fails when no input hashes can be computed; dirty trees get a patch"),
    ("dataset could be called done with unmet gates",
     "TRAINING_READY is a conjunction over five independently produced statuses"),
)


def compare(out_dir: Path, p42_audit: Optional[Path] = None) -> Dict[str, Any]:
    """Write the Pilot4.2/Pilot4.3 comparison from both audit artifacts."""
    quality = _json(out_dir, "PILOT43_DATA_QUALITY_REPORT.json")
    m43 = dict((quality.get("metrics") or {}).get("train_master") or {})
    div = m43.get("diversity") or {}
    m43.setdefault("exact_duplicate_rate", div.get("exact_duplicate_rate"))
    m43.setdefault("distinct_intent_templates", div.get("distinct_intent_templates"))
    m43["graph_explicit_share"] = (m43.get("query_mode_distribution") or {}).get(
        "GRAPH_EXPLICIT")
    m43["pattern_match_rate"] = _pattern_match_rate(out_dir)

    m42, sources = pilot42_metrics(p42_audit)

    rows: List[Dict[str, Any]] = []
    for label, key42, key43, direction in COMPARISON_ROWS:
        rows.append({
            "metric": label,
            "pilot42": m42.get(key42, ""),
            "pilot43": m43.get(key43, ""),
            "wanted_direction": direction,
        })
    with (out_dir / COMPARE_CSV).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "pilot42", "pilot43",
                                                "wanted_direction"])
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "PILOT42_VS_PILOT43_METRICS.json").write_text(
        json.dumps({"rows": rows, "pilot42_sources": sources},
                   indent=1, ensure_ascii=False), encoding="utf-8")

    lines = ["# Pilot4.2 vs Pilot4.3", "",
             f"Pilot4.3 numbers come from `PILOT43_DATA_QUALITY_REPORT.json` "
             f"(recomputed from `train_master_5000.jsonl`).",
             "Pilot4.2 numbers come from "
             + (", ".join(f"`{s}`" for s in sources) if sources
                else "`the Pilot4.2 final audit (not found)`") + ".",
             "", "## Measured comparison", "",
             "| metric | Pilot4.2 | Pilot4.3 | wanted |",
             "| --- | --- | --- | --- |"]
    lines += [f"| {r['metric']} | `{r['pilot42']}` | `{r['pilot43']}` | "
              f"{r['wanted_direction']} |" for r in rows]
    lines += ["", "## The seventeen defects and the mechanism that replaces each",
              "", "| Pilot4.2 defect | Pilot4.3 mechanism |", "| --- | --- |"]
    lines += [f"| {defect} | {fix} |" for defect, fix in DEFECTS]
    lines += ["", f"> {CLAIM}", ""]
    (out_dir / COMPARE_MD).write_text("\n".join(lines), encoding="utf-8")
    return {"rows": rows}


def _pattern_match_rate(out_dir: Path) -> Optional[float]:
    path = out_dir / "actual_pattern_classification.csv"
    if not path.exists():
        return None
    total = matched = 0
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            total += 1
            matched += int(str(row.get("matches_requested")).lower() == "true")
    return round(matched / total, 5) if total else None


#: Where the Pilot4.2 audit keeps each comparable number. Nothing is inferred by
#: name matching: a metric the Pilot4.2 audit never measured stays blank rather
#: than picking up a same-named number that means something else.
P42_SUMMARY_FILE = "PILOT42_AUDIT_SUMMARY.json"
P42_ROOT_CAUSE_FILE = "PILOT42_ROOT_CAUSE_AUDIT.json"


def pilot42_metrics(location: Optional[Path]
                    ) -> Tuple[Dict[str, Any], List[str]]:
    """Read the Pilot4.2 numbers, returning them with the files they came from."""
    if location is None:
        return {}, []
    directory = location if location.is_dir() else location.parent
    summary = _json(directory, P42_SUMMARY_FILE)
    root_cause = _json(directory, P42_ROOT_CAUSE_FILE)
    sources = [f"{directory.name}/{name}"
               for name, data in ((P42_SUMMARY_FILE, summary),
                                  (P42_ROOT_CAUSE_FILE, root_cause)) if data]
    if not sources:
        return {}, []

    measured: Dict[int, Dict[str, Any]] = {}
    for entry in root_cause.get("measurements") or []:
        if isinstance(entry, dict) and isinstance(entry.get("id"), int):
            measured[entry["id"]] = dict(entry.get("measured_value") or {})

    def m(defect_id: int, key: str) -> Any:
        return measured.get(defect_id, {}).get(key)

    def complement(value: Any) -> Any:
        return round(1.0 - value, 5) if isinstance(value, (int, float)) else ""

    queries = summary.get("queries") or {}
    booleans = summary.get("boolean_balance") or {}
    necessity = (summary.get("validation_coverage") or {}).get("node_necessity") or {}
    out = {
        "n": summary.get("n_unique_tasks"),
        "six_plus_share": m(1, "share_tasks_with_ge_6_calls"),
        "actual_primitives_used": m(3, "n_distinct_primitives_used"),
        "actual_capability_families": m(3, "n_distinct_capability_families"),
        "coding_call_share": complement(m(4, "share_arithmetic_or_comparison_calls")),
        "pattern_match_rate": complement(m(2, "declared_pattern_disagreement_rate")),
        "max_exact_sequence_share": m(3, "top1_primitive_sequence_share"),
        "boolean_true_share": booleans.get("overall_true_share"),
        "exact_duplicate_rate": queries.get("exact_duplicate_rate"),
        "v4_coverage": complement(m(7, "v4_search_skipped_share")),
        "v4_skipped": m(7, "v4_search_skipped_share"),
        "node_necessity_coverage": necessity.get("coverage"),
        "distinct_intent_templates": queries.get("n_distinct_intent"),
    }
    return {k: v for k, v in out.items() if v is not None and v != ""}, sources
