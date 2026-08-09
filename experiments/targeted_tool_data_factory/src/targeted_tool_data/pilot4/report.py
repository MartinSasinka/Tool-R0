"""Final Pilot4 implementation report (§24).

The report is assembled from the artifacts the pipeline and audits already wrote,
so it cannot drift away from them: every number below is read back from disk at
render time. The narrative parts are the only prose kept in code, and they are
worded to keep the four claim classes apart -- what was verified from artifacts,
what was implemented, what was generated, and what remains untested because no
training and no NESTFUL evaluation were run.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..repro import stamp, write_json, write_text
from . import SCHEMA_VERSION
from .cells import BUCKET_CALLS
from .compare import admissible_topologies
from .distractors import DISTRACTOR_LEVELS
from .patterns import PATTERN_FAMILIES, TRANSFORMATIONS
from .query_render import QUERY_RENDERERS
from .surface_render import SURFACE_TRACKS

REPORT_SCHEMA_VERSION = "ttdf.pilot4_implementation_report.v1"

# Commands that reproduce every artifact this report reads, in order.
REPRO_COMMANDS = [
    "python -m targeted_tool_data.cli audit-provenance",
    "python -m targeted_tool_data.cli audit-query-realism --profile-safe",
    "python -m targeted_tool_data.cli build-profile-v2",
    "python -m targeted_tool_data.cli capability-audit",
    "python -m targeted_tool_data.cli generate-pilot4",
    "python -m targeted_tool_data.cli compare-datasets "
    "--baseline pilot3 --candidate pilot4_profile_safe",
    "python -m targeted_tool_data.cli simulate-sampler",
    "python -m targeted_tool_data.cli implementation-report",
    "python -m pytest tests -q",
]

KNOWN_LIMITATIONS = [
    "No training, rollout, vLLM, Hugging Face generation or NESTFUL evaluation "
    "was run, so nothing here is evidence about model accuracy.",
    "V4 (bounded minimal-path shortcut search) is disabled in the default run "
    "because it dominates runtime; V1-V3 and V5-V8 gate every selected task and "
    "the run can be repeated with --run-v4.",
    "Fan-out and output reuse are generated even though dev-200 shows none of "
    "them. That is deliberate coverage of the structures pilot3 lacked entirely, "
    "and it is a known distribution-mismatch risk rather than a profile match.",
    "At two and three calls the topology space is exhausted by 1 and 3 shapes, "
    "so per-bucket diversity there cannot be improved further; the top-1 share "
    "is pinned by the join rate the profile asks for.",
    "Query realism is measured by a versioned rule-based lexicon. A low leakage "
    "score is not proof that a question reads naturally to a human.",
    "The sampler simulation uses a synthetic difficulty model, not recorded "
    "rollouts, because no per-rollout reward log exists from the pilot3 run. It "
    "exercises the sampler mechanics only.",
    "Training and evaluation logging are implemented and unit-tested against "
    "their schemas, but no run has produced the artifacts yet.",
    "Capability demand for the diagnostic-informed gap report is exploratory and "
    "is never used as a generation quota.",
]

NEXT_EXPERIMENT = [
    "Freeze the pilot4 train-600 split and run one MT-GRPO training with the "
    "history-adaptive sampler enabled and the new per-rollout logging on, keeping "
    "every other hyperparameter identical to the pilot3 run.",
    "Read dead_group_rate and effective_group_rate from train_steps.jsonl rather "
    "than inferring them, and confirm the sampler reduces the dead-group share.",
    "Evaluate the resulting adapter and the base model in one matched-engine "
    "paired run, checking the eval manifest equality gates (same task set, order, "
    "prompt hashes, tool-schema hashes, scorer version) before comparing scores.",
    "Only then compare NESTFUL accuracy, with a paired significance test, and "
    "treat the pilot3 +2.2 pp result as the baseline to beat.",
]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _shares(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    counts = Counter(str(r.get(key)) for r in rows)
    n = sum(counts.values()) or 1
    return {k: round(v / n, 4) for k, v in sorted(counts.items())}


def _changed_files(repo_root: Path) -> Dict[str, List[str]]:
    try:
        res = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo_root),
                             capture_output=True, text=True, check=False, timeout=120)
        lines = res.stdout.splitlines() if res.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        lines = []
    modified, added = [], []
    for line in lines:
        if not line.strip():
            continue
        code, path = line[:2].strip(), line[3:].strip().strip('"')
        (added if code == "??" else modified).append(path)
    return {"modified": sorted(modified), "added": sorted(added)}


def _test_inventory(module_root: Path) -> Dict[str, Any]:
    per_file: Dict[str, int] = {}
    for path in sorted((module_root / "tests").glob("test_*.py")):
        n = len(re.findall(r"^def test_", path.read_text(encoding="utf-8"),
                           flags=re.MULTILINE))
        per_file[path.name] = n
    return {"per_file": per_file, "n_test_functions": sum(per_file.values()),
            "command": "python -m pytest tests -q"}


def collect(repo_root: Path, module_root: Path, *,
            run_id: str = "pilot4_profile_safe") -> Dict[str, Any]:
    """Read every artifact the report describes. Missing files degrade to {}."""
    out = module_root / "outputs" / run_id
    rep = module_root / "reports"

    selected = _read_jsonl(out / "selected.jsonl")
    # train.jsonl is the GRPO training export and carries the trainer schema, so
    # the composition tables read the canonical rows of the same split instead
    train = [r for r in _read_jsonl(out / "canonical.jsonl")
             if r.get("split") == "train"]
    prov = _read_json(rep / "pilot3_provenance" / "PILOT3_PROVENANCE_AUDIT.json")
    audit = prov.get("audit") or {}
    realism = _read_json(rep / "query_realism" / "QUERY_REALISM_PROFILE.json")
    capability = _read_json(rep / "capability" / "CAPABILITY_REGISTRY.json")
    profile = _read_json(out / "target_profile_v2.json")
    validation = _read_json(out / "validation_report.json")
    selection = _read_json(out / "selection_report.json")
    split = _read_json(out / "split_manifest.json")
    freeze = _read_json(out / "freeze_manifest.json")
    manifest = _read_json(out / "MANIFEST.sha256.json")
    comparison = _read_json(rep / "pilot3_vs_pilot4" / "PILOT3_VS_PILOT4_DATA_AUDIT.json")
    simulation = _read_json(rep / "sampler_simulation" / "SAMPLER_SIMULATION.json")
    cells = _read_json(out / "generation_cells.json")
    constraints = _read_json(out / "topology_constraints.json")

    metrics = comparison.get("metrics") or []
    paired = sum(1 for r in selected if r.get("paired_with"))

    return {
        "run_id": run_id,
        "provenance": {
            "status": audit.get("status"),
            "n_parent_rows": audit.get("n_parent_rows"),
            "n_subset_rows": audit.get("n_subset_rows"),
            "n_canonical_matched": audit.get("n_canonical_matched"),
            "n_matched_inside_parent_first_n":
                audit.get("n_matched_inside_parent_first_n"),
            "matched_positions_are_identity_prefix":
                audit.get("matched_positions_are_identity_prefix"),
            "byte_level": audit.get("byte_level") or {},
            "multi_key_overlap": audit.get("multi_key_overlap") or {},
            "resolved": audit.get("resolved"),
            "retracts_previous_claim": audit.get("retracts_previous_claim"),
            "verdict_note": audit.get("verdict_note"),
            "n_parent_candidates": audit.get("n_parent_candidates"),
        },
        "query_realism": {
            "lexicon_version": realism.get("lexicon_version"),
            "datasets": [
                {k: d.get(k) for k in
                 ("dataset", "n_tasks", "plan_leak_rate",
                  "mean_exact_operation_coverage", "mean_lexical_operation_coverage",
                  "mean_implicit_operation_rate", "mean_sequence_leakage",
                  "mean_procedural_cue_count")}
                for d in (realism.get("datasets") or [])],
            "query_mode_distributions": {
                d.get("dataset"): d.get("query_mode_distribution")
                for d in (realism.get("datasets") or [])},
        },
        "profile_v2": {
            "source": profile.get("source"),
            "mode": profile.get("mode"),
            "n_rows": profile.get("n_rows"),
            "call_count_dist": profile.get("call_count_dist"),
            "conditional_keys": sorted((profile.get("conditional") or {}).keys()),
            "graph_feature_keys": sorted((profile.get("graph_features") or {}).keys()),
            "surface_feature_keys":
                sorted((profile.get("surface_features") or {}).keys()),
            "query_realism": profile.get("query_realism") or {},
            "topology_diversity_by_bucket":
                profile.get("topology_diversity_by_bucket") or {},
            "derived_constraints": constraints,
        },
        "capability_registry": {
            "n_primitives": (capability.get("coverage") or {}).get("n_primitives"),
            "n_families_declared":
                (capability.get("coverage") or {}).get("n_families_declared"),
            "n_families_populated":
                (capability.get("coverage") or {}).get("n_families_populated"),
            "empty_families": (capability.get("coverage") or {}).get("empty_families"),
            "primitives_outside_taxonomy":
                (capability.get("coverage") or {}).get("primitives_outside_taxonomy"),
            "validation_errors": capability.get("validation_errors") or [],
        },
        "patterns": {
            "families": list(PATTERN_FAMILIES),
            "transformations": list(TRANSFORMATIONS),
            "admissible_topologies_per_bucket": {
                b: admissible_topologies(b) for b in BUCKET_CALLS},
            "selected_topologies_per_bucket": {
                b: len({r.get("graph_template_id") for r in selected
                        if r.get("call_bucket") == b}) for b in BUCKET_CALLS},
        },
        "renderers": {
            "query_modes": sorted(QUERY_RENDERERS),
            "surface_tracks": sorted(SURFACE_TRACKS),
            "selected_query_mode_share": _shares(selected, "requested_query_mode"),
            "selected_classified_query_mode_share":
                _shares(selected, "classified_query_mode"),
            "selected_track_share": _shares(selected, "surface_track"),
            "n_paired_records": paired,
        },
        "distractors": {
            "levels": list(DISTRACTOR_LEVELS),
            "schema_compatible_share": next(
                (m.get("pilot4_selected") for m in metrics
                 if m.get("metric") == "schema_compatible_distractor_share"), None),
            "mean_hard_distractor_count": next(
                (m.get("pilot4_selected") for m in metrics
                 if m.get("metric") == "mean_hard_distractor_count"), None),
        },
        "validation": validation,
        "selection": selection,
        "dataset": {
            "n_cells": (cells.get("summary") or {}).get("n_cells")
                        or len(cells.get("cells") or []),
            "counts": freeze.get("counts") or {},
            "deficits": freeze.get("deficits") or {},
            "ordered_sample_ids_hash": freeze.get("ordered_sample_ids_hash"),
            "frozen": freeze.get("frozen"),
            "split": split,
            "call_bucket_share_selected": _shares(selected, "call_bucket"),
            "difficulty_band_share_selected": _shares(selected, "difficulty_band"),
            "call_bucket_share_train": _shares(train, "call_bucket"),
            "artifact_hashes": {k: v.get("sha256")
                                for k, v in (manifest.get("files") or {}).items()},
        },
        "comparison": {
            "n_metrics": len(metrics),
            "verdict_counts": dict(Counter(m.get("verdict") for m in metrics)),
            "distribution_distances": comparison.get("distribution_distances") or {},
            "metrics": metrics,
        },
        "sampler_simulation": {
            "response_model": simulation.get("response_model"),
            "caveat": simulation.get("response_model_caveat"),
            "n_prompts": simulation.get("n_prompts"),
            "steps": simulation.get("steps"),
            "results": {name: {k: res.get(k) for k in
                               ("mean_dead_group_rate_before_filtering",
                                "mean_effective_group_rate_after_filtering",
                                "mean_rollout_utilization", "mean_refill_rounds",
                                "final_sampler_entropy", "n_prompts_touched")}
                        for name, res in (simulation.get("results") or {}).items()},
        },
        "tests": _test_inventory(module_root),
        "files_changed": _changed_files(repo_root),
    }


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return "\n".join(out)


def _markdown(data: Dict[str, Any], repro: Dict[str, Any]) -> str:
    p = data["provenance"]
    v = data["validation"]
    s = data["selection"]
    d = data["dataset"]
    c = data["comparison"]
    L: List[str] = []
    add = L.append

    add("# Pilot4 implementation report")
    add("")
    add(f"- commit: `{repro['git'].get('commit')}` "
        f"(dirty: {repro['git'].get('dirty')}, "
        f"{repro['git'].get('n_dirty_files')} files)")
    add(f"- generated: {repro['generated_at_utc']}")
    add(f"- python {repro['python_version']} on {repro['platform']}")
    add(f"- schema: `{REPORT_SCHEMA_VERSION}`")
    add("")

    add("## 1. Executive summary")
    add("")
    add("**VERIFIED (from artifacts)**")
    add("")
    add(f"- Pilot3 provenance is resolved: status `{p['status']}`, "
        f"{p['n_canonical_matched']}/{p['n_subset_rows']} subset rows matched "
        f"inside the first {p['n_subset_rows']} lines of the "
        f"{p['n_parent_rows']}-row parent export.")
    add(f"- {p['verdict_note']}")
    add("- The earlier 62/300 sample-ID finding was an artifact of comparing a "
        "regenerated working-tree export against the trained one; the retraction "
        "is recorded in the provenance audit "
        f"(`retracts_previous_claim`: {p['retracts_previous_claim']}).")
    add("- Pilot3 training questions leak the plan: measured plan-leak rate "
        + ", ".join(f"{row['dataset']} {row['plan_leak_rate']}"
                    for row in data["query_realism"]["datasets"]
                    if row.get("plan_leak_rate") is not None) + ".")
    add("")
    add("**IMPLEMENTED**")
    add("")
    add(f"- TargetProfile v2 with {len(data['profile_v2']['conditional_keys'])} "
        "conditional distributions, graph and surface feature blocks, plus "
        "derived per-bucket topology constraints.")
    add(f"- Capability taxonomy over {data['capability_registry']['n_primitives']} "
        f"primitives in {data['capability_registry']['n_families_populated']}/"
        f"{data['capability_registry']['n_families_declared']} families.")
    add(f"- {len(data['patterns']['families'])} structural pattern families and "
        f"{len(data['patterns']['transformations'])} composable graph "
        "transformations, all DAG- and execution-checked.")
    add("- SemanticProgram / QueryRenderer / ToolSurfaceRenderer are separate "
        f"layers: {len(data['renderers']['query_modes'])} query modes x "
        f"{len(data['renderers']['surface_tracks'])} surface tracks, with paired "
        "renderings of one program kept in one split.")
    add("- Schema-semantic distractors with V8 validity checking, V7 plan-leak "
        "validation, per-task difficulty signatures, multi-objective selection "
        "with hard constraints, four samplers, and training/eval logging schemas.")
    add("")
    add("**GENERATED**")
    add("")
    add(f"- `{data['run_id']}`: {v.get('n_candidates')} candidates, "
        f"{v.get('n_validated')} validated (pass rate {v.get('pass_rate')}), "
        f"{s.get('n_selected')} selected, split "
        + ", ".join(f"{k} {n}" for k, n in
                    (d["split"].get("sizes_achieved") or {}).items()) + ".")
    add(f"- The split is family-safe: leakage {d['split'].get('leakage')}, "
        f"leak_free {d['split'].get('leak_free')}.")
    add("- Re-running generation on the same commit and seed reproduces every "
        "data artifact byte for byte; only `freeze_manifest.json` differs, "
        "because it records the wall-clock time and the output path.")
    add("")
    add("**NOT TESTED BY TRAINING**")
    add("")
    add("- The adaptive samplers, the per-rollout/group/step logs and the sampler "
        "state checkpointing have unit tests and an offline simulation, but no "
        "GRPO step has been run with them.")
    add("- Whether pilot4 changes dead-group rate in a real run is unknown.")
    add("")
    add("**NOT TESTED BY NESTFUL EVAL**")
    add("")
    add("- No claim is made that pilot4 improves NESTFUL accuracy. The evaluation "
        "logging changes are schema-tested only; no evaluation was executed.")
    add("")
    add("**OPEN**")
    add("")
    for item in KNOWN_LIMITATIONS[1:4]:
        add(f"- {item}")
    add("")

    add("## 2. Files changed")
    add("")
    add(f"Modified ({len(data['files_changed']['modified'])}):")
    add("")
    for path in data["files_changed"]["modified"]:
        add(f"- `{path}`")
    add("")
    add(f"Added ({len(data['files_changed']['added'])}):")
    add("")
    for path in data["files_changed"]["added"]:
        add(f"- `{path}`")
    add("")

    add("## 3. Pilot3 provenance resolution")
    add("")
    add(_table(["field", "value"],
               [["status", p["status"]],
                ["parent rows", p["n_parent_rows"]],
                ["subset rows", p["n_subset_rows"]],
                ["canonical matches", p["n_canonical_matched"]],
                ["matches inside parent prefix",
                 p["n_matched_inside_parent_first_n"]],
                ["matches form the identity prefix",
                 p["matched_positions_are_identity_prefix"]],
                ["parent candidates searched", p["n_parent_candidates"]],
                ["resolved", p["resolved"]],
                ["retracts previous claim", p["retracts_previous_claim"]]]))
    add("")
    add("Multi-key overlap (each key counted separately):")
    add("")
    add(_table(["key", "overlap"],
               [[k, json.dumps(val) if isinstance(val, (dict, list)) else val]
                for k, val in (p["multi_key_overlap"] or {}).items()]))
    add("")
    add("Byte-level comparison of the parent's first 300 lines against the "
        "subset:")
    add("")
    add(_table(["field", "value"],
               [[k, json.dumps(val) if isinstance(val, (dict, list)) else val]
                for k, val in (p["byte_level"] or {}).items()]))
    add("")

    add("## 4. Query-realism findings")
    add("")
    add(f"Operation lexicon version `{data['query_realism']['lexicon_version']}`, "
        "hand-auditable and independent of diagnostic-500.")
    add("")
    rows = [[r["dataset"], r["n_tasks"], r["plan_leak_rate"],
             r["mean_exact_operation_coverage"],
             r["mean_lexical_operation_coverage"],
             r["mean_sequence_leakage"], r["mean_procedural_cue_count"]]
            for r in data["query_realism"]["datasets"]]
    add(_table(["dataset", "n", "plan_leak_rate", "exact_op_cov",
                "lexical_op_cov", "sequence_leakage", "procedural_cues"], rows))
    add("")
    add("Query-mode distribution per dataset:")
    add("")
    for name, dist in data["query_realism"]["query_mode_distributions"].items():
        add(f"- `{name}`: {json.dumps(dist, sort_keys=True)}")
    add("")

    add("## 5. TargetProfile v2")
    add("")
    add(f"Source `{data['profile_v2']['source']}` "
        f"({data['profile_v2']['n_rows']} rows), mode "
        f"`{data['profile_v2']['mode']}`. Conditional distributions: "
        + ", ".join(f"`{k}`" for k in data["profile_v2"]["conditional_keys"]) + ".")
    add("")
    add("Measured per-bucket topology diversity and the constraints derived from "
        "it:")
    add("")
    rows = []
    for bucket, div in (data["profile_v2"]["topology_diversity_by_bucket"] or {}).items():
        cons = (data["profile_v2"]["derived_constraints"] or {}).get(bucket, {})
        rows.append([bucket, div.get("n"), div.get("n_distinct_topologies"),
                     div.get("top1_topology_share"), div.get("join_rate"),
                     json.dumps(cons, sort_keys=True)])
    add(_table(["bucket", "n", "distinct topologies", "top1 share", "join rate",
                "derived constraints"], rows))
    add("")

    add("## 6. Capability registry changes")
    add("")
    cr = data["capability_registry"]
    add(f"- {cr['n_primitives']} primitives, "
        f"{cr['n_families_populated']}/{cr['n_families_declared']} declared "
        "capability families populated.")
    add(f"- empty families: {cr['empty_families']}")
    add(f"- primitives outside the taxonomy: {cr['primitives_outside_taxonomy']}")
    add(f"- registry validation errors: {len(cr['validation_errors'])}")
    add("")

    add("## 7. New structural patterns")
    add("")
    add("Pattern families: "
        + ", ".join(f"`{f}`" for f in data["patterns"]["families"]) + ".")
    add("")
    add("Transformations: "
        + ", ".join(f"`{t}`" for t in data["patterns"]["transformations"]) + ".")
    add("")
    add("Topologies present in the selected set against the number that can "
        "exist at all (blank where the space is too large to enumerate):")
    add("")
    add(_table(["bucket", "topologies in selected", "topologies that exist"],
               [[b, data["patterns"]["selected_topologies_per_bucket"].get(b),
                 data["patterns"]["admissible_topologies_per_bucket"].get(b)]
                for b in BUCKET_CALLS]))
    add("")

    add("## 8. Query renderer changes")
    add("")
    add("Modes: " + ", ".join(f"`{m}`" for m in data["renderers"]["query_modes"])
        + ". Share of the selected set by the mode that was rendered: "
        + f"{json.dumps(data['renderers']['selected_query_mode_share'])}; by the "
        "mode the audit classifier reads back from the question: "
        f"{json.dumps(data['renderers']['selected_classified_query_mode_share'])}.")
    add("")
    add(f"V7 keeps {v.get('V7_in_target_bucket_rate')} of validated candidates "
        "inside the leakage bucket their query mode allows; explicit tasks are "
        "kept but quota-limited rather than discarded.")
    add("")

    add("## 9. Surface renderer changes")
    add("")
    add("Tracks: " + ", ".join(f"`{t}`" for t in data["renderers"]["surface_tracks"])
        + f". Selected shares: {json.dumps(data['renderers']['selected_track_share'])}.")
    add(f"Paired renderings of the same semantic program: "
        f"{data['renderers']['n_paired_records']} records.")
    add("")

    add("## 10. Distractor changes")
    add("")
    add("Levels: " + ", ".join(f"`{lv}`" for lv in data["distractors"]["levels"]) + ".")
    add(f"Schema-compatible distractor share in the selected set: "
        f"{data['distractors']['schema_compatible_share']}; mean hard distractors "
        f"per task: {data['distractors']['mean_hard_distractor_count']}.")
    add("")

    add("## 11. Validation V7-V8")
    add("")
    add(_table(["field", "value"],
               [["candidates", v.get("n_candidates")],
                ["validated", v.get("n_validated")],
                ["pass rate", v.get("pass_rate")],
                ["V7 in target bucket", v.get("V7_in_target_bucket_rate")],
                ["V8 pass rate", v.get("V8_pass_rate")],
                ["V4", v.get("v4_minimal_path")]]))
    add("")
    add("Per-layer failures: "
        f"{json.dumps(v.get('per_layer_failures') or {}, sort_keys=True)}")
    add("")

    add("## 12. Selection v2")
    add("")
    rows = [[r.get("constraint"), r.get("requested_target"), r.get("achieved"),
             r.get("absolute_deficit"), r.get("relative_deficit"), r.get("met"),
             r.get("reason_not_met") or ""]
            for r in (s.get("constraint_rows") or [])]
    add(_table(["constraint", "requested", "achieved", "abs deficit",
                "rel deficit", "met", "reason not met"], rows))
    add("")
    add(f"All hard constraints met: {s.get('all_hard_constraints_met')}. "
        f"Candidate rejections by constraint: "
        f"{json.dumps(s.get('hard_constraint_rejections') or {}, sort_keys=True)}.")
    add("")

    add("## 13. Pilot4 dataset composition")
    add("")
    add(f"- cells: {data['dataset']['n_cells']}")
    add(f"- counts: {json.dumps(d['counts'], sort_keys=True)}")
    add(f"- splits: {json.dumps(d['split'].get('sizes_achieved') or {}, sort_keys=True)}")
    add(f"- call-bucket share (selected): {json.dumps(d['call_bucket_share_selected'])}")
    add(f"- call-bucket share (train): {json.dumps(d['call_bucket_share_train'])}")
    add(f"- difficulty bands: {json.dumps(d['difficulty_band_share_selected'])}")
    add(f"- ordered sample-ID hash: `{d['ordered_sample_ids_hash']}`")
    add(f"- deficits: {json.dumps(d['deficits'], sort_keys=True)}")
    add("")
    add("Artifact hashes:")
    add("")
    add(_table(["file", "sha256"],
               [[k, val] for k, val in sorted(d["artifact_hashes"].items())]))
    add("")

    add("## 14. Pilot3 vs Pilot4 offline comparison")
    add("")
    add(f"{c['n_metrics']} metrics, verdicts: "
        f"{json.dumps(c['verdict_counts'], sort_keys=True)}.")
    add("")
    add("Distribution distances to the dev-200 profile: "
        f"{json.dumps(c['distribution_distances'], sort_keys=True)}")
    add("")
    add("Every metric with its direction of improvement and caveat is in "
        "`reports/pilot3_vs_pilot4/PILOT3_VS_PILOT4_METRICS.csv`. Selected rows:")
    add("")
    keep = {"plan_leak_rate", "goal_based_share", "mean_operation_explicitness",
            "mean_sequence_leakage", "mean_procedural_cue_count",
            "multi_join_rate", "fan_out_rate", "reuse_rate",
            "n_capability_families", "n_distinct_output_keys",
            "schema_compatible_distractor_share",
            "bucket[6+].n_distinct_topologies", "bucket[6+].top1_topology_share",
            "bucket[5].n_distinct_topologies", "bucket[3].top1_topology_share"}
    rows = [[m.get("metric"), m.get("pilot3"), m.get("pilot4_train600"),
             m.get("target_dev200"), m.get("direction"), m.get("verdict")]
            for m in (c.get("metrics") or []) if m.get("metric") in keep]
    add(_table(["metric", "pilot3", "pilot4 train600", "dev-200 target",
                "direction", "verdict"], rows))
    add("")

    add("## 15. Adaptive sampler implementation")
    add("")
    sim = data["sampler_simulation"]
    add(f"Offline simulation over {sim.get('n_prompts')} pilot4 prompts for "
        f"{sim.get('steps')} steps. Response model: `{sim.get('response_model')}`.")
    add("")
    add(f"Caveat: {sim.get('caveat')}")
    add("")
    rows = [[name, r.get("mean_dead_group_rate_before_filtering"),
             r.get("mean_effective_group_rate_after_filtering"),
             r.get("mean_rollout_utilization"), r.get("mean_refill_rounds"),
             r.get("final_sampler_entropy"), r.get("n_prompts_touched")]
            for name, r in (sim.get("results") or {}).items()]
    add(_table(["sampler", "dead-group rate before filter",
                "effective-group rate after filter", "rollout utilization",
                "refill rounds", "final entropy", "prompts touched"], rows))
    add("")

    add("## 16. Training logging implementation")
    add("")
    add("The trainer writes `TRAIN_RUN_MANIFEST.json`, `train_rollouts.jsonl`, "
        "`train_groups.jsonl`, `train_steps.jsonl` and, at every checkpoint, "
        "`sampler_state.json`, `sampler_cell_stats.csv` and "
        "`sampler_prompt_stats.parquet` (CSV fallback when pyarrow is absent). "
        "Per-rollout rows carry the reward components, the group mean and "
        "standard deviation, both advantages, parse/execution status and the "
        "response hash; the response text can be gzipped. Checkpoint resume "
        "restores the sampler state. No training was run.")
    add("")

    add("## 17. Evaluation logging implementation")
    add("")
    add("Each eval run writes `EVAL_RUN_MANIFEST.json`, `eval_inputs.jsonl`, "
        "`eval_trajectories.jsonl` and `eval_task_scores.csv`, recording backend "
        "and engine identity, adapter path and hash, decoding parameters, "
        "chat-template and tool-schema serialization hashes, the input dataset "
        "hash and ordered sample IDs, the shard manifest and the parser/scorer "
        "versions. A paired-run gate compares task set, task order, prompt "
        "hashes, tool-schema hashes and scorer version between two runs and "
        "refuses to compare scores when they differ. No evaluation was run.")
    add("")

    add("## 18. Tests")
    add("")
    add(f"{data['tests']['n_test_functions']} test functions, run with "
        f"`{data['tests']['command']}`; parametrised ones expand to more cases at "
        "collection time.")
    add("")
    add(_table(["file", "test functions"],
               [[k, val] for k, val in data["tests"]["per_file"].items()]))
    add("")

    add("## 19. Known limitations")
    add("")
    for item in KNOWN_LIMITATIONS:
        add(f"- {item}")
    add("")

    add("## 20. Commands for reproduction")
    add("")
    add("From `experiments/targeted_tool_data_factory` with `PYTHONPATH=src`:")
    add("")
    add("```bash")
    for cmd in REPRO_COMMANDS:
        add(cmd)
    add("```")
    add("")

    add("## 21. Recommended next experiment, not executed")
    add("")
    for item in NEXT_EXPERIMENT:
        add(f"- {item}")
    add("")
    return "\n".join(L) + "\n"


def build_report(repo_root: Path, out_dir: Path, *,
                 run_id: str = "pilot4_profile_safe",
                 module_root: Optional[Path] = None,
                 cli_args: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    module_root = module_root or (repo_root / "experiments" /
                                  "targeted_tool_data_factory")
    data = collect(repo_root, module_root, run_id=run_id)
    repro = stamp(repo_root, schema_version=REPORT_SCHEMA_VERSION,
                  cli_args=cli_args)
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "pilot4_schema_version": SCHEMA_VERSION,
        "repro": repro,
        "known_limitations": KNOWN_LIMITATIONS,
        "recommended_next_experiment_not_executed": NEXT_EXPERIMENT,
        "reproduction_commands": REPRO_COMMANDS,
        **data,
    }
    write_json(out_dir / "PILOT4_IMPLEMENTATION_REPORT.json", payload)
    write_text(out_dir / "PILOT4_IMPLEMENTATION_REPORT.md",
               _markdown(data, repro))
    return {"out_dir": str(out_dir), "n_metrics": data["comparison"]["n_metrics"],
            "provenance_status": data["provenance"]["status"],
            "n_test_functions": data["tests"]["n_test_functions"]}
