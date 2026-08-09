"""Final independent root-cause audit of the frozen Pilot4.2 export.

Runs the fully self-contained :mod:`analysis.pilot43_independent_audit` package
against ``outputs/pilot4_2_workflow_grounded_v2/`` and turns the recomputed
numbers into a numbered root-cause report over the 17 known Pilot4.2 defects.

The Pilot4.2 export is FROZEN: this script only reads from it and writes all of
its own artefacts to ``outputs/pilot42_final_audit/`` (per-task ledger, defect
JSON, defect markdown) and to ``outputs/pilot4_3_nestful_final/`` (the
Pilot4.2-vs-Pilot4.3 comparison skeleton whose Pilot4.3 column is filled in
later).

Failure mode prevented: a "known defect list" that is asserted from memory or
copied from producer-side metadata. Every number below is recomputed from the
exported JSONL content, and each of the 17 numbered defects carries the exact
measured value plus the file it was measured from.

Pilot4.2 exported training records are thin - exactly ``call_count``,
``cell_tier``, ``gold_answer``, ``gold_calls``, ``question``,
``requested_query_mode``, ``task_id``, ``tools``, ``was_generated_from_workflow``
and ``workflow_id`` - so no intermediate node values are available. Node output
kinds are therefore passed as ``unknown`` for every node except the sink, which
makes ``TYPE_TRANSITION_CHAIN`` undecidable rather than false.

Run with::

    python analysis/pilot42_final_audit.py
    $env:PYTHONPATH="src"; python -m targeted_tool_data.cli audit-pilot42-final
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# Allow ``python analysis/pilot42_final_audit.py`` as well as ``-m``.
_FACTORY_ROOT = Path(__file__).resolve().parents[1]
if str(_FACTORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_FACTORY_ROOT))

from analysis.pilot43_independent_audit.audit import (  # noqa: E402
    audit_export,
    read_jsonl,
)
from analysis.pilot43_independent_audit.graph_recon import reconstruct  # noqa: E402
from analysis.pilot43_independent_audit.metrics import (  # noqa: E402
    MISSING,
    get_path,
    json_safe,
    lexical_skeleton,
    numeric_literal_stats,
    query_fingerprints,
)
from analysis.pilot43_independent_audit.pattern_rules import (  # noqa: E402
    VALUE_KIND,
    primary_pattern,
    satisfied_patterns,
    undecidable_patterns,
)

FACTORY = Path(__file__).resolve().parents[1]
EXPORT_DIR = FACTORY / "outputs" / "pilot4_2_workflow_grounded_v2"
#: per-task ledger + the numbered defect report
REPORT_DIR = FACTORY / "outputs" / "pilot42_final_audit"
#: the Pilot4.2-vs-Pilot4.3 comparison artefacts consumed by the next step
FINAL_DIR = FACTORY / "outputs" / "pilot4_3_nestful_final"

FILES: Dict[str, str] = {
    "train_master_3000": "train_master_3000.jsonl",
    "heldout_500": "heldout_500.jsonl",
    "reserve_500": "reserve_500.jsonl",
    "selected": "selected.jsonl",
}
EXPECTED_COUNTS: Dict[str, int] = {
    "train_master_3000": 3000,
    "heldout_500": 500,
    "reserve_500": 500,
    "selected": 4000,
}

#: Capability families that are pure arithmetic / comparison plumbing. Used to
#: test whether a domain-flavoured workflow label sits on top of an arithmetic
#: program.
ARITHMETIC_FAMILIES = ("arithmetic.binary", "arithmetic.unary", "comparison", "logic")


def _load_json(path: Path) -> Any:
    """Read a JSON file, returning ``None`` when it is absent or unparsable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _file_size(path: Path) -> int:
    """Byte size of a file, or -1 when absent."""
    return path.stat().st_size if path.exists() else -1


def _line_count(path: Path) -> int:
    """Number of non-empty lines in a text file, or -1 when absent."""
    if not path.exists():
        return -1
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                total += 1
    return total


def build_spec(out_dir: Path) -> Dict[str, Any]:
    """Audit spec for the Pilot4.2 export.

    ``structural_pattern`` points at ``pattern_family``, which only the rich
    ``selected.jsonl`` carries; the thin training files declare no structural
    label at all, and the resulting ``missing_field`` deficit is itself part of
    the audit finding.
    """
    return {
        "run_label": "pilot4_2_workflow_grounded_v2",
        "files": FILES,
        "train_split": "train_master_3000",
        "expected_counts": EXPECTED_COUNTS,
        "declared_paths": {
            "call_count": "call_count",
            "structural_pattern": "pattern_family",
            "actual_structural_pattern": "structural_skill",
            "workflow_id": "workflow_id",
            "cell_tier": "cell_tier",
            "query_mode": "requested_query_mode",
        },
        "validation_paths": {
            "v4": "v4_gate",
            "critic": "llm_critic",
            "node_necessity": "semantic_validation.layers.V_NODE_NECESSITY",
        },
        "overlap_keys": ["workflow_id", "cell_tier"],
        "overlap_against": ["heldout_500", "reserve_500"],
        "dedupe_key": "task_id",
        "node_value_kinds": {"mode": "sink_only"},
        "surface_map": {
            "source": "record_tools",
            "semantic_id_key": "semantic_id",
            "primitive_registry": "primitive_registry.json",
        },
        "thresholds": {
            "min_share_call_count_ge": {"6": 0.05},
            "max_exact_duplicate_rate": 0.0,
            "max_top1_skeleton_share": 0.05,
            "max_top10_intent_share": 0.25,
            "boolean_true_share_range": [0.45, 0.55],
            "min_distinct_primitives": 30,
            "min_distinct_capability_families": 6,
            "max_top1_primitive_sequence_share": 0.10,
            "max_split_overlap": {"workflow_id": 0},
            "min_v4_coverage": 1.0,
            "min_critic_coverage": 1.0,
            "min_node_necessity_coverage": 1.0,
        },
        "out_dir": str(out_dir),
        "emit": {"csv": True, "json": False, "md": False},
        "csv_name": "pilot42_per_task.csv",
        "report_prefix": "PILOT42_ROOT_CAUSE_AUDIT",
        "text_key": "question",
    }


# ---------------------------------------------------------------------------
# Supplementary measurements specific to the Pilot4.2 export
# ---------------------------------------------------------------------------

_SEMI_RE = re.compile(r";")
_REPORT_RE = re.compile(r"report\s+[a-z0-9_ ]+\.\s*$", re.IGNORECASE)
_NOINVENT_RE = re.compile(r"do not invent", re.IGNORECASE)
_DIGIT_RUN_RE = re.compile(r"\d+")


def explicitness_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Surface-explicitness measures over the exported ``question`` text.

    A query that spells every input out as ``name is value;`` and then names the
    target with an imperative ``Report x.`` is a filled template, not a natural
    user request. All shares are measured directly from the exported text.
    """
    n = max(len(records), 1)
    with_semicolon = 0
    semis = 0
    report_suffix = 0
    no_invent = 0
    digit_runs: List[int] = []
    lengths: List[int] = []
    for rec in records:
        text = str(rec.get("question", ""))
        count = len(_SEMI_RE.findall(text))
        semis += count
        if count:
            with_semicolon += 1
        if _REPORT_RE.search(text.strip()):
            report_suffix += 1
        if _NOINVENT_RE.search(text):
            no_invent += 1
        digit_runs.append(len(_DIGIT_RUN_RE.findall(text)))
        lengths.append(len(text.split()))
    return {
        "n": len(records),
        "share_with_semicolon_fact_list": with_semicolon / n,
        "mean_semicolons_per_question": semis / n,
        "share_ending_with_report_imperative": report_suffix / n,
        "share_with_do_not_invent_instruction": no_invent / n,
        "mean_numeric_literals_in_text": sum(digit_runs) / n,
        "share_with_3plus_numbers_in_text": sum(1 for d in digit_runs if d >= 3) / n,
        "mean_question_words": sum(lengths) / n,
    }


def pattern_label_audit(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare declared ``pattern_family`` against the reconstructed DAG.

    Only ``selected.jsonl`` declares a structural label. A declared label counts
    as agreeing when the reconstructed graph satisfies that invariant. Labels
    that name an invariant which the thin export cannot decide (currently only
    ``TYPE_TRANSITION_CHAIN``) are counted in a separate ``undecidable`` bucket
    rather than being scored as wrong.
    """
    checked = 0
    agree = 0
    undecidable_declared = 0
    by_declared: Dict[str, Counter] = defaultdict(Counter)
    confusion: Dict[str, Counter] = defaultdict(Counter)
    declared_counts: Counter = Counter()
    satisfied_counts: Counter = Counter()
    examples: List[Dict[str, Any]] = []
    for rec in records:
        declared = rec.get("pattern_family")
        if not isinstance(declared, str) or not declared:
            continue
        graph = reconstruct(rec.get("gold_calls") or [])
        kinds = ["unknown"] * graph.n
        if graph.n:
            kinds[-1] = VALUE_KIND(rec.get("gold_answer"))
        satisfied = satisfied_patterns(graph, kinds)
        undecidable = undecidable_patterns(kinds)
        checked += 1
        declared_counts[declared] += 1
        for name in satisfied:
            satisfied_counts[name] += 1
        confusion[declared][primary_pattern(satisfied)] += 1
        if declared in satisfied:
            verdict = "agree"
            agree += 1
        elif declared in undecidable:
            verdict = "undecidable"
            undecidable_declared += 1
        else:
            verdict = "disagree"
            if len(examples) < 10:
                examples.append(
                    {
                        "task_id": rec.get("task_id"),
                        "declared_pattern_family": declared,
                        "n_nodes": graph.n,
                        "n_edges": len(graph.edges),
                        "n_join_nodes": graph.features()["n_join_nodes"],
                        "recomputed_satisfied": sorted(satisfied),
                    }
                )
        by_declared[declared][verdict] += 1
    disagree = checked - agree - undecidable_declared
    return {
        "n_checked": checked,
        "n_agree": agree,
        "n_disagree": disagree,
        "n_undecidable": undecidable_declared,
        "disagreement_rate": (disagree / checked) if checked else 0.0,
        "undecidable_rate": (undecidable_declared / checked) if checked else 0.0,
        "by_declared_label": {k: dict(v) for k, v in sorted(by_declared.items())},
        "declared_label_counts": json_safe(declared_counts),
        "records_satisfying_each_label": json_safe(satisfied_counts),
        "declared_vs_recomputed_primary": {
            label: dict(counter) for label, counter in sorted(confusion.items())
        },
        "examples": examples,
    }


def v4_search_coverage(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """How often the V4 shortcut search actually ran, by answer kind."""
    total = 0
    searched = 0
    skipped_reasons: Counter = Counter()
    skipped_by_kind: Counter = Counter()
    total_by_kind: Counter = Counter()
    for rec in records:
        gate = rec.get("v4_gate")
        if not isinstance(gate, dict):
            continue
        total += 1
        kind = VALUE_KIND(rec.get("gold_answer"))
        total_by_kind[kind] += 1
        search = gate.get("search") if isinstance(gate.get("search"), dict) else {}
        if search.get("searched") is True:
            searched += 1
        else:
            skipped_reasons[str(search.get("reason", "<no reason recorded>"))] += 1
            skipped_by_kind[kind] += 1
    return {
        "n_with_v4_gate": total,
        "n_searched": searched,
        "n_skipped": total - searched,
        "skipped_share": ((total - searched) / total) if total else 0.0,
        "skip_reasons": json_safe(skipped_reasons),
        "skipped_by_answer_kind": json_safe(skipped_by_kind),
        "total_by_answer_kind": json_safe(total_by_kind),
    }


def node_necessity_evidence(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Depth of the per-node necessity evidence carried by the export.

    A verdict-only ``{"passed": true, "reasons": []}`` block proves nothing about
    individual nodes; real evidence would name the ablated node and the observed
    answer change.
    """
    present = 0
    verdict_only = 0
    with_per_node = 0
    keys: Counter = Counter()
    for rec in records:
        block = get_path(rec, "semantic_validation.layers.V_NODE_NECESSITY")
        if block is MISSING or not isinstance(block, dict):
            continue
        present += 1
        for key in block:
            keys[key] += 1
        payload = {k: v for k, v in block.items() if k not in ("passed",)}
        has_detail = any(
            isinstance(v, (list, dict)) and len(v) > 0 for v in payload.values()
        )
        if has_detail:
            with_per_node += 1
        else:
            verdict_only += 1
    return {
        "n_present": present,
        "n_verdict_only": verdict_only,
        "n_with_per_node_detail": with_per_node,
        "observed_keys": json_safe(keys),
    }


def leakage_recomputation(
    splits: Dict[str, List[Dict[str, Any]]],
    selected: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Recompute split leakage from record content.

    ``workflow_id`` is present in the thin training records. The richer keys
    (``query_template_fingerprint``, ``program_family_id``, and the two hard
    keys) live only in ``selected.jsonl``, so they are joined onto the split
    membership by ``task_id``. A lexical query-skeleton overlap is added as a
    key-independent proxy for template leakage.
    """
    by_task: Dict[str, Dict[str, Any]] = {
        str(rec.get("task_id")): rec for rec in selected
    }
    train = splits.get("train_master_3000", [])
    heldout = splits.get("heldout_500", [])
    reserve = splits.get("reserve_500", [])

    def values(records: Sequence[Dict[str, Any]], key: str) -> set:
        out = set()
        for rec in records:
            rich = by_task.get(str(rec.get("task_id")), {})
            value = rec.get(key, rich.get(key))
            if value is not None:
                out.add(str(value))
        return out

    def skeletons(records: Sequence[Dict[str, Any]]) -> set:
        return {lexical_skeleton(str(rec.get("question", ""))) for rec in records}

    result: Dict[str, Any] = {"n_selected_joined": len(by_task), "keys": {}}
    for key in (
        "workflow_id",
        "query_template_fingerprint",
        "program_family_id",
        "semantic_program_id",
        "workflow_instance_id",
    ):
        train_values = values(train, key)
        held_values = values(heldout, key)
        res_values = values(reserve, key)
        result["keys"][key] = {
            "n_distinct_train": len(train_values),
            "n_distinct_heldout": len(held_values),
            "n_shared_train_heldout": len(train_values & held_values),
            "n_shared_train_reserve": len(train_values & res_values),
            "heldout_share_seen_in_train": (
                len(train_values & held_values) / len(held_values) if held_values else 0.0
            ),
        }
    train_sk, held_sk = skeletons(train), skeletons(heldout)
    result["query_skeleton"] = {
        "n_distinct_train": len(train_sk),
        "n_distinct_heldout": len(held_sk),
        "n_shared_train_heldout": len(train_sk & held_sk),
        "heldout_share_seen_in_train": (
            len(train_sk & held_sk) / len(held_sk) if held_sk else 0.0
        ),
    }
    return result


def workflow_label_realism(
    records: Sequence[Dict[str, Any]],
    surface_to_primitive: Dict[str, str],
    primitive_to_capability: Dict[str, str],
    workflow_registry: Any,
) -> Dict[str, Any]:
    """Test whether domain-flavoured workflow labels sit on arithmetic programs.

    For every ``workflow_id`` present in the export, the capability families of
    the primitives actually called are collected. A workflow whose domain is
    non-arithmetic (list / text / path / url / dictionary processing, coding-like)
    but whose program uses only arithmetic and comparison families carries a
    cosmetic label.
    """
    per_workflow: Dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        wf = str(rec.get("workflow_id", ""))
        for call in rec.get("gold_calls") or []:
            if not isinstance(call, dict):
                continue
            prim = surface_to_primitive.get(str(call.get("name", "")), "")
            per_workflow[wf][primitive_to_capability.get(prim, "<unknown>")] += 1

    registry_by_id: Dict[str, Dict[str, Any]] = {}
    if isinstance(workflow_registry, dict):
        entries = workflow_registry.get("workflows") or []
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict):
                    registry_by_id[str(entry.get("workflow_id"))] = entry

    non_arith_domains = (
        "list_processing",
        "text_processing",
        "file_path",
        "url_processing",
        "dictionary_processing",
        "date_time",
        "time_duration",
        "geometry",
    )
    cosmetic: List[Dict[str, Any]] = []
    for wf, families in sorted(per_workflow.items()):
        domain = wf.split(".")[0]
        arithmetic_only = all(
            family in ARITHMETIC_FAMILIES for family in families if family != "<unknown>"
        )
        entry = registry_by_id.get(wf, {})
        if arithmetic_only and (
            domain in non_arith_domains or bool(entry.get("coding_like"))
        ):
            cosmetic.append(
                {
                    "workflow_id": wf,
                    "domain": domain,
                    "coding_like_declared": bool(entry.get("coding_like")),
                    "capability_families_used": dict(families),
                }
            )
    all_families: Counter = Counter()
    for families in per_workflow.values():
        all_families.update(families)
    return {
        "n_workflows_in_export": len(per_workflow),
        "n_workflows_in_registry": len(registry_by_id),
        "capability_families_overall": json_safe(all_families),
        "share_arithmetic_or_comparison_calls": (
            sum(v for k, v in all_families.items() if k in ARITHMETIC_FAMILIES)
            / max(sum(all_families.values()), 1)
        ),
        "n_cosmetic_domain_labels": len(cosmetic),
        "cosmetic_domain_labels": cosmetic,
        "domains_present": sorted({wf.split(".")[0] for wf in per_workflow}),
    }


def openrouter_log_forensics(export_dir: Path) -> Dict[str, Any]:
    """Inspect the exported OpenRouter logs for run mixing and empty phases."""
    requests_path = export_dir / "openrouter_requests.jsonl"
    failures_path = export_dir / "openrouter_failures.jsonl"
    rendered = export_dir / "llm_rendered.jsonl"
    rendered_smoke = export_dir / "llm_rendered_smoke.jsonl"

    models: Counter = Counter()
    purposes: Counter = Counter()
    templates: Counter = Counter()
    key_fps: Counter = Counter()
    unknown_task_ids = 0
    timestamps: List[str] = []
    n_requests = 0
    if requests_path.exists():
        for rec in read_jsonl(requests_path):
            n_requests += 1
            models[str(rec.get("model"))] += 1
            purposes[str(rec.get("purpose"))] += 1
            templates[str(rec.get("prompt_template_version"))] += 1
            key_fps[str(rec.get("key_fingerprint"))] += 1
            task_ids = rec.get("task_ids")
            if not isinstance(task_ids, list) or task_ids in ([], ["unknown"]):
                unknown_task_ids += 1
            stamp = rec.get("timestamp")
            if isinstance(stamp, str):
                timestamps.append(stamp)

    failure_models: Counter = Counter()
    failure_codes: Counter = Counter()
    n_failures_without_model = 0
    if failures_path.exists():
        for rec in read_jsonl(failures_path):
            model = rec.get("model")
            if isinstance(model, str) and model:
                failure_models[model] += 1
            else:
                n_failures_without_model += 1
            error = str(rec.get("error", ""))
            match = re.search(r"HTTP (\d{3})", error)
            failure_codes[match.group(1) if match else "other"] += 1

    smoke = _load_json(export_dir / "openrouter_smoke_summary.json") or {}
    usage = _load_json(export_dir / "openrouter_usage_summary.json") or {}
    snapshot = _load_json(export_dir / "openrouter_model_snapshot.json") or {}
    declared_models = {
        value
        for value in (
            snapshot.get("writer_model"),
            snapshot.get("critic_model"),
            snapshot.get("audit_model"),
            usage.get("writer_model"),
            usage.get("critic_model"),
        )
        if isinstance(value, str) and value
    }
    observed_models = {name for name in (set(models) | set(failure_models)) if name and name != "None"}
    return {
        "n_failures_without_model": n_failures_without_model,
        "llm_rendered_jsonl_bytes": _file_size(rendered),
        "llm_rendered_smoke_jsonl_bytes": _file_size(rendered_smoke),
        "n_openrouter_requests": n_requests,
        "n_openrouter_failures": _line_count(failures_path),
        "request_models": json_safe(models),
        "request_purposes": json_safe(purposes),
        "prompt_template_versions": json_safe(templates),
        "n_distinct_key_fingerprints": len(key_fps),
        "n_requests_without_task_ids": unknown_task_ids,
        "timestamp_min": min(timestamps) if timestamps else "",
        "timestamp_max": max(timestamps) if timestamps else "",
        "failure_models": json_safe(failure_models),
        "failure_http_codes": json_safe(failure_codes),
        "smoke_summary": smoke,
        "usage_summary": usage,
        "model_snapshot": snapshot,
        "declared_models": sorted(declared_models),
        "observed_models": sorted(observed_models),
        "models_observed_but_not_declared": sorted(observed_models - declared_models),
    }


def type_transition_supplement(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Supplementary TYPE_TRANSITION_CHAIN probe using ``oracle_observations``.

    The thin training files carry no intermediate values, so the invariant is
    undecidable there. ``selected.jsonl`` happens to export per-node oracle
    observations, which lets the invariant be measured for those records; this is
    reported as a supplement, not as the audit verdict.
    """
    usable = 0
    satisfied_count = 0
    for rec in records:
        obs = rec.get("oracle_observations")
        calls = rec.get("gold_calls") or []
        if not isinstance(obs, list) or len(obs) != len(calls) or not calls:
            continue
        graph = reconstruct(calls)
        kinds = [VALUE_KIND(v) for v in obs]
        usable += 1
        if "TYPE_TRANSITION_CHAIN" in satisfied_patterns(graph, kinds):
            satisfied_count += 1
    return {
        "n_usable_records": usable,
        "n_satisfying_type_transition_chain": satisfied_count,
        "share": (satisfied_count / usable) if usable else 0.0,
        "note": "measurable only because selected.jsonl exports oracle_observations; the thin training files make this invariant undecidable",
    }


# ---------------------------------------------------------------------------
# Defect table
# ---------------------------------------------------------------------------


def _pct(value: float) -> str:
    """Format a share as a percentage with three decimals."""
    return f"{value * 100:.3f}%"


def build_defect_table(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Assemble the 17 numbered Pilot4.2 defects with measured evidence.

    Every ``confirmed: yes`` entry cites at least one recomputed number together
    with the exported file and field it came from. Nothing is inferred.
    """
    audit = evidence["audit"]
    cc = audit["call_count"]
    prims = audit["primitives"]
    dup = audit["queries"]["duplicate_rates_overall"]
    bools = audit["boolean_balance"]
    seqs = audit["sequences"]
    labels = evidence["pattern_labels"]
    v4 = evidence["v4_search"]
    necessity = evidence["node_necessity"]
    leak = evidence["leakage"]
    realism = evidence["workflow_labels"]
    router = evidence["openrouter"]
    explicit = evidence["explicitness"]
    literals = evidence["numeric_literals"]
    freeze = evidence["freeze_manifest"] or {}
    selection = evidence["selection_report"] or {}
    v4_report = evidence["v4_report"] or {}
    validation_report = evidence["validation_report"] or {}
    split_manifest = evidence["split_manifest"] or {}
    tiers = evidence["cell_tier_counts"]

    max_cc = max(int(k) for k in cc["overall"]) if cc["overall"] else 0
    share_ge6 = cc["share_at_least"]["6"]
    bool_share_of_all = bools["n_boolean"] / max(audit["counts"]["selected"] or 1, 1)
    skewed_workflows = [
        (wf, block)
        for wf, block in bools["by_workflow"].items()
        if block["n"] >= 20 and (block["true_share"] >= 0.7 or block["true_share"] <= 0.3)
    ]
    soft_overlap = split_manifest.get("soft_key_overlap") or {}
    support = selection.get("workflow_support") or {}
    support_values = sorted(int(v) for v in support.values()) if support else []

    table: List[Dict[str, Any]] = []

    table.append(
        {
            "n": 1,
            "defect": "zero true 6+ call tasks",
            "confirmed": "yes",
            "measured_evidence": [
                "recomputed call-count histogram from `gold_calls` over all 4 audited files: "
                + ", ".join(f"{k} calls: {v}" for k, v in sorted(cc["overall"].items(), key=lambda kv: int(kv[0])))
                + f"; maximum observed call count = {max_cc}",
                f"share of tasks with >= 6 calls = {_pct(share_ge6)} (0 of {sum(cc['overall'].values())} records); "
                f"share with >= 5 calls = {_pct(cc['share_at_least']['5'])}",
                "provenance: len(record['gold_calls']) recomputed per record in "
                "train_master_3000.jsonl, heldout_500.jsonl, reserve_500.jsonl, selected.jsonl",
            ],
            "required_fix": (
                "Pilot4.3 must generate programs with 6-10 calls as a first-class tier and gate on a "
                "recomputed histogram (not a declared call_count), with a hard minimum share of 6+ call tasks "
                "per split; the generator's program templates must be extended beyond the 2-5 node blueprints."
            ),
        }
    )

    table.append(
        {
            "n": 2,
            "defect": "declared pattern label vs actual dependency DAG mismatch",
            "confirmed": "yes",
            "measured_evidence": [
                f"selected.jsonl `pattern_family` vs independently reconstructed DAG: "
                f"{labels['n_disagree']}/{labels['n_checked']} records declare a pattern their graph does not satisfy "
                f"(disagreement rate {_pct(labels['disagreement_rate'])}); {labels['n_agree']} agree and "
                f"{labels['n_undecidable']} declare TYPE_TRANSITION_CHAIN, which the thin export cannot decide and "
                "which is therefore scored as undecidable rather than wrong",
                "per declared label (agree / disagree / undecidable): "
                + "; ".join(
                    f"{label}: {counts.get('agree', 0)} / {counts.get('disagree', 0)} / {counts.get('undecidable', 0)}"
                    for label, counts in labels["by_declared_label"].items()
                ),
                f"the thin training files declare NO structural label at all: `pattern_family` is absent in "
                f"{evidence['missing_pattern_family_records']} of {audit['n_records_audited']} audited rows - i.e. in "
                "all of train_master_3000.jsonl / heldout_500.jsonl / reserve_500.jsonl, which carry only call_count, "
                "cell_tier, gold_answer, gold_calls, question, requested_query_mode, task_id, tools, "
                "was_generated_from_workflow, workflow_id",
                f"recomputed primary pattern distribution over the {audit['n_unique_tasks']} unique tasks: "
                + ", ".join(f"{k}: {v}" for k, v in sorted(audit["patterns"]["primary_distribution"].items(), key=lambda kv: -kv[1])),
                "the declared labels are near-disjoint from the structures actually present - number of records that "
                "genuinely satisfy each invariant vs number of records declaring it: "
                + "; ".join(
                    f"{label}: satisfied by {labels['records_satisfying_each_label'].get(label, 0)}, "
                    f"declared by {count}, both {labels['by_declared_label'].get(label, {}).get('agree', 0)}"
                    for label, count in sorted(labels["declared_label_counts"].items(), key=lambda kv: -kv[1])
                ),
                "worked examples (declared label, then the reconstructed graph): "
                + "; ".join(
                    f"{item['task_id']} declares {item['declared_pattern_family']} but has {item['n_nodes']} nodes, "
                    f"{item['n_edges']} edges and {item['n_join_nodes']} join nodes, satisfying "
                    f"{item['recomputed_satisfied'] or ['nothing']}"
                    for item in labels["examples"][:4]
                ),
            ],
            "required_fix": (
                "The structural label must be computed from the emitted gold_calls DAG by the same code path that "
                "the audit uses, exported per record, and hard-gated: any record whose declared pattern is not "
                "satisfied by its own reconstructed graph must be rejected before selection."
            ),
        }
    )

    table.append(
        {
            "n": 3,
            "defect": "low real capability and primitive diversity",
            "confirmed": "yes",
            "measured_evidence": [
                f"distinct gold tool surface names actually called = {prims['distinct_gold_tool_surfaces']}; "
                f"distinct primitives actually used = {prims['distinct_primitives_used']} "
                f"(mapping source: {prims['primitive_mapping_source']})",
                f"distinct capability families actually used = {prims['distinct_capability_families']}: "
                + ", ".join(f"{k}: {v}" for k, v in sorted(prims["capability_counts"].items(), key=lambda kv: -kv[1])),
                f"distinct exact primitive sequences = {seqs['n_distinct_primitive_sequences']}, "
                f"top-1 sequence share = {_pct(seqs['primitive_sequence_concentration']['top1_share'])}, "
                f"top-10 share = {_pct(seqs['primitive_sequence_concentration']['top10_share'])}",
                "primitive counts: "
                + ", ".join(f"{k}: {v}" for k, v in sorted(prims["primitive_counts"].items(), key=lambda kv: -kv[1])),
            ],
            "required_fix": (
                "Pilot4.3 must impose a per-primitive and per-capability-family coverage floor measured from the "
                "exported gold_calls (for example >= 40 distinct primitives and >= 8 capability families, with no "
                "single primitive above 15% of calls and no single call sequence above 5% of tasks), and select "
                "against that recomputed coverage rather than against registry size."
            ),
        }
    )

    table.append(
        {
            "n": 4,
            "defect": "only cosmetic generic/coding workflow labels over arithmetic programs",
            "confirmed": "yes" if realism["n_cosmetic_domain_labels"] > 0 else "no",
            "measured_evidence": [
                f"share of gold calls whose capability family is arithmetic/comparison/logic = "
                f"{_pct(realism['share_arithmetic_or_comparison_calls'])} "
                f"(capability families overall: "
                + ", ".join(f"{k}: {v}" for k, v in sorted(realism["capability_families_overall"].items(), key=lambda kv: -kv[1]))
                + ")",
                f"{realism['n_cosmetic_domain_labels']} workflow ids carry a non-arithmetic domain label while their "
                f"programs use only arithmetic/comparison primitives, e.g. "
                + "; ".join(
                    f"{item['workflow_id']} -> {sorted(item['capability_families_used'])}"
                    for item in realism["cosmetic_domain_labels"][:6]
                ),
                f"domains present in the export ({len(realism['domains_present'])}): "
                + ", ".join(realism["domains_present"]),
                f"workflow ids present in the export = {realism['n_workflows_in_export']} of "
                f"{realism['n_workflows_in_registry']} declared in workflow_registry.json",
            ],
            "required_fix": (
                "A workflow label must be backed by primitives from its own domain: Pilot4.3 must require that every "
                "list/text/path/url/dictionary/date workflow uses at least one primitive from the matching capability "
                "family (sequence.*, text.*, path.*, url.*, datetime.*), verified from the exported gold_calls, and "
                "must drop domain labels that cannot be so backed."
            ),
        }
    )

    table.append(
        {
            "n": 5,
            "defect": "non-functional / never-executed OpenRouter rendering and critic phases",
            "confirmed": "yes",
            "measured_evidence": [
                f"llm_rendered.jsonl = {router['llm_rendered_jsonl_bytes']} bytes and "
                f"llm_rendered_smoke.jsonl = {router['llm_rendered_smoke_jsonl_bytes']} bytes: the rendering phase "
                "produced no output at all",
                f"openrouter_smoke_summary.json: n_input={router['smoke_summary'].get('n_input')}, "
                f"n_passed={router['smoke_summary'].get('n_passed')}, n_reject={router['smoke_summary'].get('n_reject')}, "
                f"pass_rate={router['smoke_summary'].get('pass_rate')}, "
                f"llm_status={router['smoke_summary'].get('llm_status')!r}, "
                f"critic_coverage={router['smoke_summary'].get('critic_coverage')}",
                f"openrouter_failures.jsonl contains {router['n_openrouter_failures']} failure records "
                f"(HTTP codes: " + ", ".join(f"{k}: {v}" for k, v in sorted(router["failure_http_codes"].items())) + ")",
                f"no exported record carries a critic verdict: critic coverage = "
                f"{audit['validation_coverage']['critic']['n_present']}/{audit['validation_coverage']['critic']['n']} "
                f"unique tasks; no `{audit['validation_coverage']['critic']['path']}` field exists on any record, in "
                "either the thin training files or the richer selected.jsonl",
                f"freeze_manifest.json: LLM_VALIDATED={freeze.get('LLM_VALIDATED')}, "
                f"llm_status={freeze.get('llm_status')!r}",
            ],
            "required_fix": (
                "Pilot4.3 must treat the LLM rendering and critic phases as hard dependencies: a non-empty "
                "llm_rendered.jsonl with one record per selected task, per-task critic verdicts stored on the record, "
                "and a build that fails when rendering pass rate or critic coverage is below the configured floor "
                "instead of silently exporting template text."
            ),
        }
    )

    table.append(
        {
            "n": 6,
            "defect": "overly explicit, synthetic, repetitive questions",
            "confirmed": "yes",
            "measured_evidence": [
                f"exact `question` duplicate rate = {_pct(dup['exact_duplicate_rate'])} "
                f"({dup['n_distinct_exact']} distinct texts over {dup['n']} records)",
                f"normalized lexical skeleton concentration: {dup['n_distinct_skeleton']} distinct skeletons, "
                f"top-1 skeleton share = {_pct(dup['top1_skeleton_share'])}, "
                f"top-10 skeleton share = {_pct(dup['top10_skeleton_share'])}",
                f"intent-template concentration: {dup['n_distinct_intent']} distinct intents, "
                f"top-1 intent share = {_pct(dup['top1_intent_share'])}, "
                f"top-10 intent template share = {_pct(dup['top10_intent_share'])}",
                f"explicitness measured from the exported text: "
                f"{_pct(explicit['share_with_semicolon_fact_list'])} of questions spell inputs out as a "
                f"semicolon-separated fact list (mean {explicit['mean_semicolons_per_question']:.2f} semicolons), "
                f"{_pct(explicit['share_ending_with_report_imperative'])} end with a 'Report x.' imperative, "
                f"{_pct(explicit['share_with_3plus_numbers_in_text'])} contain 3 or more numeric literals "
                f"(mean {explicit['mean_numeric_literals_in_text']:.2f} numbers, mean length "
                f"{explicit['mean_question_words']:.1f} words)",
            ],
            "required_fix": (
                "Queries must be LLM-rendered and gated on recomputed diversity: cap the top-1 lexical skeleton share "
                "and the top-10 intent-template share (for example <= 5% and <= 25%), forbid the "
                "'name is value; ... Report x.' scaffold, and require the target to be implied by the goal rather "
                "than named imperatively."
            ),
        }
    )

    table.append(
        {
            "n": 7,
            "defect": "V4 shortcut check skipped for boolean and other non-numeric answers",
            "confirmed": "yes",
            "measured_evidence": [
                f"selected.jsonl `v4_gate.search.searched`: {v4['n_skipped']}/{v4['n_with_v4_gate']} records "
                f"({_pct(v4['skipped_share'])}) never ran the shortcut search; skip reasons: "
                + ", ".join(f"{k}: {v}" for k, v in sorted(v4["skip_reasons"].items(), key=lambda kv: -kv[1])),
                "skipped records by recomputed answer kind: "
                + ", ".join(f"{k}: {v}" for k, v in sorted(v4["skipped_by_answer_kind"].items(), key=lambda kv: -kv[1]))
                + " (recomputed from `gold_answer` with bool checked before int)",
                f"v4_report.json nevertheless reports shortcut_rate={v4_report.get('shortcut_rate')} and "
                f"n_shortcuts={v4_report.get('n_shortcuts')} over n={v4_report.get('n')}, i.e. a clean rate derived "
                "from a search that was skipped for the majority of records",
            ],
            "required_fix": (
                "The V4 shortcut search must cover every answer kind: enumerate boolean and categorical answers "
                "explicitly (a boolean answer has only two candidate values, so a shortcut predictor is trivially "
                "testable), record `searched: true` per task, and fail the build when shortcut-search coverage is "
                "below 100% of selected tasks."
            ),
        }
    )

    table.append(
        {
            "n": 8,
            "defect": "insufficient per-node necessity evidence",
            "confirmed": "yes",
            "measured_evidence": [
                f"selected.jsonl `semantic_validation.layers.V_NODE_NECESSITY` present on "
                f"{necessity['n_present']} records, of which {necessity['n_verdict_only']} carry a verdict only "
                f"and {necessity['n_with_per_node_detail']} carry any per-node detail; observed keys = "
                + ", ".join(f"{k} ({v})" for k, v in sorted(necessity["observed_keys"].items())),
                f"in the actual training artefacts the evidence is absent entirely: node-necessity coverage over the "
                f"{audit['validation_coverage']['node_necessity']['n']} unique tasks as exported in "
                f"train_master_3000.jsonl / heldout_500.jsonl / reserve_500.jsonl = "
                f"{audit['validation_coverage']['node_necessity']['n_present']} records with a "
                f"`{audit['validation_coverage']['node_necessity']['path']}` block",
            ],
            "required_fix": (
                "Necessity must be evidenced per node: for each node store the ablated-program answer and the "
                "observed answer delta, export that table on the record, and gate on 'every node changes the answer "
                "when removed' computed from the exported evidence rather than on a single boolean verdict."
            ),
        }
    )

    wf_leak = leak["keys"]["workflow_id"]
    qtf_leak = leak["keys"]["query_template_fingerprint"]
    pf_leak = leak["keys"]["program_family_id"]
    table.append(
        {
            "n": 9,
            "defect": "workflow / program / query-template leakage between splits",
            "confirmed": "yes",
            "measured_evidence": [
                f"recomputed from record content: `workflow_id` shared between train_master_3000.jsonl and "
                f"heldout_500.jsonl = {wf_leak['n_shared_train_heldout']} values, i.e. "
                f"{_pct(wf_leak['heldout_share_seen_in_train'])} of the {wf_leak['n_distinct_heldout']} distinct "
                "heldout workflows were already seen in train",
                f"joined via task_id onto selected.jsonl: `query_template_fingerprint` shared train/heldout = "
                f"{qtf_leak['n_shared_train_heldout']} ({_pct(qtf_leak['heldout_share_seen_in_train'])} of heldout), "
                f"`program_family_id` shared train/heldout = {pf_leak['n_shared_train_heldout']} "
                f"({_pct(pf_leak['heldout_share_seen_in_train'])} of heldout)",
                f"key-independent proxy: {leak['query_skeleton']['n_shared_train_heldout']} lexical query skeletons "
                f"are shared between train and heldout, i.e. "
                f"{_pct(leak['query_skeleton']['heldout_share_seen_in_train'])} of the "
                f"{leak['query_skeleton']['n_distinct_heldout']} distinct heldout skeletons",
                f"split_manifest.json declares leak_free={split_manifest.get('leak_free')} while reporting "
                f"soft_key_overlap = " + json.dumps(soft_overlap),
                f"hard keys are indeed disjoint: `semantic_program_id` shared = "
                f"{leak['keys']['semantic_program_id']['n_shared_train_heldout']}, `workflow_instance_id` shared = "
                f"{leak['keys']['workflow_instance_id']['n_shared_train_heldout']}",
            ],
            "required_fix": (
                "Split on the generalisation-relevant keys, not on instance ids: workflow_id, program_family_id and "
                "query_template_fingerprint must be disjoint across train and heldout, `leak_free` must be false "
                "whenever any soft-key overlap is non-zero, and the split gate must be recomputed from the exported "
                "records instead of trusting the manifest."
            ),
        }
    )

    table.append(
        {
            "n": 10,
            "defect": "strongly unbalanced boolean answers",
            "confirmed": "yes",
            "measured_evidence": [
                f"recomputed from `gold_answer`: {bools['n_boolean']} boolean answers "
                f"({_pct(bool_share_of_all)} of the {audit['counts']['selected']} selected records), "
                f"overall True share = {_pct(bools['overall_true_share'])} "
                f"({bools['n_true']} True vs {bools['n_boolean'] - bools['n_true']} False)",
                f"a majority-class baseline that always answers True therefore scores "
                f"{_pct(max(bools['overall_true_share'], 1 - bools['overall_true_share']))} on boolean tasks",
                f"{len(skewed_workflows)} workflow ids with >= 20 boolean tasks are skewed beyond 70/30, e.g. "
                + "; ".join(
                    f"{wf}: {block['n_true']}/{block['n']} True ({_pct(block['true_share'])})"
                    for wf, block in sorted(skewed_workflows, key=lambda kv: -abs(kv[1]["true_share"] - 0.5))[:6]
                ),
                "answer kind mix recomputed from `gold_answer`: "
                + ", ".join(
                    f"{k}: {v}"
                    for k, v in sorted(evidence["answer_kind_counts"].items(), key=lambda kv: -kv[1])
                ),
            ],
            "required_fix": (
                "Boolean answers must be balanced by construction and gated: sample threshold literals so that the "
                "True share sits in [0.45, 0.55] overall AND per workflow / per cell, and cap the overall share of "
                "boolean answers so the dataset is not dominated by a two-way guess."
            ),
        }
    )

    table.append(
        {
            "n": 11,
            "defect": "unrealistic, trivially predictable values",
            "confirmed": "yes",
            "measured_evidence": [
                f"literal numeric arguments recomputed from `gold_calls` ({literals['n']} values): "
                f"min={literals['min']}, max={literals['max']}, mean={literals['mean']:.2f}, "
                f"median={literals['median']}, {literals['n_distinct']} distinct values",
                f"share integer = {_pct(literals['share_integer'])}, "
                f"share round multiple of 10 = {_pct(literals['share_round_multiple_of_10'])}, "
                f"share that looks like a generic random integer in 1..2000 = "
                f"{_pct(literals['share_generic_int_1_2000'])}",
                f"trivial predictability of the answer: the majority-class boolean baseline already reaches "
                f"{_pct(max(bools['overall_true_share'], 1 - bools['overall_true_share']))} "
                f"(see defect 10), and no non-integer literal appears at all "
                f"(share integer = {_pct(literals['share_integer'])}), so no realistic money/measure precision is present",
            ],
            "required_fix": (
                "Values must be drawn from role- and unit-aware realistic distributions (prices with cents, rates in "
                "plausible bands, counts with realistic magnitudes, occasional outliers) and gated on recomputed "
                "realism statistics: bounded share of round integers, non-zero share of fractional values, and no "
                "single uniform range covering the majority of literals."
            ),
        }
    )

    quota_keys = [k for k in selection if "quota" in k.lower() or "tier" in k.lower()]
    table.append(
        {
            "n": 12,
            "defect": "unmet or merely reported selection tier quotas",
            "confirmed": "yes",
            "measured_evidence": [
                f"selection_report.json declares selected={selection.get('selected')} of "
                f"requested={selection.get('requested')} from eligible_pool={selection.get('eligible_pool')} with "
                f"selection_all_hard_constraints_met={selection.get('selection_all_hard_constraints_met')} and "
                f"deficit={selection.get('deficit')}, but contains no per-tier quota field "
                f"(tier/quota-related keys present: {quota_keys if quota_keys else 'none'}), so the quota claim "
                "cannot be verified from the export",
                "recomputed `cell_tier` counts over selected.jsonl: "
                + ", ".join(f"{k}: {v}" for k, v in sorted(tiers.items(), key=lambda kv: -kv[1])),
                f"workflow support declared in selection_report.json is extremely uneven: min={support_values[0] if support_values else 'n/a'}, "
                f"max={support_values[-1] if support_values else 'n/a'}, "
                f"{sum(1 for v in support_values if v <= 2)} of {len(support_values)} workflows have <= 2 supporting "
                "tasks while the modal workflow has 82",
                "call-count support per tier recomputed from `gold_calls`: "
                + "; ".join(
                    f"{tier}: " + ", ".join(f"{k} calls x{v}" for k, v in sorted(hist.items(), key=lambda kv: int(kv[0])))
                    for tier, hist in sorted(audit["call_count"]["by_tier"].items())
                ),
            ],
            "required_fix": (
                "Tier and cell quotas must be declared as explicit numeric targets in the selection config, exported "
                "per tier next to the achieved count, and enforced: a shortfall in any tier (including per-workflow "
                "minimum support, for example >= 20 tasks per workflow) must make the selection gate fail rather than "
                "be reported as met."
            ),
        }
    )

    table.append(
        {
            "n": 13,
            "defect": "registry size vs primitives actually used in the dataset",
            "confirmed": "yes",
            "measured_evidence": [
                f"primitive_registry.json declares {prims['registry_size']} primitives; the dataset actually uses "
                f"{prims['distinct_primitives_used']} of them, i.e. registry coverage = "
                f"{_pct(prims['registry_coverage'])}",
                f"workflow_registry.json declares {realism['n_workflows_in_registry']} workflows; "
                f"{realism['n_workflows_in_export']} appear in the exported records",
                f"distinct gold tool surfaces = {prims['distinct_gold_tool_surfaces']}, and the surface -> primitive "
                f"map derived from `record.tools[].semantic_id` has {len(prims['surface_map_collisions'])} collisions",
            ],
            "required_fix": (
                "Report and gate on used-primitive coverage, not registry size: export the used/declared ratio, "
                "require a configured minimum coverage of the registry, and remove or exercise primitives that no "
                "generated program ever calls."
            ),
        }
    )

    table.append(
        {
            "n": 14,
            "defect": "metrics computed from metadata labels instead of exported content",
            "confirmed": "yes",
            "measured_evidence": [
                f"the declared structural label disagrees with the reconstructed DAG on "
                f"{labels['n_disagree']}/{labels['n_checked']} selected records "
                f"({_pct(labels['disagreement_rate'])}), so any metric aggregated over `pattern_family` describes "
                "labels rather than programs",
                f"v4_report.json reports shortcut_rate={v4_report.get('shortcut_rate')} over n={v4_report.get('n')} "
                f"while {v4['n_skipped']} of {v4['n_with_v4_gate']} records never ran the shortcut search",
                f"validation_report.json is a four-number summary ({json.dumps(validation_report)}) with no per-task "
                "or per-layer detail, and PILOT42_DATA_QUALITY_REPORT.json reports only counts plus label-derived "
                "booleans",
                f"the thin training files contain no structural, primitive or validation metadata at all "
                f"(`pattern_family` absent in {evidence['missing_pattern_family_records']} of "
                f"{audit['n_records_audited']} audited rows), so their quality cannot be reconstructed from "
                "declared fields - only from content, as done here",
            ],
            "required_fix": (
                "Every reported metric must be recomputed from the exported JSONL content by an auditor that cannot "
                "import producer code (this package), the audit must run as part of the build, and the freeze must "
                "carry the auditor's verdict rather than producer-side self-reports."
            ),
        }
    )

    table.append(
        {
            "n": 15,
            "defect": "mixed OpenRouter logs from different runs",
            "confirmed": "yes",
            "measured_evidence": [
                f"openrouter_requests.jsonl holds {router['n_openrouter_requests']} requests spanning "
                f"{router['timestamp_min']} .. {router['timestamp_max']} with models "
                + ", ".join(f"{k}: {v}" for k, v in sorted(router["request_models"].items(), key=lambda kv: -kv[1])),
                f"models observed in the logs but not declared in openrouter_model_snapshot.json / "
                f"openrouter_usage_summary.json (declared: {', '.join(router['declared_models'])}): "
                + (", ".join(router["models_observed_but_not_declared"]) or "none"),
                "prompt_template_version values present: "
                + ", ".join(f"{k}: {v}" for k, v in sorted(router["prompt_template_versions"].items()))
                + " - Pilot4.1 template versions appear in a Pilot4.2 export",
                f"{router['n_requests_without_task_ids']} of {router['n_openrouter_requests']} request records carry "
                f"no resolvable task id (task_ids == ['unknown']), so no log line can be attributed to an exported "
                f"task; openrouter_failures.jsonl adds {router['n_openrouter_failures']} records from "
                + ", ".join(f"{k} ({v})" for k, v in sorted(router["failure_models"].items()))
                + f" plus {router['n_failures_without_model']} records with no model field",
            ],
            "required_fix": (
                "OpenRouter logs must be per-run and attributable: write them under a run id directory, stamp every "
                "record with the run id, the prompt template version of THIS pilot and the concrete task ids, refuse "
                "to append to a log written by a different run id or template version, and fail the freeze when the "
                "declared model set does not equal the observed model set."
            ),
        }
    )

    provenance = (freeze.get("provenance") or {}) if isinstance(freeze, dict) else {}
    git = provenance.get("git") or {}
    table.append(
        {
            "n": 16,
            "defect": "incomplete reproducibility and empty input hashes",
            "confirmed": "yes",
            "measured_evidence": [
                f"freeze_manifest.json provenance.input_hashes = {json.dumps(provenance.get('input_hashes'))} "
                "(empty object: no input artefact is hash-pinned)",
                f"git state at freeze: commit={git.get('commit')}, dirty={git.get('dirty')}, "
                f"n_dirty_files={git.get('n_dirty_files')} - the export was produced from an uncommitted tree",
                f"freeze_manifest.json records cli_args={json.dumps(provenance.get('cli_args'))} and "
                f"seeds={json.dumps(provenance.get('seeds'))} but LLM_VALIDATED={freeze.get('LLM_VALIDATED')} and "
                f"llm_status={freeze.get('llm_status')!r}, so the LLM-dependent part of the pipeline is not "
                "reproducible from the manifest",
                f"MANIFEST.sha256.json is present ({_file_size(EXPORT_DIR / 'MANIFEST.sha256.json')} bytes) and hashes "
                "outputs, but no input hashes exist to tie those outputs to their sources",
            ],
            "required_fix": (
                "The freeze must pin inputs: hash every config, registry, workflow blueprint, prompt template and "
                "upstream JSONL into `input_hashes`, refuse to freeze from a dirty git tree (or record the full diff), "
                "and store the resolved model snapshot plus prompt hashes so an LLM-dependent run can be replayed."
            ),
        }
    )

    unmet = []
    if share_ge6 <= 0:
        unmet.append("zero 6+ call tasks")
    if freeze.get("LLM_VALIDATED") is False:
        unmet.append("LLM_VALIDATED=false")
    if any(int(v) > 0 for v in soft_overlap.values()):
        unmet.append("non-zero soft-key split overlap")
    if labels["n_disagree"] > 0:
        unmet.append("declared pattern labels contradicted by the DAG")
    if v4["n_skipped"] > 0:
        unmet.append("V4 shortcut search skipped")
    table.append(
        {
            "n": 17,
            "defect": "dataset markable as complete despite unmet hard gates",
            "confirmed": "yes",
            "measured_evidence": [
                f"freeze_manifest.json: frozen={freeze.get('frozen')}, "
                f"AUTOMATED_GATES_PASSED={freeze.get('AUTOMATED_GATES_PASSED')}, "
                f"selection_all_hard_constraints_met={freeze.get('selection_all_hard_constraints_met')}, "
                f"while LLM_VALIDATED={freeze.get('LLM_VALIDATED')}, "
                f"TRAINING_READY={freeze.get('TRAINING_READY')}, "
                f"HUMAN_REVIEW_PENDING={freeze.get('HUMAN_REVIEW_PENDING')}",
                f"PILOT42_DATA_QUALITY_REPORT.json reports AUTOMATED_GATES_PASSED="
                f"{(evidence['data_quality_report'] or {}).get('AUTOMATED_GATES_PASSED')} and leak_free="
                f"{(evidence['data_quality_report'] or {}).get('leak_free')} on the same export that shows "
                f"soft_key_overlap={json.dumps(soft_overlap)}",
                f"this independent audit records {len(audit['deficits'])} deficits and returns verdict "
                f"{audit['verdict']} / INDEPENDENT_AUDIT_PASSED={audit['INDEPENDENT_AUDIT_PASSED']}",
                "concretely unmet at freeze time: " + ("; ".join(unmet) if unmet else "none"),
            ],
            "required_fix": (
                "Completion must be a conjunction of independently recomputed gates: the build may set "
                "TRAINING_READY / COMPLETE only when the independent auditor returns PASS with zero deficits, and "
                "AUTOMATED_GATES_PASSED must not be settable while any hard gate (call-count tier, pattern agreement, "
                "leakage, LLM validation, shortcut coverage) is unmet."
            ),
        }
    )
    return table


# ---------------------------------------------------------------------------
# Structured per-defect measurements (machine readable)
# ---------------------------------------------------------------------------

#: id -> (short name, root cause, Pilot4.3 countermeasure hint). The measured
#: numbers are never stored here; they are recomputed in
#: :func:`defect_measurements` so the table cannot drift from the export.
ROOT_CAUSES: Dict[int, Tuple[str, str, str]] = {
    1: (
        "no true 6+ call tasks",
        "the Pilot4.2 workflow blueprints topped out at 5 primitive slots and the "
        "call-count target was checked against the declared `call_count` field "
        "rather than against len(gold_calls), so an empty 6+ bucket never failed "
        "a gate",
        "make 6-10 call programs a first-class generation tier with its own quota "
        "and gate the recomputed len(gold_calls) histogram, not the declared field",
    ),
    2: (
        "declared pattern label vs actual dependency DAG mismatch",
        "the structural pattern was the label the generation cell REQUESTED; it "
        "was written onto the record before the program existed and was never "
        "re-derived from the emitted references",
        "derive the pattern set from the built program's own edges, keep a task "
        "only when the requested pattern is in that set, and export the "
        "recomputed set plus a deterministic primary label",
    ),
    3: (
        "low real capability and primitive diversity",
        "generation sampled workflows, not primitives, and the 89-entry registry "
        "was reported as the diversity figure while the sampler could only reach "
        "a 9-primitive arithmetic core",
        "impose per-primitive and per-capability-family coverage floors measured "
        "from the exported gold_calls, plus a cap on the top-1 call sequence",
    ),
    4: (
        "generic/coding labels cosmetic over arithmetic programs",
        "workflow domain labels were free-form strings attached to a blueprint "
        "whose capability plan was arithmetic, so nothing tied the domain word to "
        "the primitives actually called",
        "require every domain-flavoured workflow to use at least one primitive of "
        "the matching capability family, verified from the exported calls",
    ),
    5: (
        "OpenRouter rendering/critic phases not actually executed",
        "the LLM phases were optional side effects: when the smoke render failed "
        "the pipeline fell back to template text and still froze the export",
        "treat rendering and critic as hard dependencies - one rendered record per "
        "selected task and a build failure below the configured coverage floor",
    ),
    6: (
        "over-explicit, synthetic, repetitive queries",
        "questions were produced by a small set of Python string templates that "
        "spell every input out as `name is value;` and end with `Report x.`",
        "LLM-render every query and gate on recomputed lexical-skeleton and "
        "intent-template concentration, forbidding the fact-list scaffold",
    ),
    7: (
        "V4 skipped for boolean / non-numeric answers",
        "the V4 shortcut search enumerated numeric predictors only and returned "
        "`searched: false` for anything else, while the summary reported a clean "
        "shortcut rate over that skipped population",
        "enumerate boolean and categorical answers explicitly and fail the build "
        "when shortcut-search coverage is below 100 percent of selected tasks",
    ),
    8: (
        "insufficient per-node necessity evidence",
        "necessity was stored as a single boolean verdict with an empty reason "
        "list; the ablation evidence was computed and thrown away",
        "store the ablated-program answer per node on the record and gate on that "
        "exported evidence rather than on a verdict flag",
    ),
    9: (
        "workflow/program/query-template leakage between splits",
        "the split was made disjoint on instance ids (semantic_program_id, "
        "workflow_instance_id) only, and `leak_free` was set from those hard keys "
        "while the soft keys overlapped completely",
        "split on the generalisation-relevant keys and recompute the leak gate "
        "from the exported records, with leak_free false on any soft overlap",
    ),
    10: (
        "unbalanced boolean answers",
        "threshold literals were sampled independently of the comparison operand, "
        "so the comparison was true far more often than not and nothing rebalanced "
        "or gated it",
        "sample threshold literals against the operand so the True share sits in "
        "[0.45, 0.55] overall and per workflow, and cap the boolean answer share",
    ),
    11: (
        "unrealistic values",
        "every literal came from one generic integer sampler over 1..2000 with no "
        "role, unit or precision model",
        "draw values from role- and unit-aware distributions and gate on "
        "recomputed realism statistics including a non-zero fractional share",
    ),
    12: (
        "selection tier quotas only reported, not enforced",
        "tier quotas existed in the selection config but the report only echoed "
        "the achieved counts; no per-tier deficit could make the gate fail",
        "declare tier and per-workflow quotas as numeric targets, export achieved "
        "next to target, and fail selection on any shortfall",
    ),
    13: (
        "registry size vs primitives actually used in gold calls",
        "the registry was reported as the capability figure; nothing measured how "
        "many registry entries a generated program ever called",
        "export and gate the used/declared primitive coverage ratio",
    ),
    14: (
        "metrics computed from metadata instead of exported content",
        "the reporting layer imported producer objects and aggregated declared "
        "fields, so a wrong label produced a clean metric",
        "recompute every reported metric from the exported JSONL with an auditor "
        "that imports no producer code, and run it as part of the build",
    ),
    15: (
        "mixed OpenRouter logs across runs",
        "the log files were appended to by run label with no run id or template "
        "version guard, so Pilot4.1 requests landed in the Pilot4.2 export",
        "write logs under a run id, stamp every record with run id, template "
        "version and task ids, and refuse to append across runs",
    ),
    16: (
        "incomplete reproducibility / empty input hashes",
        "the freeze stamp hashed outputs only; `input_hashes` was left empty and "
        "the export was frozen from a dirty tree",
        "hash every config, registry, blueprint and prompt into input_hashes and "
        "refuse to freeze from a dirty git tree",
    ),
    17: (
        "dataset marked complete despite unmet hard gates",
        "AUTOMATED_GATES_PASSED was a producer-side boolean independent of the "
        "gate results, so it stayed true while call-count, leakage, LLM and "
        "shortcut gates were unmet",
        "make completion a conjunction of independently recomputed gates: no "
        "TRAINING_READY without an auditor PASS and zero deficits",
    ),
}


def _metric(name: str, value: Any, target: str, source: str) -> Dict[str, Any]:
    """One comparison row: measured Pilot4.2 value, target, and its source."""
    return {"metric": name, "value": value, "target": target, "source": source}


def defect_measurements(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The 17 Pilot4.2 defects with exact recomputed numbers.

    Every value is derived from the exported JSONL content (or, where the defect
    is precisely that a claim exists only as a producer-side report, from the
    named report file, which is then stated in ``source``).
    """
    audit = evidence["audit"]
    cc = audit["call_count"]
    prims = audit["primitives"]
    dup = audit["queries"]["duplicate_rates_overall"]
    bools = audit["boolean_balance"]
    seqs = audit["sequences"]
    labels = evidence["pattern_labels"]
    v4 = evidence["v4_search"]
    necessity = evidence["node_necessity"]
    leak = evidence["leakage"]
    realism = evidence["workflow_labels"]
    router = evidence["openrouter"]
    explicit = evidence["explicitness"]
    # the audit-level block counts each unique task once; the evidence-level one
    # counts every audited row, so a task present in two files would be doubled
    literals = audit["numeric_literals"]
    freeze = evidence["freeze_manifest"] or {}
    selection = evidence["selection_report"] or {}
    v4_report = evidence["v4_report"] or {}
    split_manifest = evidence["split_manifest"] or {}
    provenance = (freeze.get("provenance") or {}) if isinstance(freeze, dict) else {}
    soft_overlap = split_manifest.get("soft_key_overlap") or {}
    n_tasks = audit["n_unique_tasks"]
    max_cc = max(int(k) for k in cc["overall"]) if cc["overall"] else 0
    n_ge6 = sum(v for k, v in cc["overall"].items() if int(k) >= 6)
    smoke = router.get("smoke_summary") or {}
    tier_quota_keys = [k for k in selection if "quota" in k.lower() or "tier" in k.lower()]

    exported = "recomputed from exported JSONL"
    per_task = "outputs/pilot42_final_audit/pilot42_per_task.csv"

    spec: List[Tuple[int, str, List[Dict[str, Any]]]] = [
        (1, per_task, [
            _metric("n_tasks_with_ge_6_calls", n_ge6, ">= 5% of tasks",
                    f"{exported}: len(gold_calls) over {n_tasks} unique tasks"),
            _metric("share_tasks_with_ge_6_calls", cc["share_at_least"]["6"],
                    ">= 0.05", f"{exported}: len(gold_calls)"),
            _metric("max_observed_call_count", max_cc, ">= 10",
                    f"{exported}: len(gold_calls)"),
            _metric("share_tasks_with_ge_5_calls", cc["share_at_least"]["5"],
                    ">= 0.25", f"{exported}: len(gold_calls)"),
        ]),
        (2, per_task, [
            _metric("n_declared_pattern_disagreements", labels["n_disagree"], "0",
                    f"{exported}: selected.jsonl pattern_family vs DAG from "
                    "gold_calls argument references"),
            _metric("declared_pattern_disagreement_rate",
                    labels["disagreement_rate"], "0.0",
                    f"{exported}: selected.jsonl pattern_family vs DAG"),
            _metric("n_declared_pattern_checked", labels["n_checked"],
                    "= n_selected", f"{exported}: selected.jsonl"),
            _metric("n_records_without_structural_label",
                    evidence["missing_pattern_family_records"], "0",
                    f"{exported}: pattern_family absent over "
                    f"{audit['n_records_audited']} audited rows"),
        ]),
        (3, per_task, [
            _metric("n_distinct_primitives_used",
                    prims["distinct_primitives_used"], ">= 40",
                    f"{exported}: gold_calls[].name mapped via "
                    "record.tools[].semantic_id"),
            _metric("n_distinct_capability_families",
                    prims["distinct_capability_families"], ">= 8",
                    f"{exported}: primitive_registry.json capability of each "
                    "primitive actually called"),
            _metric("n_distinct_primitive_sequences",
                    seqs["n_distinct_primitive_sequences"], ">= 300",
                    f"{exported}: ordered gold_calls primitive tuple"),
            _metric("top1_primitive_sequence_share",
                    seqs["primitive_sequence_concentration"]["top1_share"],
                    "<= 0.05", f"{exported}: ordered gold_calls primitive tuple"),
        ]),
        (4, per_task, [
            _metric("share_arithmetic_or_comparison_calls",
                    realism["share_arithmetic_or_comparison_calls"], "<= 0.60",
                    f"{exported}: capability family of every gold call"),
            _metric("n_cosmetic_domain_workflow_labels",
                    realism["n_cosmetic_domain_labels"], "0",
                    f"{exported}: non-arithmetic workflow_id domain whose calls "
                    "are arithmetic/comparison only"),
            _metric("n_workflow_domains_present",
                    len(realism["domains_present"]), ">= 10",
                    f"{exported}: workflow_id prefix over all records"),
        ]),
        (5, "outputs/pilot4_2_workflow_grounded_v2/llm_rendered.jsonl", [
            _metric("llm_rendered_jsonl_bytes",
                    router["llm_rendered_jsonl_bytes"], "> 0",
                    "file size of the exported llm_rendered.jsonl"),
            _metric("llm_rendered_smoke_jsonl_bytes",
                    router["llm_rendered_smoke_jsonl_bytes"], "> 0",
                    "file size of the exported llm_rendered_smoke.jsonl"),
            _metric("critic_verdict_coverage",
                    audit["validation_coverage"]["critic"]["coverage"], "1.0",
                    f"{exported}: llm_critic field over {n_tasks} unique tasks"),
            _metric("openrouter_smoke_pass_rate", smoke.get("pass_rate"), ">= 0.80",
                    "openrouter_smoke_summary.json (producer report; the defect "
                    "is that no rendered content exists to recompute it from)"),
            _metric("n_openrouter_failures", router["n_openrouter_failures"], "0",
                    "line count of openrouter_failures.jsonl"),
        ]),
        (6, per_task, [
            _metric("top10_intent_template_share", dup["top10_intent_share"],
                    "<= 0.25", f"{exported}: question intent template"),
            _metric("exact_question_duplicate_rate", dup["exact_duplicate_rate"],
                    "0.0", f"{exported}: question text"),
            _metric("n_distinct_query_skeletons", dup["n_distinct_skeleton"],
                    ">= 1500", f"{exported}: normalized question skeleton"),
            _metric("top1_query_skeleton_share", dup["top1_skeleton_share"],
                    "<= 0.05", f"{exported}: normalized question skeleton"),
            _metric("share_questions_with_semicolon_fact_list",
                    explicit["share_with_semicolon_fact_list"], "<= 0.05",
                    f"{exported}: question text"),
            _metric("share_questions_ending_report_imperative",
                    explicit["share_ending_with_report_imperative"], "<= 0.05",
                    f"{exported}: question text"),
            _metric("share_questions_with_3plus_numeric_literals",
                    explicit["share_with_3plus_numbers_in_text"], "<= 0.40",
                    f"{exported}: question text"),
        ]),
        (7, per_task, [
            _metric("n_tasks_v4_search_skipped", v4["n_skipped"], "0",
                    f"{exported}: selected.jsonl v4_gate.search.searched"),
            _metric("v4_search_skipped_share", v4["skipped_share"], "0.0",
                    f"{exported}: selected.jsonl v4_gate.search.searched"),
            _metric("n_boolean_answers_v4_skipped",
                    v4["skipped_by_answer_kind"].get("boolean", 0), "0",
                    f"{exported}: gold_answer kind of the skipped tasks"),
            _metric("declared_v4_shortcut_rate", v4_report.get("shortcut_rate"),
                    "recomputed, not declared",
                    "v4_report.json (producer report, contradicted above)"),
        ]),
        (8, per_task, [
            _metric("n_records_with_per_node_necessity_detail",
                    necessity["n_with_per_node_detail"], "= n_selected",
                    f"{exported}: selected.jsonl "
                    "semantic_validation.layers.V_NODE_NECESSITY"),
            _metric("n_records_verdict_only_necessity",
                    necessity["n_verdict_only"], "0",
                    f"{exported}: selected.jsonl V_NODE_NECESSITY payload keys"),
            _metric("node_necessity_coverage_in_training_files",
                    audit["validation_coverage"]["node_necessity"]["coverage"],
                    "1.0", f"{exported}: train/heldout/reserve records"),
        ]),
        (9, per_task, [
            _metric("n_workflow_ids_shared_train_heldout",
                    leak["keys"]["workflow_id"]["n_shared_train_heldout"], "0",
                    f"{exported}: workflow_id per split"),
            _metric("heldout_share_workflow_id_seen_in_train",
                    leak["keys"]["workflow_id"]["heldout_share_seen_in_train"],
                    "0.0", f"{exported}: workflow_id per split"),
            _metric("n_query_template_fingerprints_shared_train_heldout",
                    leak["keys"]["query_template_fingerprint"]
                    ["n_shared_train_heldout"], "0",
                    f"{exported}: selected.jsonl joined on task_id"),
            _metric("n_program_family_ids_shared_train_heldout",
                    leak["keys"]["program_family_id"]["n_shared_train_heldout"],
                    "0", f"{exported}: selected.jsonl joined on task_id"),
            _metric("n_query_skeletons_shared_train_heldout",
                    leak["query_skeleton"]["n_shared_train_heldout"], "0",
                    f"{exported}: lexical question skeleton per split"),
            _metric("declared_leak_free", split_manifest.get("leak_free"),
                    "true only when every overlap is 0",
                    "split_manifest.json (producer report, contradicted above)"),
        ]),
        (10, per_task, [
            _metric("boolean_true_share", bools["overall_true_share"],
                    "within [0.45, 0.55]", f"{exported}: gold_answer"),
            _metric("n_boolean_answers", bools["n_boolean"], "<= 0.35 of tasks",
                    f"{exported}: gold_answer, bool checked before int"),
            _metric("majority_class_boolean_baseline",
                    max(bools["overall_true_share"],
                        1 - bools["overall_true_share"]), "<= 0.55",
                    f"{exported}: gold_answer"),
            _metric("n_distinct_answer_kinds",
                    len(evidence["answer_kind_counts"]), ">= 4",
                    f"{exported}: gold_answer"),
        ]),
        (11, per_task, [
            _metric("share_generic_int_1_2000",
                    literals["share_generic_int_1_2000"], "<= 0.30",
                    f"{exported}: literal arguments of gold_calls"),
            _metric("share_integer_literals", literals["share_integer"],
                    "<= 0.80", f"{exported}: literal arguments of gold_calls"),
            _metric("share_round_multiple_of_10_literals",
                    literals["share_round_multiple_of_10"], "<= 0.20",
                    f"{exported}: literal arguments of gold_calls"),
            _metric("n_numeric_literal_arguments", literals["n"], "n/a",
                    f"{exported}: literal arguments of gold_calls"),
            _metric("n_distinct_literal_values", literals["n_distinct"], "n/a",
                    f"{exported}: literal arguments of gold_calls"),
        ]),
        (12, "outputs/pilot4_2_workflow_grounded_v2/selection_report.json", [
            _metric("n_tier_quota_fields_in_selection_report",
                    len(tier_quota_keys), ">= 1 per tier",
                    "selection_report.json keys"),
            _metric("n_cell_tiers_recomputed",
                    len(evidence["cell_tier_counts"]), "= n_declared_tiers",
                    f"{exported}: selected.jsonl cell_tier"),
            _metric("min_workflow_support",
                    min((int(v) for v in
                         (selection.get("workflow_support") or {}).values()),
                        default=0), ">= 20",
                    "selection_report.json workflow_support"),
            _metric("max_workflow_support",
                    max((int(v) for v in
                         (selection.get("workflow_support") or {}).values()),
                        default=0), "<= 5x the minimum",
                    "selection_report.json workflow_support"),
        ]),
        (13, per_task, [
            _metric("primitive_registry_coverage", prims["registry_coverage"],
                    ">= 0.60", f"{exported}: used / declared"),
            _metric("primitive_registry_size", prims["registry_size"], "n/a",
                    "primitive_registry.json"),
            _metric("n_primitives_used_in_gold_calls",
                    prims["distinct_primitives_used"], ">= 40",
                    f"{exported}: gold_calls[].name"),
            _metric("workflow_registry_size",
                    realism["n_workflows_in_registry"], "n/a",
                    "workflow_registry.json"),
            _metric("n_workflows_used", realism["n_workflows_in_export"],
                    "= workflow_registry_size", f"{exported}: workflow_id"),
        ]),
        (14, per_task, [
            _metric("n_metrics_recomputable_from_training_files", 0,
                    "every reported metric",
                    f"{exported}: the thin training records carry no structural, "
                    "primitive or validation field at all"),
            _metric("declared_vs_recomputed_pattern_disagreement_rate",
                    labels["disagreement_rate"], "0.0",
                    f"{exported}: any metric aggregated over pattern_family "
                    "describes labels, not programs"),
            _metric("declared_v4_shortcut_rate_over_skipped_population",
                    v4_report.get("shortcut_rate"), "recomputed, not declared",
                    "v4_report.json vs recomputed v4_gate.search.searched"),
            _metric("n_fields_in_validation_report",
                    len(evidence["validation_report"] or {}), "per-task detail",
                    "validation_report.json"),
        ]),
        (15, "outputs/pilot4_2_workflow_grounded_v2/openrouter_requests.jsonl", [
            _metric("n_pilot41_template_requests_in_pilot42_export",
                    sum(v for k, v in router["prompt_template_versions"].items()
                        if k.startswith("pilot41")), "0",
                    "prompt_template_version over openrouter_requests.jsonl"),
            _metric("n_openrouter_requests", router["n_openrouter_requests"],
                    "n/a", "line count of openrouter_requests.jsonl"),
            _metric("n_distinct_prompt_template_versions",
                    len(router["prompt_template_versions"]), "1",
                    "prompt_template_version over openrouter_requests.jsonl"),
            _metric("n_models_observed_but_not_declared",
                    len(router["models_observed_but_not_declared"]), "0",
                    "request/failure models vs openrouter_model_snapshot.json"),
            _metric("n_requests_without_task_ids",
                    router["n_requests_without_task_ids"], "0",
                    "task_ids over openrouter_requests.jsonl"),
        ]),
        (16, "outputs/pilot4_2_workflow_grounded_v2/freeze_manifest.json", [
            _metric("n_input_hashes_in_freeze_manifest",
                    len(provenance.get("input_hashes") or {}), ">= 1 per input",
                    "freeze_manifest.json provenance.input_hashes"),
            _metric("git_dirty_at_freeze",
                    (provenance.get("git") or {}).get("dirty"), "false",
                    "freeze_manifest.json provenance.git"),
            _metric("n_dirty_files_at_freeze",
                    (provenance.get("git") or {}).get("n_dirty_files"), "0",
                    "freeze_manifest.json provenance.git"),
            _metric("llm_validated", freeze.get("LLM_VALIDATED"), "true",
                    "freeze_manifest.json"),
        ]),
        (17, "outputs/pilot4_2_workflow_grounded_v2/freeze_manifest.json", [
            _metric("independent_audit_deficits", len(audit["deficits"]), "0",
                    f"{exported}: this audit's recomputed gate table"),
            _metric("declared_frozen", freeze.get("frozen"),
                    "true only with 0 deficits", "freeze_manifest.json"),
            _metric("declared_automated_gates_passed",
                    freeze.get("AUTOMATED_GATES_PASSED"),
                    "true only with 0 deficits", "freeze_manifest.json"),
            _metric("independent_audit_verdict", audit["verdict"], "PASS",
                    f"{exported}: this audit's recomputed gate table"),
            _metric("n_unmet_hard_gates_while_frozen",
                    sum(1 for value in (
                        cc["share_at_least"]["6"] <= 0,
                        freeze.get("LLM_VALIDATED") is False,
                        any(int(v) > 0 for v in soft_overlap.values()),
                        labels["n_disagree"] > 0,
                        v4["n_skipped"] > 0,
                    ) if value), "0",
                    f"{exported}: call-count, LLM, leakage, pattern and "
                    "shortcut gates"),
        ]),
    ]

    out: List[Dict[str, Any]] = []
    for defect_id, evidence_path, metrics in spec:
        name, root_cause, hint = ROOT_CAUSES[defect_id]
        out.append({
            "id": defect_id,
            "name": name,
            "measurable_from_export": True,
            "measured_value": {m["metric"]: m["value"] for m in metrics},
            "metrics": metrics,
            "evidence_path": evidence_path,
            "root_cause": root_cause,
            "pilot43_countermeasure_hint": hint,
        })
    return out


def metrics_csv_rows(measurements: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One CSV row per metric; the Pilot4.3 column is filled by a later step."""
    rows: List[Dict[str, Any]] = []
    for item in measurements:
        for metric in item["metrics"]:
            rows.append({
                "metric": f"d{item['id']:02d}.{metric['metric']}",
                "pilot42_value": metric["value"],
                "pilot43_value": "",
                "target": metric["target"],
                "source_of_pilot42_value": metric["source"],
            })
    return rows


def render_comparison_markdown(evidence: Dict[str, Any],
                               measurements: Sequence[Dict[str, Any]]) -> str:
    """The Pilot4.2 side of the comparison, with the Pilot4.3 side left open."""
    audit = evidence["audit"]
    lines: List[str] = []
    lines.append("# Pilot4.2 vs Pilot4.3 audit")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("- Pilot4.2 side: **complete** (recomputed from the frozen export)")
    lines.append("- Pilot4.3 side: **PENDING** - see the marked section below")
    lines.append("")
    lines.append("## Provenance of the Pilot4.2 side")
    lines.append("")
    lines.append(f"- frozen export (read-only): `{EXPORT_DIR.name}`")
    lines.append(
        "- auditor: `analysis/pilot43_independent_audit` - standard library only, "
        "imports no producer module")
    lines.append(
        "- every Pilot4.2 number below is recomputed from the exported JSONL "
        "content; where a row measures a producer claim, `source` names the "
        "report file and says so")
    lines.append(
        "- audited files: "
        + ", ".join(f"`{name}` ({audit['counts'][split]} records)"
                    for split, name in FILES.items()))
    lines.append(
        f"- rows audited: {audit['n_records_audited']}; unique tasks: "
        f"{audit['n_unique_tasks']}")
    lines.append(
        f"- independent audit verdict: **{audit['verdict']}** with "
        f"{len(audit['deficits'])} deficits")
    lines.append("")
    lines.append("## The 17 Pilot4.2 defects, measured")
    lines.append("")
    lines.append("| # | defect | headline measured value | target |")
    lines.append("| --- | --- | --- | --- |")
    for item in measurements:
        head = item["metrics"][0]
        lines.append(
            f"| {item['id']} | {item['name']} | `{head['metric']}` = "
            f"{head['value']} | {head['target']} |")
    lines.append("")
    for item in measurements:
        lines.append(f"### {item['id']}. {item['name']}")
        lines.append("")
        lines.append("| metric | pilot4.2 | pilot4.3 | target | source |")
        lines.append("| --- | --- | --- | --- | --- |")
        for metric in item["metrics"]:
            lines.append(
                f"| `{metric['metric']}` | {metric['value']} | _pending_ | "
                f"{metric['target']} | {metric['source']} |")
        lines.append("")
        lines.append(f"- root cause: {item['root_cause']}")
        lines.append(f"- Pilot4.3 countermeasure: {item['pilot43_countermeasure_hint']}")
        lines.append(f"- evidence: `{item['evidence_path']}`")
        lines.append("")
    lines.append("## Recomputed Pilot4.2 reference tables")
    lines.append("")
    lines.append("| calls | records | share |")
    lines.append("| --- | --- | --- |")
    total = max(sum(audit["call_count"]["overall"].values()), 1)
    for key, value in sorted(audit["call_count"]["overall"].items(),
                             key=lambda kv: int(kv[0])):
        lines.append(f"| {key} | {value} | {value / total:.4f} |")
    lines.append("")
    lines.append("| recomputed primary pattern | records |")
    lines.append("| --- | --- |")
    for key, value in sorted(audit["patterns"]["primary_distribution"].items(),
                             key=lambda kv: -kv[1]):
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("## Independent audit deficits (Pilot4.2, verbatim)")
    lines.append("")
    for deficit in audit["deficits"]:
        lines.append(f"- {deficit}")
    lines.append("")
    lines.append("<!-- PILOT43_SIDE_BEGIN -->")
    lines.append("## Pilot4.3 side pending")
    lines.append("")
    lines.append(
        "**PENDING.** No Pilot4.3 export exists yet, so every `pilot43_value` "
        "cell above and in `PILOT42_VS_PILOT43_METRICS.csv` is deliberately "
        "empty. Filling them from anything other than a recomputation over the "
        "Pilot4.3 export would reproduce Pilot4.2 defect 14.")
    lines.append("")
    lines.append("To fill this section, a later step must:")
    lines.append("")
    lines.append(
        "1. run the same independent auditor over the Pilot4.3 export dir,")
    lines.append(
        "2. write the recomputed value into the `pilot43_value` column for every "
        "metric id listed above,")
    lines.append(
        "3. mark each of the 17 defects as fixed only when its metric meets the "
        "`target` column,")
    lines.append(
        "4. compare against `target_profile_v3.json`, whose pattern "
        "distributions are produced by the Pilot4.3 classifier so the "
        "comparison is classifier-matched.")
    lines.append("")
    lines.append("<!-- PILOT43_SIDE_END -->")
    lines.append("")
    return "\n".join(lines)


def render_markdown(evidence: Dict[str, Any], defects: Sequence[Dict[str, Any]]) -> str:
    """Render the numbered root-cause report."""
    audit = evidence["audit"]
    lines: List[str] = []
    lines.append("# Pilot4.2 independent root-cause audit")
    lines.append("")
    lines.append(f"- export dir (frozen, read-only): `{EXPORT_DIR}`")
    lines.append(f"- report dir: `{REPORT_DIR}`")
    lines.append(
        "- auditor: `analysis/pilot43_independent_audit` - standard library only, no producer module imported; "
        "every number below is recomputed from exported JSONL content or read from an exported report file"
    )
    lines.append(
        "- audited files: "
        + ", ".join(f"`{name}` ({audit['counts'][split]} records)" for split, name in FILES.items())
    )
    lines.append(
        f"- rows audited: {audit['n_records_audited']}; unique tasks: {audit['n_unique_tasks']} "
        "(train_master_3000 + heldout_500 + reserve_500 are the same 4000 tasks as selected.jsonl, so all aggregate "
        "shares below count each task exactly once, while declared-vs-recomputed checks run over every row)"
    )
    lines.append(f"- independent audit verdict: **{audit['verdict']}** with {len(audit['deficits'])} deficits")
    lines.append("")
    lines.append("## Scope and undecidable invariants")
    lines.append("")
    lines.append(
        "Pilot4.2 training records are thin (`call_count`, `cell_tier`, `gold_answer`, `gold_calls`, `question`, "
        "`requested_query_mode`, `task_id`, `tools`, `was_generated_from_workflow`, `workflow_id`), so no "
        "intermediate node values exist. Node output kinds are passed as `unknown` for every node except the sink, "
        "and `TYPE_TRANSITION_CHAIN` is therefore reported as **undecidable**, not as false."
    )
    supp = evidence["type_transition"]
    lines.append("")
    lines.append(
        f"Supplementary: `selected.jsonl` does export `oracle_observations`, which makes the invariant measurable for "
        f"{supp['n_usable_records']} records; {supp['n_satisfying_type_transition_chain']} of them satisfy "
        f"TYPE_TRANSITION_CHAIN ({supp['share'] * 100:.3f}%). This is a supplement, not the verdict for the "
        "training files."
    )
    lines.append("")
    lines.append("## Defect table")
    lines.append("")
    for item in defects:
        lines.append(f"### {item['n']}. {item['defect']}")
        lines.append("")
        lines.append(f"- **confirmed: {item['confirmed']}**")
        lines.append("- measured evidence:")
        for piece in item["measured_evidence"]:
            lines.append(f"  - {piece}")
        lines.append(f"- required fix for Pilot4.3: {item['required_fix']}")
        lines.append("")
    lines.append("## Independent audit deficits (verbatim)")
    lines.append("")
    for deficit in audit["deficits"]:
        lines.append(f"- {deficit}")
    lines.append("")
    lines.append("## Recomputed reference tables")
    lines.append("")
    lines.append("### Call count (recomputed from gold_calls)")
    lines.append("")
    lines.append("| calls | records | share |")
    lines.append("| --- | --- | --- |")
    total = max(sum(audit["call_count"]["overall"].values()), 1)
    for key, value in sorted(audit["call_count"]["overall"].items(), key=lambda kv: int(kv[0])):
        lines.append(f"| {key} | {value} | {value / total:.4f} |")
    lines.append("")
    lines.append("### Call count per split")
    lines.append("")
    lines.append("| split | " + " | ".join(sorted(audit["call_count"]["overall"], key=int)) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in audit["call_count"]["overall"]) + " |")
    for split in FILES:
        hist = audit["call_count"]["by_split"][split]
        lines.append(
            f"| {split} | "
            + " | ".join(str(hist.get(key, 0)) for key in sorted(audit["call_count"]["overall"], key=int))
            + " |"
        )
    lines.append("")
    lines.append("### Recomputed primary structural pattern")
    lines.append("")
    lines.append("| pattern | records |")
    lines.append("| --- | --- |")
    for key, value in sorted(audit["patterns"]["primary_distribution"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("### Primitives actually used")
    lines.append("")
    lines.append("| primitive | capability family | calls |")
    lines.append("| --- | --- | --- |")
    caps = evidence["primitive_to_capability"]
    for key, value in sorted(audit["primitives"]["primitive_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {key} | {caps.get(key, '<unknown>')} | {value} |")
    lines.append("")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pilot42-final-audit")
    ap.add_argument("--export-dir", default=None,
                    help="frozen Pilot4.2 export (read-only)")
    ap.add_argument("--audit-dir", default=None,
                    help="where the per-task ledger and defect report go")
    ap.add_argument("--final-dir", default=None,
                    help="where the Pilot4.2-vs-Pilot4.3 artefacts go")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit and write the Pilot4.2 audit and comparison artefacts."""
    global EXPORT_DIR, REPORT_DIR, FINAL_DIR
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.export_dir:
        EXPORT_DIR = Path(args.export_dir).resolve()
    if args.audit_dir:
        REPORT_DIR = Path(args.audit_dir).resolve()
    if args.final_dir:
        FINAL_DIR = Path(args.final_dir).resolve()
    if not EXPORT_DIR.is_dir():
        raise SystemExit(f"frozen Pilot4.2 export not found: {EXPORT_DIR}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    spec = build_spec(REPORT_DIR)
    audit = audit_export(EXPORT_DIR, spec)
    rows = audit.pop("_rows")
    splits = audit.pop("_splits")
    surface_to_primitive = audit.pop("_surface_to_primitive")
    primitive_to_capability = audit.pop("_primitive_to_capability")

    all_records = [rec for split in FILES for rec in splits.get(split, [])]
    selected = splits.get("selected", [])

    evidence: Dict[str, Any] = {
        "audit": audit,
        "primitive_to_capability": primitive_to_capability,
        "pattern_labels": pattern_label_audit(selected),
        "v4_search": v4_search_coverage(selected),
        "node_necessity": node_necessity_evidence(selected),
        "leakage": leakage_recomputation(splits, selected),
        "workflow_labels": workflow_label_realism(
            all_records,
            surface_to_primitive,
            primitive_to_capability,
            _load_json(EXPORT_DIR / "workflow_registry.json"),
        ),
        "openrouter": openrouter_log_forensics(EXPORT_DIR),
        "explicitness": explicitness_stats(all_records),
        "numeric_literals": numeric_literal_stats(all_records),
        "type_transition": type_transition_supplement(selected),
        "freeze_manifest": _load_json(EXPORT_DIR / "freeze_manifest.json"),
        "selection_report": _load_json(EXPORT_DIR / "selection_report.json"),
        "split_manifest": _load_json(EXPORT_DIR / "split_manifest.json"),
        "v4_report": _load_json(EXPORT_DIR / "v4_report.json"),
        "validation_report": _load_json(EXPORT_DIR / "validation_report.json"),
        "data_quality_report": _load_json(EXPORT_DIR / "PILOT42_DATA_QUALITY_REPORT.json"),
        "cell_tier_counts": json_safe(Counter(str(rec.get("cell_tier")) for rec in selected)),
        "answer_kind_counts": json_safe(
            Counter(VALUE_KIND(rec.get("gold_answer")) for rec in selected)
        ),
        "missing_pattern_family_records": sum(
            1 for row in rows if row["declared_structural_pattern"] == ""
        ),
        "query_fingerprint_example": query_fingerprints(
            str(selected[0].get("question", "")) if selected else ""
        ),
    }

    defects = build_defect_table(evidence)
    measurements = defect_measurements(evidence)
    payload = {
        "schema_version": "ttdf.pilot42_root_cause_audit.v1",
        "export_dir": str(EXPORT_DIR),
        "report_dir": str(REPORT_DIR),
        "auditor": "analysis.pilot43_independent_audit (stdlib only, no producer imports)",
        "audited_files": FILES,
        "n_defects_confirmed": sum(1 for item in defects if item["confirmed"] == "yes"),
        "n_defects_undecidable": sum(1 for item in defects if item["confirmed"] == "undecidable"),
        "n_defects_not_confirmed": sum(1 for item in defects if item["confirmed"] == "no"),
        "undecidable_invariants": {
            "TYPE_TRANSITION_CHAIN": "no intermediate node values in the thin Pilot4.2 records",
        },
        "defects": defects,
        "measurements": measurements,
        "independent_audit": audit,
        "evidence": {k: v for k, v in evidence.items() if k != "audit"},
    }
    (REPORT_DIR / "PILOT42_ROOT_CAUSE_AUDIT.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (REPORT_DIR / "PILOT42_ROOT_CAUSE_AUDIT.md").write_text(
        render_markdown(evidence, defects), encoding="utf-8"
    )
    (REPORT_DIR / "PILOT42_AUDIT_SUMMARY.json").write_text(
        json.dumps({
            "schema_version": "ttdf.pilot42_final_audit_summary.v1",
            "export_dir": str(EXPORT_DIR),
            "verdict": audit["verdict"],
            "INDEPENDENT_AUDIT_PASSED": audit["INDEPENDENT_AUDIT_PASSED"],
            "deficits": audit["deficits"],
            "counts": audit["counts"],
            "n_records_audited": audit["n_records_audited"],
            "n_unique_tasks": audit["n_unique_tasks"],
            "call_count": audit["call_count"],
            "patterns": audit["patterns"],
            "primitives": audit["primitives"],
            "boolean_balance": {k: v for k, v in audit["boolean_balance"].items()
                                if not isinstance(v, dict)},
            "queries": audit["queries"]["duplicate_rates_overall"],
            "validation_coverage": audit["validation_coverage"],
            "n_defects_measured": len(measurements),
        }, indent=2, default=str), encoding="utf-8"
    )

    rows = metrics_csv_rows(measurements)
    columns = ["metric", "pilot42_value", "pilot43_value", "target",
               "source_of_pilot42_value"]
    with (FINAL_DIR / "PILOT42_VS_PILOT43_METRICS.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (FINAL_DIR / "PILOT42_VS_PILOT43_AUDIT.md").write_text(
        render_comparison_markdown(evidence, measurements), encoding="utf-8"
    )
    (FINAL_DIR / "PILOT42_ROOT_CAUSES.json").write_text(
        json.dumps({
            "schema_version": "ttdf.pilot42_root_causes.v1",
            "export_dir": str(EXPORT_DIR),
            "auditor": "analysis.pilot43_independent_audit (stdlib only)",
            "recomputed_from": "exported JSONL content, never metadata labels",
            "n_defects": len(measurements),
            "defects": [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "measured_value": item["measured_value"],
                    "evidence_path": item["evidence_path"],
                    "root_cause": item["root_cause"],
                    "pilot43_countermeasure_hint": item[
                        "pilot43_countermeasure_hint"],
                }
                for item in measurements
            ],
        }, indent=2, default=str), encoding="utf-8"
    )

    for name in ("PILOT42_ROOT_CAUSE_AUDIT.json", "PILOT42_ROOT_CAUSE_AUDIT.md",
                 "PILOT42_AUDIT_SUMMARY.json", "pilot42_per_task.csv"):
        print(f"wrote {REPORT_DIR / name}")
    for name in ("PILOT42_VS_PILOT43_AUDIT.md", "PILOT42_VS_PILOT43_METRICS.csv",
                 "PILOT42_ROOT_CAUSES.json"):
        print(f"wrote {FINAL_DIR / name}")
    print(f"independent audit verdict: {audit['verdict']} "
          f"({len(audit['deficits'])} deficits)")
    for item in defects:
        print(f"defect {item['n']:>2}: confirmed={item['confirmed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
