"""Independent audit driver over an exported dataset directory.

The auditor loads exported JSONL splits, re-derives every structural and
statistical property from the record CONTENT, compares the result against the
values DECLARED in the records, and emits a per-task CSV plus a JSON and a
Markdown report. It imports nothing from the producer packages.

Spec format (all keys optional unless marked required)::

    {
      "run_label": "pilot4_2_workflow_grounded_v2",
      "files": {"train": "train_master_3000.jsonl", ...},   # required
      "train_split": "train",
      "expected_counts": {"train": 3000, ...},
      "declared_paths": {                 # logical name -> dotted record path
        "call_count": "call_count",
        "structural_pattern": "declared.structural_pattern",
        "actual_structural_pattern": "actual.structural_pattern",
        "answer_type": "answer_type",
        "workflow_id": "workflow_id",
        "cell_tier": "cell_tier",
        "query_mode": "requested_query_mode"
      },
      "validation_paths": {
        "v4": "validation.v4",
        "critic": "validation.critic",
        "node_necessity": "validation.node_necessity"
      },
      "overlap_keys": ["workflow_id"],
      "overlap_against": ["heldout"],     # splits compared with train; default: all others
      "dedupe_key": "task_id",            # aggregate tables count each task once
      "node_value_kinds": {"mode": "sink_only"},   # sink_only | from_path | none
      "surface_map": {"source": "record_tools", "semantic_id_key": "semantic_id",
                      "primitive_registry": "primitive_registry.json"},
      "thresholds": {...},
      "out_dir": null,                    # defaults to export_dir
      "emit": {"csv": true, "json": true, "md": true},
      "csv_name": "independent_audit_per_task.csv",
      "report_prefix": "PILOT43_INDEPENDENT_AUDIT",
      "text_key": "question"
    }

Tolerances: counts must match exactly, recomputed rates are compared to
thresholds within 0.1 percentage point, and per-task pattern label plus call
count must match exactly.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .graph_recon import Graph, ReconError, reconstruct
from .metrics import (
    MISSING,
    boolean_balance,
    capability_usage,
    duplicate_rates,
    get_path,
    json_safe,
    normalize_answer_type,
    numeric_literal_stats,
    primitive_sequences,
    primitive_usage,
    query_fingerprints,
    recompute_call_count,
    split_overlap,
    surface_names,
    surface_to_primitive_from_tools,
)
from .pattern_rules import (
    VALUE_KIND,
    primary_pattern,
    satisfied_patterns,
    undecidable_patterns,
)

#: Rates are compared within 0.1 percentage point.
RATE_TOLERANCE = 0.001

CSV_COLUMNS: Tuple[str, ...] = (
    "task_id",
    "split",
    "recon_ok",
    "recon_error",
    "recomputed_call_count",
    "declared_call_count",
    "call_count_agrees",
    "n_nodes",
    "n_edges",
    "depth",
    "n_roots",
    "n_leaves",
    "n_join_nodes",
    "n_fan_out_nodes",
    "n_reused_outputs",
    "n_late_edges",
    "max_reference_distance",
    "mean_reference_distance",
    "n_parallel_branches",
    "critical_path",
    "recomputed_primary_pattern",
    "recomputed_satisfied_patterns",
    "undecidable_patterns",
    "declared_structural_pattern",
    "pattern_agrees",
    "declared_actual_structural_pattern",
    "actual_pattern_agrees",
    "recomputed_answer_type",
    "declared_answer_type",
    "answer_type_agrees",
    "gold_tool_surfaces",
    "gold_primitives",
    "capability_families",
    "declared_workflow_id",
    "declared_cell_tier",
    "declared_query_mode",
    "question_exact_hash",
    "question_skeleton_hash",
    "question_intent_hash",
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_primitive_registry(path: Path) -> Dict[str, str]:
    """Map primitive id -> capability family from an exported registry file.

    This reads exported DATA only; the producer registry code is never imported.
    """
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    prims = payload.get("primitives", payload) if isinstance(payload, dict) else {}
    mapping: Dict[str, str] = {}
    if isinstance(prims, dict):
        for prim_id, entry in prims.items():
            if isinstance(entry, dict):
                family = entry.get("capability_family")
                if isinstance(family, str):
                    mapping[str(prim_id)] = family
    return mapping


def node_value_kinds_for(
    rec: Dict[str, Any],
    n_nodes: int,
    mode: str,
    values_path: str = "",
) -> List[str]:
    """Per-node output value kinds, according to the configured mode.

    ``sink_only``
        Only the final node's kind is known (from ``gold_answer``); every other
        node is ``"unknown"``. This is the honest choice for exports that do not
        carry intermediate node values.
    ``from_calls``
        Each gold call carries the value it produced under ``values_path`` (a
        plain field name, ``observation`` by default). Intermediate kinds are
        what decide whether a node reduces a collection, so an export that ships
        them lets the audit judge the aggregation patterns instead of declaring
        them undecidable.
    ``from_path``
        A dotted path to a list of per-node output values.
    ``none``
        Nothing is known.
    """
    unknown = ["unknown"] * n_nodes
    if n_nodes == 0:
        return unknown
    if mode == "from_calls":
        key = values_path or "observation"
        calls = rec.get("gold_calls") or []
        if len(calls) != n_nodes:
            return unknown
        return [VALUE_KIND(c.get(key)) if isinstance(c, dict) and key in c
                else "unknown" for c in calls]
    if mode == "from_path" and values_path:
        values = get_path(rec, values_path)
        if isinstance(values, list) and len(values) == n_nodes:
            return [VALUE_KIND(v) for v in values]
        return unknown
    if mode == "sink_only":
        kinds = list(unknown)
        kinds[-1] = VALUE_KIND(rec.get("gold_answer"))
        return kinds
    return unknown


def _audit_one(
    rec: Dict[str, Any],
    split: str,
    declared_paths: Dict[str, str],
    kinds_mode: str,
    kinds_path: str,
    surface_to_primitive: Dict[str, str],
    primitive_to_capability: Dict[str, str],
    text_key: str,
) -> Dict[str, Any]:
    """Recompute every per-task property and compare with declared values."""
    row: Dict[str, Any] = {col: "" for col in CSV_COLUMNS}
    row["task_id"] = str(rec.get("task_id", ""))
    row["split"] = split

    calls = rec.get("gold_calls") or []
    recomputed_cc = recompute_call_count(rec)
    row["recomputed_call_count"] = recomputed_cc

    graph: Optional[Graph] = None
    try:
        graph = reconstruct(calls)
        row["recon_ok"] = True
    except ReconError as exc:
        row["recon_ok"] = False
        row["recon_error"] = str(exc)

    satisfied: set = set()
    undecidable: set = set()
    if graph is not None:
        feats = graph.features()
        kinds = node_value_kinds_for(rec, graph.n, kinds_mode, kinds_path)
        satisfied = satisfied_patterns(graph, kinds)
        undecidable = undecidable_patterns(kinds)
        row.update(
            {
                "n_nodes": feats["n_nodes"],
                "n_edges": feats["n_edges"],
                "depth": feats["depth"],
                "n_roots": feats["n_roots"],
                "n_leaves": feats["n_leaves"],
                "n_join_nodes": feats["n_join_nodes"],
                "n_fan_out_nodes": feats["n_fan_out_nodes"],
                "n_reused_outputs": feats["n_reused_outputs"],
                "n_late_edges": feats["n_late_edges"],
                "max_reference_distance": feats["max_reference_distance"],
                "mean_reference_distance": round(feats["mean_reference_distance"], 6),
                "n_parallel_branches": feats["n_parallel_branches"],
                "critical_path": "|".join(str(i) for i in feats["critical_path"]),
                "recomputed_primary_pattern": primary_pattern(satisfied),
                "recomputed_satisfied_patterns": "|".join(sorted(satisfied)),
                "undecidable_patterns": "|".join(sorted(undecidable)),
            }
        )
        prims: List[str] = []
        caps: List[str] = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name", ""))
            prim = surface_to_primitive.get(name) or call.get("primitive_id") or ""
            prims.append(str(prim))
            caps.append(primitive_to_capability.get(str(prim), ""))
        row["gold_tool_surfaces"] = "|".join(
            str(c.get("name", "")) for c in calls if isinstance(c, dict)
        )
        row["gold_primitives"] = "|".join(prims)
        row["capability_families"] = "|".join(caps)

    row["recomputed_answer_type"] = VALUE_KIND(rec.get("gold_answer"))

    missing: List[str] = []

    def declared(name: str) -> Any:
        path = declared_paths.get(name)
        if not path:
            return MISSING
        value = get_path(rec, path)
        if value is MISSING:
            missing.append(path)
        return value

    dcc = declared("call_count")
    if dcc is not MISSING:
        row["declared_call_count"] = dcc
        row["call_count_agrees"] = bool(dcc == recomputed_cc)

    dpat = declared("structural_pattern")
    if dpat is not MISSING:
        row["declared_structural_pattern"] = str(dpat)
        row["pattern_agrees"] = bool(
            str(dpat) in satisfied or str(dpat) in undecidable
        )

    dact = declared("actual_structural_pattern")
    if dact is not MISSING:
        row["declared_actual_structural_pattern"] = str(dact)
        row["actual_pattern_agrees"] = bool(
            str(dact) in satisfied or str(dact) in undecidable
        )

    dans = declared("answer_type")
    if dans is not MISSING:
        row["declared_answer_type"] = str(dans)
        row["answer_type_agrees"] = bool(
            normalize_answer_type(dans) == row["recomputed_answer_type"]
        )

    for logical, column in (
        ("workflow_id", "declared_workflow_id"),
        ("cell_tier", "declared_cell_tier"),
        ("query_mode", "declared_query_mode"),
    ):
        value = declared(logical)
        if value is not MISSING:
            row[column] = str(value)

    fp = query_fingerprints(rec.get(text_key, ""))
    row["question_exact_hash"] = fp["exact"]
    row["question_skeleton_hash"] = fp["skeleton_hash"]
    row["question_intent_hash"] = fp["intent_hash"]

    row["_missing_paths"] = missing
    row["_satisfied"] = satisfied
    row["_undecidable"] = undecidable
    return row


def _coverage(records: Sequence[Dict[str, Any]], path: str) -> Dict[str, Any]:
    """Presence and pass rate of a validation block at a dotted path."""
    n = len(records)
    present = 0
    passed = 0
    for rec in records:
        value = get_path(rec, path)
        if value is MISSING:
            continue
        present += 1
        if isinstance(value, dict):
            if value.get("passed") is True:
                passed += 1
        elif value is True:
            passed += 1
    return {
        "path": path,
        "n": n,
        "n_present": present,
        "coverage": (present / n) if n else 0.0,
        "n_passed": passed,
        "pass_rate_over_present": (passed / present) if present else 0.0,
    }


def audit_export(export_dir: Path, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Audit an exported dataset directory and emit the audit artefacts.

    Returns a dict with ``verdict``, ``INDEPENDENT_AUDIT_PASSED``, ``deficits``,
    ``disagreements`` and the full recomputed aggregate tables.
    """
    export_dir = Path(export_dir)
    files: Dict[str, str] = dict(spec.get("files") or {})
    if not files:
        raise ValueError("spec.files must list at least one split -> filename mapping")
    train_split = str(spec.get("train_split", "train"))
    declared_paths: Dict[str, str] = dict(spec.get("declared_paths") or {})
    validation_paths: Dict[str, str] = dict(
        spec.get("validation_paths")
        or {
            "v4": "validation.v4",
            "critic": "validation.critic",
            "node_necessity": "validation.node_necessity",
        }
    )
    thresholds: Dict[str, Any] = dict(spec.get("thresholds") or {})
    kinds_cfg = dict(spec.get("node_value_kinds") or {"mode": "sink_only"})
    kinds_mode = str(kinds_cfg.get("mode", "sink_only"))
    kinds_path = str(kinds_cfg.get("path", ""))
    surface_cfg = dict(spec.get("surface_map") or {})
    text_key = str(spec.get("text_key", "question"))
    emit = dict(spec.get("emit") or {})
    out_dir = Path(spec["out_dir"]) if spec.get("out_dir") else export_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    deficits: List[str] = []

    # ---- load ------------------------------------------------------------
    splits: Dict[str, List[Dict[str, Any]]] = {}
    for split, filename in files.items():
        path = export_dir / filename
        if not path.exists():
            deficits.append(f"missing_file:{filename} (split {split})")
            splits[split] = []
            continue
        splits[split] = read_jsonl(path)

    expected_counts: Dict[str, int] = dict(spec.get("expected_counts") or {})
    for split, expected in expected_counts.items():
        actual = len(splits.get(split, []))
        if actual != int(expected):
            deficits.append(
                f"count_mismatch:{split} measured {actual} vs required exactly {int(expected)}"
            )

    all_records: List[Dict[str, Any]] = []
    record_split: List[str] = []
    for split in files:
        for rec in splits.get(split, []):
            all_records.append(rec)
            record_split.append(split)

    # Audited files may overlap (for example a union file next to its splits).
    # Per-task checks run over every audited row, but aggregate distributions
    # count each task once so that shares are not silently multiplied.
    dedupe_key = str(spec.get("dedupe_key", "") or "")
    if dedupe_key:
        seen: set = set()
        unique_records: List[Dict[str, Any]] = []
        unique_mask: List[bool] = []
        for rec in all_records:
            value = get_path(rec, dedupe_key)
            token = str(value) if value is not MISSING else None
            if token is not None and token in seen:
                unique_mask.append(False)
                continue
            if token is not None:
                seen.add(token)
            unique_mask.append(True)
            unique_records.append(rec)
    else:
        unique_records = list(all_records)
        unique_mask = [True] * len(all_records)

    # ---- surface -> primitive map (exported data only) -------------------
    primitive_registry_name = str(surface_cfg.get("primitive_registry", "primitive_registry.json"))
    primitive_to_capability = load_primitive_registry(export_dir / primitive_registry_name)
    semantic_id_key = str(surface_cfg.get("semantic_id_key", "semantic_id"))
    derived = surface_to_primitive_from_tools(all_records, semantic_id_key)
    surface_to_primitive: Dict[str, str] = {}
    mapping_source = "none"
    if surface_cfg.get("source", "record_tools") == "record_tools" and derived["mapping"]:
        known = {
            name: prim
            for name, prim in derived["mapping"].items()
            if not primitive_to_capability or prim in primitive_to_capability
        }
        if known:
            surface_to_primitive = known
            mapping_source = f"record.tools[].{semantic_id_key} + {primitive_registry_name}"
    explicit = surface_cfg.get("mapping")
    if isinstance(explicit, dict) and explicit:
        surface_to_primitive = {str(k): str(v) for k, v in explicit.items()}
        mapping_source = "spec.surface_map.mapping"
    primitive_mapping_available = bool(surface_to_primitive)

    # ---- per task --------------------------------------------------------
    rows: List[Dict[str, Any]] = []
    for rec, split in zip(all_records, record_split):
        rows.append(
            _audit_one(
                rec,
                split,
                declared_paths,
                kinds_mode,
                kinds_path,
                surface_to_primitive,
                primitive_to_capability,
                text_key,
            )
        )

    missing_counts: Counter = Counter()
    for row in rows:
        for path in row["_missing_paths"]:
            missing_counts[path] += 1
    for path, count in sorted(missing_counts.items()):
        deficits.append(
            f"missing_field:{path} (absent in {count}/{len(rows)} records)"
        )

    recon_failures = [row for row in rows if row["recon_ok"] is not True]
    if recon_failures:
        deficits.append(
            f"graph_reconstruction_failed measured {len(recon_failures)} tasks vs required exactly 0"
        )

    cc_checked = [row for row in rows if row["call_count_agrees"] != ""]
    cc_bad = [row for row in cc_checked if row["call_count_agrees"] is False]
    pat_checked = [row for row in rows if row["pattern_agrees"] != ""]
    pat_bad = [row for row in pat_checked if row["pattern_agrees"] is False]
    act_checked = [row for row in rows if row["actual_pattern_agrees"] != ""]
    act_bad = [row for row in act_checked if row["actual_pattern_agrees"] is False]
    ans_checked = [row for row in rows if row["answer_type_agrees"] != ""]
    ans_bad = [row for row in ans_checked if row["answer_type_agrees"] is False]

    disagreements = {
        "call_count": {
            "n_checked": len(cc_checked),
            "n_disagree": len(cc_bad),
            "rate": (len(cc_bad) / len(cc_checked)) if cc_checked else 0.0,
            "examples": [row["task_id"] for row in cc_bad[:10]],
        },
        "structural_pattern": {
            "n_checked": len(pat_checked),
            "n_disagree": len(pat_bad),
            "rate": (len(pat_bad) / len(pat_checked)) if pat_checked else 0.0,
            "examples": [
                {
                    "task_id": row["task_id"],
                    "declared": row["declared_structural_pattern"],
                    "recomputed_primary": row["recomputed_primary_pattern"],
                    "recomputed_satisfied": row["recomputed_satisfied_patterns"],
                }
                for row in pat_bad[:10]
            ],
        },
        "actual_structural_pattern": {
            "n_checked": len(act_checked),
            "n_disagree": len(act_bad),
            "rate": (len(act_bad) / len(act_checked)) if act_checked else 0.0,
            "examples": [row["task_id"] for row in act_bad[:10]],
        },
        "answer_type": {
            "n_checked": len(ans_checked),
            "n_disagree": len(ans_bad),
            "rate": (len(ans_bad) / len(ans_checked)) if ans_checked else 0.0,
            "examples": [row["task_id"] for row in ans_bad[:10]],
        },
    }
    if cc_bad:
        deficits.append(
            f"declared_call_count_disagreement measured {len(cc_bad)}/{len(cc_checked)} tasks vs required exactly 0"
        )
    if pat_bad:
        deficits.append(
            f"declared_pattern_disagreement measured {len(pat_bad)}/{len(pat_checked)} tasks vs required exactly 0"
        )
    if act_bad:
        deficits.append(
            f"actual_pattern_disagreement measured {len(act_bad)}/{len(act_checked)} tasks vs required exactly 0"
        )
    if ans_bad:
        deficits.append(
            f"declared_answer_type_disagreement measured {len(ans_bad)}/{len(ans_checked)} tasks vs required exactly 0"
        )

    # ---- aggregates ------------------------------------------------------
    unique_rows = [row for row, keep in zip(rows, unique_mask) if keep]
    call_counts_by_split: Dict[str, Counter] = {
        split: Counter(row["recomputed_call_count"] for row in rows if row["split"] == split)
        for split in files
    }
    call_counts_by_tier: Dict[str, Counter] = defaultdict(Counter)
    for row in unique_rows:
        call_counts_by_tier[row["declared_cell_tier"] or "<missing>"][
            row["recomputed_call_count"]
        ] += 1

    overall_cc: Counter = Counter(row["recomputed_call_count"] for row in unique_rows)
    n_rows = max(len(unique_rows), 1)
    call_count_shares = {str(k): v / n_rows for k, v in sorted(overall_cc.items())}
    share_ge = {
        str(k): sum(v for cc, v in overall_cc.items() if cc >= k) / n_rows
        for k in (2, 3, 4, 5, 6, 7, 8)
    }

    pattern_by_call_count: Dict[str, Counter] = defaultdict(Counter)
    for row in unique_rows:
        pattern_by_call_count[str(row["recomputed_call_count"])][
            row["recomputed_primary_pattern"] or "UNCLASSIFIED"
        ] += 1
    pattern_totals: Counter = Counter(
        row["recomputed_primary_pattern"] or "UNCLASSIFIED" for row in unique_rows
    )
    satisfied_totals: Counter = Counter()
    undecidable_totals: Counter = Counter()
    for row in unique_rows:
        for name in row["_satisfied"]:
            satisfied_totals[name] += 1
        for name in row["_undecidable"]:
            undecidable_totals[name] += 1

    usage = primitive_usage(unique_records, surface_to_primitive or None)
    caps = capability_usage(unique_records, primitive_to_capability, surface_to_primitive or None)
    surfaces = surface_names(unique_records)
    seqs = primitive_sequences(
        unique_records, surface_to_primitive or None, primitive_to_capability or None
    )
    dup = duplicate_rates(unique_records, text_key)
    dup_by_split = {
        split: duplicate_rates(splits.get(split, []), text_key) for split in files
    }
    bools = boolean_balance(
        unique_records,
        declared_paths.get("workflow_id", "workflow_id"),
        declared_paths.get("cell_tier", "cell_tier"),
        text_key,
    )
    query_modes: Counter = Counter(
        row["declared_query_mode"] or "<missing>" for row in unique_rows
    )
    cell_support: Counter = Counter(
        row["declared_cell_tier"] or "<missing>" for row in unique_rows
    )
    workflow_support: Counter = Counter(
        row["declared_workflow_id"] or "<missing>" for row in unique_rows
    )
    overlap_keys = list(spec.get("overlap_keys") or [])
    overlap_against = list(spec.get("overlap_against") or [])
    overlap_splits = (
        {name: recs for name, recs in splits.items() if name == train_split or name in overlap_against}
        if overlap_against
        else splits
    )
    overlaps = split_overlap(overlap_splits, overlap_keys, train_split) if overlap_keys else {}
    coverage = {name: _coverage(unique_records, path) for name, path in validation_paths.items()}
    literals = numeric_literal_stats(unique_records)

    # ---- threshold checks ------------------------------------------------
    def check_max(name: str, measured: float, limit: Any) -> None:
        if limit is None:
            return
        if measured > float(limit) + RATE_TOLERANCE:
            deficits.append(f"{name} measured {measured:.6f} vs required <= {float(limit):.6f}")

    def check_min(name: str, measured: float, limit: Any) -> None:
        if limit is None:
            return
        if measured < float(limit) - RATE_TOLERANCE:
            deficits.append(f"{name} measured {measured:.6f} vs required >= {float(limit):.6f}")

    for key, limit in (thresholds.get("min_share_call_count_ge") or {}).items():
        measured = sum(v for cc, v in overall_cc.items() if cc >= int(key)) / n_rows
        check_min(f"share_call_count_ge_{key}", measured, limit)
    check_max("exact_duplicate_rate", dup["exact_duplicate_rate"], thresholds.get("max_exact_duplicate_rate"))
    check_max("top1_skeleton_share", dup["top1_skeleton_share"], thresholds.get("max_top1_skeleton_share"))
    check_max("top10_intent_share", dup["top10_intent_share"], thresholds.get("max_top10_intent_share"))
    check_max("top1_intent_share", dup["top1_intent_share"], thresholds.get("max_top1_intent_share"))
    bool_range = thresholds.get("boolean_true_share_range")
    if bool_range and bools["n_boolean"]:
        lo, hi = float(bool_range[0]), float(bool_range[1])
        share = bools["overall_true_share"]
        if share < lo - RATE_TOLERANCE or share > hi + RATE_TOLERANCE:
            deficits.append(
                f"boolean_true_share measured {share:.6f} vs required within [{lo:.3f}, {hi:.3f}]"
            )
    check_min("n_distinct_primitives", len(usage.counts), thresholds.get("min_distinct_primitives"))
    check_min(
        "n_distinct_capability_families",
        len([k for k in caps if not k.startswith("<")]),
        thresholds.get("min_distinct_capability_families"),
    )
    check_max(
        "top1_primitive_sequence_share",
        seqs["primitive_sequence_concentration"]["top1_share"],
        thresholds.get("max_top1_primitive_sequence_share"),
    )
    for key, limit in (thresholds.get("max_split_overlap") or {}).items():
        for split, count in (overlaps.get(key) or {}).items():
            if count > int(limit):
                deficits.append(
                    f"split_overlap[{key}][{split}] measured {count} vs required <= {int(limit)}"
                )
    check_min("v4_coverage", coverage.get("v4", {}).get("coverage", 0.0), thresholds.get("min_v4_coverage"))
    check_min(
        "critic_coverage",
        coverage.get("critic", {}).get("coverage", 0.0),
        thresholds.get("min_critic_coverage"),
    )
    check_min(
        "node_necessity_coverage",
        coverage.get("node_necessity", {}).get("coverage", 0.0),
        thresholds.get("min_node_necessity_coverage"),
    )
    if usage.disagreements:
        deficits.append(
            f"primitive_source_disagreement measured {usage.disagreements} calls vs required exactly 0"
        )

    passed = not deficits
    result: Dict[str, Any] = {
        "schema_version": "ttdf.pilot43_independent_audit.v1",
        "run_label": spec.get("run_label", export_dir.name),
        "export_dir": str(export_dir),
        "out_dir": str(out_dir),
        "independence": {
            "imports": "stdlib only",
            "producer_code_imported": False,
            "exported_data_files_read": sorted(
                set(list(files.values()) + [primitive_registry_name])
            ),
        },
        "verdict": "PASS" if passed else "FAIL",
        "INDEPENDENT_AUDIT_PASSED": passed,
        "deficits": deficits,
        "disagreements": disagreements,
        "counts": {split: len(splits.get(split, [])) for split in files},
        "n_records_audited": len(rows),
        "n_unique_tasks": len(unique_rows),
        "dedupe_key": dedupe_key,
        "dedupe_note": (
            "per-task checks cover every audited row; aggregate distributions count each "
            f"{dedupe_key or 'record'} once"
        ),
        "call_count": {
            "by_split": json_safe(call_counts_by_split),
            "by_tier": json_safe(dict(call_counts_by_tier)),
            "overall": json_safe(overall_cc),
            "shares": call_count_shares,
            "share_at_least": share_ge,
        },
        "patterns": {
            "primary_distribution": json_safe(pattern_totals),
            "satisfied_distribution": json_safe(satisfied_totals),
            "undecidable_distribution": json_safe(undecidable_totals),
            "primary_by_call_count": json_safe(dict(pattern_by_call_count)),
        },
        "primitives": {
            "primitive_mapping_available": primitive_mapping_available,
            "primitive_mapping_source": mapping_source,
            "surface_map_collisions": derived["collisions"],
            "distinct_gold_tool_surfaces": len(surfaces),
            "gold_tool_surface_counts": json_safe(surfaces),
            "distinct_primitives_used": len(usage.counts),
            "primitive_counts": json_safe(usage.counts),
            "primitive_source": usage.source,
            "primitive_source_disagreements": usage.disagreements,
            "unmapped_surfaces": json_safe(usage.unmapped),
            "registry_size": len(primitive_to_capability),
            "registry_coverage": (
                len(usage.counts) / len(primitive_to_capability)
                if primitive_to_capability
                else 0.0
            ),
            "capability_counts": json_safe(caps),
            "distinct_capability_families": len([k for k in caps if not k.startswith("<")]),
        },
        "sequences": {
            "n_distinct_primitive_sequences": len(seqs["primitive_sequences"]),
            "primitive_sequence_concentration": seqs["primitive_sequence_concentration"],
            "top_primitive_sequences": [
                {"sequence": list(seq), "n": n}
                for seq, n in seqs["primitive_sequences"].most_common(10)
            ],
            "n_distinct_capability_sequences": len(seqs["capability_sequences"]),
            "capability_sequence_concentration": seqs["capability_sequence_concentration"],
        },
        "queries": {
            "duplicate_rates_overall": dup,
            "duplicate_rates_by_split": dup_by_split,
            "declared_query_mode_distribution": json_safe(query_modes),
            "declared_query_mode_note": "declared-only: the export carries no independent evidence of the realised query mode",
        },
        "boolean_balance": bools,
        "support": {
            "cell_tier_support": json_safe(cell_support),
            "workflow_support": json_safe(workflow_support),
            "n_distinct_workflows": len([k for k in workflow_support if k != "<missing>"]),
        },
        "split_overlap": overlaps,
        "validation_coverage": coverage,
        "numeric_literals": literals,
        "tolerances": {
            "counts": "exact",
            "rates": "0.1 percentage point",
            "per_task_pattern_label": "exact",
            "per_task_call_count": "exact",
        },
    }

    if emit.get("csv", True):
        _write_csv(out_dir / str(spec.get("csv_name", "independent_audit_per_task.csv")), rows)
    prefix = str(spec.get("report_prefix", "PILOT43_INDEPENDENT_AUDIT"))
    if emit.get("json", True):
        (out_dir / f"{prefix}.json").write_text(
            json.dumps(result, indent=2, sort_keys=False), encoding="utf-8"
        )
    if emit.get("md", True):
        (out_dir / f"{prefix}.md").write_text(_render_md(result), encoding="utf-8")

    result["_rows"] = rows
    result["_splits"] = splits
    result["_surface_to_primitive"] = surface_to_primitive
    result["_primitive_to_capability"] = primitive_to_capability
    return result


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write the per-task audit table."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


def _table(title: str, mapping: Dict[str, Any], total: Optional[int] = None) -> List[str]:
    """Render a small key/count markdown table."""
    lines = [f"### {title}", "", "| key | count | share |", "| --- | --- | --- |"]
    denom = total if total else max(sum(int(v) for v in mapping.values()), 1)
    for key, value in sorted(mapping.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
        lines.append(f"| {key} | {int(value)} | {int(value) / denom:.4f} |")
    lines.append("")
    return lines


def _render_md(result: Dict[str, Any]) -> str:
    """Render the independent audit summary as Markdown."""
    lines: List[str] = []
    lines.append(f"# Independent audit: {result['run_label']}")
    lines.append("")
    lines.append(f"- verdict: **{result['verdict']}**")
    lines.append(f"- INDEPENDENT_AUDIT_PASSED: {result['INDEPENDENT_AUDIT_PASSED']}")
    lines.append(f"- export dir: `{result['export_dir']}`")
    lines.append(
        "- independence: stdlib only, no producer module imported; exported data files read: "
        + ", ".join(f"`{name}`" for name in result["independence"]["exported_data_files_read"])
    )
    lines.append("")
    lines.append("## 1. Deficits")
    lines.append("")
    if result["deficits"]:
        for item in result["deficits"]:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## 2. Split counts")
    lines.append("")
    for split, count in result["counts"].items():
        lines.append(f"- `{split}`: {count}")
    lines.append(
        f"- rows audited: {result['n_records_audited']}, unique tasks: {result['n_unique_tasks']} "
        f"({result['dedupe_note']})"
    )
    lines.append("")
    lines.append("## 3. Recomputed call-count distribution")
    lines.append("")
    lines.extend(_table("Call count (all audited records)", result["call_count"]["overall"]))
    lines.append("Share with at least N calls:")
    lines.append("")
    for key, share in result["call_count"]["share_at_least"].items():
        lines.append(f"- >= {key} calls: {share:.4f}")
    lines.append("")
    lines.append("## 4. Recomputed structural patterns")
    lines.append("")
    lines.extend(_table("Primary pattern", result["patterns"]["primary_distribution"]))
    lines.extend(
        _table("Satisfied invariants (records)", result["patterns"]["satisfied_distribution"])
    )
    if result["patterns"]["undecidable_distribution"]:
        lines.extend(
            _table(
                "Undecidable invariants (records)",
                result["patterns"]["undecidable_distribution"],
            )
        )
    lines.append("## 5. Declared vs recomputed")
    lines.append("")
    for name, block in result["disagreements"].items():
        lines.append(
            f"- {name}: {block['n_disagree']}/{block['n_checked']} disagree (rate {block['rate']:.4f})"
        )
    lines.append("")
    lines.append("## 6. Primitive and capability usage")
    lines.append("")
    prims = result["primitives"]
    lines.append(f"- primitive_mapping_available: {prims['primitive_mapping_available']}")
    lines.append(f"- primitive_mapping_source: `{prims['primitive_mapping_source']}`")
    lines.append(f"- distinct_gold_tool_surfaces: {prims['distinct_gold_tool_surfaces']}")
    lines.append(f"- distinct_primitives_used: {prims['distinct_primitives_used']}")
    lines.append(f"- registry_size: {prims['registry_size']}")
    lines.append(f"- registry_coverage: {prims['registry_coverage']:.4f}")
    lines.append(f"- distinct_capability_families: {prims['distinct_capability_families']}")
    lines.append("")
    lines.extend(_table("Capability families", prims["capability_counts"]))
    lines.append("## 7. Sequence concentration")
    lines.append("")
    seqs = result["sequences"]
    lines.append(f"- distinct primitive sequences: {seqs['n_distinct_primitive_sequences']}")
    lines.append(
        f"- top-1 share: {seqs['primitive_sequence_concentration']['top1_share']:.4f}, "
        f"top-10 share: {seqs['primitive_sequence_concentration']['top10_share']:.4f}"
    )
    lines.append("")
    lines.append("## 8. Query repetition")
    lines.append("")
    dup = result["queries"]["duplicate_rates_overall"]
    for key in (
        "n",
        "n_distinct_exact",
        "exact_duplicate_rate",
        "n_distinct_skeleton",
        "top1_skeleton_share",
        "n_distinct_intent",
        "top1_intent_share",
        "top10_intent_share",
    ):
        lines.append(f"- {key}: {dup[key]}")
    lines.append("")
    lines.append("## 9. Boolean balance")
    lines.append("")
    bools = result["boolean_balance"]
    lines.append(f"- n_boolean: {bools['n_boolean']}, overall true share: {bools['overall_true_share']:.4f}")
    lines.append("")
    lines.append("## 10. Split overlap")
    lines.append("")
    if result["split_overlap"]:
        for key, per_split in result["split_overlap"].items():
            for split, count in per_split.items():
                lines.append(f"- `{key}` train vs `{split}`: {count} shared values")
    else:
        lines.append("- not requested")
    lines.append("")
    lines.append("## 11. Validation coverage")
    lines.append("")
    for name, block in result["validation_coverage"].items():
        lines.append(
            f"- {name} (`{block['path']}`): present {block['n_present']}/{block['n']} "
            f"(coverage {block['coverage']:.4f}), pass rate over present {block['pass_rate_over_present']:.4f}"
        )
    lines.append("")
    lines.append("## 12. Numeric literal realism")
    lines.append("")
    for key, value in result["numeric_literals"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Independent dataset export audit")
    parser.add_argument("--export-dir", required=True, help="directory holding exported JSONL files")
    parser.add_argument("--spec-json", required=True, help="path to the audit spec JSON")
    parser.add_argument("--out-dir", default=None, help="override spec.out_dir")
    args = parser.parse_args(list(argv) if argv is not None else None)

    # utf-8-sig also decodes plain UTF-8, so a BOM-prefixed spec file still loads.
    spec = json.loads(Path(args.spec_json).read_text(encoding="utf-8-sig"))
    if args.out_dir:
        spec["out_dir"] = args.out_dir
    result = audit_export(Path(args.export_dir), spec)
    print(json.dumps({
        "verdict": result["verdict"],
        "INDEPENDENT_AUDIT_PASSED": result["INDEPENDENT_AUDIT_PASSED"],
        "n_deficits": len(result["deficits"]),
        "deficits": result["deficits"],
    }, indent=2))
    return 0 if result["INDEPENDENT_AUDIT_PASSED"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
