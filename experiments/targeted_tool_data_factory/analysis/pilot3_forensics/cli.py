#!/usr/bin/env python3
"""Pilot3 forensic analysis CLI (offline; no train/inference)."""
from __future__ import annotations

import argparse
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import SCHEMA_VERSION, __version__
from .coverage import build_train_indexes, coverage_by_outcome, task_coverage_features
from .discovery import discover
from .distribution_audit import (
    compare_subset_distributions,
    feature_associations,
    featurize_diag_row,
    featurize_train_row,
    ood_analysis,
)
from .failure_taxonomy import build_failure_tables
from .graph_features import (
    graph_features,
    inventory_reference_formats,
    summarize_topology_distribution,
    topology_coverage,
)
from .integrity import build_input_manifest, require_pairing_ok
from .io import (
    as_bool,
    git_info,
    read_json,
    read_jsonl,
    rel_to,
    sha256_file,
    utc_now_iso,
    write_csv,
    write_json,
    write_jsonl,
    write_md,
)
from .pairing import load_traj_index, outcome_label, reproduce_headline, write_headline_outputs
from .plotting import try_plots
from .quality_audit import audit_train_quality
from .recommendations import (
    build_generation_cells,
    rank_bottlenecks,
    registry_gap_priorities,
    selection_constraints,
    validation_gates,
)
from .report import build_final_report
from .reward_audit import missing_observability_md, reward_md, run_reward_audit
from .statistics import call_bucket, mean
from .surface_features import (
    distractor_hardness,
    namespace_overlap,
    reference_syntax_rows,
    tool_surface_record,
)
from .trajectory_features import pair_trajectory_features


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pilot3 Targeted Tool Data Factory forensics")
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/targeted_tool_data_factory/reports/pilot3_forensics"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=20000)
    for key in (
        "c0-trajectories",
        "d1-trajectories",
        "train-data",
        "full-train-data",
        "heldout-data",
        "reserve-data",
        "diagnostic-data",
        "train-log",
        "rollout-log",
        "target-profile",
        "generation-cells",
        "c0-hf-trajectories",
    ):
        p.add_argument(f"--{key}", type=Path, default=None)
    return p.parse_args(argv)


def _overrides_from_args(args: argparse.Namespace) -> Dict[str, Optional[Path]]:
    mapping = {
        "c0_trajectories": args.c0_trajectories,
        "d1_trajectories": args.d1_trajectories,
        "train_data": args.train_data,
        "full_train_data": args.full_train_data,
        "heldout_data": args.heldout_data,
        "reserve_data": args.reserve_data,
        "diagnostic_data": args.diagnostic_data,
        "train_log": args.train_log,
        "rollout_log": args.rollout_log,
        "target_profile": args.target_profile,
        "generation_cells": args.generation_cells,
        "c0_hf_trajectories": args.c0_hf_trajectories,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _gained_lost_forensics(
    ids_gained: List[str],
    ids_lost: List[str],
    pair_feats: Dict[str, Dict[str, Any]],
    fail_rows: Dict[str, Dict[str, Any]],
    cov_rows: Dict[str, Dict[str, Any]],
    diag_feats: Dict[str, Dict[str, Any]],
    outcomes: Dict[str, str],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, str, str]:
    def pack(sids: List[str], kind: str) -> List[Dict[str, Any]]:
        rows = []
        for sid in sids:
            pf = pair_feats.get(sid, {})
            fr = fail_rows.get(sid, {})
            cr = cov_rows.get(sid, {})
            df = diag_feats.get(sid, {})
            rows.append({
                "sample_id": sid,
                "kind": kind,
                "call_bucket": df.get("call_bucket"),
                "topology_hash": df.get("topology_hash"),
                "depth": df.get("depth"),
                "reference_density": df.get("reference_density"),
                "offered_tool_count": df.get("offered_tool_count"),
                "distractor_hardness": df.get("distractor_hardness"),
                "exact_tool_coverage_rate": cr.get("exact_tool_coverage_rate"),
                "combined_ood_score": cr.get("combined_ood_score"),
                "first_divergent_turn": pf.get("first_divergent_turn"),
                "divergence_category": pf.get("divergence_category"),
                "failure_transition": f"{fr.get('c0_primary')} -> {fr.get('d1_primary')}",
                "c0_tool_seq": pf.get("c0_tool_seq"),
                "d1_tool_seq": pf.get("d1_tool_seq"),
            })
        return rows

    gained_rows = pack(ids_gained, "gained")
    lost_rows = pack(ids_lost, "lost")
    # patterns
    pat_g = Counter(r["failure_transition"] for r in gained_rows)
    pat_l = Counter(r["failure_transition"] for r in lost_rows)
    div_g = Counter(r["divergence_category"] for r in gained_rows)
    div_l = Counter(r["divergence_category"] for r in lost_rows)
    patterns = []
    for key in sorted(set(pat_g) | set(pat_l), key=lambda k: -(pat_g[k] + pat_l[k])):
        ng, nl = pat_g[key], pat_l[key]
        patterns.append({
            "pattern": key,
            "n_gained": ng,
            "n_lost": nl,
            "net_gain": ng - nl,
            "support": ng + nl,
            "effect_direction": "gain" if ng > nl else ("loss" if nl > ng else "neutral"),
            "confidence": "low" if (ng + nl) < 5 else ("medium" if (ng + nl) < 12 else "medium"),
            "possible_mechanism": "failure-class transition associated with flips (not causal)",
        })
    for key in sorted(set(div_g) | set(div_l), key=lambda k: -(div_g[k] + div_l[k])):
        ng, nl = div_g[key], div_l[key]
        patterns.append({
            "pattern": f"divergence:{key}",
            "n_gained": ng,
            "n_lost": nl,
            "net_gain": ng - nl,
            "support": ng + nl,
            "effect_direction": "gain" if ng > nl else ("loss" if nl > ng else "neutral"),
            "confidence": "low" if (ng + nl) < 5 else "medium",
            "possible_mechanism": "trajectory divergence category association",
        })

    def examples_md(rows: List[Dict[str, Any]], title: str, limit: int = 15) -> str:
        # deterministic: sort by failure_transition freq then sample_id
        freq = Counter(r["failure_transition"] for r in rows)
        ordered = sorted(rows, key=lambda r: (-freq[r["failure_transition"]], r["sample_id"]))[:limit]
        lines = [f"# {title}", "", "Selected deterministically by most common failure transitions.", ""]
        for r in ordered:
            lines.append(f"## {r['sample_id']}")
            lines.append(f"- call_bucket: `{r['call_bucket']}`")
            lines.append(f"- divergence: `{r['divergence_category']}` turn={r['first_divergent_turn']}")
            lines.append(f"- failure: `{r['failure_transition']}`")
            lines.append(f"- tools C0: `{r['c0_tool_seq']}`")
            lines.append(f"- tools D1: `{r['d1_tool_seq']}`")
            lines.append(f"- coverage exact: `{r['exact_tool_coverage_rate']}` ood=`{r['combined_ood_score']}`")
            lines.append("")
        return "\n".join(lines)

    audit_md = [
        "# GAINED_LOST_AUDIT",
        "",
        f"- n_gained={len(ids_gained)} n_lost={len(ids_lost)}",
        "- Small-n warning: do not over-interpret significance.",
        "",
        "## Top patterns",
        "",
    ]
    for p in patterns[:20]:
        audit_md.append(
            f"- `{p['pattern']}`: gained={p['n_gained']} lost={p['n_lost']} net={p['net_gain']} support={p['support']}"
        )
    compare = gained_rows + lost_rows
    return compare, patterns, "\n".join(audit_md) + "\n", examples_md(gained_rows, "GAINED_REPRESENTATIVE_EXAMPLES"), examples_md(lost_rows, "LOST_REPRESENTATIVE_EXAMPLES")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    repo_root = args.repo_root.resolve()
    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    discovery = discover(repo_root, overrides=_overrides_from_args(args))
    manifest, integrity_checks, integrity_md = build_input_manifest(discovery)
    write_json(out_dir / "INPUT_MANIFEST.json", manifest)
    write_md(out_dir / "INPUT_INTEGRITY.md", integrity_md)
    require_pairing_ok(integrity_checks)

    c0_path = discovery.get("c0_trajectories")
    d1_path = discovery.get("d1_trajectories")
    assert c0_path and d1_path

    c0 = load_traj_index(c0_path)
    d1 = load_traj_index(d1_path)
    c0_hf = None
    if discovery.get("c0_hf_trajectories"):
        c0_hf = load_traj_index(discovery.get("c0_hf_trajectories"))  # type: ignore[arg-type]

    headline, paired_rows, headline_md = reproduce_headline(
        c0, d1, seed=args.seed, n_boot=args.n_boot, c0_hf=c0_hf
    )
    write_headline_outputs(out_dir, headline, paired_rows, headline_md)

    ids = [r["sample_id"] for r in paired_rows]
    outcomes = {r["sample_id"]: r["outcome"] for r in paired_rows}

    # Load raw rows for trajectory/failure
    c0_raw = {str(r["sample_id"]): r for r in read_jsonl(c0_path)}
    d1_raw = {str(r["sample_id"]): r for r in read_jsonl(d1_path)}

    diag_path = discovery.get("diagnostic_data")
    diag_rows = {str(r["sample_id"]): r for r in read_jsonl(diag_path)} if diag_path else {}
    diag_gold = {sid: (r.get("output") or []) for sid, r in diag_rows.items()}

    # Phase 2 trajectory divergence
    pair_feat_rows = []
    pair_feat_map = {}
    gained_jsonl = []
    lost_jsonl = []
    unchanged_fail_jsonl = []
    cat_counts: Counter = Counter()
    first_turns = []
    for sid in ids:
        feat = pair_trajectory_features(c0_raw[sid], d1_raw[sid], outcome=outcomes[sid])
        cat_counts[feat["divergence_category"]] += 1
        if feat["first_divergent_turn"] is not None and feat["first_divergent_turn"] >= 0:
            first_turns.append(feat["first_divergent_turn"])
        slim = {k: v for k, v in feat.items() if not k.startswith("_")}
        pair_feat_rows.append(slim)
        pair_feat_map[sid] = slim
        fp0, fp1 = feat["_fp_c0"], feat["_fp_d1"]
        blob = {
            "sample_id": sid,
            "outcome": outcomes[sid],
            "divergence_category": feat["divergence_category"],
            "first_divergent_turn": feat["first_divergent_turn"],
            "c0": {
                "tool_seq": fp0["tool_name_seq"],
                "pred_answer": fp0["predicted_answer"],
                "stop_reason": fp0["stop_reason"],
                "official_win": fp0["official_win"],
                "model_texts": fp0["model_texts"],
                "calls": fp0["calls"],
            },
            "d1": {
                "tool_seq": fp1["tool_name_seq"],
                "pred_answer": fp1["predicted_answer"],
                "stop_reason": fp1["stop_reason"],
                "official_win": fp1["official_win"],
                "model_texts": fp1["model_texts"],
                "calls": fp1["calls"],
            },
        }
        if outcomes[sid] == "loss_to_win":
            gained_jsonl.append(blob)
        elif outcomes[sid] == "win_to_loss":
            lost_jsonl.append(blob)
        elif outcomes[sid] == "loss_to_loss":
            unchanged_fail_jsonl.append(blob)

    div_summary = {
        "category_counts": dict(cat_counts),
        "mean_first_divergent_turn": mean([float(x) for x in first_turns]),
        "n_identical_text": cat_counts.get("IDENTICAL_TEXT", 0),
        "n_pairs": len(ids),
    }
    write_csv(out_dir / "TRAJECTORY_PAIR_FEATURES.csv", pair_feat_rows)
    write_json(out_dir / "TRAJECTORY_DIVERGENCE_SUMMARY.json", div_summary)
    write_md(
        out_dir / "TRAJECTORY_DIVERGENCE_SUMMARY.md",
        "# TRAJECTORY_DIVERGENCE_SUMMARY\n\n"
        + "\n".join(f"- `{k}`: {v}" for k, v in cat_counts.most_common())
        + f"\n\n- mean_first_divergent_turn: {div_summary['mean_first_divergent_turn']}\n",
    )
    write_jsonl(out_dir / "GAINED_TASKS.jsonl", gained_jsonl)
    write_jsonl(out_dir / "LOST_TASKS.jsonl", lost_jsonl)
    write_jsonl(out_dir / "UNCHANGED_FAILURES.jsonl", unchanged_fail_jsonl)

    # Phase 3 failure taxonomy
    per_task, transitions, matrix_rows, fail_summary, fail_md = build_failure_tables(
        ids, c0_raw, d1_raw, diag_gold, outcomes
    )
    write_csv(out_dir / "FAILURE_TAXONOMY_PER_TASK.csv", per_task)
    write_csv(out_dir / "FAILURE_TRANSITIONS.csv", transitions)
    write_csv(out_dir / "FAILURE_TRANSITION_MATRIX.csv", matrix_rows)
    write_md(out_dir / "FAILURE_ANALYSIS.md", fail_md)
    fail_map = {r["sample_id"]: r for r in per_task}

    # Phase 4 topology
    program_feats: List[Dict[str, Any]] = []
    topo_dist_rows: List[Dict[str, Any]] = []

    def add_split(path: Optional[Path], source: str, calls_key: str) -> List[Dict[str, Any]]:
        if not path:
            return []
        rows = read_jsonl(path)
        feats = []
        for r in rows:
            calls = r.get(calls_key) or []
            f = graph_features(calls, sample_id=str(r.get("sample_id")), source=source)
            # flatten hist dicts for CSV
            flat = dict(f)
            flat["indegree_hist"] = str(flat.get("indegree_hist"))
            flat["outdegree_hist"] = str(flat.get("outdegree_hist"))
            flat["ref_format_counts"] = str(flat.get("ref_format_counts"))
            flat["tool_names"] = "|".join(flat.get("tool_names") or [])
            feats.append(flat)
            program_feats.append(flat)
        summ = summarize_topology_distribution(feats)
        for item in summ["top_topologies"]:
            topo_dist_rows.append({"source": source, **item})
        return feats

    train300_path = discovery.get("train_data")
    full_train_path = discovery.get("full_train_data")
    heldout_path = discovery.get("heldout_data")
    reserve_path = discovery.get("reserve_data")
    pilot2_path = discovery.get("pilot2_train")

    feats_train300 = add_split(train300_path, "train300", "gold_calls")
    feats_train600 = add_split(full_train_path, "train600", "gold_calls")
    feats_heldout = add_split(heldout_path, "heldout", "gold_calls")
    feats_reserve = add_split(reserve_path, "reserve", "gold_calls")
    feats_pilot2 = add_split(pilot2_path, "pilot2_train", "gold_calls")
    feats_diag = []
    if diag_path:
        for r in diag_rows.values():
            f = graph_features(r.get("output") or [], sample_id=str(r.get("sample_id")), source="diagnostic")
            flat = dict(f)
            flat["indegree_hist"] = str(flat.get("indegree_hist"))
            flat["outdegree_hist"] = str(flat.get("outdegree_hist"))
            flat["ref_format_counts"] = str(flat.get("ref_format_counts"))
            flat["tool_names"] = "|".join(flat.get("tool_names") or [])
            feats_diag.append(flat)
            program_feats.append(flat)
        summ = summarize_topology_distribution(feats_diag)
        for item in summ["top_topologies"]:
            topo_dist_rows.append({"source": "diagnostic", **item})

    write_csv(out_dir / "PROGRAM_GRAPH_FEATURES.csv", program_feats)
    write_csv(out_dir / "TOPOLOGY_DISTRIBUTION.csv", topo_dist_rows)

    # coverage matrix topology × call_bucket etc.
    cov_matrix = []
    for source, feats in (
        ("train300", feats_train300),
        ("diagnostic", feats_diag),
    ):
        joint = Counter((f["topology_hash"], f["call_bucket"], f["motif"]) for f in feats)
        for (th, cb, motif), cnt in joint.most_common(200):
            cov_matrix.append({
                "source": source,
                "topology_hash": th,
                "call_bucket": cb,
                "motif": motif,
                "count": cnt,
            })
    write_csv(out_dir / "TOPOLOGY_COVERAGE_MATRIX.csv", cov_matrix)

    ref_inv = inventory_reference_formats(
        [(r.get("gold_calls") or []) for r in (read_jsonl(full_train_path) if full_train_path else [])]
        + [(r.get("output") or []) for r in diag_rows.values()]
    )
    topo_audit = {
        "reference_format_inventory": ref_inv,
        "train300": summarize_topology_distribution(feats_train300) if feats_train300 else {},
        "train600": summarize_topology_distribution(feats_train600) if feats_train600 else {},
        "heldout": summarize_topology_distribution(feats_heldout) if feats_heldout else {},
        "reserve": summarize_topology_distribution(feats_reserve) if feats_reserve else {},
        "pilot2_train": summarize_topology_distribution(feats_pilot2) if feats_pilot2 else {},
        "diagnostic": summarize_topology_distribution(feats_diag) if feats_diag else {},
        "coverage_train300_vs_diag": topology_coverage(feats_train300, feats_diag) if feats_train300 and feats_diag else {},
        "coverage_train600_vs_diag": topology_coverage(feats_train600, feats_diag) if feats_train600 and feats_diag else {},
        "methodology_limitations": [
            "topology_hash uses call-order node indices; order-isomorphic renumberings differ",
            "independent of tool names/labels/constants by construction",
        ],
    }
    write_json(out_dir / "TOPOLOGY_AUDIT.json", topo_audit)
    write_md(
        out_dir / "TOPOLOGY_AUDIT.md",
        "# TOPOLOGY_AUDIT\n\n"
        + f"- train300 unique: {(topo_audit.get('train300') or {}).get('n_unique_topology_hashes')}\n"
        + f"- train300 top1 share: {(topo_audit.get('train300') or {}).get('top1_share')}\n"
        + f"- diag unseen vs train300: {(topo_audit.get('coverage_train300_vs_diag') or {}).get('diagnostic_unseen_topology_rate')}\n"
        + f"- reference formats: `{ref_inv}`\n",
    )

    # Phase 5 surface
    train_for_surface = read_jsonl(full_train_path) if full_train_path else (read_jsonl(train300_path) if train300_path else [])
    train_tool_names = set()
    diag_tool_names = set()
    surface_rows = []
    train_hard = []
    diag_hard = []
    for r in train_for_surface:
        for t in r.get("tools") or []:
            if isinstance(t, dict):
                rec = tool_surface_record(t)
                train_tool_names.add(rec["name"])
                surface_rows.append({
                    "source": "train",
                    "name": rec["name"],
                    "name_norm": rec["name_norm"],
                    "n_params": rec["n_params"],
                    "n_required_params": rec["n_required_params"],
                    "param_names": "|".join(rec["param_names"]),
                    "output_keys": "|".join(rec["output_keys"]),
                    "description_len": rec["description_len"],
                    "track": (r.get("provenance") or {}).get("track") if isinstance(r.get("provenance"), dict) else "",
                })
        dh = distractor_hardness([{"name": c.get("name")} for c in (r.get("gold_calls") or [])], r.get("tools") or [])
        train_hard.append(dh["mean_distractor_hardness_proxy"])
    for r in diag_rows.values():
        for t in r.get("tools") or []:
            if isinstance(t, dict):
                rec = tool_surface_record(t)
                diag_tool_names.add(rec["name"])
                surface_rows.append({
                    "source": "diagnostic",
                    "name": rec["name"],
                    "name_norm": rec["name_norm"],
                    "n_params": rec["n_params"],
                    "n_required_params": rec["n_required_params"],
                    "param_names": "|".join(rec["param_names"]),
                    "output_keys": "|".join(rec["output_keys"]),
                    "description_len": rec["description_len"],
                    "track": "diagnostic",
                })
        dh = distractor_hardness([{"name": c.get("name")} for c in (r.get("output") or [])], r.get("tools") or [])
        diag_hard.append(dh["mean_distractor_hardness_proxy"])
        # per-task distractor rows (aggregate later)
    write_csv(out_dir / "TOOL_SURFACE_FEATURES.csv", surface_rows)
    ns = namespace_overlap(train_tool_names, diag_tool_names)
    write_csv(out_dir / "TOOL_NAMESPACE_OVERLAP.csv", [
        {"metric": k, "value": v} for k, v in ns.items() if not isinstance(v, list)
    ] + [{"metric": "exact_names_sample", "value": "|".join(ns.get("exact_names") or [])}])

    ref_rows = reference_syntax_rows(
        {str(r.get("sample_id")): (r.get("gold_calls") or []) for r in train_for_surface},
        "train",
    ) + reference_syntax_rows(diag_gold, "diagnostic")
    # stringify dicts
    ref_csv = []
    for r in ref_rows:
        ref_csv.append({
            **{k: v for k, v in r.items() if k not in ("pattern_hist", "output_key_hist")},
            "pattern_hist": str(r.get("pattern_hist")),
            "output_key_hist": str(r.get("output_key_hist")),
        })
    write_csv(out_dir / "REFERENCE_SYNTAX_AUDIT.csv", ref_csv)

    # distractor hardness table
    dh_rows = [
        {"source": "train", "mean_hardness_proxy": mean(train_hard), "n": len(train_hard)},
        {"source": "diagnostic", "mean_hardness_proxy": mean(diag_hard), "n": len(diag_hard)},
    ]
    write_csv(out_dir / "DISTRACTOR_HARDNESS.csv", dh_rows)
    surface_audit = {
        "namespace": ns,
        "distractor": {
            "train_mean": mean(train_hard),
            "diag_mean": mean(diag_hard),
        },
        "note": "lexical/schema proxies only",
    }
    write_json(out_dir / "SURFACE_SCHEMA_AUDIT.json", surface_audit)
    write_md(
        out_dir / "SURFACE_SCHEMA_AUDIT.md",
        "# SURFACE_SCHEMA_AUDIT\n\n"
        + f"- exact_overlap_rate_vs_diag: {ns.get('exact_overlap_rate_vs_diag')}\n"
        + f"- normalized_overlap_rate_vs_diag: {ns.get('normalized_overlap_rate_vs_diag')}\n"
        + f"- train distractor hardness mean: {mean(train_hard)}\n"
        + f"- diagnostic distractor hardness mean: {mean(diag_hard)}\n"
        + "\nAll overlaps are lexical/schema proxies, not semantic equivalence.\n",
    )

    # Phase 6 coverage
    train_idx = build_train_indexes(train_for_surface)
    train_topo = {f["topology_hash"] for f in feats_train300} if feats_train300 else {f["topology_hash"] for f in feats_train600}
    task_cov = []
    for sid in ids:
        if sid not in diag_rows:
            continue
        task_cov.append(task_coverage_features(
            diag_rows[sid],
            train_idx,
            topology_in_train=(graph_features(diag_rows[sid].get("output") or []).get("topology_hash") in train_topo),
            outcome=outcomes[sid],
            c0_win=c0[sid]["win"],
            d1_win=d1[sid]["win"],
        ))
    cov_by = coverage_by_outcome(task_cov)
    write_csv(out_dir / "TASK_COVERAGE_FEATURES.csv", task_cov)
    write_csv(out_dir / "REGISTRY_COVERAGE_BY_OUTCOME.csv", cov_by)
    cov_audit = {
        "by_outcome": cov_by,
        "n_tasks": len(task_cov),
        "mean_exact_coverage": mean([float(r["exact_tool_coverage_rate"]) for r in task_cov]),
        "mean_ood": mean([float(r["combined_ood_score"]) for r in task_cov]),
        "note": "proxy mappings are hypotheses",
    }
    write_json(out_dir / "REGISTRY_COVERAGE_AUDIT.json", cov_audit)
    write_md(
        out_dir / "REGISTRY_COVERAGE_AUDIT.md",
        "# REGISTRY_COVERAGE_AUDIT\n\n"
        + f"- mean exact coverage: {cov_audit['mean_exact_coverage']}\n"
        + f"- mean OOD: {cov_audit['mean_ood']}\n"
        + "\nProxy mappings ≠ semantic equivalence.\n",
    )
    cov_map = {r["sample_id"]: r for r in task_cov}

    # Phase 7 distribution / OOD
    train_feats = [featurize_train_row(r, "train300" if train300_path else "train") for r in (
        read_jsonl(train300_path) if train300_path else train_for_surface[:300]
    )]
    diag_feats_list = [featurize_diag_row(diag_rows[sid]) for sid in ids if sid in diag_rows]
    diag_feat_map = {f["sample_id"]: f for f in diag_feats_list}
    outcome_maps = {
        sid: {"c0_win": int(c0[sid]["win"]), "d1_win": int(d1[sid]["win"]), "outcome": outcomes[sid]}
        for sid in ids
    }
    joint_rows, ood_rows, dist_summary = ood_analysis(train_feats, diag_feats_list, outcome_maps)
    assoc = feature_associations(diag_feats_list, outcome_maps)
    write_csv(out_dir / "TASK_DISTRIBUTION_FEATURES.csv", train_feats + diag_feats_list)
    write_csv(out_dir / "JOINT_CELL_COVERAGE.csv", joint_rows)
    write_csv(out_dir / "OOD_ANALYSIS.csv", ood_rows)
    write_csv(out_dir / "FEATURE_ASSOCIATIONS.csv", assoc)
    dist_audit = {"summary": dist_summary}
    write_json(out_dir / "DISTRIBUTION_AUDIT.json", dist_audit)
    write_md(
        out_dir / "DISTRIBUTION_AUDIT.md",
        "# DISTRIBUTION_AUDIT\n\n"
        + f"- unseen_combination_rate: {dist_summary.get('unseen_combination_rate')}\n"
        + f"- rare_combination_rate: {dist_summary.get('rare_combination_rate')}\n"
        + "\nAssociations are not causal.\n",
    )

    # Phase 8 subset 300 vs rest / identity vs local full train
    subset_rows: List[Dict[str, Any]] = []
    subset_summary: Dict[str, Any] = {}
    if train300_path:
        sub_rows = read_jsonl(train300_path)
        sub_ids = {str(r.get("sample_id")) for r in sub_rows}
        sub_list = [str(r.get("sample_id")) for r in sub_rows]
        if full_train_path:
            full_rows = read_jsonl(full_train_path)
            full_ids = [str(r.get("sample_id")) for r in full_rows]
            full_id_set = set(full_ids)
            first300 = full_rows[:300]
            rest300 = full_rows[300:600]
            first_ids = [str(r.get("sample_id")) for r in first300]
            positional_match = first_ids == sub_list
            overlap = sub_ids & full_id_set
            # Compare actual D1 subset vs complement inside local full train when possible;
            # if identity mismatch, still compare positional first300 vs rest300 as a
            # secondary local-export audit, and flag the mismatch.
            in_full = [r for r in full_rows if str(r.get("sample_id")) in sub_ids]
            out_full = [r for r in full_rows if str(r.get("sample_id")) not in sub_ids]
            if len(in_full) >= 50 and out_full:
                f_feats = [featurize_train_row(r, "d1_subset_in_full") for r in in_full]
                r_feats = [featurize_train_row(r, "full_minus_subset") for r in out_full]
            else:
                f_feats = [featurize_train_row(r, "first300_positional") for r in first300]
                r_feats = [featurize_train_row(r, "rest300_positional") for r in rest300]
            # Always also featurize the actual D1 subset file for cell inventory
            actual_feats = [featurize_train_row(r, "d1_train_subset_300") for r in sub_rows]
            subset_rows, subset_summary = compare_subset_distributions(
                f_feats,
                r_feats,
                cat_keys=["generation_cell", "track", "call_bucket", "motif", "target_skill", "target_failure_mode",
                          "semantic_program_family", "graph_template_id", "answer_type", "paraphrase_status"],
                num_keys=["call_count", "depth", "reference_density", "offered_tool_count", "distractor_hardness"],
            )
            # Sequence audit on the actual subset file (not positional export prefix)
            cell_seq = [str(f.get("generation_cell") or "") for f in actual_feats]
            switches = sum(1 for i in range(1, len(cell_seq)) if cell_seq[i] != cell_seq[i - 1])
            subset_summary["positional_first300_matches_train_subset"] = positional_match
            subset_summary["n_overlap_subset_ids_with_local_full_train"] = len(overlap)
            subset_summary["n_subset"] = len(sub_ids)
            subset_summary["n_local_full_train"] = len(full_id_set)
            subset_summary["subset_identity_status"] = (
                "VERIFIED" if positional_match else (
                    "INCONSISTENT" if len(overlap) < int(0.9 * len(sub_ids)) else "PARTIALLY_VERIFIED"
                )
            )
            subset_summary["actual_subset_cell_sequence_switches"] = switches
            subset_summary["actual_subset_n_generation_cells"] = len({c for c in cell_seq if c})
            if subset_summary["subset_identity_status"] == "INCONSISTENT":
                subset_summary["identity_warning"] = (
                    "D1 train_subset_300.jsonl is NOT the first 300 rows of the local "
                    "train_grpo_pilot3.jsonl freeze; overlap is far below 300. "
                    "Local export may have been regenerated after the RunPod subset was frozen, "
                    "or the subset was sliced from a different artifact. "
                    "Topology/surface audits that use local train600 therefore only partially "
                    "represent the true D1 training distribution."
                )
        else:
            actual_feats = [featurize_train_row(r, "d1_train_subset_300") for r in sub_rows]
            subset_summary = {
                "subset_identity_status": "NOT_VERIFIABLE",
                "n_subset": len(sub_ids),
                "note": "full train missing; cannot compare representativeness vs rest",
            }
    write_csv(out_dir / "TRAIN300_VS_REST300.csv", subset_rows)
    write_json(out_dir / "TRAIN_SUBSET_SELECTION_AUDIT.json", subset_summary)
    write_md(
        out_dir / "TRAIN_SUBSET_SELECTION_AUDIT.md",
        "# TRAIN_SUBSET_SELECTION_AUDIT\n\n"
        + f"- subset_identity_status: `{subset_summary.get('subset_identity_status')}`\n"
        + f"- positional_match: {subset_summary.get('positional_first300_matches_train_subset')}\n"
        + f"- overlap with local full train: {subset_summary.get('n_overlap_subset_ids_with_local_full_train')}"
        f" / {subset_summary.get('n_subset')}\n"
        + f"- shuffle_interpretation: {subset_summary.get('shuffle_interpretation')}\n"
        + f"- missing cells in compared-A: {subset_summary.get('n_missing_cells_in_first300')}\n"
        + f"- identity_warning: {subset_summary.get('identity_warning')}\n"
        + f"- categorical: `{subset_summary.get('categorical')}`\n",
    )

    # Phase 9 reward
    rollout_path = discovery.get("rollout_log")
    rollout_is_pilot3 = False
    if rollout_path:
        rp = str(rollout_path).replace("\\", "/").lower()
        rollout_is_pilot3 = "pilot3" in rp and "pilot2" not in rp
    reward = run_reward_audit(
        train_log=discovery.get("train_log"),
        rollout_log=rollout_path,
        rollout_is_pilot3=rollout_is_pilot3,
    )
    write_csv(out_dir / "REWARD_GROUPS.csv", reward.get("groups") or [])
    write_csv(out_dir / "REWARD_GROUPS_BY_CELL.csv", reward.get("groups_by_cell") or [])
    write_json(out_dir / "REWARD_AUDIT.json", {
        "aggregates": (reward.get("train_log_aggregates") or {}).get("aggregates"),
        "per_rollout_available": reward.get("per_rollout_available"),
        "rollout_summary": reward.get("rollout_summary"),
        "missing_observability": reward.get("missing_observability"),
        "note": reward.get("note"),
    })
    write_md(out_dir / "REWARD_AUDIT.md", reward_md(reward))
    write_md(out_dir / "MISSING_OBSERVABILITY.md", missing_observability_md(reward))

    # Phase 10 gained/lost
    compare_rows, patterns, gl_audit, gained_ex, lost_ex = _gained_lost_forensics(
        headline.get("gained_ids") or [],
        headline.get("lost_ids") or [],
        pair_feat_map,
        fail_map,
        cov_map,
        diag_feat_map,
        outcomes,
    )
    write_csv(out_dir / "GAINED_LOST_FEATURE_COMPARISON.csv", compare_rows)
    write_csv(out_dir / "GAINED_LOST_PATTERNS.csv", patterns)
    write_md(out_dir / "GAINED_LOST_AUDIT.md", gl_audit)
    write_md(out_dir / "GAINED_REPRESENTATIVE_EXAMPLES.md", gained_ex)
    write_md(out_dir / "LOST_REPRESENTATIVE_EXAMPLES.md", lost_ex)

    # Phase 11 quality
    q_flags, q_templates, q_summary, q_md = audit_train_quality(train_for_surface)
    write_csv(out_dir / "TRAIN_DATA_QUALITY_FLAGS.csv", q_flags)
    write_csv(out_dir / "TEMPLATE_CONCENTRATION.csv", q_templates)
    write_json(out_dir / "ANTI_SHORTCUT_AUDIT.json", q_summary)
    write_md(out_dir / "ANTI_SHORTCUT_AUDIT.md", q_md)

    # Phase 12-13 recommendations
    ctx = {
        "headline": headline,
        "integrity_checks": integrity_checks,
        "reward": reward,
        "topology": topo_audit,
        "subset": subset_summary,
        "surface": surface_audit,
        "coverage": {"task_rows": task_cov, "by_outcome": cov_by},
        "distribution": {"summary": dist_summary},
        "quality": q_summary,
        "reference_syntax": ref_inv,
        "divergence_summary": div_summary,
        "gained_lost_patterns": patterns,
        "failure_matrix": matrix_rows,
        "coverage_by_outcome": cov_by,
        "topology_distribution_rows": topo_dist_rows,
        "topology_train300_full_counts": [
            c for _, c in Counter(f["topology_hash"] for f in feats_train300).most_common()
        ],
        "subset_rows": subset_rows,
    }
    bottlenecks = rank_bottlenecks(ctx)
    write_json(out_dir / "BOTTLENECK_RANKING.json", {"bottlenecks": bottlenecks})
    write_md(
        out_dir / "BOTTLENECK_RANKING.md",
        "# BOTTLENECK_RANKING\n\n"
        + "\n".join(
            f"## {b['rank']}. {b['category']}\n\n"
            f"- status: `{b['status']}`\n"
            f"- evidence: `{b['evidence_strength']}`\n"
            f"- for: {b['argument_for']}\n"
            f"- against: {b['argument_against']}\n"
            f"- next: {b['next_step']}\n"
            for b in bottlenecks
        ),
    )

    gen_cells = build_generation_cells(ctx)
    write_json(out_dir / "RECOMMENDED_GENERATION_CELLS.json", {"cells": gen_cells})
    write_csv(out_dir / "RECOMMENDED_GENERATION_CELLS.csv", [
        {k: (str(v) if isinstance(v, list) else v) for k, v in cell.items()} for cell in gen_cells
    ])
    write_json(out_dir / "RECOMMENDED_SELECTION_CONSTRAINTS.json", selection_constraints(ctx))
    write_json(out_dir / "RECOMMENDED_VALIDATION_GATES.json", validation_gates(ctx))
    gaps = registry_gap_priorities(cov_by, task_cov)
    write_csv(out_dir / "REGISTRY_GAP_PRIORITIES.csv", gaps)

    write_md(
        out_dir / "DATA_FACTORY_RECOMMENDATIONS.md",
        "# DATA_FACTORY_RECOMMENDATIONS\n\n"
        "## Modes\n\n"
        "- `PROFILE_SAFE`: TargetProfile / factory metadata only.\n"
        "- `DIAGNOSTIC_INFORMED_EXPLORATORY`: uses diagnostic-500 deeply; must not be confirmed on the same 500.\n\n"
        + "\n".join(
            f"### {c['recommendation_id']} ({c['mode']} / {c['priority']})\n\n"
            f"- quota: {c['recommended_relative_quota']}\n"
            f"- mechanism: {c['expected_mechanism']}\n"
            f"- evidence: {c['evidence']}\n"
            f"- confidence: {c['confidence']}\n"
            + (f"- disclaimer: {c.get('diagnostic_loop_disclaimer')}\n" if c.get("diagnostic_loop_disclaimer") else "")
            for c in gen_cells
        ),
    )

    ctx["bottlenecks"] = bottlenecks
    ctx["generation_cells"] = gen_cells
    final_md, final_json = build_final_report(ctx)
    write_md(out_dir / "PILOT3_FORENSIC_ANALYSIS.md", final_md)
    write_json(out_dir / "PILOT3_FORENSIC_ANALYSIS.json", final_json)

    plots = try_plots(out_dir, ctx)

    # Run manifest
    input_hashes = {
        k: (manifest["artifacts"].get(k) or {}).get("sha256")
        for k in ("c0_trajectories", "d1_trajectories", "train_data", "full_train_data", "diagnostic_data", "train_log")
    }
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "package_version": __version__,
        "timestamp_utc": utc_now_iso(),
        "git": git_info(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "cli_args": {
            "repo_root": str(repo_root),
            "output_dir": str(out_dir),
            "seed": args.seed,
            "n_boot": args.n_boot,
            **{k: str(v) for k, v in _overrides_from_args(args).items()},
        },
        "seed": args.seed,
        "input_sha256": input_hashes,
        "discovered_paths": discovery.as_dict(),
        "headline_check": {
            "wins_c0": headline.get("wins_c0"),
            "wins_d1": headline.get("wins_d1"),
            "loss_to_win": headline.get("loss_to_win"),
            "win_to_loss": headline.get("win_to_loss"),
        },
        "plots_written": plots,
    }
    write_json(out_dir / "ANALYSIS_RUN_MANIFEST.json", run_manifest)

    print(f"[pilot3_forensics] wrote outputs to {out_dir}")
    print(
        f"[pilot3_forensics] headline: C0={headline['wins_c0']}/500 D1={headline['wins_d1']}/500 "
        f"gained={headline['loss_to_win']} lost={headline['win_to_loss']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
