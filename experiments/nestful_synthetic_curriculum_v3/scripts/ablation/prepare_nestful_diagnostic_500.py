#!/usr/bin/env python3
"""Prepare & freeze the 500-task NESTFUL diagnostic eval subset for the
reward ablation (reports/reward_ablation/ABLATION_PLAN.md §4).

Source: experiments/nestful_mtgrpo_minimal/data/splits/nestful_test.jsonl
(the full 1661-task NESTFUL test split).

Selection may use ONLY dataset metadata + the already-computed C0 baseline
(outputs/runs/pure_stage3_2ep_20260719_221918/eval/C0_test) for
stratification — never any reward-arm result. Primary axis: gold call
count bucketed as {2, 3, 4, 5, 6+}, ~100 tasks/bucket (all buckets have
>=100 available tasks in nestful_test, so the ~100/bucket target is met
exactly). Secondary axis inside each bucket: (motif_type, C0 official
win/loss) via the largest-remainder method, seed 20260724.

The SAME 500 IDs are frozen for C0 and every reward arm.

Usage (repo root):
  python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/prepare_nestful_diagnostic_500.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parents[1]
_SCRIPTS = _V3 / "scripts"
_REPO = _V3.parents[1]
for p in (str(_SCRIPTS), str(_V3)):
    if p not in sys.path:
        sys.path.insert(0, p)

from motif_lib import default_test_path, extract_motifs, load_jsonl, load_task_row  # noqa: E402
from stratify_utils import group_by, largest_remainder_allocation, stratified_select  # noqa: E402

sys.path.insert(0, str(_V3 / "scripts" / "analysis"))
from two_phase_root_cause_analysis import classify_failure, official_win  # noqa: E402

SEED = 20260724
N_TARGET = 500
TARGET_PER_BUCKET = 100
SOURCE = _V3.parents[0] / "nestful_mtgrpo_minimal" / "data" / "splits" / "nestful_test.jsonl"
C0_TRAJ = (
    _V3 / "outputs" / "runs" / "pure_stage3_2ep_20260719_221918" / "eval" / "C0_test"
    / "final_eval_trajectories.jsonl"
)
OUT_DIR = _V3 / "reports" / "reward_ablation" / "data"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def call_count_bucket(n: int) -> str:
    if n <= 1:
        return "2"  # nestful_test has no 1-call tasks; guard anyway
    if n in (2, 3, 4, 5):
        return str(n)
    return "6+"


def load_c0_rows(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                sid = r.get("sample_id") or (r.get("_traj") or {}).get("task_id")
                if sid:
                    out[str(sid)] = r
    return out


def main() -> int:
    if not SOURCE.is_file():
        raise SystemExit(f"[prepare_nestful_diagnostic_500] ABORT: source not found: {SOURCE}")
    if not C0_TRAJ.is_file():
        raise SystemExit(f"[prepare_nestful_diagnostic_500] ABORT: C0 baseline not found: {C0_TRAJ}")

    raw_rows = load_jsonl(SOURCE)
    tasks = {load_task_row(r)["task_id"]: load_task_row(r) for r in raw_rows}
    assert len(tasks) == 1661, f"expected 1661 nestful_test tasks, got {len(tasks)}"

    c0_rows = load_c0_rows(C0_TRAJ)
    missing_c0 = set(tasks) - set(c0_rows)
    assert not missing_c0, f"{len(missing_c0)} tasks missing C0 baseline, e.g. {sorted(missing_c0)[:5]}"

    meta: Dict[str, Dict[str, Any]] = {}
    for tid, task in tasks.items():
        motifs = extract_motifs(task)
        c0_row = c0_rows[tid]
        ow = official_win(c0_row)
        primary, secondary = classify_failure(c0_row) if ow != 1.0 else ("win", "official_win")
        meta[tid] = {
            "gold_call_count": motifs["num_calls"],
            "call_count_bucket": call_count_bucket(motifs["num_calls"]),
            "motif_type": motifs["motif_type"],
            "tool_family": motifs["tool_family"].split(",")[0] if motifs["tool_family"] else "other",
            "c0_official_win": None if ow is None else bool(ow),
            "c0_failure_primary": primary,
            "argument_complexity": motifs["argument_complexity"],
            "dependency_depth": motifs["dependency_depth"],
        }

    # ── primary axis: call-count bucket, target ~100/bucket ────────────
    bucket_sizes = Counter(m["call_count_bucket"] for m in meta.values())
    buckets = sorted(bucket_sizes.keys())
    bucket_target: Dict[str, int] = {}
    for b in buckets:
        bucket_target[b] = min(TARGET_PER_BUCKET, bucket_sizes[b])
    shortfall = N_TARGET - sum(bucket_target.values())
    if shortfall > 0:
        # Proportionally redistribute any shortfall (buckets with <100
        # available) across buckets that still have spare capacity.
        spare_sizes = {b: bucket_sizes[b] - bucket_target[b] for b in buckets if bucket_sizes[b] > bucket_target[b]}
        extra = largest_remainder_allocation(spare_sizes, shortfall)
        for b, n in extra.items():
            bucket_target[b] += n
    assert sum(bucket_target.values()) == N_TARGET, (sum(bucket_target.values()), bucket_target)

    selected_all: List[str] = []
    per_bucket_detail: Dict[str, Any] = {}
    for b in buckets:
        ids_in_bucket = [tid for tid, m in meta.items() if m["call_count_bucket"] == b]
        # secondary axis inside the bucket: (motif_type, c0_official_win)
        cell_of = {tid: (meta[tid]["motif_type"], meta[tid]["c0_official_win"]) for tid in ids_in_bucket}
        ids_by_cell = group_by([(tid, cell_of[tid]) for tid in ids_in_bucket])
        cell_sizes = {k: len(v) for k, v in ids_by_cell.items()}
        alloc = largest_remainder_allocation(cell_sizes, bucket_target[b])
        # deterministic per-bucket seed offset (Python's built-in hash() is
        # randomized per-process by PYTHONHASHSEED, so it must not be used
        # here — crc32 over the bucket label is stable across runs/machines)
        bucket_offset = zlib.crc32(b.encode("utf-8")) % 1000
        chosen = stratified_select(ids_by_cell, alloc, seed=SEED + bucket_offset)
        selected_all.extend(chosen)
        per_bucket_detail[b] = {
            "available": bucket_sizes[b],
            "target": bucket_target[b],
            "selected": len(chosen),
            "cell_allocation": {f"{k[0]}|{k[1]}": v for k, v in alloc.items()},
        }

    selected_all = sorted(set(selected_all))
    assert len(selected_all) == N_TARGET, f"expected {N_TARGET}, got {len(selected_all)}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids_path = OUT_DIR / "nestful_diagnostic_500_ids.json"
    with open(ids_path, "w", encoding="utf-8") as fh:
        json.dump({"task_ids": selected_all, "count": len(selected_all)}, fh, indent=2, ensure_ascii=False)

    source_sha256 = _sha256_file(SOURCE)
    ids_sha256 = _sha256_file(ids_path)

    subset_counts = {
        "call_count_bucket": Counter(meta[t]["call_count_bucket"] for t in selected_all),
        "motif_type": Counter(meta[t]["motif_type"] for t in selected_all),
        "tool_family": Counter(meta[t]["tool_family"] for t in selected_all),
        "c0_official_win": Counter(str(meta[t]["c0_official_win"]) for t in selected_all),
        "c0_failure_primary": Counter(meta[t]["c0_failure_primary"] for t in selected_all),
    }
    source_counts = {
        "call_count_bucket": Counter(m["call_count_bucket"] for m in meta.values()),
        "motif_type": Counter(m["motif_type"] for m in meta.values()),
        "tool_family": Counter(m["tool_family"] for m in meta.values()),
        "c0_official_win": Counter(str(m["c0_official_win"]) for m in meta.values()),
        "c0_failure_primary": Counter(m["c0_failure_primary"] for m in meta.values()),
    }

    manifest = {
        "generated_at": _now(),
        "seed": SEED,
        "n_target": N_TARGET,
        "n_selected": len(selected_all),
        "target_per_call_count_bucket": TARGET_PER_BUCKET,
        "source_dataset_path": str(SOURCE.relative_to(_REPO)).replace("\\", "/"),
        "source_dataset_rows": len(tasks),
        "source_sha256": source_sha256,
        "ids_file_sha256": ids_sha256,
        "c0_baseline_used_for_stratification": str(C0_TRAJ.relative_to(_REPO)).replace("\\", "/"),
        "c0_baseline_sha256": _sha256_file(C0_TRAJ),
        "primary_stratification_axis": "gold_call_count_bucket",
        "secondary_stratification_axis": ["motif_type", "c0_official_win"],
        "reported_axes": ["tool_family", "c0_failure_primary", "argument_complexity", "dependency_depth"],
        "per_bucket_allocation": per_bucket_detail,
        "stratification_counts": {
            "subset": {k: dict(v) for k, v in subset_counts.items()},
            "source_1661": {k: dict(v) for k, v in source_counts.items()},
        },
        "selection_uses_only_dataset_metadata_and_c0": True,
        "identical_ids_for_c0_and_all_arms": True,
        "benchmark_reuse_disclosure": (
            "This exact NESTFUL test split has already been used in prior diagnostics "
            "(pure_stage3_2ep_20260719_221918 C0/E1/E2 evals and the reward-variant "
            "audit). This 500-task subset is an INTERNAL ablation/development "
            "evaluation set for comparing reward arms under identical conditions — "
            "it is NOT a held-out/untouched final test. Round 3's final internal "
            "confirmation evaluates on the full untrimmed n=1661 NESTFUL test set."
        ),
    }
    manifest_path = OUT_DIR / "nestful_diagnostic_500_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    _write_report(OUT_DIR / "NESTFUL_DIAGNOSTIC_SUBSET.md", manifest)

    print(f"[prepare_nestful_diagnostic_500] wrote {ids_path} ({len(selected_all)} ids)")
    print(f"[prepare_nestful_diagnostic_500] wrote {manifest_path}")
    print(f"[prepare_nestful_diagnostic_500] ids sha256={ids_sha256}")
    return 0


def _pct_table(subset_c: Dict[str, int], source_c: Dict[str, int], source_n: int, subset_n: int) -> str:
    keys = sorted(set(subset_c) | set(source_c), key=str)
    lines = [f"| value | source ({source_n}) | source % | subset ({subset_n}) | subset % |",
             "|---|---:|---:|---:|---:|"]
    for k in keys:
        sc = source_c.get(k, 0)
        ssc = subset_c.get(k, 0)
        lines.append(f"| {k} | {sc} | {sc/source_n*100:.1f}% | {ssc} | {ssc/subset_n*100:.1f}% |")
    return "\n".join(lines)


def _write_report(path: Path, manifest: Dict[str, Any]) -> None:
    sub = manifest["stratification_counts"]["subset"]
    src = manifest["stratification_counts"]["source_1661"]
    n_src = manifest["source_dataset_rows"]
    n_sub = manifest["n_selected"]

    md = f"""# NESTFUL_DIAGNOSTIC_SUBSET — 500-task fixed diagnostic eval subset

Generated: {manifest['generated_at']}

## Disclosure

{manifest['benchmark_reuse_disclosure']}

## Source

- Path: `{manifest['source_dataset_path']}`
- Rows: {n_src}
- SHA-256: `{manifest['source_sha256']}`
- C0 baseline used ONLY for stratification metadata (never for arm selection): `{manifest['c0_baseline_used_for_stratification']}`
  (sha256 `{manifest['c0_baseline_sha256']}`)

## Subset

- Seed: {manifest['seed']}
- Selected: {n_sub} / target {manifest['n_target']}
- IDs file SHA-256: `{manifest['ids_file_sha256']}`
- Identical IDs frozen for C0 and every reward arm: {manifest['identical_ids_for_c0_and_all_arms']}

## Primary axis — gold call-count bucket (target {manifest['target_per_call_count_bucket']}/bucket)

| bucket | available | target | selected |
|---|---:|---:|---:|
"""
    for b, d in sorted(manifest["per_bucket_allocation"].items()):
        md += f"| {b} | {d['available']} | {d['target']} | {d['selected']} |\n"

    md += f"""
## motif_type distribution

{_pct_table(sub['motif_type'], src['motif_type'], n_src, n_sub)}

## c0_official_win distribution (secondary stratification axis)

{_pct_table(sub['c0_official_win'], src['c0_official_win'], n_src, n_sub)}

## c0_failure_primary taxonomy (reported, not a sampling axis)

{_pct_table(sub['c0_failure_primary'], src['c0_failure_primary'], n_src, n_sub)}

## tool_family distribution (reported, not a sampling axis)

{_pct_table(sub['tool_family'], src['tool_family'], n_src, n_sub)}

## Notes / limits

- Primary stratification is the gold call-count bucket (2, 3, 4, 5, 6+); all five buckets have
  >=100 available tasks in the full 1661-task test set, so the ~100/bucket target is met exactly
  (500 = 5 x 100).
- Secondary stratification inside each bucket is (`motif_type`, `c0_official_win`) via the
  largest-remainder method, so both motif mix and C0 success/failure mix are preserved within each
  call-count bucket, not just in aggregate.
- `c0_failure_primary` / `tool_family` / argument-reference complexity are reported for
  transparency but are not independent sampling axes (crossing all of them with call-count and
  motif would produce too many near-empty cells for a stable, deterministic allocation).
- Selection depends only on dataset metadata (gold calls -> motif/call-count) and the C0 baseline;
  no reward-arm rollout or evaluation result is used anywhere in this selection.
"""
    path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
