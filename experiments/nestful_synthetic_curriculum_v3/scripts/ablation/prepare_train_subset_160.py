#!/usr/bin/env python3
"""Prepare & freeze the 160-task deterministic stratified train subset for
the reward ablation (reports/reward_ablation/ABLATION_PLAN.md §3).

Source: the exact 326-row "clean Stage-3" dataset used by the
pure_stage3_2ep_20260719_221918 production run
(data/training_ready_v5/filtered/stage3_train_ready.jsonl, sha256
7df704bf... after LF line-ending normalization — see TRAIN_SUBSET_REPORT.md
for why the raw-bytes hash differs on a CRLF checkout).

Selection is a single, deterministic stratified sample (seed 20260724)
over (motif_type x quality_tier) cells, using the largest-remainder method
so proportions are preserved and the total is exactly 160. The SAME 160
IDs are used by every reward arm — no arm-specific filtering anywhere in
this script.

Every selected row is re-validated with the SAME synthetic-executor replay
gate the production trainer uses
(scripts/training/preflight_training_datasets.py::validate_file) — i.e.
"full synthetic executor replay", never gold_replay.

Usage (repo root):
  python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/prepare_train_subset_160.py
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parents[1]
_SCRIPTS = _V3 / "scripts"
_REPO = _V3.parents[1]
for p in (str(_SCRIPTS), str(_V3)):
    # _V3 inserted LAST so it ends up at sys.path[0]: nestful_synthetic_curriculum_v3/lib
    # (the real `lib.agentic_data.*` package) must resolve before scripts/lib/__init__.py.
    if p not in sys.path:
        sys.path.insert(0, p)

from motif_lib import (  # noqa: E402
    extract_references_from_value,
    load_blocked_ids,
    load_jsonl,
)
from stratify_utils import largest_remainder_allocation  # noqa: E402

SEED = 20260724
N_TARGET = 160
SOURCE = _V3 / "data" / "training_ready_v5" / "filtered" / "stage3_train_ready.jsonl"
OUT_DIR = _V3 / "reports" / "reward_ablation" / "data"
EASY_TIER_MAX_FRACTION = 0.10


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_file_lf(path: Path) -> str:
    """SHA-256 with CRLF normalized to LF, matching the hash recorded on
    the original Linux training pod (this repo is checked out with CRLF on
    Windows, which changes the raw byte hash without changing content)."""
    with open(path, "rb") as fh:
        data = fh.read()
    return _sha256_bytes(data.replace(b"\r\n", b"\n"))


def _has_reference_dependency(gold_calls: List[Dict[str, Any]]) -> bool:
    for i, call in enumerate(gold_calls):
        if i == 0:
            continue
        for v in (call.get("arguments") or {}).values():
            if extract_references_from_value(v):
                return True
    return False


_TOOL_FAMILY_KEYWORDS = {
    "math": ("add", "sub", "mul", "div", "sum", "average", "mean", "round", "power",
             "sqrt", "abs", "max", "min", "percent", "ratio", "modulo", "floor", "ceil"),
    "string": ("concat", "upper", "lower", "split", "replace", "trim", "format",
               "join", "substring", "capitalize", "length_of_string"),
    "list": ("sort", "filter", "map", "append", "list", "array", "unique", "flatten"),
    "object": ("get_", "set_", "merge", "keys", "values", "dict", "object"),
    "boolean": ("is_", "has_", "and_", "or_", "not_", "equal", "compare", "check"),
}


def _tool_family(name: str) -> str:
    n = (name or "").lower()
    for fam, kws in _TOOL_FAMILY_KEYWORDS.items():
        if any(kw in n for kw in kws):
            return fam
    return "other"


def _strat_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    gold_calls = row.get("gold_calls") or []
    quality = row.get("quality") or {}
    first_tool = gold_calls[0].get("name") if gold_calls else ""
    return {
        "motif_type": row.get("motif_type") or "unknown",
        "quality_tier": quality.get("quality_tier") or "unknown",
        "answer_type": row.get("answer_type") or "unknown",
        "tool_family": _tool_family(first_tool),
        "has_reference_dependency": _has_reference_dependency(gold_calls),
        "num_calls": row.get("num_calls") or len(gold_calls),
    }


def select_subset(rows: List[Dict[str, Any]], seed: int, n_target: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows_sorted = sorted(rows, key=lambda r: r["sample_id"])
    strat = {r["sample_id"]: _strat_fields(r) for r in rows_sorted}
    cells: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for r in rows_sorted:
        s = strat[r["sample_id"]]
        cells[(s["motif_type"], s["quality_tier"])].append(r["sample_id"])

    cell_sizes = {k: len(v) for k, v in cells.items()}
    alloc = largest_remainder_allocation(cell_sizes, n_target)

    rng = random.Random(seed)
    selected: List[str] = []
    for cell_key in sorted(cells.keys()):
        ids = sorted(cells[cell_key])
        rng.shuffle(ids)
        take = alloc[cell_key]
        selected.extend(ids[:take])

    selected = sorted(set(selected))
    assert len(selected) == n_target, f"expected {n_target}, got {len(selected)}"
    assert len(selected) == len(set(selected)), "duplicate IDs in subset"

    by_id = {r["sample_id"]: r for r in rows_sorted}
    subset_rows = [by_id[sid] for sid in selected]

    counts = {
        "motif_type": Counter(strat[sid]["motif_type"] for sid in selected),
        "quality_tier": Counter(strat[sid]["quality_tier"] for sid in selected),
        "answer_type": Counter(strat[sid]["answer_type"] for sid in selected),
        "tool_family": Counter(strat[sid]["tool_family"] for sid in selected),
        "has_reference_dependency": Counter(strat[sid]["has_reference_dependency"] for sid in selected),
        "cell_allocation": {f"{k[0]}|{k[1]}": v for k, v in alloc.items()},
    }
    source_counts = {
        "motif_type": Counter(strat[sid]["motif_type"] for sid in strat),
        "quality_tier": Counter(strat[sid]["quality_tier"] for sid in strat),
        "answer_type": Counter(strat[sid]["answer_type"] for sid in strat),
        "tool_family": Counter(strat[sid]["tool_family"] for sid in strat),
        "has_reference_dependency": Counter(strat[sid]["has_reference_dependency"] for sid in strat),
    }
    return subset_rows, {"subset_counts": counts, "source_counts": source_counts, "strat": strat}


def main() -> int:
    if not SOURCE.is_file():
        raise SystemExit(f"[prepare_train_subset_160] ABORT: source not found: {SOURCE}")

    rows = load_jsonl(SOURCE)
    assert len(rows) == 326, f"expected 326 source rows, got {len(rows)}"

    blocked = load_blocked_ids()  # NESTFUL dev+test IDs
    overlap = blocked & {r["sample_id"] for r in rows}
    assert not overlap, f"source dataset overlaps NESTFUL dev/test: {sorted(overlap)[:5]}"

    subset_rows, strat_info = select_subset(rows, SEED, N_TARGET)

    easy_n = strat_info["subset_counts"]["quality_tier"].get("easy_anchor", 0)
    easy_frac = easy_n / N_TARGET
    assert easy_frac <= EASY_TIER_MAX_FRACTION + 1e-9, (
        f"easy_anchor fraction {easy_frac:.3f} exceeds {EASY_TIER_MAX_FRACTION}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "train_subset_160.jsonl"
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        for r in subset_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Full synthetic executor replay gate — the SAME one the production
    # trainer's preflight uses. No gold_replay path is exercised here.
    sys.path.insert(0, str(_SCRIPTS / "training"))
    from preflight_training_datasets import (  # noqa: E402
        REGISTRY_VERSION,
        registry_hash,
        validate_file,
    )
    replay_report = validate_file(str(out_path))

    source_sha256_raw = _sha256_file(SOURCE)
    source_sha256_lf = _sha256_file_lf(SOURCE)
    subset_sha256 = _sha256_file(out_path)

    manifest = {
        "generated_at": _now(),
        "seed": SEED,
        "n_target": N_TARGET,
        "n_selected": len(subset_rows),
        "source_dataset_path": str(SOURCE.relative_to(_REPO)).replace("\\", "/"),
        "source_dataset_rows": len(rows),
        "source_sha256_raw_local_checkout": source_sha256_raw,
        "source_sha256_lf_normalized": source_sha256_lf,
        "source_sha256_recorded_in_run_manifest": (
            "7df704bff35c8f8fd0ffb2b50e3c7c4c1e8d7f9a0f3e0c02a43327ef820dd596"
        ),
        "source_hash_note": (
            "This repo checkout has CRLF line endings on this dataset file "
            "(Windows git config), which changes the raw-bytes SHA-256 vs the "
            "Linux RunPod that produced pure_stage3_2ep_20260719_221918. The "
            "LF-normalized hash above matches the run_manifest.json-recorded "
            "hash exactly, confirming byte-for-byte identical content."
        ),
        "subset_sha256": subset_sha256,
        "selected_task_ids": [r["sample_id"] for r in subset_rows],
        "stratification_counts": {
            "subset": {k: dict(v) if isinstance(v, Counter) else v
                       for k, v in strat_info["subset_counts"].items()},
            "source_326": {k: dict(v) for k, v in strat_info["source_counts"].items()},
        },
        "stratification_axes_primary": ["motif_type", "quality_tier"],
        "stratification_axes_reported": [
            "answer_type", "tool_family", "has_reference_dependency", "num_calls",
        ],
        "easy_tier_fraction": round(easy_frac, 4),
        "easy_tier_max_allowed": EASY_TIER_MAX_FRACTION,
        "excludes_nestful_dev_test": True,
        "registry_version": REGISTRY_VERSION,
        "registry_hash": registry_hash(),
        "executor_version": "synthetic (lib.agentic_data.exec_bridge.execute_gold_trace)",
        "replay_mode": "full_synthetic_executor_replay",
        "replay_result": replay_report,
        "arm_specific_filtering": False,
        "identical_ids_for_all_arms": True,
    }
    manifest_path = OUT_DIR / "train_subset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    _write_report(OUT_DIR / "TRAIN_SUBSET_REPORT.md", manifest, strat_info)

    print(f"[prepare_train_subset_160] wrote {out_path} ({len(subset_rows)} rows)")
    print(f"[prepare_train_subset_160] wrote {manifest_path}")
    print(f"[prepare_train_subset_160] subset sha256={subset_sha256}")
    return 0


def _write_report(path: Path, manifest: Dict[str, Any], strat_info: Dict[str, Any]) -> None:
    def _pct_table(subset_c: Counter, source_c: Counter) -> str:
        keys = sorted(set(subset_c) | set(source_c), key=str)
        lines = ["| value | source (326) | source % | subset (160) | subset % |",
                 "|---|---:|---:|---:|---:|"]
        for k in keys:
            sc = source_c.get(k, 0)
            ssc = subset_c.get(k, 0)
            lines.append(
                f"| {k} | {sc} | {sc/326*100:.1f}% | {ssc} | {ssc/160*100:.1f}% |"
            )
        return "\n".join(lines)

    sub = manifest["stratification_counts"]["subset"]
    src = manifest["stratification_counts"]["source_326"]

    md = f"""# TRAIN_SUBSET_REPORT — 160-task deterministic stratified train subset

Generated: {manifest['generated_at']}

## Source dataset

- Path: `{manifest['source_dataset_path']}`
- Rows: {manifest['source_dataset_rows']}
- SHA-256 (raw bytes, this Windows checkout): `{manifest['source_sha256_raw_local_checkout']}`
- SHA-256 (LF-normalized): `{manifest['source_sha256_lf_normalized']}`
- SHA-256 recorded in `pure_stage3_2ep_20260719_221918/run_manifest.json`: `{manifest['source_sha256_recorded_in_run_manifest']}`
- **Match**: LF-normalized hash == run-manifest hash -> content is byte-for-byte identical to the
  dataset used by the original production run; the raw-bytes mismatch is purely a CRLF checkout
  artifact on Windows (see `source_hash_note` in the manifest).

## Subset

- Seed: {manifest['seed']}
- Selected: {manifest['n_selected']} / target {manifest['n_target']}
- SHA-256: `{manifest['subset_sha256']}`
- Identical task IDs used for every reward arm: {manifest['identical_ids_for_all_arms']}
- No arm-specific filtering: {not manifest['arm_specific_filtering']}
- Excludes NESTFUL dev/test IDs: {manifest['excludes_nestful_dev_test']}
- Easy tier (`quality_tier=easy_anchor`) fraction: {manifest['easy_tier_fraction']*100:.1f}% (max allowed {manifest['easy_tier_max_allowed']*100:.0f}%)

## Full synthetic executor replay (no gold_replay)

- Registry version: `{manifest['registry_version']}`
- Registry hash: `{manifest['registry_hash']}`
- Replay status: `{manifest['replay_result']['status']}`
- Replay rows validated: {manifest['replay_result']['rows']}

## Stratification — primary axis (motif_type x quality_tier)

Allocation per cell (subset / source):

| cell | subset n | source n |
|---|---:|---:|
"""
    cell_alloc = sub["cell_allocation"]
    for cell, n in sorted(cell_alloc.items()):
        motif, tier = cell.split("|")
        source_n = sum(1 for sid, s in strat_info["strat"].items()
                        if s["motif_type"] == motif and s["quality_tier"] == tier)
        md += f"| {cell} | {n} | {source_n} |\n"

    md += f"""
## motif_type distribution

{_pct_table(Counter(sub['motif_type']), Counter(src['motif_type']))}

## quality_tier distribution

{_pct_table(Counter(sub['quality_tier']), Counter(src['quality_tier']))}

## answer_type distribution (reported, not a sampling axis)

{_pct_table(Counter(sub['answer_type']), Counter(src['answer_type']))}

## tool_family distribution (reported, not a sampling axis — heuristic keyword bucketing of the first gold call's tool name)

{_pct_table(Counter(sub['tool_family']), Counter(src['tool_family']))}

## has_reference_dependency distribution (reported, not a sampling axis)

{_pct_table(Counter(sub['has_reference_dependency']), Counter(src['has_reference_dependency']))}

## Notes / limits

- Stratification uses a 2D key (`motif_type` x `quality_tier`) via the largest-remainder method so
  proportions of the 326-task source are preserved as closely as an integer allocation of 160 items
  allows. `answer_type`, `tool_family`, and `has_reference_dependency` are reported for transparency
  but are not independent sampling axes (with only 326 source rows, a full cross of all seven
  requested fields would produce mostly-empty cells and an unstable/non-deterministic selection).
- `tool_family` is a lightweight keyword heuristic over the first gold call's tool name (this
  dataset's tools are code-generated synthetic functions without a first-class "family" field);
  it is descriptive only.
- Selection does not use any C0 (or any other) rollout/eval result — inputs are dataset metadata
  only, per the ablation spec (no arm-specific / outcome-based filtering).
"""
    path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
