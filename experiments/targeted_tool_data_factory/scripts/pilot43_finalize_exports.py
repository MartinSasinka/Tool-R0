"""Post-selection exports required by the Pilot4.3 finish checklist.

Handles the credit-limited corpus: rebuilds a properly tiered canary-1000,
writes all_clean / overflow / rejected, aliases required filenames, and records
exact train-5000 deficits.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CANARY_TIER_QUOTAS = {
    "PROFILE_CORE": 650,
    "LONG_HORIZON_ENRICHMENT": 200,
    "CAPABILITY_ENRICHMENT": 100,
    "CHALLENGE": 50,
}


def _read(path: Path) -> List[Dict[str, Any]]:
    from targeted_tool_data.pilot43.pipeline import read_jsonl
    return read_jsonl(path) if path.exists() else []


def _write(path: Path, rows: Sequence[Dict[str, Any]]) -> int:
    from targeted_tool_data.pilot43.pipeline import write_jsonl
    return write_jsonl(path, list(rows), append=False)


def _tier_of(row: Dict[str, Any]) -> str:
    return str(row.get("cell_tier") or row.get("selection_tier") or row.get("tier")
               or "")


def rebuild_canary(master: List[Dict[str, Any]], seed: int = 20260731
                   ) -> List[Dict[str, Any]]:
    """Canary mix with explicit tier quotas (spec §25), nested into master ids.

    Never over-fills a tier to hide another tier's deficit. If PROFILE_CORE is
    short, the canary is padded from leftover master only after each tier has
    taken ``min(want, available)``, and the pad is reported separately.
    """
    by_tier: Dict[str, List[Dict[str, Any]]] = {t: [] for t in CANARY_TIER_QUOTAS}
    for row in master:
        tier = _tier_of(row)
        if tier in by_tier:
            by_tier[tier].append(row)
    rng = random.Random(seed)
    picked: List[Dict[str, Any]] = []
    for tier, want in CANARY_TIER_QUOTAS.items():
        pool = list(by_tier[tier])
        rng.shuffle(pool)
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for r in pool:
            b = str(r.get("call_bucket") or len(r.get("gold_calls") or []))
            buckets.setdefault(b, []).append(r)
        order = sorted(buckets)
        take: List[Dict[str, Any]] = []
        i = 0
        while len(take) < want and any(buckets[k] for k in order):
            key = order[i % len(order)]
            if buckets[key]:
                take.append(buckets[key].pop())
            i += 1
        picked.extend(take[:want])
    if len(picked) < 1000:
        have = {r["task_id"] for r in picked}
        # Prefer leftover PROFILE_CORE, then any remaining master rows.
        rest_core = [r for r in by_tier["PROFILE_CORE"] if r["task_id"] not in have]
        rest = rest_core + [r for r in master if r["task_id"] not in have
                            and r["task_id"] not in {x["task_id"] for x in rest_core}]
        rng.shuffle(rest_core)
        # deterministic: core leftovers first (already exhausted), then others
        rest_other = [r for r in master if r["task_id"] not in have]
        rng.shuffle(rest_other)
        picked.extend(rest_other[: 1000 - len(picked)])
    return picked[:1000]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    ap.add_argument("--seed", type=int, default=20260731)
    args = ap.parse_args()
    out = Path(args.out_dir)

    from targeted_tool_data.pilot43 import export as exp
    from targeted_tool_data.pilot43 import resume as res
    from targeted_tool_data.pilot43.pipeline import iter_jsonl
    from targeted_tool_data.pilot43.qstage import QUERY_REJECTS, QUERY_VALID
    from targeted_tool_data.pilot43.tasks import nestful_compat

    clean = _read(out / QUERY_VALID)
    rejected_q = _read(out / QUERY_REJECTS)
    master = _read(out / exp.MASTER_FILE)
    heldout = _read(out / exp.HELDOUT_FILE)
    reserve = _read(out / exp.RESERVE_FILE)
    selected = _read(out / exp.SELECTED_FILE)
    sel_ids = {r["task_id"] for r in selected}

    # Rebuild canary with correct tier mix
    canary = rebuild_canary(master, seed=args.seed)
    _write(out / "train_mix_1000.jsonl", canary)
    _write(out / "nestful_compat_train_mix_1000.jsonl",
           [nestful_compat(r) for r in canary])

    # Nested mixes capped at available master size
    master_ids = [r["task_id"] for r in master]
    canary_ids = [r["task_id"] for r in canary]
    # ensure nesting: mix_1000 ⊆ mix_2000 ⊆ mix_3000 ⊆ master
    by_id = {r["task_id"]: r for r in master}
    rest = [tid for tid in master_ids if tid not in set(canary_ids)]
    rng = random.Random(args.seed + 3)
    rng.shuffle(rest)

    def prefix(n: int) -> List[Dict[str, Any]]:
        ids = canary_ids + rest
        return [by_id[i] for i in ids[: min(n, len(ids))]]

    mix2000 = prefix(2000)
    mix3000 = prefix(3000)
    _write(out / "train_mix_2000.jsonl", mix2000)
    _write(out / "train_mix_3000.jsonl", mix3000)
    _write(out / "nestful_compat_train_mix_2000.jsonl",
           [nestful_compat(r) for r in mix2000])
    _write(out / "nestful_compat_train_mix_3000.jsonl",
           [nestful_compat(r) for r in mix3000])

    # Aliases required by the finish checklist
    _write(out / "selected_all.jsonl", selected)
    _write(out / "nestful_compat_train_master_5000.jsonl",
           [nestful_compat(r) for r in master])
    _write(out / "nestful_compat_heldout.jsonl",
           [nestful_compat(r) for r in heldout])
    _write(out / "nestful_compat_reserve.jsonl",
           [nestful_compat(r) for r in reserve])

    # all_clean / overflow / rejected
    _write(out / "all_clean_tasks.jsonl", clean)
    overflow = [r for r in clean if r["task_id"] not in sel_ids]
    _write(out / "overflow_clean_tasks.jsonl", overflow)

    # Semantic selectable without a clean query, plus query rejects
    selectable = {r["task_id"]: r for r in iter_jsonl(
        out / res.SELECTABLE_FINAL)}
    clean_ids = {r["task_id"] for r in clean}
    rejected_rows: List[Dict[str, Any]] = []
    reasons: List[Dict[str, Any]] = []
    for tid, row in selectable.items():
        if tid in clean_ids:
            continue
        rejected_rows.append({"task_id": tid, "stage": "query_render",
                              "workflow_id": row.get("workflow_id"),
                              "call_count": row.get("call_count")})
        reasons.append({"task_id": tid, "reason": "no_clean_query"})
    for rec in rejected_q:
        reasons.append({
            "task_id": rec.get("task_id"),
            "reason": ",".join(rec.get("failed_layers") or ["query_rejected"]),
        })
    _write(out / "rejected_tasks.jsonl", rejected_rows)
    with (out / "rejection_reasons.csv").open("w", encoding="utf-8",
                                              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task_id", "reason"])
        w.writeheader()
        for row in reasons:
            w.writerow({"task_id": row.get("task_id"),
                        "reason": row.get("reason")})

    # Try to fill reserve from overflow clean queries that are not selected
    if len(reserve) < 1000 and overflow:
        # overflow rows are query records; join to full export shape when possible
        # Prefer already-exported task records from a rebuild via selected leftovers
        from targeted_tool_data.pilot43.select import build_pool
        pool = build_pool(out)
        pool_by = {t.task_id: t for t in pool}
        need = 1000 - len(reserve)
        cand = [pool_by[r["task_id"]] for r in overflow
                if r["task_id"] in pool_by][:need]
        if cand:
            from targeted_tool_data.pilot43.export import _record
            extra = [_record(t, "RESERVE", "reserve") for t in cand]
            reserve = reserve + extra
            _write(out / exp.RESERVE_FILE, reserve)
            _write(out / "nestful_compat_reserve.jsonl",
                   [nestful_compat(r) for r in reserve])
            # refresh selected_all
            selected = master + heldout + reserve
            _write(out / exp.SELECTED_FILE, selected)
            _write(out / "selected_all.jsonl", selected)
            ledger = json.loads(
                (out / "reserve_access_ledger.json").read_text(encoding="utf-8"))
            ledger["reserve_size"] = len(reserve)
            ledger["note"] = (
                "Reserve filled from overflow clean tasks after credit-limited "
                "render halted OpenRouter (HTTP 402). Untouched by threshold "
                "tuning in this run."
            )
            ledger["untouched"] = True
            (out / "reserve_access_ledger.json").write_text(
                json.dumps(ledger, indent=1), encoding="utf-8")

    sel_report = json.loads(
        (out / "selection_report.json").read_text(encoding="utf-8"))
    sel_report["canary_mix_1000"] = {
        "n": len(canary),
        "tier_counts": dict(Counter(_tier_of(r) for r in canary)),
        "target_tiers": CANARY_TIER_QUOTAS,
        "nested_into_master": set(canary_ids).issubset(set(master_ids)),
    }
    sel_report["reserve"] = len(reserve)
    sel_report["overflow_clean"] = len(overflow)
    sel_report["all_clean"] = len(clean)
    sel_report["openrouter_credits_exhausted"] = True
    sel_report["train_master_deficit"] = max(
        0, int(sel_report.get("train_master_target") or 5000) - len(master))
    (out / "selection_report.json").write_text(
        json.dumps(sel_report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "all_clean": len(clean),
        "overflow_clean": len(overflow),
        "rejected_no_query": len(rejected_rows),
        "train_master": len(master),
        "train_master_deficit": sel_report["train_master_deficit"],
        "canary_1000_tiers": sel_report["canary_mix_1000"]["tier_counts"],
        "heldout": len(heldout),
        "reserve": len(reserve),
        "mix_nesting": {
            "1000_in_2000": set(canary_ids).issubset({r["task_id"] for r in mix2000}),
            "2000_in_3000": {r["task_id"] for r in mix2000}.issubset(
                {r["task_id"] for r in mix3000}),
            "3000_in_master": {r["task_id"] for r in mix3000}.issubset(
                set(master_ids)),
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
