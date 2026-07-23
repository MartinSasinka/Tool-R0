"""Merge original valid annotations with retry results."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple

from weak_audit.io_utils import read_json, read_jsonl, write_jsonl


def _pair_key(row: dict) -> Tuple[str, str]:
    return (row["task_id"], row.get("pass") or row.get("pass_label"))


def merge_final_annotations(
    out_dir: Path,
    *,
    expected_per_pass: int = 248,
) -> dict:
    manifest = read_json(out_dir / "invalid_retry_manifest.json")
    target_pairs = {
        (p["task_id"], p["pass_label"]) for p in manifest.get("pairs") or []
    }

    retry_valid = read_jsonl(out_dir / "retry_invalid_validated.jsonl")
    retry_by_pair = {_pair_key(r): r for r in retry_valid}

    stats = {
        "pass_a_valid_original": 0,
        "pass_b_valid_original": 0,
        "pass_a_replaced": 0,
        "pass_b_replaced": 0,
        "pass_a_still_invalid": 0,
        "pass_b_still_invalid": 0,
    }

    for label in ("A", "B"):
        original = read_jsonl(out_dir / f"pass_{label.lower()}_annotations.jsonl")
        final: List[dict] = []
        seen: Set[str] = set()
        for row in original:
            tid = row["task_id"]
            seen.add(tid)
            final.append(row)
            stats[f"pass_{label.lower()}_valid_original"] += 1

        for tid, pl in sorted(target_pairs):
            if pl != label:
                continue
            if tid in seen:
                continue
            rv = retry_by_pair.get((tid, pl))
            if rv:
                final.append(rv)
                stats[f"pass_{label.lower()}_replaced"] += 1
                seen.add(tid)

        write_jsonl(out_dir / f"pass_{label.lower()}_annotations_final.jsonl", final)

    invalid_final: List[dict] = []
    for tid, pl in sorted(target_pairs):
        final_path = out_dir / f"pass_{pl.lower()}_annotations_final.jsonl"
        present = {r["task_id"] for r in read_jsonl(final_path)}
        if tid not in present:
            stats[f"pass_{pl.lower()}_still_invalid"] += 1
            failed = [
                r for r in read_jsonl(out_dir / "retry_invalid_failed.jsonl")
                if r.get("task_id") == tid and r.get("pass") == pl
            ]
            if failed:
                invalid_final.append(failed[-1])
            else:
                invalid_final.append({"task_id": tid, "pass": pl, "status": "still_invalid"})

    write_jsonl(out_dir / "invalid_annotations_final.jsonl", invalid_final)

    for label in ("A", "B"):
        n = len(read_jsonl(out_dir / f"pass_{label.lower()}_annotations_final.jsonl"))
        stats[f"pass_{label.lower()}_final_valid"] = n
        stats[f"pass_{label.lower()}_final_invalid"] = expected_per_pass - n

    stats["both_pass_final_valid_plus_invalid_eq_248"] = all(
        stats.get(f"pass_{x.lower()}_final_valid", 0)
        + stats.get(f"pass_{x.lower()}_final_invalid", 0)
        == expected_per_pass
        for x in ("A", "B")
    )
    return stats
