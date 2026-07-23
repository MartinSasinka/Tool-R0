"""Provider statistics from raw annotation logs."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from weak_audit.agreement import validate_raw_rows
from weak_audit.io_utils import read_jsonl, write_json


def _is_truncation(raw: dict) -> bool:
    text = (raw.get("response") or "").strip()
    if not text:
        return False
    if not text.rstrip().endswith("}"):
        return True
    try:
        json.loads(text)
        return False
    except json.JSONDecodeError:
        return True


def compute_provider_stats(out_dir: Path) -> Tuple[dict, Optional[str]]:
    valid_keys = set()
    for label in ("A", "B"):
        for row in read_jsonl(out_dir / f"pass_{label.lower()}_annotations.jsonl"):
            valid_keys.add((row.get("task_id"), label))

    stats: Dict[str, dict] = defaultdict(lambda: {
        "requests": 0,
        "valid": 0,
        "invalid": 0,
        "truncation": 0,
        "pass_a": 0,
        "pass_b": 0,
        "latency_sum": 0.0,
        "prompt_tokens_sum": 0,
        "completion_tokens_sum": 0,
        "cost_sum": 0.0,
    })

    for label in ("A", "B"):
        raw_path = out_dir / f"pass_{label.lower()}_annotations_raw.jsonl"
        for row in read_jsonl(raw_path):
            prov = row.get("provider") or "unknown"
            s = stats[prov]
            s["requests"] += 1
            if label == "A":
                s["pass_a"] += 1
            else:
                s["pass_b"] += 1
            s["latency_sum"] += float(row.get("latency_s") or 0)
            s["prompt_tokens_sum"] += int(row.get("prompt_tokens") or 0)
            s["completion_tokens_sum"] += int(row.get("completion_tokens") or 0)
            s["cost_sum"] += float(row.get("reported_cost") or row.get("cost_usd") or 0)
            key = (row.get("task_id"), label)
            if key in valid_keys:
                s["valid"] += 1
            else:
                s["invalid"] += 1
            if _is_truncation(row):
                s["truncation"] += 1

    rows: List[dict] = []
    for prov, s in stats.items():
        n = s["requests"] or 1
        rows.append({
            "provider": prov,
            **s,
            "valid_rate": s["valid"] / n,
            "invalid_rate": s["invalid"] / n,
            "truncation_rate": s["truncation"] / n,
            "mean_latency_s": s["latency_sum"] / n,
            "mean_prompt_tokens": s["prompt_tokens_sum"] / n,
            "mean_completion_tokens": s["completion_tokens_sum"] / n,
            "mean_cost_usd": s["cost_sum"] / n,
        })

    def _rank(r: dict) -> tuple:
        return (
            r["valid_rate"],
            -r["truncation_rate"],
            r["requests"],
            -r["mean_latency_s"],
        )

    eligible = [r for r in rows if r["requests"] >= 10 and r["valid"] > 0]
    recommended = max(eligible, key=_rank)["provider"] if eligible else None

    report = {
        "providers": sorted(rows, key=lambda r: (-r["valid_rate"], r["truncation_rate"])),
        "recommended_provider": recommended,
        "selection_rule": (
            "highest valid_rate, then lowest truncation_rate, "
            "then most requests, then lower latency"
        ),
    }
    return report, recommended


def write_provider_audit(out_dir: Path) -> Tuple[dict, Optional[str]]:
    report, recommended = compute_provider_stats(out_dir)
    write_json(out_dir / "provider_stats.json", report)
    lines = [
        "# Provider audit (original real run)",
        "",
        f"**Recommended for retry:** `{recommended}`",
        "",
        "| Provider | requests | valid_rate | trunc_rate | mean_latency | mean_cost |",
        "|----------|---------:|-----------:|-----------:|-------------:|----------:|",
    ]
    for r in report["providers"]:
        lines.append(
            f"| {r['provider']} | {r['requests']} | {r['valid_rate']:.3f} | "
            f"{r['truncation_rate']:.3f} | {r['mean_latency_s']:.2f}s | "
            f"${r['mean_cost_usd']:.6f} |"
        )
    (out_dir / "PROVIDER_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return report, recommended
