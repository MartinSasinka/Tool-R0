#!/usr/bin/env python3
"""Count NESTFUL benchmark tasks by gold tool-call depth (output chain length)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATHS = [
    REPO_ROOT / "eval" / "data" / "NESTFUL-main" / "data_v2" / "nestful_data.jsonl",
    REPO_ROOT / "nestful_repo" / "data_v2" / "nestful_data.jsonl",
]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def resolve_data_path(explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"[err] dataset not found: {path}")
        return path
    for candidate in DEFAULT_DATA_PATHS:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "[err] NESTFUL dataset not found. Pass --data or clone nestful_repo / eval/data/NESTFUL-main."
    )


def parse_output(row: Dict[str, Any]) -> List[Any]:
    output = row.get("output")
    if output is None:
        return []
    if isinstance(output, str):
        output = json.loads(output)
    if isinstance(output, list):
        return output
    return []


def load_tasks(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    tasks: List[Dict[str, Any]] = []
    stats = {"lines": 0, "malformed": 0}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            stats["lines"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed"] += 1
                continue
            if not isinstance(row, dict):
                stats["malformed"] += 1
                continue
            gold_calls = parse_output(row)
            tasks.append(
                {
                    "sample_id": row.get("sample_id", f"line_{line_no}"),
                    "num_gold_calls": len(gold_calls),
                    "input_preview": (row.get("input") or "")[:120],
                }
            )
    return tasks, stats


def build_report(tasks: List[Dict[str, Any]], data_path: Path, load_stats: Dict[str, int]) -> Dict[str, Any]:
    counts = Counter(t["num_gold_calls"] for t in tasks)
    total = len(tasks)
    by_calls = []
    for num_calls in sorted(counts):
        n = counts[num_calls]
        by_calls.append(
            {
                "num_gold_calls": num_calls,
                "task_count": n,
                "percent": round(100.0 * n / total, 2) if total else 0.0,
                "cumulative_percent": round(
                    100.0 * sum(counts[k] for k in sorted(counts) if k <= num_calls) / total, 2
                )
                if total
                else 0.0,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(data_path),
        "dataset_path_relative": os.path.relpath(data_path, REPO_ROOT),
        "total_tasks": total,
        "min_gold_calls": min(counts) if counts else 0,
        "max_gold_calls": max(counts) if counts else 0,
        "avg_gold_calls": round(sum(t["num_gold_calls"] for t in tasks) / total, 3) if total else 0.0,
        "load_stats": load_stats,
        "by_num_gold_calls": by_calls,
        "curriculum_buckets": _curriculum_buckets(counts, total),
        "sample_ids_by_num_gold_calls": {
            str(k): [t["sample_id"] for t in tasks if t["num_gold_calls"] == k]
            for k in sorted(counts)
        },
    }


def _curriculum_buckets(counts: Counter, total: int) -> List[Dict[str, Any]]:
    """Group counts for 1/2/3-call curriculum stages vs longer chains."""
    specs = [
        ("exactly_1_call", lambda k: k == 1),
        ("exactly_2_calls", lambda k: k == 2),
        ("exactly_3_calls", lambda k: k == 3),
        ("4_or_more_calls", lambda k: k >= 4),
    ]
    rows = []
    for label, pred in specs:
        n = sum(v for k, v in counts.items() if pred(k))
        rows.append(
            {
                "bucket": label,
                "task_count": n,
                "percent": round(100.0 * n / total, 2) if total else 0.0,
            }
        )
    return rows


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# NESTFUL — rozložení úloh podle počtu gold tool callů",
        "",
        f"- **Dataset:** `{report['dataset_path_relative']}`",
        f"- **Celkem úloh:** {report['total_tasks']}",
        f"- **Průměr callů / úloha:** {report['avg_gold_calls']}",
        f"- **Rozsah:** {report['min_gold_calls']} – {report['max_gold_calls']} callů",
        f"- **Generováno:** {report['generated_at']}",
        "",
        "| Gold tool calls | Počet úloh | Podíl | Kumulativně |",
        "|----------------:|-----------:|------:|------------:|",
    ]
    for row in report["by_num_gold_calls"]:
        lines.append(
            f"| {row['num_gold_calls']} | {row['task_count']} | {row['percent']:.2f} % | {row['cumulative_percent']:.2f} % |"
        )
    lines.extend(
        [
            "",
            "## Curriculum-style buckets",
            "",
            "| Bucket | Počet úloh | Podíl |",
            "|--------|----------:|------:|",
        ]
    )
    for row in report["curriculum_buckets"]:
        label = row["bucket"].replace("_", " ")
        lines.append(f"| {label} | {row['task_count']} | {row['percent']:.2f} % |")
    lines.extend(
        [
            "",
            "## Poznámka",
            "",
            "Počet callů = délka pole `output` (gold trajectory) v `nestful_data.jsonl`,",
            "stejně jako `len(gold_calls)` v `nestful_evaluation/run.py`.",
            "",
            "NESTFUL v2 nemá úlohy s 1 gold callem — minimum je typicky **2 callů**.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="NESTFUL task distribution by gold tool-call count")
    ap.add_argument("--data", default=None, help="Path to nestful_data.jsonl")
    ap.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for JSON + Markdown report",
    )
    args = ap.parse_args()

    data_path = resolve_data_path(args.data)
    tasks, load_stats = load_tasks(data_path)
    if not tasks:
        raise SystemExit(f"[err] no tasks loaded from {data_path}")

    report = build_report(tasks, data_path, load_stats)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "nestful_call_distribution.json"
    md_path = out_dir / "nestful_call_distribution.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"[ok] loaded {report['total_tasks']} tasks from {data_path}")
    print(f"[ok] avg gold calls: {report['avg_gold_calls']}")
    for row in report["by_num_gold_calls"]:
        print(
            f"  {row['num_gold_calls']} call(s): {row['task_count']} tasks "
            f"({row['percent']:.1f}%)"
        )
    print(f"[ok] wrote {json_path}")
    print(f"[ok] wrote {md_path}")


if __name__ == "__main__":
    main()
