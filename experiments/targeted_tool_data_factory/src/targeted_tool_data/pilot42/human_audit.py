"""Stratified human-review artifacts."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from ..util import short_hash


def write_human_audit(records: List[Dict[str, Any]], out_dir: Path,
                      sample_size: int = 300, seed: int = 20260731) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    strata = defaultdict(list)
    for row in records:
        strata[(row.get("workflow_id"), row.get("requested_query_mode"))].append(row)
    for key in strata:
        strata[key].sort(key=lambda r: short_hash([seed, r["task_id"]]))
    sample, keys, i = [], sorted(strata), 0
    while len(sample) < min(sample_size, len(records)) and keys:
        key = keys[i % len(keys)]
        if strata[key]:
            sample.append(strata[key].pop())
            i += 1
        else:
            keys.remove(key)
    columns = ["task_id", "workflow_id", "requested_query_mode", "question",
               "gold_answer", "semantic_alignment", "naturalness",
               "ambiguity", "graph_leak", "reviewer_notes"]
    with (out_dir / "human_audit_sample.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in sample:
            writer.writerow({k: row.get(k, "") for k in columns})
    (out_dir / "HUMAN_AUDIT_GUIDE.md").write_text(
        "# Pilot4.2 human audit\n\nScore semantic alignment, naturalness, ambiguity, "
        "and graph leakage. Do not approve a row with changed facts or an exposed call graph.\n",
        encoding="utf-8")
    with (out_dir / "human_audit_import_template.csv").open(
            "w", encoding="utf-8-sig", newline="") as fh:
        csv.DictWriter(fh, fieldnames=columns).writeheader()
    return {"sample_size": len(sample), "strata": len(strata)}
