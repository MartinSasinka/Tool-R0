#!/usr/bin/env python3
"""Load prediction JSONL + gold data into trajectories_raw.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vizualisation.scripts.lib.io_utils import (  # noqa: E402
    add_repo_to_path,
    all_prediction_paths,
    ensure_run_dir,
    load_config,
    log,
    read_jsonl,
    resolve_path,
    save_config_copy,
    write_jsonl,
)
from vizualisation.scripts.lib.parse_predictions import (  # noqa: E402
    load_gold_index,
    normalize_prediction_row,
)

add_repo_to_path()


def main() -> int:
    p = argparse.ArgumentParser(description="Collect unified trajectory rows.")
    p.add_argument("--config", required=True)
    p.add_argument("--run_dir", default=None)
    p.add_argument(
        "--collect_new",
        action="store_true",
        help="Not implemented: use existing prediction JSONL paths in config.",
    )
    args = p.parse_args()

    if args.collect_new:
        log("collect", "ERROR: --collect_new not implemented; point config at existing JSONL.")
        return 2

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else ensure_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config_copy(cfg, run_dir)

    gold_path = resolve_path(cfg["gold_data_path"])
    if not gold_path.is_file():
        log("collect", f"ERROR: gold data missing: {gold_path}")
        return 2

    gold_index = load_gold_index(gold_path)
    log("collect", f"loaded {len(gold_index)} gold records from {gold_path}")

    pred_paths = all_prediction_paths(cfg)
    rows = []
    malformed = 0

    for checkpoint, path in pred_paths.items():
        log("collect", f"reading {checkpoint}: {path}")
        n = 0
        for raw in read_jsonl(path):
            raw["_source_format"] = "multiturn_eval" if "task_id" in raw else "ideal"
            row = normalize_prediction_row(raw, checkpoint)
            if row.get("_skip"):
                malformed += 1
                continue
            sid = row["sample_id"]
            if sid in gold_index:
                g = gold_index[sid]
                if not row.get("gold_output"):
                    row["gold_output"] = g.get("gold_output")
                if row.get("gold_answer") is None:
                    row["gold_answer"] = g.get("gold_answer")
                if not row.get("input"):
                    row["input"] = g.get("input", "")
                if row.get("tools") is None:
                    row["tools"] = g.get("tools")
            rows.append(row)
            n += 1
        log("collect", f"  {n} rollout rows from {checkpoint}")

    for sid, g in gold_index.items():
        rows.append(
            {
                "sample_id": sid,
                "checkpoint": "gold",
                "rollout_idx": 0,
                "score": 1.0,
                "status": "gold_reference",
                "verdict": "gold",
                "input": g.get("input", ""),
                "tools": g.get("tools"),
                "gold_output": g.get("gold_output"),
                "gold_answer": g.get("gold_answer"),
                "prediction_raw": "",
                "prediction_output": g.get("gold_output"),
                "prediction_answer": g.get("gold_answer"),
                "parse_flags": ["gold_reference"],
                "source_format": "gold",
            }
        )

    out_path = run_dir / "trajectories_raw.jsonl"
    write_jsonl(out_path, rows)
    log("collect", f"wrote {len(rows)} rows -> {out_path}")
    log("collect", f"malformed/skipped rows: {malformed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
