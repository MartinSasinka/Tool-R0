#!/usr/bin/env python3
"""Join N+1 curriculum eval with full NESTFUL final eval → nplus1_vs_full_transfer.csv."""
from __future__ import annotations

import csv
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ALL = os.path.join(_HERE, "all_runs.json")
_OUT = os.path.join(_HERE, "nplus1_vs_full_transfer.csv")


def _parse_minimal_ckpt(checkpoint: str) -> tuple[int, int, str]:
    if checkpoint == "curriculum s4e2":
        return 4, 2, checkpoint
    base = checkpoint.replace("s", "", 1)
    stage_s, epoch_s = base.split("_e")
    return int(stage_s), int(epoch_s), checkpoint


def main() -> None:
    with open(_ALL, encoding="utf-8") as fh:
        data = json.load(fh)

    curriculum: dict[tuple[str, int, int], dict] = {}
    for exp, key in (("partial", "partial_curriculum"), ("minimal", "minimal_curriculum")):
        for row in data[key]:
            s, e = int(row["stage"]), int(row["epoch"])
            curriculum[(exp, s, e)] = row

    final: dict[tuple[str, int, int], dict] = {}
    for row in data["partial_final_eval"]:
        if row["experiment"] == "baseline":
            continue
        s, e = int(row["stage"]), int(row["epoch"])
        k = ("partial", s, e)
        final.setdefault(
            k,
            {
                "checkpoint": row["checkpoint"],
                "experiment": row["experiment"],
                "stage": s,
                "epoch": e,
            },
        )
        final[k][f"full_{row['paradigm']}_win_rate"] = row["win_rate"]

    for row in data["minimal_final_eval"]:
        if row["experiment"] == "baseline":
            continue
        s, e, ck = _parse_minimal_ckpt(row["checkpoint"])
        k = ("minimal", s, e)
        final.setdefault(
            k,
            {"checkpoint": ck, "experiment": row["experiment"], "stage": s, "epoch": e},
        )
        final[k][f"full_{row['paradigm']}_win_rate"] = row["win_rate"]

    rows: list[dict] = []
    for (exp, s, e), loc in sorted(curriculum.items()):
        fe = final.get((exp, s, e))
        if not fe or fe.get("full_react_win_rate") is None:
            continue
        local_fap = float(loc["final_answer_pass"])
        local_st = float(loc["strict_gold_trace_pass"])
        react = float(fe["full_react_win_rate"])
        direct = fe.get("full_direct_win_rate")
        rows.append(
            {
                "experiment": fe["experiment"],
                "checkpoint": fe["checkpoint"],
                "stage": s,
                "epoch": e,
                "local_final_answer_pass": round(local_fap, 4),
                "local_strict_gold_trace_pass": round(local_st, 4),
                "full_react_win_rate": round(react, 4),
                "full_direct_win_rate": round(float(direct), 4) if direct is not None else "",
                "transfer_gap": round(local_fap - react, 4),
            }
        )

    fields = [
        "experiment",
        "checkpoint",
        "stage",
        "epoch",
        "local_final_answer_pass",
        "local_strict_gold_trace_pass",
        "full_react_win_rate",
        "full_direct_win_rate",
        "transfer_gap",
    ]
    with open(_OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {_OUT}")


if __name__ == "__main__":
    main()
