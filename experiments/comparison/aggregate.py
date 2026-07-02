"""Aggregate ALL NESTFUL runs into one comparison dataset.

Pulls from three experiments and emits a single JSON + CSV so the canvas /
report can compare them apples-to-apples:

  1. nestful_mtgrpo_minimal  — final_outputs/consolidated_metrics.json
                               (baseline + curriculum pilot final eval + v2 rescore)
  2. nestful_mtgrpo_partial  — outputs/curriculum/**/eval/metrics.json (per-epoch)
                               + outputs/final_eval/<ckpt>_<paradigm>/metrics_official.json
  3. nestful_grpo            — rescored_official.json (original curriculum, today's metrics)

All "official" numbers come from the same nestful_official_score scorer.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.abspath(os.path.join(_HERE, ".."))

_MINIMAL = os.path.join(_EXP, "nestful_mtgrpo_minimal")
_PARTIAL = os.path.join(_EXP, "nestful_mtgrpo_partial")
_GRPO = os.path.join(_EXP, "nestful_grpo")


def _load(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _partial_curriculum():
    """Per-epoch training eval (strict / final_answer pass) for the partial run."""
    rows = []
    pat = os.path.join(_PARTIAL, "outputs", "curriculum",
                       "stage_*", "epoch_*", "eval", "metrics.json")
    for p in sorted(glob.glob(pat)):
        m = re.search(r"stage_(\d+)[\\/]+epoch_(\d+)", p)
        if not m:
            continue
        d = _load(p)
        rows.append({
            "stage": int(m.group(1)),
            "epoch": int(m.group(2)),
            "num_tasks": d.get("num_tasks"),
            "strict_gold_trace_pass": d.get("strict_gold_trace_pass"),
            "final_answer_pass": d.get("final_answer_pass"),
            "zero_tool_calls": d.get("zero_tool_calls"),
        })
    rows.sort(key=lambda r: (r["stage"], r["epoch"]))
    return rows


def _partial_final_eval():
    """Final full-NESTFUL eval per checkpoint x paradigm for the partial run."""
    rows = []
    base = os.path.join(_PARTIAL, "outputs", "final_eval")
    for d in sorted(glob.glob(os.path.join(base, "partial_*_*"))):
        name = os.path.basename(d)
        mp = os.path.join(d, "metrics_official.json")
        if not os.path.isfile(mp):
            continue
        m = re.match(r"partial_s(\d+)_e(\d+)_(direct|react)", name)
        if not m:
            continue
        md = _load(mp)
        rows.append({
            "experiment": "mtgrpo_partial",
            "checkpoint": f"s{m.group(1)}_e{m.group(2)}",
            "stage": int(m.group(1)),
            "epoch": int(m.group(2)),
            "paradigm": m.group(3),
            "f1_func": md.get("f1_func"),
            "f1_param": md.get("f1_param"),
            "partial": md.get("partial_sequence_accuracy"),
            "full": md.get("full_sequence_accuracy"),
            "win_rate": md.get("win_rate"),
            "num_examples": md.get("num_examples"),
        })
    rows.sort(key=lambda r: (r["stage"], r["epoch"], r["paradigm"]))
    return rows


def _minimal_final_eval():
    """Baseline + ALL minimal curriculum checkpoints from outputs/final_eval/*.

    Prefers per-run metrics_official.json under outputs/final_eval/ (newer, has
    more checkpoints); falls back to consolidated_metrics.json for runs only
    present there (e.g. baseline_react)."""
    cm = _load(os.path.join(_MINIMAL, "final_outputs", "consolidated_metrics.json"))
    rows = []
    seen = set()

    base = os.path.join(_MINIMAL, "outputs", "final_eval")
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        name = os.path.basename(d)
        if name.startswith("_logs") or not os.path.isdir(d):
            continue
        mp = os.path.join(d, "metrics_official.json")
        if not os.path.isfile(mp):
            continue
        md = _load(mp)
        is_base = name.startswith("baseline")
        m = re.match(r"stage(\d+)_epoch(\d+)_(direct|react)", name)
        if is_base:
            ckpt = "baseline (no LoRA)"
            paradigm = md.get("paradigm", "direct")
        elif m:
            ckpt = f"s{m.group(1)}_e{m.group(2)}"
            paradigm = m.group(3)
        else:
            ckpt = name
            paradigm = md.get("paradigm", "direct")
        rows.append({
            "experiment": "baseline" if is_base else "mtgrpo_minimal",
            "checkpoint": ckpt,
            "paradigm": paradigm,
            "f1_func": md.get("f1_func"),
            "f1_param": md.get("f1_param"),
            "partial": md.get("partial_sequence_accuracy"),
            "full": md.get("full_sequence_accuracy"),
            "win_rate": md.get("win_rate"),
            "num_examples": md.get("num_examples"),
        })
        seen.add((ckpt, paradigm))

    # Fill in runs only present in consolidated (e.g. baseline_react).
    for r in cm.get("full_eval", []):
        ckpt = r["model"]
        if (ckpt, r["paradigm"]) in seen:
            continue
        rows.append({
            "experiment": "baseline" if "baseline" in r["run"] else "mtgrpo_minimal",
            "checkpoint": ckpt,
            "paradigm": r["paradigm"],
            "f1_func": r.get("f1_func"),
            "f1_param": r.get("f1_param"),
            "partial": r.get("partial"),
            "full": r.get("full"),
            "win_rate": r.get("win_rate"),
            "num_examples": r.get("num_examples"),
        })
    return rows, cm


def main() -> int:
    partial_curric = _partial_curriculum()
    partial_final = _partial_final_eval()
    minimal_final, minimal_cm = _minimal_final_eval()

    out = {
        "partial_curriculum": partial_curric,
        "partial_final_eval": partial_final,
        "minimal_final_eval": minimal_final,
        "minimal_curriculum": minimal_cm.get("curriculum", []),
        "v2_curriculum_official": minimal_cm.get("curriculum_v2_official", []),
    }
    out_json = os.path.join(_HERE, "all_runs.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    # Combined final-eval CSV (all experiments, both paradigms)
    cols = ["experiment", "checkpoint", "paradigm",
            "f1_func", "f1_param", "partial", "full", "win_rate", "num_examples"]
    all_final = minimal_final + partial_final
    out_csv = os.path.join(_HERE, "final_eval_all.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_final)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    print(f"\npartial curriculum epochs: {len(partial_curric)}")
    print(f"partial final-eval rows  : {len(partial_final)}")
    print(f"minimal final-eval rows  : {len(minimal_final)}")

    print("\n=== PARTIAL final eval (full NESTFUL) ===")
    hdr = f"{'ckpt':<8}{'parad':<8}{'F1func':>8}{'F1par':>7}{'Part':>7}{'Full':>7}{'Win':>7}"
    print(hdr); print("-"*len(hdr))
    for r in partial_final:
        print(f"{r['checkpoint']:<8}{r['paradigm']:<8}{r['f1_func']:>8.3f}{r['f1_param']:>7.3f}"
              f"{r['partial']:>7.3f}{r['full']:>7.3f}{r['win_rate']:>7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
