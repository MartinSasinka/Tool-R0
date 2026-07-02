"""Diagnostic metrics that contextualize the saturated official macro-F1.

For every final_eval run (Direct + ReAct, all experiments) we compute, over the
shared NESTFUL gold (1861 tasks):

  - macro_f1_func   : official metric (MultiLabelBinarizer, average='macro')
  - micro_f1_func   : same labels, average='micro' (frequency-weighted)
  - set_match       : per-task share where pred function-name MULTISET == gold
  - seq_match       : per-task share where pred ordered name sequence == gold

Full Sequence Acc + Win Rate are read from each run's metrics_official.json.

Predicted call names come from:
  - Direct : <run>/direct_predictions.jsonl  -> predicted_calls[].name
  - ReAct  : <run>/final_eval_trajectories.jsonl -> _traj.turns[].parsed_call.name
             (terminal / empty turns skipped)
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.abspath(os.path.join(_HERE, ".."))
_MIN = os.path.join(_EXP, "nestful_mtgrpo_minimal")
_PARTIAL = os.path.join(_EXP, "nestful_mtgrpo_partial")
_DATASET = os.path.join(_MIN, "data", "NESTFUL-main", "data_v2", "nestful_data.jsonl")

from sklearn.preprocessing import MultiLabelBinarizer  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402


def _load_gold():
    gold = {}
    with open(_DATASET, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = str(r.get("sample_id") or r.get("task_id") or r.get("id"))
            out = r.get("output")
            if isinstance(out, str):
                try:
                    out = json.loads(out)
                except json.JSONDecodeError:
                    out = []
            gold[sid] = [c.get("name") for c in out if isinstance(c, dict)]
    return gold


def _names_direct(path):
    """sample_id -> list of predicted function names (Direct predictions)."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id") or row.get("task_id"))
            out[sid] = [c.get("name") for c in (row.get("predicted_calls") or [])
                        if isinstance(c, dict) and c.get("name")]
    return out


def _names_react(path):
    """sample_id -> predicted names from a ReAct trajectory file."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id") or row.get("task_id"))
            traj = row.get("_traj")
            if isinstance(traj, str):
                try:
                    traj = json.loads(traj)
                except json.JSONDecodeError:
                    traj = {}
            turns = (traj or {}).get("turns", []) if isinstance(traj, dict) else []
            names = []
            for t in turns:
                if t.get("is_terminal"):
                    break
                pc = t.get("parsed_call")
                if isinstance(pc, dict) and pc.get("name"):
                    names.append(pc["name"])
            out[sid] = names
    return out


def _diag(gold, pred_by_id):
    gi, pi = [], []
    for sid, g in gold.items():
        gi.append(g)
        pi.append(pred_by_id.get(sid, []))
    b = MultiLabelBinarizer()
    b.fit(gi)
    Y, P = b.transform(gi), b.transform(pi)
    macro = float(f1_score(Y, P, average="macro", zero_division=0))
    micro = float(f1_score(Y, P, average="micro", zero_division=0))
    n = len(gi)
    set_match = sum(1 for g, p in zip(gi, pi)
                    if collections.Counter(g) == collections.Counter(p)) / n
    seq_match = sum(1 for g, p in zip(gi, pi) if g == p) / n
    return {
        "macro_f1_func": round(macro, 3),
        "micro_f1_func": round(micro, 3),
        "set_match": round(set_match, 3),
        "seq_match": round(seq_match, 3),
        "n": n,
    }


def _metrics_file(run_dir):
    p = os.path.join(run_dir, "metrics_official.json")
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _discover(base, experiment):
    """Yield (label, experiment, paradigm, pred_path, kind, run_dir)."""
    for run_dir in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(run_dir):
            continue
        name = os.path.basename(run_dir)
        if name.startswith("_logs"):
            continue
        direct = os.path.join(run_dir, "direct_predictions.jsonl")
        react = os.path.join(run_dir, "final_eval_trajectories.jsonl")
        if os.path.isfile(direct):
            yield (name, experiment, "direct", direct, "direct", run_dir)
        elif os.path.isfile(react):
            yield (name, experiment, "react", react, "react", run_dir)


def main():
    gold = _load_gold()
    rows = []
    sources = list(_discover(os.path.join(_MIN, "outputs", "final_eval"), "minimal")) + \
        list(_discover(os.path.join(_PARTIAL, "outputs", "final_eval"), "partial"))

    for label, experiment, paradigm, path, kind, run_dir in sources:
        pred = _names_direct(path) if kind == "direct" else _names_react(path)
        d = _diag(gold, pred)
        m = _metrics_file(run_dir)
        rows.append({
            "run": label,
            "experiment": experiment,
            "paradigm": paradigm,
            "macro_f1_func": d["macro_f1_func"],
            "micro_f1_func": d["micro_f1_func"],
            "set_match": d["set_match"],
            "seq_match": d["seq_match"],
            "partial": m.get("partial_sequence_accuracy"),
            "full": m.get("full_sequence_accuracy"),
            "win_rate": m.get("win_rate"),
            "n": d["n"],
        })

    out_json = os.path.join(_HERE, "diagnostics.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    cols = ["run", "experiment", "paradigm", "macro_f1_func", "micro_f1_func",
            "set_match", "seq_match", "partial", "full", "win_rate", "n"]
    with open(os.path.join(_HERE, "diagnostics.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out_json} ({len(rows)} runs)\n")
    hdr = f"{'run':<24}{'par':<7}{'macroF1':>8}{'microF1':>8}{'set':>7}{'seq':>7}{'full':>7}{'win':>7}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        def f(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) else "-"
        print(f"{r['run']:<24}{r['paradigm']:<7}{f(r['macro_f1_func']):>8}{f(r['micro_f1_func']):>8}"
              f"{f(r['set_match']):>7}{f(r['seq_match']):>7}{f(r['full']):>7}{f(r['win_rate']):>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
