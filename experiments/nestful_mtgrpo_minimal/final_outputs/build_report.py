"""Build a consolidated NESTFUL evaluation report from outputs/.

Re-scores EVERY run with the *same* official NESTFUL scorer
(`nestful_official_score.score_items`) so the comparison is apples-to-apples and
free of the stale / internal-definition metrics that were stored at run time.

What it does
------------
1. Reconstructs the predicted call sequence for each run:
     - Direct runs:      `direct_predictions.jsonl` -> `predicted_calls`
     - ReAct full eval:  `final_eval_trajectories.jsonl` -> `_traj.turns[].parsed_call`
     - Curriculum evals: `eval/rollout_eval_trajectories.jsonl` -> turns parsed_call
2. Scores them with the official scorer (corpus macro-F1 Func/Param + grounded
   partial/full sequence accuracy). Win Rate is only computed on Linux (SIGALRM);
   on Windows it is left as None and any pre-existing official win_rate is reused.
3. Writes CSV + Markdown tables, per-run `metrics_official.json` files under
   `final_outputs/runs/`, and a machine-readable JSON into this folder.

Run:  python final_outputs/build_report.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # experiments/nestful_mtgrpo_minimal
_OUT = os.path.join(_ROOT, "outputs")
_RUNS_DIR = os.path.join(_HERE, "runs")  # per-run metrics_official.json copies
sys.path.insert(0, _ROOT)

from nestful_official_score import build_item, load_raw_dataset, score_items  # noqa: E402

_DATASET = os.path.join(_ROOT, "data", "NESTFUL-main", "data_v2", "nestful_data.jsonl")
_FUNC_DIR = os.path.join(_ROOT, "data", "NESTFUL-main", "data_v2", "executable_functions")
_WANT_WIN = os.name != "nt" and os.path.isdir(_FUNC_DIR)  # Win Rate needs SIGALRM (Unix)

# Original curriculum eval framework (multi-turn ReAct rollouts on full NESTFUL).
_REPO_ROOT = os.path.dirname(os.path.dirname(_ROOT))  # .../Tool-R0
_V2_DIR = os.path.join(_REPO_ROOT, "curricullum", "evaluation", "results_v2_20260617")

RAW = load_raw_dataset(_DATASET)


def _resolve_path(*candidates: str) -> str | None:
    """Return the first existing path (supports local vs pod output layouts)."""
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def _prior_win_from_runs(label: str):
    """Win Rate previously saved in final_outputs/runs/<label>/metrics_official.json.

    Win Rate can only be computed on Linux (SIGALRM). When the report is later
    re-run on Windows, this lets us keep the Linux-computed value instead of
    overwriting it with None.
    """
    p = os.path.join(_RUNS_DIR, label, "metrics_official.json")
    if not os.path.isfile(p):
        return None, None
    try:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        return d.get("win_rate"), d.get("win_source")
    except (OSError, json.JSONDecodeError):
        return None, None


# ---------------------------------------------------------------------------
#  Prediction reconstruction
# ---------------------------------------------------------------------------

def _calls_from_react_turns(traj: dict) -> list:
    """Collect the predicted call sequence from a stored ReAct trajectory.

    Uses the per-turn `parsed_call` (already parsed at run time); stops at the
    first terminal/empty turn, mirroring the live rollout's hard-stop semantics.
    """
    calls = []
    for t in traj.get("turns", []):
        if t.get("is_terminal"):
            break
        c = t.get("parsed_call")
        if isinstance(c, dict) and (c.get("name") or "").strip():
            calls.append(c)
        else:
            break
    return calls


def _load_direct(path: str) -> dict:
    preds = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = str(r.get("sample_id") or r.get("task_id") or "")
            preds[sid] = r.get("predicted_calls") or []
    return preds


def _load_react(path: str) -> dict:
    preds = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = str(r.get("sample_id") or r.get("task_id") or "")
            traj = r.get("_traj", r)
            preds[sid] = _calls_from_react_turns(traj)
    return preds


def _score(preds: dict) -> dict:
    """Official corpus metrics for {sample_id: predicted_calls}. No Win Rate on Windows."""
    items, n_missing = [], 0
    for sid, calls in preds.items():
        gold_row = RAW.get(sid)
        if gold_row is None:
            n_missing += 1
            continue
        items.append(build_item(calls, gold_row))
    if not items:
        return {"num_examples": 0, "num_missing_in_dataset": n_missing}
    m = score_items(items, executable_func_dir=_FUNC_DIR, win_rate=_WANT_WIN)
    m["num_missing_in_dataset"] = n_missing
    return m


def _save_run_metrics(
    label: str,
    scored: dict,
    *,
    model: str,
    paradigm: str,
    source_input: str,
    win_rate=None,
    win_source: str | None = None,
) -> str:
    """Write one run's official metrics into final_outputs/runs/<label>/."""
    run_dir = os.path.join(_RUNS_DIR, label)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "metrics_official.json")
    payload = {
        "run": label,
        "model": model,
        "paradigm": paradigm,
        "source_input": os.path.relpath(source_input, _ROOT) if source_input else None,
        "f1_func": scored.get("f1_func"),
        "f1_param": scored.get("f1_param"),
        "partial_sequence_accuracy": scored.get("partial_sequence_accuracy"),
        "full_sequence_accuracy": scored.get("full_sequence_accuracy"),
        "num_examples": scored.get("num_examples"),
        "num_pred_parsing_errors": scored.get("num_pred_parsing_errors"),
        "num_missing_in_dataset": scored.get("num_missing_in_dataset"),
        "win_rate": win_rate if win_rate is not None else scored.get("win_rate"),
        "win_source": win_source,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[save] {out_path}")
    return out_path


# ---------------------------------------------------------------------------
#  Run registry  (label -> how to load predictions)
# ---------------------------------------------------------------------------

# Each run: (label, model, paradigm, loader, path_candidates, preexisting_official_json_candidates)
FULL_RUNS = [
    ("baseline_react", "baseline (no LoRA)", "react", _load_react, [
        os.path.join(_OUT, "curriculum", "final_eval_baseline_react", "final_eval_trajectories.jsonl"),
        os.path.join(_OUT, "final_eval_baseline_react", "final_eval_trajectories.jsonl"),
    ], []),
    ("baseline_direct", "baseline (no LoRA)", "direct", _load_direct, [
        os.path.join(_OUT, "final_eval", "baseline_direct", "direct_predictions.jsonl"),
        os.path.join(_OUT, "baseline_direct", "direct_predictions.jsonl"),
    ], [
        os.path.join(_OUT, "final_eval", "baseline_direct", "metrics_official.json"),
        os.path.join(_OUT, "baseline_direct", "metrics_official.json"),
    ]),
    ("stage4e2_react", "curriculum s4e2", "react", _load_react, [
        os.path.join(_OUT, "curriculum", "final_eval_ckpt_react", "final_eval_trajectories.jsonl"),
        os.path.join(_OUT, "final_eval_stage4_epoch2_react", "final_eval_trajectories.jsonl"),
    ], []),
    ("stage4e2_direct", "curriculum s4e2", "direct", _load_direct, [
        os.path.join(_OUT, "final_eval", "stage4_epoch2_direct", "direct_predictions.jsonl"),
        os.path.join(_OUT, "stage4_epoch2_direct", "direct_predictions.jsonl"),
    ], [
        os.path.join(_OUT, "final_eval", "stage4_epoch2_direct", "metrics_official.json"),
        os.path.join(_OUT, "stage4_epoch2_direct", "metrics_official.json"),
    ]),
]

METRIC_KEYS = ["f1_func", "f1_param", "partial_sequence_accuracy", "full_sequence_accuracy", "win_rate"]


def build_full_eval_table() -> list:
    rows = []
    for label, model, paradigm, loader, path_candidates, pre_json_candidates in FULL_RUNS:
        path = _resolve_path(*path_candidates)
        if path is None:
            print(f"[skip] {label}: none of {path_candidates!r}")
            continue
        pre_json = _resolve_path(*pre_json_candidates) if pre_json_candidates else None
        print(f"[load] {label}: {path}")
        preds = loader(path)
        scored = _score(preds)
        # Win Rate priority: freshly recomputed (Linux) > prior runs/ value (Linux) >
        # a pre-existing official json in outputs/. On Windows the first is always None.
        win = scored.get("win_rate")
        win_source = "recomputed" if win is not None else None
        if win is None:
            prior_win, prior_src = _prior_win_from_runs(label)
            if prior_win is not None:
                win, win_source = prior_win, (prior_src or "prior_run")
        if win is None and pre_json and os.path.isfile(pre_json):
            with open(pre_json, encoding="utf-8") as fh:
                win = json.load(fh).get("win_rate")
            if win is not None:
                win_source = "prior_run"
        if win_source is None:
            win_source = "needs_linux"
        row = {
            "run": label,
            "model": model,
            "paradigm": paradigm,
            "source_input": os.path.relpath(path, _ROOT),
            "metrics_file": os.path.relpath(
                _save_run_metrics(
                    label, scored,
                    model=model, paradigm=paradigm, source_input=path,
                    win_rate=win, win_source=win_source,
                ), _HERE,
            ),
            "num_examples": scored.get("num_examples", 0),
            "num_pred_parsing_errors": scored.get("num_pred_parsing_errors", 0),
            "f1_func": scored.get("f1_func"),
            "f1_param": scored.get("f1_param"),
            "partial": scored.get("partial_sequence_accuracy"),
            "full": scored.get("full_sequence_accuracy"),
            "win_rate": win,
            "win_source": win_source,
        }
        rows.append(row)
        print(f"[ok] {label}: f1_func={row['f1_func']} partial={row['partial']} "
              f"full={row['full']} win={row['win_rate']} ({row['win_source']})")
    return rows


# ---------------------------------------------------------------------------
#  Original curriculum eval framework (results_v2_*) -> official metrics
# ---------------------------------------------------------------------------

def _load_v2_predictions(path: str, rollout_idx: int = 0) -> dict:
    """One predicted call sequence per task from a results_v2 predictions file.

    These files store `rollout_idx` rollouts (0..k-1) per `task_id`; for
    paper-comparable single-prediction NESTFUL metrics we take one rollout.
    """
    preds = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("rollout_idx", 0) != rollout_idx:
                continue
            preds[str(r["task_id"])] = r.get("predicted_calls") or []
    return preds


V2_RUNS = [
    # (label, model, predictions_file, executor_accuracy_from_summary)
    ("v2_baseline",        "baseline (no LoRA)",  None,                                                       67.18),
    ("v2_stage3_epoch1",   "curriculum s3e1",     "curriculum_stage_3_epoch1_multiturn_predictions.jsonl",    70.74),
    ("v2_stage5_epoch2",   "curriculum s5e2",     "curriculum_stage_5_epoch2_multiturn_predictions.jsonl",    70.14),
]


def build_v2_table() -> list:
    rows = []
    if not os.path.isdir(_V2_DIR):
        print(f"[v2] folder not found: {_V2_DIR}")
        return rows
    for label, model, fname, exec_acc in V2_RUNS:
        row = {"run": label, "model": model, "paradigm": "react(4-rollout)",
               "executor_accuracy_pct": exec_acc}
        if fname is None:
            # baseline has only the executor-accuracy summary, no predictions to re-score
            row.update({"num_examples": None, "f1_func": None, "f1_param": None,
                        "partial": None, "full": None, "win_rate": None, "rollout": "—"})
            rows.append(row)
            print(f"[v2] {label}: only executor accuracy {exec_acc}% (no predictions file)")
            continue
        path = os.path.join(_V2_DIR, fname)
        if not os.path.isfile(path):
            print(f"[v2] skip {label}: missing {path}")
            continue
        preds = _load_v2_predictions(path, rollout_idx=0)
        scored = _score(preds)
        win = scored.get("win_rate")
        win_source = "recomputed" if win is not None else None
        if win is None:
            prior_win, prior_src = _prior_win_from_runs(label)
            if prior_win is not None:
                win, win_source = prior_win, (prior_src or "prior_run")
        if win_source is None:
            win_source = "needs_linux"
        metrics_file = _save_run_metrics(
            label, scored,
            model=model, paradigm="react(4-rollout)", source_input=path,
            win_rate=win, win_source=win_source,
        )
        row.update({
            "rollout": "idx0",
            "source_input": os.path.relpath(path, _REPO_ROOT),
            "metrics_file": os.path.relpath(metrics_file, _HERE),
            "num_examples": scored.get("num_examples", 0),
            "num_pred_parsing_errors": scored.get("num_pred_parsing_errors", 0),
            "f1_func": scored.get("f1_func"),
            "f1_param": scored.get("f1_param"),
            "partial": scored.get("partial_sequence_accuracy"),
            "full": scored.get("full_sequence_accuracy"),
            "win_rate": win,
            "win_source": win_source,
        })
        rows.append(row)
        print(f"[v2] {label}: f1_func={row['f1_func']} partial={row['partial']} "
              f"full={row['full']} exec_acc={exec_acc}%")
    return rows


# ---------------------------------------------------------------------------
#  Curriculum training progression
# ---------------------------------------------------------------------------

def build_curriculum_table() -> list:
    rows = []
    cur = os.path.join(_OUT, "curriculum")
    if not os.path.isdir(cur):
        return rows
    for stage in sorted(os.listdir(cur)):
        sdir = os.path.join(cur, stage)
        if not (stage.startswith("stage_") and os.path.isdir(sdir)):
            continue
        for ep in sorted(os.listdir(sdir)):
            edir = os.path.join(sdir, ep)
            metrics_p = os.path.join(edir, "eval", "metrics.json")
            traj_p = os.path.join(edir, "eval", "rollout_eval_trajectories.jsonl")
            if not (ep.startswith("epoch_") and os.path.isfile(metrics_p)):
                continue
            with open(metrics_p, encoding="utf-8") as fh:
                m = json.load(fh)
            row = {
                "stage": stage.replace("stage_", ""),
                "epoch": ep.replace("epoch_", ""),
                "num_tasks": m.get("num_tasks"),
                "strict_gold_trace_pass": m.get("strict_gold_trace_pass"),
                "final_answer_pass": m.get("final_answer_pass"),
                "zero_tool_calls": m.get("zero_tool_calls"),
                "clipped_rate": m.get("clipped_completion_rate"),
            }
            # Re-score officially from trajectories (eval set = NESTFUL filtered to stage N+1).
            if os.path.isfile(traj_p):
                preds = _load_react(traj_p)
                scored = _score(preds)
                row["off_f1_func"] = scored.get("f1_func")
                row["off_partial"] = scored.get("partial_sequence_accuracy")
                row["off_full"] = scored.get("full_sequence_accuracy")
            rows.append(row)
            print(f"[curriculum] {stage} {ep}: strict={row['strict_gold_trace_pass']} "
                  f"final={row['final_answer_pass']} off_f1={row.get('off_f1_func')}")
    return rows


# ---------------------------------------------------------------------------
#  Writers
# ---------------------------------------------------------------------------

def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return "—" if v is None else str(v)


def write_csv(path: str, rows: list, cols: list):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def write_md_table(fh, rows: list, cols: list, headers: list):
    fh.write("| " + " | ".join(headers) + " |\n")
    fh.write("|" + "|".join(["---"] * len(headers)) + "|\n")
    for r in rows:
        fh.write("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |\n")
    fh.write("\n")


def main() -> int:
    print(f"[report] dataset={_DATASET}")
    print(f"[report] Win Rate {'ENABLED (Linux)' if _WANT_WIN else 'DISABLED (Windows/SIGALRM)'}")

    full = build_full_eval_table()
    v2 = build_v2_table()
    curric = build_curriculum_table()

    # Machine-readable
    with open(os.path.join(_HERE, "consolidated_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"full_eval": full, "curriculum_v2_official": v2, "curriculum": curric,
                   "win_rate_enabled": _WANT_WIN}, fh, indent=2, ensure_ascii=False)

    # CSVs
    full_cols = ["run", "model", "paradigm", "source_input", "metrics_file", "num_examples",
                 "num_pred_parsing_errors", "f1_func", "f1_param", "partial", "full",
                 "win_rate", "win_source"]
    write_csv(os.path.join(_HERE, "nestful_full_eval.csv"), full, full_cols)
    v2_cols = ["run", "model", "paradigm", "rollout", "source_input", "metrics_file",
               "num_examples", "executor_accuracy_pct", "f1_func", "f1_param", "partial",
               "full", "win_rate", "win_source"]
    write_csv(os.path.join(_HERE, "curriculum_v2_official.csv"), v2, v2_cols)
    cur_cols = ["stage", "epoch", "num_tasks", "strict_gold_trace_pass", "final_answer_pass",
                "zero_tool_calls", "clipped_rate", "off_f1_func", "off_partial", "off_full"]
    write_csv(os.path.join(_HERE, "curriculum_training.csv"), curric, cur_cols)

    # Markdown summary
    with open(os.path.join(_HERE, "RESULTS.md"), "w", encoding="utf-8") as fh:
        fh.write("# NESTFUL — consolidated results\n\n")
        fh.write("All numbers below are produced by the **official NESTFUL scorer** "
                 "(corpus macro-F1 + grounded partial/full). They were re-computed "
                 "from saved predictions/trajectories so every run uses identical "
                 "scoring. See `ANALYSIS.md` for methodology and caveats.\n\n")
        have_win = [r for r in full if r.get("win_rate") is not None]
        if _WANT_WIN:
            win_note = "recomputed on this run (Linux/SIGALRM)"
        elif have_win:
            win_note = ("**reused from a prior Linux run** (stored in `runs/<run>/metrics_official.json`); "
                        "Win Rate cannot be (re)computed on Windows — SIGALRM is Linux-only")
        else:
            win_note = ("**not available** — Win Rate needs Linux/SIGALRM; run `build_report.py` on Linux")
        fh.write(f"> Win Rate: {win_note}.\n\n")

        fh.write("## Full NESTFUL eval (1861 tasks): baseline vs curriculum, ReAct vs Direct\n\n")
        write_md_table(
            fh, full,
            ["run", "model", "paradigm", "f1_func", "f1_param", "partial", "full", "win_rate", "win_source", "num_pred_parsing_errors"],
            ["run", "model", "paradigm", "F1 Func", "F1 Param", "Partial", "Full", "Win", "win src", "parse_err"],
        )

        fh.write("## Original curriculum eval (results_v2) re-scored to NESTFUL metrics\n\n")
        fh.write("Multi-turn ReAct rollouts on the full 1861-task NESTFUL. The original "
                 "framework reported only executor-based final-answer accuracy "
                 "(`executor_accuracy_pct`); the official NESTFUL metrics below are "
                 "re-computed from `predicted_calls` (rollout idx 0, one prediction/task). "
                 "Baseline has no stored predictions, only its executor accuracy.\n\n")
        write_md_table(
            fh, v2,
            ["run", "model", "rollout", "executor_accuracy_pct", "f1_func", "f1_param", "partial", "full", "win_rate"],
            ["run", "model", "rollout", "exec acc %", "F1 Func", "F1 Param", "Partial", "Full", "Win"],
        )

        fh.write("## Curriculum training progression (small per-stage eval)\n\n")
        fh.write("`strict_gold_trace_pass` / `final_answer_pass` are the training-time "
                 "eval metrics; `off_*` are official re-scores of the same trajectories.\n\n")
        write_md_table(
            fh, curric,
            ["stage", "epoch", "num_tasks", "strict_gold_trace_pass", "final_answer_pass",
             "zero_tool_calls", "off_f1_func", "off_partial", "off_full"],
            ["Stage", "Epoch", "N", "strict_pass", "final_pass", "zero_calls",
             "off F1 Func", "off Partial", "off Full"],
        )

    print(f"\n[report] wrote into {_HERE}:")
    print("  RESULTS.md, nestful_full_eval.csv, curriculum_training.csv, consolidated_metrics.json")
    print(f"  per-run metrics: {_RUNS_DIR}/<run>/metrics_official.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
