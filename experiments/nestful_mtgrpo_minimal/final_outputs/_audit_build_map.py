#!/usr/bin/env python3
"""One-shot audit: build checkpoint_eval_map.csv + CHECKPOINT_EVAL_MAP.md."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse the same prediction reconstruction + official scorer as build_report.py.
from build_report import _load_react, _score  # noqa: E402

CUR = _REPO / "outputs" / "curriculum"
RUNS = _HERE / "runs"
V2 = _REPO.parent.parent / "curricullum" / "evaluation" / "results_v2_20260617"

EVAL_SUBSET_DESC = {
    "1": "NESTFUL tasks with num_calls==1 (1-call subset)",
    "2": "NESTFUL tasks with num_calls==2 (2-call subset)",
    "3": "NESTFUL tasks with num_calls==3 (3-call subset)",
    "4": "NESTFUL tasks with num_calls==4 (4-call subset)",
    "5": "NESTFUL tasks with num_calls==5 (5-call subset)",
    "full": "Full NESTFUL benchmark (all call counts, 1861 tasks)",
}

COLUMNS = [
    "source", "run_family", "wandb_name_or_log_name", "display_name",
    "model_checkpoint_stage", "model_checkpoint_epoch", "model_checkpoint_path",
    "eval_subset_stage", "eval_subset_description", "eval_num_tasks",
    "metrics_file_path", "trajectories_file_path", "predictions_file_path",
    "strict_gold_trace_pass", "final_answer_pass", "zero_tool_calls", "clipped_rate",
    "off_f1_func", "off_f1_param", "off_partial", "off_full",
    "official_f1_func_full", "official_f1_param_full", "official_partial_full",
    "official_full_full", "official_win_full", "paradigm", "notes",
]


def _rel(p: Path | str | None) -> str:
    if not p:
        return ""
    p = Path(p)
    # Normalize absolute RunPod paths to repo-relative when possible.
    s = str(p).replace("\\", "/")
    marker = "/experiments/nestful_mtgrpo_minimal/"
    if marker in s:
        s = s.split(marker, 1)[1]
    elif s.startswith("outputs/") or s.startswith("final_outputs/"):
        pass
    else:
        try:
            s = os.path.relpath(s, _REPO).replace("\\", "/")
        except ValueError:
            pass
    return s


def _load_json(p: Path) -> dict:
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _norm_ckpt(train_stage: str, epoch: str) -> str:
    ckpt = CUR / f"stage_{train_stage}" / "checkpoints" / f"adapter_epoch_{epoch}"
    return _rel(ckpt) if ckpt.is_dir() else f"outputs/curriculum/stage_{train_stage}/checkpoints/adapter_epoch_{epoch}"


def _load_v2_preds(path: Path) -> dict:
    preds = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            tid = r.get("sample_id") or r.get("task_id")
            rollouts = r.get("rollouts") or r.get("predictions") or []
            if rollouts and isinstance(rollouts[0], list):
                preds[tid] = rollouts[0]
            elif rollouts:
                preds[tid] = rollouts
            else:
                preds[tid] = r.get("predicted_calls") or []
    return preds


def _off_cols(scored: dict, prefix: str = "") -> dict:
    if not scored or scored.get("error"):
        return {
            "off_f1_func": "", "off_f1_param": "", "off_partial": "", "off_full": "",
            "official_f1_func_full": "", "official_f1_param_full": "",
            "official_partial_full": "", "official_full_full": "", "official_win_full": "",
        }
    is_full = prefix == "full" or scored.get("num_examples", 0) >= 1800
    base = {
        "off_f1_func": scored.get("f1_func"),
        "off_f1_param": scored.get("f1_param"),
        "off_partial": scored.get("partial_sequence_accuracy"),
        "off_full": scored.get("full_sequence_accuracy"),
    }
    if is_full:
        base.update({
            "official_f1_func_full": scored.get("f1_func"),
            "official_f1_param_full": scored.get("f1_param"),
            "official_partial_full": scored.get("partial_sequence_accuracy"),
            "official_full_full": scored.get("full_sequence_accuracy"),
            "official_win_full": scored.get("win_rate"),
        })
    else:
        base.update({
            "official_f1_func_full": "",
            "official_f1_param_full": "",
            "official_partial_full": "",
            "official_full_full": "",
            "official_win_full": "",
        })
    return base


def _curriculum_rows() -> list[dict]:
    rows = []
    if not CUR.is_dir():
        return rows
    for stage_dir in sorted(CUR.glob("stage_*")):
        train_stage = stage_dir.name.replace("stage_", "")
        eval_stage = str(int(train_stage) + 1)
        for ep_dir in sorted(stage_dir.glob("epoch_*")):
            epoch = ep_dir.name.replace("epoch_", "")
            ckpt_path = _norm_ckpt(train_stage, epoch)
            metrics_p = ep_dir / "eval" / "metrics.json"
            traj_p = ep_dir / "eval" / "rollout_eval_trajectories.jsonl"
            if not metrics_p.is_file():
                continue
            m = _load_json(metrics_p)
            scored = _score(_load_react(str(traj_p))) if traj_p.is_file() else {}
            wandb_name = f"eval-stage{eval_stage}-e{epoch}"
            display = f"train-s{train_stage}-e{epoch} → eval subset {eval_stage}-call"
            notes = (
                f"W&B name eval-stage{eval_stage}-e{epoch} uses EVAL subset stage "
                f"(num_calls=={eval_stage}), NOT model stage. "
                f"Model is stage {train_stage} epoch {epoch}."
            )
            if train_stage == "2" and epoch == "3":
                notes += " SOURCE OF 0.488943 confusion: W&B 'eval-stage3-e3' = this row."
            if train_stage == "3" and epoch == "3":
                notes += " Actual stage3-e3 model eval is W&B 'eval-stage4-e3' (0.168 final_answer)."
            row = {
                "source": "mt-grpo curriculum rollout_eval",
                "run_family": "mt-grpo",
                "wandb_name_or_log_name": wandb_name,
                "display_name": display,
                "model_checkpoint_stage": train_stage,
                "model_checkpoint_epoch": epoch,
                "model_checkpoint_path": ckpt_path,
                "eval_subset_stage": eval_stage,
                "eval_subset_description": EVAL_SUBSET_DESC.get(eval_stage, ""),
                "eval_num_tasks": m.get("num_tasks"),
                "metrics_file_path": _rel(metrics_p),
                "trajectories_file_path": _rel(traj_p) if traj_p.is_file() else "",
                "predictions_file_path": "",
                "strict_gold_trace_pass": m.get("strict_gold_trace_pass"),
                "final_answer_pass": m.get("final_answer_pass"),
                "zero_tool_calls": m.get("zero_tool_calls"),
                "clipped_rate": m.get("clipped_completion_rate"),
                "paradigm": "react (rollout_eval)",
                "notes": notes,
            }
            row.update(_off_cols(scored))
            rows.append(row)
    return rows


def _final_eval_rows() -> list[dict]:
    specs = [
        ("baseline_react", "baseline (no LoRA)", None, None,
         _REPO / "outputs" / "final_eval_baseline_react",
         "final_eval_baseline_react", "react"),
        ("stage4e2_react", "curriculum s4e2", "4", "2",
         _REPO / "outputs" / "final_eval_stage4_epoch2_react",
         "final_eval_stage4_epoch2_react", "react"),
        ("baseline_direct", "baseline (no LoRA)", None, None,
         _REPO / "outputs" / "baseline_direct",
         "baseline_direct", "direct"),
        ("stage4e2_direct", "curriculum s4e2", "4", "2",
         _REPO / "outputs" / "stage4_epoch2_direct",
         "stage4_epoch2_direct", "direct"),
    ]
    alt_paths = {
        "stage4e2_direct": [_REPO / "outputs" / "final_eval" / "stage4_epoch2_direct"],
        "baseline_direct": [_REPO / "outputs" / "final_eval" / "baseline_direct"],
    }
    rows = []
    for _key, model_label, stg, ep, out_dir, wandb_subdir, paradigm in specs:
        if not out_dir.is_dir():
            for alt in alt_paths.get(_key, []):
                if alt.is_dir():
                    out_dir = alt
                    break
            else:
                continue
        if paradigm == "direct":
            pred_p = out_dir / "direct_predictions.jsonl"
            traj_p = out_dir / "direct_eval_trajectories.jsonl"
        else:
            pred_p = out_dir / "final_eval_predictions.partial.jsonl"
            traj_p = out_dir / "final_eval_trajectories.jsonl"
        metrics_train = _load_json(out_dir / "metrics.json")
        run_label = _key
        metrics_off = _load_json(RUNS / run_label / "metrics_official.json")
        if not metrics_off:
            metrics_off = _load_json(out_dir / "metrics_official.json")
        if not metrics_off and pred_p.is_file():
            from build_report import _load_direct
            metrics_off = _score(_load_direct(str(pred_p)))
        elif not metrics_off and traj_p.is_file():
            metrics_off = _score(_load_react(str(traj_p)))

        ckpt_path = _norm_ckpt(stg, ep) if stg and ep else ""

        row = {
            "source": "mt-grpo final_eval",
            "run_family": "mt-grpo",
            "wandb_name_or_log_name": wandb_subdir,
            "display_name": f"{model_label} / {paradigm} / full NESTFUL",
            "model_checkpoint_stage": stg or "",
            "model_checkpoint_epoch": ep or "",
            "model_checkpoint_path": ckpt_path,
            "eval_subset_stage": "full",
            "eval_subset_description": EVAL_SUBSET_DESC["full"],
            "eval_num_tasks": metrics_off.get("num_examples") or metrics_train.get("num_tasks"),
            "metrics_file_path": _rel(RUNS / run_label / "metrics_official.json")
            if (RUNS / run_label / "metrics_official.json").is_file()
            else _rel(out_dir / "metrics_official.json")
            if (out_dir / "metrics_official.json").is_file()
            else _rel(out_dir / "metrics.json"),
            "trajectories_file_path": _rel(traj_p) if traj_p.is_file() else "",
            "predictions_file_path": _rel(pred_p) if pred_p.is_file() else "",
            "strict_gold_trace_pass": metrics_train.get("strict_gold_trace_pass"),
            "final_answer_pass": metrics_train.get("final_answer_pass"),
            "zero_tool_calls": metrics_train.get("zero_tool_calls"),
            "clipped_rate": metrics_train.get("clipped_completion_rate"),
            "paradigm": paradigm,
            "notes": "Full-benchmark eval (1861 tasks). W&B run name = output subdir when launched via RUN_FINAL_EVAL=1.",
        }
        row.update(_off_cols(metrics_off, prefix="full"))
        rows.append(row)
    return rows


def _v2_rows() -> list[dict]:
    if not V2.is_dir():
        return []
    specs = [
        ("v2_baseline", "baseline", None, None, None, 67.18),
        ("v2_stage3_epoch1", "curriculum s3e1", "3", "1",
         "curriculum_stage_3_epoch1_multiturn_predictions.jsonl", 70.74),
        ("v2_stage5_epoch2", "curriculum s5e2", "5", "2",
         "curriculum_stage_5_epoch2_multiturn_predictions.jsonl", 70.14),
    ]
    rows = []
    for label, model, stg, ep, fname, exec_acc in specs:
        row = {
            "source": "legacy curriculum v2 (results_v2_20260617)",
            "run_family": "curriculum-v2",
            "wandb_name_or_log_name": label,
            "display_name": f"{model} / react 4-rollout / full NESTFUL",
            "model_checkpoint_stage": stg or "",
            "model_checkpoint_epoch": ep or "",
            "model_checkpoint_path": "",
            "eval_subset_stage": "full",
            "eval_subset_description": EVAL_SUBSET_DESC["full"],
            "eval_num_tasks": 1861,
            "metrics_file_path": _rel(_HERE / "runs" / label / "metrics_official.json"),
            "trajectories_file_path": "",
            "predictions_file_path": "",
            "strict_gold_trace_pass": "",
            "final_answer_pass": "",
            "zero_tool_calls": "",
            "clipped_rate": "",
            "paradigm": "react(4-rollout)",
            "notes": f"Legacy run; executor_accuracy={exec_acc}%. Uses rollout idx0 only.",
        }
        if fname:
            pred_p = V2 / fname
            row["predictions_file_path"] = _rel(pred_p)
            if pred_p.is_file():
                from nestful_official_score import build_item
                from build_report import RAW
                preds = _load_v2_preds(pred_p)
                items = [build_item(preds[tid], RAW[tid]) for tid in preds if tid in RAW]
                from nestful_official_score import score_items
                from build_report import _FUNC_DIR, _WANT_WIN
                scored = score_items(items, executable_func_dir=_FUNC_DIR, win_rate=_WANT_WIN) if items else {}
                row.update(_off_cols(scored, prefix="full"))
                mfile = _HERE / "runs" / label / "metrics_official.json"
                if mfile.is_file():
                    row["metrics_file_path"] = _rel(mfile)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def write_md(path: Path, rows: list[dict]):
    lines = [
        "# Checkpoint → Eval → Metrics Map (NESTFUL MT-GRPO)",
        "",
        "Audit report generated from on-disk artifacts. **No code or scoring changes.**",
        "",
        "## Root cause of the `0.48898` vs `stage3-e3` confusion",
        "",
        "W&B run names for **in-curriculum** `rollout_eval` follow this pattern "
        "(see `run_curriculum.sh` line 373):",
        "",
        "```text",
        "WANDB_RUN_NAME = eval-stage{EVAL_STAGE}-e{EPOCH}",
        "where EVAL_STAGE = train_stage + 1",
        "```",
        "",
        "So **`eval-stage3-e3` means**:",
        "- **Eval subset**: stage 3 → NESTFUL tasks with `num_calls == 3` (407 tasks in this run)",
        "- **Model checkpoint**: **stage 2, epoch 3** (NOT stage 3 epoch 3)",
        "- **`final_answer_pass ≈ 0.488943`** belongs to **train-s2-e3**, as in `consolidated_metrics.json`",
        "",
        "The **actual stage 3 epoch 3** model is logged as **`eval-stage4-e3`** "
        "(evaluated on 4-call subset, 250 tasks) with **`final_answer_pass = 0.168**`.",
        "",
        "| What you might read | W&B name | Actual model ckpt | Eval subset | final_answer_pass |",
        "|---|---|---|---|---|",
        "| \"stage3 epoch3\" (ambiguous) | `eval-stage3-e3` | **s2e3** | 3-call (407) | **0.488943** |",
        "| \"stage3 epoch3\" (correct) | `eval-stage4-e3` | **s3e3** | 4-call (250) | **0.168** |",
        "",
        "## Naming conventions",
        "",
        "| Pattern | Meaning | Set in |",
        "|---|---|---|",
        "| `train-stage{N}-e{E}` | GRPO training, stage N epoch E | `run_curriculum.sh:328` |",
        "| `eval-stage{S}-e{E}` | Rollout eval on **subset S** after training stage (N=S−1) epoch E | `run_curriculum.sh:373` |",
        "| `final_eval_*` | Full NESTFUL benchmark (1861 tasks), Direct or ReAct | `run_curriculum.sh:598` |",
        "",
        "## Eval subset sizes (this run)",
        "",
        "| eval_subset_stage | Filter | Tasks in metrics.json |",
        "|---|---|---|",
        "| 2 | num_calls==2 | 609 |",
        "| 3 | num_calls==3 | 407 |",
        "| 4 | num_calls==4 | 250 (stage 3 epochs); 32 (stage 4 epoch 1 pilot cap) |",
        "| 5 | num_calls==5 | 173 (stage 4 epoch 2) |",
        "| full | all call counts | 1861 |",
        "",
        "## Metric types",
        "",
        "- **Training-time** (`strict_gold_trace_pass`, `final_answer_pass`, …): from `mode_rollout_eval`, "
        "stored in `epoch_*/eval/metrics.json`, logged to W&B as `eval/*`.",
        "- **Subset official** (`off_*`): official NESTFUL scorer re-run on the same trajectories, "
        "same filtered subset (not full 1861).",
        "- **Full official** (`official_*_full`): from `final_eval` on full benchmark, or re-scored predictions.",
        "",
        "## Full mapping table",
        "",
    ]
    # markdown table - subset of key cols for readability
    md_cols = [
        ("wandb_name_or_log_name", "W&B / log name"),
        ("model_checkpoint_stage", "Model S"),
        ("model_checkpoint_epoch", "Model E"),
        ("eval_subset_stage", "Eval subset"),
        ("eval_num_tasks", "N tasks"),
        ("final_answer_pass", "final_answer"),
        ("strict_gold_trace_pass", "strict_pass"),
        ("off_f1_func", "off_f1"),
        ("official_f1_func_full", "full_f1"),
        ("paradigm", "paradigm"),
    ]
    lines.append("| " + " | ".join(h for _, h in md_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(md_cols)) + "|")
    for r in rows:
        vals = []
        for k, _ in md_cols:
            v = r.get(k, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}" if v != int(v) else str(v))
            elif v is None or v == "":
                vals.append("—")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines.extend([
        "",
        "## Per-row detail",
        "",
    ])
    for r in rows:
        lines.append(f"### `{r.get('display_name', '?')}`")
        lines.append("")
        lines.append(f"- **Source**: {r.get('source')}")
        lines.append(f"- **W&B name**: `{r.get('wandb_name_or_log_name')}`")
        lines.append(f"- **Checkpoint**: `{r.get('model_checkpoint_path')}`")
        lines.append(f"- **Metrics**: `{r.get('metrics_file_path')}`")
        if r.get("trajectories_file_path"):
            lines.append(f"- **Trajectories**: `{r.get('trajectories_file_path')}`")
        if r.get("predictions_file_path"):
            lines.append(f"- **Predictions**: `{r.get('predictions_file_path')}`")
        if r.get("notes"):
            lines.append(f"- **Notes**: {r.get('notes')}")
        lines.append("")

    lines.extend([
        "## Gaps and duplicate paths",
        "",
        "- **Stage 2 epoch 2** has no `epoch_2/eval/metrics.json` — training resumed at epoch 3 "
        "(see `epoch_summary.jsonl`: jumps 1 → 3 → 4). No W&B run `eval-stage3-e2` exists for this curriculum.",
        "- **Legacy duplicate**: `outputs/curriculum/stage_4/eval/epoch_1/` (173 tasks) is NOT written by "
        "current `run_curriculum.sh` (which uses `stage_4/epoch_1/eval/`). Treat as stale/orphan unless proven otherwise.",
        "- **`consolidated_metrics.json` / `curriculum_training.csv`** index rows by **model** `(stage, epoch)` "
        "from disk path — this is the correct key for comparing checkpoints.",
        "- **W&B graphs** index in-curriculum evals by **eval subset** in the run name — easy to misread as model stage.",
        "",
        "## Files audited",
        "",
        "- `run_curriculum.sh` — W&B naming, eval subset = train_stage+1",
        "- `run.py` — `mode_rollout_eval`, `mode_final_eval`, W&B logging",
        "- `build_report.py` — `build_curriculum_table()` keys rows by **model** stage/epoch from disk path",
        "- `outputs/curriculum/stage_*/epoch_*/eval/metrics.json`",
        "- `final_outputs/consolidated_metrics.json`",
        "",
        "Regenerate: `python final_outputs/_audit_build_map.py`",
        "",
        "Machine-readable full table: [`checkpoint_eval_map.csv`](checkpoint_eval_map.csv)",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows = _curriculum_rows() + _final_eval_rows() + _v2_rows()
    csv_path = _HERE / "checkpoint_eval_map.csv"
    md_path = _HERE / "CHECKPOINT_EVAL_MAP.md"
    write_csv(csv_path, rows)
    write_md(md_path, rows)
    print(f"Wrote {len(rows)} rows -> {csv_path.name}, {md_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
