"""Quick schema peek at raw artifacts (read-only)."""
import io
import json
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[3]
R1 = V3 / "outputs" / "runs" / "_local_round1_analysis"


def first_row(p, n=1):
    rows = []
    with io.open(p, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= n:
                    break
    return rows


def main():
    out = {}
    c0_dir = R1 / "shared_C0_eval_500" / "shared_C0_eval_500" / "eval" / "C0" / "20260724"
    traj = first_row(c0_dir / "final_eval_trajectories.jsonl")[0]
    out["traj_keys"] = sorted(traj.keys())
    turns = traj.get("turns") or []
    out["n_turns"] = len(turns)
    if turns:
        out["turn0_keys"] = sorted(turns[0].keys()) if isinstance(turns[0], dict) else str(type(turns[0]))
        out["turn0_sample"] = json.dumps(turns[0], ensure_ascii=False)[:900]
    # some eval dirs may have task_results.jsonl in arm dirs only
    a0_dir = (R1 / "reward_ablation_r1_A0_R0_CURRENT_seed20260724" /
              "reward_ablation_r1_A0_R0_CURRENT_seed20260724" / "eval" / "A0_R0_CURRENT" / "20260724")
    for name in ("task_results.jsonl",):
        p = a0_dir / name
        if p.is_file():
            r = first_row(p)[0]
            out[f"a0_{name}_keys"] = sorted(r.keys())
            out[f"a0_{name}_sample"] = json.dumps(r, ensure_ascii=False)[:900]
    p = c0_dir / "task_results.jsonl"
    out["c0_has_task_results"] = p.is_file()
    if p.is_file():
        r = first_row(p)[0]
        out["c0_task_results_keys"] = sorted(r.keys())
    # predictions partial
    p = c0_dir / "final_eval_predictions.partial.jsonl"
    if p.is_file():
        r = first_row(p)[0]
        out["c0_pred_keys"] = sorted(r.keys())
        out["c0_pred_sample"] = json.dumps(r, ensure_ascii=False)[:600]
    # metrics
    for lbl, d in (("c0", c0_dir), ("a0", a0_dir)):
        mp = d / "metrics_official.json"
        if mp.is_file():
            out[f"{lbl}_metrics_official"] = json.loads(mp.read_text(encoding="utf-8"))
    # train subset row
    ts = V3 / "reports" / "reward_ablation" / "data" / "train_subset_160.jsonl"
    if ts.is_file():
        r = first_row(ts)[0]
        out["train_subset_keys"] = sorted(r.keys())
        out["train_subset_sample"] = json.dumps(r, ensure_ascii=False)[:900]
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
