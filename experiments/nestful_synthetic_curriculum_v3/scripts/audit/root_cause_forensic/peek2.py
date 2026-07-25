import io
import json
from pathlib import Path

V3 = Path(__file__).resolve().parents[3]
R1 = V3 / "outputs" / "runs" / "_local_round1_analysis"

c0_dir = R1 / "shared_C0_eval_500" / "shared_C0_eval_500" / "eval" / "C0" / "20260724"
with io.open(c0_dir / "final_eval_trajectories.jsonl", encoding="utf-8") as fh:
    row = json.loads(fh.readline())
tr = row.get("_traj") or {}
print("_traj type:", type(tr).__name__)
if isinstance(tr, dict):
    print("_traj keys:", sorted(tr.keys()))
    for k in ("turns", "rollouts", "messages"):
        v = tr.get(k)
        if v:
            print(k, "len", len(v))
            print(json.dumps(v[0], ensure_ascii=False)[:1000])
# count lines + wins
n = 0
with io.open(c0_dir / "final_eval_trajectories.jsonl", encoding="utf-8") as fh:
    for line in fh:
        if line.strip():
            n += 1
print("c0 trajectories rows:", n)
n = 0
with io.open(c0_dir / "final_eval_predictions.partial.jsonl", encoding="utf-8") as fh:
    for line in fh:
        if line.strip():
            n += 1
print("c0 predictions rows:", n)

# arm trajectories
a0_dir = (R1 / "reward_ablation_r1_A0_R0_CURRENT_seed20260724" /
          "reward_ablation_r1_A0_R0_CURRENT_seed20260724" / "eval" / "A0_R0_CURRENT" / "20260724")
with io.open(a0_dir / "final_eval_trajectories.jsonl", encoding="utf-8") as fh:
    row = json.loads(fh.readline())
print("a0 traj keys:", sorted(row.keys()))
tr = row.get("_traj") or {}
if isinstance(tr, dict):
    print("a0 _traj keys:", sorted(tr.keys()))

# nestful test size
nt = V3.parent / "nestful_mtgrpo_minimal" / "data" / "splits" / "nestful_test.jsonl"
n = 0
with io.open(nt, encoding="utf-8") as fh:
    for line in fh:
        if line.strip():
            n += 1
print("nestful_test rows:", n)
