import csv, json, os
from collections import defaultdict

OUT = os.path.dirname(__file__)
corr = list(csv.DictReader(open(os.path.join(OUT, "execution_reward_correlation.csv"), encoding="utf-8")))
def fl(x):
    try: return float(x)
    except: return 0.0

# ---- Offline alignment: FP / FN per reward, per run ----
print("=== OFFLINE REWARD ALIGNMENT ===")
by_run = defaultdict(list)
for row in corr:
    by_run[row["run"]].append(row)
print(f"{'run':22s} n    win   FP_exec FN_exec FP_part FP_strict fap_dom")
for run, rows in list(by_run.items()) + [("ALL", corr)]:
    n = len(rows)
    win = sum(fl(r["official_win"]) for r in rows) / n
    # FP: win=0 but reward high (>0.7); FN: win=1 but reward low (<0.3)
    fp_e = sum(1 for r in rows if fl(r["official_win"]) == 0 and fl(r["execution_reward"]) > 0.7) / n
    fn_e = sum(1 for r in rows if fl(r["official_win"]) == 1 and fl(r["execution_reward"]) < 0.3) / n
    fp_p = sum(1 for r in rows if fl(r["official_win"]) == 0 and fl(r["partial_reward"]) > 0.7) / n
    fp_s = sum(1 for r in rows if fl(r["official_win"]) == 0 and fl(r["strict_reward"]) > 0.7) / n
    # how often final_answer_pass alone == win (dominance)
    fap_dom = sum(1 for r in rows if fl(r["final_answer_pass"]) == fl(r["official_win"])) / n
    print(f"{run:22s} {n:5d} {win:.3f}  {fp_e:.3f}   {fn_e:.3f}   {fp_p:.3f}   {fp_s:.3f}     {fap_dom:.3f}")

# ---- Build manual_failure_samples.csv from best checkpoint trajectories ----
RUN = "partial_s1_e4_react"
traj_path = os.path.abspath(os.path.join(OUT, "..", "nestful_mtgrpo_partial", "outputs", "final_eval", "partial_s1_e4_react", "final_eval_trajectories.jsonl"))
ds_path = os.path.abspath(os.path.join(OUT, "..", "nestful_mtgrpo_minimal", "data", "NESTFUL-main", "data_v2", "nestful_data.jsonl"))

# index per-sample rewards for this run
pers = {r["sample_id"]: r for r in corr if r["run"] == RUN}
# load gold answers
gold = {}
for l in open(ds_path, encoding="utf-8"):
    l = l.strip()
    if not l: continue
    row = json.loads(l)
    sid = str(row.get("sample_id") or row.get("task_id"))
    gold[sid] = row

trajs = {}
if os.path.exists(traj_path):
    for l in open(traj_path, encoding="utf-8"):
        l = l.strip()
        if not l: continue
        row = json.loads(l)
        sid = str(row.get("sample_id") or row.get("task_id") or "")
        trajs[sid] = row

def summarize_traj(t):
    tj = t.get("_traj", t)
    turns = tj.get("turns", [])
    calls = []
    for tt in turns:
        c = tt.get("parsed_call")
        if c: calls.append(c.get("name"))
    return {"n_turns": len(turns), "stop": tj.get("stop_reason"), "pred_names": calls,
            "final_obs": tj.get("final_observation", t.get("final_observation"))}

# categories
cats = {
    "exec_high_win0": lambda r: fl(r["official_win"]) == 0 and fl(r["execution_reward"]) > 0.7,
    "win1_exec_low": lambda r: fl(r["official_win"]) == 1 and fl(r["execution_reward"]) < 0.3,
    "strict_fail": lambda r: fl(r["strict_reward"]) == 0,
    "no_tool_call": lambda r: r["execution_cap"] == "no_tool_call",
    "too_few_calls": lambda r: fl(r["num_successful_calls"]) < fl(r["gold_num_calls"]),
}
rows_out = []
for cat, fn in cats.items():
    picked = [r for r in corr if r["run"] == RUN and fn(r)][:20]
    for r in picked:
        sid = r["sample_id"]
        g = gold.get(sid, {})
        tinfo = summarize_traj(trajs.get(sid, {})) if sid in trajs else {}
        rows_out.append({
            "category": cat, "sample_id": sid, "gold_num_calls": r["gold_num_calls"],
            "official_win": r["official_win"], "strict": r["strict_reward"],
            "partial": r["partial_reward"], "execution": r["execution_reward"],
            "final_answer_pass": r["final_answer_pass"], "exec_cap": r["execution_cap"],
            "num_successful_calls": r["num_successful_calls"],
            "pred_names": "|".join(str(x) for x in tinfo.get("pred_names", [])),
            "stop_reason": tinfo.get("stop", ""), "n_turns": tinfo.get("n_turns", ""),
        })
with open(os.path.join(OUT, "manual_failure_samples.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader(); w.writerows(rows_out)
print(f"\nWROTE manual_failure_samples.csv ({len(rows_out)} rows across {len(cats)} categories), run={RUN}, trajs_loaded={len(trajs)}")
