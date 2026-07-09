import json, os, glob
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
RUNS = os.path.join(REPO, "experiments", "nestful_synthetic_curriculum_v3", "outputs", "runs")

def jload(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None

for rid in ("20260707_103035_v3_1", "20260707_152750_v3_1", "20260707_183801_v3_1",
            "20260708_212347_v3_1"):
    print("=====", rid)
    cfgs = glob.glob(os.path.join(RUNS, rid, "stage_*", "checkpoints", "*", "config_used.json"))
    if cfgs:
        c = jload(cfgs[0])
        data = c.get("data", {})
        roll = c.get("rollout", {})
        print(" data:", {k: data.get(k) for k in ("train_stage", "eval_stage", "replay_ratio", "replay_weights", "mixed_replay")})
        print(" rollout keys:", roll if roll else "(none)")
        print(" teacher_forced:", {k: v for k, v in c.items() if "teacher" in str(k).lower()})
        for sect, val in c.items():
            if isinstance(val, dict):
                tf = {k: v for k, v in val.items() if "teacher" in k.lower() or "replay" in k.lower() or "forced" in k.lower()}
                if tf:
                    print(f"  [{sect}] {tf}")
    # first 25 lines of train.log for env echoes
    logs = sorted(glob.glob(os.path.join(RUNS, rid, "stage_*", "epoch_1", "train.log")))
    if logs:
        with open(logs[0], encoding="utf-8", errors="replace") as fh:
            head = [next(fh, "") for _ in range(30)]
        for ln in head:
            ln = ln.strip()
            if any(s in ln.lower() for s in ("replay", "teacher", "forced", "mixed", "train_jsonl", "tasks", "stage")):
                print("  log:", ln[:180])
    # eval dir metrics: what data was rollout-eval run on?
    evs = sorted(glob.glob(os.path.join(RUNS, rid, "stage_*", "epoch_*", "eval", "metrics.json")))
    for ev in evs:
        m = jload(ev) or {}
        rel = os.path.relpath(ev, RUNS)
        print(f"  eval {rel}: num_tasks={m.get('num_tasks')} keys={list(m.keys())[:8]}")
        if "react_win_rate" in m:
            print(f"    react_win_rate={m.get('react_win_rate')} strict={m.get('strict_gold_trace_pass')}")
    print()
