import json, os

def load_json(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except:
        return {}

def load_jsonl(p):
    rows = []
    try:
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if l:
                rows.append(json.loads(l))
    except:
        pass
    return rows

base = "experiments/nestful_mtgrpo_minimal/outputs/curriculum"

print("=== EVAL METRICS ===")
for stage in [1, 2, 3]:
    for epoch in [1, 2, 3, 4]:
        p = os.path.join(base, f"stage_{stage}", f"epoch_{epoch}", "eval", "metrics.json")
        if os.path.exists(p):
            m = load_json(p)
            strict = m.get("strict_gold_trace_pass", m.get("strict_gold_trace", -1))
            fa = m.get("final_answer_pass", -1)
            zt = m.get("zero_tool_calls", -1)
            cr = m.get("clipped_completion_rate", -1)
            n  = m.get("num_tasks", "?")
            es = m.get("eval_stage", "?")
            print(f"  S{stage}e{epoch} (eval_stage={es}, n={n}): "
                  f"strict={strict:.4f}  final={fa:.4f}  zero_tool={zt:.4f}  clipped={cr:.4f}")
        else:
            p2 = os.path.join(base, f"stage_{stage}", f"epoch_{epoch}", "eval")
            if os.path.exists(p2):
                print(f"  S{stage}e{epoch}: metrics.json missing (eval dir exists)")

print()
print("=== TRAIN SUMMARIES ===")
for stage in [1, 2, 3]:
    for epoch in [1, 2, 3, 4]:
        p = os.path.join(base, f"stage_{stage}", f"epoch_{epoch}", "train_summary.json")
        if os.path.exists(p):
            m = load_json(p)
            print(f"  S{stage}e{epoch}: steps={m.get('steps')}  "
                  f"fallback={m.get('fallback_used')}  vllm={m.get('vllm_rollout')}  "
                  f"mt_mode={m.get('mt_grpo_mode')}")

print()
print("=== CURRICULUM SUMMARY ===")
rows = load_jsonl(os.path.join(base, "curriculum_summary.jsonl"))
for r in rows:
    best = r.get("best_strict_gold_trace_pass", 0)
    print(f"  stage={r.get('stage')}  best_strict={best:.4f}  "
          f"advance={r.get('advance_reason')}  gate={r.get('gate_pass')}")

print()
print("=== EPOCH SUMMARIES PER STAGE ===")
for stage in [1, 2, 3]:
    p = os.path.join(base, f"stage_{stage}", "epoch_summary.jsonl")
    if os.path.exists(p):
        rows = load_jsonl(p)
        for r in rows:
            strict = r.get("strict_gold_trace_pass", 0)
            print(f"  S{stage}e{r.get('epoch')}: strict={strict:.4f}  fallback={r.get('fallback_used')}")

print()
print("=== TRAIN LOG STATS (dead groups per stage) ===")
for stage in [1, 2, 3]:
    for epoch in [1, 2, 3, 4]:
        p = os.path.join(base, f"stage_{stage}", f"epoch_{epoch}", "train_log.jsonl")
        if not os.path.exists(p):
            continue
        rows = load_jsonl(p)
        tasks = [r for r in rows if "dead_group" in r]
        if not tasks:
            continue
        dead = sum(1 for r in tasks if r.get("dead_group"))
        all_one = sum(1 for r in tasks if r.get("group_all_one"))
        all_zero = sum(1 for r in tasks if r.get("group_all_zero") and not r.get("dead_group"))
        mixed = sum(1 for r in tasks if r.get("group_mixed"))
        mean_r = sum(r.get("mean_reward", 0) for r in tasks) / max(1, len(tasks))
        print(f"  S{stage}e{epoch}: {len(tasks)} tasks | dead={dead}({dead*100//max(1,len(tasks))}%)  "
              f"all_one={all_one}  all_zero={all_zero}  mixed={mixed}  avg_reward={mean_r:.3f}")
