import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

root = Path(r"c:\Users\Šunka\Downloads\qwen3_p43_profile1000_dynamic_online_continue256")
rows = [json.loads(l) for l in open(root / "train_log.jsonl", encoding="utf-8") if l.strip()]
task = [r for r in rows if "mean_reward" in r]
opt = [r for r in rows if r.get("update") == "optimizer_step"]
summary = json.loads((root / "train_summary.json").read_text(encoding="utf-8"))

print("task_rows", len(task))
print(
    "opt_steps",
    len(opt),
    "range",
    opt[0]["global_step"] if opt else None,
    "->",
    opt[-1]["global_step"] if opt else None,
)


def stats(xs):
    if not xs:
        return None
    mr = [r["mean_reward"] for r in xs]
    ow = [r.get("official_win_rate") or 0 for r in xs]
    uniq = [r.get("n_unique_completion_hashes") or r.get("n_unique_completions") or 0 for r in xs]
    dead = sum(1 for r in xs if r.get("update") == "skipped_dead_group" or r.get("dead_group"))
    fa = [r.get("final_answer_accuracy") or 0 for r in xs]
    fa_pres = [r.get("final_answer_presence_rate") or 0 for r in xs]
    execf = [r.get("execfail_total") or 0 for r in xs]
    kl = [r["kl"] for r in xs if r.get("kl") is not None]
    return {
        "n": len(xs),
        "mean_rew": sum(mr) / len(mr),
        "win": sum(ow) / len(ow),
        "uniq": sum(uniq) / len(uniq),
        "dead_rate": dead / len(xs),
        "fa_acc": sum(fa) / len(fa),
        "fa_pres": sum(fa_pres) / len(fa_pres),
        "execfail": sum(execf) / len(execf),
        "kl": (sum(kl) / len(kl)) if kl else None,
    }


by_ep = defaultdict(list)
for r in task:
    by_ep[r.get("epoch")].append(r)

print("=== per epoch ===")
for ep in sorted(k for k in by_ep if k is not None):
    s = stats(by_ep[ep])
    print(
        f"ep{ep}: n={s['n']} rew={s['mean_rew']:.3f} win={s['win']:.3f} "
        f"uniq={s['uniq']:.2f} dead={s['dead_rate']:.2f} "
        f"fa_acc={s['fa_acc']:.3f} fa_pres={s['fa_pres']:.3f} "
        f"execfail={s['execfail']:.2f} kl={s['kl']}"
    )

for label, xs in [("first50", task[:50]), ("last50", task[-50:]), ("overall", task)]:
    s = stats(xs)
    print(label, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})

print("updates", Counter(r.get("update") for r in task))
gn = [r["grad_norm"] for r in opt if r.get("grad_norm") is not None]
print("grad_norm mean/min/max", sum(gn) / len(gn), min(gn), max(gn), "n", len(gn))

fmt = "%Y-%m-%dT%H:%M:%S"
times = []
for ep in range(1, 9):
    p = root / "checkpoints" / f"adapter_epoch_{ep}" / "trainer_state.json"
    ts = json.loads(p.read_text(encoding="utf-8"))
    times.append((ep, ts["global_step"], datetime.strptime(ts["saved_at"], fmt)))
    print(f"adapter_epoch_{ep}: step={ts['global_step']} saved_at={ts['saved_at']}")

t0, t1 = times[0][2], times[-1][2]
print("wall_h_ep1_to_ep8", (t1 - t0).total_seconds() / 3600)
print("min_per_step", (t1 - t0).total_seconds() / 60 / max(1, times[-1][1] - times[0][1]))
print(
    "summary",
    {
        "global_step_start": summary.get("global_step_start"),
        "global_step_end": summary.get("global_step_end"),
        "dead_group_rate": summary.get("dead_group_rate"),
        "no_tool_call_rate": summary.get("no_tool_call_rate"),
        "eligible_for_best": summary.get("eligible_for_best"),
        "fractional_rewards_present": summary.get("fractional_rewards_present"),
        "n_unique_reward_values": summary.get("n_unique_reward_values"),
        "contributing_turns_total": summary.get("contributing_turns_total"),
        "avg_predicted_calls": summary.get("avg_predicted_calls"),
    },
)
