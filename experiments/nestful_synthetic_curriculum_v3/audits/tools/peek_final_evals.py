import json, os, glob
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
RUNS = os.path.join(REPO, "experiments", "nestful_synthetic_curriculum_v3", "outputs", "runs")

def show(p):
    d = json.load(open(p, encoding="utf-8"))
    internal = d.get("internal_metrics_diagnostic", {})
    ours = d.get("our_metrics", {})
    print(os.path.relpath(p, RUNS))
    keys = [k for k in d if not isinstance(d[k], dict)]
    print("  top-level:", {k: d[k] for k in keys})
    if internal:
        print("  internal:", {k: round(v, 4) if isinstance(v, float) else v for k, v in internal.items()})
    if ours:
        print("  ours:", {k: round(v, 4) if isinstance(v, float) else v for k, v in ours.items()})
    for extra in ("official", "win_rate", "react_win_rate"):
        if extra in d:
            print(f"  {extra}:", d[extra])
    print()

for pat in ("final_eval_all_runs_20260707_215620/*/metrics.json",
            "final_eval_all_runs_20260708_164607_temp0/*/*/metrics.json",
            "final_eval_stage3_e1e2_20260709_093453_temp0/*/*/metrics*.json"):
    for p in sorted(glob.glob(os.path.join(RUNS, pat))):
        show(p)
