"""Render only the OpenRouter subset named in render_allocation_llm.jsonl.

Bounds (hard):
  - task ids ⊆ render_allocation_llm.jsonl ∪ llm_queries_reused.jsonl
  - OpenRouter max_total_cost_usd from configs/pilot4_3_openrouter.yaml (36 USD)
  - resumes existing llm_rendered.jsonl; never re-spends on completed keys
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260731)
    args = ap.parse_args()

    from targeted_tool_data.pilot43 import orrun, profile as prof, qstage, resume as res
    from targeted_tool_data.pilot43.orclient import (
        OpenRouterClient, get_api_key, load_openrouter_config)
    from targeted_tool_data.pilot43.pipeline import iter_jsonl

    out = Path(args.out_dir)
    alloc = out / "render_allocation_llm.jsonl"
    if not alloc.exists():
        raise SystemExit(f"missing {alloc}; run allocate-pilot43-render first")
    allowed = {r["task_id"] for r in iter_jsonl(alloc)}
    reused_path = out / res.REUSED_LLM
    if reused_path.exists():
        allowed |= {r["task_id"] for r in iter_jsonl(reused_path)}

    if not get_api_key():
        print(json.dumps({"executed": False, "reason": "no OPENROUTER_API_KEY"}))
        return 2

    cfg = load_openrouter_config()
    print(json.dumps({
        "phase": "allocated_full_render",
        "allowed_task_ids": len(allowed),
        "max_total_cost_usd": cfg.max_total_cost_usd,
        "models": cfg.models,
        "cache_namespace": cfg.cache_namespace,
    }, indent=2), flush=True)

    client = OpenRouterClient(cfg, out)
    # Honor allocation planned_mode so spend matches the render plan (not a
    # second independent mode draw from assign_modes).
    mode_overrides = {
        r["task_id"]: r["planned_mode"]
        for r in iter_jsonl(alloc)
        if r.get("planned_mode") in {
            "DOMAIN_GROUNDED_IMPLICIT", "GOAL_BASED_IMPLICIT", "SEMI_IMPLICIT",
        }
    }
    tasks = qstage.build_render_tasks(
        out, profile=prof.build_profile_v3(), seed=args.seed,
        mode_overrides=mode_overrides)
    tasks = [t for t in tasks if t["task_id"] in allowed]
    print(f"render tasks after allocation filter: {len(tasks)}", flush=True)
    print(f"mode_overrides: {len(mode_overrides)}", flush=True)
    report = orrun.run_stage("full", tasks, client, out, seed=args.seed,
                             workers=max(1, args.workers),
                             progress_every=args.progress_every)
    client.close()
    (out / "stage_gate_pilot43_full_allocated.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in
                      ("stage", "metrics", "passed", "stopped",
                       "resumed", "n_selected")}, indent=2, default=str))
    return 0 if report.get("passed") or report.get("metrics") else 1


if __name__ == "__main__":
    raise SystemExit(main())
