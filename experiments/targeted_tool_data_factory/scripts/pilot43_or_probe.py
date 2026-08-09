"""One-task OpenRouter round trip: is the writer/critic plumbing actually live?

Run before a staged render so a broken key, an unpinned slug or a schema
mismatch costs one request instead of fifty.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from targeted_tool_data.pilot43 import orrun, profile as prof, qstage
from targeted_tool_data.pilot43.orclient import (OpenRouterClient, get_api_key,
                                                 load_openrouter_config)

OUT = Path("outputs/pilot4_3_nestful_final")


def main() -> int:
    print("key present:", bool(get_api_key()))
    cfg = load_openrouter_config()
    print("configured models:", json.dumps(cfg.models))
    print("prompt versions:", json.dumps(cfg.prompt_versions))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    tasks = qstage.build_render_tasks(OUT, profile=prof.build_profile_v3(),
                                     limit=limit)
    print("render tasks built:", len(tasks),
          "modes:", sorted({t["requested_mode"] for t in tasks}))
    if not tasks:
        return 1
    client = OpenRouterClient(cfg, OUT)
    print("client available:", client.available())
    task = orrun.RenderTask.from_dict(tasks[0])
    record = orrun.render_one(task, client, sample_rate=1.0)
    print(json.dumps({k: record.get(k) for k in
                      ("task_id", "requested_mode", "call_count", "answer_type",
                       "structured_output_ok", "query", "model",
                       "prompt_version", "blocked", "blocked_reason", "error")},
                     indent=1, ensure_ascii=False))
    print("first critic:", (record.get("critic") or {}).get("verdict"),
          "| second critic:", (record.get("second_critic") or {}).get("verdict"),
          "| routed because:", record.get("second_critic_reason"))
    print("deterministic validation:", record["validation"].get("passed"),
          record["validation"].get("failed_layers"))
    print("usage:", json.dumps(client.totals.as_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
