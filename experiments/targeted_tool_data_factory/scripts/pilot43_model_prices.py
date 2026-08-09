"""Price and structured-output support for the critic models under consideration."""
from __future__ import annotations

import json
import urllib.request

CANDIDATES = (
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash",
    "openai/gpt-4o-mini-2024-07-18",
    "openai/gpt-4.1-mini",
    "openai/gpt-5-mini",
    "anthropic/claude-haiku-4.5",
    "deepseek/deepseek-chat-v3.1",
    "qwen/qwen3-235b-a22b-instruct-2507",
    "mistralai/mistral-medium-3.1",
    "x-ai/grok-4-fast",
)


def main() -> int:
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models",
                                timeout=60) as fh:
        models = json.load(fh)["data"]
    by_id = {m["id"]: m for m in models}
    print(f"{'model':<40} {'in $/Mtok':>10} {'out $/Mtok':>11}  structured")
    for slug in CANDIDATES:
        model = by_id.get(slug)
        if model is None:
            print(f"{slug:<40} {'MISSING':>10}")
            continue
        pricing = model.get("pricing") or {}
        params = model.get("supported_parameters") or []
        print(f"{slug:<40} {float(pricing.get('prompt', 0)) * 1e6:>10.3f} "
              f"{float(pricing.get('completion', 0)) * 1e6:>11.3f}  "
              f"{'response_format' in params or 'structured_outputs' in params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
