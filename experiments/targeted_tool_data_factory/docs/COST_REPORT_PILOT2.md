# Pilot2 cost report

Generated 2026-07-26 12:03 UTC.

## Paid services

| item | value |
|---|---|
| provider | OpenRouter |
| model id | `mistralai/mistral-small-24b-instruct-2501` |
| date (UTC) | 2026-07-26T11:50:32Z |
| API calls, final run | 899 (guard: 2000) |
| cached responses on disk (all runs) | 1925 |
| prompt tokens, final run | 318258 |
| completion tokens, final run | 104060 |
| **cost, final run (USD)** | **0.0242** (guard: 2.0) |

Earlier calibration runs against the same cache cost a further $0.024 in total (600 requests at $0.0154, then 600 at $0.0062). Everything is two orders of magnitude below the 2 USD guard.

The budget guard is enforced before each request, not audited afterwards: the client refuses to send request 601, or any request that would push the accumulated cost past the cap.

## Free / local

| item | cost |
|---|---|
| pilot2 generation, validation, selection, split, export | local CPU |
| gold-replay preflight | local CPU |
| local Qwen3-4B probe | `NOT_RUN_LOCAL` — no OpenAI-compatible endpoint answered; see `LOCAL_PROBE_REPORT.md` for the exact PowerShell command. |

## Not yet spent

The RunPod D0/D1 run is the next cost item and is **not** included here. Rough shape at 4 GPUs: canary ~2 GPU-hours, D0 and D1 sequentially, then evaluation across the four GPUs. The full NESTFUL test (1661 tasks) is deliberately disabled by default and is a separate, later decision.
