# Pilot3 data card — `pilot3`

Generated 2026-07-27 03:15 UTC.

## What this dataset is

1000 synthetic nested-tool-use tasks produced program-first: a semantic program
is sampled first, executed by the factory executor to obtain the oracle answer,
and only then rendered into a natural-language question. Pilot2 artefacts are
untouched.

| field | value |
|---|---|
| version | `pilot3` |
| seed | `20260727` |
| selected tasks | 1000 |
| train / structural held-out / reserve | 600 / 200 / 200 |
| adaptation / generalization | 55.2 % / 44.8 % |
| target benchmark profile | NESTFUL dev |
| target student | `Qwen/Qwen3-4B-Instruct-2507` |
| oracle | executor-only, deterministic |
| replay rate (validated pool) | 100.0 % |
| verdict | **NOT_READY** |

## Composition

### call count
| value | share |
|---|---|
| `2` | 31.4 % |
| `3` | 17.5 % |
| `4` | 13.4 % |
| `5` | 12.1 % |
| `6` | 13.3 % |
| `7` | 5.5 % |
| `8` | 6.8 % |

### motif
| value | share |
|---|---|
| `branch_aggregate` | 7.6 % |
| `fan_in` | 41.6 % |
| `linear` | 50.8 % |

### answer type
| value | share |
|---|---|
| `bool` | 4.2 % |
| `float` | 71.6 % |
| `int` | 6.2 % |
| `list` | 6.8 % |
| `numeric_string` | 3.5 % |
| `string` | 7.7 % |

- max generation-cell share: 4.30 % (cap 8 %)
- max graph-template share: 8.30 % (cap 5 %)
- candidates generated: 5671
- paraphrase accepted: 1793
  cost=$0.076372

## Quality gates

| gate | status |
|---|---|
| gold replay 100 % | see preflight |
| leakage 0 | PASS |
| cell share ≤ 8 % | PASS |
| template share ≤ 5 % | FAIL |

## Intended use

GRPO training on the frozen train split after a RunPod signal probe selects a
NESTFUL-matched Phase-1 subset with real terminal/process mixed signal.
Structural held-out is for in-domain measurement; reserve stays sealed.

## Out of scope

- Not a public benchmark.
- Do not regenerate on RunPod — use `runpod_bundle_pilot3/`.
