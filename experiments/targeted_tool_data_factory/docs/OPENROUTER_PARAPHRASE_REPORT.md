# OpenRouter paraphrase report

Generated 2026-07-26 12:03 UTC.

## Configuration

| field | value |
|---|---|
| model id | `mistralai/mistral-small-24b-instruct-2501` |
| endpoint | `https://openrouter.ai/api/v1` |
| run date (UTC) | 2026-07-26T11:50:32Z |
| budget guard | 2000 requests / 2.0 USD |
| key fingerprint | `sha256:65a944ccf175` (not the key) |
| cache | content-hash keyed, resume-safe |

The model id is pinned in the config. `openrouter/auto` is never used: a routing alias would make the surface distribution unreproducible. The model is deliberately non-Qwen so the paraphrases do not inherit the student's own phrasing distribution.

## Usage

| metric | value |
|---|---|
| shortlisted tasks | 1800 |
| API calls this run | 899 |
| cache hits this run | 901 |
| retries / errors | 0 / 0 |
| prompt tokens | 318258 |
| completion tokens | 104060 |
| **cost this run (USD)** | **0.0242** |
| paraphrases proposed | 3604 |
| paraphrases accepted | 755 |
| tasks kept on the deterministic template | 1045 |
| reverted at re-validation | 0 |
| dropped by dedup / contamination | 0 |
| paraphrased records in the validated pool | 755 / 2175 |

## Why paraphrases were rejected

The validator has to prove the program is unchanged. Anything it cannot prove is discarded and the deterministic template survives, so a high rejection rate costs surface diversity but can never cost correctness.

| check | n |
|---|---|
| `numeric tokens changed` | 2063 |
| `dependency references dropped` | 432 |
| `operation 2 (ratio_of) missing or reordered` | 35 |
| `operation 1 (floor_divide) missing or reordered` | 31 |
| `V3` | 27 |
| `too long` | 26 |
| `operation 2 (floor_divide) missing or reordered` | 25 |
| `operation 1 (inverse) missing or reordered` | 20 |
| `operation 3 (ratio_of) missing or reordered` | 18 |
| `operation 3 (floor_divide) missing or reordered` | 16 |
| `operation 4 (ratio_of) missing or reordered` | 13 |
| `operation 1 (ratio_of) missing or reordered` | 11 |
| `operation 2 (average_two) missing or reordered` | 10 |
| `operation 1 (count_values) missing or reordered` | 9 |
| `operation 1 (modulo) missing or reordered` | 9 |

## Safety properties

- The API key is read from the repo-root `.env` at call time and is never logged, never written into an artefact and never committed.
- Only an already-validated synthetic question is sent. Raw NESTFUL text never leaves the machine.
- A paraphrase can only replace the question string. The program, tools, arguments, constants, dependency order and oracle answer are untouched by construction — they are not part of the request.
- Every returned paraphrase is re-validated deterministically (constants preserved as an exact multiset, operation keywords present and in order, dependency markers intact, no oracle or intermediate leak, contamination and dedup re-run). A paraphrase that fails any check is discarded and the deterministic template is kept.
