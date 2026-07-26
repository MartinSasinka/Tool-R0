# DECISIONS — Targeted Tool Data Factory

Numbered, binding decisions with rationale. Anything not listed follows
DESIGN.md.

## D01 — New standalone module, minimal deps
`experiments/targeted_tool_data_factory/` with `pip install -e .`.
Core deps: pydantic, PyYAML, numpy, jsonschema (all verified installable
locally). Extras: `[analysis]` scipy/scikit-learn/rapidfuzz,
`[local-llm]` transformers/torch, `[dev]` pytest/ruff. No LangChain — the
pipeline is a deterministic batch DAG, an agent framework adds nothing.

## D02 — Do NOT reuse `lib/synthetic_tools.py`
The v3 registry is one of the audited root causes of the transfer gap
(verbose renames of arithmetic, no numeric-strings, no surface-variant
control, no distractor machinery). We build a typed semantic-primitive
registry with explicit G/A surface morphology. The *reference syntax*,
*tool-schema shape* and *train-ready row format* of the existing trainer are
reused exactly so exports stay drop-in compatible.

## D03 — NESTFUL dev (n=200) is the only profiling source
`nestful_test.jsonl` (1661) is loaded exclusively as a contamination
blocklist (hashes/similarity index), never for quotas or seeds. The
diagnostic-500 is never read by this module.

## D04 — Aggregate-only profiling
TargetProfile stores distributions and morphology statistics; it never
stores target queries or gold programs. Enforced by schema (profile JSON has
no raw-text fields except tool-name morphology token counts).

## D05 — Exact NESTFUL core-math names allowed in A-track only
`divide`, `multiply`, `add`, `subtract`, `power`, `inverse`, `sqrt`,
`negate`, `floor` are NESTFUL's core vocabulary (85 % of dev gold calls).
Domain adaptation without them is fiction. Guards: A-track gold *programs*
(name sequences + arg skeletons) must not match any dev/test gold program;
offered-tool sets are sampled fresh per task; G-track bans all target names.

## D06 — Failure-driven cells from the forensic audit (with one correction)
StudentFailureProfile is seeded from measured Qwen3-4B (C0) behavior on the
NESTFUL diagnostic set (forensic a05, 2026-07-25): too_few_calls dominant
(103–119/500), no_tool_call 41–55, executable_wrong_final 16–23, 2-call
bucket weakest (45 % vs 62 %), undercalling 67 %. Cells therefore
oversample: 2-call tasks, continuation-after-observation, catalog search
with hard distractors, premature-stop pressure ("then…finally" phrasing).

**Correction found during this project's discovery (2026-07-25):** the a06
claim "NESTFUL args are 15–24 % references + str-encoded numbers" was a
measurement bug — the test split writes labels/references as `$var_1`
(underscore), which a06's `$var\d` detector missed and counted as plain
strings. Re-measured: **100 % of dev AND test rows contain references**
(dev per-arg: int 909 / reference 671 / list 20; test: int 7802 /
reference 5812 / list 160). Numeric-string pressure survives only via
string-typed tool params and string gold answers (18/200 dev answers are
strings), so numeric-string cells get a small, profile-derived quota, not
the a06-inflated one.

## D07 — 2-call oversampling bound to evidence
Train quota 35–40 % for 2-call (dev share 33 %) — justified by the measured
2-call weakness of the exact student checkpoint, not hardcoded taste.

## D08 — Reference/direct quotas from profile (per-argument, not per-row)
Row-level reference presence is ~100 % in NESTFUL (see D06 correction) —
identical to Stage-3, so row-level ref rate is NOT the transfer gap. The
real, profile-measured knobs are: per-argument reference share (dev ≈ 42 %),
direct-constant share, and the mix of chain vs fan-in vs branch motifs.
Quotas come from the measured dev profile at runtime; the generator controls
per-arg ref density per cell, never forcing all-ref or all-direct.

## D09 — Deterministic template realization first, LLM optional
≥ 6 template families per motif; template share hard-capped at 5 %.
Providers: `template_only` (default), `openai_compatible_local`,
`transformers_local`. LLM may only paraphrase/score; V3 re-validates; cache
keyed by content hash; `--no-llm` forces template_only.

## D10 — Minimal-path search is bounded and value-based
Search over offered tools with candidate argument pool = question constants
∪ produced intermediates, depth ≤ min(declared−1, 3), breadth capped;
success = terminal value equal (tol 1e-6 / string-exact) to oracle answer.
Deterministic, pure CPU. Declared-N tasks solvable in < N calls are rejected
or relabeled multi-path (then excluded from exact-path scoring downstream).

## D11 — Splits by group keys, not rows
Disjointness on: semantic_program_family, graph_template_id,
tool_combination_hash, paraphrase_family, argument_skeleton_hash,
value_seed. Greedy family-level assignment 160/80/80; leakage audit stores
all group IDs and fails hard on any cross-split collision.

## D12 — Selection = hard gates → deficit matching → coverage; no opaque score
Gates (validity, contamination, dedup) are binary. Remaining candidates fill
generation-cell quotas via greedy deficit matching with novelty tie-breaks
(program family counts, template counts, tool-combination counts). Every
selection decision appends to a JSONL decision trace.

## D13 — Student probe is optional and never oracle
Cascade P0 (structural, free) → P1 (1 greedy rollout, limited pool) →
P2 (4 rollouts) → P3 (8 rollouts, borderline/final only). Local CUDA or an
openai-compatible local server required; otherwise probe = NOT_RUN_LOCAL and
the exact command is emitted. Probe results only add selection metadata.

## D14 — Versioning & hygiene
Every artifact records: config hash, generator version+hash (source file
SHA), profile hash, registry hash, dataset SHA256. Any config/generator
change ⇒ new `--version` output directory. Overwrite requires `--overwrite`.

## D15 — Thresholds are engineering knobs
JSD 0.10 / AUC 0.75 / template 5 % / cell 10 % / hard-distractor ≥ 50 %
live in `configs/base.yaml`, reported explicitly, changeable with
justification — not silently.

## D16 — What this task does NOT do
No GRPO training, no RunPod run, no paid API, no full-test tuning, no
LLM-judge oracle, no final large dataset before the pilot is validated.
