# DESIGN — Targeted Tool Data Factory

Target-conditioned, program-first, executor-verified, failure-driven,
student-in-the-loop, transfer-validated data generation for tool-use models.

First student: **Qwen/Qwen3-4B-Instruct-2507** (exact checkpoint used by the
GRPO trainer, `experiments/nestful_mtgrpo_partial/config.yaml`).
First target: **NESTFUL** (arXiv:2409.03797).

---

## 1. Why program-first

Every prior failure in this repo traces back to surface-first generation:
the v3 synthetic curriculum was internally valid (100 % replay) yet
structurally alien to NESTFUL (forensic audit 2026-07-25: gold-tool Jaccard
0.003–0.006, reference args 100 % vs 15–24 %, fixed 3-call vs 2–6+, offered
tools 163 vs 816). LLM-first pipelines (query → trajectory) cannot guarantee
oracle correctness without an LLM judge, which this project forbids.

Order of generation is therefore fixed:

```
semantic program (typed DAG)
→ executable graph → oracle execution → oracle observations
→ oracle final answer → tool schemas → offered-tool set
→ hard distractors → user query (templates; optional local LLM paraphrase)
```

The oracle answer exists **before** any natural language exists. An LLM can
only re-word a query; it can never define correctness (APIGen's execution
verification taken to its logical end; ToolACE-style LLM complexity
evolution rejected — see DECISIONS.md).

## 2. Why target-conditioned profiling

"Valid synthetic data" failed to transfer once already (pure_stage3: train
signal ↑, NESTFUL −1.14 pp). Intrinsic validity is a *hard gate*, not an
objective. The objective is matching the **capability-relevant structural
distribution** of the target: call-count distribution, motif mix, argument
typing (incl. numeric-strings), reference/direct ratio, per-task offered-tool
count, distractor hardness, name/description morphology. All quotas are
derived from a machine-extracted `TargetProfile` of NESTFUL **dev (n=200)**
— never hardcoded, never taken from the test split.

## 3. Why intrinsic validity is not enough

A dataset can pass 100 % replay and still teach shortcuts (template echo,
fixed 3-call rhythm, reference-always habit). Hence three extra layers on
top of execution validity:

- **minimal-path & shortcut audit** (V4): declared call count must equal the
  minimal valid path over the *offered* set, or the task is explicitly
  multi-path; single-tool solutions and decorative gold steps are rejected;
- **distribution audit** (V6): no template > 5 %, no generation cell > 10 %
  (unless justified), no dominant tool family, JSD/Wasserstein/classifier-AUC
  match reported vs the target profile *and* vs old Stage-3;
- **downstream data-only experiment** (planned, not run here): D0 (old
  Stage-3 160) vs D1 (new 160) with everything else frozen.

## 4. How benchmark memorization is prevented

- Profiling uses **aggregate statistics only** from NESTFUL dev; no dev/test
  query, gold trajectory or offered-tool set is ever a generative seed.
- Contamination audit (V5) against dev+test: exact query overlap, normalized
  overlap (lowercase/digit-masked), char-3-gram near-duplicate similarity,
  gold tool-call skeleton overlap (name sequence), paraphrase similarity
  (rapidfuzz token_set_ratio). Hard gate: 0 exact overlaps; near-dup
  similarity above threshold ⇒ reject.
- The fixed NESTFUL diagnostic-500 is reserved for downstream comparison of
  *frozen* dataset versions; it is never read by the generator.

## 5. Generalization vs adaptation (two tracks)

- **G-track (40 %)**: independent tool vocabulary (no NESTFUL names),
  same *structure* (schemas, types, motifs, difficulty). Measures
  transferable schema-reading/planning ability.
- **A-track (60 %)**: NESTFUL-like surface morphology — short math-core
  names (`divide`, `multiply`, …) and snake_case descriptive utilities,
  NESTFUL-style flat-dict parameter schemas, `$varN.output_0$` references,
  str-encoded numeric arguments. New programs, values, questions and tool
  combinations; explicitly labeled `track="A"`, `domain_adaptation=true`.
- Held-out results are reported jointly and per track, so generalization
  gains cannot be confused with naming adaptation.

## 6. What is deterministic and what the LLM may do

Deterministic (always): program synthesis, execution, oracle answers,
schema rendering, distractor construction, template query realization,
validation, dedup, selection, splits, exports.

Optional local LLM (`openai_compatible_local` / `transformers_local`,
default **off**): paraphrase an already-valid query, naturalness scoring,
ambiguity flagging. Post-paraphrase the semantic-consistency validator
(V3) re-runs; on failure the template query is kept. All LLM outputs are
cached by content hash. No remote/paid endpoint is ever enabled by default.

## 7. How target-distribution similarity is measured

Per feature versus the NESTFUL dev profile, computed identically for
old Stage-3 (326) and the new pilot:

- **Jensen–Shannon divergence** for categorical features (call-count
  buckets, motif, argument types, reference usage, offered-tool buckets);
- **Wasserstein distance** for numeric features (offered-tool count,
  question length, dependency depth);
- **classifier two-sample AUC** (logistic regression on structural
  features): AUC → 0.5 means indistinguishable from target structure.

Default warnings: JSD > 0.10 on a major feature, AUC > 0.75, any empty
significant target bucket, template > 5 %, cell > 10 %. These are
configurable engineering thresholds, not scientific laws.

## 8. How quality is verified downstream

Final quality criterion is the **data-only training experiment**
(docs/NEXT_TRAINING_EXPERIMENT.md): identical C0 checkpoint, reward
(A4_GATED_VERIFIABLE after the dispatch canary), seed, rollouts, optimizer
steps, LR/KL/LoRA, credit assignment, decoding and eval pipeline; the only
change is the dataset (D0 old Stage-3 vs D1 new). Evaluated on the
80-task structural held-out (jointly + per track) and the frozen NESTFUL
diagnostic-500. This experiment is *planned only* in this task.

## 9. Anchor methods — adopted / rejected

| Source | Adopted | Rejected (and why) |
|---|---|---|
| NESTFUL (2409.03797) | exact data format (`input/output/tools`, `$varN.output_0$` refs, flat-dict parameters), nested dependency semantics, F1/win metrics for downstream | using its test split for tuning (hygiene) |
| APIGen (2406.18518) | three-stage verification: format → execution → semantics (our V1/V2/V3); rejection-sampling into a large candidate pool | LLM-generated queries as the primary source of task semantics (oracle must be executed, not judged) |
| ToolACE (2409.00920) | complexity as an explicit, controlled dimension; tool-catalog diversity goals; self-consistency checks | agentic multi-LLM interplay for generation (needs strong paid LLMs; non-deterministic oracle) |
| NesTools (2410.11805) | nested-call emphasis; distractor-tool inclusion in offered sets; per-task tool-catalog control | LLM-judged solvability filtering |
| AutoData (2606.25996) | staged DAG pipeline with caching, provenance hashes, resume-safety; separation of discovery/generation/validation roles | multi-agent web crawling architecture (irrelevant to executor-verified synthesis; remote APIs) |

## 10. Capability mapping

Every generator decision is tied to at least one target capability
(numbers from §0 of the task):

| Design element | Capabilities |
|---|---|
| per-task offered sets of 8–18 tools sampled to the dev profile | 1, 11 |
| hard distractors (same-signature/near-semantics/similar-name) ≥ 50 % of tasks | 1, 8 |
| schema rendering variants (names, param names, descriptions, optional args) | 2 |
| typed primitives with int/float/numeric-string/list/enum params | 3, 4 |
| real executor observations + reference args where profiled | 5, 7 |
| motif × call-count cells (2–6+, linear/fan-in/branch/aggregation) | 6 |
| numeric-string cells (parse + emit) | 4 |
| path-invariant terminal check + alternative-path labeling | 9 |
| P0–P3 student probing cascade (informative difficulty 1/8–7/8) | 10 |
| G/A track split + structural held-out + NESTFUL-500 downstream plan | 11 |

## 11. Architecture layers

```
semantic core        src/targeted_tool_data/registry, graph, executor
generic generator    src/targeted_tool_data/generate (cells → candidates)
target profile       src/targeted_tool_data/profile (+ outputs/profiles)
target adapter       targets/<name>/ (data loading, morphology, quotas)
target renderer      src/targeted_tool_data/render (+ adapter conventions)
exporter             src/targeted_tool_data/export (canonical/NESTFUL/GRPO/CSV)
```

NESTFUL-specific code lives only in `targets/nestful/` and
`configs/targets/nestful.yaml`. A dummy target adapter in tests proves the
pipeline is not NESTFUL-bound.

## 12. Reuse from the existing repo

Reused (read-only, verified): NESTFUL splits
(`experiments/nestful_mtgrpo_minimal/data/splits/nestful_dev.jsonl` for
profiling; test only for the contamination blocklist), the GRPO train-ready
row schema (mirrored from `stage3_train_ready.jsonl`), the trainer's
reference syntax and tool-schema conventions, the forensic failure profile
(`reports/root_cause_forensic/analysis/a05/a06`). NOT reused: the v3
`synthetic_tools.py` registry (its verbose-arithmetic surface vocabulary and
100 %-reference habit are two of the audited transfer-gap causes; we need
typed multi-surface primitives with G/A morphology control). Executor logic
is re-implemented in ~150 lines because the graph model differs (typed DAG
with oracle observations), while keeping identical reference semantics
(`$varN.output_0$`) so exported rows replay in the existing trainer
preflight.
