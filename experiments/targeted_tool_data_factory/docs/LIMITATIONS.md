# LIMITATIONS

1. **Template English.** Queries come from 5 deterministic template families
   (with per-family variants, concentration ≤ 5 % per surface template).
   NESTFUL questions are more varied natural language. A local-LLM paraphrase
   pass exists but was not run in the pilot (no local server). This is the
   largest remaining surface-level gap; it is measured (question-length
   Wasserstein, two-sample AUC includes q_len) rather than hidden.
2. **Registry breadth.** 35 typed primitives across arithmetic, comparison,
   selection, string, list and conversion semantics — richer than Stage-3's
   renamed arithmetic, but far narrower than NESTFUL's long tail (tensor
   utilities, dict-processing, domain functions). List/dict-typed programs
   are underrepresented (list args ≈ 1 % in target dev, so quota impact is
   small).
3. **Bounded shortcut search.** V4 proves the absence of *cheap* shortcuts
   (value-reachable within depth ≤ 3, ≤ 20 000 evals, scalar-typed pools);
   it cannot prove absence of all alternative decompositions. Numeric-string
   coercion shortcuts (passing 26181 where "26181" is expected) are not
   counted as shortcuts because they violate the declared schema — same
   convention as NESTFUL itself.
4. **Student probe not run locally.** No CUDA GPU / local server on this
   machine: probe = P0 structural heuristic + `NOT_RUN_LOCAL`; informative
   difficulty (1/8–7/8 band) is unverified until Phase C runs (exact command
   in GENERATION_RUNBOOK.md).
5. **No deterministic-selection motif.** max/min/clamp *sinks* were removed
   from generation after Phase B1 showed 0 % acceptance: a selection output
   always equals one of its inputs, which is irreconcilable with
   path-invariant value-based correctness and the duplicate-observation
   guard. Selection primitives remain in offered sets as hard distractors;
   selection-style comparison tasks would need answer formats richer than a
   single value (e.g. "which branch") and are deferred.
6. **Answer-type gap.** List/bool answers (9 % of target) are only partially
   covered (count/aggregation primitives produce ints; bool producers were
   excluded from chains to avoid degenerate programs).
7. **Trainer executor integration.** GRPO export matches the
   stage3_train_ready row contract, but the trainer's synthetic rollout
   executor must be pointed at this factory's primitives for D1 training
   (integration step described in NEXT_TRAINING_EXPERIMENT.md). Gold-replay
   verification of exported rows is test-covered here, not in the trainer.
8. **Profile from n=200.** Dev-split estimates carry sampling error (~3.5 pp
   on a 33 % bucket at n=200). Quotas inherit that noise; acceptable for a
   320-task pilot.
9. **Intrinsic ≠ transfer.** All gates here are necessary conditions. The
   only sufficient evidence is the pre-registered data-only experiment
   (D0 vs D1) — deliberately not run in this task.
