# BOTTLENECK_RANKING

## 1. EVAL_PROTOCOL

- status: `SUPPORTED`
- evidence: `HIGH`
- for: Original +8.8pp used mismatched HF vs vLLM backends; matched contrast is smaller.
- against: Matched C0/D1 still share residual LoRA/vLLM path differences.
- next: Keep matched vLLM eval as the only training-effect contrast; freeze eval script hash.

## 2. TRAIN_SUBSET_SELECTION

- status: `SUPPORTED`
- evidence: `HIGH`
- for: D1 train_subset_300 is not identical to the local train_grpo_pilot3 prefix/freeze; selection provenance is broken or the export was regenerated.
- against: Even a balanced 300 may still yield small transfer.
- next: Freeze SHA256 of exact D1 subset; nested stratified subset by cell×call_bucket×motif×track; never silent file-prefix slices.

## 3. SURFACE_SCHEMA_MISMATCH

- status: `PARTIALLY_SUPPORTED`
- evidence: `HIGH`
- for: Diagnostic gold tools largely outside factory exact namespace; transfer relies on schema/lexical proxies.
- against: Factory intentionally uses synthetic surfaces; some transfer still observed.
- next: Paired A-native/G-general renderers; NESTFUL-like output keys; harder schema-compatible distractors.

## 4. REWARD_SIGNAL

- status: `PARTIALLY_SUPPORTED`
- evidence: `MEDIUM`
- for: Aggregate dead_group_rate=0.51 and low unique rewards imply many non-informative GRPO groups.
- against: Without per-rollout groups cannot separate all-success vs all-fail cells.
- next: Add difficulty-targeted cells + persist per-rollout rewards; balance effective-group rate.

## 5. LORA_INFERENCE_PATH

- status: `NOT_IDENTIFIABLE`
- evidence: `LOW`
- for: D1 uses adapter; C0 is base. Path differences can shift decoding even at T=0.
- against: Cannot quantify from stored trajectories alone.
- next: Future A/B: base+null-LoRA vs D1 under identical vLLM loader.

## 6. TOPOLOGY_DIVERSITY

- status: `PARTIALLY_SUPPORTED`
- evidence: `MEDIUM`
- for: High top-1 topology share and/or high diagnostic unseen-topology rate indicate shape mismatch risk.
- against: Topology coverage may not associate with gained/lost if surface mismatch dominates.
- next: Cap top-1 topology family share; raise unique topology quota; joint topology×call_count matching.

## 7. REGISTRY_SEMANTIC_COVERAGE

- status: `PARTIALLY_SUPPORTED`
- evidence: `MEDIUM`
- for: Unmapped diagnostic gold tools on critical path associate with persistent failures (proxy).
- against: Registry size alone does not explain gap; exact IBM clone is not required.
- next: Expand abstract operation families with high eval frequency + low coverage + low D1 gain.

## 8. JOINT_DISTRIBUTION_MISMATCH

- status: `PARTIALLY_SUPPORTED`
- evidence: `MEDIUM`
- for: Margin match can hide unseen joint cells (topology×calls×answer×track).
- against: Nearest-neighbor OOD may not predict flips at n=27/16.
- next: Selection objective: joint deficit matching + rare-cell floors.

## 9. DISTRACTOR_REALISM

- status: `PARTIALLY_SUPPORTED`
- evidence: `MEDIUM`
- for: If train distractors are lexically far / type-impossible, model learns weak discrimination.
- against: Hardness proxies are not semantic.
- next: Minimum distractor hardness gate; schema-compatible near-miss distractors.

## 10. DATA_SCALE

- status: `PARTIALLY_SUPPORTED`
- evidence: `MEDIUM`
- for: n=300 with 51% dead groups yields few effective updates; underpowered for +2pp.
- against: More easy data can worsen dead groups.
- next: Scale only after effective-group and joint-coverage constraints.

## 11. REFERENCE_SYNTAX_MISMATCH

- status: `PARTIALLY_SUPPORTED`
- evidence: `MEDIUM`
- for: Train and diagnostic may differ in $var vs $var_ and output key conventions.
- against: Parser accepts multiple formats; may not drive official_win.
- next: Validation gate on reference syntax + output-key distribution vs TargetProfile.

## 12. OTHER

- status: `PARTIALLY_SUPPORTED`
- evidence: `LOW`
- for: Template concentration / shortcut cues can inflate train reward without transfer.
- against: Shortcuts may be limited after paraphrase.
- next: Template-skeleton concentration gate; anti-leak checks in V7+.

## 13. TRAINING_SEED_VARIANCE

- status: `NOT_IDENTIFIABLE`
- evidence: `LOW`
- for: Single seed run; flips could be seed noise.
- against: No multi-seed artifacts available.
- next: Multi-seed train only after data composition fix.
