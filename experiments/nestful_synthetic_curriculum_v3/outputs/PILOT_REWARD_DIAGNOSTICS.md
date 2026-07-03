# Pilot Reward Diagnostics

## Timeline

| stage | epoch | synth strict_pass | synth mean_reward | dead_group_rate | kl |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 0.5779376498800959 | 0.5779376498800959 | 0.3884892086330935 | 0.0007735639810562134 |
| 1 | 2 | 0.5779376498800959 | 0.5779376498800959 | 0.3932853717026379 | 0.00788724422454834 |
| 2 | 1 | 0.2671875 | 0.26875 | 0.6859375 | 0 |
| 2 | 2 | 0.26875 | 0.26875 | 0.6796875 | 0 |

## Diagnosis

1. **GRPO signal:** Adequate in stage 1 (~39% dead groups). Stage 2 **inadequate** (~68% dead groups; many steps with zero contributing turns).
2. **Dead groups:** ~39% (s1) → ~68% (s2) — majority of groups had zero reward variance.
3. **Reward vs dev Win:** Stage 1 epoch 2 improves dev Win (+2pp) while synthetic strict_pass flat (~0.578). Stage 2 synthetic strict_pass ~0.268 but dev Win **drops** — negative transfer.
4. **motif_trace_consistency:** Logged indirectly via strict_pass; W&B did not export component-level reward rates — need train.log JSONL from pod.
5. **final_answer_pass vs Win:** NESTFUL eval shows high final_answer_pass (~64–69%) but low Win on 3-call — classic wrong-answer-with-partial-trace pattern.
6. **Gaming risk:** mean_reward spikes to 1.0 on last steps while epoch mean ~0.58 — possible short-trace ceiling hits on easy synthetic tasks.
7. **Weight changes recommended:** Increase too_few_calls penalty; cap final_pass with severe short trace; reduce stage2 mixed-replay weight until baseline beat is stable.

## Missing

- Per-component reward rates (motif_trace_consistency, valid_references) — export from `train.log` / W&B custom metrics next run.
