# W&B Status Overview (pulled after RunPod loss)

Pulled: see `wandb_training_pull.json`  
Entity: `sasinka-martin`  
Projects: `nestful_v5_pure_stage3`, `nestful-v5-curriculum`

**Security:** API key was pasted in chat — rotate it in W&B → Settings → API keys.

---

## 1. Two-phase GRPO (`two_phase_20260718_192902`)

Project: [nestful-v5-curriculum](https://wandb.ai/sasinka-martin/nestful-v5-curriculum)

| Phase | Dead-group rate | Mean reward | Train win_rate | global_step |
|------|----------------:|------------:|---------------:|------------:|
| Phase 1 (429× Stage2) | **78.3%** | 0.533 | 0.031 | 0→24 |
| Phase 2 (466× S3+replay) | **31.0%** | 0.588 | 0.197 | 24→105 |
| Phase2 / stage3 only | 17.2% | 0.625 | — | — |
| Phase2 / stage2 replay | 62.9% | 0.500 | — | — |

Official nestful_test (n=1661), from W&B final_eval runs:

| Arm | Win Rate | F1 func | F1 param |
|-----|---------:|--------:|---------:|
| C0 | 0.5352 | 0.898 | 0.439 |
| C1 | 0.5370 | 0.889 | 0.442 |
| C2 | 0.5394 | 0.888 | 0.442 |

→ Matches local root-cause analysis: **+0.42 pp C2−C0, not meaningful.**  
Local format audit still stands: format mostly OK, semantics dominate.

---

## 2. Pure Stage 3 overnight (`pure_stage3_2ep_20260719_221918`)

Project: [nestful_v5_pure_stage3](https://wandb.ai/sasinka-martin/nestful_v5_pure_stage3)

| Epoch | Tasks | Dead-group | Mean reward | Epoch win_rate | Steps |
|------|------:|-----------:|------------:|---------------:|------:|
| E1 | 326 | **27.9%** | 0.626 | 0.283 | 0→59 |
| E2 | 326 | **15.0%** | 0.634 | 0.294 | 59→129 |

- Continuous training: E2 `continuous_training=True`, monotonic `global_step`
- `no_tool_call_rate` ≈ 0.04% (excellent format during train)
- `too_few_calls_rate` ≈ 14–15% (still under-calling vs gold 3-call)

**Eval status on W&B:**
- C0 dev (n=200): win **0.54** — finished  
- E1 dev: run exists (`pq9k4plx`) but **empty summary** → almost certainly crashed (same vLLM/GPU issue)  
- E2 / final test: **not logged** → overnight eval did not finish before pod died  

→ Full overnight train **did complete** on W&B; official Nestful transfer for E1/E2 is **unknown**.

---

## 3. Pure Stage 3 smoke (`pure_stage3_smoke_20260719_213722`)

Tiny train (8 tasks) then later full evals appeared on W&B:

| Eval | n | Win Rate | Notes |
|------|--:|---------:|-------|
| C0 / E1 / E2 dev | 20 | 0.60 each | too small / noisy |
| C0 final test | 1661 | **0.5382** | |
| E2 final test | 1661 | **0.5340** | **−0.42 pp vs C0** |

Smoke E2 on full test is slightly **worse** than baseline — consistent with “no free lunch” / underpowered 8-task train, but also a warning that Stage3-only can hurt if unstable.

---

## 4. What’s wrong (priority)

1. **Learning signal:** Phase1 dead groups ~78%; pure-S3 E1 still ~28% (better, but not great).  
2. **Transfer:** two-phase Nestful win flat; smoke E2 slightly down. Overnight E1/E2 Nestful scores **missing**.  
3. **Infra:** eval TP=4 after train was fragile (fixed in code: unload learner + util cap) — overnight lost post-train eval.  
4. **Not the main issue:** raw tool-call format (local audit + tiny `no_tool_call_rate` on train).

---

## 5. What to do next

1. **Rotate W&B API key** (exposed in chat).  
2. On next RunPod credit: sync fixed eval teardown code; re-run **only eval** if checkpoints still on a volume, else retrain pure-S3 overnight with immediate artifact download (`checkpoints/`, `eval/`, `console.log`).  
3. **Decision gate after overnight E2 nestful_test:**  
   - if win ≤ C0 + ~1 pp → stop “more Stage3 epochs”, change objective (credit / reward / data mix);  
   - if win up on 3-call and 6+ without killing 2-call → continue.  
4. Do **not** add format reward; focus semantic credit on long chains.  
5. Always `rclone`/scp run dir before killing the pod.

### Handy W&B links
- Overnight train E1: https://wandb.ai/sasinka-martin/nestful_v5_pure_stage3/runs/adt14fso  
- Overnight train E2: https://wandb.ai/sasinka-martin/nestful_v5_pure_stage3/runs/jutdo9fv  
- Smoke E2 test: https://wandb.ai/sasinka-martin/nestful_v5_pure_stage3/runs/iorhhhqf  
- Two-phase C2 test: https://wandb.ai/sasinka-martin/nestful-v5-curriculum/runs/esmftg0u  
