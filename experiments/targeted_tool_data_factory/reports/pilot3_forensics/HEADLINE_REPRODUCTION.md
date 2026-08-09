# HEADLINE_REPRODUCTION

## Matched-engine C0-vLLM vs D1-vLLM

- C0 wins: **277/500** (55.4%)
- D1 wins: **288/500** (57.6%)
- Absolute delta: **+2.20 pp**
- Transitions: win→win=261, loss→loss=196, loss→win=27, win→loss=16
- McNemar exact p = `0.1263` (b=16, c=27)
- Paired bootstrap 95% CI (pp): [-0.40, 4.80]
- Stratified (call-bucket) bootstrap 95% CI (pp): [-0.40, 4.80]

## Call-count buckets

- 2: n=100 C0=48.0% D1=47.0% Δ=-1.0 pp (gained=3, lost=4)
- 3: n=100 C0=59.0% D1=62.0% Δ=+3.0 pp (gained=6, lost=3)
- 4: n=100 C0=64.0% D1=65.0% Δ=+1.0 pp (gained=3, lost=2)
- 5: n=100 C0=56.0% D1=60.0% Δ=+4.0 pp (gained=8, lost=4)
- 6+: n=100 C0=50.0% D1=54.0% Δ=+4.0 pp (gained=7, lost=3)

## Interpretation guards

- diagnostic-500 is a balanced slice (100 tasks per call-count bucket 2/3/4/5/6+).
- Overall win equals the macro average across call-count buckets.
- It is not an estimate of naturally distributed NESTFUL official win rate.
- +delta_pp is a matched-engine point estimate, not a proven causal training effect.
- Residual LoRA inference-path confound cannot be removed from these two trajectory sets alone.

## Three-arm (separate; not main contrast)

- C0-HF: 48.8%
- C0-vLLM: 55.4%
- D1-vLLM: 57.6%
- C0-HF must not be mixed into the main training-effect contrast.

## Methodology

- alignment: `sample_id`
- bootstrap: 20000 iters, seed=42
- McNemar: exact two-sided binomial(b+c, 0.5)
