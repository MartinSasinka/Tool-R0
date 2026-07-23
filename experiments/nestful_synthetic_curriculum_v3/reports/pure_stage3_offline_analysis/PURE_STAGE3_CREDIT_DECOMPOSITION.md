# Credit assignment decomposition (train logs)

**Generated:** 2026-07-23T06:35:03.604081+00:00

R²(G₀ ~ R_episode) = **0.7562** (correlation only; see note below)

R²(G₀~R_ep) is correlation of first-turn return vs episode reward, NOT fraction of advantage variance from episode term. Per-turn table uses P_t (discounted future turn rewards) vs E_t (discounted terminal episode bonus) with trainer-normalized advantages.

| Turn | n | mean |A| | mean |P_t| | mean |E_t| | var share P | sign mismatch | good turn, neg adv | bad turn, pos adv |
|------|--:|-----------:|-----------:|-----------:|------------:|--------------:|-------------------:|------------------:|
| 1 | 5216 | 0.6650 | 2.6221 | 0.6297 | 0.823 | 0.498 | 1725 | 0 |
| 2 | 5052 | 0.6444 | 1.7096 | 0.6443 | 0.704 | 0.480 | 1440 | 26 |
| 3 | 4944 | 0.6308 | 0.8580 | 0.6546 | 0.484 | 0.405 | 1404 | 36 |