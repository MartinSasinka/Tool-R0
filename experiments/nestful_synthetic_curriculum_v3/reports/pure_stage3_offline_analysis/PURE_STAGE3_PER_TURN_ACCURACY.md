# Per-turn conditional accuracy (NESTFUL test)

**Generated:** 2026-07-23T06:35:03.621507+00:00

| Metrika | C0 | E1 | E2 |
|---------|---:|---:|---:|
| správný první tool | 51.48% | 51.78% | 51.78% |
| správný 2. tool při správném 1. | 79.70% | 77.84% | 76.75% |
| správný 3. tool při správném prefixu 1–2 | 77.20% | 75.27% | 80.22% |
| správná reference na observation (turn 2–3) | 25.00% | 27.46% | 29.32% |
| správný terminal outcome při executable | 68.07% | 67.81% | 67.87% |

## Under-calling vs premature stop

| Metrika | C0 | E1 | E2 |
|---------|---:|---:|---:|
| pred_calls < gold_calls (eval metric) | 59.72% | 60.45% | 60.69% |
| taxonomy: too few calls | 0.78% | 0.96% | 0.84% |

Rozdíl ~60 pp vs ~0.8 pp potvrzuje, že under-calling metrika **nesmí** řídit SFT.