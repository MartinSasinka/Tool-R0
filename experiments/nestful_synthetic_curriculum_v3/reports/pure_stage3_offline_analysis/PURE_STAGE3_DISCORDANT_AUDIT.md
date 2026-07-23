# Discordant task audit (C0↔E2)

**Generated:** 2026-07-23T06:35:03.620872+00:00
**Discordant tasks:** 167 (74 gained, 93 lost)

## Change type (all discordant)

| change_type | count | share |
|-------------|------:|------:|
| kratší než gold trace (metrika, ne nutně chyba) | 67 | 40.1% |
| jiná, ale validní cesta | 65 | 38.9% |
| stejný tool, jiné values | 16 | 9.6% |
| změněný pozdější tool | 7 | 4.2% |
| změněný první tool | 5 | 3.0% |
| nesprávné použití observation | 3 | 1.8% |
| vykonatelná cesta se špatným výsledkem | 2 | 1.2% |
| předčasné ukončení | 2 | 1.2% |

## C0 win → E2 loss — first divergence turn

| first_changed_turn | count |
|-------------------:|------:|
| 1 | 45 |
| 2 | 36 |
| 3 | 4 |
| 4 | 2 |
| 5 | 4 |
| 7 | 1 |
| 16 | 1 |

## Under-calling metric vs taxonomy (all test, reminder)

Metric `pred_calls < gold_calls` is **not** premature stop. See per-turn report for taxonomy `too few calls` rate (~0.8%).

- Discordant lost with under-calling metric: 69/93
- Discordant lost with too_few taxonomy: 2/93

Full rows: `PURE_STAGE3_DISCORDANT_AUDIT.jsonl`