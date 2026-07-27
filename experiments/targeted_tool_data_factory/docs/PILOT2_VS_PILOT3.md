# Pilot2 vs Pilot3

Generated 2026-07-27 03:15 UTC.

| | pilot2 | pilot3 |
|---|---|---|
| selected | 320 | 1000 |
| train / heldout / reserve | 160 / 80 / 80 | 600 / 200 / 200 |
| seed | 20260726 | 20260727 |
| G-track share | 40.3 % | 44.8 % |
| cell_max_share cap | 10 % | 8 % |
| long-horizon boost | none | +2.5 pp on 5-call, +3.5 pp on 6+ |

## Call-count share

| bucket | pilot2 | pilot3 | NESTFUL |
|---|---|---|---|
| `2` | 37.5 % | 31.4 % | 33.0 % |
| `3` | 17.5 % | 17.5 % | 22.0 % |
| `4` | 14.1 % | 13.4 % | 13.5 % |
| `5` | 9.7 % | 12.1 % | 9.5 % |
| `6` | 12.5 % | 13.3 % | 22.0 % |
| `7` | 3.1 % | 5.5 % | 22.0 % |
| `8` | 5.6 % | 6.8 % | 22.0 % |
| `6+` | 0.0 % | 0.0 % | 22.0 % |

## Profile match (selected vs NESTFUL)

| metric | pilot3 |
|---|---|
| JSD call_bucket | 0.004112 |
| JSD motif | 0.048685 |
| JSD answer_type | 0.005675 |
| Wasserstein n_tools | 0.755 |
| two-sample AUC | 0.5448 |

Pilot2 artefacts under `outputs/**/*pilot2*` and `runpod_bundle_pilot2/` were
**not** modified.
