# Clean curriculum — cleaning report

- Vstup: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_mtgrpo_minimal\data\filtered_toolr0_synthetic`
- Výstup: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_mtgrpo_minimal\data\clean_curriculum`
- IBM gold executor dostupný: **True** (spuštěny i execution checks/repairs)

## Zásada

Odstraňují se **pouze objektivně vadné / neexekuovatelné** příklady. **Neodstraňují se** těžké, dlouhé ani multi-call úlohy, ani úlohy, které model neumí, ani úlohy s alternativní cestou — pokud je gold trace validní a spustitelná.

## Počty po stage

| stage | total | kept | removed | repaired |
|-------|-------|------|---------|----------|
| 1 | 400 | 400 | 0 | 0 |
| 2 | 400 | 381 | 19 | 0 |
| 3 | 400 | 381 | 19 | 0 |
| 4 | 400 | 384 | 16 | 0 |
| 5 | 400 | 375 | 25 | 0 |
| 6 | 268 | 236 | 32 | 0 |
| **Σ** | **2268** | **2157** | **111** | **0** |

## Důvody odstranění (per stage)

### stage 2
- `gold_answer_unresolved`: 19

### stage 3
- `gold_answer_unresolved`: 19

### stage 4
- `gold_answer_unresolved`: 15
- `missing_field:gold_answer`: 1

### stage 5
- `gold_answer_unresolved`: 25

### stage 6
- `gold_answer_unresolved`: 32

## Provedené opravy (per stage)

- (žádné opravy nebyly potřeba)

## Detaily

- `removed_examples.csv` — odstraněné příklady (stage, line_index, sample_id, reason).
- `repaired_examples.csv` — opravené příklady (stage, line_index, sample_id, repairs).
- `validation_summary.json` — strojově čitelný souhrn.

Reward se touto úpravou **nemění**; jde pouze o čistotu trénovacích dat.
