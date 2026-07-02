# NESTFUL / MT-GRPO — meeting analysis

Rychlá rozhodovací analýza existujících běhů (bez dalšího tréninku).

## Spuštění

```bash
python experiments/comparison/meeting_analysis.py
```

První běh může trvat ~10–15 min, pokud chybí cache pro baseline Direct (`official_win` není v `direct_eval_trajectories.jsonl`). Cache: `experiments/comparison/.cache/`.

## Výstupy

| Soubor | Popis |
|--------|--------|
| [MEETING_BRIEF.md](../comparison/MEETING_BRIEF.md) | 1stránkový briefing (CS) pro schůzku |
| [meeting_summary.csv](../comparison/meeting_summary.csv) | Hlavní srovnávací tabulka 8 klíčových běhů |
| [win_loss_overlap.csv](../comparison/win_loss_overlap.csv) | Win/loss overlap (4 bucketů) |
| [failure_taxonomy.csv](../comparison/failure_taxonomy.csv) | Heuristická taxonomie chyb |
| [meeting_analysis.py](../comparison/meeting_analysis.py) | Generátor (opakovatelné) |

Zdrojová data: `experiments/comparison/{all_runs.json,diagnostics.json,final_eval_all.csv}`, `outputs/final_eval/**`.

## Hlavní závěr (2025-06-29)

- Fine-tuning **nepřekonal** baseline na Full Acc / Win Rate.
- **Strict** reward → ReAct kolaps (Win 0.544 → 0.325).
- **Partial s1e4** ReAct ≈ baseline (Win 0.543); delší partial trénink degraduje.
- Další krok: **execution-dominant reward** + early stopping na val Win.
