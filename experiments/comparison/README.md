# comparison — srovnání všech NESTFUL bězích

Agreguje výsledky tří experimentů do jedné srovnatelné sady. Vše přepočítané
**jedním oficiálním NESTFUL scorerem** (`nestful_official_score.py`).

## Zdroje

| experiment | co se bere |
|------------|-----------|
| `nestful_mtgrpo_minimal` | `final_outputs/consolidated_metrics.json` (baseline + curriculum pilot + v2) |
| `nestful_mtgrpo_partial` | `outputs/curriculum/**/eval/metrics.json` (per-epoch) + `outputs/final_eval/<ckpt>_<paradigm>/metrics_official.json` |
| `nestful_grpo` | `rescored_official.json` (původní curriculum, dnešní metriky) |

## Výstupy

| soubor | obsah |
|--------|-------|
| `all_runs.json` | vše strojově (partial curriculum + final eval + minimal) |
| `final_eval_all.csv` | sjednocená tabulka final-eval všech experimentů (Direct + ReAct) |
| `diagnostics.json` / `diagnostics.csv` | per-run macro-F1, **micro-F1**, per-task **set-match** + **seq-match**, Full, Win |
| `aggregate.py` | regenerace `all_runs.json` + `final_eval_all.csv` |
| `diagnostics.py` | regenerace diagnostiky (přepočítává z predikcí/trajektorií) |

## Vizualizace

Canvas summary: `canvases/nestful-vsechny-behy.canvas.tsx` (otevři v IDE vedle chatu).

## Klíčové závěry

- **Direct**: všechny checkpointy ≈ baseline (F1 Param ~0.64, Win ~0.27) — finetuning Direct nepoškodil.
- **ReAct**: minimal (strict) kolabuje (F1 Func 0.894 → 0.153); partial degraduje postupně,
  `part·s1e4` je prakticky na baseline (F1 Func 0.926, Win 0.543).
- **Trénink**: partial drží `final_answer_pass` ~0.5+ napříč stage (minimal padá k 0.16),
  přestože `strict_gold_trace_pass` je u obou nízký → graded reward = lepší generalizace.
- **F1 caveat**: official F1 Func je macro-F1 přes ~904 názvů funkcí (881 singleton/near-singleton) →
  nasycené k ~0.92. Diagnostika to ukazuje: micro-F1 ~0.84, set-match ~0.31, seq-match ~0.28.
  **Vysoké F1 ≠ task success.**
- **vs paper** (arXiv:2409.03797 **v3** / EMNLP 2025, Table 1, Direct, 1-shot): bezpečné tvrzení —
  *„Our model shows high format/function-name compliance, but execution-oriented metrics (Full Acc,
  Win Rate) remain below GPT-4o / DeepSeek-V3."* Netvrdíme převahu podle F1.

## Regenerace

```bash
python experiments/comparison/aggregate.py
python experiments/comparison/diagnostics.py
```
