# helper_calculations

Pomocné skripty pro statistiky nad daty (bez tréninku / evalu).

## NESTFUL — rozložení podle počtu gold tool callů

Stejný benchmark jako eval: `eval/data/NESTFUL-main/data_v2/nestful_data.jsonl`  
(fallback: `nestful_repo/data_v2/nestful_data.jsonl`).

```bash
python helper_calculations/nestful_call_distribution.py
```

Výstup:

- `helper_calculations/output/nestful_call_distribution.json` — tabulka + seznam `sample_id` per bucket
- `helper_calculations/output/nestful_call_distribution.md` — přehledná tabulka

Volitelně:

```bash
python helper_calculations/nestful_call_distribution.py \
  --data eval/data/NESTFUL-main/data_v2/nestful_data.jsonl \
  --output-dir helper_calculations/output
```

Počet callů = `len(output)` v JSONL (gold trajectory), konzistentně s `nestful_evaluation/run.py`.
