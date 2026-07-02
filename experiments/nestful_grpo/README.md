# nestful_grpo — původní curriculum přepočítané na dnešní metriky

Tahle složka **nespouští žádný trénink ani inferenci**. Bere *původní trajektorie*
z curriculum bězích (`curricullum/evaluation/...`) a přepočítává je na **dnešní
oficiální NESTFUL metriky** – stejným scorerem, jaký používají `nestful_mtgrpo_minimal`
a `nestful_mtgrpo_partial`. Cílem je mít původní (plain GRPO/SFT) curriculum
**apples-to-apples** vedle nových MT-GRPO bězích, bez nutnosti cokoli přepočítávat na GPU.

## Zdroj dat

Predikce z `curricullum/evaluation/`:

| dir | co to je |
|-----|----------|
| `results1/` | první curriculum eval (baseline + stage1/2/3), `predicted_calls` většinou prázdné |
| `results_toolr0/` | toolr0 varianta (baseline + stage3) |
| `results_v2_20260617/` | novější eval (stage3-e1, stage5-e2), `predicted_calls` většinou vyplněné |

Každý task má víc rolloutů; scorer je deduplikuje na **1 řádek/task** (rollout 0).
Počet unikátních tasků: **1861**.

## Výstupy

| soubor | obsah |
|--------|-------|
| `rescored_official.json` | surové oficiální metriky per profil, režimy `stored` + `reparse` |
| `curriculum_official_metrics.csv` | konsolidace: původní `final_answer_accuracy` + nové oficiální metriky |
| `rescore_official.py` (ve zdroji) | `curricullum/evaluation/rescore_official.py` |
| `consolidate.py` | spojí JSON + původní summary do CSV |

## Dvě varianty parseru

- **stored** – skóruje `predicted_calls` uložené ve starém souboru. Pro `results1`
  a `results_toolr0` je ale `predicted_calls` skoro vždy prázdné, takže tahle čísla
  podhodnocují skutečnost.
- **reparse** – znovu vytáhne tool cally z `raw_completions` **dnešním lenient
  parserem**. Tohle je férové číslo pro všechny profily (parser je stejný jako v MT-GRPO).

> Pro srovnání s MT-GRPO používej **reparse**.

## Význam metrik (dnešní oficiální NESTFUL scorer)

| metrika | význam |
|---------|--------|
| `f1_func` | F1 přes názvy volaných funkcí (correct function selection) |
| `f1_param` | F1 přes (název funkce + parametry) – přísnější |
| `partial_sequence_accuracy` | podíl tasků, kde se sekvence shoduje aspoň zčásti |
| `full_sequence_accuracy` | podíl tasků s přesně správnou celou sekvencí (**je v JSON/CSV**) |
| `win_rate` | oficiální execution-based metrika (**chybí na Windows**, viz níže) |
| `avg_pred_calls` | průměrný počet predikovaných volání na task |
| `final_answer_accuracy` | **původní** metrika – executor pass-rate (zda vyšel finální výsledek) |

### Proč je Full Acc ≈ 0?

Všechny tyto běhy jsou **ReAct multi-turn**. Model emituje konkrétní hodnoty místo
`$varN$` referencí → oficiální grounded Partial/Full jsou ~0 i když
`final_answer_accuracy` je 67–71 %. To **není chybějící metrika**, ale očekávané
chování (stejně jako u baseline ReAct v MT-GRPO: Full 0.000, Win ~0.54).

## Win Rate (dopočteno offline, funguje i na Windows)

`win_rate` počítá oficiální IBM re-exekuce predikovaných callů. Dřív chybělo kvůli
Windows (`SIGALRM`) a chybějícím `output_parameters` ve starých prediction souborech —
obojí je opraveno:

- **Windows shim** v `nestful_official_score.py` (threading `signal.alarm`)
- **tools z NESTFUL datasetu** podle `task_id` (predictions měly jen `parameters`)

Regenerace (~4 min, bez GPU):

```bash
python curricullum/evaluation/rescore_official.py --reparse \
  --out experiments/nestful_grpo/rescored_official.json
python experiments/nestful_grpo/consolidate.py
```

Volitelně na Linuxu: `bash experiments/nestful_grpo/rescore_linux.sh`

## Jak to znovu vygenerovat

```bash
# přepočet trajektorií včetně Win Rate (~4 min, bez GPU)
python curricullum/evaluation/rescore_official.py --reparse \
  --out experiments/nestful_grpo/rescored_official.json

# konsolidace do CSV
python experiments/nestful_grpo/consolidate.py
```
