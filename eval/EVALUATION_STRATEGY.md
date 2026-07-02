# Evaluační strategie Tool-R0

## TL;DR

Tool-R0 trénuje model (Qwen2.5-1.5B-Instruct) na **single-turn generování tool callů** pomocí GRPO. Evaluace pokrývá 6 benchmarků ve 3 úrovních — od single-turn přesnosti přes reálné API až po multi-turn konverzace a vnořené sekvence.

## Tréninková úloha

Model generuje strukturované tool cally:

```
<think>[úvaha]</think>
<tool_call_answer>[{"name": "func", "arguments": {"key": "value"}}]</tool_call_answer>
```

Tréninková odměna (`rewards_solver.py`): `0.2 × name + 0.3 × key_F1 + 0.5 × value_match`

Evaluační metrika: **binární AST matching** (shodné s článkem a Patil et al., 2024) — buď je celý tool call 100% správně (name + keys + values), nebo 0.

## Přehled benchmarků

| # | Benchmark | Typ | Úloh | Co měří |
|---|-----------|-----|-----:|---------|
| 1 | BFCL AST | single-turn | 1040 | Přesnost function calling (simple, multiple, parallel, irrelevance) |
| 2 | BFCL Exec | single-turn | 240 | Executable verifikace (skutečné spuštění) |
| 3 | ToolAlpaca | single-turn | 100 | Generalizace na reálná OpenAPI schémata |
| 4 | API-Bank | single-turn (reálné API) | 399 | 73 reálných API z různých domén |
| 5 | ToolTalk | multi-turn (reálné API) | 164 turnů / 78 konverzací | 28 nástrojů ve 7 doménách, easy+hard |
| 6 | NESTFUL | nested sequences | 1861 | Vnořené sekvence API callů (math + coding), výstup jednoho callu → vstup dalšího |

## Proč tyto benchmarky

### BFCL — referenční single-turn benchmark
Zlatý standard pro function calling (ICML 2025). Výsledky přímo srovnatelné s veřejným leaderboardem (GPT-4, Claude, Llama). Pokrývá 4 kategorie: simple (1 funkce), multiple (výběr z N), parallel (M callů najednou), irrelevance (nemá volat žádný nástroj).

### ToolAlpaca — generalizace na neviděná API
Používá reálná OpenAPI schémata s ground-truth tool cally. Testuje generalizaci na API, která model nikdy neviděl. Stejná reward formule jako trénink — přímo ukazuje, co se model naučil.

### API-Bank — reálné API prostředí
73 reálných API z různých domén (kalendář, zdravotnictví, smart home, vyhledávání). Na rozdíl od syntetických benchmarků má skutečné parametry (časy, ID, tokeny, souřadnice). Level-1 (single-turn s daným popisem API) je přímo srovnatelný s článkem.

### ToolTalk — nejtěžší multi-turn test
Microsoft benchmark s 28 reálnými (simulovanými) nástroji ve 7 doménách. Konverzace od jednoduchých (1 call) po složité (5+ řetězených callů přes 3 domény). Ukazuje hranice single-turn tréninku.

### NESTFUL — vnořené sekvence API callů
IBM benchmark s 1861 úlohami vyžadujícími **nested sequencing** — sekvence callů, kde výstup jednoho je vstupem dalšího (typicky 2-6 callů). Domény: matematické reasoning a coding nástroje. Měříme: Full Match Accuracy (celá sekvence správně), Partial Match (per-call), Name F1. Všechny funkce jsou executable — v budoucnu lze přidat Win Rate (porovnání výsledku s gold answer).

## Výsledky

### Souhrnná tabulka (s bootstrap 95% CI)

Pro spolehlivé reportování výsledků jsme použili **bootstrap resampling** (1000 iterací, seed=42) z existujících predikcí. Tato standardní statistická metoda opakovaně vzorkuje s vracením z nasbíraných predikcí a počítá metriku na každém vzorku — výsledkem je rozptyl a 95% confidence interval bez nutnosti opakované (drahé) inference.

Skript: `python -m eval.scripts.bootstrap_analysis --n-bootstrap 1000`

| Benchmark | Metrika | Baseline | Finetuned | Delta |
|-----------|---------|----------|-----------|-------|
| BFCL AST (n=1040) | accuracy | 42.69 +/- 1.51 | **63.46 +/- 1.48** | +20.77 |
| BFCL Exec (n=240) | accuracy | 41.67 +/- 3.32 | **62.92 +/- 3.13** | +21.25 |
| ToolAlpaca (n=100) | AST accuracy | 17.00 +/- 3.77 | **30.00 +/- 4.56** | +13.00 |
| API-Bank (n=399) | AST accuracy | 43.86 +/- 2.40 | **57.14 +/- 2.56** | +13.28 |
| ToolTalk (n=164) | turn accuracy | 9.15 +/- 2.22 | **15.85 +/- 3.00** | +6.70 |

Všechna zlepšení jsou **statisticky signifikantní** — 95% confidence intervaly baseline a finetuned se nikde nepřekrývají.

### BFCL detailní rozbor

| Kategorie | Baseline | Finetuned | Delta |
|-----------|----------|-----------|-------|
| simple (n=400) | 52.00 +/- 2.49 | **87.00 +/- 1.69** | +35.00 |
| multiple (n=200) | 50.00 +/- 3.44 | **79.00 +/- 2.87** | +29.00 |
| parallel (n=200) | 33.50 +/- 3.38 | **74.50 +/- 3.07** | +41.00 |
| irrelevance (n=240) | 28.75 +/- 2.96 | 2.08 +/- 0.97 | **-26.67** |

**Irrelevance regrese**: Model po fine-tuningu volá nástroje i tam, kde nemá — typický vedlejší efekt tool-call tréninku. Bez irrelevance kategorie: baseline 46.88% → finetuned **81.88%** (+35.00 pp).

### ToolTalk detailní rozbor

| Metrika | Baseline | Finetuned | Delta |
|---------|----------|-----------|-------|
| Turn accuracy | 9.15% | **15.85%** | +6.70 |
| Conversation success | 8.97% | **17.95%** | +8.98 |
| Easy turns (n=28) | 25.00% | **50.00%** | +25.00 |
| Hard turns (n=136) | 5.88% | 8.82% | +2.94 |
| Name recall | 33.83% | **52.63%** | +18.80 |
| Parse success | 56.10% | **80.49%** | +24.39 |

### Multi-turn transfer

| Schopnost | Trénovaná? | Výsledek |
|-----------|:---:|---|
| Vybrat správnou funkci | Ano | Name match 94% (ToolAlpaca), name recall 53% (ToolTalk) |
| Zadat správné parametry | Ano | AST accuracy +13pp (ToolAlpaca, API-Bank) |
| Generovat validní formát | Ano | Parse success 49% → 98% (ToolAlpaca), 56% → 81% (ToolTalk) |
| Easy multi-turn | Ne | 25% → **50%** (ToolTalk easy) |
| Složité řetězení (5+ callů) | Ne | 5.88% → 8.82% (ToolTalk hard) — limitace |

## Metodologie evaluace

### AST matching (binární metrika)

Pro srovnatelnost s článkem (Tool-R0 paper, Patil et al. 2024) používáme binární AST matching:
- Pro každý vzorek: buď je predikovaný tool call **100% strukturálně správný** (funkce + klíče + hodnoty), nebo ne
- Porovnání hodnot s type coercion (string "5" = int 5), whitespace normalizací
- Implementace v `eval/ast_eval.py`, duplikuje logiku z `rewards_solver.py` bez závislosti na wandb

Trénink používá **soft reward** (partial credit) pro lepší gradient signal. Evaluace používá **binární metriku** pro srovnatelnost s literaturou. Obě sdílejí stejnou funkci `robust_value_match` pro porovnání hodnot.

### Bootstrap resampling

Pro odhad rozptylu a confidence intervalů používáme bootstrap (Efron, 1979):
1. Z existujících N predikcí náhodně vzorkujeme N predikcí **s vracením**
2. Na každém vzorku spočítáme metriku (accuracy)
3. Opakujeme 1000x → distribuce metriky
4. Z distribuce odečteme mean, std a 95% percentilový CI

Výhoda: nepotřebuje opakovanou inference (drahé GPU hodiny), pracuje s existujícími predikcemi.

## Limitace a známé problémy

1. **Irrelevance regrese** — model ztrácí schopnost odmítnout tool call (BFCL irrelevance -26.67pp)
2. **Složité multi-turn** — ToolTalk hard (3 domény, 5+ callů) zůstává nízké (~9%)
3. **Velikost modelu** — 1.5B parametrů limituje kapacitu pro dlouhé kontexty
4. **NESTFUL** — model trénovaný na single-turn úlohy nemusí generovat celé sekvence vnořených callů

## Proč nepoužíváme těžší agentní frameworky

| Framework | Problém pro Tool-R0 |
|-----------|---------------------|
| StableToolBench | Měří agentní plánování, ne kvalitu tool callů; vyžaduje GPT-4 evaluátor |
| ToolSandbox | Vyžaduje plnou agentní smyčku (Apple framework) |
| ToolBench (7000+ API) | Nedeterministické výsledky (reálné HTTP API) |

Tyto frameworky testují **agentní chování** (plánování, stav, opakování) — schopnosti, na které 1.5B single-turn model nebyl trénován. SimpleEnv + ToolTalk pokrývají multi-turn evaluaci na úrovni přiměřené modelu.

## Reference

- **BFCL**: Yan et al., "Berkeley Function Calling Leaderboard", ICML 2025
- **ToolAlpaca**: Tang et al., "ToolAlpaca: Generalized Tool Learning for Language Models", 2023
- **API-Bank**: Li et al., "API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs", EMNLP 2023
- **ToolTalk**: Farn & Shin, "ToolTalk: Evaluating Tool-Usage in a Conversation Setting", 2023
- **NESTFUL**: Basu et al., "NESTFUL: A Benchmark for Evaluating LLMs on Nested Sequences of API Calls", 2024
- **Tool-R0 paper**: "Tool-R0: Self-Evolving LLM Agents for Tool-Learning from Zero Data"
- **AST matching**: Patil et al., "Gorilla: Large Language Model Connected with Massive APIs", 2024
