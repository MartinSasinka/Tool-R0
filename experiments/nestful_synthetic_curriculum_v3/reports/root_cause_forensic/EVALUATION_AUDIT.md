# EVALUATION_AUDIT — audit evaluace

Zdroj: `analysis/a05_eval_audit.json` (přepočteno z raw `task_results.jsonl`
a `_traj`, nezávisle na uložených metrics soubory).

## Párování a reprodukce metrik: OK

- 500 eval ID: unikátní, kompletní; `ids_match_500 = true` u všech armů.
- **Win Rate reprodukovaný přesně** z raw řádků u všech armů
  (A0 0.574, A1 0.566, A2 0.562, A3 0.530, A4 0.574; C0 na 500 = 0.570).
- **Gained/lost vs C0 reprodukované přesně** proti uloženým paired výsledkům
  (A0 +31/−29, A1 +27/−29, A2 +19/−23, A3 +18/−38, A4 +30/−28).
- A0 vs A4: 21 úloh vyhrává jen A0, 21 jen A4 (potvrzeno) — a **355/500
  (71 %) mají bitově identickou sekvenci predicted calls**. A0 je identický
  s C0 na 326/500 (65 %). Vyhodnocované modely se od C0 téměř neliší.

## Anomálie C0 eval setu (nalezena, bez dopadu na párovanou analýzu)

`shared_C0_eval_500` běžel na **1861 řádcích** (nestful_test 1661 + dev),
ačkoli jméno/manifest implikuje 500; `metrics_official_win_rate = 0.5513` se
vztahuje k n=1861, ne k n=500. Pokrytí 500 párovaných ID je ale kompletní a
C0=57.0 % na 500 je korektní. Manifest neuvádí checkpoint. Riziko: záměna
0.5513 za baseline na 500 v budoucích reportech — označeno jako chybně
pojmenovaná metrika.

## Kratší validní cesty a premature stop

- Oficiální scorer akceptuje alternativní/kratší validní řešení:
  `alternative_valid_solution_pass` u 254/500 C0 úloh.
- `predicted calls < gold calls` má C0 na 67.4 % úloh — většinou jde o
  legitimní kratší cesty NEBO skutečný předčasný stop; rozlišení dává
  taxonomie: skutečné `too_few_calls` selhání je 103–119 úloh/arm,
  `no_tool_call` 41–55, `executable_wrong_final` 16–23, `too_many_calls`
  18–23, `execution_failure` 19–23.

## Win Rate dle počtu gold calls (potvrzeno: 2-call nejslabší)

| Bucket | n | C0 | A0 | A4 |
|---|---|---|---|---|
| 2-call | 100 | 0.45 | 0.49 | 0.48 |
| 3-call | 100 | 0.62 | 0.64 | 0.63 |
| 4+ | 300 | 0.593 | 0.58 | 0.587 |

2-call slabost je vlastnost C0/benchmarku (krátké úlohy mají menší toleranci
k undercallingu a vyšší podíl str-argumentů), ne efekt tréninku.

## Verdikt

Eval pipeline (parser, executor, oficiální scorer, párování, výpočty) je
**bez nálezu** — všechna čísla reprodukovatelná z raw dat. Jediný problém je
mislabeling C0 eval setu (1861 vs 500) popsaný výše.
