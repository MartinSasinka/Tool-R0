# NESTFUL — analýza výsledků a kontrola správnosti

Tento dokument shrnuje, **co je ve `outputs/`**, **které běhy měly/neměly spočítané
metriky**, jak je celá evaluace udělaná a **jestli je správně**, a vysvětluje dvě
podezřelé věci, kterých sis všiml: „F1 kolem 90 %" a „finetuned vychází hůř než
baseline".

Všechna čísla v tabulkách (`RESULTS.md`, CSV) jsou **přepočítaná jedním
oficiálním NESTFUL scorerem** ze surových predikcí/trajektorií, takže jsou
navzájem porovnatelná a nezávislá na tom, co se uložilo za běhu.

> **Zdroj pravdy:** `official_*` z `nestful_official_score.py` (oficiální scorer
> v `data/NESTFUL-main/src`). Interní `metrics.py` je jen diagnostika. Detailně
> viz `docs/AUDIT.md`.

---

## 1. Inventář běhů a stav metrik

| Běh (složka) | Paradigma | Model | Co tam bylo | Stav metrik |
|---|---|---|---|---|
| `final_eval_baseline_react/` | ReAct | baseline | `metrics.json` (interní, **starý formát**) + trajektorie | ⚠️ jen interní; **chybělo `metrics_official.json`** |
| `baseline_direct/` | Direct | baseline | `metrics_official.json` + predictions + trajektorie | ✅ kompletní (vč. Win Rate z Linuxu) |
| `final_eval_stage4_epoch2_react/` | ReAct | curriculum s4e2 | `metrics.json` (interní, starý) + trajektorie | ⚠️ jen interní; **chybělo `metrics_official.json`** |
| `stage4_epoch2_direct/` | Direct | curriculum s4e2 | **jen `direct_predictions.jsonl`** | ❌ metriky se nedopočítaly (pády na scoringu) |
| `final_eval_baseline/` | ReAct | baseline | jen trajektorie, žádné metriky | ⚠️ starší/neúplný běh |
| `test_nestful_eval/` | rollout | — | malý test | testovací, ignorováno |
| `curriculum/stage_1..4/` | rollout eval | mezistupně | per-epoch `eval/metrics.json` + trajektorie | ⚠️ malý eval, interní formát |

**Závěr inventáře:**
- Žádný z **ReAct** běhů neměl oficiální metriky — uložil se jen interní
  `metrics.json` (protože v době běhu chyběl `jsonlines` a oficiální scoring
  spadl do `try/except`).
- **`stage4_epoch2_direct`** neměl spočítané vůbec nic kromě predikcí (pády na
  bugu s `gold_answer` jako holým stringem a na `set`/`tuple` v argumentech —
  oba teď opravené).
- Vše jsem dopočítal z uložených dat. Nově vznikl
  `outputs/stage4_epoch2_direct/metrics_official.partial.json` (bez Win Rate, viz
  §6).

---

## 2. Je metodika správně? (review)

**Ano, po opravách je scoring korektní a jednotný.** Klíčové body:

1. **Oficiální scorer je kanonický.** `nestful_official_score.score_items`
   spouští reálný NESTFUL kód (`scorer.calculate_scores`): grounding proměnných,
   korpusové macro-F1 přes `MultiLabelBinarizer`, poziční partial/full po
   zarovnání délek, a Win Rate přes re-exekuci predikovaných callů.

2. **Proč staré `metrics.json` ukazovalo jiná čísla.** Starý `metrics.py` počítal
   F1 jako *průměr per-sample multiset F1*, ne korpusové macro-F1. Proto starý
   soubor pro baseline ReAct ukazoval `f1_func = 0.61`, kdežto oficiální scorer
   dává `0.894`. **Jiná definice, ne chyba** — ale do reportů patří jen
   oficiální číslo. (Toto je přesně ten rozdíl, který audit řešil.)

3. **Opravené bugy, kvůli kterým Direct padal** (oba teď pokryté testy):
   - `gold_answer` jako holý string (např. `022-01-01T00:00:00`) → oficiální
     `json.loads` padalo na „Extra data". Opraveno `_json_field_str`.
   - `set`/`tuple` v argumentech z `ast.literal_eval` → `json.dumps` padalo.
     Opraveno `_json_safe`.
   - Direct větev navíc dřív zahodila hodinu generování při pádu scoringu — teď
     se predikce ukládají hned a scoring je v `try/except`.

4. **Konzistence porovnání.** V `final_outputs` se **všechny** běhy přepočítaly
   stejným scorerem ze surových predikcí (Direct) / trajektorií (ReAct,
   rekonstrukce z `turns[].parsed_call`). Žádné míchání starých a nových definic.

**Jediná výhrada:** Win Rate vyžaduje `signal.SIGALRM` (jen Linux). Pro 4 hlavní
full-eval běhy je už přepočítaný na Linuxu a uložený v `runs/`; chybí jen pro v2
curriculum — viz §6.

---

## 3. Hlavní výsledky (full NESTFUL, 1861 úloh)

| Běh | Model | Paradigma | F1 Func | F1 Param | Partial | Full | Win |
|---|---|---|---|---|---|---|---|
| baseline_react | baseline | ReAct | **0.894** | 0.360 | 0.130 | 0.000 | **0.544** |
| baseline_direct | baseline | Direct | **0.921** | 0.650 | 0.294 | 0.169 | 0.292 |
| stage4e2_react | curriculum | ReAct | **0.153** | 0.131 | 0.050 | 0.001 | 0.325 |
| stage4e2_direct | curriculum | Direct | **0.917** | 0.616 | 0.272 | 0.152 | 0.243 |

(F1 = korpusové macro; Partial/Full = po groundingu; **Win = re-exekuce, nově
přepočteno na Linuxu** a uloženo v `runs/<běh>/metrics_official.json`. Strojově:
`consolidated_metrics.json`.)

**Win Rate odhalil dvě věci, které strukturní metriky nezachytí:**
1. **ReAct má vyšší Win Rate než Direct** (baseline 0.544 vs 0.292), přestože má
   nižší F1/Partial/Full. Win je *execution-based* (stačí správná finální
   hodnota), takže ReAct — který emituje konkrétní mezivýsledky místo `$varN$`
   referencí a iteruje ve více krocích — se častěji „protrefí" ke správné
   odpovědi, i když nereprodukuje gold-trace strukturu. Naopak Direct je
   penalizován za jeden chybný krok v celé sekvenci.
2. **Finetuned je horší ve Win Rate v OBOU paradigmatech** (viz §5).

---

## 4. Anomálie „F1 kolem 90 %" — vysvětlení

**Není to chyba, je to korektní oficiální macro-F1 a má jinou interpretaci než
per-sample/micro skóre.**

- F1 Func je **korpusové macro-F1 přes ~900 názvů funkcí**. Macro průměruje skóre
  přes třídy (názvy funkcí), ne přes vzorky. Slovník je obrovský a řídký — drtivá
  většina názvů funkcí je vzácná a snadno rozlišitelná, takže model je trefí
  „správně" (často tím, že je vůbec nepoužije a ani gold je nemá → třída se
  počítá jako shoda). To macro průměr přirozeně táhne vysoko.
- Proto je `F1 Func ≈ 0.9`, **zatímco Partial/Full jsou nízké** (0.13 / 0.0 u
  ReAct). To není rozpor: F1 Func měří jen *množinu názvů funkcí*, Partial/Full
  měří *přesnou poziční sekvenci s argumenty a groundingem*. Sekvence je mnohem
  těžší.
- **Důsledek pro paper:** F1 Func/Param reportuj jako oficiální macro (to je
  definice NESTFULu), ale nečti je jako „90 % úspěšnost". Skutečnou kvalitu
  sekvencí ukazují Partial/Full a Win Rate. Doporučuji vedle macro F1 uvádět i
  Partial/Full, aby byl obrázek úplný (přesně jak říká `docs/AUDIT.md`).

---

## 5. Anomálie „finetuned je horší než baseline" — vysvětlení

Klíč je **rozlišit paradigma a metriku**. Po dopočtení Win Rate je obrázek
přesnější než původně:

| Metrika | Direct: baseline → finetuned | ReAct: baseline → finetuned |
|---|---|---|
| F1 Func | 0.921 → 0.917 (≈) | 0.894 → 0.153 (**kolaps**) |
| Partial | 0.294 → 0.272 (≈) | 0.130 → 0.050 (pokles) |
| Full | 0.169 → 0.152 (≈) | 0.000 → 0.001 (≈) |
| **Win** | 0.292 → **0.243** (−0.049) | 0.544 → **0.325** (−0.219) |

- **V Direct jsou strukturní metriky (F1/Partial/Full) prakticky shodné**, ale
  **Win Rate klesl** (0.292 → 0.243). Tedy finetuned model generuje sekvence se
  stejnou „strukturní" podobností goldu, ale o něco méně z nich reálně
  doběhne ke správné odpovědi. Drobná, ale reálná degradace — execution-based
  metrika ji odhalí, kdežto F1/Partial ne.

- **V ReAct je propad velký napříč vším:** F1 0.894 → 0.153, Win 0.544 → 0.325.
  Trénink probíhal ve **strict gold-trace** režimu na málo datech a rozhodil
  ReAct chování modelu (drift formátu / dílčí zapomínání). V **Direct** se to
  skoro neprojeví, protože prompt obsahuje ICL příklady a striktní formát, který
  model „navede" zpět; v **ReAct** (volnější multi-turn) se poškození projeví
  naplno — model generuje špatné/chybějící cally, proto spadne i F1 Func.

- **Potvrzení z tréninkových evalů:** `strict_gold_trace_pass` klesá se stupni
  curricula: stage 1 ≈ 0.38 → stage 3 ≈ 0.01 → stage 4 = 0.00 (viz §7). Model se
  multi-call úlohy nikdy nenaučil; pilot scale na to nestačil.

**Shrnutí:** „finetuned horší" je reálné a po dopočtení Win Rate platí v **obou**
paradigmatech — v ReAct dramaticky (kolaps), v Direct mírně (jen na Win Rate).
Příčina je mrňavý pilotní trénink (16 úloh), který poškodil zejména ReAct
chování, ne chyba v měření.

---

## 6. Win Rate — stav a metodika

Win Rate počítá oficiální scorer **re-exekucí** predikovaných callů a používá
`signal.SIGALRM` (timeout), který je **jen na Linuxu**. **Pro 4 hlavní full-eval
běhy je už přepočítaný na Linuxu** a uložený v
`final_outputs/runs/<běh>/metrics_official.json`:

| Běh | Win Rate |
|---|---|
| baseline_react | 0.544 |
| baseline_direct | 0.292 |
| stage4e2_react | 0.325 |
| stage4e2_direct | 0.243 |

`build_report.py` tyto hodnoty **přebírá z `runs/`** i při spuštění na Windows
(kde se přepočítat nedají), takže se v tabulkách neztratí.

**Chybí jen Win Rate pro v2 curriculum** (`v2_stage3_epoch1`, `v2_stage5_epoch2`)
— ty mají `win_rate: null` / `needs_linux`. Dopočítání na podu (bez
re-generování) přímo přes report skript:

```bash
cd /workspace/nestful_mtgrpo_minimal
pip install "jsonlines>=4.0" "scikit-learn>=1.3"
python final_outputs/build_report.py   # na Linuxu zapne Win Rate i pro v2
```

> Pozn. ke „class IndexError/KeyError scorer.py" hláškám během běhu: to **není
> chyba skriptu**. Oficiální `scorer.py` je tiskne u každého vzorku, jehož
> predikovaná sekvence nejde spustit (chybějící `$varN.result$` reference,
> neznámá funkce…), a takový vzorek započítá jako Win = 0. Běh doběhne korektně.

---

## 7. Curriculum trénink (jednoduchý) — progrese

| Stage | Epoch | N | strict_pass | final_pass | off F1 Func | off Partial |
|---|---|---|---|---|---|---|
| 1 | 4 | 609 | 0.378 | 0.470 | 0.787 | 0.262 |
| 2 | 4 | 407 | 0.093 | 0.494 | 0.780 | 0.106 |
| 3 | 3 | 250 | 0.000 | 0.168 | 0.397 | 0.038 |
| 4 | 2 | 173 | 0.000 | 0.087 | 0.473 | 0.043 |

(plná tabulka v `RESULTS.md` / `curriculum_training.csv`.)

- Eval každého stupně běží na NESTFULu filtrovaném na počet callů stupně N+1
  (proto klesá N: 609 → 173).
- **`strict_gold_trace_pass` padá s rostoucí obtížností** (1-call → 4-call+):
  0.38 → 0.09 → 0.00. Model se delší sekvence nenaučil reprodukovat.
- Pozn.: `stage_2/epoch_2` nemá uložené `eval/metrics.json` (jen trajektorie),
  takže ve souhrnu chybí — drobná mezera v datech, ne chyba scoringu.

---

## 7b. Původní curriculum evaluace (`results_v2_20260617`) přepočtená na NESTFUL metriky

Tahle složka je **jiný (starší) eval framework**: multi-turn ReAct rollouty
curriculum checkpointů na **celém** NESTFULu (1861 úloh × 4 rollouty). Původně
reportoval **jen executor-based final-answer accuracy** (`mean_score`), ne paper
metriky. Přepočítal jsem ji oficiálním scorerem z uloženého `predicted_calls`
(beru `rollout_idx == 0`, tj. jedna predikce na úlohu, paper-style).

| Běh | Model | exec. acc. % | F1 Func | F1 Param | Partial | Full | Win |
|---|---|---|---|---|---|---|---|
| v2_baseline | baseline | 67.18 | (jen predikce nejsou uložené) | — | — | — | — |
| v2_stage3_epoch1 | curriculum s3e1 | 70.74 | 0.941 | 0.447 | 0.158 | 0.000 | (Linux) |
| v2_stage5_epoch2 | curriculum s5e2 | 70.14 | 0.951 | 0.453 | 0.163 | 0.000 | (Linux) |

**Co z toho plyne:**
- Na rozdíl od mrňavého „pilot" curricula ze `§5/§7` **tahle curriculum varianta
  reálně pomohla**: executor accuracy 70.7 % / 70.1 % vs. baseline 67.2 %
  (+3.5 pp), a `no_calls` propadlo z ~20 % na ~5–7 % (model si víc volá nástroje).
- F1 Func je opět vysoké (0.94–0.95) — stejný macro-F1 efekt jako v §4.
- **Full = 0.000** u obou: ReAct emituje konkrétní hodnoty místo `$varN.result$`
  referencí, takže přesná gold-trace sekvence (i po groundingu) nesedí nikdy.
  Partial (0.16) zachytí část poziční shody názvů+hodnot. To je očekávané a je to
  hlavní důvod, proč **ReAct čísla nejsou srovnatelná s NESTFUL paper tabulkou**
  (ta je Direct).
- Baseline v této složce má uložený jen accuracy souhrn (žádné `predicted_calls`),
  takže ho oficiálním scorerem přepočítat nelze — ale srovnatelný baseline máš v
  `final_eval_baseline_react` (§3).

> Pozn.: „curriculum trénink", který jsi měl na mysli, jsou tyto checkpointy
> (`results_v2`), ne pilot běh ve `outputs/curriculum/`. Pilot v §7 byl jen
> rychlý sanity běh (16 úloh/epocha). Tabulky a CSV teď obsahují obojí.

Win Rate pro v2 se taktéž počítá jen na Linuxu — stačí spustit
`python final_outputs/build_report.py` na podu (předikce už jsou uložené, nic se
negeneruje znovu) a Win se doplní.

---

## 8. Doporučení

1. **Do paperu reportuj oficiální čísla** z `RESULTS.md` (F1 Func/Param, Partial,
   Full, Win), ne stará interní `metrics.json`.
2. **Win Rate** pro 4 hlavní běhy už je hotový (§3/§6). Zbývá jen dopočítat Win
   pro v2 curriculum spuštěním `build_report.py` na Linuxu (jen re-score,
   negeneruje se znovu).
3. **Pro paper-srovnatelná čísla používej Direct** (= NESTFUL Table 1). ReAct
   čísla nejsou srovnatelná s paperem (jiné paradigma, konkrétní hodnoty místo
   `$varN$` referencí → nízké Partial/Full).
4. **Finetuning:** pilotní curriculum (16 úloh) je příliš malé; pro reálné
   zlepšení je potřeba plný trénink. Aktuálně je finetuned (pilot s4e2) v Direct
   na strukturních metrikách ≈ baseline, ale na Win Rate mírně horší (0.243 vs
   0.292), a v ReAct výrazně horší. Reálné zlepšení ukazuje až větší curriculum
   varianta z `results_v2` (§7b: +3.5 pp executor accuracy).
5. Vedle macro F1 vždy uváděj Partial/Full, ať je jasné, že 0.9 F1 není 90 %
   úspěšnost sekvencí.

---

## Soubory v `final_outputs/`

- `report.html` — **interaktivní vizualizace** (grafy + porovnání s NESTFUL paper modely + vysvětlení metrik). Otevři v prohlížeči.
- `RESULTS.md` — hlavní tabulky (full eval + v2 curriculum + tréninková progrese), oficiální scoring.
- `nestful_full_eval.csv` — strojová tabulka full-eval běhů.
- `curriculum_v2_official.csv` — původní `results_v2` curriculum přepočtené na NESTFUL metriky.
- `curriculum_training.csv` — strojová tabulka tréninkové progrese (pilot).
- `consolidated_metrics.json` — vše strojově (vč. `win_rate_enabled`).
- `runs/<run>/metrics_official.json` — **oficiální metriky per běh** (vše na jednom místě).
- `build_report.py` — skript, který vše přepočítá (spustitelný i na Linuxu pro Win Rate).
- `ANALYSIS.md` — tento dokument.

Příklad struktury po `python final_outputs/build_report.py`:

```
final_outputs/
  RESULTS.md
  nestful_full_eval.csv
  runs/
    baseline_react/metrics_official.json
    baseline_direct/metrics_official.json
    stage4e2_react/metrics_official.json
    stage4e2_direct/metrics_official.json
    v2_stage3_epoch1/metrics_official.json
    v2_stage5_epoch2/metrics_official.json
```
