# Eval metriky — co čísla znamenají

---

## Jako bych to říkal kamarádovi

Po zkoušce dostaneš víc známek než jen „prošel / neprošel“. Každá říká něco jiného.

---

## Hlavní metriky

### `exec_pass_rate` ⭐ (ta důležitá)

**Úloha kompletně správně** — všechny tool cally OK + finální odpověď sedí.

> Jako by celý recept dopadl a dort chutnal.

### `parse_fail_rate`

Model něco vygeneroval, ale **počítač to nepřečetl** jako tool call.

> Student napsal odpověď, ale písmenko nelze přečíst.

### `tool_call_acc` / `partial_score`

Kolik **jednotlivých kroků** bylo správně (jméno nástroje + správné klíče argumentů).

> Polovina receptu správně, ale finální dort špatně.

### `clipped_frac`

Kolik generací **narazilo na limit tokenů** (`max_new_tokens`).

> Student psal tak dlouho, že učitel utrhl papír — možná ještě nebyl u konce.

### `avg_turns_completed`

Kolik kroků model v průměru stihl v multi-turn evalu.

---

## Jak číst baseline (base model, bez tréninku)

| n_calls | exec_pass | parse_fail | clipped (při 2048 tokenech) |
|---|---|---|---|
| 2 | 50 % | 22 % | 7.6 % |
| 5 | 5 % | **83 %** | 10.5 % |

**Poznámka:** Na těžkých úlohách většina nepadá na „špatný výpočet“, ale na **formát**.

---

## Častá chyba v interpretaci

| Špatně | Správně |
|---|---|
| „Clip 22 % = potřebujeme větší okno“ | U stage 4 je clip vysoký, protože model **dlouho žvaní** před parse-failem |
| „tool_acc 30 % = model je blízko“ | Bez exec_pass to nestačí — kroky mohou být částečně OK, ale celek špatně |
| „Partial eval 150 úloh = nic“ | Stačí na trend v průběhu tréninku; finální report = full eval |

---

## Kde čísla najdeš

```
curricullum/training/results/
├── baseline_nestful.json      # base model, full
├── stage_3_epoch1_val.json    # po epochě, 150 úloh
└── stage_*_val.failures.jsonl # konkrétní chyby
```

---

## Cvičení (praktické)

Otevři jeden řádek v `stage_4_epoch1_val.failures.jsonl` a zkus odpovědět na otázky 22–24 níže.

---

## Díra v mém chápání

- [x] `exec_pass` = 3 podmínky najednou — odpověď 2
- [x] `tool_call_acc` vs `partial_score` — odpověď 6 (v kódu stejné)
- [x] Co je `n_calls` — odpověď 14 (počet callů v úloze, ne rollouts)
- [x] Čtení failures.jsonl — odpovědi 22–24
- [ ] Jak často „správné číslo“ = náhoda při špatném řetězci (failure taxonomy)

---

## Tvoje slova (doplň!)

Piš vlastními slovy — klidně hrubě, špatně, neúplně. Cíl je zjistit, co víš a co ne.

```
1. Co je `exec_pass_rate`?
→ Podíl úloh, kde model **celou multi-turn trajektorii** zvládl správně (ne jen „něco se spustilo“).
  Hlavní metrika curriculum, protože gating a best checkpoint jedou podle ní.

2. Kdy `exec_pass = true`? (všechny 3 podmínky)
→ `parse_fail = false` (každý očekávaný turn má parsovatelný call)
  AND `correct_calls == total_calls` (správný nástroj + správné **klíče** argumentů na každém kroku)
  AND `final_result` sedí s `gold_answer`

3. Co je `parse_fail`?
→ V nějakém turnu, kde ještě čekáme gold call, parser (`parse_tool_calls`) **nevytáhl** call.
  Typicky: samotný text/„thinking“, chybí `<tool_call_answer>`, špatný JSON uvnitř tagu, oříznutý výstup.
  Ne jen „špatná závorka“ — u nás jde hlavně o **XML tag + JSON uvnitř**.

4. `parse_fail = true` i když model „něco zavolal“?
→ **Ano.** Failure může mít `pred_calls` z prvních turnů — parse_fail nastane na **pozdějším** turnu,
  nebo parser vůbec nic nevzal z posledního výstupu. Částečné cally ≠ úspěch.

5. Co měří `tool_call_acc`?
→ Průměr přes úlohy: `correct_calls / total_calls`.
  „Správný krok“ = stejné **jméno nástroje** + stejná **jména klíčů** v arguments (hodnoty se nekontrolují!).

6. Rozdíl `tool_call_acc` vs `partial_score`?
→ V `evaluate_nestful_stage.py` se počítají **identicky** (oba = correct_calls/total_calls).
  V logu jsou dvě jména, ale číslo je stejné.

7. Vysoký `tool_call_acc`, nízký `exec_pass`?
→ Ano. Např. 3/5 kroků má správný nástroj+klíče, ale špatné **hodnoty** v args → exec spadne.
  Nebo parse_fail na kroku 4, nebo `final_result` ≠ gold_answer i při částečně OK krocích.

8. `exec_pass = 0`, ale `final_result` = `gold_answer`?
→ **Ano** — viz první failure stage 4: špatný řetězec (`divide`×3 místo `multiply`…),
  ale náhodou stejné finální číslo. Exec_pass vyžaduje i **správné kroky**, ne jen shodu čísla.

9. `clipped_frac`?
→ Podíl generací, které dosáhly `max_new_tokens` (useknutí).
  Problém: když **validní** call by byl až za limitem. U nás často OK ~10 %;
  vysoký clip + vysoký parse_fail = model **žvaní** (thinking), ne že potřebujeme 2× větší okno.

10. Baseline 2-call?
→ Base model na krátkých úlohách zvládne ~polovinu celých trajektorií; ~22 % úloh spadne už na formát.

11. Baseline 5-call — bottleneck?
→ Správně: **formát** (parse_fail ~83 %). Většina úloh se ani nedostane k pořádné exekuci.

12. Proč vysoký clip ≠ větší okno (stage 4)?
→ Model generuje dlouhý thinking/text **před** tím, než vůbec vypíše call — a stejně failne parse.
  Zvětšení okna nepomůže, dokud model neemituje `<tool_call_answer>` včas.

13. `avg_turns_completed`?
→ Průměrný počet turnů, kde model **něco** vygeneroval jako call (parsovatelných i ne).
  U 5-call evalu **nízké** = model končí brzy (parse_fail / málo kroků) — nedojede řetězec.

14. Co je `n_calls` v evalu?
→ Oprava: **ne rollouts**. Filtr na NESTFUL úlohy s přesně **N** gold tool cally v trajektorii.
  Stage 4 eval → `n_calls=5` (úlohy s 5 kroky).

15. Jedna NESTFUL úloha v evalu (hrubě)?
1. System prompt + user (otázka + tools)
2. Model → `<tool_call_answer>` (nebo fail)
3. IBM registry spustí call, výsledek jako user zpráva
4. Opakovat pro každý gold krok
5. Na konci porovnat `final_result` vs `gold_answer` → exec_pass

16. Kde metriky — epoch vs final?
→ `curricullum/training/results/stage_X_epochY_val.json` (často **150 úloh** — trend)
  vs `final_n*.json` / full eval (`max_tasks_final: null` — report pro paper/schůzku).
  Wandb = průběh; JSON = zdroj pravdy.

17. `*.failures.jsonl`?
→ Ukázky **neúspěšných** úloh (ne všechny). Číst první: `parse_fail`, pak `gold_calls` vs `pred_calls`,
  pak `gold_answer` vs `final_result`.

18. `delta` v logu?
→ `exec_pass − baseline` v **procentních bodech** (např. +0.03 = +3 p.b.).
  Kladné = nad baseline, záporné = horší než netrénovaný model.

19. Eval vs train reward?
→ Podobné ingredience (formát, call, exec), ale:
  - train = **jeden turn**, gold prefix, soft call_match
  - eval = **celý řetězec**, autoregresivně, tvrdější podmínka exec_pass
  Proto reward může růst a exec_pass ne.

20. Po naučení XML formátu?
→ **`parse_fail_rate` klesá**, pak **`exec_pass_rate` roste**. `tool_call_acc` může růst později.

21. Stage 4: exec 4 %, parse 85 %, tool_acc 15 %?
→ Priorita: **formát / parse_fail** (85 %). Dokud neumí emitovat call, zbytek je vedlejší.

22. [Cvičení] První řádek — parse_fail?
→ **Ano**, `"parse_fail": true`. Parser v nějakém turnu nevzal validní call
  (u 5 kroků model stihl jen 3 `divide` — řetězec nedokončen/špatný).

23. [Cvičení] gold vs pred?
→ Gold: `multiply, multiply, multiply, add, divide` (5 kroků aritmetiky)
  Pred: `divide, divide, divide` (3 kroky, **špatné nástroje od začátku**).
  Rozbití: už **turn 1** — měl `multiply`, dal `divide`. Navíc jen 3/5 kroků.

24. [Cvičení] exec_pass = 0 při shodném final_result?
→ Exec_pass potřebuje: žádný parse_fail + **všechny** kroky správně + finální odpověď.
  Tady: parse_fail=true, correct_calls < 5, špatné nástroje — číslo sedí **náhodou** špatného řetězce.

25. Co ještě nevím?
→ Normalní na tomto místě. Důležité teď: exec_pass je **AND** tří podmínek, parse_fail je brána,
  failures.jsonl je učebnice — jeden řádek ti řekne víc než jedno procento v tabulce.
```
