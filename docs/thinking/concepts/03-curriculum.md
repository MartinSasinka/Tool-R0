# Curriculum — proč učíme po stupních

---

## Jako bych to říkal kamarádovi

Nehodíš žáka rovnou na maturitu. Nejdřív jednoduché příklady, pak těžší.

**Curriculum** = stejné u modelu:

| Stage | Trénink (synthetic) | Eval (NESTFUL) |
|---|---|---|
| 1 | úlohy s **1** tool callem | úlohy se **2** cally |
| 2 | 2 cally | 3 cally |
| … | … | … |
| 4 | 4 cally | **5** callů |

Všimni si: **vždy testujeme o 1 krok těžší**, než na čem trénujeme. To měří **generalizaci**.

---

## Analogie: žebřík

Každý stupeň = naučíš se chodit o jeden stupeň výš. Zkouška je vždy **o stupeň výš**, než kde stojíš.

Když na zkoušce spadneš, možná jsi šel na další stupeň moc brzy.

---

## Kdy postoupíme na další stage?

Tři způsoby (kterýkoliv):

1. **exec_pass ≥ baseline[n+1] + 0.12** — jsme aspoň o 12 p.b. lepší než base model na těžší úloze
2. **Plateau** — 2 epochy bez zlepšení (ale min 3 epochy celkem)
3. **max_epochs** — pojistka, abychom nezůstali věčně

---

## Replay buffer (15 %)

Do tréninku stage 4 přimícháme **15 % dat ze starších stages**.

Proč? Aby model **nezapomněl** jednoduché věci, když řeší těžké.

Jako v mateře: pořád procvičuješ násobilku, i když už řešíš rovnice.

---

## Co fungovalo / nefungovalo

| Stage | Výsledek | Interpretace |
|---|---|---|
| 1→2 | ✅ nad baseline | Curriculum na krátkých řetězcích dává smysl |
| 3 | ✅ nad baseline | Generalizace +1 call ještě jde |
| 4 | ❌ pod baseline | Skok na 5-call eval je moc / formát / data |

---

## Díra v mém chápání

- [x] Synthetic vs NESTFUL — odpověď 5–6 níže
- [x] Threshold 0.12 — odpověď 10 (12 p.b., ne „12 %“)
- [x] Teacher forcing — odpověď 21
- [ ] Jak často stage 4+ končí přes plateau vs threshold (z manifestu)

---

## Tvoje slova (doplň!)

Piš vlastními slovy — klidně hrubě, špatně, neúplně. Cíl je zjistit, co víš a co ne.

```
1. Co je curriculum v našem projektu jednou větou?
→ Postupně trénujeme na synthetic datech s rostoucím počtem callů (1→6),
  po každé stage evaluujeme na NESTFUL o **1 call těžším** a rozhodujeme, jestli postoupit dál.

2. Proč stage 1 trénuje na 1-call, ale evaluje na 2-call NESTFUL?
→ Měříme **generalizaci**: naučil se model na jednodušším, umí to o krok těžší?
  Kdyby eval = train, viděli bychom jen memorování, ne přenos.

3. Vzorec „train N → eval N+1“ pro stage 4?
→ **Train:** `epoch_4_4call.jsonl` (synthetic, 4 cally v gold trajektorii).
  **Eval:** NESTFUL úlohy s **5** cally (`val_n_calls = stage + 1`).

4. Analogie?
→ Ve škole: nejdřív sčítání, pak rovnice. U nás: nejdřív 1 call, pak zkouška na 2.
  Tvoje analogie sedí.

5. Co je synthetic data a čím se liší od NESTFUL?
→ LLM (DeepSeek přes OpenRouter) vygeneruje kandidáty → **verifikace** (IBM exec, filtrování)
  → `filtered_toolr0_synthetic/epoch_X_Ycall.jsonl`.
  **Liší se od NESTFUL:** jiné příběhy/prompty, jiná distribuce úloh, IBM nástroje z registry
  (ne přímo NESTFUL benchmark set). Formát je stejný (XML tool_r0), ale **doména není 1:1**.

6. Proč filtr/verifikace ≠ „umí NESTFUL“?
→ Filtr garantuje, že **gold** trajektorie je spustitelná a konzistentní.
  Model se ale učí **generovat** vlastní výstupy — to verifikace neřeší.
  Navíc eval = jiné úlohy, multi-turn bez gold prefixu, delší řetězce.

7. Co dělá `run_curriculum_training.py`?
→ Orchestrátor celého běhu: baseline eval → loop stages (train epoch → eval → gating)
  → manifest checkpointů → finální eval. Ne jen „jeden cyklus“, ale **celá pipeline**.

8. Jeden epoch cyklus stage 4 (hrubě)?
1. Vzorkne se replay (15 % řádků ze stages 1–3)
2. GRPO train na `epoch_4_4call.jsonl` + replay, navázání na `previous_adapter`
3. Eval checkpointu na NESTFUL **5-call**
4. Gating: postup na stage 5 pokud `exec_pass ≥ threshold` NEBO plateau NEBO max_epochs
5. Jinak další epoch **ve stejné stage 4** (ne hned další stage)

9. Baseline eval — kdy, proč, jaké n_calls?
→ **Jednou na začátku** běhu, base Qwen **bez LoRA**, na NESTFUL groups **2..7**.
  Slouží jako referenční čára: `threshold = baseline[val_n_calls] + 0.12`.

10. Threshold pro postup ze stage 3 na 4?
→ Oprava: při **dokončení stage 3** eval běží na **4-call** (ne 5!).
  `threshold = baseline["4"] + 0.12`. Pokud baseline 4-call ≈ 8 % → threshold ≈ **20 %**.
  Postoupíme, když `exec_pass ≥ 20 %` (ne „o 12 % víc než baseline“ jako relativní procenta).

11. Tři důvody postupu — který je ideální?
→ **Ideální:** `threshold` — exec_pass dosáhl baseline + 12 p.b.
  **Plateau:** 2 epochy bez zlepšení oproti nejlepšímu z předchozích → jdeme dál i pod threshold.
  **Pojistka:** `max_epochs` — stage skončí, i když se nic nezlepšilo.

12. `plateau_patience: 2`?
→ Potřebujeme min. **3 epochy** historie. Pokud aktuální exec_pass ≤ nejlepší
  z předchozích 2 epoch → advance s důvodem `plateau`.
  Stage 4 často skončí takhle — pod threshold, ale „už se to nezlepšuje“.

13. `max_epochs_per_stage`?
→ Tvoje intuice OK: pojistka proti věčnému tréninku a přeučení.
  Stage 4 má max 5 epoch — pak postoupíme i bez thresholdu.

14. Replay buffer?
→ Oprava: ne „z předchozích **epoch**“, ale z předchozích **stages** (1..s−1).
  Zapíná se od stage 2+, bere `epoch_1_1call.jsonl` … `epoch_{s-1}_{s-1}call.jsonl`,
  vzorkuje **15 %** velikosti aktuálního datasetu, přimíchá do trainu této epochy.

15. Proč replay při těžších úlohách?
→ Správně: **catastrophic forgetting** — model by zapomněl krátké řetězce.
  Bez replay by po stage 4 mohl hůř na 1–2 call věcech.

16. `previous_adapter`?
→ Cesta k **nejlepšímu LoRA checkpointu** z předchozí stage.
  Stage 3 načte adapter ze stage 2 (`PeftModel.from_pretrained`) a **pokračuje v učení**
  stejného adapteru — ne začíná od nuly.

17. Který checkpoint jde dál?
→ Oprava: ukládáme každou epochu, ale dál jde **nejlepší podle exec_pass** v dané stage
  (`best_epoch_adapter`), ne poslední. Viz manifest `best_path`.

18. Proč rostou `max_completion_length` / `max_prompt_length`?
→ Delší řetězce = delší konverzace (víc turnů v promptu) + delší generovaný call.
  Stage 1: 1536 / default; stage 5: prompt 2560, completion 3072.
  „Více přemýšlení“ je vedlejší — hlavně **víc tokenů na více kroků**.

19. Funguje curriculum vždy?
→ Ne. Stage 1–3 nad baseline = krátké řetězce + generalizace +1 funguje.
  Stage 4 pod baseline = limit délky / formát / GRPO signál. Curriculum není záruka.

20. Proč stage 4 selže i když logika dává smysl?
→ Tvoje body OK (+ těžší řetězení). Doplň:
  - všech 8 rollouts ≈ stejný reward → advantage ≈ 0 (GRPO)
  - parse_fail ~85 % v eval (formát, ne logika)
  - teacher forcing v train vs autoregresivní eval

21. Teacher forcing + stage 4?
→ V train promptu jsou **předchozí turny vždy správně** (gold prefix).
  GRPO hodnotí jen **další** call. Na stage 4 se učí „doplnit krok 4“, když kroky 1–3
  jsou perfektní. V eval musí **sám** zvládnout kroky 1–4 — chyba v turn 1 rozbije zbytek.

22. Eval bez +1 (stejný počet callů jako train)?
→ Správně: výsledky by byly **vyšší** (méně generalizace), ale neviděli bychom,
  jestli model zvládne o krok těžší úlohu — ztratíme smysl curriculum designu.

23. Kdy zastavit curriculum dřív?
→ Signály pro stop / přehodnocení:
  - exec_pass **pod baseline** 2+ stages za sebou
  - `train/reward` stagnuje, vysoká parse_fail v eval
  - gating vždy `plateau` nebo `max_epochs`, nikdy `threshold`
  - stage 5 běží, ale stage 4 best < baseline → další stage pravděpodobně nepomůže

24. Co ještě nevím?
→ Otevřené: podíl promptů s advantage ≈ 0 na stage 4+, jestli replay 15 % stačí,
  jestli threshold 0.12 je moc nízký/vysoký pro 5–6 call. „Vím všechno“ určitě ne — a to je OK.
```
