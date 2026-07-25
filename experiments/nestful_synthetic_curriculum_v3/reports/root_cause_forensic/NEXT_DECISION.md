# NEXT_DECISION

## 1. Co je nyní nejpravděpodobnější hlavní problém?

Dvě vrstvy:
- **Experimentální vrstva (vyřešena):** Round-1 reward ablace byla invalidní
  — všech 5 armů trénovalo se stejným rewardem (dispatch bug, CONFIRMED).
- **Výzkumná vrstva (otevřená):** i korektní trénink na tomto syntetickém
  curriculu pravděpodobně NESTFUL nezvedne kvůli **transfer gapu**
  (tool overlap Jaccard ≤0.006, inverze reference-usage 100 % vs 15–24 %,
  fixní 3-call vs 2–6, katalog 163 vs 816–3850 tools) — pure_stage3 to
  empiricky ukázal: train signál ↑, NESTFUL −1.14 pp.

## 2. Co je prokazatelně špatně?

- Reward dispatch: `setdefault(REWARD_POLICY)` v `two_phase_train_session.py`
  + env-preferující hook v `run.py` přepsaly konfigurované ablation politiky
  (důkaz: 160/160 řádků train logu per arm, identické rewardy na
  hash-matched completions, max_abs_diff=0.0).
- `config_used.json` deklaroval politiku, která neběžela.
- Offline audit používal: raw-weight cosine (init artefakt), chybný success
  threshold 0.92 (správně 0.90) a hardcoded závěr „rewards distinct".
- `shared_C0_eval_500` běžel na 1861 řádcích; jeho win rate 0.5513 je n=1861
  metrika, ne baseline na 500 (na 500 je C0 = 57.0 %).

## 3. Co už bylo vyvráceno?

- „A0/A4 adaptery téměř identické" (delta-to-init cosine 0.835; raw ~1 byl artefakt).
- „A3 process reward kauzálně škodí" (identický reward všude; 53.0–57.4 % je replikační šum jedné konfigurace).
- „A4 nejlepší díky gated rewardu" (tatáž vada + chybný threshold proxy metriky).
- „Credit assignment je bugnutý" (97–99 % good-turn-negativ je pod group mean; 96–100 % execwrong-pozitiv v all-failure groups; dead-flag parity 100 % — očekávané GRPO chování). Zbytek je pouze netestovatelný, ne podezřelý.
- Záměna checkpointů, chybné párování evalu, chybný scorer (vše přesně reprodukováno z raw dat).

## 4. Jaké opravy byly provedeny?

Viz FIXES_APPLIED.md: oprava dispatch priority v `run.py`, env pre-empce +
tvrdý `assert_dispatched_policy` guard v runneru, dispatch check + verdikt
short-circuit v offline auditu, delta-to-init adapter metrika (float64,
trace identita), oprava success thresholdu 0.92→0.90 + přeznačení na proxy,
parita `_turn_returns`, encoding fix testů. 17 nových regresních testů, vše
zelené.

## 5. Jeden nejmenší další experiment, který rozliší zbývající hypotézy

**Instrumentovaný canary „dispatch-fix + trajectory logging": 24 tasků × 8
rolloutů × ~10 optimizer steps, pouze 2 army (A1_OUTCOME_ONLY a
A4_GATED_VERIFIABLE), stejný seed a train subset jako Round-1.**

Proč právě tento: (a) ověří opravu dispatche na GPU stacku dřív, než se
spálí plný re-run; (b) s persistencí per-turn trajektorií (parsed calls,
predикáty, rewardy per turn) odblokuje matched-prefix credit audit, který je
dnes UNTESTABLE; (c) změří, zda různé rewardy vytvářejí různé advantages
po group normalizaci na REÁLNÝCH skupinách (dnes jen non-kauzální re-score).

## 6. Kolik tasků, rolloutů, kroků a GPU hodin vyžaduje?

- 24 tasků × 8 rolloutů = 192 epizod/arm; ~10 optimizer steps/arm; 2 army.
- Odhad dle Round-1 (160 groups ≈ 6 h/arm na 1 GPU): 24 groups ≈ **~1 GPU-h
  na arm, ~2 GPU-h celkem** (bez evalu; eval na 500 se u canary nespouští).

## 7. Jaké přesné metriky rozhodnou výsledek?

1. `reward_policy_resolved` v každém řádku train logu == konfigurovaná
   politika armu (guard nesmí spadnout).
2. Na hash-matched completions mezi army: `reward_pearson_hash_matched < 1.0`
   a `max_abs_diff > 0` (dnes 1.0 / 0.0).
3. Podíl rewardů vysvětlitelných intended terminálními skaláry ≥ 0.95
   (dnes 0.00–0.05).
4. Cross-arm advantage cosine na hash-matched skupinách < 0.9 (různé rewardy
   ⇒ různé advantages po normalizaci).
5. Z nových trajektorií: matched-prefix audit — podíl good-prefix turnů
   s negativní advantage v epizodách NAD group mean (skutečný test credit
   assignmentu; očekávání ≈ 0).

## 8. Stop conditions

- Guard abortuje nebo metrika 1/2/3 selže → dispatch oprava je neúplná;
  STOP, žádný další trénink, zpět do kódu.
- Metriky 1–4 projdou → dispatch fix potvrzen; rozhodnutí o plném Round-2
  je SAMOSTATNÉ a musí nejdřív adresovat transfer gap (bod 9).
- Metrika 5 > 0.1 → otevřít credit assignment znovu, s daty.

## 9. Co se nyní nemá spouštět?

- **Plný Round-2 reward ablation re-run (5 armů × 1 epocha × eval 500)** —
  dokud canary nepotvrdí fix; a i potom zvážit, že H2 (transfer gap)
  predikuje nulový NESTFUL efekt bez ohledu na reward design.
- Další epochy `pure_stage3` (2 epochy již NESTFUL zhoršily o 1.14 pp).
- Jakékoli závěry z Round-1 cross-arm srovnání (statisticky jde o 5 replik
  téže konfigurace).
- Nový reward design, SFT, generování dat — mimo mandát a bez důkazní opory,
  dokud canary neproběhne.
