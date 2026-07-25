# ROOT_CAUSE_REPORT — forenzní audit 2026-07-25

Otázka: proč GRPO trénink mění syntetické metriky, ale nezvyšuje NESTFUL
Win Rate. Všechny důkazy ověřeny v raw JSON/JSONL/kódu; odkazy na detailní
reporty v této složce a strojová data v `analysis/`.

| Hypotéza | Stav | Důkazy pro | Důkazy proti | Confidence |
|----------|------|------------|---------------|------------|
| H1: Reward dispatch bug — všechny army A0–A4 trénovaly s `execution_aware_v3_2_dense`; Round-1 „ablace" nikdy nevariovala reward | CONFIRMED | Header + všech 160 řádků train_log/arm: `reward_policy_resolved=execution_aware_v3_2_dense`; console `training reward = execution_aware_v3_2_dense` hned po `[override] …reward_ablation_<ARM>`; hash-matched completions napříč 10 páry armů mají identické rewardy (max_abs_diff=0.0); logované rewardy nevysvětlitelné intended skaláry (0–5 %); mechanismus reprodukován v kódu (`setdefault(REWARD_POLICY)` + hook preference) | žádné | 0.99 |
| H2: Transfer gap synthetic→NESTFUL (tooly, argumenty, délky, šířka katalogu) | STRONGLY_SUPPORTED | Gold-tool Jaccard 0.003–0.006; reference args 100 % train vs 15–24 % NESTFUL; str-encoded argumenty jen v NESTFUL; train fixně 3-call vs NESTFUL 2–6; offered tools 163 vs 816–3850; pure_stage3 zlepšil train signál (loss 0.217→0.063), NESTFUL −1.14 pp | kauzalitu prokáže až trénink na NESTFUL-like datech | 0.90 |
| H3: Round-1 update příliš slabý na behaviorální změnu | STRONGLY_SUPPORTED | 27–31 optimizer steps @ LR 3e-7; KL mean 2.5–3.8e-5; clipping ~0; dead groups 21–31 %; A0 identický s C0 na 65 % úloh, A0 s A4 na 71 %; ‖ΔW‖_F ~0.009 | ΔW nenulové a vzájemně odlišné (cosine 0.835–0.885); silnější update (pure_stage3) stejně NESTFUL nezvedl | 0.85 |
| H4: A3 process reward kauzálně škodí NESTFUL (53.0 %) | REJECTED (jako kauzální tvrzení) | — | A3 trénoval s IDENTICKÝM rewardem jako ostatní (H1); rozdíly 53.0–57.4 % jsou replikační šum téže konfigurace (stejný seed, data, reward; jen rollout nedeterminismus) | 0.95 |
| H5: A0/A4 adaptery „téměř identické" (raw cosine ~1) → žádný trénink | REJECTED | — | raw cosine je artefakt sdílené seedované lora_A inicializace; delta-to-init (B@A, float64) cosine A0×A4 = 0.835, rel_l2 = 0.575; všech 5 checkpointů odlišné SHA256 | 0.98 |
| H6: Credit assignment bug (good-turn negativní adv, execwrong pozitivní adv) | UNLIKELY; matched-prefix část UNTESTABLE_WITH_CURRENT_LOGS | good-turn neg. adv 27–33 % | 97–99 % těchto případů je v epizodách pod group mean (očekávané u outcome-based GRPO); 96–100 % execwrong-pozitivních je v all-failure groups (GRPO tam MUSÍ někoho ohodnotit kladně); dead-flag parity 100 %; žádný double-counting (G0 episode share ~0.19) | 0.80 (že bug není) |
| H7: Záměna checkpointů / C0 není base / resume-LoRA chyba | REJECTED | — | provenance SHA256 všech FINAL odlišné a = adapter_epoch_1; manifesty ukazují správné zdrojové runy; C0 win rate reprodukován z raw | 0.97 |
| H8: Chyba v eval pipeline (parser/scorer/párování/výpočet) | REJECTED | — | 500/500 párovaných ID; win rate, gained/lost přesně reprodukovány z raw trajektorií u všech armů; scorer akceptuje kratší validní cesty (254/500) | 0.97 |
| H9: v3_2_dense reward je fakticky trace-imitation signál škodící alternativním cestám | UNTESTABLE_WITH_CURRENT_LOGS (kauzálně) | dense process komponenty jsou gold-aware; ablace navržená k testu tohoto nikdy neběžela (H1) | eval scorer alternativní cesty akceptuje; counterfactual re-score ukazuje, že intended politiky by signál výrazně změnily | — |
| H10: „A4 nejvyšší train-side success" = efekt gated rewardu | REJECTED (interpretace) | — | metrika byla reward-threshold proxy s chybným prahem 0.92 (správně 0.90 = fully_correct dolní mez) a rewardy byly identické → rozdíl je šum rolloutů | 0.95 |
| H11: NESTFUL 2-call slabší kvůli tréninku | REJECTED | — | 2-call slabost existuje už u C0 (45 % vs 62 % 3-call); army ji mírně zlepšují (0.46–0.49) | 0.9 |

## Hlavní příčiny (max 3, dle pravděpodobnosti)

1. **CONFIRMED — Reward dispatch bug.** Round-1 reward ablace je invalidní
   experiment: všech 5 armů trénovalo se stejným rewardem
   (`execution_aware_v3_2_dense`). Veškeré cross-arm závěry (včetně „A3
   škodí" a „A4 nejlepší train-side") jsou artefakty replikačního šumu.
   Opraveno + regresní testy.
2. **STRONGLY_SUPPORTED — Transfer gap.** Synthetic curriculum sdílí
   s NESTFUL 1–2 tooly ze ~181, invertuje reference-usage, netrénuje výběr
   z velkého katalogu ani variabilní délku. Model se syntetickou úlohu učil
   (pure_stage3: loss 0.217→0.063, KL ↑), ale NESTFUL šel o 1.14 pp dolů —
   in-domain learning bez transferu.
3. **STRONGLY_SUPPORTED — Slabý update v Round-1.** ~30 steps @ 3e-7,
   KL ~3e-5: výsledné politiky jsou z 65–71 % bitově identické s C0/mezi
   sebou. Round-1 nemohl vyprodukovat měřitelný NESTFUL efekt ani při
   správném dispatchi.

Poznámka k prioritě: H1 je chyba experimentu, H2+H3 jsou kandidáti na
odpověď na výzkumnou otázku. Po opravě H1 zbývá rozlišit H2 vs H3 — viz
NEXT_DECISION.md.
