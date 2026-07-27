# Od AutoData / Stage-3 k Pilot2 a Phase-1 canary

Dokument shrnuje, jak jsme postupně zjišťovali, proč syntetická data
„fungovala“ při tréninku, ale nepřinášela spolehlivý transfer na IBM
NESTFUL, a jak z toho vznikl cílený generátor Pilot1 → Pilot2 a malý
Phase-1 GRPO canary.

**Čtenář:** výzkumník znalý LLM a GRPO.  
**Pravidlo:** čísla a tvrzení jsou vázaná na konkrétní artefakty; kde jde
o interpretaci nebo otevřenou otázku, je to výslovně označené.

**Model / metoda (společný kontext):**
`Qwen/Qwen3-4B-Instruct-2507`, multi-turn GRPO (MT-GRPO) s QLoRA,
typicky 8 rolloutů na úlohu, eval při temperature 0. Headline metrika:
oficiální NESTFUL win rate (ne interní train „win“).

---

## Krátká časová osa

| Období (2026) | Co se stalo | Klíčový artefakt |
|---|---|---|
| ≈ 2.–9. 7. | Curriculum v3.1, Stage 1–3 GRPO piloty; audit: žádný same-batch zisk na NESTFUL | `docs/NEXT_TRAINING_DECISION.md`, `DEEP_RESEARCH_BRIEF_STATUS_2026_07_22.md` |
| ≈ 9.–16. 7. | Nestful-like + AutoData/agentic OpenRouter generace; Stage-3 accept ~10 % | `experiments/nestful_synthetic_curriculum_v3/docs/AGENTIC_DATA_GENERATION.md` |
| ≈ 18. 7. | Two-phase GRPO v5 (Stage-2 → Stage-3+replay) | run `two_phase_20260718_192902`, W&B `nestful-v5-curriculum` |
| ≈ 19.–20. 7. | Pure Stage-3 2 epochy; train signál ↑; overnight Nestful eval zpočátku ztracená | `pure_stage3_2ep_20260719_221918`, `WANDB_STATUS_OVERVIEW.md` |
| 22. 7. | Status brief: +0.42 pp C2−C0 n.s.; decision gate | `DEEP_RESEARCH_BRIEF_STATUS_2026_07_22.md` |
| 25. 7. | Forenzní audit: reward dispatch bug + transfer gap; oprava reference detectoru v profilu | `reports/root_cause_forensic/*`, `TARGET_PROFILE_NESTFUL.md` |
| 25.–26. 7. | Targeted Tool Data Factory: Pilot1, pak Pilot2 (engine v2) | `experiments/targeted_tool_data_factory/` |
| 26. 7. | Signal probe P2/P3 → offline reward audit → Phase-1 canary C0→C1 | `outputs/runpod_pilot2/` |
| po canary | NESTFUL-500 +1.0 pp (n.s.); held-out 80× `not_executable` | `C0_VS_C1_PHASE1_REPORT.md` |

---

## 1. Výchozí AutoData / Stage-3 přístup

### Jak se data generovala

Existovaly dvě navazující linie:

1. **Deterministický curriculum** (Stage 1–4 podle délky gold trajektorie /
   počtu tool callů), gold replay 100 %, toy math/string registry.
2. **AutoData / agentic OpenRouter smyčka** (inspirováno Kulikov et al.):
   challenger navrhne úlohu, weak solver selže, strong solver projde,
   deterministický executor je primární verifier. Viz
   `experiments/nestful_synthetic_curriculum_v3/docs/AGENTIC_DATA_GENERATION.md`.

Curriculum podle počtu callů:

| Stage | Typický obsah | Poznámka |
|---|---|---|
| Stage-1 | krátké / jednoduché | v praxi často saturované (dead groups ~1.0) |
| Stage-2 | 2-call | ve v5: 429–496 train-ready úloh |
| Stage-3 | 3-call | 326 train-ready; Round-1 subset 160 |
| Stage-4 | delší | méně dominantní v pozdějších GRPO bězích |

Materiál `training_ready_v5` (Phase C/D briefu): Phase 1 = 429 Stage-2;
Phase 2 = 466 (= 326 Stage-3 + 140 Stage-2 replay).

### Executor, reward, formát

- Train executor: `synthetic` (lokální registry).
- Reward (produkční recept po densifikaci): `execution_aware_v3_2_dense`
  (`lib/reward_v3_2_dense.py`), registry v5.0.2.
- Formát: ReAct + `<tool_call_answer>`, single-call gate.
- Eval: oficiální IBM NESTFUL scorer, T=0, 1 rollout.

Zdroj: `DEEP_RESEARCH_BRIEF_STATUS_2026_07_22.md` §1.

### Proč se zpočátku zdálo, že data jsou vhodná

- 100 % gold replay a syntaktická shoda referencí (Stage-3 syntax audit:
  `NO_MISMATCH` vůči kanonickému Tool-R0 tvaru).
- Po densifikaci rewardu klesly dead groups u Stage-3 (Phase-2 stage3-only
  slice ~17 %; pure Stage-3 E1→E2: 27.9 % → 15.0 % —
  `WANDB_STATUS_OVERVIEW.md`).
- Agentic smyčka běžela s ~10 % accept a contamination gates.
- Train metriky (mean reward, train win_rate, loss) se hýbaly „správným“
  směrem.

To stačilo k dojmu, že problém je spíš v rewardu / počtu epoch než
ve struktuře dat.

---

## 2. Co bylo na Stage-3 dobré

Potvrzená pozitiva (nejsou totéž co transfer):

| Vlastnost | Evidence |
|---|---|
| Syntaktická validita / reference syntax | Stage-3 Nestful syntax audit: `NO_MISMATCH` (`stage3_nestful_syntax_audit.md`) |
| Deterministické oracle + gold replay | design curriculumu; 100 % replay v early pilots |
| Vícekrokové `$var` reference | každá Stage-2/3 úloha má referenční argumenty (`DATA_TRANSFER_AUDIT.md`) |
| Část groups má reward/action variance | pure Stage-3 dead groups klesly na 15 %; dense reward snižoval sparse GRPO |

**Důležité rozlišení:** *variance uvnitř GRPO group* znamená jen, že
optimizer vidí relativní výhody mezi 8 rollouty. Neznamená, že se učí
NESTFUL tool katalog, NESTFUL argumentové typy, ani že se zlepší oficiální
win rate. Stage-3 to ukázal empiricky: silnější train update bez transferu
(viz §3).

---

## 3. Jak jsme zjistili problémy Stage-3

### Trénink vs NESTFUL

| Experiment | Train-side | NESTFUL | Zdroj |
|---|---|---|---|
| Two-phase v5 `two_phase_20260718_192902` | Phase-1 dead 78.3 %; Phase-2 dead 31.0 % | C0 53.52 % → C2 53.94 % (**+0.42 pp**, n.s.) | `WANDB_STATUS_OVERVIEW.md`, brief |
| Pure Stage-3 smoke (8 tasks) | — | full test **−0.42 pp** vs C0 | brief §4 |
| Pure Stage-3 2ep `…221918` | loss 0.217→0.063; dead 28 %→15 % | oficiální **−1.14 pp** (n=1661) | `UPDATE_STRENGTH_AUDIT.md`, `ROOT_CAUSE_REPORT.md` |

Interpretace (forenzní H2, STRONGLY_SUPPORTED): **in-domain learning ANO,
transfer NE**.

### Dead groups a collapse signálu

- Stage-2 (Phase 1): **78.3 %** dead groups — většina GRPO kroků bez
  užitečného gradientu (`WANDB_STATUS_OVERVIEW.md`).
- Stage-2 replay v Phase 2 stále **62.9 %** dead.
- Agentic Stage-3: dominantní reject `all_same_reward` — generátor našel
  úlohy, kde je politika homogenní, ne diskriminativní (brief §4).

*Dead group* = všech 8 rolloutů má stejný (nebo téměř stejný) reward →
group-normalized advantage ≈ 0.

### Tool / executor / reward / eval mismatch

Forenzní audit 2026-07-25 (`ROOT_CAUSE_REPORT.md`):

1. **CONFIRMED — reward dispatch bug:** Round-1 A0–A4 všechny trénovaly s
   `execution_aware_v3_2_dense` (env `setdefault` + hook preference). Cross-arm
   „ablace“ byla invalidní; win spread 53.0–57.4 % na n=500 = replikační šum.
2. **STRONGLY_SUPPORTED — transfer gap** (viz tabulka níže).
3. **STRONGLY_SUPPORTED — slabý Round-1 update** (~30 steps @ LR 3e-7).

### Profile audit Stage-3 vs NESTFUL

Z `DATA_TRANSFER_AUDIT.md` / `a06_data_transfer.json` a
`profile_match_pilot2.json` (stage3_old řádek):

| Vlastnost | Stage-3 / Round-1 | NESTFUL |
|---|---|---|
| Call counts | 100 % 3-call (Stage-3) | 2–6+ (dev: 33 % 2-call, 22 % 6+) |
| Gold-tool Jaccard vs NESTFUL-500 | **0.003–0.006** | — |
| Offered tools | ~163 | 816 (diag-500) / až 3850 (full) |
| Dominant tools | verbose aritmetika (`round_to_decimals`, …) | holá `add/multiply/…` |
| Classifier two-sample AUC vs NESTFUL | **0.728** (`profile_match_*.json` stage3_old) | ideál 0.5 |
| JSD call_bucket | **0.585** | ideál 0 |
| JSD answer_type | **0.152** | ideál 0 |

**Proč stejný počet tool callů ≠ stejné rozložení úloh:** Stage-3 může mít
„3 cally“ a přesto (a) jiné tool jména, (b) jiný graf závislostí (motif),
(c) jiné typy odpovědí, (d) jinou šířku offered katalogu, (e) jiný poměr
reference vs literál. Curriculum podle délky trajektorie tyto osy
nesrovnává.

### Reference detector bug (oprava profilu)

Forenzní a06 tvrdil, že NESTFUL má reference jen v 15–24 % úloh.
`docs/TARGET_PROFILE_NESTFUL.md` (factory) to opravuje: detektor
nerozpoznal tvar `$var_1` (underscore). Po opravě má **100 %** NESTFUL
dev/test řádků reference; per-arg reference share ~40 %. Quoty factory
používají opravená čísla. Starší forenzní texty stále citují 15–24 % —
to je **nekonzistence k opravě** (viz závěr dokumentu).

---

## 4. Hlavní změna v uvažování

Kvalitní dataset pro GRPO→NESTFUL musí současně splnit:

| Požadavek | Význam | Stage-3 typicky |
|---|---|---|
| **(a) Correctness / executability** | gold je executor-ověřený, replayovatelný | ano |
| **(b) Podobnost s NESTFUL profilem** | call/motif/answer/args/offered | ne (AUC 0.73) |
| **(c) Model-relative learnability** | u C0 existují mixed groups (ne all-success / all-fail) | částečně u S3, špatně u S2 |
| **(d) Správný reward ordering** | lepší trajektorie → vyšší reward (bez terminálních inverzí) | po opravách sledováno |
| **(e) Transfer na benchmark** | oficiální win / paired gained-lost | opakovaně selhalo |

Tři vrstvy, které se nesmí zaměňovat:

```
strukturální validita  →  „úloha je správně spočítaná“
        ↓
GRPO signál            →  „model má na úloze směs úspěch/neúspěch;
                          reward řadí trajektorie“
        ↓
transfer               →  „po update se zlepší NESTFUL, ne jen train loss“
```

Factory design (`DESIGN.md`, `SCIENTIFIC_RATIONALE.md`): program-first,
target-conditioned quotas z NESTFUL **dev n=200**, failure-driven cells,
structural held-out před jakýmkoli claimem o transferu.

---

## 5. Pilot1

**Co to je:** první freeze Targeted Tool Data Factory — program-first
generátor, V1–V6 validace, greedy selection k NESTFUL profilu.
320 úloh (160/80/80). Reporty: `docs/DATA_CARD_PILOT.md`,
`PILOT_REPORT.md`, srovnání v `PILOT1_VS_PILOT2.md`.

### Co se zlepšilo proti Stage-3

Z `PILOT2_REPORT.md` §8 (řádek pilot1 vs stage3_old):

| Metrika vs NESTFUL | Stage-3 (old) | Pilot1 |
|---|---:|---:|
| JSD call_bucket | 0.585 | 0.003 |
| JSD answer_type | 0.152 | 0.114 |
| two-sample AUC | 0.728 | 0.550 |

Call-count mix už sleduje NESTFUL (ne 100 % 3-call). Oracle zůstává
executor-only.

### Co zůstalo špatně

| Problém | Pilot1 | NESTFUL / cíl | Zdroj |
|---|---:|---|---|
| Float odpovědi | **97.2 %** | 77.0 % | `PILOT1_VS_PILOT2.md` |
| Fan-in | 21.9 % | ~43 % | totéž |
| Linear grafy | 70.3 % | — | totéž |
| Hard distractory na každé úloze | **100 %** | — | totéž |
| Největší šablona | 5.3 % | ≤5 % gate | totéž |
| Typed answer mix (int/bool/string/list) | téměř chybí | přítomný | totéž |
| LLM parafráze | 0 % | — | totéž |

Pilot1 tedy vyřešil hrubý call-count mismatch, ale stále učil „skoro vždy
float + lineární řetězec + maximálně adversarial distractory“. Proto nebyl
finální.

---

## 6. Pilot2 / engine v2

Freeze: **320 úloh** (train 160 / held-out 80 / reserve 80), seed
`20260726`, verdict **READY** (`docs/PILOT2_REPORT.md`).

### Technické změny (stručně)

| Prvek | Účel |
|---|---|
| Unit-aware typed DAG | odpovíď a mezivýsledky mají typy/jednotky |
| Deterministický factory executor + `trainer_adapter` | stejný oracle při generaci i GRPO |
| Primitiva int/bool/string/list/numeric-string | přiblížení answer-type mixu NESTFUL |
| Motivy fan-in + branch-aggregate | méně čistě lineárních grafů |
| Unit propagation / plausibility | odmítá nesmyslné složení jednotek |
| Shortcut / contamination / dedup / leakage / replay gates (V1–V6) | tvrdá validita |
| OpenRouter parafráze | **jen surface**; po přijetí znovu V-validace; neovlivní oracle |

Pipeline (zjednodušeně):

```
NESTFUL dev profil (n=200)
        ↓
generation cells (kvóty × failure profile)
        ↓
typed DAG → execute → oracle → surface (+ optional paraphrase)
        ↓
V1–V6 validate
        ↓
select (deficit matching + profile JSD/AUC)
        ↓
split 160/80/80 (family leakage audit)
        ↓
export GRPO + nestful + canonical  (+ SHA256 freeze)
```

Kód: `experiments/targeted_tool_data_factory/src/targeted_tool_data/`,
bundle `runpod_bundle_pilot2/`.

### Porovnání distribucí (celý selected set)

| Vlastnost | Pilot1 | Pilot2 | NESTFUL dev |
|---|---:|---:|---:|
| Float odpovědi | 97.2 % | 74.1 % | 77.0 % |
| Fan-in | 21.9 % | 37.5 % | ~43 % |
| Největší šablona | 5.3 % | 4.1 % | — |
| Hard distractor tasky | 100 % | 80.3 % | — |
| Answer-type JSD | 0.114 | 0.0026 | ideál 0 |
| Classifier AUC | 0.550 | 0.525 | ideál 0.5 |
| LLM-paraphrase share | 0 % | 57.2 % | — |

Zdroje: `PILOT1_VS_PILOT2.md`, `PILOT2_REPORT.md`,
`outputs/selected/profile_match_pilot{1,2}.json`.

Pilot2 je tedy první dataset, který je současně (a) executor-validní a
(b) blízko NESTFUL v answer/call/motif metrikách. To stále **neprokazuje**
learnability ani transfer — jen nutnou podmínku (b).

---

## 7. Jak jsme ověřili učitelnost

### Proč generátor sám dead groups nepozná

Validace V1–V6 kontroluje správnost programu a distribuci poolu. Neví, zda
konkrétní C0 checkpoint na úloze uspěje ve 0/4, 2/4 nebo 4/4 roloutech.
To je **model-relative** vlastnost → potřeba inference probe.

### Signal probe (P2 / P3)

Skript: `runpod_bundle_pilot2/run_signal_probe_4gpu.sh`  
Report: `outputs/runpod_pilot2/signal_probe_from_zip/signal_probe/SIGNAL_PROBE_REPORT.md`

| Fáze | Rozsah | Dead | Terminal-mixed | All-success dead | All-failure dead |
|---|---|---:|---:|---:|---:|
| P2 | 160×4 | 70.0 % | 15.0 % | 53.1 % | 16.9 % |
| P3 | 64×8 (boundary) | 50.0 % | 35.9 % | 34.4 % | 15.6 % |

Slovníček:

- **all-success dead** — všechny rollouty uspějí → úloha pro C0 příliš snadná.
- **all-failure dead** — všechny selžou → příliš těžká / špatný surface.
- **terminal-mixed** — směs success/fail → GRPO má terminální signál.
- **process-only-mixed** — terminál stejný, liší se process komponenta.

Zjištění: velká část train setu je pro C0 trivialně řešitelná (P2:
63.7 % groups 4/4 success); informativní masa je menší, ale neprázdná.

### Reward-ordering audit

Počáteční probe verdikt: **STOP** — 32 Pareto inverzí / rate 0.0339 >
tolerance 0.02 (`SIGNAL_PROBE_REPORT.md`).

Offline audit na stejných rolloutách
(`phase1_canary_from_zip/offline_reward_audit/OFFLINE_REWARD_AUDIT.md`)
pak rozlišil:

- **terminální / clear-dominance inverze** (hard gate) → u `A4_current`: **0**;
- **success–success disagreement** (dvě validní cesty, jiný gold prefix) →
  **není** gate failure (150 párů u A4_current).

Vybráno: **`A4_current`** / policy `reward_ablation_A4_GATED_VERIFIABLE`,
hard_gate **PASS** (`SELECTED_REWARD_VARIANT.json`).

### 80-task Phase-1 subset

`recommended_phase1_train.jsonl` — **80** úloh:

| reason | n |
|---|---:|
| terminal_mixed | 23 |
| process_only_mixed | 7 |
| easy_anchor | 10 |
| hard_all_failure_control | 10 |
| structural_topup | 30 |

Verifikace: gold replay 100 %, leakage 0, max major-feature JSD 0.038
(`PHASE1_SUBSET_VERIFICATION.md`, verdict PASS).

---

## 8. Phase-1 GRPO canary

Skript: `runpod_bundle_pilot2/run_phase1_canary_4gpu.sh`  
Train: `run_phase1_train.py` — 80 úloh × 8 rolloutů, grad_accum 4 →
**~20 optimizer steps**, GPU0 learner / GPU1–3 rollouts, `--skip-eval`
(eval až `run_eval_all.py`).

### NESTFUL diagnostic-500 (hlavní číslo)

Z `C0_VS_C1_PHASE1_REPORT.md` / `.json`:

| | C0 | C1 | Δ |
|---|---:|---:|---:|
| Win rate | **48.8 %** | **49.8 %** | **+1.0 pp** |
| Gained (C1 win, C0 lose) | — | — | **21** |
| Lost (C0 win, C1 lose) | — | — | **16** |
| Bootstrap 95 % CI Δ | — | — | **[−1.4, +3.4] pp** |
| Exact McNemar (37 discordant) | — | — | **p = 0.5114** (n.s.) |

**Jak to číst:** +1 pp a 21:16 gained:lost je **první pozitivní
transferový trend** na factory datech po sérii Stage-3 nul/regresí. Není to
statisticky průkazný důkaz; CI obsahuje nulu. Full D1-160 ani NESTFUL-1661
neběžely.

### Structural held-out 80 — chyba evaluace

| | C0 | C1 |
|---|---:|---:|
| Win rate | 0 % | 0 % |
| Failure class | `not_executable` ×80 | `not_executable` ×80 |

Held-out má měřit in-domain učení factory úloh přes **synthetic** executor
(`run_eval_all.py` nastavuje `executor.mode=synthetic` +
`SYNTHETIC_TOOLS_DIR`). Výsledek 80/80 `not_executable` znamená, že tato
větev evaluace v daném běhu **neběžela korektně** (registry / tool resolution
/ klasifikace), ne že model „nic neumí“. NESTFUL-500 ve stejném reportu
běžel (244→249 success) — problém je specifický pro factory held-out cestu.

Důsledek: z canary **nelze** tvrdit nic o in-domain held-out gain; transferový
signál stojí jen na NESTFUL-500.

---

## 9. Co jsme se naučili

1. **Stage-3 mohl mít reward variance a klesající loss, a přesto učit jinou
   funkcionalitu** než NESTFUL (tool Jaccard ≈ 0, fixní 3-call, jiný katalog).
2. **Profilová podobnost nestačí bez learnability** — Pilot2 má výborné JSD,
   ale P2 dead 70 % (hlavně all-success).
3. **Learnability nestačí bez NESTFUL coverage** — informativní subset musí
   zachovat straty (ověřeno v Phase-1 JSD gates).
4. **Více epoch stejných Stage-3 dat není řešení** — pure_stage3 2ep:
   NESTFUL −1.14 pp.
5. **Reward experiment bez dispatch guardu je neplatný** — Round-1 A0–A4.
6. **Data se mají vybírat podle NESTFUL kvót i podle rollout signálu**
   (cells + probe → `recommended_phase1_train.jsonl`).

---

## 10. Současný doporučený postup

Odvozeno z factory runbooků, Phase-1 výsledků a forenzního
`NEXT_DECISION.md` (transfer gap zůstává hlavní otevřenou výzkumnou vrstvou):

1. Zachovat informativní Pilot2 (a později Pilot3) train tasky s mixed
   signálem; neodkládat jen „validní“ úlohy.
2. Generovat primárně **deficitní** NESTFUL-like cells (call/motif/answer /
   distractor mezery), ne uniformní dump.
3. Pipeline:
   ```
   CPU generate → deterministic V1–V6 → freeze+SHA256
        → RunPod probe (P2/P3) → offline reward audit
        → constrained selection (Phase-1 / full)
        → GRPO z čistého C0
   ```
4. Nový větší dataset trénovat **z C0**, ne navazovat na Stage-3 adapter.
5. Eval: C0 vs C1 (vs C2) na **opraveném** factory held-out *a* NESTFUL-500;
   paired bootstrap + McNemar.
6. Full NESTFUL-1661 až po reprodukci pozitivního trendu na dalším seedu /
   větším budgetu — ne jako první metr po každém canary.

Pilot3 (1000 = 600/200/200) je škálování stejné logiky; tento dokument ho
neanalyzuje do hloubky (viz `experiments/targeted_tool_data_factory/docs/PILOT3_REPORT.md`).

---

## Tabulka: Stage-3 vs Pilot1 vs Pilot2

| Vlastnost | Stage-3 (old) | Pilot1 | Pilot2 | NESTFUL dev |
|---|---:|---:|---:|---:|
| Typický call mix | 100 % 3-call | NESTFUL-like | NESTFUL-like | 2–6+ |
| Float answer | (numeric-only) | 97.2 % | 74.1 % | 77.0 % |
| Fan-in | nízko / jiné motify | 21.9 % | 37.5 % | ~43 % |
| Hard distractor tasks | nekontrolované | 100 % | 80.3 % | — |
| JSD answer_type | 0.152 | 0.114 | 0.0026 | 0 |
| JSD call_bucket | 0.585 | 0.003 | 0.003 | 0 |
| two-sample AUC | 0.728 | 0.550 | 0.525 | 0.5 |
| Gold-tool Jaccard vs NESTFUL-500 | ≤0.006 | (factory registry) | (factory registry) | — |
| Train→NESTFUL (best measured) | +0.42 pp n.s. / −1.14 pp | (ne Phase-1 canary) | C0→C1 **+1.0 pp** n.s. na 500 | — |

Pozn.: Stage-3 „best measured“ míchá two-phase (+0.42) a pure_stage3 (−1.14);
Pilot2 řádek je Phase-1 canary na 80 úlohách, ne full 160.

---

## Tabulka: problém → diagnostika → změna → výsledek

| Problém | Diagnostika | Změna | Výsledek |
|---|---|---|---|
| Falešný pokrok z train win | Internal win > official o ~5–7 pp | Headline = official Nestful @ T=0, same-batch | Hygiene v briefu §1 |
| Sparse GRPO / dead groups | Stage-2 78 % dead | Dense reward `v3_2_dense`; skip Stage-1; later probe filtering | S3 dead ↓; transfer stále slabý |
| Invalid reward ablace | Všechny army stejný `resolved_policy` | Dispatch fix + guards + tests | Round-1 zneplatněn; canary A1/A4 reálné |
| Transfer gap (tools/calls/catalog) | Jaccard ≤0.006; AUC 0.73 | Target-conditioned factory | Pilot2 AUC 0.525, JSD answer 0.0026 |
| Float-only / linear Pilot1 | 97 % float, 70 % linear | Engine v2 typed + fan-in | Pilot2 74 % float, 37.5 % fan-in |
| All-success pool | P2 53 % all-success dead | Phase-1 mixed selection (80) | Canary na informativním subsetu |
| Pareto reward inversions | 32 inversí v probe | Offline audit: oddělit alt. cesty | `A4_current` PASS (0 term. inv.) |
| Held-out 0 % win | 80× `not_executable` | (oprava eval path — open) | In-domain metrika z canary nepoužitelná |
| +1 pp na 500 | CI [−1.4,+3.4], McNemar n.s. | Neškálovat slepě na D1-160 | Trend, ne důkaz |

---

## Potvrzená zjištění vs interpretace vs otevřené otázky

### Potvrzeno (artefaktově)

- Two-phase C2−C0 = +0.42 pp n.s. na n=1661.
- Pure Stage-3 2ep: train signál ↑, NESTFUL −1.14 pp (n=1661) dle forenzního
  auditu (eval trajektorie později nalezeny v run dir).
- Round-1 reward dispatch bug (CONFIRMED).
- Pilot1→Pilot2 posun float/fan-in/JSD dle reportů.
- Probe P2/P3 dead/mixed rates; Phase-1 = 80 úloh; offline `A4_current` PASS.
- Canary NESTFUL-500: 48.8 → 49.8 %, 21 gained / 16 lost, McNemar n.s.
- Held-out canary: 80/80 `not_executable`.

### Interpretace (plausible, ne kauzálně uzavřené)

- Hlavní dlouhodobá překážka Stage-3 byla transfer gap, ne „málo epoch“.
- +1 pp canary je první pozitivní transferový *signál* factory dat, ale
  může být šum (CI obsahuje 0).
- Held-out selhání = eval/registry path, ne absenci učení.

### Otevřené otázky

- Reprodukuje se +1 pp na jiném seedu / větším budgetu (full 160 nebo Pilot3 600)?
- Jaká je skutečná in-domain held-out křivka po opravě executor path?
- Je A4 process komponenta nutná, nebo stačí outcome-only na větším mixed setu?
- Kolik all-success úloh lze nechat jako kotvy vs kolik škodí?

---

## Nekonzistence v reportech (k opravě)

1. **Reference rate NESTFUL:** `DATA_TRANSFER_AUDIT.md` / `ROOT_CAUSE_REPORT.md`
   stále uvádějí 15–24 %; `TARGET_PROFILE_NESTFUL.md` dokumentuje bug detektoru
   a 100 % tasků s referencemi. Forenzní H2 text by měl citovat opravu.
2. **C0 baseline na „500“:** forensic 57.0 % vs Phase-1 canary 48.8 % —
   různé běhy/setupy; nesmí se míchat do jedné křivky bez explicitního
   same-batch párování.
3. **Overnight Nestful „unknown“ vs −1.14 pp:** `WANDB_STATUS_OVERVIEW.md`
   (22. 7.) říká, že E1/E2 eval nebyl zalogován; forenzní audit (25. 7.)
   později našel lokální `final_eval_trajectories` a reportuje −1.14 pp.
   Oba texty jsou historicky pravdivé v čase napsání — doplnit křížový odkaz.
4. **Signal probe VERDICT STOP vs Phase-1 start:** počáteční Pareto gate
   FAIL; offline terminal-class audit PASS. README probe by měl odkazovat na
   offline audit jako rozhodující pro canary.
5. **`PILOT2_VS_PILOT3.md` call table:** řádek `6+` = 0 % u Pilot2/3 při
   současném výpisu 6/7/8 zvlášť — matoucí agregace vůči NESTFUL 22 % `6+`.

---

## Příloha: klíčové cesty

| Role | Cesta |
|---|---|
| Stage-3 / AutoData docs | `experiments/nestful_synthetic_curriculum_v3/docs/` |
| Deep research brief | `experiments/nestful_synthetic_curriculum_v3/reports/DEEP_RESEARCH_BRIEF_STATUS_2026_07_22.md` |
| W&B overview | `…/reports/WANDB_STATUS_OVERVIEW.md` |
| Forenzní audit | `…/reports/root_cause_forensic/` |
| Factory design/method | `experiments/targeted_tool_data_factory/docs/{DESIGN,METHOD,SCIENTIFIC_RATIONALE}.md` |
| Pilot1 vs Pilot2 | `…/docs/PILOT1_VS_PILOT2.md`, `PILOT2_REPORT.md` |
| NESTFUL profile (+ ref fix) | `…/docs/TARGET_PROFILE_NESTFUL.md` |
| RunPod Pilot2 | `…/runpod_bundle_pilot2/` |
| Probe report | `…/outputs/runpod_pilot2/signal_probe_from_zip/signal_probe/SIGNAL_PROBE_REPORT.md` |
| Offline reward | `…/outputs/runpod_pilot2/phase1_canary_from_zip/offline_reward_audit/` |
| C0 vs C1 report | `…/outputs/runpod_pilot2/phase1_canary_from_zip/C0_VS_C1_PHASE1_REPORT.md` |
