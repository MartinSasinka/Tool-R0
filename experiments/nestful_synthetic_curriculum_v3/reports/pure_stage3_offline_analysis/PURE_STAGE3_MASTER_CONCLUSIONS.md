# Pure Stage 3 Overnight — Finální analýza (326×2 GRPO, test n=1661)

**Run:** `outputs/runs/pure_stage3_2ep_20260719_221918`  
**Updated:** 2026-07-22  
**Eval:** C0 / E1 / E2 na `nestful_test.jsonl` (1661), parity OK, stejný batch  
**Adapter hashes:** E1 `7419b731…` · E2 `92aca741…`

---

## 0. Verdikt jednou větou

**Plný pure Stage-3 GRPO (326 úloh × 2 epochy) na NESTFUL testu monotonně zhoršuje model** (−0.78 pp E1, −1.14 pp E2 vs C0), přičemž se potvrzuje vzorec **„wrong values mírně dolů, wrong tool a executable-wrong-result nahoru"** — objective učí lokální argumentovou podobnost, ale zhoršuje globální strategii řešení. Druhá epocha navíc **41 úloh** získaných po E1 znovu ztrácí. Problém není formát ani eval; je to **transfer + slabý/shpatně směřovaný update + episode-level kredit**.

---

## 1. Headline — official win (test, n=1661)

| Arm | Win rate | Δ vs C0 | Paired net | McNemar p |
|-----|---------:|--------:|-----------:|----------:|
| **C0** | **54.43 %** | — | — | — |
| **E1** | 53.64 % | **−0.78 pp** | −13 (75↑ / 88↓) | 0.35 |
| **E2** | 53.28 % | **−1.14 pp** | −19 (74↑ / 93↓) | 0.16 |
| E2 vs E1 | — | −0.36 pp | −6 (79↑ / 85↓) | 0.70 |

Bootstrap 95 % CI pro E2−C0: **[−2.65 pp, +0.36 pp]** — statisticky hraniční, ale směr konzistentně negativní across E1→E2.

**Důležité:** nový C0 test (54.43 %) je vyšší než starý dev-only odhad (~53.8 % ze smoke) — srovnání vždy within-batch C0/E1/E2.

---

## 2. Eval parity — OK

Stejných 1661 IDs, stejný prompt/scorer/decoder (temp=0, top_p=1, react, IBM executor). Rozdíly ve win rate **nejsou** eval artefakt.

---

## 3. Párové přechody (C0 → E1 → E2)

| Kategorie | počet | význam |
|-----------|------:|--------|
| stabilní výhry | 772 | |
| stabilní prohry | 642 | |
| ztracené po E1 (C0 win → E1 loss) | 49 | |
| získané E1, ztracené E2 | **41** | ⚠️ druhá epocha undo |
| získané E1, ponechané E2 | 34 | |
| ztracené E1, obnovené E2 | 39 | flip oběma směry |
| získané až po E2 (bez E1) | 40 | |

**E1→E2 flip je symetrický** (41 vs 39) — druhá epocha není čisté přeučení, ale **41 konkrétních úloh** E1 zisku E2 zruší. E2 je celkově horší než E1 (−0.36 pp).

---

## 4. Klíčová otázka — failure shift na testu

| Failure (podíl z 1661) | C0 | E1 | E2 | E2−C0 |
|------------------------|---:|---:|---:|------:|
| **correct keys, wrong values** | 8.97 % | 8.73 % | **8.61 %** | **−0.36 pp** ✓ |
| **wrong tool** | 9.51 % | 9.75 % | **10.05 %** | **+0.54 pp** ✗ |
| **executable wrong result** | 10.17 % | 11.32 % | **11.68 %** | **+1.51 pp** ✗ |
| no tool call | 6.56 % | 6.86 % | 6.50 % | ~0 |
| under-calling (rate) | 59.7 % | 60.4 % | **60.7 %** | +1.0 pp ✗ |
| parse/format | 4.33 % | 3.79 % | 4.03 % | −0.30 pp |

### **Vzorec potvrzen: ANO**

Wrong argument values mírně klesají, ale **wrong tool roste** a **executable-wrong-result roste o +1.5 pp** — model častěji produkuje vykonatelné trajektorie vedoucí ke špatnému výsledku. To je přesně signatura **lokálního reward hackingu** bez správné globální strategie.

---

## 5. Win rate podle bucketů a motivů

### Gold call count

| Bucket | n | C0 | E1 | E2 | E2−C0 |
|--------|--:|---:|---:|---:|------:|
| **2-call** | 543 | 45.9 % | 45.5 % | **44.2 %** | **−1.7 pp** |
| **3-call** | 363 | 60.1 % | 59.2 % | 59.0 % | −1.1 pp |
| 4-call | 223 | 61.0 % | 61.0 % | 59.2 % | −1.8 pp |
| 5-call | 154 | 59.7 % | 55.8 % | 61.7 % | +2.0 pp (noise?) |
| 6+ | 378 | 55.3 % | 54.8 % | 54.0 % | −1.3 pp |

Regrese je **rozptýlená across délky**, ne jen 3-call. Stage-3 train (vše 3-call synthetic) **nepomáhá** 3-call na NESTFUL a mírně škodí 2-call.

### Motif

| Motif | C0 | E1 | E2 |
|-------|---:|---:|---:|
| linear_dependency (850) | 51.3 % | 50.4 % | **49.5 %** |
| long_chain (532) | 56.6 % | 55.1 % | 56.2 % |
| fan_in (255) | 62.7 % | **64.3 %** | 62.4 % |

Největší pokles: **linear_dependency** (−1.8 pp E2 vs C0).

---

## 6. Train-side (326×2 synthetic) — proč NESTFUL klesá

| Metrika | Epoch 1 | Epoch 2 | NESTFUL test |
|---------|--------:|--------:|-------------|
| synthetic group win | 28.3 % | 29.4 % | 54.4 % → 53.3 % |
| dead_group rate | 27.9 % | 15.0 % | — |
| mean train reward | 0.626 | 0.634 | — |
| E1→E2 weight move | — | **0.26 %** | — |
| R²(G₀ ~ episode_reward) | — | **0.982** | — |

**Interpretace:**

1. **Train reward roste, NESTFUL klesá** → buď overfitting na synthetic, nebo reward–outcome misalignment na transferu (ne „málo kroku").
2. **Episode reward dominuje G₀** (98 %) — turn-level MT-GRPO je fakticky episode-level; správné prefixy v neúspěšných rolloutech dostávají záporný advantage (4569 turnů v auditu).
3. **Update je extrémně malý** (E2−E1 cosine 0.999997), ale stačí k **škodlivému směru** na NESTFUL — LR 3e-7 + kl_beta 0.15 možná nutí model k minimálním změnám, které optimalizují synthetic dense reward, ne IBM win.
4. Na train rolloutech wrong_tool/execfail **mírně klesá** — train metriky **klamou** vůči NESTFUL.

---

## 7. Co NENÍ problém

- **Formát / parse** — mírně lepší nebo flat (−0.3 pp)
- **Eval pipeline** — parity OK, 1661×3 kompletní
- **Reference syntax** — NO_MISMATCH (audit)
- **Dead groups** — E2 train 15 % (lepší než two-phase Stage2 78 %), ale NESTFUL stejně klesá

---

## 8. Root cause (seřazeno)

| # | Příčina | Důkaz | Confidence |
|---|---------|-------|------------|
| 1 | **Synthetic→NESTFUL transfer gap** | train win ~29 % vs NESTFUL ~54 %; 326 toy 3-call ≠ NESTFUL mix 2–6+ | vysoká |
| 2 | **Reward učí lokální args, ne globální strategii** | values↓, exec-wrong↑ +1.5 pp, wrong tool↑ | vysoká |
| 3 | **Episode-level kredit** | R²(G₀)=0.98; A2/A3 horší než A0 offline | vysoká |
| 4 | **Under-calling přetrvává/roste** | ~60 % na testu; dominantní first-error | vysoká |
| 5 | **Slabý update špatným směrem** | weight Δ 0.26 %, ale −1.14 pp NESTFUL | střední |
| 6 | **E2 horší než E1** | −0.36 pp; 41× E1 gain → E2 loss | střední |
| 7 | Přeučení druhou epochou | **vyvráceno** jako hlavní — flip je symetrický | nízká |

---

## 9. Decision gate — splněno

> „Pokud win ≤ C0 + ~1 pp → stop more Stage3 epochs, change objective"

**E2 = C0 − 1.14 pp → STOP.** Nepokračovat třetí epochou stejného receptu.

---

## 10. Co dělat dál (prioritizované)

### A. Okamžitě STOP
- Další epochy pure Stage-3 GRPO se stejným reward/config
- Format reward / další Stage-3 epochy

### B. Největší EV — 1 izolovaná ablace (vyber jednu)

**B1. Terminal-outcome reward shaping** (doporučeno první)  
- Rozšířit gap: `fully_correct` vs `executable_wrong_final`  
- Cíl: snížit exec-wrong-result (+1.5 pp) bez format rewardu  
- Abort: pokud exec-wrong na testu roste nebo win ≤ C0 − 1 pp

**B2. SFT warmup → GRPO**  
- SFT na gold continuations Stage-3 (proti under-calling ~60 %)  
- Pak krátký GRPO s B1 rewardem  
- Abort: pokud SFT alone nezvedne 2-call bucket

**B3. Silnější / jiný update**  
- LR 1e-6 nebo kl_beta 0.05, **1 epocha**, stejná data  
- Měřit weight Δ i NESTFUL — pokud Δ velký a win pořád klesá → problém je data/reward, ne LR

**B4. Delší synthetic curriculum**  
- Přidat Stage 4/5 (4–5 call) + Stage-2 replay pro 2-call  
- Bez změny rewardu — test jestli transfer gap

**B5. On-policy NESTFUL-dev mini-loop** (50 held-out, leak-safe)  
- Probe transfer vs synthetic-only

### C. Credit assignment (druhá vlna, ne první páka)
- Offline A0 je nejlepší varianta — **nelokalizovat** na A2  
- Experiment: terminal-outcome band + optional `lambda_episode` ablace (0.5 vs 1.0)  
- Nejdřív B1, pak credit

### D. Infra (hotovo, držet)
- Eval teardown + `VLLM_GPU_UTIL=0.70`  
- Artifact sync před kill podu

---

## 11. Srovnání se smoke eval (8-task train)

| | Smoke E2−C0 | **Overnight E2−C0** |
|--|------------:|--------------------:|
| Win delta | −0.42 pp | **−1.14 pp** |
| Pattern values↓ tool↑ | ne | **ano** |
| Train tasks | 8 | **326×2** |

Plný train **škodí víc** než smoke — smoke nebyl reprezentativní varování, ale směr stejný.

---

## 12. Soubory

| Soubor | Obsah |
|--------|--------|
| `PURE_STAGE3_C0_E1_E2_PAIRED.md` | Párová test analýza |
| `analysis_c0_e1_e2_test_overnight.json` | JSON agregace |
| `pure_stage3_task_level_analysis.jsonl` | 1661 řádků C0/E1/E2 |
| `PURE_STAGE3_FAILURE_TRANSITIONS.csv` | Failure přechody |
| `PURE_STAGE3_REWARD_ALIGNMENT.md` | Train credit audit |
| `analysis.json` | Train logs + checkpoint delta |

Skripty: `scripts/analysis/pure_stage3_c0_e1_e2_eval_analysis.py`, `pure_stage3_offline_analysis.py`

---

## 13. One-liner pro deep research / supervizora

Pure Stage-3 GRPO na 326 synthetic 3-call úlohách **monotonně degraduje** NESTFUL test (−1.14 pp E2 vs C0) s potvrzeným failure shiftem **wrong values ↓, executable-wrong-result ↑ (+1.5 pp)**; train synthetic reward roste, transfer ne; episode-level kredit (R²=0.98) a under-calling (~60 %) jsou hlavní páky — **stop same recipe**, next: terminal-outcome reward ablace nebo SFT→GRPO, ne třetí epochu.
