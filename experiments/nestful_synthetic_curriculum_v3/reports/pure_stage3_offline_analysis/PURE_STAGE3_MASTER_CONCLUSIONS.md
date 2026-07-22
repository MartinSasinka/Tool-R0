# Pure Stage 3 — Master Conclusions (C0 vs E2 test + offline audits)

**Updated:** 2026-07-22  
**Scope:** paired NESTFUL **test** eval C0→E2 (n=1661, no E1) + train reward/credit from overnight logs + checkpoint deltas.

---

## 0. TL;DR

1. **Eval parity C0 vs E2 je OK** — stejných 1661 task IDs, stejný prompt/scorer/decoder, temp=0, top_p=1. Win rate rozdíl **není** artefakt evalu.
2. **E2 vs C0 na testu: −0.42 pp** (53.82 % → 53.40 %), net **−7** úloh (79 gained / 86 lost), McNemar **p≈0.64** (n.s.). Bootstrap 95 % CI pro delta zahrnuje nulu.
3. **Klíčový vzorec „wrong values ↓ ale wrong tool / exec-wrong ↑" se NEOPAKUJE** — wrong values mírně **rostou** (+0.48 pp), wrong tool mírně klesá, exec-wrong flat.
4. **Dominantní regrese je 3-call bucket (−2.2 pp)** a mírný pokles long_chain / linear_dependency. Under-calling zůstává ~60 %.
5. **Train-side (overnight, 326×2):** episode reward dominuje G₀ (R²=0.98), E2 weights ≈ E1 (0.26 % posun) → slabý update, ne přeučení.
6. **E1 test eval chybí** (jen C0/E2). **Overnight checkpointy nemají test eval** — smoke eval ≠ overnight train.

**Data sources:**

| Analýza | Run | Poznámka |
|---|---|---|
| C0/E2 test paired | `pure_stage3_smoke_20260719_213722` | 8-task smoke train, plný test eval |
| Reward/credit | `pure_stage3_2ep_20260719_221918` | 326×2 epoch train logs |
| E1 test | — | **chybí** (`S3_E1_test` neexistuje) |

Detailní reporty: `PURE_STAGE3_C0_E2_PAIRED.md`, `PURE_STAGE3_REWARD_ALIGNMENT.md`, `pure_stage3_task_level_analysis.jsonl`, CSV soubory ve stejné složce.

---

## 1. Eval parity (sekce 1) — C0 vs E2

| Check | Výsledek |
|---|---|
| 1661 stejných task IDs | ✓ |
| Párování podle `sample_id` | ✓ |
| `nestful_test.jsonl` stejný soubor | ✓ (sha `917ce6ec…`) |
| Base model revision | ✓ `cdbee75f17c0…` |
| Prompt (sha `prompt.py`) | ✓ stejný |
| Official scorer (sha) | ✓ stejný |
| Parser + IBM executor pipeline | ✓ stejný `run.py final_eval` |
| temperature=0, top_p=1, 1 rollout, react | ✓ |
| E2 adapter | `bbc8cbb17c3354fa…` (smoke checkpoint) |

**Závěr:** rozdíl −0.42 pp lze připsat checkpointu E2 (smoke), ne eval asymetrii.

E1: dev eval existuje (20 úloh, smoke cap), **test eval ne** — triplet C0/E1/E2 na 1661 nelze.

---

## 2. Párová analýza C0 → E2 (sekce 2, bez E1)

### Headline (official win, n=1661)

| Metrika | C0 | E2 | Δ |
|---|---:|---:|---:|
| Win rate | 0.5382 | 0.5340 | **−0.0042** |
| Function F1 (mean) | 0.621 | 0.623 | +0.002 |
| Parameter F1 (mean) | 0.297 | 0.297 | ~0 |
| First-tool accuracy | 0.571 | 0.574 | +0.003 |
| Full seq accuracy | 0.022 | 0.022 | ~0 |
| Executability | 0.790 | 0.789 | −0.001 |
| Final-answer accuracy | 0.588 | 0.588 | ~0 |
| Under-calling rate | 60.5 % | 60.6 % | +0.06 pp |
| Avg predicted calls | 2.26 | 2.27 | +0.01 |

### Přechody (E1 vynechán)

| Kategorie | počet |
|---|---:|
| stabilní výhry (C0=1, E2=1) | 808 |
| stabilní prohry | 688 |
| **získané po E2** | 79 |
| **ztracené po E2** | 86 |
| net | **−7** |

E1→E2 flip analýza **n/a** (chybí E1 test).

### Failure taxonomy (non-success counts)

| Failure | C0 | E2 | Δ |
|---|---:|---:|---:|
| success | 894 | 887 | −7 |
| correct keys, wrong values | 143 | 151 | **+8** |
| wrong tool | 166 | 164 | −2 |
| executable wrong result | 182 | 182 | 0 |
| no tool call | 110 | 116 | +6 |
| too few calls | 18 | 13 | −5 |
| parse/format | 69 | 66 | −3 |
| wrong keys | 38 | 45 | +7 |

### Klíčová otázka: lokální args ↓, globální strategie ↑?

**NE.** Wrong values **rostou** (+0.48 pp), wrong tool **klesají** (−0.12 pp), exec-wrong **flat**. Objective neukazuje „učí args, ničí tool choice" — spíš **žádný smysluplný posun** + mírné zhoršení v wrong-value a no-call.

### Win rate podle bucketů

| Gold calls | n | C0 | E2 | Δ |
|---|---:|---:|---:|---:|
| 2 | 543 | 0.457 | 0.460 | +0.004 |
| **3** | 363 | **0.620** | **0.598** | **−0.022** |
| 4 | 223 | 0.601 | 0.601 | 0 |
| 5 | 154 | 0.565 | 0.571 | +0.007 |
| 6+ | 378 | 0.529 | 0.524 | −0.005 |

| Motif | n | C0 | E2 | Δ |
|---|---:|---:|---:|---:|
| linear_dependency | 850 | 0.515 | 0.509 | −0.006 |
| long_chain | 532 | 0.539 | 0.538 | −0.002 |
| fan_in | 255 | 0.639 | 0.631 | −0.008 |

---

## 3. First-error analýza (sekce 3, C0 vs E2)

| First error class | C0 | E2 | E2−C0 |
|---|---:|---:|---:|
| too_early_stop (under-call) | 301 | 296 | −5 |
| executable_wrong_result | 123 | 128 | +5 |
| no_tool_call | 110 | 116 | +6 |
| invalid_format | 69 | 66 | −3 |
| wrong_first_tool | 65 | 61 | −4 |
| wrong_keys | 38 | 45 | +7 |
| wrong_value | 20 | 25 | +5 |
| wrong_final_answer | 30 | 30 | 0 |

**Čtení:** formát a wrong-first-tool mírně lepší; under-call mírně lepší; ale **executable-wrong-result a wrong-value/keys mírně horší**. Chyba se neposouvá konzistentně „dál v trajektorii jako zlepšení" — spíš **mírný mix bez net gain**.

Příklad typu „parse fail → validní 3 cally, špatný outcome": existuje u jednotlivých úloh, ale agregovaně to nevynese pozitivní transfer.

Detail: `PURE_STAGE3_FIRST_ERROR_ANALYSIS.csv`, `pure_stage3_task_level_analysis.jsonl`.

---

## 4. Reward alignment (sekce 4, overnight train — 326 groups)

Z `epoch_1/2/train/train_log.jsonl` (plný Stage-3 train, ne smoke 8-task):

| Test | Hodnota |
|---|---|
| mean reward win / loss rollouts | 1.00 / 0.479 |
| pairwise ordering (call-count proxy) | 95.9 % správně |
| **R²(G₀ ~ episode_reward)** | **0.982** |
| corr(G₀, traj_length) | 0.70 |
| too_few reward vs full | 0.217 vs 0.572 |
| lokálně dobrý turn + záporný adv (A0) | 4569 turnů |

**Závěr:** reward ordering v rámci synthetic train vypadá zdravě, ale **kredit je fakticky episode-level**. Nezávislý IBM re-score rolloutů stále chybí.

Detail: `PURE_STAGE3_REWARD_ALIGNMENT.md`.

---

## 5. Offline credit schemes A0–A3 (sekce 5)

| Schéma | dead positions | good&neg adv |
|---|---:|---:|
| **A0 current** (Σr + R_ep) | **24.9 %** | 4569 |
| A1 no episode | 27.1 % | 4335 |
| A2 local r_t | 54.4 % | 2438 |
| A3 local + outcome | 39.1 % | 3790 |

**A0 je nejlepší offline varianta** — naivní „lokálnější" kredit zhorší signál.

---

## 6. Synthetic held-out (sekce 6)

**BLOCKED** — nebyl spuštěn held-out eval C0/E2 na nezávislém synthetic splitu. Viz `PURE_STAGE3_SYNTHETIC_HELDOUT.md`.

Interpretační tabulka z tvého zadání zatím **neaplikovatelná** pro NESTFUL (máme jen test C0 vs E2).

---

## 7. Checkpoint delta (sekce 7)

### Smoke (eval checkpointy)
E2 adapter `bbc8cbb17c3354fa…` — eval z tohoto runu.

### Overnight (plný train)
| Metrika | E1 | E2 | E1→E2 |
|---|---:|---:|---:|
| adapter norm | 36.65 | 36.65 | Δnorm **0.093** |
| cosine(E1,E2) | — | — | **0.999997** |
| relativní posun | — | — | **0.26 %** |

**Interpretace:** E2 téměř nezměnila váhy oproti E1 → **slabý update**, ne přeučení. Smoke E2 test mírně pod C0 je konzistentní s „málo efektivních změn chování".

Behaviorální KL / first-tool entropy přes forward pass **nebyl měřen** (vyžaduje GPU inference).

---

## 8. Rozhodovací strom (aktualizovaný)

```
Eval parity C0/E2?  → ANO (smoke test 1661)
│
├─ Synthetic held-out C0/E2?  → NE (blocked)
│
├─ Reward pairwise ordering (train)?  → spíš ANO (95.9 % proxy)
│     └─ ale G₀ episode-dominated (R²=0.98) → credit assignment issue, ne band ordering
│
├─ Správné early turns + záporný advantage?  → ANO (4569); first-tool na testu +0.3 pp only
│
├─ E1 lepší než E2?  → NEZNÁMO (chybí E1 test)
│
└─ E2 vs C0 na testu?  → −0.42 pp, n.s.; pattern „values↓ tool↑" → NE
       └─ hlavní regrese: 3-call bucket (−2.2 pp)
```

---

## 9. Co je špatně (seřazeno)

1. **Transfer na NESTFUL po Stage-3 GRPO nefunguje** (smoke E2 test −0.42 pp; overnight test nezměřen).
2. **Update je příliš slabý** (overnight E1→E2 weight move 0.26 %; train win +1 pp synthetic).
3. **Under-calling ~60 %** přetrvává; too_early_stop dominuje first-error (~300/1661).
4. **Episode-level kredit** (R²=0.98) — turn-level MT-GRPO je jen jméno.
5. **3-call bucket** je nejzranitelnější na testu (−2.2 pp smoke E2).
6. **Smoke ≠ overnight** — rozhodnutí o plném 326-task běhu vyžaduje eval overnight checkpointů.

---

## 10. Doporučené další kroky (priorita)

1. **Eval overnight S3_E1 + S3_E2** na stejném batchi (C0 + E1 + E2, test 1661) s OOM fixem — jediný chybějící důkaz pro plný train.
2. **Ne třetí epocha Stage 3** — místo toho jedna ablace: silnější update (LR↑ / kl_beta↓) **nebo** SFT warmup proti under-call **nebo** terminal-outcome reward shaping.
3. **Synthetic held-out** split + C0/E2 eval (transfer vs objective).
4. IBM re-score train rolloutů (nezávislý reward-vs-outcome test).

---

## 11. Soubory

| Soubor | Obsah |
|---|---|
| `PURE_STAGE3_C0_E2_PAIRED.md` | parity + metriky + buckety |
| `PURE_STAGE3_FAILURE_TRANSITIONS.csv` | per-task failure přechody |
| `PURE_STAGE3_FIRST_ERROR_ANALYSIS.csv` | first-error matice |
| `PURE_STAGE3_REWARD_ALIGNMENT.md` | train reward/credit |
| `PURE_STAGE3_SYNTHETIC_HELDOUT.md` | blocked status |
| `pure_stage3_task_level_analysis.jsonl` | 1661 řádků C0/E2 |
| `analysis_c0_e2_test.json` | strojově čitelná agregace |
| `analysis.json` | overnight train offline audit |

Skript: `scripts/analysis/pure_stage3_c0_e2_eval_analysis.py` (eval) + `pure_stage3_offline_analysis.py` (train).
