# Velký obrázek — o čem celý projekt je

> **Cíl složky:** Pochopit *proč* to děláme, ne jen *jak* spustit skript.

---

## Jednou větou

Učíme malý jazykový model (**Qwen3-4B**), aby uměl **postupně volat nástroje** (kalkulačka, vyhledávání, …) v správném pořadí a dostal správnou finální odpověď — a měříme to na benchmarku **NESTFUL**.

---

## Vysvětli to jako dítěti

Představ si úlohu:

> „Vezmi číslo 10, vyděl 2, výsledek vynásob 3, a řekni mi finální číslo.“

Model musí udělat **kroky za sebou**:

1. Krok 1: `10 ÷ 2 = 5`
2. Krok 2: `5 × 3 = 15`
3. Odpověď: `15`

To není jedna odpověď na jeden dotaz. Je to **řetězec akcí** — jako recept: nejdřív rozbij vejce, pak míchej, pak peč.

**NESTFUL** je sbírka takových receptů. Některé mají 2 kroky, některé 6+.

**Náš projekt** zkouší: když modelu dáme lehčí recepty a postupně těžší, naučí se to lépe než když ho hodíme rovnou do pekla?

---

## Tři věci, které děláme

```
1. TRÉNINK (GRPO + LoRA)
   Model zkouší řešit úlohy → dostane známku → učí se

2. CURRICULUM (stages 1→6)
   Nejdřív 1 krok, pak 2, pak 3… jako škola

3. EVAL (NESTFUL)
   Zkouška: umí to opravdu, nebo jen na cvičišti?
```

---

## Co je „úspěch“?

Ne „model je nejlepší na světě“. Úspěch je:

- **exec_pass_rate** roste oproti baseline (base model bez tréninku)
- Na těžších úlohách (víc kroků) to **neklesá hned**
- Víme **proč** to padá (formát? špatný nástroj? špatný výpočet?)

---

## Kde jsme teď (červen 2026)

| Stage | Co trénujeme | Co testujeme | Jak to jde |
|---|---|---|---|
| 1 | 1-call synthetic | 2-call NESTFUL | ✅ ~55 % vs baseline 50 % |
| 2 | 2-call | 3-call | ✅ ~23 % vs baseline 21 % |
| 3 | 3-call | 4-call | ✅ ~11 % vs baseline 8 % |
| 4 | 4-call | 5-call | ❌ ~4 % vs baseline 5 % |
| 5–6 | … | … | 🔄 běží / plánováno |

**Hlavní zjištění zatím:** Krátké řetězce curriculum pomáhá. Na delších narážíme spíš na **formát odpovědi a parsování** než na „model neumí počítat“.

---

## Co z toho může být PhD příběh

> Curriculum na synthetic multi-call datech zlepšuje krátké řetězce, ale generalizace na delší NESTFUL úlohy vyžaduje [X] — kde X zjišťujeme (reward? data? threshold? formát?).

To je v pořádku i když čísla nejsou skvělá. Důležité je **vědět proč**.

---

## Tvoje vlastní vysvětlení (doplň!)

*Napiš sem 5–10 vět vlastními slovy. Špatně je OK — opravíme v journalu.*

```
[Místo pro tebe]

Vycházíme z Tool-R0: původně dva modely (generátor úloh + solver), oba se zlepšují
přes GRPO. My generátor nahradíme syntetickými daty rozdělenými do stages podle
počtu tool callů (1 až 6). Jeden model (Qwen3-4B + LoRA) se učí řešit tyto úlohy
a dostává reward za formát, správné nástroje a správný výsledek.
Po každé stage testujeme na NESTFUL úlohách o jeden krok těžších, než na čem
trénujeme (stage 1 → eval na 2-call úlohách, stage 4 → eval na 5-call). Úspěch
znamená exec_pass_rate vyšší než u baseline modelu bez tréninku.
Zatím víme: curriculum pomáhá na krátkých řetězcích (stages 1–3 nad baseline),
ale na stage 4 už jsme horší než baseline. Většina chyb na těžkých úlohách
je parse_fail — model neřekne odpověď ve formátu, který eval umí přečíst,
ne špatný výpočet.
Moje hlavní hypotéza: GRPO často nemá učící signál, protože všech 8 rollouts
na stejný prompt dostane podobný (nebo nulový) reward → advantage je ~0 a model
se z toho promptu skoro nic nenaučí. Tohle chci ověřit v reward logu / wandb.
Druhá věc, které nerozumím: eval NESTFUL běží po skupinách n_calls (2, 3, 4…),
ne na celém datasetu najednou — proto se čísla mezi soubory liší. Celý přehled
je v baseline_nestful.json.
Směr pro mě: zjistit, kde curriculum přestává fungovat (data? formát?
train single-shot vs eval multi-turn? GRPO signál?) a navrhnout jeden cílený
experiment — ne další náhodný běh na cloudu.

```
