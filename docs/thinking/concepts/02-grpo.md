# GRPO — jak se model učí

---

## Jako bych to říkal kamarádovi

Normální učení: učitel ukáže správnou odpověď, žák ji opíše.

**GRPO** je jinak:

1. Model dostane úlohu
2. **Sám vymyslí několik pokusů** (u nás 8 = `num_generations`)
3. Každý pokus dostane **známku** (reward)
4. Model se učí: dělej víc toho, co mělo vyšší známku, míň toho, co mělo nízkou

Není potřeba „učitelova“ jedna správná odpověď v každém kroku — stačí **porovnat pokusy mezi sebou**.

---

## Analogie: soutěž v psaní úlohy

8 žáků napíše řešení stejného příkladu. Učitel neřekne „toto je správně“, ale řekne „č. 3 a 7 byly lepší než č. 1 a 5“. Žáci se příště přiblíží stylu 3 a 7.

---

## Co je LoRA (proč ne celý model)

Měníme jen **malou přídavnou vrstvu** modelu (adapter), ne všech 4 miliard parametrů.

- Levnější na GPU
- Rychlejší
- Checkpoint je malý (~64 MB)

Jako bys na velké knize lepil jen post-it poznámky, ne přepisoval celý text.

---

## Reward u nás (známka)

| Část | Váha | Co měří (`tool_r0`) |
|---|---|---|
| format | 20 % | Je tam `<tool_call_answer>` a jde vyparsovat call? |
| call_match | 30 % | Správný nástroj + argumenty vs gold **pro tento turn** |
| exec_match | 50 % | IBM spustí řetězec (prefix + pred) a výsledek sedí s `expected_result` |

Model může dostat body za **formát**, ale pořád failnout na **exec_pass** v evalu — to jsou různé věci.

---

## Důležité parametry

| Parametr | Naše hodnota | Intuice |
|---|---|---|
| `num_generations` | 8 | Víc pokusů = stabilnější učení, dražší |
| `learning_rate` | 5e-6 | Malé kroky — LLM se neraduje |
| `beta` | 0.02 | KL penalizace — drží model blízko původnímu (nesmí „zbláznit“) |

---

## Příklad z projektu

Trénink stage 1 na **1-call** synthetic datech → eval na **2-call** NESTFUL:

- Před tréninkem (baseline): **50.2 %**
- Po stage 1: **~55 %**

→ GRPO + curriculum aspoň na krátkých řetězcích **něco naučí**.

---

## Díra v mém chápání

- [x] Jak se z 8 completions spočítá gradient — hrubě v odpovědi 23 níže
- [x] Proč `beta` — odpověď 15
- [ ] Jak měřit podíl promptů s advantage ≈ 0 v praxi (wandb / log)

---

## Tvoje slova (doplň!)

```
1. Čím se GRPO liší od SFT?
→ SFT: jedna gold odpověď, model ji napodobuje. GRPO: na stejný prompt X rollouts,
  každý dostane reward, učí se z **rozdílů mezi rollouts** (kdo byl lepší v rámci skupiny).
  Gold pořád používáme — ale v **reward funkci**, ne jako přímý copy target jako u SFT.

2. Co je „group“ u nás?
→ Skupina = **8 completions** (`num_generations`) na **jeden stejný prompt** (jeden training row).
  Porovnáváme je navzájem, ne s jinými úlohami.

3. Proč stačí porovnávat pokusy mezi sebou?
→ GRPO použije **průměr rewardu ve skupině** jako baseline. Lepší rollout → kladný advantage,
  horší → záporný. Postupně se posouváme k lepším odpovědím i bez jedné „perfektní“ vzorové věty
  pro každý token. (Gold je v reward, ne v loss jako u SFT.)

4. Analogie?
→ V kuchyni vyzkoušíš 4 způsoby krájení cibule. Nemusíš znát „ideální“ techniku dopředu —
  porovnáš čas + rovnoměrnost + bezpečnost a příště zopakuješ tu lepší. GRPO dělá totéž s texty.

5. Jeden training step (hrubě)?
1. Vezme se batch promptů (u nás `tool_r0` prefix + úloha)
2. vLLM vygeneruje 8 completions na každý prompt
3. Reward funkce ohodnotí každý completion (format / call_match / exec_match)
4. Spočítá se advantage = reward − průměr ve skupině (+ normalizace)
5. Gradient posílí tokeny z lepších rollouts, oslabí horší (+ KL penalty přes `beta`)
6. Po `gradient_accumulation_steps` micro-batchech → update LoRA vah

6. Proč `num_generations: 8` dražší než 2?
→ 4× víc generovaného textu na stejný prompt = víc VRAM (KV cache), víc času v vLLM,
  víc reward výpočtů. Odměna: stabilnější odhad „co je lepší“ ve skupině.

7. LoRA?
→ Low-Rank Adaptation: přidáme malé trénovatelné matice do vybraných vrstev, zbytek modelu
  je zmrazený. Učíme ~64 MB adapter místo 4B parametrů — levnější, rychlejší, menší checkpoint.

8. Proč vLLM?
→ vLLM je optimalizovaný inference engine (PagedAttention, batching). Pro GRPO potřebujeme
  **hodně rychlých rollouts** — vLLM colocate na každé GPU generuje mnohem rychleji než
  opakované `model.generate()` v PyTorch train loopu. Bez vLLM by trénink trval řádově déle.

9. Reward `tool_r0`?
→ Vážený součet 0–1:
  - **format (0.2):** je `<tool_call_answer>` a jde vytáhnout call? (ne „libovolný JSON“)
  - **call_match (0.3):** jméno nástroje + klíče/hodnoty argumentů vs gold call **tohoto turnu**
  - **exec_match (0.5):** IBM spustí prefix+pred call a výsledek sedí s `expected_result`
  Oprava: není to jen „spustitelný JSON“, ale XML tag + skutečná exekuce.

10. Vysoký train reward, ale fail exec_pass v eval?
→ **Ano.** Train hodnotí **jeden turn** s gold prefixem; eval hodnotí **celý multi-turn řetězec** sám.
  Model může umět jeden krok, ale spadnout na turn 2. Nebo mít reward za formát, ale špatný řetězec celkově.

11. Advantage jednou větou?
→ „O kolik je tento rollout lepší/horší než průměr své skupiny“ — učící signál pro gradient.

12. Kdy advantage ≈ 0?
→ Když všech 8 rollouts dostane **stejný** (nebo velmi podobný) reward — typicky všechny 0.0.
  Pak gradient prakticky nic nezmění → **model se z tohoto promptu nic nenaučí**. (Ne „nezapomíná“ — prostě stagnuje.)

13. Jak to poznat ve wandb?
→ `train/reward` stagnuje nízko, malý rozptyl mezi kroky; vysoká `invalid_json_rate` / parse fail v rollouts;
  po stage 4+ exec_pass v eval neklesá. Ideálně: logovat variance rewardů **uvnitř skupiny** (zatím ručně z logu).

14. Proč je špatných 8× reward 0.0?
→ Žádný rollout není lepší než ostatní → **žádný směr učení**. Model neví, kam se posunout.
  To je přesně tvoje hypotéza ze stage 4.

15. Co dělá `beta` (KL penalty)?
→ Penalizuje odchylku od **referenčního modelu** (base Qwen před RL). Model se smí učit,
  ale ne úplně „rozbít“ původní chování. **`beta = 0`:** agresivnější updaty, riziko degenerace
  (gibberish, opakování, ztráta základního jazyka). **`beta` moc velké:** skoro se nic neučí.

16. Proč malý `learning_rate` (5e-6)?
→ LLM + RL je nestabilní; velký LR rozbije model rychle. Malé kroky + KL (`beta`) = bezpečnější curriculum.

17. Proč `temperature: 0.7`, ne 0?
→ Správně intuice: při T=0 by všechny rollouts vycházely skoro stejně → advantage ≈ 0.
  T=0.7 dává **různorodé** pokusy ve skupině, aby GRPO mělo čím porovnat.

18. `gradient_accumulation_steps: 2`?
→ Oprava: není to „2 úlohy = update“. Na **jedné GPU** se spočítá gradient z 2 micro-batchů,
  pak teprve jeden optimizer step. S 8 GPU × grad_accum × batch dostaneme větší efektivní batch
  bez OOM. `num_generations` s tím přímo nesouvisí — to je počet rollouts **na prompt**.

19. Stage 1 — co GRPO vidí jako dobré/špatné?
→ Jeden training row = jeden turn (XML call). Dobrý rollout: parsovatelný tag, správný tool+args,
  IBM exec sedí s `expected_result`. Špatný: chybí tag, špatný nástroj, špatný výsledek.

20. Proč GRPO na stage 4 málo pomáhá?
→ Kombinace: (a) všechny rollouts podobně špatné → advantage ≈ 0, (b) teacher forcing v train
  vs autoregresivní eval, (c) +1 call generalizace, (d) těžší úlohy. Ne jen „špatný formát“.

21. Reward (train) vs exec_pass (eval)?
→ **Train reward:** jeden turn, gold prefix, částečné skóre (format/call/exec pro tento krok).
  **exec_pass:** celá úloha end-to-end, všechny kroky správně + finální odpověď, bez gold prefixu.
  Proto reward může růst a exec_pass stagnovat — měří jinou věc.

22. Teacher forcing — vliv na GRPO?
→ V promptu jsou **předchozí turny vždy správně** (gold). GRPO hodnotí jen **další** call.
  Model se neučí opravit vlastní chybu z turn 1 — v evalu ale musí. Na delších řetězcích to bolí víc.

23. Jak z 8 completions vznikne gradient? (bez vzorce)
→ Pro každý token v každém rolloutu: „jak moc jsem ho generoval“ × advantage toho rolloutu.
  Rollout s vysokým reward → posílit tyto tokeny. Nízkým → oslabit. Průměr přes skupinu = baseline.
  KL (`beta`) táhne zpět k base modelu. Součet přes 8 rollouts → jeden update LoRA.

24. GRPO vs PPO (volitelně)?
→ PPO potřebuje critic (value network). GRPO baseline = průměr ve skupině — jednodušší,
  méně paměti, běžné pro LLM RL (DeepSeek-R1 styl).
```
