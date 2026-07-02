# Aktuální fokus

> **Jedna otázka najednou.** Když se mění, starou otázku přesuň do `insights-log.md`.

---

## Otázka teď

**Proč stage 4 (trénink na 4-call) nedává lepší výsledek na 5-call NESTFUL než baseline?**

---

## Co už víme (fakta, ne domněnky)

- Baseline na 5-call: **5.2 %** exec_pass
- Stage 4 epoch 1 val: **4.0 %** exec_pass
- Parse_fail na stage 4 eval: **~85 %** (model často neprodukuje parsovatelný tool call)
- Clipped_frac 22 % u stage 4 — spíš **dlouhý nesmyslný výstup**, ne malé okno
- Curriculum stages 1–3 **fungovaly** (exec_pass nad baseline)

---

## Hypotézy (k otestování)

| # | Hypotéza | Jak ověřit | Stav |
|---|---|---|---|
| H1 | Model se učí formát na synthetic, ale NESTFUL eval je jiný formát | Porovnat failure examples synthetic vs NESTFUL | ⬜ |
| H2 | Advance threshold 0.12 je moc měkký — jdeme dál moc brzy | Zpřísnit na 0.35, sledovat stage 4 | ⬜ |
| H3 | +1 call generalizace je moc skok | Eval i na stejném n-call jako train | ⬜ |
| H4 | Replay 15 % nestačí — zapomíná starší stages | Zvýšit replay, měřit | ⬜ |

---

## Další konkrétní krok

- [ ] Přečíst 10 řádků z `stage_4_epoch1_val.failures.jsonl` a kategorizovat chyby
- [ ] Dopsat vlastní vysvětlení do `concepts/04-eval-metrics.md` (sekce „Tvoje slova“)

---

## Co NENÍ teď priorita

- Optimalizace cloud GPU (už běží)
- Snižování oken (2048 pro eval stačí)
- Stage 6 dokončit za každou cenu před schůzkou

---

## Poznámky

```
[Místo pro tebe — co tě dnes trápí, co jsi zkusil]


# Dobré otázky pro analýzu:
Synteteické úlohy jsem schválně generoval tak, že jsem je vygeneroval a pak pršly verifikací, kde musely jít splnit podle NESTFUL funkcí (u větší stage to bylo těžší), myslel jsem si, že tím zajistím, že data budou lepší a bude víc učícího se signálum, to se ovšem nestalo. 
1. Co otestovat, kolik z tasků na každé stage má advantage 0? Na jakých metrikách model selhává u delších úlohách? Je to formát, je to parseování, je to zvolení správných tool callů?
2. Jak optimalizovat tokeny pro každou stage pro optimalizovanější běh tréningu?
3. Jak jsme vygenereovali tasky, kde jsou prompty atd?
4. Jak funguje trénink celý a jak jsme získali data?
5. Compounding errors, jak měřit, jak odstranit, lze zjistit, kolikrát se to stalo, kde je problém?
6. Co je to plně autoregresivní GRPO a k čemu by to bylo?
```
