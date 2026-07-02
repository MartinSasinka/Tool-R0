# Log poznatků

> Krátké záznamy — **i negativní výsledky jsou poznatky**.
> Formát: datum + jedna věta + (volitelně) co z toho plyne.

---

## 2026-06

### 2026-06-17 — Curriculum na krátkých řetězcích funguje

Stages 1–3 dávají exec_pass **nad baseline** na (+1)-call eval. To není náhoda — pipeline měří správně.

→ **Plyne:** Příběh pro schůzku není „nic nefunguje“, ale „funguje do určité hloubky“.

---

### 2026-06-17 — Stage 4 pod baseline

4-call train → 5-call eval: **4.0 %** vs baseline **5.2 %**. Parse_fail **~85 %**.

→ **Plyne:** Problém není jen „víc kroků“, ale **formát / režim evalu / generalizace**. Viz `current-focus.md`.

---

### 2026-06-17 — Okna pro eval: 2048 stačí

Baseline clip při 2048 tokenech: **8–10 %**. Při 1024 bylo ~22 %. Train okna 3072 jsou vůči gold datům nafouklá (~2×).

→ **Plyne:** Optimalizace spíš v parse-fail a datech, ne ve zvětšování kontextu.

---

### 2026-06-17 — Infrastruktura je výsledek

Reprodukovatelný eval, baseline cache, cloud běh, shard merge — to je **výzkumná platforma**, ne overhead.

→ **Plyne:** PhD hodnota není jen číslo na leaderboardu.

---

## Šablona pro nový záznam

```markdown
### YYYY-MM-DD — [titulek]
[1–3 věty co jsme zjistili]

→ **Plyne:** [rozhodnutí / další experiment]
```

---

## Tvoje záznamy

```
[Místo pro tebe]
```
