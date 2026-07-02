# Feynman workflow — jak přemýšlet nad problémy

---

## Metoda v 4 krocích

### 1. Vyber jeden koncept (ne celý projekt)

Špatně: „Dnes pochopím GRPO, curriculum, NESTFUL a cloud.“

Dobře: „Dnes pochopím, co přesně znamená parse_fail_rate.“

### 2. Vysvětli na prázdný papír (nebo do journalu)

Pravidla psaní:

- Žádné zkratky bez vysvětlení
- Jedna analogie ze života
- Příklad z **našeho** běhu (číslo z JSON, ne abstraktní)

Šablona:

```markdown
## Koncept: [název]

### Jako bych to říkal kamarádovi
[3–8 vět]

### Analogie
[Jedna věc ze života]

### Příklad z našeho projektu
[Konkrétní číslo / soubor / chyba]

### Kde mám díru
[Co mi ještě nedává smysl]
```

### 3. Najdi díry

Signály, že ještě nerozumíš:

- Musíš použít slovo, které neumíš vysvětlit
- Vysvětlení trvá víc než 10 vět
- Nemůžeš říct, co by se stalo, když parametr změníš
- Mícháš trénink a eval

→ Díra jde do `open-questions.md`, ne do zásuvky v hlavě.

### 4. Zjednoduš a vrať se ke zdroji

- Přečti **jeden** kódový úsek (ne celý soubor)
- Spusť **jeden** malý experiment
- Zeptej se AI: „Je moje vysvětlení v journalu správně? Kde je chyba?“

---

## Jak pracovat s frustrací

Frustrace často znamená **příliš velký koncept najednou**.

Rozděl:

```
„Stage 4 nefunguje“
        ↓
├── Model generuje špatný JSON?     → parse_fail
├── Model volá špatný nástroj?      → tool_call_acc
├── Nástroje OK, špatná odpověď?    → exec_pass
└── Odpověď uříznutá?               → clipped_frac
```

Jedna větev = jeden den v journalu.

---

## Týdenní rytmus

| Den | Aktivita |
|---|---|
| Po–Čt | 1 koncept + journal |
| Pá | Aktualizuj `current-focus.md` + `insights-log.md` |
| Před schůzkou | Přečti `00-big-picture.md` nahlas za 2 min |

---

## Checklist „rozumím tomu“

- [ ] Umím vysvětlit bez slova, které neznám
- [ ] Umím říct, co by se změnilo, když změním jeden parametr
- [ ] Umím ukázat příklad z `training/results/`
- [ ] Vím, co je hypotéza a co jen pozorování
