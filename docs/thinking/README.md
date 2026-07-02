# Thinking space — jak tu pracovat

Tahle složka není dokumentace kódu. Je to **tvůj laboratorní deník pro pochopení**.

Inspirace: [Richard Feynman](https://en.wikipedia.org/wiki/Feynman_technique) — problém pochopíš teprve tehdy, když ho umíš vysvětlit **svými slovy**, jednoduše, bez žargonu.

---

## Struktura

```
docs/thinking/
├── README.md                 ← tady jsi
├── 00-big-picture.md         ← celý projekt jednou větou + analogie
├── 01-feynman-workflow.md    ← jak pracovat krok za krokem
├── current-focus.md          ← JEDEN problém, na kterém teď pracuješ
├── open-questions.md         ← co ještě nechápu (bez ostychu)
├── insights-log.md           ← malé objevy, i negativní výsledky
├── concepts/                 ← jeden soubor = jeden koncept
│   ├── 01-tool-calling.md
│   ├── 02-grpo.md
│   ├── 03-curriculum.md
│   ├── 04-eval-metrics.md
│   └── 05-train-vs-eval.md
└── journal/                  ← denní Feynman zápisy
    ├── _template.md
    └── (YYYY-MM-DD-*.md)
```

---

## Denní rytmus (15–30 min)

1. Otevři `current-focus.md` — co řešíš dnes?
2. Přečti příslušný soubor v `concepts/`
3. Napiš journal: vysvětli to **jako bys to říkal kamarádovi**, ne profesorovi
4. Co nešlo jednoduše → přidej do `open-questions.md`
5. Co jsi pochopil → jedna věta do `insights-log.md`

---

## Pravidla (aby to nešlo vniveč)

| Dělej | Nedělej |
|---|---|
| Piš vlastními slovy | Kopírovat definice z paperů |
| Jedna myšlenka = jedna věta | 10 stran teorie bez příkladu |
| Přiznej „nevím“ | Předstírat, že rozumíš |
| Negativní výsledek = zápis | Frustraci ignorovat |
| Každý běh = jedna otázka | „Uvidíme co vyjde“ |

---

## Jak spolupracovat s AI (Cursor)

Když něčemu nerozumíš, napiš do journalu:

```markdown
## Nechápu
[Tvoje pokus o vysvětlení — i když je špatně]

## Otázka
Proč X když Y?
```

A v chatu: „Přečti `docs/thinking/journal/...` a pomoz mi to doplnit Feynmanovsky.“

AI by měla **doplňovat tvůj text**, ne nahrazovat tvoje myšlení.

---

## Odkazy na kód (když chceš hlouběji)

| Téma | Kde v repu |
|---|---|
| Curriculum orchestrátor | `curricullum/train/run_curriculum_training.py` |
| GRPO trénink | `curricullum/train/train_grpo_stage.py` |
| Eval | `curricullum/train/evaluate_nestful_stage.py` |
| Config | `curricullum/train/configs/qwen3_4b_curriculum_v2.yaml` |
| Výsledky | `curricullum/training/results/` |
