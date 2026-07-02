# Trénink vs eval — proč to není totéž

---

## Jako bych to říkal kamarádovi

**Trénink** = domácí úkol: model napíše **celou odpověď najednou** (všechny kroky v jednom JSONu).

**Eval** = zkouška naživo: model dělá **jeden krok**, dostane výsledek od „počítače“, pak další krok, …

To jsou **dva různé režimy**. Model může být dobrý v jednom a špatný v druhém.

---

## Analogie

- **Trénink:** Napíšeš celý recept od začátku do konce na papír
- **Eval:** Vaříš v kuchyni — po každém kroku ochutnáš a pokračuješ

V kuchyni tě něco překvapí (nástroj vrátí chybu). Na papíru ne.

---

## Tabulka rozdílů

| | Trénink (GRPO) | Eval (NESTFUL) |
|---|---|---|
| Formát | Jeden velký JSON completion | Multi-turn chat |
| Nástroje | Simulované v reward | Skutečně spuštěné (IBM registry) |
| Délka | `max_completion_length` 1536–3072 | `max_new_tokens` 2048 **na každý turn** |
| Data | Synthetic JSONL | NESTFUL benchmark |
| Cíl | Učit adapter | Měřit generalizaci |

---

## Proč na tom záleží pro stage 4

Trénujeme na **synthetic 4-call** (jeden shot JSON).

Testujeme na **NESTFUL 5-call** (multi-turn, jiné prompty, větší tool menu u 2-call).

Když exec_pass klesne, může to být:

1. Model neumí 5 kroků (schopnost)
2. Model neumí multi-turn formát (režim)
3. Synthetic ≠ NESTFUL (data)
4. Všechno dohromady

**Hypotéza H3** v `current-focus.md`: eval na stejném n-call jako train oddělí (1–2) od (3).

---

## Context windows — kde co platí

| Parametr | Kde | Jedna věta |
|---|---|---|
| `max_completion_length` | Train | Max délka celé učebné odpovědi |
| `max_new_tokens` | Eval | Max délka jedné odpovědi v konverzaci |
| `vllm_max_model_length` | Train vLLM | Kolik VRAM rezervuje KV cache |

Gold data mají completion p95 ~**1000 tokenů** — train okna 3072 jsou **rezerva**, ne nutnost.

---

## Tvoje slova (doplň!)

```
[Místo pro tebe]
```
