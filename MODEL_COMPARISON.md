# Porovnání modelů: Qwen 2.5 vs Qwen 3.5

> `eval/` → Qwen 2.5-1.5B-Instruct | `eval_qwen3.5/` → Qwen 3.5

---

## Souhrnné výsledky — všechny benchmarky

| Benchmark | Metrika | Qwen 2.5 Base | Qwen 2.5 + R0 | Qwen 3.5 Base | Qwen 3.5 + R0 |
|-----------|---------|:---:|:---:|:---:|:---:|
| **BFCL AST** | Accuracy % | 42.7 | 63.5 | **80.96** | 68.56 |
| **BFCL Exec** | Accuracy % | 41.7 | 62.9 | **65.83** | 65.42 |
| **API-Bank** | AST Accuracy % | 43.9 | 57.1 | 70.43 | **72.18** |
| **ToolAlpaca** | AST Accuracy % | 17.0 | **30.0** | 17.0 | 30.0 |
| **ToolAlpaca** | Soft Score % | 31.3 | **69.1** | 31.3 | 69.1 |
| **ToolTalk** | Turn Accuracy % | 9.2 | 15.9 | **40.85** | **40.85** |
| **NESTFUL** | Partial Match % | 10.4 | 13.2 | **14.6** | 14.31 |
| **AppWorld** | Task Success % | 0.0 | 0.0 | 0.0 | 0.0 |

> **Tučně** = nejlepší v řádku

---

## NESTFUL — detailní pohled (1861 úloh)

| Metrika | Qwen 2.5 Base | Qwen 2.5 + R0 | Qwen 3.5 Base | Qwen 3.5 + R0 |
|---------|:---:|:---:|:---:|:---:|
| **Partial Match %** | 10.4 | 13.2 | **14.6** | 14.31 |
| Full Match % | — | — | 0.0 | 0.0 |
| Name F1 % | — | — | **45.64** | 46.35 |
| Arg Match Ratio % | — | — | **24.79** | 23.03 |
| Length Match % | — | — | 26.71 | **30.90** |
| Parse Success % | — | — | **67.92** | 62.01 |

> Qwen 2.5 Name F1, Arg Match a Parse Success nejsou k dispozici (eval složka přepsána).

### NESTFUL shrnutí

| | Qwen 2.5 Base | Qwen 2.5 + R0 | Qwen 3.5 Base | Qwen 3.5 + R0 |
|--|:--:|:--:|:--:|:--:|
| Partial Match | 10.4% | 13.2% (+2.8pp) | **14.6%** | 14.31% (-0.29pp) |

- **Qwen 3.5 Base vede** na partial match (14.6% vs 10.4%, +4.2 p.p. oproti Qwen 2.5 Base)
- Tool-R0 finetuning pomáhá Qwen 2.5 (+2.8 p.p.), ale u Qwen 3.5 finetuned mírně škodí (-0.29 p.p.) — model trénovaný na single-turn generování tool callů se hůře vypořádá s vnořenými sekvencemi
- **Full match = 0 u všech modelů** — žádný model nedokáže spolehlivě generovat celou sekvenci nested API callů

---

## BFCL AST — kategorie

| Kategorie | Qwen 2.5 Base | Qwen 2.5 + R0 | Qwen 3.5 Base | Qwen 3.5 + R0 |
|-----------|:---:|:---:|:---:|:---:|
| Simple (400) | 52.0 | 87.0 | 88.5 | **91.0** |
| Multiple (200) | 50.0 | 79.0 | 86.0 | **88.5** |
| Parallel (200) | 33.5 | 74.5 | **83.5** | **83.5** |
| Irrelevance (240) | 28.8 | ❌ 2.1 | **62.1** | ❌ 2.1 |

> ❌ **Irrelevance regrese**: Tool-R0 finetuning způsobuje, že oba modely (2.5 i 3.5) téměř přestanou odmítat tool call — ztráta 27–60 p.p.

---

## Klíčové závěry

| Téma | Závěr |
|------|-------|
| **Baseline síla** | Qwen 3.5 Base výrazně vede na BFCL (+38 p.p.) a ToolTalk (+31 p.p.) |
| **Přínos Tool-R0** | Největší gain u Qwen 2.5 (BFCL +20.8 p.p., API-Bank +13.3 p.p.); u Qwen 3.5 menší nebo nulový |
| **NESTFUL** | Qwen 3.5 Base vede (+4.2 p.p. vs Qwen 2.5), finetuning nepomáhá žádnému modelu |
| **Irrelevance** | Oba finetuned modely padají na 2.1% — vedlejší efekt tool-call tréninku |
| **AppWorld** | Všichni na 0% — benchmark mimo scope single-turn tréninku |

---

*Zdroj: `eval/results/` (Qwen 2.5) a `eval_qwen3.5/results/` (Qwen 3.5) | 2026-04-24*

Metrika	Qwen 3.5 Base	Qwen 3.5 + R0	Rozdíl
Partial Match %	14.60	14.31	-0.29pp
Name F1 %	45.64	46.35	+0.71pp
Arg Match %	24.79	23.03	-1.76pp
Parse Success %	67.92	62.01	-5.91pp
Length Match %	26.71	30.90	+4.19pp