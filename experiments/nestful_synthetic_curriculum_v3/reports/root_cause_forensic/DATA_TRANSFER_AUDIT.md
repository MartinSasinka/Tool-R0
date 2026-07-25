# DATA_TRANSFER_AUDIT — Stage-2/Stage-3 vs NESTFUL

Zdroj: `analysis/a06_data_transfer.json` (profily počítané přímo z JSONL dat).

## Strukturální srovnání

| Vlastnost | Stage-2 (496) | Stage-3 / Round-1 train (326/160) | NESTFUL 500 | NESTFUL test full (1661) |
|---|---|---|---|---|
| Gold call counts | 100 % 2-call | 100 % 3-call | 100×{2,3,4,5,6} | 2:543, 3:363, 4:223, 5:154, 6:378 |
| Unikátní gold tools | 34 | 129–152 | 181 | 817 |
| Nabízené tools (offered) | 34 | 163 | 816 | 3850 |
| Dominantní tools | doménové (`is_above_threshold`, `percentage_of`) | verbose aritmetika (`round_to_decimals`, `square_root_of`, `multiply_numbers`) | holá aritmetika (`divide`, `multiply`, `add`, `subtract` = ~85 % volání) | totéž |
| Řádky s reference argumenty | 100 % | 100 % | **15 %** | **24 %** |
| Argumenty | int+reference | int+reference | int + **str-encoded čísla** (1810 str args) | totéž |

## Tool overlap (gold tools, Jaccard vůči NESTFUL 500)

| Train sada | Sdílené tools | Jaccard |
|---|---|---|
| Round-1 subset 160 | 1 (`rectangle_area`) | **0.0032** |
| Stage-3 (326) | 2 (`rectangle_area`, `square_area`) | **0.0060** |
| Stage-2 (496) | 1 | 0.0047 |

## Klíčové mezery

1. **Prakticky nulový tool overlap** — trénink nemůže učit NESTFUL toolům;
   může učit jen obecný formát/proceduru volání.
2. **Reference-usage inverze**: synthetic trénink má referenční argumenty
   (`$varN.result$` styl) v každé úloze; NESTFUL je má v 15–24 % úloh a čísla
   často předává jako string. Trénink tedy posiluje návyk, který NESTFUL
   většinou nevyžaduje (a může škodit u str argumentů).
3. **Fixní délka**: Stage-3 je 100 % 3-call. NESTFUL 2-call úlohy jsou
   nejslabší bucket (C0 45 % vs 62 % u 3-call; viz EVALUATION_AUDIT.md) —
   trénink na fixních 3-call je nijak neadresuje a undercalling (67 % C0
   úloh má predicted < gold calls) je dominantní failure mode společně
   s `too_few_calls` taxonomií (103–119 úloh/arm).
4. **Šířka nabídky tools**: 163 nabízených tools v tréninku vs 816–3850
   v NESTFUL → výběr správného toolu z velkého katalogu se netrénuje.

## Shortcuty a in-domain learning

- Synthetic executor je deterministický nad malým katalogem; úlohy jsou
  řešitelné naučením mapování „motiv → 3 volání se šablonou referencí" bez
  obecné kompoziční dovednosti. Zlepšení syntetických metrik (pure_stage3)
  proto nemusí implikovat žádnou NESTFUL-relevantní schopnost.
- Empiricky: pure_stage3 2ep zlepšoval train-side signál (loss 0.217→0.063),
  ale oficiální NESTFUL šel o 1.14 pp dolů → **in-domain learning ANO,
  transfer NE**.

## Verdikt

Transfer gap je strukturálně doložený a je nejpravděpodobnější příčinou,
proč by ani korektně provedená reward ablace tohoto typu nezvedla NESTFUL
Win Rate (STRONGLY_SUPPORTED; kauzálně to prokáže jen trénink na
NESTFUL-like datech, což není v mandátu tohoto auditu).
