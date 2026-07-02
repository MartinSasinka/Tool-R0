# Tool calling — co to vlastně je

---

## Jako bych to říkal kamarádovi

Model nedostane jen otázku „kolik je 2+2“. Dostane **seznam nástrojů** (jako aplikace v telefonu) a musí sám rozhodnout:

1. Kterou „aplikaci“ spustit
2. S jakými parametry
3. Co udělat s výsledkem
4. Kdy skončit a říct finální odpověď

Každé zavolání nástroje = **jeden krok** v řetězu.

---

## Analogie: recept v kuchyni

- **Nástroje** = spotřebiče (mixér, trouba, váha)
- **Tool call** = „zapni mixér na 30 sekund“
- **Výsledek nástroje** = co mixér vrátil (těsto je hotové)
- **Další krok** = použiješ výsledek v dalším kroku
- **Finální answer** = „dort je hotový, váží 1,2 kg“

---

## Formát u nás (Tool-R0)

**Eval i náš curriculum trénink** (`training_format: tool_r0`) používají **XML tagy**:

```text
<think>
Step 1: divide(arg_0=10, arg_1=2) → 5
</think>
<tool_call_answer>[{"name": "divide", "arguments": {"arg_0": 10, "arg_1": 2}}]</tool_call_answer>
```

Po výsledku nástroje model pošle další `<tool_call_answer>…</tool_call_answer>`. Konec řetězce: `<tool_call_answer>[]</tool_call_answer>`.

**Legacy cesta `json`** (jen při ručním spuštění bez orchestrátoru) používá plain JSON:

```json
{"output": [{"name": "divide", "label": "$var_1", "arguments": {"arg_0": 10, "arg_1": 2}}], "answer": "15"}
```

| | `tool_r0` (náš běh) | `json` (legacy) |
|---|---|---|
| Trénink | 1 turn, XML tag, gold prefix předchozích kroků | celá trajektorie v jednom JSON |
| Eval | multi-turn XML | multi-turn XML (stejné) |

---

## Proč je to těžké

1. **Formát** — validní `<tool_call_answer>` (eval) nebo JSON (legacy train)
2. **Pořadí** — krok 2 závisí na výsledku kroku 1
3. **Správný nástroj** — z nabídky 4–6 nástrojů vybrat správný
4. **Správné argumenty** — ne jen název nástroje
5. **Finální odpověď** — musí sedět s tím, co nástroje skutečně spočítaly

---

## Příklad z našeho projektu

Baseline model na **5-call** úlohách:

- **5.2 %** úloh celých správně (`exec_pass`)
- **83 %** úloh — nepodařilo se ani **parsovat** první tool call (`parse_fail`)

→ Většina chyb není „špatný výpočet“, ale „model neřekl to ve formátu, který počítač rozumí“.

---

## Díra v mém chápání

- [x] Přesně jak vypadá jeden turn v eval vs jeden training row (`tool_r0`) — viz odpovědi 16–18 níže
- [x] Co je `label` ($var_1) a proč — viz odpověď 7
- [ ] Teacher forcing (gold prefix v train) vs plně autoregresivní eval — jak moc to vadí na stage 4+?

---

## Tvoje slova (doplň!)

```
1. Co je tool call jednou větou?
→ Zavolání konkrétního nástroje se zadanými argumenty; spustí se skutečný kód a vrátí výsledek. Liší se od chatbota tím, že odpověď nevzniká „z hlavy“, ale z exekuce nástroje.

2. Proč nestačí říct jen finální číslo „15“?
→ Eval musí ověřit celý řetězec kroků — pozdější kroky závisí na reálném výstupu předchozího. Bez tool callů nejde poznat, jestli model počítal správně, nebo halucinoval. U složitějších úloh to navíc nefunguje vůbec.

3. Co dostane model navíc?
→ Seznam dostupných nástrojů (jméno, popis, parametry) — jako menu API, ze kterého si musí vybrat.

4. Analogie?
→ Vaření: recept = sekvence kroků (nakoupit → nakrájet → uvařit). Každý krok má vstup a výstup; finální jídlo = `answer`.

5. Co jsou `output` a `answer` v JSONu (legacy / gold data)?
→ `output` = sekvence všech tool callů v pořadí. `answer` = finální odpověď uživateli po provedení celého řetězce. (V našem `tool_r0` train/eval se místo toho používá XML `<tool_call_answer>`.)

6. Co znamená jeden objekt v `output` (name, arguments, label)?
→ `name` = který nástroj. `arguments` = vstupy. `label` = jméno kroku v řetězu (např. `$var_1`), přes které se odkazujeme na výstup tohoto kroku.

7. Proč `label` ($var_1), když máme `name`?
→ `name` říká *jaký* nástroj (divide, multiply). `label` říká *který krok* v řetězu — stejný nástroj může být volán víckrát, každý krok má jiný výstup.

8. Co znamená `"$var_1.output_0$"` v argumentech?
→ „Vezmi výstup kroku `$var_1` (první return value nástroje) a použij ho jako vstup sem.“ V eval promptu je podobná syntaxe `$var1.result$`.

9. Proč train jinak než eval?
→ Oprava: náš curriculum (`tool_r0`) taky učí **jeden XML call na turn**, ne celou sekvenci najednou.
  - **Train:** jeden training row = jeden turn; předchozí kroky jsou v promptu jako **gold prefix** (teacher forcing).
  - **Eval:** model musí **sám** řetězit všechny kroky; chyba v turn 1 rozbije turn 2.
  Legacy `json` formát učí celou trajektorii v jednom JSON — ten nepoužívá orchestrátor.

10. Typy chyb:
- **Špatný nástroj** → místo `divide` zavolá `multiply` (`tool_call_acc` klesá).
- **Špatné argumenty** → správný nástroj, špatné hodnoty/klíče (partial score, ne exec_pass).
- **Parse_fail** → eval neumí vytáhnout `<tool_call_answer>` (chybí tag, špatný JSON uvnitř tagu, thinking místo callu, oříznutý výstup…). Není to jen „špatná závorka v JSONu“.
- **Všechny kroky OK, špatná answer** → nástroje proběhly správně, ale finální odpověď nesedí s gold (`exec_pass` = 0, `tool_acc` může být vysoký). To není „špatný nástroj“ — to je špatný závěr.

11. Baseline 5-call: exec_pass ~5 %, parse_fail ~83 %?
→ Model nejvíc neumí **emitovat parsovatelný tool call** ve eval formátu (XML). Aritmetika je vedlejší — většina úloh spadne dřív, než se vůbec něco spustí. (Baseline = netrénovaný model, takže to není jen train/eval mismatch.)

12. Proč je formát samostatný problém?
→ Bez parsovatelného výstupu se nástroj **nespustí vůbec** — exec_pass je automaticky 0. Formát je brána před výpočtem.

13. Stage 1 train → 2-call eval: co musí umět?
→ V **XML formátu** vygenerovat první call, po reálném výsledku z IBM registry druhý call (s referencí na výsledek), správně vybrat tool z menu 4–6 nástrojů **dvakrát**, a ukončit řetězec. Nestačí „umět 1 call“ — musí **navázat** na výsledek prvního kroku sám.

14. Proč parse_fail i po synthetic train s verifikací?
→ Synthetic data garantují správný *gold*, ne že se model naučil *generovat* XML pod tlakem. Navíc NESTFUL úlohy mají jiné prompty/tool menu. A train má gold prefix (model nevidí vlastní chyby z předchozích turnů).

15. Multi-turn eval — kdo co říká?
→ **User:** otázka + tools. **Model (assistant):** `<tool_call_answer>[{...}]</tool_call_answer>`. **User (simulace):** výsledek nástroje z IBM registry. Opakuje se, dokud řetězec neskončí nebo nefailne parse.

16. Jeden turn eval krok za krokem:
1. `messages` = system prompt (`TOOL_R0_SYSTEM_PROMPT`) + user (otázka + tools) + historie předchozích turnů
2. Model vygeneruje text (volitelně thinking + jeden `<tool_call_answer>`)
3. Parser vytáhne call → IBM registry ho spustí
4. Výsledek se přidá jako user zpráva (formátovaná tool response)
5. Další turn, dokud není hotovo nebo parse_fail

17. Jedna training completion (`tool_r0`):
→ Model dostane **prefix konverzace** (system + user + gold předchozí turny až po aktuální krok). Má vygenerovat **jeden** další `<tool_call_answer>`. Reward (`rewards_toolr0_exec`) hodnotí formát, shodu callu a IBM exec. GRPO porovná 8 rollouts — pokud všechny stejně špatné, advantage ≈ 0.

18. Train vs eval — proč vadí u stage 4+?
→ Formát je **stejný** (XML). Liší se **režim**:
  - Train: gold prefix — model jen „dopíše další krok“
  - Eval: plně autoregresivní — chyba se kumuluje přes 4–5 turnů
  Na delších řetězcích se teacher forcing méně přenáší do reálného multi-turn chování.
```
