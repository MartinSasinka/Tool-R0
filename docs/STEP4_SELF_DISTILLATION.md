Manuál: Step 4 — SDPO (Self-Distillation Policy Optimization)

Tento dokument popisuje správnou strukturu a pořadí rozšíření tréninku o SDPO tak, aby:

nesahalo do step0–step3 a původní pipeline zůstala beze změny,
běželo offline ve dvou fázích: sběr rolloutů → samostatný trénink,
nepoužívalo BFCL exec_* jako hlavní tréninkové prostředí,
používalo lokální verifikovatelné prostředí s bohatým textovým feedbackem,
zůstalo co nejblíže doméně a formátu step3_solver.py,
produkovalo auditovatelná JSONL data + souhrnné reporty,
bylo realisticky spustitelné na DGX bez Docker/Podman závislosti.

Implementační soubory budou pojmenované jednotně step4_*.py (a případně run_step4_*.sh), analogicky k step2_*.py / step3_solver.py.

1. Rozhodnutí a scope

V Step 4 implementujeme pouze SDPO.

Neimplementujeme SDFT, protože:

SDFT je metodicky zaměřené na continual learning from demonstrations a redukci forgettingu při sekvenčním učení nových dovedností, což není hlavní problém, který teď u Tool-R0 řešíte; vy řešíte hlavně irrelevance regresi, lepší práci s chybou a posílení multi-step tool use.
SDFT podle autorů výrazněji těží z lepší in-context learning schopnosti větších modelů; na 3B na jejich project page podávalo horší výsledky než SFT a zisky se objevily až ve vyšších škálách. To z něj dělá slabší první volbu pro 1.5B Tool-R0.
SDPO je naopak přímo navržené pro verifiable tasks s rich feedbackem, včetně tool use, a převádí textový feedback na hustší učící signál než čistý outcome reward.
2. Co použít místo BFCL-exec
2.1 Primární doporučení: Tool-R0 ExecBank

Jako hlavní prostředí doporučujeme vytvořit Tool-R0 ExecBank: lokální sadu single-turn tool-use úloh ve stejném stylu jako step3_solver.py, ale s možností deterministické exekuce a textového feedbacku.

Každá úloha má:

question ve stejném stylu jako ve step 3,
očekávaný tool call nebo sémantickou specifikaci výsledku,
lokální tool / checker napsaný v Pythonu,
bohatý feedback při chybě, např.:
missing_required_argument: city
wrong_argument_type: start_date expected ISO-8601
invalid_enum_value: currency must be one of [USD, EUR, GBP]
semantic_mismatch: returned flights outside allowed layover window
irrelevance_error: no tool should be called for this request

To je doporučení, ne převzatý benchmark. Důvodem je, že SDPO paper stojí na prostředích, která vracejí textovou informaci o chybě, a ta je pak použita jako feedback-conditioned self-teacher signal. BFCL exec_* je pro vás méně vhodný, protože jeho executable režimy jsou navržené jako benchmarkové eval kategorie a podle BFCL dokumentace vyžadují API klíče a volání externích API.

2.2 Sekundární zdroj: API-Bank simulator subset

Jako sekundární zdroj doporučujeme malý lokální subset API-Bank, ne celé API-Bank end-to-end. API-Bank paper popisuje runnable evaluation system se 73 API tools a veřejné eval repo skutečně obsahuje soubory simulator.py a tool_manager.py, což z něj dělá dobrý základ pro lokální, ne-síťové nebo semi-simulované tool-use úlohy.

Prakticky:

vyberte jen ty API / tool kategorie, které lze spustit lokálně a deterministicky,
převeďte jejich validaci do vašeho jednotného step3 formátu,
používejte je jako doplňkový mix pro rozmanitější chybové zprávy.
2.3 Co nepoužít jako primární zdroj

ToolTalk nepoužívat jako hlavní SDPO prostředí. Repo říká, že jde o 28 easy + 50 hard conversations, s 28 nástroji v 7 pluginech, a evaluace probíhá po každé uživatelské replice v konverzaci. To je výborné pro evaluaci multi-turn chování, ale jako první SDPO training loop je to zbytečně vzdálené vašemu single-turn solveru a dat je málo.

3. Jak má SDPO v tomto repu metodicky fungovat
3.1 Ne “teacher jako generátor”, ale “teacher jako scorer”

V této implementaci teacher negeneruje novou sekvenci, kterou by student napodoboval token po tokenu.

Místo toho:

student vygeneruje rollout na původním promptu,
prostředí vrátí feedback,
teacher větev vezme ten samý studentův rollout a spočítá jeho tokenové log-probability pod feedback-conditioned promptem,
loss porovnává studentovy a teacherovy distribuce na stejných tokenech.

To je zásadní konstrukční rozhodnutí. Je v souladu s SDPO paperem, který popisuje, že model conditioned on feedback slouží jako self-teacher a jeho feedback-informed next-token predictions se destilují zpět do policy. Podobný princip privileged-context teacher vs student na studentových vlastních trajektoriích je i v Self-Distilled Reasoner.

3.2 Teacher je stop-gradient větev, ideálně EMA teacher

Teacher nesmí být “stejný model, co se právě učí” bez stabilizace.
V implementaci tedy platí:

teacher branch běží bez gradientů,
preferovaná varianta je EMA teacher nebo snapshot teacher,
student je jediná větev, do které teče gradient.

Tím se zabrání triviálnímu “chasing your tail” efektu. Tohle odpovídá duchu SDPO, kde teacher je feedback-conditioned self-teacher, ale učící signál musí být stabilní.

3.3 Rich feedback je nutnost, ne detail

SDPO funguje dobře tehdy, když feedback není jen binární 0/1, ale nese informaci o tom, co přesně se pokazilo. Proto v tomto návrhu není BFCL exec_* hlavní zdroj. BFCL je výborný eval benchmark, ale pro tréninkový feedback je vhodnější lokální prostředí, které umí produkovat delší a sémanticky užitečnější chybové zprávy. To je návrh vyplývající z SDPO paperu a z praktických vlastností BFCL/API-Bank prostředí.

4. Prostředí exekuce
4.1 Primární prostředí: lokální deterministic checker

Step 4 bude používat lokální checker, nikoli živé HTTP API.

Každý task v ExecBank bude mít:

parser predikovaného <tool_call_answer>...</tool_call_answer>,
kontrolu jména nástroje,
kontrolu argumentové struktury,
kontrolu typů,
volitelnou sémantickou kontrolu výsledku,
generátor textového feedbacku.

Příklad typů tasků:

kalkulace a transformace,
kalendářové operace,
lookup nad malým lokálním KB,
filtrování / sort / aggregation,
schema-sensitive formulářové operace,
“no tool” / irrelevance úlohy,
dvoukrokové latentní mapování, kde je pořád vyžadován jen jeden finální tool call.
4.2 Doplňkový mix: API-Bank-derived local tasks

Menší podíl batchů může být z API-Bank-inspired nebo API-Bank-simulator úloh, ale jen pokud:

jsou lokálně spustitelné,
nepotřebují síť,
jdou převést do jednoho question -> tool_call_answer kroku,
vrací konzistentní textový feedback.

Tento mix doporučuji držet malý, například 10–20 %, aby hlavní distribuce zůstala blízko step 3 doméně. To je metodický návrh, ne tvrzení z paperu. Opírá se o to, že API-Bank má široké API pokrytí a veřejný runnable eval scaffolding, ale vaše hlavní potřeba je stále formátová kompatibilita se solverem.

5. Zarovnání domény se step 3

step3_solver.py je pro Step 4 referenční formát.
Step 4 proto musí zachovat:

stejný system prompt nebo jeho minimálně odvozenou variantu,
stejný styl user -> tool_call_answer,
stejný parser cílového výstupu,
stejné vyhodnocení validního / nevalidního tool call formátu.

Cílem není vytvořit nový agentní framework, ale přidat feedback-aware refinement nad stávající solver policy. Tento závěr je inference z vaší současné pipeline a ze skutečnosti, že ToolTalk či BFCL multi-turn/agent režimy by jinak otevřely nový distribution shift.

6. Architektura: offline dvě fáze
6.1 Fáze 1 — sběr rolloutů

Sběr dělá pouze toto:

načte checkpoint po step 3,
spustí inference na ExecBank taskách,
uloží studentovu odpověď,
checker vrátí:
exec_ok
exec_feedback
volitelně error_code
uloží JSONL + souhrnný report.

V této fázi není nutné generovat teacher completion.
Stačí uložit studentův rollout a feedback. Teacher scoring lze dopočítat až ve fázi 2.

6.2 Fáze 2 — SDPO trénink

Trénink dělá:

načte JSONL ze sběru,
vybere řádky:
primárně exec_ok == false,
volitelně menší podíl exec_ok == true,
vytvoří dvojici promptů:
student prompt = původní system + user
teacher prompt = původní system + user + feedback
teacher branch spočítá log-proby nad původní studentovou sekvencí,
loss aktualizuje jen studenta,
checkpoint uloží do nového adresáře.

Online varianta “generate + feedback + update v jednom loopu” se v první verzi nepoužívá, protože je hůř laditelná a hůř auditovatelná na clusteru. To je implementační doporučení.

7. Bezpečnost a provozní jednoduchost

Tento Step 4 nepředpokládá Docker/Podman.

Místo toho:

checker běží jako lokální Python proces,
používá timeout,
pracuje jen v temp adresáři,
nepoužívá síťové tooly,
loguje jen ořezaný feedback,
nikdy nesmí shodit celý collect job kvůli jedné chybě.

Neříkáme “máme hard network sandbox”; říkáme přesněji:
Step 4 používá pouze lokální deterministic tools a síťové volání není součástí training environmentu.

8. Datový formát JSONL

Jeden řádek = jeden rollout.

Doporučená pole:

Pole	Typ	Popis
schema_version	int	např. 2
stage	str	"sdpo_collect"
source	str	"toolr0_execbank" nebo "apibank_local"
task_id	str	unikátní ID
checkpoint	str	použitý student checkpoint
messages_student	list	system + user
completion_student	str	studentův rollout
exec_ok	bool	zda checker uznal výsledek
error_code	str | null	např. missing_arg, wrong_tool, irrelevance, semantic_mismatch
exec_feedback	str	textová zpětná vazba
gold_answer	str | null	kanonická odpověď, pokud existuje
meta	dict	seed, temperature, domain, source split, atd.

Záměrně zde není completion_teacher. Teacher completion se negeneruje; teacher funguje jako scorer.

9. Loss a tréninkový objective

Primární objective:

SDPO distillation loss na failed rollouts,
volitelně malý doplněk přes standardní CE/SFT na gold answer pro stabilizaci.

Doporučení:

začít se samotným SDPO loss + malý SFT anchor,
nezkoušet na začátku komplikovaný hybrid s GRPO v tomtéž jobu.

První verze Step 4 má být:

jednoduchá,
auditovatelná,
snadno ablovatelná.

To je designové doporučení odvozené z toho, že Tool-R0 už má GRPO ve step 3 a Step 4 má být izolované rozšíření, ne nový monolit.

10. Hyperparametry — výchozí návrh

Startovní hodnoty:

Parametr	Doporučení
teacher_mode	ema_scorer
lambda_sdpo	0.1 až 0.3
teacher_temperature	0.7 až 1.0
max_feedback_chars	512 až 1024
lr_step4	nižší než step 3, např. 1e-6 až 3e-6
epochs	1 až 2
failed_only_ratio	začít 1.0
success_mix_ratio	později 0.1 až 0.2
irrelevance_mix	explicitně držet nenulový podíl

Hlavní princip:

Step 4 je krátký corrective stage,
ne nový dlouhý plný post-training.
11. Navrhované soubory
Soubor	Účel
step4_sdpo_collect.py	inference na ExecBank / API-Bank-local + checker + JSONL
step4_sdpo_train.py	načte JSONL, spočítá teacher-scored SDPO loss, uloží checkpoint
step4_execbank.py	lokální tools, checker, error message templates
step4_feedback.py	normalizace a redakce feedbacku
run_step4_sdpo.sh	orchestrace collect + train
step4_sdpo.yaml	konfigurace cest a hyperparametrů

Doporučení:

dát tyto soubory vedle stávajících training stepů,
neotevírat novou vícehlavou strukturu typu train_extensions/ v první verzi,
mít jednu zřejmou cestu implementace.
12. Evaluace po Step 4

Po Step 4 vždy spustit stejnou eval sadu jako po step 3 a zvlášť sledovat:

BFCL AST,
BFCL irrelevance,
BFCL exec nebo vlastní exec eval split,
ToolAlpaca AST,
API-Bank AST,
SimpleEnv,
ToolTalk.

Primární očekávané zlepšení:

vyšší parse success,
nižší wrong-argument / wrong-tool chybovost,
lepší exec success na verifikovatelných úlohách,
částečné zlepšení easy multi-turn transferu.

Nečekal bych zásadní skok na nejtěžším ToolTalk hard; Step 4 stále zůstává single-turn corrective stage, ne plný agentní planner training. To je inference navázaná na charakter ToolTalk benchmarku.

13. Pořadí experimentů
Baseline: checkpoint po step 3
SDPO-v1: pouze Tool-R0 ExecBank
SDPO-v2: ExecBank + malý API-Bank-local mix
SDPO-v3: přidat success-mix a irrelevance-heavy sampling
SDPO-v4: EMA teacher vs snapshot teacher ablace
14. Hlavní rizika a jak je přiznat v článku
Feedback quality bottleneck
Když budou error messages moc strohé, SDPO efekt bude slabý. SDPO paper stojí na bohatém feedbacku.
Dataset size bottleneck
Malý počet verifikovatelných tasků může vést k nestabilním ziskům; proto je lepší vlastní ExecBank než spoléhat jen na BFCL exec_*.
Distribution overfitting
Příliš úzký ExecBank může model specializovat jen na lokální checker styl.
Irrelevance drift
Bez explicitních “no tool” úloh může model dál přeučovat na nadměrné volání nástrojů.
15. Shrnutí rozhodnutí

Tento Step 4:

implementuje pouze SDPO,
nepoužívá SDFT,
nepoužívá BFCL-exec jako hlavní training environment,
staví na lokálním Tool-R0 ExecBank,
volitelně přimíchává API-Bank-inspired local tasks,
používá teacher-as-scorer na studentových vlastních rolloutech,
používá stop-gradient / EMA teacher,
zachovává plnou kompatibilitu se step 3 formátem,
zůstává offline, auditovatelný a snadno ablovatelný.