# FIXES_APPLIED — provedené opravy (pouze prokázané chyby)

Každá oprava má regresní test v
`tests/test_root_cause_forensic_fixes.py` (17 testů, všechny PASS).
Existující sady (`test_reward_ablation*`, `test_reward_v3_2_dense`,
`test_pure_stage3_pipeline`) po opravách PASS (69 passed, 1 skipped).

## 1. Reward dispatch bug (CONFIRMED) — 3 vrstvy obrany

**a) `run.py` (`_hook_select_train_reward` + `_patch_reward_ablation`):**
explicitně nakonfigurovaná `reward_ablation_*` politika v
`config["reward"]["train_policy"]` má nyní absolutní přednost před
`REWARD_POLICY`/`REWARD_NAME` env proměnnými i defaulty. Neznámý
`reward_ablation_*` název vyhazuje `ValueError`. Chování
`execution_aware_v3_2_dense` větve nezměněno.
Testy: `TestRewardDispatchFix` (4).

**b) `scripts/ablation/run_reward_ablation.py`:** runner nastavuje
`os.environ["REWARD_POLICY"]` na politiku armu PŘED importem
`TwoPhaseTrainSession` (pre-empce `setdefault` v modulu).
Testy: součást (a) a (c).

**c) `scripts/ablation/run_reward_ablation.py::assert_dispatched_policy`:**
tvrdý runtime guard hned po vytvoření session — pokud se resolvovaná politika
liší od konfigurované, běh se ABORTUJE (SystemExit) místo tichého tréninku se
špatným rewardem. Přesně tento guard by Round-1 bug zachytil před spotřebou
GPU.
Testy: `TestRunnerGuard` (2).

## 2. Offline audit — detekce a verdikt dispatch mismatch

**`lib/offline_audit/discovery.py::_reward_dispatch_check`:** porovnává
deklarovanou politiku armu s politikami skutečně logovanými v
`train_log.jsonl`; mismatch = error v discovery.
Testy: `TestDiscoveryDispatchCheck` (4).

**`lib/offline_audit/verdict.py`:** dispatch mismatch nyní short-circuituje
verdikt na `REWARD_DISPATCH_BUG` (nejvyšší priorita); executive summary už
netvrdí natvrdo „Rewards distinct across arms" — řádek se generuje z
`_rewards_identical_on_hash_matched` (u Round-1 správně hlásí IDENTICKÉ).
Testy: `TestVerdictShortCircuit` (2).

## 3. Adapter metrika (init artefakt)

**`lib/offline_audit/adapters.py`:** primární metrika je nyní
`cosine_delta_to_init_BA` — cosine přes efektivní update ΔW = B@A (lora_B je
zero-init ⇒ B@A ≡ delta-to-init), počítáno ve float64 trace-identitou
`dot(B₁A₁, B₂A₂) = trace((B₂ᵀB₁)(A₁A₂ᵀ))` bez materializace ΔW (řeší i OOM
14.5 GB). Raw flat cosine ponechán jen jako diagnostika s vysvětlením
artefaktu.
Testy: `TestAdapterDeltaMetric` (2) — mj. konstrukce, kde opačné updaty dají
raw cosine > 0.99 a delta cosine < −0.99.

## 4. Chybná definice metriky „synthetic success"

**`lib/offline_audit/__init__.py`:** `SYNTHETIC_SUCCESS_REWARD` opraven
z 0.92 na **0.90** (dolní mez pásma `fully_correct` v3_2_dense) a
**`lib/offline_audit/on_policy.py`** přeznačuje metriku na
„reward-threshold success proxy" (není path-invariant terminal success).
Testy: `TestSuccessThresholdDefinition` (2).

## 5. Parita forenzní kopie `_turn_returns`

**`scripts/audit/root_cause_forensic/common.py`** obsahuje kopii trainerovy
`_turn_returns`; test drží byte-ekvivalentní chování s
`grpo_train._turn_returns` (50 náhodných případů, γ/λ ∈ {0, 0.5, 1}).
Testy: `TestTurnReturnsParity` (1).

## 6. Windows encoding v testech pipeline

**`tests/test_pure_stage3_pipeline.py`:** subprocess výstup se dekóduje
`encoding="utf-8", errors="replace"` (dítě píše v legacy codepage, když
cesta obsahuje ne-ASCII znaky — „Šunka" → 0x8a → pád reader threadu)
+ oprava `TypeError` při skládání None stderr.

## Vedlejší zjištění (neopraveno záměrně, jen zdokumentováno)

- `config_used.json` se zapisuje před vytvořením session → deklaruje
  intended politiku; skutečnost nyní vynucuje guard 1c, takže manifest už
  nemůže lhát o běžícím tréninku.
- Regenerační skripty manifestů zapisují timestamp do repa i při běhu nad
  temp výstupem (timestamp-only churn; obsahové SHA256 nezměněny; změny
  revertovány).
- Grad normy se nelogují — instrumentace navržena v NEXT_DECISION.md
  (canary), nepřidávána do produkčního traineru v rámci tohoto auditu
  (zákaz velkých změn traineru).

## Co NEBYLO provedeno (dle mandátu)

Žádný nový GRPO trénink, změna LR/KL, nový reward design, generování dat,
SFT, RunPod běh.
