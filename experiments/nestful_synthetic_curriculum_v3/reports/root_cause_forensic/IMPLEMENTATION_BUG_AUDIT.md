# IMPLEMENTATION_BUG_AUDIT — implementační chyby

Zdroje: `analysis/a02_reward_dispatch.json`, `analysis/a03_adapter_audit.json`,
raw `train_log.jsonl`, `console.log`, checkpoint manifesty.

## BUG 1 (CONFIRMED): Reward dispatch — všechny army trénovaly s A0 rewardem

**Všech 5 Round-1 armů trénovalo s `execution_aware_v3_2_dense`.** Ablace
rewardu se nikdy nekonala.

### Mechanismus

1. `nestful_mtgrpo_partial/two_phase_train_session.py` při **importu modulu**
   volá `os.environ.setdefault("REWARD_POLICY", "execution_aware_v3_2_dense")`.
2. Starý `run.py::_hook_select_train_reward` preferoval env
   `REWARD_POLICY`/default před explicitním `config["reward"]["train_policy"]`
   pro `reward_ablation_*` politiky.
3. Runner `run_reward_ablation.py` env proměnnou nenastavoval (ani
   `run_reward_ablation_round1.sh`), takže setdefault vyhrál.

### Důkazy (raw artefakty, per arm A1–A4)

- `console.log` obsahuje OBĚ řádky:
  `[override] reward.train_policy = 'reward_ablation_<ARM>'` (zápis configu)
  a hned poté `[v3/run.py] training reward = execution_aware_v3_2_dense`
  (skutečný dispatch). Worker log: `config_policy=execution_aware_v3_2_dense`.
- `train_log.jsonl` header: `configured_policy = resolved_policy =
  execution_aware_v3_2_dense`; **všech 160 group řádků v každém armu** má
  `reward_policy_resolved = execution_aware_v3_2_dense`.
- **Hash-matched completions**: napříč všemi 10 páry armů bylo nalezeno 14–35
  identických completions (shodný SHA hash) a jejich rewardy jsou **identické
  s max_abs_diff = 0.0** ve 100 % případů (a02, `cross_arm_hash_matched_rewards`).
  Kdyby army měly různé rewardy, musely by se lišit (A1 outcome-only nemá
  dense pásma; a07 ukazuje, že intended politiky dávají průměrné rewardy
  0.62–0.76 — rozdíl ~0.14 na stejných trajektoriích).
- `frac_rewards_explainable_by_intended_scalars` = 0.00–0.05 (logované rewardy
  NEJSOU vysvětlitelné terminálními skaláry intended politik; jsou to dense
  v3.2 hodnoty jako 0.181667, 0.245833, …).

### `config_used.json` lže

Manifest se zapisuje před vytvořením session, takže deklaruje
`reward_ablation_<ARM>`, který nikdy neběžel. Opraveno guardem (viz
FIXES_APPLIED.md).

## Checkpoint / adapter provenance: OK (žádná záměna)

- Všech 5 FINAL checkpointů má **odlišný SHA256** (a03 `provenance`);
  `final == adapter_epoch_1` všude (žádný resume/přepis).
- Manifesty odkazují správný zdrojový run (`…r1_<ARM>_seed20260724/train/checkpoints/adapter_epoch_1`).
- C0 = base checkpoint bez adapteru; C0 eval win rate na 500 (57.0 %) je
  reprodukovatelná z raw `task_results.jsonl` (viz EVALUATION_AUDIT.md).

## „Raw LoRA cosine ≈ 1" byl artefakt metriky (REJECTED jako důkaz slabého update)

- `lora_A` matice jsou seedovaně inicializované a téměř netrénované →
  cosine přes zřetězené absolutní váhy je ~0.99999996 pro JAKÉKOLI dva runy
  stejného seedu.
- Správná metrika: cosine přes **ΔW = B@A** (lora_B je zero-init, takže B@A je
  přesně delta-to-init). A0 vs A4: **cosine_delta_to_init_BA = 0.835**,
  rel_l2_delta = 0.575. Adaptery se reálně liší; „identické adaptery" neplatí.
- Původní hodnoty cosine > 1.0 byly float32 akumulační chyba; přepočteno ve
  float64 trace-identitou (bez materializace ΔW, viz `lib/offline_audit/adapters.py`).

## Rollout worker vs. trainer, eval pipeline

- Worker i trainer resolvují stejnou politiku (shodné header/row/console
  záznamy) — konzistentní, jen konzistentně špatnou.
- Eval používá oficiální scorer; win rate i gained/lost přesně reprodukovány
  z raw trajektorií (EVALUATION_AUDIT.md) → parser/executor/scorer bez nálezu.
