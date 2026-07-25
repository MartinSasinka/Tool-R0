# DISCOVERY — mapa artefaktů (forenzní audit 2026-07-25)

Všechna tvrzení v této sadě reportů jsou ověřena přímo z JSON/JSONL/zdrojového
kódu; Markdown souhrny byly použity jen jako ukazatele. Strojová data:
`reports/root_cause_forensic/analysis/a01..a08*.json`.

## Runy

| Run | Umístění | Stav artefaktů |
|---|---|---|
| `pure_stage3_2ep_20260719_221918` | `outputs/runs/pure_stage3_2ep_20260719_221918/` | train_log.jsonl (2 epochy), config, checkpointy, **plné eval adresáře včetně final_eval_trajectories.jsonl (dříve považované za ztracené)** |
| Round-1 ablation A0–A4 | `outputs/runs/_local_round1_analysis/reward_ablation_r1_<ARM>_seed20260724/` | train_log.jsonl (160 groups/arm), console.log, config_used.json, adapter checkpointy (FINAL + adapter_epoch_1), eval `task_results.jsonl` + `_traj` |
| `shared_C0_eval_500` | `outputs/runs/shared_C0_eval_500/` | C0 baseline eval — POZOR: obsahuje 1861 řádků (test+dev), ne 500; pokrytí 500 párovaných ID je ale kompletní (viz EVALUATION_AUDIT.md) |

## Kód (relevantní pro root cause)

- Reward registry: `lib/reward_ablation_registry.py` (v5.0.2), `lib/reward_v3_2_dense.py`
- Dispatch řetězec: `scripts/ablation/run_reward_ablation.py` → `run.py`
  (`_hook_select_train_reward`) → `nestful_mtgrpo_partial/two_phase_train_session.py`
  (`os.environ.setdefault("REWARD_POLICY", "execution_aware_v3_2_dense")` při importu)
  → `nestful_mtgrpo_minimal/grpo_train.py` (`episode_turn_reward_seq`)
- Credit assignment: `grpo_train._turn_returns`, `nestful_mtgrpo_minimal/group_stats.py`
- Rollout worker: `vllm_dp_pool.py` (loguje `config_policy`/`resolved_policy` do konzole)

## Datové sady

- Train subset Round-1: 160 úloh (vše 3-call), hash `b64d3ec2…` shodný ve všech armech
- Stage-3 train-ready: 326 úloh; Stage-2: 496 úloh (2-call)
- NESTFUL diagnostic 500: fixní stratifikovaný subset (100×{2,3,4,5,6}-call)
- NESTFUL test full: 1661 úloh

## Co CHYBÍ (limity auditu)

1. **Train-side rollout trajektorie nejsou persistované** — v train_log.jsonl jsou
   jen hashe completions, episode/turn rewards a agregáty. Matched-prefix /
   first-divergence credit audit a skutečný train-side counterfactual re-score
   jsou `UNTESTABLE_WITH_CURRENT_LOGS`.
2. **Grad normy nejsou logované** (`has_grad_norm_field=false` všude).
3. C0 eval manifest neuvádí checkpoint (`manifest_checkpoint: null`).

Detailní inventura polí train logu: `analysis/a04_update_strength.json`
(`logged_fields`).
