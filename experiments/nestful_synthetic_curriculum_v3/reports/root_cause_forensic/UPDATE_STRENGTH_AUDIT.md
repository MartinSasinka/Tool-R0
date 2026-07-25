# UPDATE_STRENGTH_AUDIT — síla update

Zdroje: `analysis/a04_update_strength.json`, `analysis/a03_adapter_audit.json`.

## Round-1 army (1 epocha, 160 groups)

| Arm | Optimizer steps | LR | KL mean | KL max | Clip rate | Dead groups | Groups s nenulovou adv |
|---|---|---|---|---|---|---|---|
| A0 | 28 | 3e-7 | 2.6e-5 | 2.6e-4 | 0.000 | 45/160 | 115 |
| A1 | 31 | 3e-7 | 3.8e-5 | 7.6e-4 | 0.001 | 33/160 | 127 |
| A2 | 30 | 3e-7 | 3.5e-5 | 4.9e-4 | 0.000 | 40/160 | 120 |
| A3 | 31 | 3e-7 | 3.2e-5 | 3.4e-4 | 0.002 | 36/160 | 124 |
| A4 | 27 | 3e-7 | 2.5e-5 | 3.0e-4 | 0.000 | 50/160 | 110 |

Grad normy nejsou logované (chybějící instrumentace, poznamenáno).

## LoRA delta-to-init (float64, trace identita)

| Arm | ‖ΔW‖_F (B@A) | ‖lora_A‖_F (kontext) |
|---|---|---|
| A0 | 0.00852 | 36.66 |
| A1 | 0.01018 | 36.66 |
| A2 | 0.00953 | 36.66 |
| A3 | 0.01035 | 36.66 |
| A4 | 0.00875 | 36.66 |

Cross-arm `cosine_delta_to_init_BA` = 0.835–0.885 (A0×A4 = 0.835). Updaty
jsou **nenulové a vzájemně odlišné** — ale odlišnost pochází z rollout
nedeterminismu, ne z reward designu (rewardy byly identické).

## Referenční kotva: `pure_stage3_2ep_20260719_221918`

| Epocha | Steps | KL mean | Loss mean |
|---|---|---|---|
| 1 | 58 | 1.3e-4 | 0.217 |
| 2 | 69 | 1.2e-3 | 0.063 |

Dvouepochový běh (127 steps, KL o 1.5–2 řády vyšší než Round-1) vyprodukoval
měřitelnou změnu oficiálního evalu: **−1.14 pp na n=1661** (zhoršení). To je
horní odhad toho, co ~30 steps @ KL 3e-5 může udělat: prakticky nic.

## Behaviorální potvrzení slabého update (z EVALUATION_AUDIT)

- A0 produkuje **identickou sekvenci predicted calls jako C0 na 326/500
  (65 %)** úloh.
- A0 vs A4: identické call sekvence na 355/500 (71 %).

## Závěr

1. Round-1 update byl **příliš slabý na behaviorální změnu**: ~30 optimizer
   steps, LR 3e-7, KL ~3e-5, žádný clipping, 21–31 % dead groups. Win Rate
   rozdíly ±2–4 pp jsou konzistentní s rollout šumem téže politiky.
2. Tvrzení „A0/A4 update byl slabý" platí; tvrzení „A0/A4 updaty jsou stejné"
   (cosine ~1) je REJECTED — byl to init artefakt (viz IMPLEMENTATION_BUG_AUDIT.md).
3. I silnější update (pure_stage3, 2 epochy) NESTFUL nezlepšil, ale zhoršil —
   síla update tedy není jediný problém; viz DATA_TRANSFER_AUDIT.md.
