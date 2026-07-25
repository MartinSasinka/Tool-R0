# CONFIG_PARITY — parita konfigurací C0 / A0–A4

Zdroj: `analysis/a01_config_parity.json` (62 plochých config klíčů per arm).

## Výsledek: konfigurace jsou identické až na identitu armu

Jediné rozdílné klíče (všechny očekávané „identity fields"):

| Klíč | Poznámka |
|---|---|
| `description` | textový popis armu |
| `reward.train_policy` | A0=`execution_aware_v3_2_dense`, A1–A4=`reward_ablation_<ARM>` — **deklarovaná** reward, do tréninku se nedostala (viz IMPLEMENTATION_BUG_AUDIT.md) |
| `reward_id` | ID armu |
| `wandb.extra_tags`, `wandb.run_name` | jména |

`unexpected_config_diff_count = 0`.

## Hash parita (identická napříč všemi 5 army)

| Artefakt | SHA256 (prefix) |
|---|---|
| dataset_hash (train subset 160) | `b64d3ec2…` |
| eval_subset_hash (NESTFUL 500) | `90e018f2…` |
| reward_spec_hash | `4ee7dcbe…` |
| executor_hash | `f945b18c…` |
| registry_version | 5.0.2 |

## Seed a commit

- Seed: **20260724 ve všech armech** (stejný!)
- Git commit: `fdf8e579…` ve všech armech

## Důsledek

Protože army sdílely data, executor, seed, commit — a (kvůli dispatch bugu)
i reward funkci — byl Round-1 fakticky **5× replikace téhož tréninkového
konfigu**, lišící se pouze nedeterminismem vLLM rolloutů. Rozptyl Win Rate
53.0–57.4 % na n=500 je tedy **odhad replikačního šumu jedné konfigurace**,
ne efekt reward designu.
