# Pilot 4.3 targeted data factory

Aktuální generátor a validátor tool-use dat. Starší Pilot 1–4.2 implementace,
RunPod bundly a jednorázové reporty byly odstraněny; podporovaná je pouze větev
Pilot 4.3 a výběr `NESTFUL_PROFILE_1000`.

## Instalace

```bash
cd experiments/targeted_tool_data_factory
python -m pip install -e .
```

Pro OpenRouter rendering nastav `OPENROUTER_API_KEY`. Modely a rozpočtové limity
jsou v `configs/pilot4_3_openrouter.yaml`.

## Generování a validace

CLI vypíše všechny dostupné kroky a argumenty:

```bash
targeted-data --help
```

Hlavní deterministická část pipeline:

```bash
targeted-data build-workflow-registry-v3
targeted-data validate-primitive-registry-v3
targeted-data build-target-profile-v3
targeted-data generate-pilot43-semantic --stage full --resume
targeted-data validate-pilot43-semantic --resume
targeted-data shortlist-pilot43
targeted-data run-pilot43-v4 --resume
```

Rendering a následné validační brány:

```bash
targeted-data freeze-pilot43-selectable
targeted-data allocate-pilot43-render
targeted-data render-pilot43-openrouter --resume
targeted-data validate-pilot43-queries
targeted-data select-pilot43
targeted-data build-pilot43-nested-subsets
targeted-data gate-pilot43
targeted-data independent-audit-pilot43
targeted-data freeze-pilot43
```

Výstupní adresář lze u všech kroků nastavit přes `--output-dir`. Výchozí pracovní
adresář pipeline je `outputs/pilot4_3_nestful_final`.

## Profilový dataset pro GRPO

Ze zmrazeného Pilot 4.3 poolu sestaví přesný profilový výběr:

```bash
python scripts/build_nestful_profile_1000.py \
  --pilot-out outputs/pilot4_3_nestful_final \
  --out-dir outputs/pilot4_3_nestful_profile_1000
```

Repo uchovává dvě přímo použitelné datové sady:

- `outputs/pilot4_3_nestful_profile_1000/train_nestful_profile_1000.jsonl`
- `outputs/pilot4_3_nestful_profile_1000/train_nestful_enrichment_500.jsonl`

Před každým tréninkem launch skript automaticky spustí úplný gold-replay přes
`trainer_adapter_p43/preflight_gold_replay.py`.

## Testy

```bash
python -m pytest tests -q
```
