# PILOT3_PROVENANCE_AUDIT

**Status:** `EXACT_FIRST_300_BYTES`

Subset is byte-identical to the parent's first 300 lines. Provenance fully verified.

## Compared artifacts

- parent: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\reports\pilot3_provenance\_git_revisions\train_grpo_pilot3@e83f57de.jsonl`
- subset: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\runpod_pilot3\train_nestful500_from_zip\train_nestful500\train_subset_300.jsonl`
- parent rows: 600
- subset rows: 300

## Level 1 — byte-level prefix

- n_lines_taken: `300`
- parent_prefix_sha256: `8b0a16a39a1b5815ae768ac4cd57a9941114426c91465e28783dfad941939c69`
- subset_sha256: `8b0a16a39a1b5815ae768ac4cd57a9941114426c91465e28783dfad941939c69`
- exact_bytes_match: `True`
- match_after_trailing_newline_normalization: `False`
- note: ``

## Level 2 — multi-key identity overlap

| key | subset defined | overlap (parent any) | overlap (parent first N) |
|---|---:|---:|---:|
| `sample_id` | 300 | 300 | 300 |
| `task_id` | 0 | 0 | 0 |
| `semantic_program_id` | 300 | 295 | 295 |
| `semantic_program_family` | 300 | 295 | 295 |
| `graph_template_id` | 300 | 143 | 143 |
| `generation_cell_id` | 300 | 36 | 36 |

## Level 3/4 — canonical content match

- `exact`: 300
- `order_insensitive`: 0
- `semantic`: 0
- `question_only`: 0
- `calls_only`: 0
- `none`: 0

- matched total: 300 / 300
- matched inside parent's first N: 300
- matched positions monotonic: True
- matched positions are identity prefix: True

## Artifact inventory

- `experiments/targeted_tool_data_factory/outputs/runpod_pilot3/train_nestful500_from_zip/train_nestful500/train_subset_300.jsonl` rows=300 sha256=8b0a16a39a1b5815… first=ttdf_0982b2624c98 last=ttdf_30a35ee0f7b0
- `experiments/targeted_tool_data_factory/reports/pilot3_provenance/_git_revisions/train_grpo_pilot3@e83f57de.jsonl` rows=600 sha256=b1bf1d7e24e71521… first=ttdf_0982b2624c98 last=ttdf_15b13550bf04
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\pilot4_smoke\freeze_manifest.json` rows=None sha256=477c65d69730ec4a… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\pilot4_smoke\MANIFEST.sha256.json` rows=None sha256=de016209a6c86e69… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\pilot4_smoke\split_manifest.json` rows=None sha256=a79c192b8aa6e958… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\runpod_pilot2\phase1_canary_from_zip\eval\C0_heldout80\eval_manifest.json` rows=None sha256=7b674f34e17a581c… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\runpod_pilot2\phase1_canary_from_zip\eval\C0_nestful500\eval_manifest.json` rows=None sha256=e8769f61dde3b0e5… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\runpod_pilot2\phase1_canary_from_zip\eval\C1_heldout80\eval_manifest.json` rows=None sha256=6b471032df7e9f16… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\runpod_pilot2\phase1_canary_from_zip\eval\C1_nestful500\eval_manifest.json` rows=None sha256=55553835ee7eb4c7… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\runpod_pilot2\signal_probe_from_zip\signal_probe\manifest_p2_0.json` rows=None sha256=6ab298e57fc39487… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\runpod_pilot2\signal_probe_from_zip\signal_probe\manifest_p2_1.json` rows=None sha256=20c85c5e50b7dbcc… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\runpod_pilot2\signal_probe_from_zip\signal_probe\manifest_p2_2.json` rows=None sha256=cf0f31c05d9cd907… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\runpod_pilot2\signal_probe_from_zip\signal_probe\manifest_p2_3.json` rows=None sha256=f8466b2347ecb53e… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\runpod_pilot2\signal_probe_from_zip\signal_probe\manifest_p3_0.json` rows=None sha256=1acfafa4085de428… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\runpod_pilot2\signal_probe_from_zip\signal_probe\manifest_p3_1.json` rows=None sha256=abf423c7364b46f2… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\runpod_pilot2\signal_probe_from_zip\signal_probe\manifest_p3_2.json` rows=None sha256=6b181d2bcdc60a31… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\runpod_pilot2\signal_probe_from_zip\signal_probe\manifest_p3_3.json` rows=None sha256=e7dc4b65bc10a12a… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\runpod_pilot3\train_nestful500_from_zip\train_nestful500\eval\D1_nestful500\eval_manifest.json` rows=None sha256=8cf0d8f52674b016… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\runpod_pilot3_from_zip2\runpod_pilot3\eval_C0_nestful500_vllm_matched_v2\eval_manifest.json` rows=None sha256=62f5721063ccf026… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\runpod_pilot3_from_zip2\runpod_pilot3\train_nestful500\eval\D1_nestful500\eval_manifest.json` rows=None sha256=8cf0d8f52674b016… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\selected\export_phaseA\manifest_phaseA.json` rows=None sha256=be07ad1cb791a723… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\selected\export_pilot1\manifest_pilot1.json` rows=None sha256=407b561f27c9f136… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\selected\export_pilot2\manifest_pilot2.json` rows=None sha256=032afcbb7955390e… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\selected\export_pilot3\manifest_pilot3.json` rows=None sha256=35722de2ce7f675d… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\selected\export_vsmoke\manifest_vsmoke.json` rows=None sha256=887175b89441eefa… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\reports\pilot3_forensics\ANALYSIS_RUN_MANIFEST.json` rows=None sha256=6da881ae44b89501… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\reports\pilot3_forensics\INPUT_MANIFEST.json` rows=None sha256=4b7b5e79d45ed509… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\runpod_bundle_pilot2\MANIFEST.sha256.json` rows=None sha256=6a359128e886f59e… first=None last=None
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\runpod_bundle_pilot3\MANIFEST.sha256.json` rows=None sha256=850b40395f8fa6a0… first=None last=None

## Alternative parents considered

- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\selected\export_phaseA\grpo_train_ready_phaseA.jsonl` → DIFFERENT_PARENT_EXPORT (0 matched)
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\selected\export_phaseA\train_grpo_phaseA.jsonl` → DIFFERENT_PARENT_EXPORT (0 matched)
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\selected\export_pilot1\grpo_train_ready_pilot1.jsonl` → DIFFERENT_PARENT_EXPORT (0 matched)
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\selected\export_pilot1\train_grpo_pilot1.jsonl` → DIFFERENT_PARENT_EXPORT (0 matched)
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\selected\export_pilot2\grpo_train_ready_pilot2.jsonl` → PARTIAL_SEMANTIC_MATCH (58 matched)
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\selected\export_pilot2\train_grpo_pilot2.jsonl` → PARTIAL_SEMANTIC_MATCH (23 matched)
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\selected\export_pilot3\grpo_train_ready_pilot3.jsonl` → PARTIAL_SEMANTIC_MATCH (88 matched)
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\selected\export_pilot3\train_grpo_pilot3.jsonl` → PARTIAL_SEMANTIC_MATCH (62 matched)
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\selected\export_vsmoke\grpo_train_ready_vsmoke.jsonl` → DIFFERENT_PARENT_EXPORT (0 matched)
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\selected\export_vsmoke\train_grpo_vsmoke.jsonl` → DIFFERENT_PARENT_EXPORT (0 matched)
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\runpod_bundle_pilot2\data\train_grpo_pilot2.jsonl` → PARTIAL_SEMANTIC_MATCH (23 matched)
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\runpod_bundle_pilot3\data\train_grpo_pilot3.jsonl` → PARTIAL_SEMANTIC_MATCH (62 matched)
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\reports\pilot3_provenance\_git_revisions\train_grpo_pilot3@735edd6f.jsonl` → PARTIAL_SEMANTIC_MATCH (62 matched)
- `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\reports\pilot3_provenance\_git_revisions\train_grpo_pilot2@0d0f6663.jsonl` → PARTIAL_SEMANTIC_MATCH (23 matched)

## Interpretation rules applied

- Low `sample_id` overlap alone is NOT evidence of a different training dataset.
- Export ids depend on generator seed/attempt counters and change on regeneration.
- Only byte-level or canonical content match is treated as identity evidence.

