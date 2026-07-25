# Held-out Stage-3 166 plan

- Held-out rows: **166** (expected 166)
- Disjoint from train 160: **True**
- SHA-256: `b74662a9f0adf879aa42ed0a293f797aa2c37236ae92107486486bb6091da287`

## Optional inference (not run in offline audit phase)

```powershell
python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_local_heldout_eval.py --heldout C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\reports\reward_ablation\offline_audit\heldout_stage3_166.jsonl --checkpoint C0|A0|A4 --allow-model-inference
```