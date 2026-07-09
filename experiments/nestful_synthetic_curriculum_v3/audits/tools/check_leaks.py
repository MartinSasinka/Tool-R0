import json, re, os
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
B = os.path.join(REPO, "experiments", "nestful_mtgrpo_minimal", "data", "filtered_toolr0_synthetic")

pats = ("cluster", "stage1", "stage2", "stage3", "stage4")
for fn in ("epoch_1_1call.jsonl", "epoch_4_4call.jsonl", "epoch_5_5call.jsonl", "epoch_6_6call.jsonl"):
    p = os.path.join(B, fn)
    with open(p, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            row = json.loads(line)
            q = (row.get("input") or "").lower()
            hits = [x for x in pats if x in q]
            if hits:
                print(f"{fn}:{i} {hits} :: {row['input'][:220]}")
print()
# null answer in epoch_4
with open(os.path.join(B, "epoch_4_4call.jsonl"), encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        row = json.loads(line)
        if row.get("gold_answer") is None:
            print(f"NULL gold_answer epoch_4 row {i}: {row['sample_id']} :: {row['input'][:150]}")
print()
# sample of var-ref gold answers
with open(os.path.join(B, "epoch_2_2call.jsonl"), encoding="utf-8") as fh:
    shown = 0
    for i, line in enumerate(fh):
        row = json.loads(line)
        ga = json.dumps(row.get("gold_answer"), ensure_ascii=False, default=str)
        if re.search(r"\$var_?\d+", ga) and shown < 3:
            print(f"varref epoch_2 row {i}: gold_answer={ga[:150]}")
            shown += 1
