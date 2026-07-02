import json
from pathlib import Path
from collections import Counter

raw = json.loads(Path("curricullum/data/raw_toolr0/epoch_6_candidates.json").read_text(encoding="utf-8"))
rejected = json.loads(Path("curricullum/data/rejected_toolr0/epoch_6_rejected.json").read_text(encoding="utf-8"))

print(f"Total raw: {len(raw)}")

# Check ordering - topup vs original
topup_indices = [i for i, r in enumerate(raw) if "topup" in r.get("meta", {}).get("raw_id", "").lower()]
original_indices = [i for i, r in enumerate(raw) if "topup" not in r.get("meta", {}).get("raw_id", "").lower()]
print(f"Original count: {len(original_indices)}, first_idx={original_indices[0] if original_indices else 'N/A'}, last_idx={original_indices[-1] if original_indices else 'N/A'}")
print(f"Topup count:    {len(topup_indices)}, first_idx={topup_indices[0] if topup_indices else 'N/A'}")

# Rejection reasons
reasons = Counter(r.get("reason") for r in rejected)
print(f"\nTotal rejected: {len(rejected)}")
print("Top reasons:")
for reason, count in reasons.most_common(10):
    print(f"  {count:5d}  {reason}")

# How many rejected are duplicates?
dup_reasons = sum(c for r, c in reasons.items() if r and "duplicate" in str(r).lower())
print(f"\nDuplicate rejections: {dup_reasons}")
