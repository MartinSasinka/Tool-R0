import json
from pathlib import Path
from collections import Counter

rejected_path = Path("curricullum/data/rejected_toolr0/epoch_6_rejected.json")
rejected = json.loads(rejected_path.read_text(encoding="utf-8"))

reasons = Counter(r.get("reason") for r in rejected)

print(f"Total rejected: {len(rejected)}")
print("\nAll rejection reasons:")
for reason, count in reasons.most_common(30):
    pct = count / len(rejected) * 100
    print(f"  {count:5d}  ({pct:5.1f}%)  {reason}")

# Categorize
structural = sum(c for r, c in reasons.items() if r and not r.startswith("exec_ibm"))
ibm_exec = sum(c for r, c in reasons.items() if r and r.startswith("exec_ibm"))
print(f"\nStructural failures: {structural} ({structural/len(rejected)*100:.1f}%)")
print(f"IBM exec failures:   {ibm_exec} ({ibm_exec/len(rejected)*100:.1f}%)")
