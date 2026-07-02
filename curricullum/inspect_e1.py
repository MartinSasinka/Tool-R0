import json
from pathlib import Path

path = Path("curricullum/data/filtered_toolr0_synthetic/epoch_1_1call.jsonl")
lines = [l for l in path.open(encoding="utf-8") if l.strip()]
sizes = sorted(enumerate(lines), key=lambda x: len(x[1]), reverse=True)

print(f"Total lines: {len(lines)}")
print(f"File size: {path.stat().st_size / 1024 / 1024:.1f} MB")
print(f"Max line: {len(sizes[0][1]):,} chars ({len(sizes[0][1])//1024} KB)")
print(f"Top 5 sizes: {[len(s[1]) for s in sizes[:5]]}")
print(f"Median size: {len(sizes[len(sizes)//2][1]):,} chars")

big = [(i, len(l)) for i, l in enumerate(lines) if len(l) > 50_000]
print(f"\nLines >50KB: {len(big)}")
for idx, sz in big[:5]:
    d = json.loads(lines[idx])
    ga = str(d.get("gold_answer", ""))
    print(f"  line {idx}: {sz:,} chars | gold_answer ({len(ga)} chars): {ga[:200]}")
