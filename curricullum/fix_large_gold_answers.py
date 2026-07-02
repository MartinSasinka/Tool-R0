"""
Fix oversized gold_answer fields in filtered_toolr0_synthetic JSONL files.
Truncates any gold_answer exceeding MAX_GOLD_ANSWER_CHARS to that limit.
"""
import json
from pathlib import Path

MAX_GOLD_ANSWER_CHARS = 8192
FILTERED_DIR = Path("curricullum/data/filtered_toolr0_synthetic")

for jsonl_file in sorted(FILTERED_DIR.glob("epoch_*.jsonl")):
    lines = [l for l in jsonl_file.open(encoding="utf-8") if l.strip()]
    fixed = 0
    out_lines = []
    for line in lines:
        d = json.loads(line)
        ga = d.get("gold_answer")
        if ga is not None and len(str(ga)) > MAX_GOLD_ANSWER_CHARS:
            d["gold_answer"] = str(ga)[:MAX_GOLD_ANSWER_CHARS]
            fixed += 1
        out_lines.append(json.dumps(d, ensure_ascii=False))

    if fixed > 0:
        jsonl_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        size_mb = jsonl_file.stat().st_size / 1024 / 1024
        print(f"{jsonl_file.name}: fixed {fixed} entries -> {size_mb:.2f} MB")
    else:
        size_mb = jsonl_file.stat().st_size / 1024 / 1024
        print(f"{jsonl_file.name}: OK ({size_mb:.2f} MB)")
