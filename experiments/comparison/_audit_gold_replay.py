import json, sys, os, random
root = os.path.join(os.path.dirname(__file__), "..", "nestful_mtgrpo_minimal")
root = os.path.abspath(root)
sys.path.insert(0, root)
import nestful_official_score as nos

nf = os.path.join(root, "data", "NESTFUL-main", "data_v2", "nestful_data.jsonl")
raw = []
for l in open(nf, encoding="utf-8"):
    l = l.strip()
    if l:
        raw.append(json.loads(l))
random.seed(7)
sample = random.sample(raw, 150)
items = []
for r in sample:
    out = r["output"]
    if isinstance(out, str):
        out = json.loads(out)
    items.append(nos.build_item(out, r))  # predictions == gold calls
res = nos.score_items_per_sample(items, win_rate=True)
n = len(res)
win = sum(x["official_win"] for x in res) / n
ex = sum(1 for x in res if x["executable"]) / n
full = sum(x["official_full_match"] for x in res) / n
parse = sum(1 for x in res if x["parse_valid"]) / n
print(f"GOLD REPLAY n={n}: official_win={win:.3f} executable={ex:.3f} full_match={full:.3f} parse_valid={parse:.3f}")
fails = [(s.get("sample_id"), r) for s, r in zip(sample, res) if r["official_win"] < 1.0]
print("num gold-replay win<1:", len(fails))
for sid, r in fails[:10]:
    print("  ", sid, "win", r["official_win"], "exec", r["executable"], "full", r["official_full_match"], "err", r["execution_error"])
