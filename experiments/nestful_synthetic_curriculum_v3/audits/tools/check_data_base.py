import json, os, hashlib
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
V3 = os.path.join(REPO, "experiments", "nestful_synthetic_curriculum_v3")
RUNS = os.path.join(V3, "outputs", "runs")
A = os.path.join(V3, "outputs", "curriculum_v3_1", "filtered")

def rowhashes(p):
    hs = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                hs.append(row.get("sample_id"))
    return hs

for run in ("20260708_212347_v3_1", "20260702_112150"):
    p = os.path.join(RUNS, run, "data_base", "epoch_2_2call.jsonl")
    sz = os.path.getsize(p)
    with open(p, encoding="utf-8", errors="replace") as fh:
        head = fh.read(300)
    print(f"{run}: size={sz} head={head[:200]!r}")
    if sz > 1000:
        ids = rowhashes(p)
        print(f"  rows={len(ids)} first_id={ids[0]}")
        a_ids = rowhashes(os.path.join(A, "stage2_2call_dependency.jsonl"))
        inter = len(set(ids) & set(a_ids))
        print(f"  sample_id overlap with A/stage2: {inter}/{len(ids)}")
    print()
