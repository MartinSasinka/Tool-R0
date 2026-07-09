import json, os, hashlib
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
V3 = os.path.join(REPO, "experiments", "nestful_synthetic_curriculum_v3")
A = os.path.join(V3, "outputs", "curriculum_v3_1", "filtered")
RUNS = os.path.join(V3, "outputs", "runs")

def canon_rows(p):
    out = {}
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                sid = row.get("sample_id") or row.get("task_id")
                out[sid] = hashlib.sha1(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return out

pairs = [
    ("stage1_1call_atomic.jsonl", "epoch_1_1call.jsonl"),
    ("stage2_2call_dependency.jsonl", "epoch_2_2call.jsonl"),
    ("stage3_3call_composition.jsonl", "epoch_3_3call.jsonl"),
    ("stage4_4to6call_persistence.jsonl", "epoch_4_4call.jsonl"),
]
for run in ("20260708_212347_v3_1", "20260707_103035_v3_1"):
    print("=====", run)
    for a_fn, r_fn in pairs:
        a_rows = canon_rows(os.path.join(A, a_fn))
        r_rows = canon_rows(os.path.join(RUNS, run, "data_base", r_fn))
        same_ids = set(a_rows) == set(r_rows)
        identical = sum(1 for k in r_rows if a_rows.get(k) == r_rows[k])
        print(f"  {a_fn} vs {r_fn}: same_id_set={same_ids} identical_rows={identical}/{len(r_rows)}")
