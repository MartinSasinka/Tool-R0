import hashlib, json, os, glob

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
V3 = os.path.join(REPO, "experiments", "nestful_synthetic_curriculum_v3")
A = os.path.join(V3, "outputs", "curriculum_v3_1", "filtered")
B = os.path.join(REPO, "experiments", "nestful_mtgrpo_minimal", "data", "filtered_toolr0_synthetic")
RUNS = os.path.join(V3, "outputs", "runs")

def h(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]

a_hashes = {fn: h(os.path.join(A, fn)) for fn in os.listdir(A) if fn.endswith(".jsonl")}
b_hashes = {fn: h(os.path.join(B, fn)) for fn in os.listdir(B) if fn.endswith(".jsonl")}
print("A hashes:", a_hashes)
print("B hashes:", b_hashes)
print()
for run in sorted(os.listdir(RUNS)):
    db = os.path.join(RUNS, run, "data_base")
    if not os.path.isdir(db):
        continue
    for fn in sorted(os.listdir(db)):
        p = os.path.join(db, fn)
        hh = h(p)
        src = [k for k, v in a_hashes.items() if v == hh] or [k for k, v in b_hashes.items() if v == hh]
        print(f"{run}/data_base/{fn}: {hh} matches={src or 'NOTHING'}")
print()
# peek formats
r = os.path.join(RUNS, "20260708_212347_v3_1")
print("=== train_summary stage_3 epoch_1 ===")
print(open(os.path.join(r, "stage_3", "epoch_1", "train_summary.json"), encoding="utf-8").read()[:2500])
print("=== first + last line of train_log.jsonl ===")
lines = open(os.path.join(r, "stage_3", "epoch_1", "train_log.jsonl"), encoding="utf-8").read().splitlines()
print("n_lines:", len(lines))
print(lines[0][:1500])
print("...")
print(lines[-1][:1500])
print("=== curriculum_summary ===")
print(open(os.path.join(r, "curriculum_summary.jsonl"), encoding="utf-8").read()[:1500])
print("=== val_eval metrics.json (internal) ===")
print(open(os.path.join(r, "stage_3", "epoch_1", "val_eval", "metrics.json"), encoding="utf-8").read()[:2500])
print("=== config_used.json (checkpoint) keys ===")
cfg = json.load(open(os.path.join(r, "stage_3", "checkpoints", "adapter_epoch_1", "config_used.json"), encoding="utf-8"))
print(json.dumps(cfg, indent=1)[:3000])
