import json, glob, os
base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
files = glob.glob(os.path.join(base, "**", "metrics_official.json"), recursive=True)
print("label | win | full | part | f1f | f1p | n | parse_err")
def g(d, k):
    v = d.get(k)
    return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
for f in sorted(files):
    d = json.load(open(f, encoding="utf-8"))
    rel = os.path.relpath(os.path.dirname(f), base)
    print(f"{rel:62s} win={g(d,'win_rate')} full={g(d,'full_sequence_accuracy')} part={g(d,'partial_sequence_accuracy')} f1f={g(d,'f1_func')} f1p={g(d,'f1_param')} n={d.get('num_examples')} perr={d.get('num_pred_parsing_errors')}")
