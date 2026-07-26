import collections
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from targeted_tool_data import registry as reg
from targeted_tool_data.graph import build_program_v2, classify_program_motif
from targeted_tool_data.plausibility import analyze
from targeted_tool_data.executor import execute, ExecutionError
from targeted_tool_data.graph import GraphBuildError
from targeted_tool_data.render import render_query, pick_surfaces, render_calls, TEMPLATE_COUNT
from targeted_tool_data.schemas import GenerationCell
from targeted_tool_data.util import arg_type_of

print("primitives:", len(reg.all_primitives()), "surfaces:", len(reg.all_surfaces()))
print("surface uniqueness errors:", reg.validate_surface_uniqueness())
print("templates:", TEMPLATE_COUNT)
print("registry_hash:", reg.registry_hash()[:16])

ok = collections.Counter()
fail = collections.Counter()
samples = []
for kind in ["float", "int", "bool", "string", "list", "numeric_string"]:
    for motif, cc in [("linear", 2), ("linear", 4), ("fan_in", 4), ("fan_in", 5),
                      ("branch_aggregate", 5)]:
        cell = GenerationCell(
            generation_cell_id=f"T_{cc}call_{motif}", track="A", mode="adaptation",
            call_count=cc, motif=motif, target_skill="s", target_failure="f",
            answer_kind=kind)
        for i in range(24):
            rng = random.Random(f"{kind}:{motif}:{cc}:{i}")
            try:
                prog = build_program_v2(cell, rng)
            except (GraphBuildError, ExecutionError) as e:
                fail[f"{kind}:build:{str(e)[:40]}"] += 1
                continue
            try:
                obs, ans = execute(prog)
            except ExecutionError as e:
                fail[f"{kind}:exec:{str(e)[:40]}"] += 1
                continue
            at = arg_type_of(ans)
            pl = analyze(prog)
            key = f"{kind}->{at} {motif}->{prog.motif} n={len(prog.nodes)}"
            ok[key] += 1
            if len(samples) < 400:
                q, tid, fam = render_query(prog, rng)
                samples.append((kind, motif, at, prog.motif, pl["plausibility_class"], ans, q))

for k, v in sorted(ok.items()):
    print(f"  {v:4d}  {k}")
print("--- failures")
for k, v in fail.most_common(14):
    print(f"  {v:4d}  {k}")
print("--- plausibility", collections.Counter(s[4] for s in samples))
print("--- sample queries")
random.Random(1).shuffle(samples)
for s in samples[:14]:
    print(f"[{s[0]}/{s[3]}/{s[4]}] ans={s[5]!r}")
    print("   ", s[6][:250])
