import json, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(HERE, "DATASET_AUDIT.json"), encoding="utf-8"))
for k, v in d["datasets"].items():
    print(f"{k}: rows={v['rows']} uq={v['unique_questions']} dupq={v['duplicate_question_rows']} "
          f"dupt={v['duplicate_gold_trace_rows']} calls={v['call_count_distribution']} "
          f"nullans={v['null_gold_answers']} varref_ans={v['gold_answer_contains_unresolved_var_ref']} "
          f"tools_str={v['tools_field_is_string_not_list']} leaks={v['prompt_metadata_leaks']}")
print()
print("=== nonzero overlaps ===")
for k, v in d["overlaps"].items():
    if v["question_hash_overlap"] or v["gold_trace_hash_overlap"] or v["sample_id_overlap"]:
        print(f"{k}: q={v['question_hash_overlap']} t={v['gold_trace_hash_overlap']} id={v['sample_id_overlap']}")
print()
print("=== A vs B zero-overlap check (all pairs) ===")
zero = sum(1 for k, v in d["overlaps"].items()
           if k.startswith("A/") and " B/" in k.replace("<->", " ")
           and not (v["question_hash_overlap"] or v["gold_trace_hash_overlap"] or v["sample_id_overlap"]))
tot = sum(1 for k in d["overlaps"] if k.startswith("A/") and "B/" in k)
print(f"A-B pairs with zero overlap: {zero}/{tot}")
