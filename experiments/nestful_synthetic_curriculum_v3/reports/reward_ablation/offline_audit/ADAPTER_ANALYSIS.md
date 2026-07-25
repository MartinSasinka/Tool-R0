# Adapter analysis (CPU, LoRA only)

Primary metric: `cosine_delta_to_init_BA` — cosine over the effective
update DeltaW = B@A (lora_B is zero-initialized, so B@A is exactly the
delta to initialization). The raw flat cosine over absolute adapter
weights is dominated by the shared seeded lora_A init and reads ~1.0
for any two same-seed runs; it is kept only as a diagnostic.

**A0 vs A4**: delta-to-init cosine=0.835053045345478, raw flat cosine=0.9999999567861761 (init artifact), rel_dist_delta=0.5749578374193427