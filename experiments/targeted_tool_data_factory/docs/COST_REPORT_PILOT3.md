# Pilot3 cost & time estimate

Generated 2026-07-27 03:15 UTC.

| phase | estimate | notes |
|---|---|---|
| CPU generate + validate (~8k candidates) | **4–10 h** wall | 1 workstation CPU; B2 expand may add 1–3 h |
| OpenRouter paraphrase (≤4500 req, ≤$5) | **1–3 h** / **≤ $5** | mistral-small; measured pilot2 ≈ $0.00003/req |
| Select / split / export / report | **5–15 min** | CPU |
| Gold-replay preflight (1000 tasks) | **2–10 min** | factory executor |
| RunPod signal probe (600×4 + P3×8, 4 GPU) | **2–5 h** | BF16 Qwen3-4B, no training |
| Subsequent GRPO (Phase-1 ~400, 8 gens, small budget) | **3–8 h** | not auto-started |
| Full D1-style 600-train GRPO (later decision) | **8–20 h** | not part of this freeze |

OpenRouter spend this run: $0.076372
(requests=4491).
