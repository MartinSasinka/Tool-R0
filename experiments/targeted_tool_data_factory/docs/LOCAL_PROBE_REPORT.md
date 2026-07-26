# Local Qwen3-4B probe report

Generated 2026-07-26 12:03 UTC.

The probe is a **difficulty proxy only**. It never determines the oracle, never changes the structural split and never gates task acceptance.

## Status: `NOT_RUN_LOCAL`

No OpenAI-compatible endpoint answered at the configured base URL, so the cascade was skipped. Pilot2 generation was **not** blocked by this, which is the intended behaviour.

To run it later, start LM Studio (or any OpenAI-compatible server) with `Qwen/Qwen3-4B-Instruct-2507` and run, from the repo root:

```powershell
$env:LOCAL_LLM_BASE_URL = 'http://127.0.0.1:1234/v1'
$env:LOCAL_LLM_MODEL    = 'qwen/qwen3-4b-2507'
cd experiments\targeted_tool_data_factory\src
& ..\..\nestful_synthetic_curriculum_v3\.venv\Scripts\python.exe -X utf8 -m targeted_tool_data.cli probe --config ../configs/pilot2_local.yaml --version pilot2 --seed 20260726
```

The probe is resumable and content-hash cached, so an interrupted run continues where it stopped.
