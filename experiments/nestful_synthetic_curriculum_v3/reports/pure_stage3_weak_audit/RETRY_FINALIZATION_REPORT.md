# Retry finalization report

**Generated:** 2026-07-23T10:23:00.949347+00:00

## Invalid counts

- Originally invalid pairs: 14
- Pass A originally invalid: 9
- Pass B originally invalid: 5
- Retry validated: 14
- Retry failed: 0
- Still invalid after merge: 0

## Retry configuration

- Model: deepseek/deepseek-v3.2
- Provider: DeepInfra
- Structured JSON Schema: True
- Reasoning: none
- Retry cost USD: 0.008391979999999999
- Retry prompt tokens: 30997
- Retry completion tokens: 3306

## Agreement before vs after

- Before exact agreement: 0.1574468085106383
- After exact agreement: 0.16129032258064516
- Before both-valid n: 235
- After both-valid n: 248
- Before root κ: 0.4103906216204524
- After root κ: 0.41615269373698854

## High-priority handoff

- Before count: 80
- After count: 80
- Added: ['ec654414-df15-4621-9b4d-3570a4ae6864']
- Removed: ['52136fd0-87d3-4086-968b-fa7b8523c333']

## Still invalid task IDs


## Integrity

- Original raw/validated files preserved (see backup manifest).
- Only invalid task/pass pairs were retried.
- Pass B mapping unchanged.

## Final manifest SHA-256

- See `WEAK_AUDIT_FINAL_MANIFEST.json`
- Manifest file SHA-256: 6847cc249179c776a3669fb35c3d921814e7ef61422e20e9590e127b91b45aee