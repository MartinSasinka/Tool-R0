# Tool Family Realism Report

Status: **partial_tool_realism**

- NESTFUL tasks: 1861
- Synthetic v3 tasks: 1030
- Tool name diversity (nestful / v3): 4202 / 25
- Tool family overlap (Jaccard): 0.3230
- Tool sequence bigram overlap: 0.0091
- Tool sequence trigram overlap: 0.0417
- Mean distractor tools (nestful / v3): 8.56 / 7.56

## Output / answer type
- NESTFUL output types: {'scalar': 1531, 'list': 101, 'boolean': 39, 'string': 173, 'object': 17}
- v3 output types: {'scalar': 857, 'boolean': 88, 'string': 61, 'list': 24}

## Status meanings
- `math_only`: only math/distractor tools — pipeline validation only
- `mixed_synthetic_prototype`: multi-family synthetic tools, low NESTFUL overlap
- `partial_tool_realism`: improved diversity/overlap — pilot with caveats
- `final_ready`: high overlap — suitable for transfer claims

Current classification: **partial_tool_realism**
- scalar output share: 83.2%
- non-scalar output share: 16.8%
