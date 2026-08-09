# GAINED_LOST_AUDIT

- n_gained=27 n_lost=16
- Small-n warning: do not over-interpret significance.

## Top patterns

- `SUCCESS_ALTERNATIVE_VALID -> FAIL_WRONG_FIRST_TOOL`: gained=0 lost=6 net=-6 support=6
- `FAIL_EXECUTOR_ERROR -> SUCCESS_ALTERNATIVE_VALID`: gained=6 lost=0 net=6 support=6
- `FAIL_WRONG_FIRST_TOOL -> SUCCESS_ALTERNATIVE_VALID`: gained=5 lost=0 net=5 support=5
- `FAIL_WRONG_FIRST_TOOL -> SUCCESS_OTHER_OFFICIAL`: gained=3 lost=0 net=3 support=3
- `FAIL_NO_TOOL_CALL -> SUCCESS_ALTERNATIVE_VALID`: gained=3 lost=0 net=3 support=3
- `SUCCESS_ALTERNATIVE_VALID -> FAIL_WRONG_TOOL_SEQUENCE`: gained=0 lost=2 net=-2 support=2
- `SUCCESS_ALTERNATIVE_VALID -> FAIL_EXECUTOR_ERROR`: gained=0 lost=2 net=-2 support=2
- `FAIL_PARSE_INVALID -> SUCCESS_ALTERNATIVE_VALID`: gained=2 lost=0 net=2 support=2
- `FAIL_EXECUTOR_ERROR -> SUCCESS_STRICT_GOLD`: gained=2 lost=0 net=2 support=2
- `FAIL_WRONG_TOOL_SEQUENCE -> SUCCESS_ALTERNATIVE_VALID`: gained=2 lost=0 net=2 support=2
- `FAIL_WRONG_TOOL_SEQUENCE -> SUCCESS_STRICT_GOLD`: gained=2 lost=0 net=2 support=2
- `SUCCESS_ALTERNATIVE_VALID -> FAIL_NO_TOOL_CALL`: gained=0 lost=1 net=-1 support=1
- `SUCCESS_STRICT_GOLD -> FAIL_EXECUTOR_ERROR`: gained=0 lost=1 net=-1 support=1
- `SUCCESS_STRICT_GOLD -> FAIL_PARSE_INVALID`: gained=0 lost=1 net=-1 support=1
- `SUCCESS_STRICT_GOLD -> FAIL_NO_TOOL_CALL`: gained=0 lost=1 net=-1 support=1
- `FAIL_WRONG_TOOL_SEQUENCE -> SUCCESS_OTHER_OFFICIAL`: gained=1 lost=0 net=1 support=1
- `FAIL_EXECUTOR_ERROR -> SUCCESS_OTHER_OFFICIAL`: gained=1 lost=0 net=1 support=1
- `SUCCESS_OTHER_OFFICIAL -> FAIL_WRONG_FIRST_TOOL`: gained=0 lost=1 net=-1 support=1
- `SUCCESS_ALTERNATIVE_VALID -> FAIL_EXECUTABLE_WRONG_ANSWER`: gained=0 lost=1 net=-1 support=1
- `divergence:TOOL_COUNT_DIFFERENCE`: gained=15 lost=7 net=8 support=22
