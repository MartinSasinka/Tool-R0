# PLAN_LEAK_EXAMPLES

Deterministically sampled, balanced across query modes.

## `PROCEDURAL_EXPLICIT` — pilot3_train_600_worktree / ttdf_001c7f87d8d1

- lexical coverage: 1.00
- sequence leakage: 1.00
- procedural cues: 12

```text
Please compute 47 percent of 1485. then convert -3 degrees Celsius to Fahrenheit. then add 286 and that result. then compute how many whole times that result fits into 2940. then compute how many whole times that result fits into 2589. and finally compute the ratio of the result of step 1 to that result. What is the final result?
```

## `PROCEDURAL_EXPLICIT` — pilot3_heldout_200 / ttdf_560ee9d6856e

- lexical coverage: 1.00
- sequence leakage: 1.00
- procedural cues: 2

```text
First, convert 18 degrees Celsius to Fahrenheit. Then, subtract 6 from that value.
```

## `PROCEDURAL_PARTIAL` — pilot3_train_600_worktree / ttdf_0001c0bc1f80

- lexical coverage: 0.75
- sequence leakage: 0.88
- procedural cues: 14

```text
Compute the ratio of 675 to 48, after that round that result to the nearest whole number, after that find the remainder of that result divided by 4, after that find the absolute difference between that result and 70, after that decrease that result by 9 percent, after that round that result up to a whole number, after that add 772 and that result, and at the end scale every item of [8, 49, 70] by that result.
```

## `PROCEDURAL_PARTIAL` — pilot3_reserve_200 / ttdf_52e35bd6ee76

- lexical coverage: 0.50
- sequence leakage: 0.12
- procedural cues: 2

```text
What I finally need is to increase the value those steps produce by 10 percent; the preparation is to compute how many whole times 17 fits into 3280. (Background: the device serial is AX-40, irrelevant to the math.)
```

## `SEMI_IMPLICIT` — pilot3_train_600_worktree / ttdf_01f1f3477121

- lexical coverage: 0.40
- sequence leakage: 0.70
- procedural cues: 8

```text
First, round 19.58 up to the nearest whole number. After that, turn -14 degrees Celsius into Fahrenheit. Then, lower that result by 27 percent.  Now, find the average of the results from step 1 and that result. Finally, label that result with the unit 'boxes'.
```

## `SEMI_IMPLICIT` — pilot3_heldout_200 / ttdf_5c13fb461b3a

- lexical coverage: 0.33
- sequence leakage: 0.08
- procedural cues: 4

```text
Round 75.17 down to a whole number, and then increase that result by 22 percent, concluding by find the absolute difference between 264 and that result.
```

## `GOAL_BASED_IMPLICIT` — pilot3_train_600_worktree / ttdf_0a363f4ef90e

- lexical coverage: 0.00
- sequence leakage: 0.00
- procedural cues: 1

```text
Invert 23 to get a new value, then reduce 678 by the percentage that is the result.
```

## `GOAL_BASED_IMPLICIT` — pilot3_train_600_worktree / ttdf_6f8c0f3fb0a4

- lexical coverage: 0.00
- sequence leakage: 0.00
- procedural cues: 3

```text
First, reduce 626.79 to a whole number by rounding down. After that, see if that result can be divided by 6 with no remainder.
```

## `UNCLASSIFIED` — pilot3_train_600_worktree / ttdf_1146308536be

- lexical coverage: 0.50
- sequence leakage: 0.12
- procedural cues: 0

```text
Identify the 1-based position of the greatest item in [34, 10, 23, 1, 48] using this result, increase 1236 by that percent.
```

## `UNCLASSIFIED` — pilot3_reserve_200 / ttdf_55dc4ad32d54

- lexical coverage: 0.50
- sequence leakage: 0.12
- procedural cues: 1

```text
Determine the outcome of this procedure: increase 1703 by 51 percent; multiply 253 by that result. (Background: the device serial is AX-40, irrelevant to the math.)
```

