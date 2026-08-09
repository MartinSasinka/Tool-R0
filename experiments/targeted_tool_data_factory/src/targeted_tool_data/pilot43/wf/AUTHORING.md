# Authoring Pilot4.3 workflow blueprints

A blueprint module lives in this package and exposes one function:

```python
def blueprints() -> list[Blueprint]: ...
```

`targeted_tool_data.pilot43.blueprints.all_blueprints()` imports every module here,
validates every plan fail-closed, and rejects the whole registry if any plan is
invalid. So a broken module blocks the entire pilot: verify before you finish.

## The contract

A `Plan` is an explicit capability DAG. Each `Step` names a capability and, per
parameter position, where the value comes from:

* `"role_name"` — a fact the user states in the query (a `Role` of the plan),
* `"@n2"` — the output of an earlier step.

Nothing is labelled after the fact. Call count is `len(steps)`, the structural
pattern is classified from the built graph, capability coverage is counted from
the bound ops. If you want a diamond, you write a diamond.

## Worked example

```python
from ..blueprints import Blueprint, Plan, Role, Step

def blueprints():
    return [Blueprint(
        workflow_id="inventory.replenishment_check",
        domain="inventory",
        natural_user_goal="decide whether a depot must reorder a part",
        target_description="whether projected stock falls below the safety level",
        value_generator_id="inventory.replenishment",
        query_asset_family="inventory_depot",
        entity_family="warehouse",
        boolean_balancing_strategy="threshold_band",
        hard_distractor_families=("arithmetic", "comparison"),
        plans=(
            Plan(
                plan_id="replen.v4",
                intent="daily usage over the lead time versus stock on hand",
                roles=(
                    Role("daily_usage", "quantity_units", "units consumed per day"),
                    Role("lead_days", "count_small", "days until the delivery"),
                    Role("on_hand", "quantity_stock", "units in the depot"),
                    Role("safety_level", "threshold_count", "safety stock"),
                ),
                steps=(
                    Step("n1", "arithmetic.multiply", ("daily_usage", "lead_days"),
                         "consumption during the lead time"),
                    Step("n2", "arithmetic.subtract", ("on_hand", "@n1"),
                         "projected stock at delivery"),
                    Step("n3", "comparison.less_than", ("@n2", "safety_level"),
                         "below the safety level?"),
                    Step("n4", "boolean.identity", ("@n3",), "decision"),
                ),
                sink="n4",
            ),
        ),
    )]
```

## Hard rules (the validator enforces all of these)

1. Every role must be used by at least one step.
2. Every step must have a path to the sink; no dead branches.
3. `@ref` may only point at an *earlier* step.
4. The capability must have an op of exactly the arity you pass. Check with
   `python scripts/pilot43_op_dump.py <capability-prefix>`.
5. Semantic types must line up. `Money` may feed `Money`/`GenericScalar`/`Quantity`
   parameters; a physical quantity (hours, km, kg…) only feeds a parameter of the
   same quantity or an explicit converter. `Percentage` (20) and `Ratio` (0.2) are
   different types on purpose. The op dump prints the resolved signature.
6. `workflow_id` must be globally unique and shaped `family.name`.

## Quality rules (not machine-checked, but the pilot fails its gates without them)

* **Real dependency depth.** Aim for the plan set of one blueprint to span short
  and long: e.g. plans with 2, 4, 6 and 8 steps. A blueprint whose only plan is a
  3-step linear chain is close to worthless for this dataset.
* **Real topologies.** Use `@ref` reuse (same output consumed by two later steps),
  late references (an early output consumed 3+ steps later), several join nodes,
  and aggregations that consume aggregations. Do not write everything as a chain.
* **No `add → increase_by_percent → compare` monoculture.** No exact primitive
  sequence may exceed 3 % of the dataset, so every module should contribute
  genuinely different capability sequences.
* **Domain-true capabilities.** A workflow named `path_processing` must actually
  bind `path.*` ops. Renaming an arithmetic chain is exactly the Pilot4.2 defect
  this pilot exists to fix.
* **Answer-type spread.** Across your module produce float, integer, boolean,
  string, list, object and category sinks — not only float and boolean.
* **Boolean plans need a balancing strategy.** Set
  `boolean_balancing_strategy="threshold_band"` and give the predicate its
  threshold through a `threshold_*` role; the builder calibrates that literal after
  execution so True/False comes out balanced. Never hand a predicate two
  independent roles and hope.
* **Roles must read like facts a user would state.** Pick the value hint that
  matches the domain (`money_price`, `path_file`, `url_link`, `date_iso`,
  `record_list`, `mapping_rates`, …), never `generic_value` for a domain quantity.

## Available value hints

Run:

```bash
python -c "import sys; sys.path.insert(0,'src'); from targeted_tool_data.pilot43 import values as V; print(sorted(V.HINTS))"
```

`threshold_*`, `cut_*`, `range_*` and `tolerance_*` hints are *calibrated*: the
builder overwrites them after execution to balance boolean and category sinks.
Use them only as the comparison operand of a predicate step.

## Verify before finishing

```bash
python scripts/pilot43_dev_check.py --only <your.family>          # must show ok=N/N
python scripts/pilot43_dev_check.py                              # whole registry
```

`ok=8/8` on every plan, zero build errors, and a call-count / answer-type /
pattern spread you can defend. If a core module (`ops`, `values`, `build`,
`program`) blocks you, report it instead of editing it — those files are shared.
