# {{MODEL}} fusion ladder

Derive this from the actual model DAG. Partitions must execute every logical stage
exactly once and honor data, program-order and state dependencies.

## Logical stages and edges

| Stage ID | Formula/rounding | Inputs/outputs/layout | State/cache effect | Current implementation |
|---|---|---|---|---|
| unknown | unknown | unknown | unknown | unknown |

## Runnable partitions

| Version | Partition list / launch boundary | Newly fused edge | Removed cost | Added cost | Correctness/progress evidence | Layer performance | Role: B/C/rejected |
|---|---|---|---|---|---|---|---|
| baseline | to derive from actual staged execution | none | none | none | not run | not run | unqualified |

Example only: [A][B][C] -> [A+B][C] -> [A+B+C].
Bind each partition version to its parallel_plan_id. Do not copy this order where
the actual model branches or has recurrence, TP, EP dispatch/return or CP merge dependencies.

For the first island decide output tile owner, required producer tiles, unified CTA
geometry, handoff protocol and N/P/F comparison before broad fusion.
Keep epoch consistent across the entire logical invocation and exact stage partition.

## Current frontier

- Next edge and why it is the best available opportunity: not selected
- Smallest implementation and rejection experiment: not selected
- If slower functional C is retained: named downstream saving and exploration limit: none
