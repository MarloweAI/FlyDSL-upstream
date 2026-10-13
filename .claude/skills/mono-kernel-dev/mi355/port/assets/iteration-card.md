# {{MODEL}} iteration card

Optional detail template for a complex candidate. A complete entry in the existing
experiment log is sufficient; link to this card when used rather than duplicating
its conclusions. Do not overwrite settled evidence.

## Before implementation

- Experiment ID / current phase / target bucket: unassigned
- R / B / C source paths and hashes: unassigned
- Current and proposed partitions: unassigned
- Baseline/candidate parallel_plan_id, per-stage TP/EP/CP and equal-resource/global-workload boundary: unassigned
- Comparison isolates fusion, parallel strategy, or combined deployment gain; missing controls: unassigned
- Observation from existing evidence: unassigned
- One causal hypothesis and competing explanation: unassigned
- Smallest main change: unassigned
- If searching parameters: axes, fixed invariants, pruning assumptions, compile/binary-size/GPU budget and coverage boundaries: not applicable
- Predicted task/instruction/byte/completion-point signal: unassigned
- If using PMC: predicted count from actual routes/tasks/body; metric definition, agent/interval/instances, main and auxiliary calls: unassigned
- If using ATT: target stage/CTA coverage, binary/PC-to-inline-call mapping, hits/stall normalization and competing explanations: unassigned
- What would falsify it: unassigned
- Required numerical/liveness gates: unassigned
- N/P/F or baseline/candidate protocol, timing/output boundary: unassigned
- Branch cost limit and stop condition from task constraints: unassigned

## After execution

- Exact source/binary/command/environment/fixture and raw artifacts: not run
- Compile/resource/occupancy result: not run
- Correctness/state/replay result and first divergence: not run
- Paired uninstrumented island and full-layer results/dispersion: not run
- If parallel work changes: per-rank token/expert/context work, communication/merge tail and redistribution costs: not run
- mechanism_status (supported / falsified / unknown), evidence and alternative explanations: unknown
- Each attempted change -> observed effect, unaffected observation, ruled-out condition and evidence identity: not run
- PMC/ATT when collected: raw totals, denominators, valid samples, normalized values, prediction comparison and missing/invalid evidence: not collected
- Work/dependency cost eliminated by this accepted change; cost still remaining: not established
- Status: not_run
- Update B per bucket? Update functional C? Why: no
- Static-feasible / correctness-validated / performance-confirmed ranges and remaining gaps: not established
- Ruled-out condition and what new evidence permits retry: none
- Next executable action and exit criterion: awaiting this experiment

Allowed statuses: not_run, compile_failed, correctness_failed, liveness_failed,
inconclusive, functional_only, rejected_perf, accepted.
An accepted local speedup does not establish the full model goal.
