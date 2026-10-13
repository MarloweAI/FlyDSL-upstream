# Symptoms → minimal experiments → direct cases → next actions

Choose the row matching current evidence and read that case section. Conditions, failures, and evidence limits are part of each conclusion. Historical tiles/thresholds are not defaults. Return all decisions to the [RSI loop](../../port/references/iteration-loop.md).

## Resolve correctness and bring-up failures first

| Symptom | First action | Direct case/rule | Decision |
|---|---|---|---|
| Same precision label, different golden | Locate the first A/W/scale/MFMA/rounding difference | [Numerical paths](../../numerics/references/contracts.md#numerical-paths) | Fix the contract, not the tolerance |
| Routes/probabilities differ | Fix logits/bias; separate permutation, exact ties, near scores | [Routing mismatch classes](../../numerics/references/contracts.md#routing) | Expert alignment is diagnostic, not an ABI waiver |
| Reference drifts | Repeat R from the same initial state; isolate nondeterminism | [Native instability](../../numerics/references/contracts.md#unstable-reference) | Name controlled-oracle scope and retain native failures |
| Short replay passes, long replay hangs | Preserve binary/state; inspect adjacent bounds and old LDS readers | [GLM threshold/scan race](../../schedule/references/synchronization.md#replay-race) | Replay the original uninstrumented failure after fixing it |
| New partition never receives payload | Align builder, generation, and committer | [DSV4 epochs](../../schedule/references/staged-epoch-case.md) | One consistent invocation generation before more fusion |
| Tile change leaves holes or stale padding | Enumerate actual body outputs and consumer addresses | [Tile mapping/masks](../../schedule/references/tile-contract.md#mapping) | Repair coverage, neutral values, and bounds before queue tuning |
| All workers wait while producers remain unclaimed | Add worker-order and resource-wait edges | [Two-worker deadlock](../../schedule/references/persistent-design.md#ready-scheduler) | Finish production or claim only ready tasks; residency does not break cycles |
| Empty queue triggers exit, or stop strands peers | Audit closure, in-flight work, completion/drain | [Completion](../../schedule/references/persistent-design.md#completion), [service loop](../../schedule/references/persistent-design.md#service-loop) | Empty is not complete; stop must preserve protocol progress |
| Hidden is correct, next step fails | Compare written/untouched state and request predecessors | [State](../../numerics/references/contracts.md#state) | Fix ownership/lifetime and repeat consecutive steps |

## Choose performance experiments from evidence

| Evidence | Smallest experiment | Direct case | Falsification or follow-up |
|---|---|---|---|
| Fast standalone, slow resident | Same-math/plan N/P/F | [K3 regression](../../port/references/resident-regression-case.md) | P regression: body/tasks/resources; F-only regression: handoff/joint resources |
| Serial producer delays dependents | Distribute only its independent tasks | [Cache completion chain](pmc-cases.md#pmc-cache) | Same work can finish sooner; unchanged dependents question the critical path |
| Routed slots greatly exceed distinct experts | Group UG, retain original slots/mid/down, predict MFMA savings | [187,392 fewer MFMA](pmc-cases.md#pmc-grouped-ug) | Missing signal: task/body/dispatch; matching signal: inspect repeated quantization next |
| Consumers repeatedly quantize FP32 mid | One full-group quantization owner | [Quant-once](pmc-cases.md#pmc-quant-once) | Fewer LDS instructions is not fewer bank conflicts; logical bytes are not HBM bytes |
| Quantization is already shared, down repeats work | Reuse within original token slot/K order | [Ordered down](pmc-cases.md#pmc-down) | Verify count prediction/numerics; more cross-slot reuse may regress |
| Larger working set exposes VMEM stalls | Match samples/callsites/hits, then inspect addresses/dependencies | [One/eight addresses](att-cases.md#att-workset) | Sampled-stall share is not latency share; HBM/L2 claims need more evidence |
| Known addresses load late | Move legal shared weight/scale loads; inspect ISA/waits | [Shared early](att-cases.md#att-prefetch), [poll-wait counterexample](../../schedule/references/layout-prefetch-case.md) | Prefetch can be caught by polling waits; judge full-layer pairs |
| Scattered weight/scale gathers | Change packer and loader together, same math | [A4W4 layout](att-cases.md#att-layout) | Normalize changed hits; increased VGPR does not automatically reject |
| Fewer resources, slower kernel, old profiler only | Match current binary and observe the target stage | [v13 gap](pmc-cases.md#pmc-missing), [ATT coverage failure](att-cases.md#att-coverage) | Keep cause unknown; do not reuse old hotspots as current evidence |
| Larger B/S repeats dense/shared loads | Token-column reuse against current schedule | [Reuse and negative results](../../schedule/references/reuse.md) | Check VGPR/rounds/tail and actual LDS lane mapping |
| Down atomic/combine/TP sets the tail | Isolate output-owned down or tile-ready publication; prove progress | [K3 next design](../../port/references/resident-regression-case.md) | Conditional candidate; unresolved golden/performance stays unresolved |
| Long-context score/selection dominates | Bounded exact selector, then same-algorithm CP off/on | [Non-monotonic indexer CP](../../schedule/references/indexer.md) | Include exchange/merge; preserve each query's causal bound |
| Prototype wins, ATOM loses | Trace packing, materialization, Graph/step, cache, and dispatch | [Integration gaps](../../atom/references/integration.md) | Fix real interface/state costs, then measure serving scope |

## Parallel strategy is also a candidate

For over-sharded TP matrices/reduction tails, EP hotspots/small messages, or unpartitioned long-context scans, build staged controls using the [parallel plan](../../port/references/parallel-plan.md). EP grouping uses post-dispatch work per owner. Separate indexer CP, attention CP, and KV storage. EP guidance here is design reasoning, not a historical measurement.

Fix the plan to measure fusion, fix implementation organization to measure strategy, and compare deployment with the best known equal-resource unfused path. Missing controls allow only combined attribution. Rank-local counter reductions may be work relocation.

Turn a case into the current candidate's observation, prediction, falsifying outcome, and next action. After grouped UG, inspect quantization; after quant-once, inspect down/handoff. Do not repeat a failed grid without changed premises. Before profiling, apply the [diagnostic scope checks](diagnostics.md).
