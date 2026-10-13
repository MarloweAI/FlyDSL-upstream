# From optimization opportunity to bounded search

Use once the layer is correct and the next structural change or parameter range must be chosen. Produce an opportunity estimate, fixed contracts, screening rationale, coverage, and stopping conditions. Use [diagnostics](diagnostics.md) for counters; profiling is not required every iteration.

## 1. Separate bounds, attainable performance, and removable time

| Model | Construction | Meaning and limitation |
|---|---|---|
| Compute/bandwidth roof | Work F, traffic Q_l at each memory level, matching compute peak P_compute and bandwidth B_l; `AI_l=F/Q_l`, `P_roof=min(P_compute, min_l(B_l*AI_l))` | Throughput bound under stated assumptions, not predicted low-concurrency utilization |
| Empirically attainable performance | Fastest validated implementation at matched shape, precision, working set, environment, and timing scope | Comparable target, not a mathematical lower bound |
| Dependency/latency bound | Unoverlappable DAG paths and required communication/data-arrival conditions | Serial constraints outside a throughput roofline; requires valid timebase and defined events |

State whether F counts useful FLOPs or executed work. MFMA count is not useful FLOPs. Match compute peak to dtype, accumulation, and instruction mode. Separate HBM/L2/LDS traffic; logical tensor bytes are not measured HBM bytes. For copies, report effective bandwidth using declared read+write bytes. Match clocks, device/CU scope, and units.

A throughput model gives `t_lb=max(F/P_compute, max_l(Q_l/B_l))` only when its assumptions apply. Take the maximum of independent valid lower bounds. Add segments only when they are necessarily serial and do not double-count work. If `t_lb/t_measured>1`, audit work, peak definitions, units, traffic, and timing scope. Do not invent a target such as 90% of theoretical peak.

Describe each opportunity as **work removed/moved earlier → effect on layer completion → added cost → discriminating experiment**. Removing a tensor may retain cross-CTA payload traffic; earlier shared loads need not improve routed loads. Predict work from real routes/tasks and test critical-path propagation with completion markers. [PMC cases](pmc-cases.md) show both unchanged-work speedups and exact MFMA savings. Bounds prioritize work; paired uninstrumented full-layer timing accepts it.

## 2. Separate tunable axes from invariants

Start with a compilable, independently checkable body. Choose meaningful axes: tile, wave count, pipeline depth, buffers, vector width, layout/loader, legal task mapping. State the reuse, parallelism, access, or live-range effect of each; avoid an immediate Cartesian product.

Fix quantization groups/rounding, accumulation/reduction order, output ownership, publication/wait scope, epoch lifetime, and supported masks/tails. Changes to these contracts are separate structural/numerical candidates. Read resource constraints from current gfx950 compilation/device properties, not historical tiles or residency thresholds.

Track three distinct ranges:

- **Statically feasible:** alignment, divisibility/tail handling, capacity, coverage, and progress constraints hold; execution remains unvalidated.
- **Correctness validated:** specific shapes/configurations pass required endpoints, full layer, state, and replay.
- **Performance confirmed:** qualified buckets/configurations/environments with fresh paired confirmation.

Select boundary checks by risk: either side of tile divisibility, task capacity/final group, high/low route overlap, LDS/occupancy tiers, and B/S/TP dispatch switches. Corner checks do not validate all interior configurations. The [tile contract](../../schedule/references/tile-contract.md) makes mapping, masks, and byte bounds explicit.

## 3. Screen in stages to bound cost

| Step | Action | Decision |
|---|---|---|
| Hard constraints | Check capacity, layout, tails, and progress, then compile/resources | Prune contract violations; smaller resources do not guarantee speed |
| Conditional pruning | Exclude combinations under current working-set/reuse/parallelism assumptions | Save premises; fewer VGPRs or a shallower pipeline may lose overlap |
| Ranking/bucketing | Rank by opportunity and cost; select a few families by shape/routes | Predictions prioritize; outside the predicted region is not proven inferior |
| Cheap screening | Correctness first, then short uninstrumented B/C pairs with trial differences/dispersion | Stop clear regressions only with sufficient evidence; near-noise remains inconclusive |
| Independent confirmation | Fresh winner measurements and affected buckets/switch boundaries | Update per-bucket B; combined optimizations are a new candidate |

Set candidate-count, compilation-time, instantiation/code-size, and GPU-time budgets from task constraints, including branch stop conditions. When budget limits coverage, report the best known result and gaps, not global optimality. Do not remove necessary numerical/state checks to search more candidates.

Save pruned values, reasons, premises, and environment. Reopen regions when workload, compiler, loader/layout, or evidence changes those premises. If a winner falls outside the predicted region, revise the ranking model before expanding search.

## 4. Leave resumable results

Record `bucket/family → invariants → candidates/pruning reasons → source/binary identity → three coverage ranges → raw pairs → B/C decision → next action/stop condition` in current task records and results. Keep one current state and immutable historical evidence. The [RSI loop](../../port/references/iteration-loop.md) decides whether to tune, redesign, fix observation, or fuse the next edge.
