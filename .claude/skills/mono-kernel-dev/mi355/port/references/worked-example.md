# A complete walkthrough from bring-up to evidence-driven iteration

Part A uses DSV4-like MoE structure to illustrate a development sequence; current config determines the actual model. It does not claim historical work followed this exact order. Part B uses measured K3 results. No new GPU measurements are introduced.

## A. Start a new model with a DSV4-like MoE

Suppose the task requests resident MoE on MI355 for B1/S1 and S4, followed by ATOM full-layer integration. These are teaching targets, not default support. Use [config discovery](model-config-discovery.md) for hidden H, experts E, top-k, global intermediate I, shared count, TP sharding, branch-specific A/W/scales/groups, and layer kinds. Apply hash/bias or HCState/mHC details only if current code has them. A MoE-only task does not require fusing external attention/mHC.

### First: make a real layer repeatable

If bias-routing layers exist, choose one and plan coverage of hash/other branches. Borrow prepared rank-local weights/metadata; capture real inputs, state, positions, and nonzero cache or another legal initial state. Run the native layer twice with restoration, recording router logits/IDs/probabilities, MoE inputs/outputs, and cache/state effects.

The deliverable is a fixture, reference, and check command. **Reading config is not the exit criterion; replaying a complete input→output/effect contract is.** If native drifts, fix logits to isolate routing from upstream arithmetic. Fusion code cannot resolve an undefined reference.

### Second: build replaceable staged MoE

Split `router → selection`, routed/shared input quantization, `UG+activation`, `down+combine`, and TP. Start with mature operators, replace one device body at a time, and preserve quantization/rounding.

Check G1 on fixed inputs/routes, then G2, then real routing. Put the new MoE back into the native layer shell and check all state/cache, including HCState where present. Now R and a qualified staged B exist.

If G1/G2 are not composable device bodies, implement those next. If they already exist, proceed to resident compatibility; do not rewrite unrelated attention to standardize the project.

### Third: build the UG/down island

Choose the down output owner and whether one task gathers all local routes, then derive required mid tiles. Align UG production with quantization group q: a scale owner must see the whole group. A tile smaller than q cannot substitute a local amax. Historical q32/UG8 alternatives are in the [reuse guide](../../schedule/references/reuse.md); verify the [tile mapping/lifetimes](../../schedule/references/tile-contract.md) before coding.

Build N/P/F: ordinary stages, resident-configured separate stages, and one resident partition. Keep router/quant/TP external. Check mid, output, and changing-input epochs. Do not simultaneously change quantization, CTA assignment, tie rules, and TP algorithm.

- P much slower: fix body/tasks/resources before wrapping router/TP into the kernel.
- P reasonable, F blocked on whole-mid completion: keep functional C and test tile-ready/tagged payload.
- F faster: pair the complete layer, decide B promotion, then consider down→TP.

### Fourth: extend to one MoE launch

Connect down to TP, compare local partials and reduced outputs, and prove task claims/peer waits cannot starve production. Then connect router/selection/quantization with explicit producer roles. Retain same-builder staged controls and advance epochs once per logical invocation.

Keep runnable staged, current partial, and full-MoE partitions with the same fixture and actual launch lists. Adding one Python `forward()` is not device fusion.

### Fifth: scale S1 to S4 without repeating S1 four times

Preserve dense/shared token-column reuse and examine whether within-stage `token×task` distribution reduces resident tails. If T4 cost comes from repeated weights/quantization, use actual routes to decide grouping. S1 performance does not predict S4, and four rows do not establish four independent requests.

Within-stage distribution, token-major pipelining, and expert grouping are different changes. Validate the first's numerics/coverage before trying the others. Re-identify the bottleneck after each accepted change.

### Sixth: complete the real ATOM call

Implement prepare/dispatch/Graph/close with guards before cache writes. Retain native external nodes and use actual mHC/attention/indexer ABIs without duplicate cache updates. Prove the MoE launch count and list remaining full-layer launches.

For a full-layer mono target, this remains a milestone: return to P3 for mHC/attention/compressor islands and edges. For a MoE-only target, complete full-layer performance/state regressions and acceptance. The task defines the boundary.

## B. K3: evidence that requires a new direction

The [resident regression case](resident-regression-case.md) preserves controls, resources, negative results, correctness limits, and implementation follow-up. Representative S4 timing was ordinary staged≈32.5 µs, resident separate stages≈50.7 µs, grid-barrier pair≈86.5 µs. Each configuration had its own paired baseline; avoid precise cross-table subtraction.

**Implemented:** general 256-thread A16W4 GEMM1/GEMM2 in a persistent loop, retaining SiTU, BF16 mid, global BF16 atomic scatter, and external router/tail. Scoped numerical/Graph checks passed, performance regressed, and independent full-layer golden gaps remained.

**Correct state:** P2/P3 functional C exists, B stays staged, `status=rejected_perf`. Neither “K3 cannot use mono” nor “one launch is enough to keep fusing” follows.

**Why stop grid tuning:**256/512 CTAs, uniform indices, unrolling, and waves-per-EU had negative evidence alongside changed resources/common LDS. Ownership and handoff offered new information; repeating the grid did not.

**Next concrete experiment:** standalone output-owned down gathers routes, removes atomic scatter, and prepares a tile-TP boundary. Preserve BF16 input/mid, SiTU, and numerical criteria. Validate the new reduction organization independently and at layer scope before connecting a tagged UG producer. Do not simultaneously switch inputs to A8.

**Prediction:** less atomic/materialization work, direct consumption of ready fragments, possible down/TP overlap. These are predictions, not measured gains; full-layer uninstrumented pairs still decide.

**Follow-up:** slow standalone → tile/K/wave organization; standalone reasonable/fused slow → handoff/progress; local win/layer loss → latent RMSNorm, shared/latent tail, and frontend dependencies.

## C. GLM: iterate after functionality works

G4/G5 mechanisms are in [grouped UG/quant-once](../../schedule/references/reuse.md) and their interpretation in [diagnostics](../../benchmark/references/diagnostics.md). Historical C2/S6 routes exposed repeated UG across tokens.

1. **Group UG first.** Preserve FP32 mid, original slots/down/quantization. The first task-domain error left half-rows unpublished; repair coverage before treating timeout as a performance issue.
2. **Change the question after acceptance.** Once UG tasks/instructions fall, inspect repeated down quantization rather than continuing to optimize already-removed UG work.
3. **Quantize once.** Compare a producer that gathers 32 elements with UG8 plus a separate owner. The latter preserves UG parallelism. Keep down order to isolate the change.
4. **Update attribution.** Large logical-payload savings but small DRAM-request change reject proportional HBM savings as the explanation; examine quantization/LDS/consumer preparation.
5. **Continue to the actual target.** Beating old mono does not establish a win over real unfused. Keep both comparisons and investigate down/TP/layer tails.

The [PMC chain](../../benchmark/references/pmc-cases.md) makes this quantitative: 96 slots→35 experts predicts 187,392 fewer MFMA and matches observation. Quant-once keeps 636,992 MFMA, reduces LDS 5.36%, and DRAM requests only 0.58%. Ordered down then predicts and observes 19,968 fewer MFMA. Pairs have separate baselines/fixtures/timing; do not merge them into an exact time decomposition.

The [ATT walkthrough](../../benchmark/references/att-cases.md) links larger working sets and fixed-hit callsites to shared-early prefetch, opposing stall signals, scale-only failure, and fresh timing. High stalls alone do not select an optimization.

## D. Check whether the workflow produces an executable next step

| Starting situation | Required next output |
|---|---|
| Only model config and ATOM code | Representative fixture, snapshot/restore, R, DAG, runnable command |
| Correct staged path, no composable bodies | One stage emitter/standalone wrapper and endpoint comparison |
| N fast, P slow, F slower | Resident/body/owner ablation; retain B and stop unsupported fusion expansion |
| F correct/slower, next edge may remove atomics/TP | Bounded functional C branch with a falsifiable downstream mechanism |
| Short replay passes, long replay hangs | Exact-binary task/tag/bounds/LDS investigation; suspend performance promotion |
| Grouped UG improves but layer loses to unfused | Record local progress; find remaining critical path; retain real unfused B |
| Resume has latest source and an old “best” claim | Reconcile R/B/C identities, completed results, and rejected conditions |

This is a decision-path walkthrough, not an independent agent evaluation or a guarantee of new-model performance.
