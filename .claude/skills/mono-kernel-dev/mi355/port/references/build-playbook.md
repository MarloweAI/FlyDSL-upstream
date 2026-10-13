# Incremental fusion build playbook

Use this when you have a model but no mono-kernel. Build code that can be replaced and validated one stage at a time. Function names below are proposed project interfaces, not existing FlyDSL APIs.

## 1. Make one real layer the experiment unit

Follow [config discovery](model-config-discovery.md) through config → arguments/overrides → loader/sharding → forward/dispatch. Select a representative real layer and the smallest legal shape, plus a target shape that exposes scaling constraints. For a C2/S6 target, C1/S1 is a useful starting point, but P0 metadata must already represent two independent page tables and positions; do not hard-code `pos0+s` into the design.

| File/object | Responsibility | First runnable result |
|---|---|---|
| `spec.py` / ModelSpec | Resolve global/local dimensions, layer kind, quantization, and state ABI from config | Report effective config; reject the wrong model or unsupported format |
| `fixture.py` / LayerFixture | Borrow loaded weights; hold inputs/metadata and snapshot/restore; define allowed side effects | Run the reference twice from the same initial state and check drift |
| `reference.py` | Native layer or independent quantization-aware specification; capture stage endpoints | Check output, routing, cache, and state independently |
| `staged.py` | Compose existing operators and replace stages incrementally | Run and compare a complete staged layer with R |
| `packing.py` / `layout.py` | Shared host/device storage mapping, capacity assertions, scratch offsets | Check pack/unpack or address mapping independently |
| `device/*.py` | Current stage's device math, loaders, and epilogues | Compile, run, and measure a standalone stage |
| `kernel.py` / `op.py` | Partition compilation, resident scheduling, one-time prepare, execution entry | Switch between staged, partial fusion, and full mono |
| `tools/check.py` / `bench.py` | Same-state comparison, Graph timing, binary/environment records | Reproduce one well-defined comparison with one command |

Fit these responsibilities into the existing project; do not create empty modules just to fill a template. Make the fixture/reference/staged subgraph run first.

Expand the actual GPU DAG from host calls. Python function boundaries are not stage boundaries. Number nodes and record formulas, rounding, layouts, and launches. Distinguish views/aliases from copies and model cache/state writes as explicit outputs.

## 2. Preserve sensitive boundaries in the first runnable chain

A structural MoE example; use the current model's math:

```text
x ── norm ──┬─ router_logits → selection → ids,prob ─────────┐
            └─ quant → xq,scale ── UG → activation/quant ── down/combine → y
                     shared branch ────────────────────────┘
```

Connect ordinary launches with global tensors first. Resolve G1/G2 local dimensions and expert ownership under the chosen TP/EP plan, route-weight placement, shared rounding, and residual addition. Put EP dispatch/return explicitly around expert computation; expand cross-rank partials and CP merges using the [parallel plan](parallel-plan.md). Validate selection on fixed logits and retain G1/activation/quant endpoints.

**For every replacement:** capture its input from R → compare the standalone body against an independent oracle → insert it into the staged chain → check the full subgraph/layer and side effects → preserve that version. On failure, return to that node; do not alter downstream code or tolerances to conceal it.

Mature GPU operators can remain external. Only stages entering the same launch require composable device bodies. Host `aiter.xxx`/PyTorch calls cannot execute inside a GPU kernel; port the compute/layout or use a device-inline emitter.

## 3. Let the dependency determine the fusion method

| Edge | Initial implementation | Requirements for further fusion |
|---|---|---|
| Element/tile-local epilogue | Apply consumer math to producer results | Preserve rounding, vector width, and masks; global intermediates may disappear |
| Intra-CTA reduction | LDS/reduction with workgroup synchronization | Compatible wave cooperation; old readers finish before reuse |
| Cross-CTA tile dependency | Global payload with explicit publication/consumption | Resident progress, memory scope, tile readiness; global payload may remain |
| Full-vector statistics/routing | Separate statistics/selection stage | Owners, partials, completion counts, downstream task dependencies; not a local epilogue |
| Cross-rank TP | Keep the external collective initially | Peer addresses, slots, epochs, visibility, reduction order, rank progress |
| Cross-rank EP | Keep external dispatch/return; validate expert-local math and token combine | Original token/slot mapping, live counts, empty ranks, transport progress, hotspot tail, buffer reuse |
| Context CP | Separate sharded scan/score/attention and valid merge | Complete scores/exact selection or attention partial semantics, KV/page ownership, exchange capacity, cross-rank publication |
| Cross-token recurrence | Explicit predecessor state | Per-token snapshots and ordered commit; move independent preparation, retain state dependencies |

Choose an edge with measurable cost and manageable implementation. MoE is not a mandatory starting point: attention/indexer may be the better first island when it dominates and suitable bodies exist. Compare removable launches, materialization, and repeated work with added residency/synchronization cost.

Before coding that edge, complete the [tile and storage contract](../../schedule/references/tile-contract.md). Derive task coverage, masks, physical addresses, quantization groups, and overlapping buffer lifetimes together.

## 4. Share stage math, keep the reference independent

Suggested interfaces:

```python
# Interface sketch, not executable FlyDSL code.
prepared = prepare(spec, loaded_weights, cache_views)
run_reference(fixture.restore())
run_partitioned(prepared, fixture.restore(), partitions=[['A'], ['B'], ['C']])
run_partitioned(prepared, fixture.restore(), partitions=[['A', 'B'], ['C']])
run_partitioned(prepared, fixture.restore(), partitions=[['A', 'B', 'C']])

# Each partition compiles to a kernel. Emitters expand at compile time.
emit_A(task, tensor_views, scratch_view, sync_context)
emit_B(task, tensor_views, scratch_view, sync_context)
```

Partitions execute each logical operation exactly once and respect inter-partition topology. A branching DAG cannot always be cut into an arbitrary linear list. List cross-partition inputs/outputs and enforce legal stream/event dependencies. Compile/cache the exact partition list for a capture; host loop counts are not launch counts.

The [DSV4 builder/epoch case](../../schedule/references/staged-epoch-case.md) uses one builder for `router/route/quant/up/down` and the full version. Compile-time selection controls device math; the host selects the launch list. Only the full kernel or final down stage commits CTA epochs, so earlier stages share one generation.

That case includes pseudocode, CTA/epoch mapping assumptions, and partial-partition validation. Advancing the epoch after every standalone stage can make producers and consumers wait on different generations. A live buffer cannot serve two unfinished invocations without an isolation protocol. New grids or concurrent streams require a new generation mapping.

## 5. Diagnose resident compatibility before joint redesign

Read [persistent execution design](../../schedule/references/persistent-design.md) for worker grids, task loops, ready queues, completion, and exit. Hold execution lifetime fixed for this ablation; a cross-invocation service is a separate candidate.

- **N:** efficient ordinary stages, separate launches in the same Graph.
- **P:** proposed block/LDS/task-loop/storage configuration, still separate launches. Do not wait on flags whose producers are absent; use real preceding output or valid protocol initialization.
- **F:** the combined launch and proposed handoff.

Match math, effective work, initial state, and timing scope. If N uses a different grouping algorithm, the whole difference is not a resident penalty.

If P regresses, inspect matrix tiles, active waves, task tails, unrolling, weight layout, common LDS footprint, and actual occupancy. Preserving semantics does not require keeping the original tile; a general GEMM may need a new resident body.

If P is reasonable but F regresses, inspect joint compilation/resources, handoff, grid waits, consumer granularity, idle roles, atomics, and external tails. Use one discriminating control at a time.

## 6. Extend the island to the required layer boundary

For each partition, identify the removed launch, saved/added data movement, retained dependencies, resource changes, and deciding check/experiment. Keep a reversible ladder, for example:

```text
L0: [norm] [router] [select] [quant] [UG+act] [down/combine] [TP/residual]
L1: [norm+quant] [router+select] [UG+act] [down/combine] [TP/residual]
L2: [norm+quant] [router+select] [UG+act+down/combine] [TP/residual]
L3: [norm+quant] [router+select] [UG+act+down/combine+TP/residual]
L4: [whole MoE island], attention/indexer still external
L5: join [attention/front end] and [MoE] into the required full-layer partition
```

This is a TP MoE illustration without a latent branch. Redraw EP/CP ladders around their [communication chains](parallel-plan.md). K3 adds latent/down-RMS/AttnRes dependencies; DSV4 may retain external mHC/compressor work, so L4 is not full-layer completion. Respect the real DAG between L1 branches; do not duplicate norm to match the diagram.

A first L2 can use a correct serial resident handoff, then tile readiness, then L3. An output-owned down+TP island may be a better start. Compare the full old/new partition at each step and retain switches. Roll back the failing edge when correctness, liveness, or performance evidence is lost.

TP fusion must avoid a cycle where all resident CTAs claim some outputs and wait for peers while other required outputs remain unclaimed. Historical K3 used “output tasks do not exceed resident CTAs” as one premise of its particular queue proof. A new shape outside that premise needs a new proof/schedule, not a copied constant.

## 7. Distinguish two valid milestones

**Promote to B:** required correctness, replay, paired full-layer performance, and affected-bucket checks pass.

**Advance only C:** the implementation correctly covers more of the target fusion scope but is slower than B. Continue only with a named downstream saving, a falsifying experiment, and a branch time/candidate limit. Reject the direction if that experiment fails; sunk implementation effort does not justify more tuning.

Budgets and criteria come from the current task. Without GPU access, continue real body work, static constraints, and runnable harnesses where useful, but mark unexecuted gates as `not_run`.
