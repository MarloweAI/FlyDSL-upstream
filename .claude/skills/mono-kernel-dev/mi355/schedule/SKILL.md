---
name: mi355-mono-kernel-schedule
description: Design MI355 persistent worker loops and incrementally fused mono-kernels with tile ownership, task scheduling, completion protocols, and TP/EP/CP progress. Use for persistent execution, staged fusion, resident regressions, replay hangs, grouped MoE, or fused indexer CP.
---

# Build a resident fusion island from staged operators

Start with an executable staged subgraph, input/output snapshots, and a reference. Otherwise complete [P0/P1](../port/SKILL.md) first. Deliver runnable standalone/resident/fused code and a specific next fusion edge.

For a new persistent executor, read [worker lifetime and execution design](references/persistent-design.md). Declare finite invocation or cross-invocation service mode, then design the worker grid, task loop, readiness, completion/exit, and communication progress. Validate Graph replay, resource residency, and a cross-invocation service loop as separate properties.

## 1. Derive tasks from output ownership

Confirm rank, expert, and KV ownership in the [parallel plan](../port/references/parallel-plan.md). Choose the final output tile: which rank/CTA/wave owns it, which route/K/tensor/context partials it collects, and when its final write completes. Work backward to the consumer's intermediate tiles and their producers.

For example, `(expert, output tile) → atomic scatter` and `(token group, output tile) → intra-CTA route combine → TP` imply different schedules. The latter can extend fusion into combine/TP, but a changed accumulation or rounding order needs independent validation. Removing atomics does not establish equivalence.

EP usually computes on the expert owner and returns contributions to the token owner. Compute grouping opportunities from post-dispatch work. Establish external dispatch/return/merge first, then fuse one edge. Communication progress, empty-rank completion, and hotspot tails are part of the resident progress argument.

Write a task table: `stage / task domain / actual output elements / CTA-wave mapping / input dependencies / payload and tag / last reader`. Use the body's actual tile. GLM once described 16 elements per task while the body wrote 8, causing a permanent wait. Apply the [tile and storage contract](references/tile-contract.md) before changing geometry, layouts, or buffer reuse.

## 2. Share a device body between two wrappers

Extract the compute body, prologue/epilogue, and loaders. Keep a standalone `build_stage(stage)` launch and add `build_partition(stages)` or equivalent compile-time selection. These are suggested project interfaces, not existing FlyDSL APIs.

The host prepares arguments and chooses partitions; a device emitter is compiled into its caller. Parameterize task index, input/output views, workspace layout, and synchronization/publication context. Share mathematical and quantization bodies where practical; keep the oracle independent of the formulas under test. See the [DSV4 staged/fused example](references/staged-epoch-case.md).

Before unifying CTA geometry, audit every `gpu.barrier()`, lane/wave assumption, matrix fragment, and LDS region. Two 256-thread bodies cannot independently run in opposite halves of a 512-thread CTA if their barriers synchronize the entire workgroup.

**Compile and run the rewritten body on its own first.** Fix standalone failures before adding mailboxes or TP.

## 3. Replace a launch boundary with an in-kernel handoff

Map every payload, ready flag, and queue: offset, capacity, writer, epoch, readers, and reuse condition. Prefer verified project helpers and check their actual memory scope.

Use a first schedule that is easy to audit: producers finish the independent work they own before waiting on consumers. Use static roles or let completed producers join the consumer queue. A producer must not wait for a consumer that depends on its own unproduced tasks.

A grid barrier is a diagnostic baseline only where co-residency and progress are established. Otherwise retain the launch boundary or reorder tasks. Check HIP occupancy, XCD/partition placement, and roles; one capacity inequality is not a liveness proof.

Publication requires **all payload writers complete → visibility at the required scope → ready for the current epoch**. Consumption requires **matching epoch → valid read → last reader completes before reuse**. A thread-0 wait does not complete stores issued by other waves. See [synchronization](references/synchronization.md).

Check poisoned scratch, changed inputs/epochs, original numerical criteria, unwritten regions, and repeated uninstrumented replay. On a hang, freeze the candidate and inspect the first unfinished dependency and adjacent memory; do not blindly add retries or remove barriers.

Direct cases: [long-replay LDS race](references/synchronization.md#replay-race), [partition generation mismatch](references/staged-epoch-case.md), [K3 N/P/F regression](../port/references/resident-regression-case.md).

## 4. Turn a correct handoff into useful overlap

Record the producer's first/last completion and the consumer's first useful computation/last completion. If a consumer waits for a whole tensor but needs one tile, test tile readiness. Tagged payloads may return data during polling and avoid a separate ready round trip; this requires a producer epilogue and consumer loader change, not just an extra flag.

Move independent work earlier when evidence supports it: shared weights, current-token norm/decay/beta, or the next tile's prefetch. Check actual ISA placement and wait domains; a polling `vmcnt(0)` may wait for prefetched loads too. Keep predecessor-state loads/updates behind their required recurrence fence.

To demonstrate UG/down compute overlap, observe `first_down_math < last_UG_complete`; for down/TP, observe `first_TP_work < last_down_complete`. Label markers as issue, visible completion, or useful math. They test the mechanism; uninstrumented full-layer timing establishes the gain.

## 5. Target reuse before broad parameter search

Choose one change based on the bottleneck:

- Repeated dense/shared weights: reuse weights across token columns; first count the extra rounds and bytes introduced by splitting tokens.
- Overlapping routes: expert union/masks; change UG first and preserve original slots, then tackle down.
- Repeated consumer quantization: assign one owner to a complete quantization group and publish packed values/scales.
- Scattered loads: change weight/scale/gate-up packing and readers together; derive LDS banks from lane addresses.
- TP tail: publish completed tiles early, then compare owner reduction or all-gather designs and check unclaimed-task starvation.
- EP dispatch/return or hotspot tail, or long-context scans: use the [parallel plan](../port/references/parallel-plan.md). Measure per-rank work and merge cost before choosing communication, expert placement, or CP.

Use the [reuse guide](references/reuse.md) for mappings and the [indexer guide](references/indexer.md) for block/token top-k, candidate ownership, and CP entry-point constraints.

## 6. Return a decision to the main loop

Return `implemented partition + build/resource identity + numerical/replay results + N/P/F and full-layer pairs + changed bottleneck + next edge or ablation`.

If N→P already regresses, investigate the resident body/tasks/resources. If P is close to N but F regresses, investigate joint resources, handoff, and program order. If the island improves without a layer gain, inspect the critical path and interface tail. Update construction C and validated B using the [iteration rules](../port/references/iteration-loop.md); fewer launches alone do not justify promotion.

For [delivery](../port/references/delivery.md), provide real entry points, retained staged/fallback paths, progress/replay evidence, and the final fusion boundary. A diagram or wrapper name does not replace a launch trace.
