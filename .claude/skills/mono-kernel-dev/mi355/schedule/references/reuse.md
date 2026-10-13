# Reuse and task mapping at low concurrency

Sources: [S1/S2, G1, G4/G5, M1/M2](../../port/references/sources.md). The models below describe logical work. Counters and paired latency are needed to establish performance.

## Dense projections and token groups

Using W for g query columns amortizes a weight load across rows. Splitting T rows into `ceil(T/g)` groups can multiply logical weight reads for some projections by that factor. Cache, schedule, and working set determine how much reaches HBM.

In historical K3 S4, group4/2/1 produced 68.25/136.5/273 MiB of logical output/router/latent/shared-up weight reads per layer/rank. A runtime group1 loop also increased VGPR usage and reduced residency. Compile-time unrolling reduced resources but retained repeated-work/code-size costs. Preserve multi-token dense/tail reuse first, then move independent tasks locally.

Count maximum tasks per CTA, not only total tasks. Reducing backend workers can add an entire tail round. Capacity released by early-exiting frontend CTAs does not automatically repair static backend imbalance.

### DSV4: within-stage token distribution differs from token-major pipelining

One historical DSV4 change distributed `token × stage_task` within each stage while preserving dependencies and math. For n tasks/token, T tokens, and W role-local workers, stride over `[0,T*n)` from the worker's local ID, reconstructing `token = task // n` and `local_task = task % n`. Derive n, offsets, and workers separately per stage; irregular domains need prefix ranges/task tables.

This fills task tails across tokens. It differs from `for token: run the whole norm→MoE→TP chain`, which can lose dense/shared column reuse, as K3 did. It does not create expert grouping or permit cross-request KV sharing.

First change only task mapping. Retain token-tagged scratch, route/K accumulation, and state order. Check every live task exactly once, masked publication, capacity, and maximum rounds per CTA, then full-layer tail. If tail balance improves but reads/resources increase, use a control to choose rollback or smaller grouping. Add a [liveness review](synchronization.md) when introducing cross-stage waits.

## Expert union

For T rows, top-k routes per row, and U distinct routed experts, token-major UG has roughly `T*k` logical expert instances; expert reuse has roughly U. Account for shared separately. Actual instructions also depend on K/N tiles, valid columns, and wave layout.

A useful intermediate representation:

```text
unique_experts[u]
token_mask[u]                 # Cover T bits; use multiple words when needed.
slot_of[u, token]             # Original routing slot; explicit inactive value.
mid_row[u, token] / zero_row  # Down lookup.
route_weight[token, slot]     # Never index weights by union position by mistake.
```

Build union/masks on device → assign `(u, Ntile)` UG tasks → map active tokens to MFMA columns → write the original `(token, slot, intermediate)` domain. Initially preserve down and FP32 mid so intermediates remain directly comparable. Compute shared as one logical expert instance covering all valid tokens.

Check worst-case union capacity `min(E, T*k)`, shared extras, rank consistency, inactive columns, multiple token masks within a wave, odd rows, and T boundaries. Debug buffers also need worst-case capacity; observed small U is not a safe allocation bound.

GLM's first grouped UG confused tile16 tasks with a tile8 body. Derive domains from actual output coverage. The [PMC case](../../benchmark/references/pmc-cases.md#pmc-grouped-ug) predicts and observes 187,392 fewer MFMA instructions from real routes; capacity, live tasks, and masked execution are distinct counts.

## Preserve each token's down reduction order

UG can often change token-to-column mapping while preserving each row's K order. Down combines expert/K contributions into one output and is more sensitive to reordering.

Start by preserving original slot/K/wave/TP accumulation. Group identical experts within a fixed slot, or combine currently executable queue heads without crossing a token's predecessor. These are conditional designs, not universal winners. Prove every original slot executes once and loops make progress.

M3 wide-down kept unrelated tokens' routes from changing another token's summation grouping. Re-derive this for GLM/K3 top-k, grouping, precision, and inter-wave reduction.

GLM later implemented same-slot grouping within original waves: predicted and measured MFMA savings were 19,968. Cross-slot queues had negative performance results. See [ordered down](../../benchmark/references/pmc-cases.md#pmc-down), including the LDS lifetime failure. Do not describe a subsequently implemented mechanism as still missing.

## Quantize intermediate activations once

A q-element quantization group's scale needs the full-group amax. A UG8 tile cannot independently produce a block32 scale. Two designs are:

- Assign four adjacent tiles to one producer, gathering 32 elements at the cost of task parallelism and longer live ranges.
- Keep UG granularity; one owner wave gathers published FP32 mid, applies the original quantizer, and publishes packed values/scales. This adds a device stage, not necessarily a kernel.

Historical GLM Pack3 used the latter, preserving FP32-mid checks and down K/slot/TP order. Tagged packed payloads avoided a global completion flag. At T12, logical payload per consumer fell 71.875%; DRAM requests did not fall proportionally. The gain involved repeated quantization, LDS work, and reads.

[PMC](../../benchmark/references/pmc-cases.md#pmc-quant-once) showed unchanged MFMA, 5.36% fewer LDS instructions, and 0.58% fewer DRAM requests. This supports investigating remaining down work; it does not independently establish fewer LDS bank conflicts.

## Fix layouts before blindly deepening prefetch

Historical GLM #1241 packed E8M0 scales by 16 rows ×128 K, with 8 gate + 8 up rows in one tile. Host packing and device reads changed together. A later native-MFMA path was different; importing a new file did not establish that it consumed the layout.

See the [layout/prefetch case](layout-prefetch-case.md) for implementation, scope, negative results, and retry conditions. Re-derive current layouts from lane consumption and quantization groups rather than copying tile constants.

Use actual DS lane addresses to check LDS bank mapping. A 16B padding candidate worked in some M3/GLM paths, not every tensor. Recompute the total overlapping live footprint when changing stride/padding; see the [tile contract](tile-contract.md#lifetime).

Check paired weight/scale loads, register lifetime, generated placement, and polling wait domains. Shared expert IDs are known before routing; routed IDs are not. K3's early first-down load was caught by an input poll's `vmcnt(0)`, eliminating the expected overlap.
