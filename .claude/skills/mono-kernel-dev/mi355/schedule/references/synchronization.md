# Mailboxes, residency, and buffer lifetimes

Sources: [S1/S2 K3, G3 GLM replay race, D1 DSV4](../../port/references/sources.md). This is a protocol review method. Actual release/acquire, cache scope, and instruction semantics come from current helpers and compiler.

## Questions for every dependency

| Item | Required fact |
|---|---|
| Producer domain | Task counts match actual writes; masked-task completion is defined |
| Payload writer | Unique owner or explicit aggregation; all participating waves/CTAs/ranks are known |
| Publication | Which writes complete before ready, and at which memory scope? |
| Consumer | Which epoch is read; can reordering or stale cache data invalidate consumption? |
| Capacity | Payload, flags, tables, and guard-region maximum indices are bounded |
| Reuse | All old readers finish before the next epoch/stage writes |
| Progress | Producers can execute; program order and waits cannot form a cycle |

For a multi-wave payload published by one lane, each writer must complete relevant stores, participate in CTA aggregation, then publish at the helper's required scope. A lane-0 `s_waitcnt` cannot complete other waves' writes. A CTA barrier does not replace cross-rank visibility.

Reason separately about tagged payloads and separate ready flags. Some helpers transfer value/tag together; others store value then tag. Check atomicity, compiler reordering, and consistency retries in the implementation.

## Liveness and compiled resources

Resident stages reuse a CTA set. A CTA waiting on a consumer before producing its own later tasks can create a program-order cycle despite an acyclic stage DAG.

Calculate `ceil(tasks/workers)` and tail owners after role count, offsets, or tile changes. Historical K3's 220-worker backend mismatched 224 down/448 tail tiles. GLM grouped UG described tile16 while the body published only 8 elements, leaving half the rows unwritten and down waiting forever.

Retain VGPR counts, spill metadata, private bytes, LDS, and code size separately. HIP occupancy is required runtime evidence; SGPR spill metadata is not scratch traffic, and a resource number alone does not explain a stall. Check actual CUs/partitions rather than copying residency assumptions.

<a id="replay-race"></a>

## GLM short-selector long-replay race

On 2026-10-10, a TP8/S8 fused-indexer case passed three functional replays but hung in long uninstrumented Graph replay. Q and score epochs were correct; one rank's `indices_ready` contained token IDs.

The final radix pass stored threshold/count in `keys[256:258]`. Compaction immediately reused `keys[256:264]` for wave counts before every wave had read the old values. A late wave read overwritten thresholds, emitted excess indices, and overwrote the mailbox out of bounds.

The fix loaded threshold/count into registers in every wave, then used a workgroup barrier before LDS reuse. The long selector already had this boundary. An extra mailbox memory-clobber candidate did not fix the race and was removed.

Reusable investigation sequence:

1. Preserve the hanging binary, layout, epochs, rank completion points, and exit status.
2. If tracing/retry counters mask the failure, use an independent stream/SDMA snapshot where supported instead of adding instructions to the kernel.
3. Inspect adjacent payload bounds, LDS reuse, and task granularity before assuming an ordering defect. A waiter does not imply a missing flag publication.
4. Replay the original **uninstrumented long failure**, delayed-rank cases, and original numerical/index checks after the fix. Short smoke is insufficient.

## Graph epochs and MTP

A Graph may update step once for many layer calls, but every invocation needs a non-conflicting layer/ordinal tag, including repeated calls to the same layer. Two buffer slots are safe only under the read/write progress contract; define tag wrap too.

Historical K3 true MTP used `state_indices[S+1]`: token s reads slot[s] and writes slot[s+1]. With B requests, each predecessor chain is independent; flattening must not connect requests. Invalid tokens may still owe state publication awaited downstream.

Concurrent streams cannot freely share epochs/scratch. Without independent arenas or a synchronization protocol, declare a single-stream runtime contract. Wait for all Graph consumers before closing the runtime.
