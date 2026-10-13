# Integration and state contracts

Sources: [D1/D2, M1, K1, G1/G2, R1](../../port/references/sources.md). Populate this matrix after discovering current config, loader, metadata, and dispatch through [config discovery](../../port/references/model-config-discovery.md). Implement only mechanisms present in the target model.

| Interface | Contract | Validation focus |
|---|---|---|
| Checkpoint/weights | Parameter identity, pack version, scale layout, TP/EP shard/expert map, dtype | Avoid duplicate full expert copies; host pack matches device branch |
| Positions/slots | Per-token request, position, context bound, write slot | B1/S8 is not B2/S4; permuted pages, padding, negative slots |
| KV/index cache | Page size/stride, packed layout, live/capacity, owner | Current token written once; untouched bytes unchanged |
| KDA state | Decode/MTP, predecessor/successor, preserved input snapshot | Consecutive steps, partial accept/reject, restart/aliases |
| mHC/compressor | Delayed residual/combination state, ring, commit point | Uncommitted rows, ring wrap, next layer/step |
| Peer/epoch | TP/EP/CP groups/members, arena, slots/parity, step/layer ordinal | Delayed/empty ranks, transport progress, long replay, close before teardown |
| Graph | Compile/pointer keys, scratch lifetime, stream contract | Capture→change inputs→replay; rebuild/reject after cache rebind |
| Dispatch | Supported buckets and semantic switches | Side-effect-free fallback; host chunking is not one launch |

When mono and unfused plans differ, check activation redistribution, expert/KV ownership, and state migration through the [deployment contract](../../port/references/parallel-plan.md). Graph/peer buffers bind to concrete plans; equal degrees do not imply equal groups/maps. Fallback must consume current layouts, not switch parallelism against stale weights/cache.

## GLM layout integration

The old adapter's elementwise gathers differed in cost from contiguous scale/gate-up packing. Verify final packed addresses, reader branch, and MFMA mode. Pointing Python at a new kernel file does not establish fast-path selection.

Multi-request support in `with_indexer=False` did not establish fused-indexer support. Early fused `pos0+s`, a single cache, and chunk rejection differed from later per-request paged metadata. Prefer consuming production caches directly. If a temporary conversion is needed, disclose whether it is outside timing and how serving pays for it.

Graph organization matters: “16 calls to one layer, one shared step update” differs from a step per call. Avoid tag collisions and unnecessary separate replay/step operations. Reusing one layer's weights 16 times is not a real 16-layer model.

## K3 full-output boundary

KDA state dtype, attention/projection quantization, and latent/routed paths may differ from native ATOM. Record formats and validate each path; latency across differing math is not isolated fusion gain.

When native returns prefix/routed/shared pieces, include residual/output materialization so both arms return consumable equivalent-boundary tensors. List the application kernel separately from Graph-external resets/epoch work.

Historical MTP used S+1 state indices to preserve input. New B×S support needs independent request starts, per-token snapshots, and row bounds. A full mono may support more shapes than it wins; choose fallback from bucket data.

## DSV4 layered fusion

Historical `Dsv4MonoKernel.forward(HCState, positions)` was a full-layer API with native mHC/indexer/attention plus one resident MoE kernel. Mono MoE and five-stage MoE were compared in the same shell; neither automatically represented full native ATOM MoE. Inspect current wrappers and traces rather than inferring fusion from `forward`. See the [shared builder/epoch case](../../schedule/references/staged-epoch-case.md).

CSA attention already called the indexer; an outer indexer call would duplicate cache writes. HCState delayed post-state passed to the next layer, so eagerly folding it into an ordinary residual changed semantics. Follow actual attention calls to identify the indexer owner and discover HCA/CSA/hash/bias layer kinds from current config/constructors.

Derive support from public B/S/TP guards, specializations, and completed tests/traces, not old B1/S1..4 releases. Ordinary fallback and explicit-mono rejection both happen before cache writes. Allocate epoch/scratch/IPC by actual concurrency. Borrowed routed Parameters require shuffle-completion/ownership records; replacing weights while runtime lives requires rebind/rebuild. Close after Graph completion and before TP teardown under the current IPC protocol.

A single layer with private initial cache does not establish real long-context/nonzero-history generation or MTP acceptance. Keep these unvalidated until in-scope tests complete.

## Carry evidence from prototype to production

A prototype establishes math/body, tails, resources, and scheduling in a declared range. Integration establishes actual specialization selection, guards/fallback, state/cache/Graph lifetime, and required E2E behavior. These stages can interleave through the [build playbook](../../port/references/build-playbook.md); prototype evidence does not automatically cover serving.

Record at the adapter handoff:

- Source revision/hash, compiler/flags, actual code object, replayable fixture/reference, semantics, and tolerances.
- Parameter axes, numerical/synchronization/layout invariants, resources, and separate static/correctness/performance ranges and gaps; see [experiment design](../../benchmark/references/experiment-design.md).
- Actual I/O, weight-pack/cache/state ownership, parallel/epoch contracts, and prepare/runtime boundary.
- Pairs, working set/Graph/device environment, rejected directions, and search limits; best configurations remain environment-bound.

Trace the real entry to prove the target specialization runs, then revalidate claims invalidated by actual changes rather than filenames or “only a template parameter” intent:

| Actual change | Old evidence no longer covers | Required update |
|---|---|---|
| Same body, new tile/wave/constants | Geometry, tail, register/LDS/residency behavior | Compile/resources, affected boundary numerics/performance; liveness if residency changes |
| New specialization/tail or broader guard | Branch correctness/dispatch boundaries; old buckets may take a new path | Actual selected branch, new-path numerics, affected old buckets/fallback |
| Math/reduction, layout, owner, publication/wait/epoch | Related numerical, state, or progress claims | Isolate; endpoints→full layer; replay/rank liveness for sync/state changes before timing |
| Framework guard/cache binding/Graph/dispatch | Same kernel can read wrong weights, duplicate writes, or select the wrong path | Real supported/unsupported entry boundaries and affected lifecycle; serving/E2E timing where relevant |

Matching code objects/`.text` strengthens evidence that device code is unchanged, not that host dispatch, inputs, or cache behavior is unchanged. Changed code requires affected resource/numerical/bucket checks. Reuse unaffected evidence, retain B during fixes, reproduce the original failure, and update status through the [iteration rules](../../port/references/iteration-loop.md).
