# Fused sparse indexing and context parallelism

Sources: [M1 M3 review, G2 GLM CP, G3 replay race](../../port/references/sources.md). CP partitions context score work; it does not automatically shard physical KV storage.

These historical mechanisms concern indexer scoring/selection. They do not establish decode-attention CP. Use the [parallel plan](../../port/references/parallel-plan.md) to inspect attention scans, page/KV ownership, and partial merges. Mono may use a different strategy from unfused, with separate evidence.

## Specify selection semantics first

Use [config discovery](../../port/references/model-config-discovery.md) on indexer config, score/select code, and attention consumers. Record selection unit, head aggregation, top-k, causal bounds, and owner. These historical numbers illustrate semantic differences; they are not new-model defaults:

| | Reviewed M3 path | Reviewed GLM path |
|---|---|---|
| Selection unit | 128-token block | Individual token |
| Score | Per-head query-key dot, causal max within each block | `sum_h(weight_h * ReLU(dot(q_h,k_token)))` over all 32 heads |
| Top-k | Top 16 blocks per index head | Top 2048 tokens after score aggregation |
| Consumer | Owner rank of each head | Attention TP ranks need the same request's indices |

A final count near 2048 keys does not imply the same selection problem. GLM requires complete token scores before local top-k; selecting per-head top-k first generally does not preserve top-k of the weighted sum.

With a consistent total score/ID order and valid-value policy, global top-k lies in the union of shard-local top-k. Exact merging can use this, but candidate capacity is about `P*k`. Replacing 16 with 2048 in a small-candidate M3 routine ignores that cost. Define ties, NaNs, and fewer-than-k behavior.

## Bound LDS and separate live length from capacity

Putting all context's 32-bit keys in one CTA's LDS scales linearly with L: 1M keys require 4 MiB. Prefer existing bounded tile histogram/radix/compaction or local-candidate/merge selectors.

Historical GLM CP v9 exchanged complete aggregated scores for a short-live-length exact selector; the long path used 4K-token tiles with distributed radix/count/scan/emit. The short selector dispatched on live length, while some projection specializations still used capacity. Both lengths belong in the contract; these thresholds are not prescriptions.

Compare CP off/on with the same long selector first. Otherwise algorithm changes confound the CP effect. Search the current project for bounded selectors, CP code, and public entry points before implementing a historical TODO.

## Reuse and parallelism for few queries

Map multiple query tokens from the same request to MFMA columns to reuse index-K, retaining each query's causal bound. Different requests may share projection weights, not KV pages or visible-token sets.

CP reduces score work per rank but can reduce short-context tasks far below the resident grid. Estimate `score_tasks_per_rank ≈ sum_i ceil(L_i/(P*key_tile)) * query_groups_i`, then inspect actual assignment. This is not an occupancy measurement.

Long contexts can amortize communication, yet historical GLM 128K/S1 CP still regressed. Do not define a monotonic “CP wins above L” rule. Measure boundaries across S, B, live length, capacity, TP, selector, and real inputs.

## Publication and production interfaces

- After writing the current key, only score tasks that consume that request/slot need its dependency; derive this from real metadata.
- Define epochs, owners, and last-consumer completion for CP Q, scores, histograms/candidates, and final indices. Do not reuse a previous layer's ready state.
- Skip projection/score/select only on model-defined index-reuse layers, not arbitrary adjacent layers.
- Align BF16 contiguous prototypes with FP8/FP4 paged production caches separately. Multi-request support needs per-request metadata. Prototype 1M support does not establish production 1M integration.
- Check short/long dispatch boundaries, empty shards, delayed ranks, ties, short live/large capacity, independent requests, page edges, and long Graph replay.

A CP helper does not establish public enablement. Reviewed M3 ATOM optional paths differed from AITER's default one-head path. Record actual builder parameters, traces, and output ownership.
