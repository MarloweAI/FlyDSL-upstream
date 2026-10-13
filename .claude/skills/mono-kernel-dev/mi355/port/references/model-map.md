# Match reusable mechanisms to model semantics

GLM/K3/M3/DSV4 are historical case labels, not current configuration or capability tables. First use [config discovery](model-config-discovery.md) on the current config, resolved arguments, loader/sharding, forward, and dispatch. Then match semantics below. No historical source checkout is required.

| Structure in the current model | Reusable mechanism | Confirm in current code/config |
|---|---|---|
| MLA/indexer → router → MoE → TP, GLM-like | [Scale/gate-up layout](../../schedule/references/layout-prefetch-case.md), [expert union/quant-once](../../schedule/references/reuse.md), [bounded selection/CP](../../schedule/references/indexer.md) | Head aggregation/selection unit, top-k, bias/probabilities, IndexShare, cache format, MFMA/quantization branch |
| Recurrence plus latent/shared/routed branches, K3-like | [Resident redesign](resident-regression-case.md), [multi-token dense/tail reuse](../../schedule/references/reuse.md), [state/integration](../../atom/references/integration.md) | Predecessor chains, decode/MTP snapshots, activation, latent dimensions, AttnRes, TP/rounding, output materialization |
| Sparse indexer with multiple queries per request, M3-like | [Context partitioning/candidate ownership/query-K reuse](../../schedule/references/indexer.md), [expert union/token masks](../../schedule/references/reuse.md) | Per-head/block/token selection order, causal masks, reuse layers, real CP entry, independent page tables |
| Resident MoE with distinct routed/shared numerics and complex layer state, DSV4-like | [Shared staged/fused builder](../../schedule/references/staged-epoch-case.md), [within-stage token scheduling](../../schedule/references/reuse.md), [borrowed weights/state](../../atom/references/integration.md) | Routing/special layers, quantization rules, HCState/mHC, compressor commit, MoE versus full-layer launch boundary |

## Re-derive three decisions

**Token-column reuse versus earlier pipelining.** Historical K3 group1/2 changes increased weight reads, task rounds, and resources. DSV4's within-stage task distribution is a different change. Calculate rounds per CTA and repeated reads for current weights/T before splitting tokens; “low concurrency” does not prescribe a loop structure.

**Expert grouping versus a narrow path.** Compare actual routed slots with distinct-expert union U. Small U offers reuse; when U approaches slots, table construction can dominate. Derive thresholds from supported configurations and paired measurements, not old M3 wide/TP-reduce constants. Verify dispatch and numerics, then retain winners per bucket.

**Indexer structure versus indexer mathematics.** M3 per-head block top-k and GLM token top-k after head aggregation solve different problems, even with similar final key counts. Reuse bounded LDS, context partitioning, publication, and same-request K reuse; reimplement the actual score/selector/consumer contract. Search for current implementations before treating a historical TODO as missing functionality.

## Rediscover current capability

Inspect public guards, builder arguments, completed checks, and traces. Report compile feasibility, numerical/state/replay coverage, and performance coverage separately. Historical B/S/TP coverage, 7/9 staged launches, MoE/full-layer scope, and paged/contiguous branches are version-specific facts.

Grid/block, tiles, LDS, prefetch, grouping/CP thresholds, tolerances, and latency belong to exact versions. Check current gfx950 tools/device properties and helper implementations for hardware capacity, occupancy, and ordering. The cases are not a hardware-constant table or synchronization library.
