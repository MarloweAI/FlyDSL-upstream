# Embedded cases and historical provenance

Curated on 2026-10-12 from historical experiments and code reviews; no new GPU measurements. G/K/M/D/R/S identifiers retain historical relationships. All operational guidance is embedded; original-machine reports, sessions, and source snapshots are not prerequisites.

Discover current configuration/capability through [current config, loader, arguments, metadata, and dispatch](model-config-discovery.md). Historical cases are not model parameter tables.

## Optimization and engineering cases

| ID | Embedded reference | Supported lesson and limits |
|---|---|---|
| G1 | [Layout/prefetch](../../schedule/references/layout-prefetch-case.md), [integration gap](../../atom/references/integration.md) | Scale tiles/gate-up, shared early, working sets/actual dispatch, negative results; historical BF16 MFMA differs from native scaled MFMA |
| G2 | [Bounded selection/CP](../../schedule/references/indexer.md) | Complete scores, exact candidates, live/capacity, query reuse, CP controls/non-monotonic gains |
| G3 | [GLM replay race](../../schedule/references/synchronization.md#replay-race) | LDS threshold/scan overwrite→out-of-bounds ready corruption; register load then barrier before reuse; observer masking |
| G4 | [Expert union/task domains](../../schedule/references/reuse.md), [walkthrough](worked-example.md) | Masks/original slots, UG tile16/8 missing writes, worst-case capacity; local UG wins do not establish a layer win over unfused |
| G5 | [Unique quantization owner](../../schedule/references/reuse.md), [diagnostics](../../benchmark/references/diagnostics.md) | Complete groups, two owner designs, preserved order; logical payload and DRAM requests do not scale together |
| G6 | [Multi-request integration](../../atom/references/integration.md), [query-K reuse](../../schedule/references/indexer.md) | B/S/T, independent pages/causal bounds, single cache updates; B1/S8 is not B2/S4 |
| K1 | [K3 output/production boundaries](../../atom/references/integration.md), [numerical paths](../../numerics/references/contracts.md) | Prefix/routed/shared materialization, format differences, B per bucket; keep environment/timing scopes separate |
| K2 | [Winner score/bias](../../numerics/references/contracts.md), [prefetch caught by waits](../../schedule/references/layout-prefetch-case.md) | Winner ID/score/bias consistency; earlier ISA loads do not establish overlap |
| K3 | [MTP/epoch](../../schedule/references/synchronization.md), [integration](../../atom/references/integration.md) | S+1 predecessor snapshots; rediscover current ABI/support |
| K4 | [Resident regression](resident-regression-case.md) | N/P/F timing/resources, failed tuning, output-owned down/tagged payload follow-up; unresolved golden stays unresolved |
| K5 | [Build playbook](build-playbook.md), [structural redesign](resident-regression-case.md) | Runnable subgraphs before full fusion, real savings budgets, CTA cooperation, latent/TP dependencies |
| M1 | [M3/GLM indexer semantics](../../schedule/references/indexer.md), [cache integration](../../atom/references/integration.md) | Block/token top-k, candidate owners, CP helper versus public entry; review is not a new benchmark |
| M2 | [Expert/query reuse](../../schedule/references/reuse.md), [CP/query tasks](../../schedule/references/indexer.md) | Token masks, per-token summation, measured TP/quantization dispatch; old thresholds or PR latency are not current measurements |
| D1 | [Staged/fused epochs](../../schedule/references/staged-epoch-case.md), [branch numerics](../../numerics/references/contracts.md), [HCState/borrowed weights](../../atom/references/integration.md) | Shared builder/generation, numerical/ABI counterexamples; current config/layer kinds come from the project |
| D2 | [Within-stage token distribution](../../schedule/references/reuse.md), [native drift](../../numerics/references/contracts.md), [fusion boundary](../../atom/references/integration.md) | Stage-major versus token-major, controlled reference versus native, MoE versus full-layer launch count |
| R1 | [Routing mismatch classes](../../numerics/references/contracts.md#routing) | Fixed logits/bias, slot/set/tie distinctions, near-score truncation, expert alignment; diagnosis does not waive ABI order |

## Session knowledge retained in the bundle

[PMC cases](../../benchmark/references/pmc-cases.md), [ATT cases](../../benchmark/references/att-cases.md), and the [recomputable extract/source hashes](../../benchmark/references/data/pmc-att-evidence.json) preserve S4/S5 quantitative reasoning: cache serialization→UG work prediction→quant-once→ordered down, plus working-set/prefetch/layout ablations. Raw quantities, scope, and identities do not depend on machine paths.

Earlier curation read persistent context from six related sessions and checked it against engineering reports. These labels map knowledge, not required external inputs; original sessions are not bundled.

| ID | Historical topic | Embedded guidance |
|---|---|---|
| S0 | Initial GLM/K3 fusion boundaries | [Model mechanisms](model-map.md), [build boundaries](build-playbook.md) |
| S1 | K3 staged/mono, resident structure, performance goals | [Regression case](resident-regression-case.md), [RSI loop](iteration-loop.md) |
| S2 | Failed token pipelines, dense/tail reuse, selector, replay | [Reuse](../../schedule/references/reuse.md), [synchronization](../../schedule/references/synchronization.md), [numerics](../../numerics/references/contracts.md) |
| S3 | DSV4 routed/shared paths, full-layer API, ATOM | [Numerics](../../numerics/references/contracts.md), [integration](../../atom/references/integration.md) |
| S4 | GLM layout, M3 review, indexer CP, multiple requests | [Layout](../../schedule/references/layout-prefetch-case.md), [indexer](../../schedule/references/indexer.md) |
| S5 | Grouped UG, quant-once, next direction | [Reuse](../../schedule/references/reuse.md), [walkthrough](worked-example.md) |

Historical memory search found no matching mono/Kimi/GLM/DSV4/M3 entries, so no memory-derived conclusions were invented. The bundle retains case/method relationships, evidence limits, and numerical extracts, not original-machine locators, sessions, or database entries. Source hashes identify historical inputs; they do not establish a GPU binary or a new run. Missing original inputs prevent full reproduction from the bundle alone.

## Development method index

These methods are embedded and require no external agent repository or checkout:

| Decision | Reference | Required result |
|---|---|---|
| Estimate remaining opportunity | [Experiment design](../../benchmark/references/experiment-design.md), [diagnostics](../../benchmark/references/diagnostics.md) | Separate theoretical, empirical, and dependency assumptions; scoped evidence for a falsifiable mechanism |
| Bound parameter search | [Experiment design](../../benchmark/references/experiment-design.md) | Axes, invariants, pruning premises, three coverage ranges, compile/code-size/GPU budgets |
| Map body tiles and storage | [Tile contract](../../schedule/references/tile-contract.md) | Coverage, physical addresses/masks, quantization groups, live ranges, compiled resources; new boundary checks are design guidance |
| Design persistent execution | [Worker lifetime/scheduling](../../schedule/references/persistent-design.md) | Finite/service mode, coverage/readiness, progress, completion/drain/exit; service rules are not historical measurements |
| Choose parallelism with fusion/workload | [Parallel plan](parallel-plan.md) | Stage mappings, ownership/communication, fixed-plan/strategy controls; new EP rules are design reasoning |
| Resume and choose an action | [RSI loop](iteration-loop.md), [iteration card](../assets/iteration-card.md) | Reconciled R/B/C, attempted→effect, rejected conditions, evidence identity, next action |
| Move prototype into the framework | [Integration/revalidation](../../atom/references/integration.md) | Handoff contracts; branch, numerical, state/Graph, or performance evidence affected by actual changes |
| Convert failures into reusable rules | [Rule updates](iteration-loop.md) | Original failure plus unaffected-neighbor checks, trigger/expiry conditions, unvalidated/rejected changes |
| Accept and deliver | [Delivery](delivery.md) | Final-version functional/numerical/state/performance matrix, runnable artifacts, incomplete scope |

Historical GPU claims remain scoped by their cases. Document curation and decision-path review are not new GPU measurements or agent-performance certification.

Add new counterexamples to the relevant embedded reference as `trigger → prior hypothesis → experiment → observation → rejected/applicable conditions → next action`. Do not add only a machine path or generalize a local failure into a ban across models.
