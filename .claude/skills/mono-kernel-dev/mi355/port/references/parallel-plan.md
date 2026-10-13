# Co-design parallelism and fusion

Use at P0 for deployment boundaries, P4 for cross-rank execution, or when performance stalls/workloads change. TP, EP, and CP are composable candidate axes. The current unfused plan is a starting point, not a proven optimum. This page contains design/validation rules, not new EP/CP measurements; historical CP scope is in the [indexer case](../../schedule/references/indexer.md).

Fusion changes launches, intermediate traffic, communication initiation/overlap, and resident resource cost. The best parallel plan can change with the partition. Revisit it when the critical path warrants it, not by rescanning every combination each iteration.

## 1. Separate semantics, deployment constraints, and tunable choices

Hold math, checkpoint, numerical criteria, global requests, and state semantics fixed. Record GPU count, topology, memory, latency/throughput goals, and required fusion scope. Treat user-fixed parallelism as a constraint; otherwise rank layouts, expert/weight/context splits, and communication protocols may be candidates.

Assign current and proposed deployments a `parallel_plan_id` and record per stage:

| Stage | Groups/members | Split axes and replicas | Weight/KV/state owner | Communication/output owner |
|---|---|---|---|---|
| Attention/indexer | Actual runtime groups | Head/context/query mapping | Physically sharded or replicated KV; current-token writer | Q/score/candidate or attention-partial exchange/merge |
| Routed/shared MoE | Separate TP/EP and nested groups | Expert allocation, intra-expert dimensions, shared replication | Global expert→rank/local expert; contribution owner | Dispatch, local partials, return, combine |

Attention may use TP/CP while MoE uses EP or TP within EP. Include activation redistribution between groups. Multiplying degree labels does not establish total GPU count, and `CP=1` does not imply one GPU. Distinguish cross-rank context parallelism from intra-GPU CTA partitioning, indexer score CP from decode-attention CP, and both from physical KV sharding.

Define concurrency by globally unique requests and per-request S_i; also record rank-local input tokens. Dispatched token-expert slots are not additional requests, and CP ranks cooperating on one request do not multiply throughput.

## 2. Identify the critical path changed by each plan

| Plan | Opportunity and cost | First discriminating experiment |
|---|---|---|
| TP: split tensors/intra-expert dimensions | Less local math; small-matrix inefficiency and output reduction/gather may dominate | Same-math external collective control; verify required partials, then test tile-ready communication |
| EP: distribute experts | Owners keep fuller expert matrices and may avoid some TP reductions; dispatch/return, small messages, and hotspot tails increase | Fix real routes; measure received slots/tasks, communication, and final combine; start with external communication |
| CP: distribute context | Share score/KV scan or attention work; add Q/partial/candidate exchange/merge and reduce tasks per rank | Hold selector/attention math, quantization, and cache semantics fixed; compare CP off/on or a few degrees, including scan and merge completion |

Communication follows actual layouts: TP is not always all-reduce; EP may use all-to-all, sparse push/pull, or an existing valid protocol. Renaming a down→TP stage to EP does not port its schedule.

### EP: establish a complete token round trip

Build `route → dispatch → owner-local UG/activation/down → return → original token-owner combine`, joining adjacent steps only where the actual implementation permits. Preserve `(request/token ID, original route slot, global expert ID, destination/local expert, route weight)` per contribution. Apply the route weight exactly once at the declared endpoint. Shared/residual contributions also have one accountable owner.

Check received records, expert-local math, returned values, and final token output before fusing dispatch→compute or down→return/combine. Estimate grouped-UG reuse from **actual slots/distinct experts on each owner after dispatch**; TP's global union may misestimate EP work. Retain intra-expert TP reductions where present.

At low concurrency, many EP ranks may be idle while one expert owner sets the tail. Record empty ranks and per-expert/peer live counts versus capacity. Test skew, zero messages, repeated experts, delayed peers, and buffer reuse. Do not silently drop tokens to fit fixed capacity. Zero-work participants still publish required completion. Identify who drives communication; resident waits must not occupy resources needed by a communication kernel or block unclaimed producers.

### Long context: distinguish scan, selection, and attention

Record real per-request live-length distribution, cache capacity, query count, refresh/reuse layers, and page tables; separate prefill and decode. Measure which work grows with live length and which scans repeat before choosing CP. Small batches can still offer context parallelism; long context does not guarantee merge/communication amortization.

Sparse-indexer CP must form complete scores before exact selection across context shards. With disjoint logical shards and a consistent total score/ID order, global top-k is contained in the union of local top-k. Budget candidate/histogram exchange for actual k. Do not select on incomplete per-head scores. See [indexer semantics and non-monotonic gains](../../schedule/references/indexer.md).

For standard softmax attention split across KV, retain each query/head's local maximum m_p, exponential sum l_p, and unnormalized weighted sum n_p. Merge using `m=max(m_p)`, `a_p=exp(m_p-m)`, and `sum(a_p*n_p)/sum(a_p*l_p)`. Do not average normalized local outputs. Handle empty shards according to masks and avoid `-inf-(-inf)` when all are empty. Derive other attention merges from their actual formulas and check original numerical criteria.

Replicated KV with split computation differs from physical KV sharding. The latter can save memory but changes writes, page ownership, migration, and Graph pointer contracts. Indexer CP does not establish attention CP. After sparse selection reduces attention work, further attention partitioning may only add merge cost.

## 3. Introduce plans through staged development

1. **P0:** inspect config/loaders/groups/traces; fix R and global semantics; record current/tunable plans. Represent remote experts, sharded KV, and independent requests in the first fixture when required.
2. **P1/P2:** build a staged chain for the chosen plan, initially with external communication. Derive local shapes, owners, and task domains. Single-rank checks do not establish rank progress.
3. **P3/P4:** fuse one measurable edge, validate numerics/state/communication progress, then measure resident/handoff costs. After a plan change, revisit the earliest invalidated contract without discarding independently validated math bodies.
4. **Each iteration/P5:** if hotspots, repeated scans, communication, or over-sharded small matrices dominate, propose one plan change with a prediction, falsification test, and cost limit. Do not simultaneously change precision and selection to obscure cause.

R's mathematical standard stays fixed; distribution and reconstruction are checked separately. Keep B for every `bucket + parallel_plan_id`, plus the deployment winner under equal constraints. Bind C/partition to a plan; never resume an EP candidate using TP weights/epochs.

A plan may require weight/KV redistribution and new process groups/Graphs. Per-bucket plan choice is valid only with prepared runtimes and legal state migration. Report setup/migration costs separately from steady state, and include transitions that actually occur in serving metrics.

## 4. Separate effects with two comparison axes

Let Q0 be the original plan, Q1 the candidate, U an available unfused/staged implementation, and F a fused implementation:

| | Q0 | Q1 |
|---|---|---|
| U | U0: original staged baseline | U1: staged implementation of the new plan |
| F | F0: fused original plan | F1: fused new plan |

- `U0↔F0` or `U1↔F1`: fusion at a fixed plan; N/P/F also fixes the plan.
- `U0↔U1` or `F0↔F1`: strategy at a fixed implementation organization; disclose necessary adaptation changes.
- Final deployment: compare with the **best known unfused deployment under equal resource constraints**, including dataset, outputs/state, communication, and inter-layer conversions. `U0↔F1` can establish a combined gain but cannot attribute it to fusion alone.

Build only controls needed for the hypothesis, not four mandatory production implementations. Mark missing cells unsupported/not measured and limit attribution. Do not ignore a known better unfused plan. Hold total GPUs/topology, global request trace, context distribution, numerical criteria, and memory constraints fixed. With different resources, report cost/throughput separately rather than claiming equal-cost speedup.

Interpret PMC/ATT by actual work ownership. Retain per-rank token/expert/context work and raw counts; sum only compatible intervals/units. **A rank-local MFMA/DRAM decrease may mean work moved to another rank.** ATT explains covered local stages; transport completion and hotspot ranks need rank/peer traces. Measure the valid request/layer completion path rather than hiding tails in rank averages. Full-layer and required E2E uninstrumented pairs remain acceptance evidence.

Specify the next strategy and discriminating result: for example, staged EP dispatch/return on fixed routes to separate TP reduction cost from an EP hotspot, or CP off/on with the same long selector to test scan savings against merge cost. If no improvement is established, retain B and the rejected condition; enabling EP/CP alone is not progress.
