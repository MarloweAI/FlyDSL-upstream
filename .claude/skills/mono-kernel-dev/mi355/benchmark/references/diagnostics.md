# Use PMC, ATT, and completion markers to choose the next change

Deliver a testable causal chain: **candidate anomaly → work/dependency prediction → scoped data → supported/excluded explanations → minimal change → uninstrumented acceptance → next iteration**. Counter percentages or “inspect PMC/ATT” alone are not an analysis.

## Establish what the data can answer

Before interpretation, check each arm's exact binary/fixture, comparable output/work scope, metric/time units and interval/auxiliary calls, and actual ATT coverage of the target stage. Rebuild work ownership after strategy changes. Limit claims or repair observation when fields are missing; never substitute zero or another version's data.

Quick routing: [unchanged counts with earlier dependencies](pmc-cases.md#pmc-cache) → serialization; [exact MFMA reduction](pmc-cases.md#pmc-grouped-ug) → routes/tasks/body; [large logical-byte reduction, little DRAM change](pmc-cases.md#pmc-quant-once) → preparation/LDS; [higher local stall, faster layer](att-cases.md#att-prefetch) → redistributed waiting; [uncovered stage](att-cases.md#att-coverage) → evidence gap.

Use [PMC cases](pmc-cases.md), [ATT cases](att-cases.md), and, only for recomputation, the [numerical extract](data/pmc-att-evidence.json). The extract contains raw-instance PMC totals, normalized values, trimmed ATT aggregates, and source hashes, not all raw traces.

## 1. State competing explanations

Start from a full-layer paired anomaly with matched baseline/candidate, fixture, bucket, binary, and timing scope. Write at least two explanations distinguishable by observation:

| Hypothesis | Prediction before coding | First observation | Change justified by support |
|---|---|---|---|
| Repeated expert computation across tokens | Route/task/body-derived MFMA reduction | `SQ_INSTS_MFMA`, live tasks, independent numerics | Expert/token-column reuse; UG before down |
| Repeated quantization/mid preparation | Same MFMA, less quant/DS work; account for tags in logical bytes | LDS instructions, quant ISA, consumer preparation interval | Unique quant owner, original rounding |
| Single-owner serialization | Similar instructions/traffic, earlier producer and dependents | Task mapping, rank-local completion points, paired PMC | Query/tile distribution, same state ownership |
| Register pressure causes a resource cliff | Occupancy tier change or scratch; dynamic attribution needed | ELF/occupancy, then scratch/wait PCs | Shorten real live ranges or redesign body |
| Exposed load addresses/dependencies | Higher cycles/hit at equal hits, or more gathers/address instructions | Matched ATT callsites, ISA, applicable cache/DRAM counters | Packer+loader or legal earlier loads |
| TP sets the tail | Last local output/peer availability determines completion | Local-ready, peer-consume, rank completion | Output-tile publication/reduction scheduling |
| EP load/communication sets the tail | Hotspot slots/tasks or dispatch/return delays combine | Rank/expert work, peer messages/completion, local PMC/ATT | Owner-specific grouping/communication granularity/plan |
| Long context is insufficiently partitioned | Same-algorithm scan can shrink; merge may offset it | Live length, per-rank context work, scan/merge/layer completion | Indexer/attention CP off/on, separately |

Select tools to distinguish the hypotheses; do not collect everything every round. Existing valid evidence may suffice. Predictions need a count, load, or completion point, not just “faster.”

## 2. PMC: turn interval records into comparable work counts

Read current tool metadata for definitions, units, dimensions, agent scope, and compatible metric sets. Record tool version/warnings and the exact loaded metric configuration. Do not copy historical gfx950 event selectors or infer units from names. Distinguish unsupported, defined-but-uncollected, and valid zero. Different passes are not simultaneous; ratios of independently maximized values can mix different instances.

Key raw records by `pair/run → interval/pass → agent → label → metric → instance/record ID`. Sum dimensions only within one valid interval. Confirm collector behavior for cumulative reads, resets, and rerun passes; do not sum repeated snapshots or ranks unconditionally.

Derive main/auxiliary calls and grid/waves from traces/Graphs. Check empty intervals and expected waves. Historical 32 layers plus two 4-wave step kernels produced 65,544 waves, not 65,536. The extra eight establish auxiliary work, not a license to discard it or assume its counters are zero. Wave agreement checks accounting, not permissions, all metrics, or absence of perturbation.

Report:

| Metric/definition/unit | Raw instance total and scope | Main/auxiliary calls | Baseline normalized | Candidate normalized | Absolute/relative change | Quality limits |
|---|---|---|---|---|---|---|
| Actual supported metric | Actual records | Trace-derived | Measured/unknown | Measured/unknown | Only when valid | Warnings, missing data, pass relationship |

Interpret evidence carefully:

- MFMA agreement with task/body predictions supports reduced matrix work. Equal instructions do not imply equal useful columns/FLOPs.
- Fewer LDS instructions means less LDS execution work, **not fewer bank conflicts**. Conflicts require defined metrics/normalization and lane/DS mapping; mark unmeasured if unavailable.
- Convert DRAM requests using the metric's request unit, not unique tensor or weight capacity. TCC hit rate is total hits/(total hits+total misses), not an average of slice percentages.
- Separate private bytes and VGPR/SGPR spill metadata from dynamic scratch traffic. Stage attribution needs actual scratch instructions/callsites or suitable dynamic evidence.
- Device-counting intervals include polls, auxiliary kernels, and all covered work. Stage attribution needs controlled stages/minimal changes plus ISA/timeline support.

Retain recomputable raw values, normalization, and structural predictions. On a missing MFMA reduction, first check capacity versus live tasks, masked work, branches, binary, and denominator. A speedup with unchanged counts points to dependencies/parallelism, as in the cache case.

After TP/EP/CP changes, rebuild rank-local predictions for identical global inputs. A local count reduction may be relocation, not deletion. Keep load distribution and aggregate only compatible intervals/units; use the [parallel controls](../../port/references/parallel-plan.md).

## 3. ATT: establish stage coverage before ranking hotspots

Select sampling locations from actual stage/task/CTA roles. CU/SE/SIMD IDs alone do not ensure coverage; inspect captured wave PCs/source ranges. Without UG/down samples, the trace cannot rank MoE bottlenecks. Additional CUs/seeds check stability, not whole-GPU extrapolation.

Verify decoder version/columns, lost/incomplete status, valid waves, launch counts, trim rules, and per-wave/aggregate reconstruction. Do not discard initial anomalies merely to improve results. Explain capture-start/working-set warmup and apply the same exclusion rule to both arms, retaining exclusions.

Map the captured code object to exact `.text`/addresses and a matching debug line table, including the full inline chain. Keep:

```text
binary/.text identity, PC, ISA, source operation/callsite, stage/CTA coverage
hits, stall cycles, cycles/hit, retained waves/launches/CUs/seeds
load/wait operands, producer, wait scope, independent work available earlier
```

Compare groups with `sum(stall)/sum(hits)`. When hits change, report totals and per-hit values. Raw stalls from different wave counts are not directly comparable; unweighted averages of unequal groups are not weighted ratios. PCs move across builds, so align semantic callsites rather than reusing old addresses.

| Hotspot | Dependencies to inspect | Discriminating experiment |
|---|---|---|
| Weight/scale load | When addresses become known; gather/descriptor dependencies; lane addresses; load-to-MFMA distance | Same-math layout control or earlier known-address loads; applicable PMC |
| `s_waitcnt vmcnt(...)` | Payload, weights, scratch, or multiple outstanding loads; whether prefetch shares the wait | Generated order and legal wait domain; change loader organization, not blind wait deletion |
| Mixed vmcnt/lgkmcnt | VMEM and LDS/SMEM producers both in the wait set | Trace both and retain a mixed category |
| `s_barrier` | Early/last waves and their preceding work, memory, reduction | Last-completion chain and task/producer adjustment; stall sum is not removable time |
| MFMA/adjacent DS | Valid columns/tasks, operand arrival, dependencies/resources | MFMA prediction and lane layout, not tile size alone |

VMEM-load stalls do not independently prove HBM misses, saturated bandwidth, or full queues. Those need cache/traffic/backpressure evidence. Shared-early demonstrates that a local stall category can increase while the layer gets faster.

## 4. Connect work changes to layer completion

Label markers as issue, store completion, visible publication, first useful math, or last completion; “end” alone is ambiguous. Stage intervals can overlap and must not be added indiscriminately.

Use a valid origin per rank; cross-rank clocks are not assumed synchronized. Verify the current `s_memrealtime` timebase; historical 100MHz is not inferred from core clock. Compare instrumented/uninstrumented LDS/VGPR/private/spill and `.text`. Identical observer source can perturb candidates differently.

Test dependency propagation: cache completing about 14 µs earlier with indexer/O/router also earlier supports a critical-path explanation. Fewer UG tasks while down waits on the entire mid tensor suggests a handoff opportunity. `down start < UG end` may mean only that down started waiting. Compute overlap requires `first_down_math < last_UG_complete` with valid marker semantics.

## 5. Make the explanation determine the next action

| Combined evidence | Conclusion | Next action |
|---|---|---|
| MFMA reduction exactly matches routes/tasks and layer improves | Repeated compute was removed | Keep the change, remeasure downstream, mark work eliminated |
| Same MFMA, fewer LDS instructions, little DRAM change, faster layer | Less preparation/quantization/LDS work; HBM explanation insufficient | Inspect remaining down math/handoff |
| Similar counts with earlier dependent completion | Less serialization/scheduling delay | Keep scoped change and address unchanged work |
| ATT stalls move, event timing reliably improves | Scheduling redistributes waiting | Preserve both the gain and opposing signal; target remaining hotspots |
| Fewer resources but slower, without matched dynamic evidence | Regression established, cause unresolved | Targeted observation or another mechanism; do not borrow old ATT |
| Missing data, uncovered stage, wrong binary | Mechanism cannot be judged | Repair observation/coverage; do not fill zero or claim falsification |

Validate original numerical/state/replay criteria and confirm with uninstrumented full-layer pairs plus affected buckets. Counter percentages, sampled cycles, and static resources are not performance acceptance gates. Record supported explanations, remaining alternatives, eliminated cost, and the next experiment.

The diagnostic handoff needs five items: **observation/scope; structural prediction and measured difference; supported explanation; unresolved alternatives/gaps; next discriminating experiment**. Record correctness/performance status separately; “bottleneck found” does not mean passed.
