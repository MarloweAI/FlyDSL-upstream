# PMC case studies: guide each GLM optimization with measured work and readiness

These analyses were performed in historical GLM sessions; no new GPU measurements were invented for this guide. The cases retain predictions, data, tradeoffs, and follow-up decisions. v6/v8/Group4/Pack3/v9/v10 are historical version labels, not current defaults.

The tables below preserve normalized historical observations, metric definitions and structural predictions. They are self-contained extracts, not bundled raw traces or new measurements. Uninstrumented timing, PMC and stage clocks are separate paired experiments, not one additive time decomposition.

**Timing scope for cases 1–3.** The cache, grouped-UG, and quant-once C2/S6 uninstrumented pairs used `5 trials × 256 replays × 16 layer calls`, aggregated as **rank median, then trial median**. Their improvements establish gains in that historical statistic; they do not by themselves establish lower slowest-rank or request-completion latency. The historical decisions below retain that scope. Promoting a current B for a completion-time target requires valid timing of that completion boundary and supporting rank/trial records; do not relabel these summaries or infer missing rank tails.

<a id="pmc-scope"></a>

## Define collection scope and normalization

All four PMC pairs used MI355X (gfx950), TP4, C2/S6, indexed native A8W4, and GPU0 device counting. Each interval contained `2 graphs × 16 layer calls` plus two step updates. Each main kernel used 256 workgroups ×8 wavefronts; each step update used 4 wavefronts. Counts were not summed across the four ranks.

```text
expected_waves = 32 × 256 × 8 + 2 × 4 = 65,544
waves_per_layer_with_step = 65,544 / 32 = 2,048.25
counter_per_layer = sum(valid metric instances in one arm/agent/interval) / 32
DRAM_request_MB = TCC_EA0_RDREQ_DRAM_32B × 32 / 1,000,000
TCC_hit_pct = 100 × sum(TCC_HIT) / (sum(TCC_HIT) + sum(TCC_MISS))
```

`SQ_INSTS_MFMA` counts issued MFMA instructions; `SQ_INSTS_LDS` counts LDS instructions, including FLAT in the provider definition. Neither is FLOPs or bank conflicts. `TCC_EA0_RDREQ_DRAM_32B` counts 32-byte requests, with 64/128-byte requests counted as 2/4. It does not identify weights, payload, or scratch. TCC misses include UC reads.

Historical collection reported SYS_PERFMON permission/lock warnings. Empty intervals, wavefront accounting, instance records, and paired controls strengthen internal consistency but do not remove that limitation. Use these counters for diagnosis/cross-checking, not validated absolute HBM bandwidth. Re-read definitions/dimensions for a new toolchain rather than copying event selectors.

<a id="pmc-cache"></a>

## 1. Cache parallelization: same work, earlier completion

**Observation/hypothesis.** One workgroup serially performed cache norm/RoPE/write for multiple queries, delaying the indexer. The hypothesis concerned readiness, not excessive GEMM work.

**Minimal change.** v6→v8 distributed cache tasks by query in the validated range, preserving per-query math, write ownership/readiness, and downstream workgroup offsets.

| Per-layer metric | v6 | v8 | Interpretation |
|---|---:|---:|---|
| MFMA instructions | 824,384 | 824,384 | No arithmetic-instruction reduction |
| LDS instructions | 3,114,615 | 3,110,823 | About 0.12% lower |
| DRAM request MB | 592.292 | 591.338 | About 0.16% lower |
| TCC hit rate | 63.998% | 64.087% | No evidence of a large hit-rate improvement |
| Waves, full interval | 65,544 | 65,544 | Consistent launch/wavefront accounting |

Matched reduced-observer `s_memrealtime` used the verified 100MHz timebase of that environment. These completion milestones are relative to rank-local origins, aggregated as rank median then snapshot median over 15 snapshots per rank. They are neither cross-rank latest-completion times nor additive stage durations:

| Completion point, µs | v6 | v8 | Earlier by |
|---|---:|---:|---:|
| QKV | 23.960 | 23.830 | 0.130 |
| Cache | 39.190 | 25.420 | 13.770 |
| Index selection | 70.300 | 56.455 | 13.845 |
| Output projection | 128.040 | 114.540 | 13.500 |
| Router | 134.430 | 120.730 | 13.700 |

Uninstrumented C2/S6 indexed rank-median timing was 256.291→242.410 µs: 1.0573× speedup, about 5.42% lower latency under that statistic. Similar work plus earlier cache and dependent milestones supports reduced serialization. Unchanged PMC does not imply no optimization, and 13.77 µs is not a universal recoverable budget.

**Observer counterexample.** v6/v8 private-segment allocation was 428/432 B/thread without markers and 452/468 with markers. The instrumented composite down interval increased from 49.14 to 52.54 µs. The same observer source can perturb two candidates differently; historical acceptance used the uninstrumented rank-median pairs.

**Historical decision/next step.** The scoped cache change passed the original eight-point checks. A broader policy regressed C1/S6 off, so dispatch was narrowed. Unchanged MFMA/DRAM left repeated MoE work as the next opportunity: expert grouping. If cache advances without downstream/layer movement in a new model, reassess criticality before adding more cache workgroups.

<a id="pmc-grouped-ug"></a>

## 2. Grouped UG: predict and verify the MFMA reduction

**Fixture facts.** C2/S6 had 96 routed slots and 35 distinct routed experts. The shard's intermediate was 512, with UG tile8, 64 tiles/expert, and 48 issued MFMA instructions/tile. Shared already reused across tokens and was accounted separately.

```text
removed routed tile instances = (96 - 35) × 64 = 3,904
predicted MFMA reduction = 3,904 × 48 = 187,392
grouped live tasks, including shared = (35 + 1) × 64 = 2,304
```

For a new model, derive the prediction from actual routes, task mapping, and generated ISA. The debug value 6400 was capacity, not 2304 live tasks. An initial 16-element task mapping with an eight-element device implementation left outputs unwritten and stalled consumers; coverage was fixed before timing.

**Change.** Expert union, token masks, and original-slot mapping placed tokens sharing an expert into MFMA columns. FP32 intermediate activations, down math/quantization/reduction order stayed fixed. UG staging padding also changed, so LDS/DRAM effects cannot all be assigned to one submechanism.

| Per-layer count | v8 | Group4 | Change |
|---|---:|---:|---:|
| MFMA | 824,384 | 636,992 | −187,392; −22.73% |
| LDS instructions | 3,110,823 | 2,578,217 | −17.12% |
| DRAM 32B requests | 18,421,314.625 | 12,729,284.375 | −30.90% |
| Waves, amortized step included | 2,048.25 | 2,048.25 | Unchanged |

**Interpretation.** Exact MFMA agreement establishes that repeated UG execution was skipped, rather than merely zeroing inactive columns while issuing every instruction. DRAM changes can include cache, scratch, and other accesses. Whole-kernel private-segment allocation of 432→120 B/thread and VGPR spill metadata 107→29 indicate resource changes but do not locate all spilling in UG.

UG last completion moved 184.330→158.875 µs and down last completion 259.010→210.710 µs. The composite down interval 59.22→41.44 µs includes staging of intermediate activations, quantization, loads, MFMA, and local reduction; it is not 17.78 µs of pure MFMA savings. Uninstrumented indexed rank-median timing improved 237.437→197.272 µs, or 16.92% under that statistic. An unfused path on another fixture needs its own paired comparison.

**Historical decision/next step.** Work prediction, counters, completion points, and timing agreed, supporting acceptance under the historical rank-median metric. This does not establish a slowest-rank or request-completion win. Next inspect repeated consumer quantization. If a new candidate misses the predicted MFMA reduction, check dynamic task skipping, masked MFMA, dispatch/binary, and denominator before blaming noise.

<a id="pmc-quant-once"></a>

## 3. Quant-once: 71.875% fewer logical bytes, only 0.58% fewer DRAM requests

**Starting point.** Group4 had reduced UG work, but every down workgroup still read tagged FP32 intermediate activations and repeated the same quantization. UG8 alone could not form a 32-element scale group.

Pack2 assigned four UG8 tiles to one producer, reducing task parallelism. Pack3 retained UG scheduling and used one owner to gather a complete group, apply the original FP32→MXFP8 rule, and publish packed values/scales. The down projection preserved the accumulation order across slots, K tiles, wavefronts, and TP ranks. Pack3 added neither a kernel nor a global completion flag.

At T12, logical payload per down workgroup fell 442,368→124,416 bytes, or 71.875%. The owner additionally read FP32 once per rank and published packed data. Repeated logical requests are not HBM reads; many may hit cache.

| Per-layer count | Group4 | Pack3 | Change |
|---|---:|---:|---:|
| MFMA | 636,992 | 636,992 | 0 |
| LDS instructions | 2,578,217 | 2,439,977 | −5.36% |
| DRAM 32B requests | 12,855,439.75 | 12,780,522.4375 | −0.58% |
| Waves, step included | 2,048.25 | 2,048.25 | 0 |

This Group4 DRAM baseline differs from the previous pair because it came from a fresh collection. Compare within this pair. Matched uninstrumented indexed rank-median timing was 199.860→193.182 µs, 3.34% lower under that statistic; the historical C2 matrix reported about 3.34%–5.06% improvement. These figures do not establish lower slowest-rank or request-completion latency.

**Supported/rejected explanations.** Code moves repeated quantization to one owner; lower LDS work with unchanged MFMA is consistent with less consumer preparation. The data rejects proportional HBM-weight-traffic reduction as the main explanation. It does not isolate exact quantization time or demonstrate fewer bank conflicts; those require finer instruction mapping/metrics.

**Next step.** Quantization is now shared; stop proposing it as missing work. Down still issues the same token×slot×K MFMA sequence, so investigate reuse within the original summation order. If a later change reduces only logical bytes without changing preparation/ISA or layer completion, revisit its proposed mechanism.

<a id="pmc-down"></a>

## 4. Ordered down: verify work and numerical order

Historical v9→slot3 grouped tokens with the same expert only within each original wavefront and slot. It preserved each token's slot/K contribution sequence and final wavefront reduction. Old LDS tables were reused for reduction only after all readers finished; slot2 had failed this lifetime rule.

The PMC fixture had 432/380 old/new units per output group, 2 MFMA/unit, and 192 groups:

```text
predicted_saved = (432 - 380) × 2 × 192 = 19,968
observed_saved = 636,992 - 617,024 = 19,968
```

| Per-layer count | v9 | slot3 | Change |
|---|---:|---:|---:|
| MFMA | 636,992 | 617,024 | −3.13% |
| LDS instructions | 2,439,977 | 2,222,633 | −8.91% |
| DRAM 32B requests | 12,634,031.3125 | 11,472,098.1875 | −9.20% |

This supports reduced execution from ordered grouping. Do not substitute the 432→392 route table from another real-weight fixture. Native bitwise checks still matter; a CPU order audit does not replace them. Compatibility and uninstrumented results supported adoption. Real-weight C2/S6 indexed timing was v9 218.342→slot3 203.897 µs versus unfused 204.365 µs. The roughly 0.23% lead over unfused was not claimed as a stable win.

**Next step.** After reducing repeated MFMA and eliminating reported private-memory allocation and VGPR spills in the relevant shapes, reassess the remaining critical path. Old spill diagnoses are no longer current. A cross-slot queue offered more theoretical reuse but regressed historical S6/on by 2.50%; logical reuse alone is insufficient.

<a id="pmc-missing"></a>

## 5. Later v13: retain gaps when matched PMC/ATT is missing

v13 had already added norm reuse, UG double tiles, and earlier down preparation. Four-round UG batching and two-tile prefetch candidates still regressed about 0.53%–1.38% and 0.58%–1.23%, respectively. LDS decreased, private-segment allocation and reported VGPR spills were zero, and resident workgroup capacity was unchanged. Resource numbers alone do not explain the regression. Neither failed candidate had matching new PMC/ATT; old v5 or A8W8 hotspots cannot establish the cause.

UG/down intervals overlapped, but down start included waiting. It did not establish useful down math overlapping UG. A discriminating follow-up would mark full-32-element-group publication/consumption and first useful down-projection computation, then collect ATT that actually covers UG/down. Indexed observer SGPR spill metadata rose 110→752, so reduce observer perturbation and verify the original binary first. Performance acceptance remains uninstrumented.

The reusable loop is to update eliminated work with matched evidence, then investigate remaining work/dependencies. A v13 resume must not propose grouping or quant-once as unimplemented.
