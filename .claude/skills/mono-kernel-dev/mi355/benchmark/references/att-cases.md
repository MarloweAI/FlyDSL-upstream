# From ATT hotspots to changes, ablations, and confirmation

Sources are the 2026-10-08 GLM working-set analysis, shared-early/A4W4 layout experiments, and a later TP4 sampling review. Keep these precision, working-set, and binary scopes separate. The [extract](data/pmc-att-evidence.json) contains trimmed aggregates and source hashes, not large raw ATT traces or model weights.

<a id="att-workset"></a>

## 1. Working-set growth: locate the additional stalls

**Control.** TP8, H8/S4, one request. One/eight means independent addresses holding identical weights, not eight concurrent requests or eight real layers. Inputs, weights, and algorithm stayed fixed while the address working set increased. Matched unprofiled FlyDSL latency was 57.166→62.102 µs; TileRT was 54.470→54.612 µs. The problem appeared in a single-layer test.

**Sampling.** GPU0, CU1/CU3, four SE streams each, selected SIMD, three seeds. Each region had 16 launches; dropping the first eight left eight launches and 256 sampled waves. Initial capture disturbance and the first eight-address rotation were excluded consistently. There were 12 pairs/24 regions; fixed native→Fly order was a limitation of the ATT capture. Empty-wave CU9 exploration was excluded.

Per-wave Hit/Latency/Stall accounting matched the decoder and `num_stitched == num_insts`. Columns were `ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle`; the last column was not Stall. Below are stall sums for 256 waves/region, averaged over seeds/CUs, in Mcycles:

| FlyDSL category | One address | Eight addresses | Increase |
|---|---:|---:|---:|
| VMEM-load | 5.150 | 7.373 | +2.223 |
| VMEM-wait | 7.547 | 7.933 | +0.387 |
| LDS/SMEM-wait | 1.533 | 1.757 | +0.224 |
| Mixed VMEM+LDS/SMEM-wait | 0.114 | 0.152 | +0.039 |
| Barrier | 8.995 | 9.082 | +0.088 |
| MFMA | 0.037 | 0.031 | −0.006 |
| All categories | 25.237 | 28.414 | +3.177 |

VMEM loads accounted for about 70% of the sampled-stall increase; barrier growth was small. Investigate load/dependency behavior first. This is not 70% of the layer-latency difference. Large barrier totals can mean early waves waiting for late ones, not 8.995M cycles recoverable by deleting barriers.

**Control for hits.** UG weight loads had 15,360 hits/region; cycles/hit rose 77.52→101.49 (+30.9%), increasing in all six CU×seed pairs. Inline-callsite breakdown:

| Load callsite | Hits/region | One address cycles/hit | Eight addresses cycles/hit |
|---|---:|---:|---:|
| Shared weight | 3,072 | 54.00 | 64.57 |
| First routed weight | 3,072 | 81.04 | 125.77 |
| Next sample weight | 9,216 | 84.18 | 105.70 |
| Shared scale | 1,536 | 420.94 | 546.99 |
| First routed scale | 1,536 | 82.18 | 120.70 |

More stalls at fixed hits supports more waiting per sampled execution, not simply more executions. VMEM-load stalls are not direct HBM latency. No matching L2/HBM PMC existed in that round, so L2 misses or a full VMEM queue remained unproven.

**PC → source → dependency.** With debug `.text` matching the captured binary, `0x92a4 buffer_load_dword` mapped to shared UG scales after the post-route barrier; `0x9380 buffer_load_dwordx4` to first routed weights; `0x9698 s_waitcnt vmcnt(3)` to UG input staging. These PCs identify that historical binary only.

Shared addresses were known before routing but loaded afterward; routed addresses depended on selection. The next experiment therefore moved the complete shared weight/scale set earlier while preserving math/resources. It did not apply NT hints to every load. Cache/DRAM/backpressure evidence would be needed for stronger memory-system attribution; missing data is not zero.

<a id="att-prefetch"></a>

## 2. Shared early: one hotspot improves while another worsens

**Change/ISA.** Only S>1 shared weight/scale prefetch moved before `dn_route`. Eighteen loads moved to `0x876c–0x8808`; the new `dn_route` started at `0x8810`. VGPR 220, LDS 95,024 bytes, private 0, and 8,545 static instructions were unchanged. Machine code established placement.

**Follow-up sampling.** Eight addresses, GPU0/CU1, four SEs, three seeds, 256 retained waves/region. Ratios use `sum(stall)/sum(hits)` for the same callsite groups; hits were fixed across seeds:

| Group | Hits/region | Control cycles/hit | Early cycles/hit | Change |
|---|---:|---:|---:|---:|
| Shared scale | 1,536 | 532.57 | 284.78 | −46.53% |
| Shared weight | 3,072 | 55.83 | 58.25 | +4.34% |
| Routed weight | 12,288 | 101.26 | 155.56 | +53.63% |

Total sampled stalls fell 28.314→27.575 Mcycles; barriers fell 10.576→9.773 Mcycles. Shared-scale stalls dropped while routed-weight stalls rose. Scheduling redistributed stall attribution; it did not make all accesses faster or prove a scale-bandwidth bottleneck was solved.

**Discriminating ablations.** Scale-only prefetch gave 62.239→62.260 µs, no gain. Moving the full shared set worked. Batched down consumption raised VGPR to 252 and regressed; deleting the preparation barrier produced no stable gain. A visible hotspot alone did not prescribe a successful change.

**Acceptance.** Uninstrumented confirmation used three rounds ×three seeds, reversed comparison order, 256 launches/Graph, and maximum-rank event median. One address: 57.245→55.887 µs (−2.37%); eight addresses: 62.158→61.520 µs (−1.03%). Original bitwise outputs and routing matched. ATT explained the mechanism; events accepted the result.

**Next step.** Keep the scoped gain, then inspect routed layouts/reuse/waits because the eight-address gap remains. Higher routed cycles/hit does not reject a reliable layer win, and the small win does not solve the full gap. This was A8W8, not validated A8W4 transfer.

<a id="att-layout"></a>

## 3. A4W4 layout: changed hits require normalization

A separate experiment started from FP4 access patterns. Scale rows were 192 bytes apart, scattering a tile's 16×4 scale bytes. Packing by 16 rows ×128 K and pairing 8 gate + 8 up changed the initializer and device loader together while preserving quantization/MFMA math. Screening also compared down batch 9/18 and prefetch distance 1/2.

Scope: GPU0/CU1, four SEs, eight addresses, S4, 256 retained waves, seed 42; debug and executed `.text` matched. Per-hit values are derived from raw hits/stalls:

| Group | Original hits / stall cycles | Final hits / stall cycles | Original→final cycles/hit |
|---|---:|---:|---:|
| UG scale | 7,680 / 815,856 | 7,680 / 461,540 | 106.23→60.10 |
| UG weight | 7,680 / 1,288,324 | 7,680 / 801,168 | 167.75→104.32 |
| Down scale | 4,352 / 278,280 | 4,608 / 161,480 | 63.94→35.04 |
| Down weight | 4,608 / 485,976 | 4,608 / 474,100 | 105.46→102.89 |

Down-scale hits changed. Compare `1-(161480/4608)/(278280/4352)≈45.2%`, not only total-stall reduction. Changed hits still indicate execution/path differences; normalization does not remove every confounder.

**Why layout was selected.** Eight-address screening gave control_start 57.244 and control_end 57.195 µs; scale_tile 54.119, scale+paired 53.678, down18 alone 57.308, paired+down18 53.459. Deeper prefetch did not win. Layout was the main direction, with down18 a later increment on that layout. Ordering/drift prevents treating these differences as strictly additive contributions.

**Retain opposing evidence.** Scale+paired with batch 9 increased barrier stalls 8,510,208→9,296,636 cycles; final batch 18 used 7,048,456. Final down-weight cycles/hit improved only 2.4%. Do not explain the whole gain using the largest scale percentage. Native control total stalls also drifted about 2.3%.

Independent confirmation used two rounds ×three seeds, 256 launches/Graph, slowest rank then mean across cases. Eight-address A4W4 latency improved 57.225→53.566 µs (−6.39%); old/new output SHA matched. VGPR rose 162→194 with unchanged LDS and zero private/spill. More registers can still win. The PMC pilot had permission/perturbation limits and did not validate HBM/L2 traffic improvements.

**Next step.** Re-derive co-consumed weight/scale units for the current path, then compare actual loads and same-math layer timing. The later A8W4 BF16-MFMA port was another experiment; these ATT numbers do not establish its hotspots or those of native scaled MFMA. See [packer implementation](../../schedule/references/layout-prefetch-case.md).

<a id="att-coverage"></a>

## 4. Stop conclusions at the sampled coverage boundary

Later TP4 v5 narrow ATT passed instruction reconstruction, but GPU0/CU3 with limited SE/SIMD mainly covered indexer/attention. Each region had 32 waves and retained 16 after skipping the first 8 launches, far below 256 waves/region above. Raw stall sums across those designs are not comparable.

The session explicitly limited old ATT to captured barrier/VMEM waits. It could not establish down as the largest layer bottleneck or be relabeled v8/v13 ATT. Investigating UG/down requires current role/task→CTA→wave coverage, matching debug line tables, and exact `.text`. Without valid samples, use PMC/completion points for a limited hypothesis and retain the ATT gap.

On resume, first identify binary, stage, and valid-wave count. If only old uncovered samples exist, collect targeted evidence instead of repeatedly recommending MoE changes from an old top-PC list.
