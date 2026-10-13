# GLM layout and early loads: change the actual access path

Sources G1 and K2: historical reports/code review. This page preserves implementation, conditions, and counterexamples. Gains belong to the stated precision, working set, and measurement boundary.

Read [ATT cases](../../benchmark/references/att-cases.md) for fixed-hit working-set analysis, PC/inline mapping, opposing stall changes after shared prefetch, A4W4 normalization, and deciding ablations. Keep the distinct numerical paths separate.

## Case 1: co-locate weights and scales consumed together

A historical GLM A8W4 path used BF16 MFMA: GEMM1 FP8×MXFP4, GEMM2 BF16×MXFP4. Related scales/weights were scattered, and the old ATOM layout could add elementwise gathers.

Pack E8M0 scales by `16 rows × 128 K`, making each tile's `16×4` scale bytes contiguous. Put 8 gate and 8 up rows in the same 16-row tile. Change host packing and device readers together, only during initialization; preserve runtime activation quantization, MFMA, and rounding.

Apply the mechanism to a new project:

1. Derive co-consumed weight/scale tiles from current MFMA/loader lane addresses; locate gathers/discontinuity.
2. Specify logical-index→packed-offset mappings on both sides, including gate/up rows and quantization groups. Check real-data pack/unpack or address sets.
3. Change the initializer and reader together, retain numerical endpoints, and verify the selected branch receives the new tensors.
4. Compare generated loads/address arithmetic, VGPR/LDS, and paired full-layer timing. Investigate integration if runtime repacking/gathers remain.

Historical TP8/B1/no-indexer hot-weight Graph timing for S1/2/4/8 changed from 38.023/45.042/57.606/88.011 µs to 37.116/43.117/54.847/84.512 µs, using slowest rank then trial median. This establishes that BF16-MFMA path's paired improvement, not a promise for another model.

Switching the old adapter also removed extra gathers; do not attribute that entire gain to tile packing. Later native scaled MFMA used another numerical/layout path and needs its own co-consumption mapping.

## Case 2: known shared addresses do not guarantee overlap

Shared expert IDs are known before routing. Historical A8W8 `shared_early` moved the full shared weight/scale load set ahead of route waiting. ISA showed 18 loads moved, VGPR/LDS were unchanged, and outputs were bitwise identical. Reversed-order pairs showed a small gain. A scale-only ablation did not help; A8W4 transfer was not established.

Ask which addresses are already known and move only legal loads:

```text
Original: route wait → shared weight/scale load → compute
Candidate: shared weight/scale load → independent route work/wait → compute
```

Verify placement and useful overlap separately. K3's first-down load moved in ISA but a later input poll's `vmcnt(0)` also waited for the prefetch, removing overlap. The next experiment concerns loader/wait domains and resources, not automatically greater depth. Preserve correctness-required waits.

Do not add another prefetch to a bucket already doing it. If values live across more stages, check occupancy/resources and full-layer tail. Source order is not evidence of machine execution order.

## Negative results and retry conditions

Historical GLM regressed after directly copying cache `nt` hints. A VGPR 220→158 candidate remained slower while both configurations occupied the same residency tier. Some batched down consumers increased VGPR and regressed. These results favor actual load/working-set/critical-path evidence over hints or register-count rankings.

Retry when layout, numerical path, working set, compiler, or wait organization changes the premise. If local load improvements do not reach ATOM, inspect packed tensors, dispatch, Graph, and output scope through the [integration matrix](../../atom/references/integration.md).
