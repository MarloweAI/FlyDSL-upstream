---
name: mi355-mono-kernel-numerics
description: Preserve and validate MI355 mono-kernel numerical contracts across FP8, MXFP4, native scaled MFMA, routing, recurrent state, and cache updates. Use for golden mismatches, quantization ports, expert alignment, or stateful decode/MTP correctness.
---

# Establish executable correctness gates for each replacement

Inputs are the changed stage/fusion edge, R, candidate C, and a fixture with identical initial state. Return the first divergence and a replayable failure, or evidence sufficient for the next gate.

## 1. Capture stage oracles before development

Use [config discovery](../port/references/model-config-discovery.md) to identify the current config, quantization loader, and selected math path. Capture real stage inputs/outputs, routes, and initial cache/state at the layer boundary. Pin the reference revision and existing criteria. Record each GEMM's A/W/scale/MFMA/accumulator and output rounding. If the reference does not model the target quantization, build the required independent oracle before using it to judge the kernel.

Run `run_reference(snapshot)` and `run_candidate(restore(snapshot))` from separate copies of the same initial state. Reference cache writes must not become candidate inputs. Repeat a potentially unstable reference on identical inputs and retain failures.

Direct cases: [numerical paths](references/contracts.md#numerical-paths), [three routing mismatch classes](references/contracts.md#routing), [reference instability](references/contracts.md#unstable-reference), and [state](references/contracts.md#state). The same A8W4 label does not imply identical native scaled-MFMA and BF16-MFMA arithmetic.

## 2. Choose the first check from the actual change

| Change | Check first | Then check |
|---|---|---|
| Host packer/device view | Pack/unpack and actual weight/scale addresses | Independent stage oracle and full layer |
| GEMM/activation replacement | Accumulation, rounding, and quantization endpoints on identical inputs | Consumer output and full layer |
| Producer/consumer fusion | Every staged/fused boundary tensor | Poisoned scratch, changed inputs/epochs, state, long replay |
| Tasks/synchronization/queues | Per-task write coverage, tags, capacity, progress | Outputs/intermediates, delayed producer/rank, uninstrumented stress |
| Expert grouping/token columns | Original slots, masks, shared/zero columns, per-token intermediates | Identical/disjoint/partly overlapping routes, odd rows and tails |
| Reduction/atomics/TP order | Independent quantization oracle and partial/reduced values | Full layer under original criteria; neither require old bitwise order blindly nor relax tolerances arbitrarily |
| EP or TP/EP ownership | Global/local expert IDs, original token/slot, dispatch/return, route-weight placement | Skew, empty ranks, capacity boundaries, shared contribution exactly once, final token output |
| Context/KV sharding or CP merge | Complete scores/global top-k or valid attention partial merge; cache ownership | Empty/all-masked shards, page boundaries, multiple requests, original criteria and state |
| New B/S/MTP/cache ABI | Independent request predecessors, page tables, slots, untouched regions | Consecutive steps, accept/reject, rebind, Graph lifetime |

Early fusion can retain explicit rounding even when an intermediate no longer reaches global memory. Fusion does not authorize a precision change. For tile, mask, or layout changes, use the [tile contract](../schedule/references/tile-contract.md): masked loads/stores alone do not define padding values, reduction identities, or collective participation.

When parallelism changes, rebuild distribution/reconstruction using the [parallel contract](../port/references/parallel-plan.md); R's mathematical standard stays fixed. Do not compare rank-local partials that no longer represent the same logical work, or average CP-local normalized softmax outputs.

## 3. Bisect at the first divergence

1. Compare input/initial-state and actual weight/scale identity. Resolve fixture or dispatch mismatches first.
2. Compare stages and replay the first divergent endpoint with fixed inputs. Preserve raw slot data; expert-ID alignment is diagnostic and does not waive required ordering.
3. Fix logits/bias to isolate routing. Separate near scores, exact ties, permutations, and probabilities. Truncated near scores are not a legal tie difference.
4. Locate the first numerical boundary: accumulation → activation → rounding → quantization → route weight → TP/residual. For hangs, locate an unpublished task, out-of-bounds tag, or LDS lifetime error and return it to scheduling.
5. Fix one located cause, replay the failure, then return to full-layer checks. Old performance does not qualify a new numerical/layout version.

GLM's LDS race was masked by short replay and instrumentation. A synchronization fix must survive the original uninstrumented long-replay failure; one non-hanging run does not establish the mechanism is fixed.

## 4. Keep four conclusions separate

- Equivalence to the previous baseline.
- Staged/fused agreement under the same device math.
- Independent oracle/framework golden acceptance.
- Full-model generation and MTP commit/rollback acceptance, when in scope.

For each, record `pass / fail / not_run / not_applicable`, scope, and evidence. Explain why an item is not applicable. If the baseline itself fails, “no additional difference” is valid; “golden passed” is not. Candidates missing required correctness gates may retain diagnostic timings but cannot become B.

## 5. Return actionable evidence

Return the changed scope, reference/candidate identities, check matrix, first failing endpoint and reproduction command, and the body/protocol to fix next. On success, state which fusion or performance step is now justified. Link evidence rather than writing only PASS.

Place final numerical/state results in the [delivery matrix](../port/references/delivery.md), bound to the final version, buckets, original criteria, and untested scope. Performance acceptance remains separate.

State/cache matters as much as hidden output: per-token KDA snapshots, HCState, compressor ring/commit, logical values at written positions, and untouched bytes belong to their ABI's acceptance criteria. Preserve the definitions of NRMSE, relative L2, bitwise equality, and top-k overlap; they are not interchangeable.
