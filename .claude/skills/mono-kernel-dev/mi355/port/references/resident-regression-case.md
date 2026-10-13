# K3 expert pair: why one launch became slower

Sources K4/K5 and S1. Historical experiment: 2026-09-28, MI355X/gfx950, correctness baseline `cb147e89`. This page preserves the reasoning needed for another iteration without the original machine, report, session, or source checkout. Values are historical observations, not new-model defaults or new measurements.

Evidence consists of N/P/F timing, code objects, and runtime occupancy. No PMC/ATT was invented for this case. Learn counter/trace reasoning from the separate [PMC](../../benchmark/references/pmc-cases.md) and [ATT](../../benchmark/references/att-cases.md) cases without mixing experiment identities.

## Starting point and controls

The working A16W4 routed UG/down path retained SiTU, BF16 intermediate, BF16 atomic scatter, and external router/tail. The first pair placed two general 256-thread GEMM bodies in a resident task loop, removing one launch without implementing output-owned combine/TP.

N used the efficient ordinary GEMMs; P used resident task/resource configurations in separate launches; F used one launch for the same target math. F compared tile-ready handoff with one grid barrier. Each experiment had its own paired baseline; cross-table differences are not individual instruction costs.

Historical single-GPU protocol: no profiler, Graph with 16 expert sections and common clear/step work, 100 replays, alternating order. Representative seed 1234 values in µs/expert section:

| Execution | S4 | S8 |
|---|---:|---:|
| N: ordinary staged | About 32.5 | About 51.3 |
| P: separate resident stages | 50.7 | 79.3 |
| F: tile-ready, 256 CTAs | 102.0 | 162.0 |
| F: grid barrier, 256 CTAs | 86.5 | 122.6 |
| F: grid barrier, 512 CTAs | 107.1 | 150.2 |

Multiple seeds and same-expert-set inputs also regressed, as did TP8 KDA/MLA full-layer comparisons. Fixed-routing intermediate/staged checks, an independent dequantization reference, and changing-input Graph replay had passing evidence. Independent full-layer quantization/routing golden issues remained unresolved. This established scoped functional C, not arbitrary-checkpoint full-layer correctness.

## What the evidence supports

P already regressed before fusion, so investigate resident bodies, tasks, and shared resource constraints first. F regressed further, requiring joint-code/resource and handoff investigation. Differences include code generation and scheduling; they are not pure fence time.

Historical code objects and HIP occupancy:

| Kernel | VGPR: S4 / S8 | Maximum blocks/CU: S4 / S8 |
|---|---:|---:|
| Staged UG | 152 / 158 | 3 / 3 |
| Staged down | 72 / 72 | 7 / 7 |
| Pair | 190 / 232 | 2 / 2 |

The pair lost standalone down's resource/concurrency advantage. Grid 512 was slower despite passing that experiment's co-residency check. Uniform task indices, first-iteration unrolling, and `waves_per_eu=4` did not repair the structural loss; the latter also spilled to scratch. More CTAs or a few fewer VGPRs were no longer the highest-value next experiment.

This does not establish that tile readiness is always slower than a grid barrier. The prototype retained ordinary BF16 payloads and added ready/poll/acquire operations. A tagged loader that directly retrieves data is a different path.

## Next implementation sequence

1. **Redesign down ownership independently.** Work backward from a final output tile to all required route/K contributions. Combine locally in the CTA while retaining input, SiTU, and intermediate formats. Compare with scatter+combine at the same output boundary. Removing BF16 atomics changes summation/rounding organization; check partials, independent oracle, and layer criteria rather than requiring old bitwise order blindly or relaxing tolerance.
2. **Redesign intermediate handoff.** Publish payload+generation in the producer epilogue and validate/retrieve fragments in the consumer loader. Specify atomicity, scope, reuse, and progress. A flag wrapped around an old GEMM does not create this data path.
3. **Retune the resident body.** Select tile/wave/K/prefetch batches against maximum task rounds, real occupancy, and joint LDS/VGPR constraints. Measure standalone bodies before fusion; historical sizes are not defaults.
4. **Connect output to TP/tail.** Begin legal tile communication when ready, preserving latent RMSNorm and shared/latent dependencies. Prove peer waits cannot starve unclaimed outputs, then validate at shared MoE/full-layer boundaries.

Each step is falsifiable: slow standalone owner-down → tile/reduction resources; reasonable standalone but slow fused → loader/publication/joint compilation; local win but no layer win → latent norm, tail, or interface. Retry old parameter directions only when their premises change.

## State returned to the development loop

Retain staged B. The pair remains a diagnostic/functional C only within passed checks, with performance status `rejected_perf`; unresolved golden items stay open. A further structural branch needs named removable atomic/handoff/TP costs, a minimal falsification test, and an exploration limit.

This case applies when directly making efficient stages resident produces a large loss. It neither rules out mono nor mandates output-owned down. Rebuild [N/P/F controls](iteration-loop.md) for different arithmetic, task domains, or atomic requirements.
