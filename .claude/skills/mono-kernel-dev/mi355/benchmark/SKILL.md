---
name: mi355-mono-kernel-benchmark
description: Drive evidence-based MI355 mono-kernel optimization using resident/staged/fused ablations, TP/EP/CP comparisons, paired Graph benchmarks, and ISA/PMC/ATT diagnostics. Use to select the next optimization, explain failed fusion, or validate workload and deployment gains.
---

# Drive performance iteration with discriminating experiments

Start from current R/B/C, bucket/parallel plan, partition, and recent valid results. Return accept/reject/inconclusive, scope, eliminated work, and the next experiment. Resolve incorrect or hanging candidates at the earliest failed [RSI gate](../port/references/iteration-loop.md).

## 1. Select one hypothesis from the symptom

Inspect the uninstrumented full-layer completion tail. The largest accumulated stage duration need not lie on the critical path. Use the [symptom-to-case guide](references/optimization-priors.md) to write an observation, mechanism, smallest change, and falsifying outcome. Cases guide experiments, not default tile sizes or thresholds.

Rank opportunities by potential impact on target buckets, evidence, and implementation/validation cost. Use an ablation when the mechanism is unclear. For bounds and search ranges, use [experiment design](references/experiment-design.md): hold numerical/synchronization invariants fixed and bound compilation, code size, and GPU cost.

Bodies, fusion boundaries, and TP/EP/CP are all candidate axes. The [parallel plan](../port/references/parallel-plan.md) separates fusion at a fixed plan, strategy at a fixed implementation organization, and combined deployment gains. Compare against the best known unfused deployment under equal resources/workload; narrow attribution when controls are missing.

## 2. Build a valid comparison

- **N/P/F:** hold math, inputs, and parallel plan fixed. N uses ordinary separate launches; P uses resident configurations with separate launches; F fuses them. Differences measure execution organization, not an isolated fence or poll cost.
- **Timing boundary:** match initial input/cache state, output materialization, warmup, layer/step calls per Graph, and replay. Exclude compilation/packing from steady state. Distinguish repeated hot weights, independent-address copies, and a real layer stack.
- **Pair quality:** retain raw trials, A/B order, dispersion, exclusions, and environment. Serialize comparable jobs on the same devices. A worktree does not isolate a GPU, and a brief idle sample does not establish exclusive use. Exclude contaminated results from ranking.
- **Cross-rank timing:** measure the actual request/layer completion path with a valid protocol; retain rank/trial records and participation. Do not relabel rank-median results as slowest-rank timing or hide EP/CP tails with an average.

Check contracts, compilation/resources, endpoints/full-layer numerics, and affected state/liveness before short timing screens. Confirm winners with fresh pairs and affected buckets. A combination of optimizations is a new candidate. Reuse still-valid evidence instead of repeating checks without a reason.

Report `speedup=t_base/t_candidate` separately from `latency_reduction=1-t_candidate/t_base`. Use task/reference criteria; without an explicit performance threshold, judge paired uncertainty instead of inventing a percentage.

## 3. Use PMC/ATT for mechanisms and event timing for gains

When explanations compete or profiler data already exists, use the [diagnostic workflow](references/diagnostics.md). Match each arm's binary, scope, units, and coverage, then predict work/dependency changes. PMC needs raw instances, denominators, normalization, and task predictions. ATT needs target-stage coverage, PC/inline-call mapping, hits, stalls, and cycles per hit.

Read the relevant [PMC case](references/pmc-cases.md) or [ATT case](references/att-cases.md). Load the [numerical extract](references/data/pmc-att-evidence.json) only for recomputation. Missing is not zero; sampled stalls are not layer time. An old binary or uncovered stage cannot establish a current hotspot. With a changed parallel plan, rule out work relocation before interpreting a rank-local counter decrease as eliminated work.

Valid paired measurements can establish functionality/performance without profiler data. Keep the mechanism limited or unknown when its evidence is missing.

A persistent service publishes request completion before its kernel exits. Use the [persistent measurement boundary](../schedule/references/persistent-design.md#validation) for timing, PMC denominators, and active-stage ATT coverage; finite layer-launch Graph/event conventions do not automatically apply.

## 4. Decide and deliver

Use [RSI result handling](../port/references/iteration-loop.md): accept and update scoped B when required numerical/state/liveness gates pass and fresh uninstrumented pairs confirm reliable improvement at the target completion boundary. Record `mechanism_status` separately as `supported / falsified / unknown`; an unknown or falsified original hypothesis does not block an otherwise qualified performance win. A performance win alone does not prove the proposed mechanism. If a local win does not reach the layer, inspect the critical path/new costs. Keep noisy performance results inconclusive.

Record source/binary, command, fixture/environment, raw results, supported and unresolved explanations, falsification conditions, and `next_action`. Update B for the qualified plan/bucket or advance only functional C. Record only evidenced eliminated work, leave unexplained causes open, and re-identify the remaining critical path. Use [delivery acceptance](../port/references/delivery.md) for the final functional/performance matrix and reproduction artifacts.
