---
name: mi355-mono-kernel-atom
description: Integrate MI355 mono-kernels into ATOM decode/MTP with checkpoint weight packing, paged caches, Graph epochs, state lifecycle, and guarded fallback. Use when a validated mono prototype must become a real ATOM layer/model path or retain its gains in serving.
---

# Integrate a validated partition into the real ATOM lifecycle

Use this skill twice: establish interfaces early, then complete integration. Discover cache ABI constraints before the kernel is finished; early fixtures do not require changing production defaults.

## I0 — Establish integration seams during P0

Use [config discovery](../port/references/model-config-discovery.md) to locate current model config, loader, and guards. Follow the real Block/ModelRunner forward through weight preparation, metadata construction, cache updates, Graph capture, step/MTP commit, and process-group teardown. Specify the objects the candidate consumes and returns. Expand HCState, KDA snapshots, or IndexShare only when the model has them.

Build a layer fixture/adapter that borrows loaded Parameters and real metadata. Replace one island while retaining native/staged code elsewhere. Restore identical initial state and check the full layer after every replacement. List which stages execute inside mono and which remain external; one host forward does not define device fusion scope.

**Exit:** prototype work runs against real interfaces, and shape guards execute before any side effect. See the model-specific [integration matrix](references/integration.md).

## I1 — Prepare weights, runtime, and specialization dispatch

Move packing, scratch allocation, peer mapping, epoch setup, and JIT preparation into one-time setup. Verify the selected layout/compute path with actual packed tensors and traces. For borrowed weights, record when shuffle completes; for conversions, record memory use and when old copies can be released.

Compile/Graph keys must cover shape, dtype/layout, TP/EP/CP plan, index refresh, and cache binding where they affect generated code or pointer interpretation. Runtime bindings also include concrete groups/members, expert/page mappings, and communication buffers. Define scratch and stream concurrency for each new bucket. Rebuild bindings or reject stale runtimes after cache/weight reallocation.

**Exit:** steady state does not repack layer weights/full caches, JIT, or allocate new required buffers. Single-layer replay supports changing inputs with valid pointer lifetimes.

For a cross-invocation service, implement the [submission/completion/stop protocol](../schedule/references/persistent-design.md#service-loop). It has its own state/visibility contract. Do not submit required inputs behind a waiting persistent kernel on the same stream.

## I2 — Integrate real decode/MTP and validate state first

Check applicable model contracts: GLM/M3 per-request positions/pages/causal bounds and single cache writes; K3 predecessors, S+1 snapshots, acceptance/rollback; DSV4 delayed HCState, compressor ring/commit, and indexer calls inside attention; TP epoch/slots and close order.

Unsupported ordinary-forward inputs use native fallback; an explicit mono entry reports a concrete error. Never fall back after modifying cache. Check dummy/prefill/padding/microbatch and unsupported shapes for absence of side effects. Restore full cache/state for both arms, not just hidden inputs.

**Exit:** same-state layer execution, consecutive decode, long Graph replay, real independent requests, and in-scope MTP commit/rollback meet their original criteria. Untested behavior is not supported behavior.

## I3 — Explain the prototype-to-framework gap

Use the same checkpoint, inputs, and output-materialization boundary for paired layer timing and device traces. Verify actual launches, auxiliary step/reset work, layout gathers, host/device copies, cache conversions, collectives, and fallback.

If the island improves without a layer gain, inspect external boundaries. If the layer improves without a serving gain, inspect the real layer-stack working set, Graph organization, unfused layer fraction, MTP acceptance, and serving scheduler.

Keep algorithm/precision differences explicit. If the old framework returns prefix/routed/shared pieces, include final materialization in the full boundary. Count a multi-launch DSV4 full-layer forward separately from a single-launch MoE.

**Exit:** gains hold through the real ATOM call, or a measured gap identifies the next experiment. A faster prototype alone does not qualify a production policy.

## I4 — Enable qualified buckets and complete in-scope E2E work

Dispatch according to correctness and performance coverage; retain faster staged/native buckets. A new shape or template branch can affect old buckets' generated code/resources, even if their Python branch is unchanged. Check affected numerics and timing.

If mono uses another TP/EP/CP plan, apply the [deployment comparison and switching contract](../port/references/parallel-plan.md). Include inter-layer redistribution and real state migration. Switching plans may rebuild weights/KV/groups/Graphs; it is not a free per-bucket kernel dispatch. Fallback must match the current layout and state.

For full-model tasks, complete multi-layer generation, MTP acceptance, and same-workload step latency/TPOT. Before teardown, wait for Graph consumers and in-flight communication, close IPC according to its protocol, then destroy related process groups. Record support, defaults, fallback, and reproduction commands.

Deliver actual integration code and results. If required fusion/performance goals remain unmet, return the gap and smallest next step to the [main loop](../port/references/iteration-loop.md). Publication follows current authorization, not historical push permissions.

Use [delivery acceptance](../port/references/delivery.md) to organize real calls/configuration, supported/fallback scope, functional/performance matrices, and commands for the final version. Prototype single-layer timing does not complete a task that requires serving/E2E evidence.
