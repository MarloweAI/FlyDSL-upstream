---
name: mi355-mono-kernel-port
description: Build MI355/gfx950 low-concurrency mono-kernels for new models, from real layer fixtures through staged bring-up, incremental fusion, persistent scheduling, TP/EP/CP planning, and ATOM integration. Use to start or resume mono-kernel development; excludes unrelated multi-node MegaMoE throughput tuning.
---

# Bring up a new mono-kernel and drive the RSI loop

Start with a replayable real layer, then fuse incrementally. Let evidence determine the next change. Read this entry and the references needed for the current gate; do not preload every case or the PMC/ATT dataset.

## Establish four things first

1. **Target:** layer inputs, outputs, and state/cache effects; request, token, and context buckets; numerical criteria; resource and parallelism constraints; required fusion boundary.
2. **R — reference:** an independent executable specification of the math and numerical contract. Do not change it to conceal candidate errors.
3. **B — best validated implementation:** the fastest qualified path per bucket and parallel plan, plus the deployment winner under equal resource constraints. It may be native, staged, partially fused, or mono.
4. **C — construction branch:** the current functional implementation. A slower C needs a concrete downstream saving, a falsifying experiment, and an exploration limit; it does not replace B automatically.

Here **B** denotes the baseline role. In historical shape labels such as **B1/S4**, B denotes request batch size; S is tokens per request and T is the total token count. Record the actual convention, including whether S includes the current token.

Start new models at P0. For existing work, [reconcile the saved state](references/iteration-loop.md) with code and results, then find the earliest unmet gate. The first coding task is a layer fixture/reference runner, not a directory scaffold or a copied historical kernel.

## Produce a runnable result at each gate

| Gate | Implementation work | Exit criteria and reading |
|---|---|---|
| **P0: real layer and contract** | Resolve config, loader, and dispatch; implement snapshot/restore, reference execution, and stage captures; expand the actual GPU DAG | Repeatable initial state, checked outputs/effects, explicit timing and fusion boundaries. [Config discovery](references/model-config-discovery.md), [build playbook](references/build-playbook.md) |
| **P1: staged bring-up** | Build an executable staged chain; replace one device body at a time, retain its standalone wrapper, and use mature operators elsewhere | Independent stage oracle and full-layer checks pass; preserve quantization and rounding boundaries. [Numerics](../numerics/SKILL.md) |
| **P2: persistent workers and first subgraph** | Declare execution lifetime; derive workers/tasks from output ownership; define stage progress, payload/epoch, completion, and exit; run bodies independently first | Body correctness, feasible resources, residency, and progress. Use N/P/F when needed. [Tile contract](../schedule/references/tile-contract.md), [persistent design](../schedule/references/persistent-design.md), [schedule](../schedule/SKILL.md) |
| **P3: fuse one edge at a time** | Replace one fusion island while the rest stays staged; establish correct handoff before tile readiness, reuse, or communication overlap | New partition endpoints, full-layer effects, and uninstrumented replay pass; measure both island and layer. Roll back the failing edge. [Build playbook](references/build-playbook.md) |
| **P4: target parallelism and workload** | Connect real TP partials, EP dispatch/return, or CP split/merge; join remaining islands and cover target requests, contexts, and state transitions | Cross-rank semantics and liveness, required fusion scope, and target buckets have evidence. [Parallel plan](references/parallel-plan.md) |
| **P5: optimize, integrate, deliver** | Run the RSI loop; when in scope, implement real ATOM prepare/runtime/Graph/dispatch; select qualified paths per bucket | Meet functional, quality, and performance goals with reproducible artifacts. [Benchmark](../benchmark/SKILL.md), [ATOM](../atom/SKILL.md), [delivery](references/delivery.md) |

These are gates, not a one-way waterfall. If the target includes multiple ranks, requests, or MTP, represent that ownership in P0. Single-rank/S1 runs are useful diagnostics, not full-scope validation. P1 does not require rewriting every layer operator. Host launches cannot be embedded in device code.

Choose fusion edges by removable work and added resource/synchronization cost, rather than source-file boundaries. A larger island can change the generated code of existing bodies; revalidate affected claims. Partial fusion is a milestone. It does not complete a task that requires a full-layer mono-kernel.

## Run the RSI loop

**Reconcile facts → identify the earliest failed gate or current critical path → state one falsifiable hypothesis → run the smallest discriminating experiment → check numerics, liveness, and uninstrumented performance → update B/C, rejected conditions, and the next action.** Use the [iteration rules](references/iteration-loop.md) for execution and result handling.

For a failure or optimization opportunity, use the [symptom-to-case guide](../benchmark/references/optimization-priors.md) and read only the relevant section. With PMC/ATT data, or before collecting it, read the [diagnostic workflow](../benchmark/references/diagnostics.md): establish binary identity, scope, units, and coverage before making causal claims. Historical priors guide experiments; current evidence decides. Record work already eliminated so a resumed session does not propose it again.

Parallelism is an optimization axis. Start from the current unfused TP/EP/CP plan, but allow alternatives within task constraints. Rebuild ownership and communication contracts and distinguish strategy gains from fusion gains. When progress stalls, use evidence to change the body, handoff, or parallel plan instead of expanding an unsupported parameter search.

## Keep records small and delivery complete

Reuse project records when available. For a new task, optionally set `MONO_PORT_SKILL_ROOT` to the actual directory of this loaded port skill and run:

```bash
python3 "$MONO_PORT_SKILL_ROOT/scripts/bootstrap.py" --model NAME --output NEW_DIR
```

The initializer creates only `workflow-state.json` and an empty `experiments.jsonl`; it neither loads a model nor runs a GPU job. Update existing records in place. Save input/code identity and evidence using the [recording rules](references/iteration-loop.md), without a separate provenance manifest.

Use the [model contract](assets/model-contract.md), [dependency plan](assets/dependency-plan.md), and [fusion plan](assets/fusion-plan.md) as needed; the [iteration card](assets/iteration-card.md) is optional detail for complex experiments. Keep the workflow in the skill rather than copying it into every project. Once results exist, organize entry points, commands, validation matrices, performance tables, and remaining work using [delivery acceptance](references/delivery.md).

Use the [model map](references/model-map.md) to select mechanisms, the [source index](references/sources.md) for historical provenance, and the [dependency notes](references/local-dependencies.md) for sharing. Environment, budgets, and publication permissions come from the current task, not a historical session.
