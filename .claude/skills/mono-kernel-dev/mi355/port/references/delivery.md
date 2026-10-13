# Acceptance and delivery artifacts

Use this when the target scope is connected and ready for handoff. During iteration, run only checks affected by the change. Final acceptance belongs to the final source/binary, real entry point, and declared buckets; do not assemble a pass from unrelated historical versions.

## 1. Decide completion by dimension

| Dimension | Required answer and evidence |
|---|---|
| Functionality/fusion scope | Which layer families, B/S/context buckets, precisions, and TP/EP/CP plans does the real layer/serving entry support? Trace application kernels per layer/rank; list auxiliary step/reset, external operators, and communication separately. One forward or Graph replay is not proof of one full-layer kernel |
| Numerics | Which independent oracle/framework R and original criteria check endpoints and final outputs? Record errors/routes. Old-baseline equivalence, same-math staged/fused agreement, framework golden, and model quality are separate claims |
| State/liveness | Check affected cache regions and untouched bytes, independent requests, consecutive steps/MTP, epochs/replay, and rank progress. Reproduce the original failure; short smoke or an instrumented non-hang is insufficient |
| Performance | Pair with qualified B and best known unfused under equal resources, global workload, initial state, and materialization boundary. Retain trials, dispersion, aggregation, winning/fallback buckets, and fresh winner confirmation |
| Real integration | For ATOM/full-model tasks, validate dispatch, weight/cache binding, fallback, Graph lifecycle, and required generation/MTP acceptance/step/TPOT. Hot-weight single-layer timing does not establish serving gains |

For each dimension record `pass / fail / not_run / not_applicable`, scope, and evidence. Explain non-applicability against the task boundary. Avoid unrelated testing, but missing required checks prevent completion. Preserve reference instability; a controlled oracle is not the original production path.

Persistent implementations also declare lifetime, worker/task mapping, and progress/completion/exit protocols. Finite replay does not cover a service loop, which additionally needs submission/backpressure, slot reuse, stop/drain with in-flight work, and request-level timing evidence. See [persistent validation](../../schedule/references/persistent-design.md#validation).

Without a fixed performance threshold, use the user's goal and paired uncertainty; do not invent a percentage. A near-noise lead remains unconfirmed. Separate simultaneous strategy/fusion gains only with suitable controls. PMC/ATT explains mechanisms; uninstrumented timing accepts performance. Missing profiler data does not invalidate an otherwise sound functional/performance result, but limits attribution.

## 2. Make the handoff runnable by the next session

Use the project's structure; no mandatory new folders or copied templates. Provide actual relative paths for:

| Artifact | Contents |
|---|---|
| Implementation | Public call/dispatch, prepare/packing, device bodies, retained staged/fallback paths, supported scope and switches |
| Correctness entry | Runnable fixture/reference/check commands, state restoration and failure replay, current project/model config discovery/loading |
| Performance entry | Runnable benchmark command, parameters/buckets, Graph/working-set/parallel configuration, aggregation, setup versus steady-state boundary |
| State and evidence | Current R/B/C and plan identities, experiment log, final raw results, relevant compiler/build flags and source/binary identity. If used, include profiler metric configuration/scope, PC mapping, and recomputable summaries |
| Delivery note | Completion status, usage/reproduction, validation/performance tables, limitations/fallback, remaining work and next action |

Use project/model IDs, revision/hash, and paths relative to declared roots. Explain how the current project supplies external checkpoints/runtime. Do not link author-machine files or private historical repositories, or describe absent data as bundled. Large weights/raw traces may live in the task's artifact store; give an accessible locator and identify omitted inputs.

Check commands against final entry points, default dispatch against validated plans, and unfinished experiments against B. Retain useful negative results and remove misleading temporary entries or stale commands. If nothing affecting a claim changed after its last valid check, do not rerun the full suite merely for handoff.

## 3. One concise delivery note is enough

Use an existing project report, or `DELIVERY.md` if none exists. Create it when results exist, not as an empty bootstrap artifact.

```text
Status: complete / incomplete; final version and target scope
Usage: entry point, required configuration, minimal call/check/benchmark commands
Validation: scope | oracle/criteria | measured result | status | evidence
Performance: bucket/plan | baseline | candidate | change/dispersion | accept/fallback | evidence
Files: implementation, checks, benchmark, current state, experiments, raw results
Limitations: untested scope, known failures, fallback, next action/exit criterion
```

If functionality, quality, required fusion scope, or performance remains unmet, provide an explicit incomplete handoff with usable B, C's progress, and the quantified gap. Complete files or a local speedup do not establish task completion.
