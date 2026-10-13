# Persistent execution: workers, progress, and completion

Read when first placing staged bodies in a persistent executor at P2, or changing queues/lifetime. Produce an implementable worker/task/communication/exit design, then compile and validate through the [build playbook](../../port/references/build-playbook.md). Service-loop guidance here is design reasoning, not a new cross-step persistent-kernel measurement. Historical [K3 resident](../../port/references/resident-regression-case.md) and [DSV4 epoch](staged-epoch-case.md) results do not validate a service loop.

<a id="mode-scope"></a>

## 1. Declare how long workers persist

| Mode | CTA/invocation lifetime | Required proof |
|---|---|---|
| Finite per-invocation workers | A bounded grid processes multiple tasks/stages within one layer/island call, then returns | Task coverage, dependencies/progress, completion, epochs; usually the first implementation |
| Cross-invocation service loop | The kernel stays active, accepts later layer/step/request descriptors, publishes completion, and exits on a stop protocol | Submission, visibility/wakeup, backpressure, invocation isolation, drain/stop, host lifecycle |
| Ordinary Graph replay | Each application kernel exits at its own boundary; Graph execution launches subsequent calls | Capture/pointers/epochs/replay; reusing an arena or Graph does not make a service loop |

Resident describes resource residency; persistent describes worker reuse and lifetime. One launch, simultaneous CTA residency, and remaining active across decode steps are distinct properties. Record `execution_mode`, completion unit, and scope rather than only `persistent=True`. Persistent workers do not imply all weights remain in registers, LDS, or cache.

<a id="worker-plan"></a>

## 2. Derive tasks before choosing worker count

Expand the DAG into `generation / stage / request-token / expert or context shard / output tile / inputs / output owner`. Derive task counts/masks from actual body tiles and distinguish capacity from live work. Routing/EP-generated counts must be published for the current generation; capacity is not the completion target. Use the [tile contract](tile-contract.md) for physical mapping and lifetime checks.

Choose grid G, CTA geometry, and roles using the current binary's VGPR/LDS/private allocation, HIP occupancy, and actual XCD/CU/partition configuration. Include producer/consumer/communication roles and maximum rounds per worker. Loops cover tasks beyond G; neither one CTA per task nor one CTA per CU is a universal optimum.

If workers block on other CTAs or use a grid barrier, prove that all required participants can execute. `G <= aggregate capacity` is a clue, not proof of actual placement, co-residency, or progress. Check external communication-kernel resources too. A nonblocking task design may use another progress argument; barrier assumptions do not apply automatically.

Task selection must be uniform across the waves participating in a CTA body: for example, a leader claims and broadcasts a task, then all required waves execute. Dynamic scheduling does not permit arbitrary wave-local stage selection around workgroup barriers. Audit fragments, LDS partitions, barrier scope, and live ranges.

<a id="first-loop"></a>

## 3. Start with a static stage-major loop

Retain a standalone body for every stage. A simple auditable assignment gives worker w tasks `w, w+G, w+2G, ... < live_tasks(stage)`. With roles, use that stage's worker set and role-local index/stride.

This is control-flow pseudocode. Emitters, publication, and handoff must map to real project primitives; it is neither an API nor a liveness proof:

```text
host/runtime: prepare generation g, inputs, and shared control state in legal stream/event order
launch finite_worker_grid(g):
    worker = CTA_ID
    for stage in topological_order(current_partition):
        for task in assigned_tasks(stage, worker):
            emit_stage_body(stage, task)       # Follow the body's CTA cooperation contract.
            finish_all_payload_writers(task)
            publish_payload_for_generation(task, g)
        phase_handoff(stage, g)               # Only where the edge requires it.
    finish_this_worker_protocol(g)
    return
```

A phase barrier is acceptable only with established grid co-residency and synchronization semantics; all participants, including empty-task CTAs, must arrive. Otherwise retain separate launches or use producer/consumer roles with a different progress proof. Do not introduce an unproven grid spin barrier.

Initialize counters/queues before consumers can access them. One CTA must not reset state already in use by others. A finished worker is not a finished invocation. Finite kernel completion/event means all CTAs returned; cross-rank/asynchronous transfers still need their own completion protocol. Define generation owner, commit point, and slot reuse separately; see the [epoch case](staged-epoch-case.md).

Validate coverage, same-math endpoints, and finite completion before N/P/F. If P already regresses, fix the body/resources/task rounds; a longer lifetime does not automatically recover the loss.

<a id="ready-scheduler"></a>

## 4. Schedule ready work without blocking every worker

Introduce finer scheduling only with evidence of waiting or imbalance:

| Policy | When useful | Cost and progress requirement |
|---|---|---|
| Fixed producer/consumer roles | Clear dependencies; first correct overlap | Producers finish required production before waiting; idle consumers cannot consume every production resource |
| Dynamic claims/chunked claims within a stage | Uneven task cost or routes | Better tail balance costs atomics, scheduling/indexing, and locality; claiming is not completion |
| Cross-stage ready queue | Tile-ready consumers can shorten the critical path | Implement readiness publication, unique claims, role/resource compatibility, and termination detection |

Tasks need at least generation-local `WAITING → READY → CLAIMED → DONE` states. After producers complete and publish payloads, aggregate predecessor completion. Only an owner that has established visibility of all required payloads may enqueue a consumer once. An atomic counter reaching zero alone does not establish payload visibility. Claim ownership atomically or equivalently, broadcast, execute, then mark DONE.

An empty queue can coexist with running producers or in-flight messages. Workers must seek legal work or wait for an explicit generation-completion condition. Never retain the only opportunity to produce missing input while blocking on it. Readiness does not authorize arbitrary numerical reduction reordering; preserve token/slot/K constraints.

**Minimal deadlock example, not a GPU measurement:** two workers must execute P0/P1/P2. After P0/P1 they both wait on a consumer that needs all three producers; P2 remains unclaimed. Co-residency does not help. Complete the static producer loop, including P2, or let available workers continue production. A logical DAG can be acyclic while worker program order and resource waits create a cycle.

Compare stage-major execution with one selected policy before adding general queues, work stealing, and deeper prefetch together. Report queue/poll work, latest task, first useful consumer math, and layer time. Entering a consumer wait is not compute overlap.

<a id="completion"></a>

## 5. Prove completion, safe reuse, and rank progress

Define how each generation's task set becomes closed before defining completion. Fixed tasks may use known live counts; routing/EP tasks require count/dispatch publication. Account for a generated child before releasing the parent's final outstanding responsibility, avoiding a false zero-work interval.

A termination proof must establish all of the following: no more inputs/tasks can enter the generation; all required work executed and published; no unaccounted in-flight work/communication remains; final output/state is committed; and last readers of reusable buffers are done. Use a coherent coordinator/counting/closure protocol, not unrelated counter snapshots. **An empty queue, all tasks claimed, or one done flag is insufficient on its own.**

TP must leave a path to produce missing partials. EP covers dispatch→expert→return→token combine, empty peers/counts, and hotspot owners. CP covers scan/score→partial/candidate merge, empty shards, and output owners. Zero-work participants still complete required publication. If transport needs another GPU kernel, host progress, or a remote rank, state when it can run; workers cannot occupy all its resources and then wait for it.

A slot follows `FREE → IN_USE(g) → OUTPUT_READY(g) → CONSUMED(g) → FREE` according to actual readers. Tag/parity reuse and wrap must reject stale messages. Output readiness does not necessarily permit scratch/IPC release. An empty queue does not justify early layer/step-state commit.

<a id="service-loop"></a>

## 6. Add a service protocol only when cross-invocation persistence is needed

First validate finite workers, then wrap them in `WAIT_SUBMISSION → EXECUTE(g) → PUBLISH_COMPLETION(g) → WAIT_SUBMISSION`. Process generations sequentially initially unless concurrency/isolation is separately established. Put request pointers, shapes/plans, scratch, and state contracts in descriptors or fixed runtime configuration; retain pointers until consumption is acknowledged.

Designate a submission coordinator. It claims a descriptor, initializes control state, and publishes the active generation. All participating workers acquire the same g before entering the task loop. Workers report completion; after task closure, required participants, and transfers finish, the coordinator publishes the result. It must finish its own production too. CTAs must not independently dequeue different requests and then enter one shared grid barrier. Concurrent generations need separate worker/arena ownership or another scheduling proof.

Implement these interfaces explicitly:

- **Submission/visibility:** bounded slots/ring, sequence numbers, descriptor writes → publication → GPU acquire. Verify host/device or device/device memory scope in current HIP/runtime. Ordinary CPU stores are not assumed visible to GPU polling.
- **Wakeup/backpressure:** define full-queue behavior and read-pointer ownership. Never overwrite unconsumed descriptors or drop requests. A producer queued behind the waiting kernel on the same stream cannot run. Another stream still needs resource/dependency proof.
- **Completion acknowledgement:** bind completion to a generation and distinguish output-ready from permission to reuse input/scratch. Host-observed completion also needs result-data visibility.
- **Stop/failure:** stop accepting submissions, drain accepted work and communication, publish STOPPED, confirm kernel/transport exit, then release arena/IPC/groups. A worker exiting immediately on stop can strand peers. Timeout triggers error handling; it does not establish data completion. If cancellation/recovery is unsupported, say so and do not reuse contaminated state.

Finite Graph replay does not prove a service loop can be captured/replayed identically. Events/Graph nodes behind a never-returning kernel cannot execute. The runtime needs an independent submission/completion channel and timing method. Without it, retain finite invocation mode.

<a id="validation"></a>

## 7. Validate and measure the selected lifetime

Select checks affected by the design: tasks below/equal/above workers, masked final rounds, zero work, skew, delayed producers/peers, changing inputs, poisoned scratch, multiple generations, legal slot reuse/wrap. Service mode adds idle-then-submit, full queue, and stop/drain with in-flight work. Check each task/state effect exactly once, complete output, and absence of stale payloads. Establish the wait/progress argument first, then test whether implementation matches it; non-hanging runs alone are not a proof.

Run N/P/F at fixed lifetime and treat a lifetime change as a separate candidate. When comparing worker counts/roles, static/dynamic claims, or handoff granularity, hold math/plan fixed and record compiled resources, maximum rounds, poll/queue cost, and final tail.

| Symptom | Direct case and next action |
|---|---|
| Standalone regresses after adding the worker loop | [K3 resource loss](../../port/references/resident-regression-case.md): fix body/tasks before expanding fusion |
| New partition waits for the wrong generation | [DSV4 commit point](staged-epoch-case.md): align invocation owner/lifetime |
| Long replay corrupts ready state | [GLM LDS race](synchronization.md#replay-race): inspect old readers, overwrite, and bounds |
| Stage starts overlap but useful math is unknown | [v13 observation gap](../../benchmark/references/pmc-cases.md#pmc-missing): mark first useful consumer math and check observer perturbation |

Do not time service requests with an event that waits for kernel exit. Report submission-to-result latency, throughput, startup/drain, and idle cost separately. Use a completion signal that can progress concurrently with valid visibility, or request boundaries on a validated timebase. Different scopes are not directly comparable.

PMC intervals may include idle polling, multiple generations, and auxiliary communication. Normalize by actual completed tasks/invocations, not loop iterations or resident waves as layer counts. ATT samples of polling do not establish GEMM coverage; match generation/stage and exact binary. An idle control can help interpretation but subtraction does not isolate pure compute time. Apply the [diagnostic rules](../../benchmark/references/diagnostics.md), then confirm performance without profiling at the same boundary.

Deliver lifetime, worker/task mapping, initialization/completion/exit protocol, progress argument, validated scope, and runnable entry points. Change one evidence-supported mechanism next and update B/C through the [RSI loop](../../port/references/iteration-loop.md). Enabling persistent mode is not an acceptance result.
