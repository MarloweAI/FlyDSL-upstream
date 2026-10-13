# RSI loop: turn current evidence into the next action

This loop governs implementation, experiment decisions, and updates to reusable guidance. End each iteration with an executable `next_action`; continue when it is authorized and prerequisites are available.

## 1. Reconcile facts and keep one current state

Reuse equivalent project records. By default, `workflow-state.json` holds the objective/buckets, P gate, inputs, R/B/C, partition, active experiment, latest result, rejected directions, limits, and next action/exit criterion. Populate inputs after resolving actual config/checkpoint/fixtures. Identify source, binary, input, and results with project IDs, paths relative to declared roots, and revision/hash. A mutable filename alone is insufficient; no separate provenance manifest is needed.

`parallel_plans[plan_id]` records per-stage rank/expert/context/KV mappings, resource constraints, and B per bucket. `active_parallel_plan` binds C. `fastest_validated_B_by_bucket` records the equal-resource deployment winner and plan ID. Add missing fields to old tasks instead of reinitializing them.

Resume in this order: **objective → actual R/B/C files and identities → weight/group/cache bindings → completed jobs/results → rejected directions → next action**. Reconcile conflicts before acting. A chat claim, newest file, or old pending job is not evidence for promotion or automatic rerun.

Append iterations to `experiments.jsonl`, one JSON object per line; connect plan and result records with `experiment_id`. An iteration card means this structured record, not a mandatory Markdown file. Use the [detail template](../assets/iteration-card.md) only for complex experiments and link raw artifacts. Maintain each conclusion in one place.

## 2. Choose the smallest experiment that can change the decision

Resolve the earliest failed gate: fixture/reference → standalone body → resident/progress → new fusion edge → full layer/parallel scope → real framework. Do not expand fusion or rank performance while a prerequisite fails.

For unfinished functionality, specify the next edge's removable launch/materialization/repeated computation/communication and added registers, LDS, polling, copies, and tail. Preserve full-vector, routing, recurrence, and cross-rank dependencies. A costly construction branch needs a concrete recovery mechanism and exploration limit before entry.

For a functional implementation, inspect the uninstrumented layer completion path: repeated work, loaders, residency efficiency, handoff, rank/CTA tail, communication, or framework overhead. TP/EP/CP is another [candidate axis](parallel-plan.md). Use observation to distinguish unknown mechanisms, then implement. Not every iteration requires a kernel edit or profiler capture.

Record before execution:

```text
experiment_id / gate / bucket / parallel_plan_id / exact B and C identities
observation and source: current evidence or historical prior
one main change, causal hypothesis, competing explanation
prediction: task/instruction/byte/completion-point change; falsifying outcome
validation: numerical/state/liveness criteria; timing and output boundary
limits: candidates, compilation/code size/GPU time, stopping conditions
```

Select a first experiment from the [symptom guide](../../benchmark/references/optimization-priors.md). Historical tiles/thresholds are not defaults. Use bounded [parameter screening](../../benchmark/references/experiment-design.md), not a wider grid under a disproven mechanism.

## 3. Execute, decide, update

Run required checks: contracts/compile/resources → independent endpoints, full layer, state/progress → uninstrumented pairs → necessary diagnostics → fresh winner confirmation and affected buckets. Reuse valid evidence; repeat checks invalidated by the change.

Use N/P/F when resident compatibility is unknown or regresses. Hold math, inputs, parallel plan, and execution boundary fixed: N ordinary standalone, P resident configuration with separate launches, F fused launch. Three permanent production paths are unnecessary.

| Observation | Decision | Next implementation or experiment |
|---|---|---|
| Compile failure or infeasible resources | `compile_failed`; no performance conclusion | Fix interface/body/resource axis and record pruning condition |
| Numerical/state error | `correctness_failed`; do not promote B | [Freeze the first divergent endpoint](../../numerics/SKILL.md), fix it, replay |
| Hang or unstable replay | `liveness_failed`; instrumentation masking is not a fix | Check coverage, epochs, bounds, and progress cycles; [race case](../../schedule/references/synchronization.md#replay-race) |
| P already much slower than N | Resident body/task/resource cost or a confounded comparison | Verify equal work; use the [K3 case](resident-regression-case.md) to isolate the body |
| P close to N, F slower | Joint resources, handoff, or program order | Compare barrier/tile readiness, ownership, metadata, and tail; not a pure fence measurement |
| Island faster, layer/serving unchanged | Local gain not delivered | Inspect critical path, conversions/communication, and [integration boundaries](../../atom/references/integration.md) |
| Target completion reliably improves in fresh pairs and required gates pass | `accepted`; update scoped B and retain old B, independently of mechanism status | Re-identify the bottleneck; record verified eliminated work and unresolved causes |
| Mechanism appears, layer not faster | Off critical path or offset by new costs | Run one distinguishing control; otherwise reject under current conditions |
| Predicted instruction/work signal absent | Mechanism not established | Check dispatch/binary, compiler, and observation before judging the mechanism |
| Noise, contamination, or missing required acceptance evidence | `inconclusive` | Same-code control or bounded follow-up for the specific gap |
| Some buckets improve, others regress | Scoped conclusion only | Preserve per-bucket B; validate dispatch and state boundaries |
| C adds correct functionality but remains slower | `functional_only`; retain B | Run the predefined downstream experiment; revert/change structure on failure or budget exhaustion |

Use `rejected_perf` for a valid experiment showing clear regression and `not_run` for unexecuted work. Alongside experiment status, report numerics, state/liveness, performance, and fusion scope separately. Validation dimensions use `pass / fail / not_run / not_applicable`; explain non-applicability.

Record `mechanism_status` for the tested hypothesis separately as `supported / falsified / unknown`. Missing or inadequate observation leaves it `unknown`; suitable evidence contradicting its prediction makes it `falsified`. Neither status blocks updating B when all required acceptance evidence passes. Conversely, a reliable speedup does not establish the predicted mechanism. Diagnose an unexplained gain when it can change the next decision, not merely to permit promotion.

Pair each attempted change with its observed effect and exact source/binary, fixture, command, environment, and raw results. Then record the decision, B/C update, rejected conditions/retry triggers, and next action. PMC/ATT records include scope, raw quantities, denominators, normalization, prediction agreement, and gaps under the [diagnostic rules](../../benchmark/references/diagnostics.md).

## 4. Prevent repeated dead ends and improve the rules

Save rejected directions with shape/precision/environment and the new evidence that would justify retry. A renamed candidate or seed is not a new mechanism. Reconsider when routing overlap, loader, compiler, or observations invalidate the premise. Prefer larger structural opportunities over expected gains below noise or far smaller than known structural losses.

After acceptance, update the list of work already eliminated only where evidence supports it; leave unexplained causes open. If grouping or quant-once is implemented, investigate remaining work instead of proposing it again. Preserve the quantified gap to unfinished functionality; a partially fused B does not complete a full-layer mono task.

Update a general prior only when a concrete failure/counterexample changes a decision: **trigger → action → prediction → evidence → falsification/expiry condition**. Explain the old rule's wrong choice and check both the original failure and an unaffected neighboring case. For example, P regression points to the body; P≈N with F regression still points to joint resources/handoff.

Keep original references, tolerances, and raw results fixed; append interpretation rather than rewriting observations. Scope a single-case rule narrowly and mark unmeasured guidance as unvalidated. Retain reasons for rejected/deferred rule changes. A documentation walkthrough is a decision-path check, not a new GPU benchmark or independent agent evaluation.

When the goal is met, use [delivery acceptance](delivery.md) to verify final code, evidence, and commands. Otherwise report usable B, remaining gaps, and the next action.
