# Share device math between staged and fused execution

Source D1: review of a historical DSV4 MoE builder/kernel. This is a reusable structural example, not an executable FlyDSL API or a prescribed five-stage model.

## Compile different partitions of one subgraph

One builder generated a full MoE or individual `router / route / quant / up / down` stages. The host selected one launch or five dependency-ordered launches; compile-time conditions expanded the same device math.

```python
# Conceptual pseudocode: emit_* bodies compile into the caller.
stage_order = ("router", "route", "quant", "up", "down")

def build(selected_stages, final_partition):
    def kernel(args):
        bid = cta_id()
        tag = epochs[bid] + 1
        for stage in constexpr(stage_order):
            if constexpr(stage in selected_stages):
                emit_stage(stage, args, tag)
        if constexpr(final_partition):
            # After this CTA's work and all protocol-required completion.
            epochs[bid] = tag
    return compile_kernel(kernel)

# Host sketch: launches in one invocation follow real dependencies.
staged = [build({s}, final_partition=(s == "down")) for s in stage_order]
fused = [build(set(stage_order), final_partition=True)]
```

The sketch shows the commit point only. Thread ownership, memory scope, mailbox code, task mapping, and completion synchronization still need implementation and proof.

## Commit the epoch only at the final stage

If `epochs[bid]=e` before the call, preceding standalone stages all use `e+1`. Only the full kernel or final down stage writes `epochs[bid]=e+1`; the next invocation uses `e+2`.

Incrementing after router, quant, up, and every other standalone stage makes producer and consumer tags differ, causing a hang or stale read even when math bodies match.

This design assumes:

- Compatible CTA-to-epoch-slot mapping across partitions. Cross-CTA mailbox generations must also match; one CTA's count does not establish that.
- Every required CTA advances according to contract; masks/empty tasks cannot skip required publication/commit.
- Real stream/event order between launches and no overlapping invocations on one scratch/epoch instance.
- Last readers complete before payload reuse; a final counter update alone does not prove this.
- Defined initialization, repeated Graph calls, wrap, and recovery after interrupted execution.

With a changed grid/mapping or overlapping requests/streams, do not copy `epochs[bid]+1`. Use an explicit invocation generation and stable mailbox ownership, isolated arenas, or another justified mapping. Every edge still needs matching tags and valid lifetimes.

## Apply the structure to incremental fusion

Keep standalone wrappers and independent R, then try `[router][route][quant][up+down]`. Execute each operation once and follow actual shared/quant/route dependencies rather than copying a list.

For each partition, record input/output edges, launch list, final epoch committer, in-launch mailboxes, and data completed by preceding launches. A P ablation must not wait for a producer absent from that run.

Run staged/partial/fused on the same snapshot and compare endpoints/cache/state. Then change inputs, poison scratch, and repeat Graph replay, including masks, empty tasks, and target CTA geometry. A single fixed-input run can accidentally accept stale payloads.

Retain partition switches after acceptance so regressions can be isolated under identical device math. See [synchronization](synchronization.md) for protocol details and the real LDS race.
