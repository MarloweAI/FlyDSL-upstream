# Derive a tile and storage contract before fusion

Use when selecting a body tile, changing task geometry, or joining producer and consumer layouts. The output is an implementable mapping and a small coverage/address check, not another project-wide checklist. These rules generalize the [tile16/body8 failure](reuse.md), [quantization-group constraint](reuse.md), [LDS reuse race](synchronization.md#replay-race), and [resident resource regression](../../port/references/resident-regression-case.md). New boundary checks below are design guidance, not additional historical measurements.

<a id="mapping"></a>

## 1. Map one logical output all the way to storage

For the chosen stage, write the mapping in both directions:

```text
logical output (request, token, expert/route, M/N/K or context range)
  ↔ task ID and valid extent
  ↔ rank / CTA role / wave / lane / MFMA fragment
  ↔ register or LDS element
  ↔ packed payload offset and scale group
  ↔ consumer tile and final output owner
```

Declare reduction partials separately from final outputs. Each required contribution executes exactly once, and every final element has one owner or a specified reduction protocol. A task count derived from an unrelated tile constant is invalid even when buffer capacity looks sufficient.

Build a small host-side enumeration/address check for the actual mapping: tile boundaries, final partial tiles, odd token counts, minimum/maximum route union, and role-local offsets. Check missing/duplicate contributions, output coverage, and expected consumer addresses. This is useful before GPU execution; it does not replace device numerical/progress checks. Derive MFMA counts from the body and valid/masked work for later [PMC predictions](../../benchmark/references/pmc-cases.md#pmc-grouped-ug).

<a id="mask-bounds"></a>

## 2. Define masks, padding, and address bounds

A masked store does not make an invalid lane safe. Define values consumed by reductions/MFMA and selector logic: additive identity for sums, appropriate masked maxima/scores, and model-specific empty/all-masked behavior. Do not assume zero is valid for every operation. Quantization groups must include their full logical extent; define padded elements and scale handling using the numerical contract. Never read a stale padded lane and rely on a later output mask.

All required participants still reach barriers and completion publication. A CTA with no useful math may owe protocol work. Wave-local early exits cannot bypass workgroup synchronization used by other waves.

Compute maximum byte offsets for weights, scales, pages, routes, scratch, and tags, including padding and vector-access width. Check alignment and intermediate index arithmetic, not only the final pointer type; long contexts or large expert arenas can overflow a narrow intermediate. Use wider arithmetic or an explicitly proven bound where needed. Validate page/slot boundaries and poisoned guard regions on device for affected mappings.

<a id="lifetime"></a>

## 3. Choose the storage level and prove reuse

For each intermediate, record:

| Value | Producer/consumer tile | Storage/layout/bytes | First write → last read | Visibility/reuse event |
|---|---|---|---|---|
| Actual tensor or control region | Exact mapping | Registers, LDS, global, or peer buffer | Include all waves/ranks and asynchronous readers | Owner and synchronization scope |

Keep same-owner values in registers when practical; use LDS for legal intra-CTA exchange and global/peer payloads where ownership requires them. Removing a launch does not remove a cross-CTA payload. Consider recomputation only as a separate candidate with its added instructions and unchanged numerical contract.

Disjoint live intervals can share storage only after the last reader completes. Overlapping producer/consumer stages, double buffering, prefetch, or an asynchronous send extend lifetimes. Budget the simultaneous live set, control metadata, alignment, and padding together. A two-slot scheme requires a consumer acknowledgement before a producer reuses a slot; parity alone is not a proof. For LDS, all waves must finish reading old data before repurposing it, as the GLM threshold/scan race demonstrates.

## 4. Validate the compiled joint body, then choose the next step

Compile the standalone and combined body. Record CTA geometry, actual VGPR/SGPR/LDS/private allocation, code size, occupancy, and key generated loads/waits/MFMA. A source-level lifetime reduction need not lower register allocation, and branch-local LDS declarations may still affect the whole kernel. Do not assume the joint footprint is the sum or maximum of standalone values; inspect the binary and runtime.

If coverage/addresses fail, fix the mapping before queues. If the resident body regresses, use N/P/F before adding fusion. If layout conversion or payload storage dominates, change one producer-consumer boundary and predict saved work. If overlap extends live ranges enough to erase the gain, compare a smaller island, fewer buffers, or another tile. Carry the mapping and its invalidation conditions into the existing dependency plan; avoid duplicating the same facts in multiple documents.
