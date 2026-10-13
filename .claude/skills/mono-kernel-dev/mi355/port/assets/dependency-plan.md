# {{MODEL}} resident dependency plan

Status: draft; this is not a runnable kernel.

## Persistent execution contract

- Execution mode and completion unit: finite invocation / cross-invocation service loop / other, to define
- Worker grid and role sets; static striding / dynamic claim / ready-task policy: unknown
- Task state transitions, unique claim/publication and dynamic live-count closure: unknown
- Shared-state initialization order, generation owner and completion/termination proof: unknown
- Communication progress resources and blocked-worker escape/production path: unknown
- If service loop: submission visibility, bounded slots/backpressure, completion acknowledgement and stop/drain lifecycle: not applicable until selected

## Device tasks

| Stage | Task domain/count | Actual elements per task | Rank/CTA/wave owner | Waited inputs | Published payload/tag | Last reader/reuse fence |
|---|---|---|---|---|---|---|
| unknown | unknown | unknown | unknown | unknown | unknown | unknown |

- Logical output → task → rank/CTA/wave/lane/fragment → payload → consumer mapping: unknown
- Mask/padding identities, maximum byte offsets and intermediate index-width bounds: unknown
- Overlapping register/LDS/global/peer buffer lifetimes and last-reader acknowledgement: unknown
- Data edges AND per-CTA program-order edges: unknown
- Worst-case tasks per CTA and tail imbalance: unknown
- Compiled grid/block / LDS / VGPR / SGPR / private bytes / binary hash: unknown
- Actual runtime occupancy and resident-grid progress argument: unknown
- Masked/invalid task completion behavior: unknown
- TP/EP/CP plan, per-rank live counts, communication progress owner and empty-rank completion: unknown
- Payload visibility / release-acquire scope / publisher aggregation: unknown
- Epoch uniqueness, buffer slot reuse and wrap policy: unknown
- Global scratch/table bounds and LDS lifetime intervals: unknown

## Reuse hypotheses

- Token-column reuse versus repeated dense weight loads: unknown
- Observed routed slots / distinct expert union / masks / shared expert: unknown
- Original token/route slot, expert mapping and K/cross-rank combine order: unknown
- Quantization group versus producer tile; unique quant owner: unknown
- Lane weight/scale addresses and LDS bank mapping: unknown
- Input-independent prefetch and actual ISA wait domain: unknown
- Indexer score/top-k semantics and CP result ownership, if present: unknown

## Next discriminating experiment

Observation / hypothesis / single change / predicted signal: unknown

Numerical gates / uninstrumented timing protocol / rejection condition: unknown

Baseline/candidate identity and raw artifact locations: unknown
