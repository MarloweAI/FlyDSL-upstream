# {{MODEL}} mono-kernel model contract

Status: draft; unknown fields have not been validated.

## Workload and finish condition

- Checkpoint/config/source revision: unknown
- Layer families and exceptions: unknown
- Goal, target buckets, baseline and performance boundary: unknown
- B/conc (requests), per-request S_i, total T=sum(S_i): unknown
- Does S include the current token? MTP draft/verify convention: unknown
- Current TP / EP / attention-CP (DCP) / indexer-CP, with actual groups and scope: unknown
- Hidden / latent / global and local intermediate / experts / top-k: unknown
- Live context lengths / cache capacities: unknown
- Numerical criteria and existing reference revision: unknown
- Fallback/unsupported inputs: unknown

## Configuration discovery and resolved values

Locate these in the current project/model; do not copy historical model constants.

- Actual project/model entry, checkpoint config and model registry selection: unknown
- Parsed config/args, defaults, aliases and effective runtime overrides: unknown
- Representative layer constructor and layer-specific exceptions: unknown
- Weight loader, per-tensor TP/EP shard, pack/shuffle and selected kernel branch: unknown

| Semantic field | Config/source symbol and revision | Effective value/override | Global vs rank-local; logical vs packed | Runtime tensor/dispatch evidence |
|---|---|---|---|---|
| unknown | unknown | unknown | unknown | unknown |

Support comes from current guards plus completed validation; performance coverage is separate.

## Parallel plan and deployment comparison

- Current / candidate parallel_plan_id; fixed constraints versus tunable strategy: unknown
- GPU count/topology/memory budget; global request trace versus per-rank tokens: unknown
- Context-length distribution, capacity, prefill/decode and refresh/reuse coverage: unknown

| Stage | Rank group/members | Tensor/expert/context split and replicas | Weight/KV/state owner | Communication and output owner |
|---|---|---|---|---|
| unknown | unknown | unknown | unknown | unknown |

- EP token/route-slot/global-to-local expert mapping, shared owner and empty-rank completion: unknown
- CP score/attention merge semantics; compute-only split versus physical KV sharding: unknown
- Layer-boundary redistribution, preparation/state migration and legal fallback: unknown
- Fixed-plan fusion comparison / fixed-implementation strategy comparison / best available unfused deployment: unknown
- Missing comparison cells and limits on attribution: unknown

## Stage numerical contract

| Stage | Logical/packed shape | A/W dtype | Scale dtype/group/layout/rule | MFMA/accumulator | Activation/output rounding | Reference |
|---|---|---|---|---|---|---|
| unknown | unknown | unknown | unknown | unknown | unknown | unknown |

Routing score, selection bias, probabilities, tie/order rule, shared placement: unknown

## State and runtime ABI

- Request IDs / positions / slots / page tables / causal bounds: unknown
- KV/index cache format and single-write owner: unknown
- Recurrent/MTP predecessor and snapshot mapping: unknown
- Residual/HCState/compressor lifecycle, if present: unknown
- Weight packing and memory ownership: unknown
- Graph keys, stream contract, step/layer epochs and close order: unknown

## Fusion boundary and validation

- Actual target GPU DAG and remaining external operations: unknown
- Execution lifetime: finite per-invocation worker loop or cross-invocation service; task and kernel completion boundaries: unknown
- Full output materialization boundary: unknown
- Launch count evidence (per layer/rank, auxiliary kernels separate): unknown
- Same-math staged comparison: unknown
- Independent quantization-aware oracle: unknown
- Framework golden/state/cache checks: unknown
- Full-model generation/MTP acceptance, if in scope: unknown

## Parameter envelope

| Axis | Intended values | Fixed invariants | Static constraints | Compiled | Correct | Performance |
|---|---|---|---|---|---|---|
| unknown | unknown | unknown | unknown | unknown | unknown | unknown |

Compilation/GPU/time budgets and experiment stopping conditions: unknown
