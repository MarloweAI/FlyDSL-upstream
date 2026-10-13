# Resolve the model contract from the current project

Use during P0, numerical-path ports, and ATOM integration. Historical cases explain mechanisms; dimensions, formats, layer kinds, and support must come from the current project.

## 1. Find the config and implementation actually loaded

Identify the requested project, checkpoint/model ID, revision, and launch entry. Search from the project root, then follow callers. These are search examples, not confirmed interfaces:

```bash
rg --files -g '*config*' -g '*args*' -g '*model*' -g '*loader*' -g '*runner*'
rg -n 'model_type|architectures|AutoConfig|from_pretrained|load_config' .
rg -n 'load_weights|weight_loader|load_state_dict|quantization_config' .
```

Read the checkpoint config and the config class, argument parser, and layer constructor selected by `model_type`, `architectures`, or the registry. If unavailable, record the gap and continue static analysis of available code. Do not infer values from a related model or download a full checkpoint merely to inspect configuration.

Follow each relevant value to a runtime object:

```text
checkpoint config
  → config class/defaults/aliases
  → launch arguments and runtime overrides
  → representative layer constructor and layer-specific branches
  → weight loader, TP/EP sharding, packing/shuffle
  → forward guards/dispatch
  → actual builder arguments, tensor views, and kernel
```

When config and tensors disagree, locate the override/sharding/packing cause. With an available runtime, record resolved arguments, selected layer class, weight shapes/strides/dtypes, and branch in the fixture; configuration discovery does not require running the full model first.

## 2. Resolve only values relevant to the contract

| Question | Where to inspect | Record |
|---|---|---|
| What does a layer compute? | Config, registry, constructor, forward | Attention/recurrence, dense/MoE, first/last-layer exceptions, index refresh/reuse, hash/bias routing |
| What are the logical dimensions? | Checkpoint/config and resolved arguments | Hidden/latent, heads/head dimension, global intermediate, routed/shared experts, top-k, units |
| What does this rank compute? | Shard loader, TP/EP plan, actual Parameters | Global→local mapping, replication/split axes, padding; not every dimension divides by TP |
| Is the current parallel plan suitable? | Launch args, groups, expert/KV maps, communication trace | Per-stage TP/EP/CP groups, replicas, indexer/attention CP and KV sharding; current plan versus constraints and tunable choices |
| What is the numerical path? | Quant config, loader, quant functions, GEMM/epilogue dispatch | A/W/scales, groups/rules, MFMA, accumulator, activation/rounding; separate routed/shared branches |
| What does routing mean? | Score, selector, probability code | Selection score, bias, normalization, weight placement, tie/order rule, shared contribution |
| What does the indexer select? | Config, score/selection, attention consumer | Blocks/tokens, head aggregation, causal bounds, top-k, output owner, CP enablement |
| How are requests represented? | Metadata builder, runner, MTP | B, per-request S_i, T, current-token convention, pages/positions/slots, predecessor/successor |
| Who updates cache/state? | Allocator, writers, inter-layer state, commit/rollback | Dtype/stride/page size, live/capacity, write owner, aliases, delayed commit/lifetime |
| Which shapes really work? | Public guards, builders, dispatch, tests, traces | Native/staged/mono scope, chunking/fallback; supported buckets versus performance winners |

For each needed value retain `semantic name → config field/source symbol → override → resolved value → local tensor/selected branch → revision`. Do not copy every default.

Useful search terms; resolve to actual project symbols:

```bash
rg -n 'top_k|topk|num_experts|intermediate_size|hidden_size|num_attention_heads' .
rg -n 'fp8|mxfp4|e8m0|quant|scale|shuffle|pack|tensor_parallel|expert_parallel' .
rg -n 'page_table|block_table|slot_mapping|state_indices|commit|rollback|index.*reuse' .
rg -n 'supports|fallback|dispatch|with_indexer|context_parallel|capture' .
```

## 3. Resolve three common ambiguities

**Logical versus stored.** One packed FP4 byte holds two logical elements. Packed/scale shapes, padding, and logical K differ. Derive device views from loader/packer address formulas and check a tile. `numel()` alone does not determine expert intermediate size.

**Global versus local.** Routed experts, per-expert intermediate, shared experts, latent projections, and heads may shard differently. Record each weight's global shape, rank range, and local valid shape against its tensor. Use `global / TP` only where the loader confirms uniform sharding.

**Configured versus validated.** A CP parameter does not prove the public path enables CP; a multi-request wrapper does not prove fused-indexer support; host chunking does not prove one device launch. Test files show intended checks; completed results and traces establish capability.

## 4. Turn discovery into executable work

Populate the project's model contract, optionally using the [template](../assets/model-contract.md), then build ModelSpec and LayerFixture with the [playbook](build-playbook.md). Record config/source revision or hash, relative locators, effective parameters, and tensor metadata with current inputs/reference and experiments. Current TP/EP/CP describes today's deployment; evaluate alternatives through the [parallel plan](parallel-plan.md) instead of freezing deployment as mathematics.

Select representative layers for actual branches: hash/bias, dense/sparse, refresh/reuse, or attention/recurrence. Record initial coverage and the next layer family. Historical DSV4/K3/GLM layer indices are not defaults.

Exit when every target node/effect in forward is understood, same-state reference execution is reproducible, and weights/metadata map to real kernel arguments. Finding a config file alone does not complete P0.
