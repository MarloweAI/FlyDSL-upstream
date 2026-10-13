# Numerical, routing, and state counterexamples

Sources: [G1/G4/G5, D1/D2, R1, S1/S2, K1](../../port/references/sources.md). Current numerical configuration comes from [config/loader/dispatch discovery](../../port/references/model-config-discovery.md), not historical constants or tolerances.

<a id="numerical-paths"></a>

## Keep a numerical-path ledger

Record per stage: `logical shape / packed shape / A dtype / W dtype / scale dtype+group+layout / MFMA / accumulation / activation / output rounding / route weighting / TP order`.

FP8 requires the E4M3 variant, scale/amax algorithm, and group. FP4 requires packed-byte versus logical-K mapping. BF16 requires RNE versus truncation and rounding before/after activation. The independent oracle must implement these boundaries without calling the epilogue under test.

### DSV4 routed and shared branches

Resolve hidden dimensions, global/local intermediate, expert counts, top-k, route scale, and hash/bias layer distribution from checkpoint → arguments → constructor → loader/dispatch. Do not copy a Pro snapshot's dimensions or first-layer rules.

The historical routed path used MXFP4/per32 E8M0, E4M3 inputs with per32 upward scales, SwiGLU remaining FP32 until FP8 conversion, and the native fused exponent rule. Shared used E4M3 weights, 128×128 E8M0 scales, and its own BF16 RNE boundary. Rounding routed mid to BF16 before FP8 quantization violated that contract. Preserve each branch's distinct rounding; use these formats only when current code confirms them.

Port scale rules from original `quant_mxfp8`/fused epilogues and an independent oracle using extrema, zeros, rounding boundaries, and saturation. “Upward” or “nearest” alone is not a sufficient formula.

### GLM legacy and native MFMA

Historical layout optimization retained BF16-MFMA numerics. Later native scaled MFMA used block32 MXFP8 activations; old FP8-dequant/BF16-mid references did not model it. Bitwise old/new binary agreement validates that change's equivalence, not a missing native golden.

### K3 framework comparisons

One checkpoint-backed K3 comparison used different activation, routed-expert, recurrent-state dtypes, and TP combination in ATOM and mono. Independent state/replay checks did not establish bitwise or model-quality equivalence. If native returns prefix/routed/shared separately, materialize and time the same final output boundary.

<a id="routing"></a>

## Separate three routing mismatch classes

1. **Permutation:** same expert set, different slots. Align probabilities/mid by expert ID for diagnosis, handle shared separately, and preserve raw slot results.
2. **Exact ties:** state the tie rule. An AITER pivot/compact fast path did not guarantee lowest-ID-first; a stable low-ID reference could choose another legal set. Required ABI consistency still needs resolution.
3. **Near scores:** do not classify them as ties. Historical GLM packed keys cleared the low 8 score bits to store expert IDs, misordering genuinely distinct scores. Full FP32 scores fixed precision, not merely output order.

In one GLM sigmoid+bias path, selection used `s_i+bias_i`, while weights normalized unbiased `s_i` over selected experts, then applied route scale; shared weighting was separate. Do not apply this formula to K3/DSV4 without verification.

A K3 selector carried winner bias through max/min reduction and retained `best_score-best_bias`. Replacing it with raw score could change FP32 rounding despite algebraic equivalence. Winner ID, score, and bias must identify the same candidate.

Fixed logits/bias isolates selection. A near-tie flip caused by one upstream attention ulp remains an end-to-end issue; fixed-routing agreement does not clear it.

<a id="unstable-reference"></a>

## The framework baseline can itself drift

Historical DSV4 native ATOM used a BF16 atomic split-K router that could change near-tie routing; shared ASM truncation/atomics also affected output. Preserve repeated native results and name any separate stable/RNE profile explicitly. Do not overwrite failures or call a modified baseline original production.

A full-layer control using the same stable compressor projection or FlyDSL MoE in both arms tests that controlled path. It does not replace native-MoE or full-model generation validation.

<a id="state"></a>

## State is part of the output contract

Check all KDA convolution/recurrent snapshots, AttnRes sources, the four HCState tensor classes, compressor ring, and compressed KV/index caches. Hidden output alone misses errors that appear on the next decode step.

Retain decoded logical values at modified positions, raw bytes in declared write regions, byte-exact untouched regions, and padding/negative-slot/uncommitted-row behavior. Restore/clone separate copies of the same initial buffers so one arm cannot contaminate the other.

Select consecutive decode, restart, partial acceptance/all rejection, interleaved requests, reallocation, and long replay according to the affected ABI. Do not use only zero caches or artificially correlated routes; they can hide address errors and grouping costs.
