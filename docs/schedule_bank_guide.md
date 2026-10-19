# sched_bank — VGPR Bank Scheduling Hint (gfx1250)

## Overview

`sched_bank` is a VGPR register-bank placement hint for the AMD **gfx1250**
architecture. It is an **identity operation** at runtime — the value passes
through unchanged — but it instructs the LLVM register allocator to place the
value into a specific 256-register VGPR bank (0–3).

By controlling bank placement, kernel authors can reduce VGPR bank conflicts
that stall the execution pipeline.

## Requirements

This feature requires a **custom LLVM build** that includes the
`rocdl.schedule.bank` op. The standard upstream LLVM does not have this op.

Build FlyDSL with the custom LLVM:

```bash
LLVM_SOURCE=custom-upstream-old bash scripts/build_llvm.sh -j64
bash scripts/build.sh -j64
```

## API

```python
from flydsl.expr.rocdl import sched_bank

sched_bank(value, bank, soft=False)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `value` | DSL numeric / `ir.Value` | The value to annotate. Any scalar type (f32, i32, etc.). |
| `bank` | `int` | Target VGPR bank: **0, 1, 2, or 3**. Each bank covers 256 registers. |
| `soft` | `bool` | Default `False` (strict). When `True`, the hint is advisory — the allocator may ignore it if it conflicts with other constraints. |

### Return value

Returns the same value with the same type (identity semantics).

## Usage

### Strict placement (default)

Force a value into VGPR bank 2:

```python
from flydsl.expr.rocdl import sched_bank

some_value = sched_bank(some_value, bank=2)
```

Generated IR:

```mlir
%0 = rocdl.schedule.bank %val, 2 : f32
```

### Soft (advisory) placement

Suggest bank 1, but allow the allocator to override:

```python
some_value = sched_bank(some_value, bank=1, soft=True)
```

Generated IR:

```mlir
%0 = rocdl.schedule.bank %val, 1 soft : f32
```

### Typical kernel pattern

Annotate MFMA/WMMA operands to avoid bank conflicts in the inner loop:

```python
import flydsl.expr as fx
from flydsl.expr.rocdl import sched_bank

@flyc.kernel
def my_kernel(...):
    # Load tiles
    a_tile = ...  # load operand A
    b_tile = ...  # load operand B

    # Place A and B in different banks to avoid conflicts
    a_tile = sched_bank(a_tile, bank=0)
    b_tile = sched_bank(b_tile, bank=1)

    # MMA — register allocator respects the bank hints
    accumulator = fx.gemm(a_tile, b_tile, accumulator)
```

## Banks

gfx1250 has 4 VGPR banks, each spanning 256 registers:

| Bank | Register range |
|---|---|
| 0 | v0 – v255 |
| 1 | v256 – v511 |
| 2 | v512 – v767 |
| 3 | v768 – v1023 |

## Testing

Run the unit tests (skipped automatically if the custom LLVM is not present):

```bash
python3 -m pytest tests/unit/test_sched_bank.py -v
```
