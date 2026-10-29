"""mcdrv.py MLIR OUT [options]: FlyDSL's final code generation (``gpu-module-to-binary``,
the LLVM bundled in FlyDSL 0.4.0) on a dumped MLIR module, in a fresh process.

The module (a FlyDSL stage dump holding the ``gpu.module``, e.g. stage 19
``reconcile_unrealized_casts``) is parsed into a context like FlyDSL's (no MLIR threads,
all dialects loaded) and the binary pass runs on it, as FlyDSL's own compile does.
Writes OUT (the code object; the ISA text with ``--format isa``) and prints
``MCDRV out=.. sha256=.. bytes=.. seconds=..``.

  --format fatbin|bin|isa|llvm  the binary pass's format (FlyDSL's compile uses fatbin; llvm: the
                            module's bitcode after LLVM's O2 pipeline, the codegen input)
  --opt NAME=VALUE          an LLVM option for the run, set in-process through FlyDSL's
                            set_llvm_option_{bool,int,str} (true/false, an integer, else a
                            string), repeatable
  --perturb SEED            first malloc and free a seeded pattern of blocks, so the
                            compiler's allocations land elsewhere in the heap
  --fill BYTE               fill those blocks with BYTE before freeing them (old contents)
  --pipeline FRAGMENTS      ';'-separated pass fragments to run before the binary pass
                            (to start from an earlier stage)
  --from-stage K            run FlyDSL's own pre-binary fragments K+1.. (its ROCm backend's,
                            no compile hints) on MLIR, a stage-K dump (0: 00_origin)
  --between SEED            heap perturbation (as --perturb) before each fragment
  --mt                      MLIR multithreading on (FlyDSL turns it off)
  --before-binary SEED      heap perturbation (as --perturb) right before the binary pass
  --imports MODULES         import these modules (comma-separated) first, as a build would
  --final FILE              write the module text right before the binary pass
  --final-bc FILE           write the module's bytecode (keeps use-list orders) there too
  --roundtrip               run the binary pass on a copy parsed back from the module's text
  --repeat N                compile N times in this process (fresh parse each time):
                            OUT.1, OUT.2, ...
  --stderr FILE             send fd 2 (LLVM's -print-* output) to FILE
"""

import argparse
import ctypes
import hashlib
import os
import random
import sys
import time


def perturb(seed, fill):
    libc = ctypes.CDLL(None)
    libc.malloc.restype = ctypes.c_void_p
    libc.malloc.argtypes = [ctypes.c_size_t]
    libc.free.argtypes = [ctypes.c_void_p]
    libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    r = random.Random(seed)
    kept = []
    for _ in range(r.randint(2000, 20000)):
        n = r.choice((16, 24, 32, 40, 48, 64, 96, 128, 192, 256, 512, 1024, 4096, 16384, 70000))
        n *= r.randint(1, 3)
        p = libc.malloc(n)
        if fill is not None:
            libc.memset(p, fill, n)
        if r.random() < 0.6:
            libc.free(p)
        else:
            kept.append(p)
    return kept


def set_opt(F, name, value):
    if value.lower() in ("true", "false"):
        old = F.set_llvm_option_bool(name, value.lower() == "true")
    else:
        try:
            v = int(value, 0)
        except ValueError:
            old = F.set_llvm_option_str(name, value)
        else:
            old = F.set_llvm_option_int(name, v)
    print(f"MCDRV opt {name}={value} (was {old})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir")
    ap.add_argument("out")
    ap.add_argument("--format", default="fatbin", choices=("fatbin", "bin", "isa", "llvm"))
    ap.add_argument("--opt", action="append", default=[])
    ap.add_argument("--perturb", type=int, default=None)
    ap.add_argument("--fill", type=int, default=None)
    ap.add_argument("--pipeline", default="")
    ap.add_argument("--from-stage", type=int, default=None)
    ap.add_argument("--between", type=int, default=None)
    ap.add_argument("--mt", action="store_true")
    ap.add_argument("--before-binary", type=int, default=None)
    ap.add_argument("--imports", default="")
    ap.add_argument("--final", default="")
    ap.add_argument("--final-bc", default="")
    ap.add_argument("--roundtrip", action="store_true")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--stderr", default="")
    a = ap.parse_args()

    import importlib

    for name in filter(None, a.imports.split(",")):
        importlib.import_module(name)
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import gpu as _gpu
    from flydsl._mlir.passmanager import PassManager
    from flydsl._mlir._mlir_libs import _mlirDialectsFly as F

    if a.stderr:
        fd = os.open(a.stderr, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(fd, 2)
    for o in a.opt:
        name, _, value = o.partition("=")
        set_opt(F, name, value)
    text = open(a.mlir, "rb").read() if a.mlir.endswith(".mlirbc") else open(a.mlir).read()
    frags = [f for f in a.pipeline.split(";") if f.strip()]
    if a.from_stage is not None:
        from flydsl.compiler.backends import get_backend

        pre = get_backend(arch="gfx950").pipeline_fragments(compile_hints={})[:-1]
        frags = pre[a.from_stage:] + frags
        print(f"MCDRV fragments from stage {a.from_stage}: {len(frags)}", flush=True)
    kept = perturb(a.perturb, a.fill) if a.perturb is not None else []
    ctx = ir.Context()
    ctx.enable_multithreading(bool(a.mt))
    ctx.load_all_available_dialects()
    for n in range(1, a.repeat + 1):
        begin = time.time()
        with ctx:
            module = ir.Module.parse(text)
            for k, frag in enumerate(frags):
                if a.between is not None:
                    kept += perturb(a.between * 1000 + k, None)
                PassManager.parse(f"builtin.module({frag})").run(module.operation)
            if a.final:
                with open(a.final, "w") as f:
                    f.write(module.operation.get_asm(enable_debug_info=True))
            if a.final_bc:
                with open(a.final_bc, "wb") as f:
                    module.operation.write_bytecode(f)
            if a.roundtrip:
                module = ir.Module.parse(module.operation.get_asm(enable_debug_info=True))
            if a.before_binary is not None:
                kept += perturb(a.before_binary * 7919 + n, None)
            fmt = "isa" if a.format == "isa" else a.format
            PassManager.parse(
                f'builtin.module(gpu-module-to-binary{{format={fmt} opts=""}})'
            ).run(module.operation)
            found = []

            def visit(op):
                if op.name == "gpu.binary":
                    found.append(_gpu.ObjectAttr(ir.ArrayAttr(op.attributes["objects"])[0]).object)
                return ir.WalkResult.ADVANCE

            module.operation.walk(visit)
        if len(found) != 1:
            raise SystemExit(f"expected one gpu.binary, found {len(found)}")
        obj = found[0] if isinstance(found[0], bytes) else found[0].encode("latin-1")
        out = a.out if a.repeat == 1 else f"{a.out}.{n}"
        with open(out, "wb") as f:
            f.write(obj)
        print(f"MCDRV out={out} sha256={hashlib.sha256(obj).hexdigest()[:16]} bytes={len(obj)} "
              f"seconds={time.time() - begin:.1f} perturb={a.perturb} fill={a.fill} "
              f"from={a.from_stage} between={a.between} mt={int(a.mt)} "
              f"opts={','.join(a.opt) or '-'}", flush=True)
        del module
    del kept


if __name__ == "__main__":
    main()
