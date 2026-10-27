// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

// RUN: %fly-opt %s --split-input-file --convert-fly-ktrace-to-rocdl | FileCheck %s

// Straight-line events take their index from a register counter: no atomic after the claim.
// The wave's attempted count goes to its header word right before the kernel returns.
// CHECK-LABEL: gpu.func @straight
// CHECK: llvm.atomicrmw add {{.*}} syncscope("agent")
// CHECK-NOT: llvm.atomicrmw
// CHECK: llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"
// CHECK-COUNT-6: llvm.store
// CHECK-NOT: llvm.atomicrmw
// CHECK: llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"
// CHECK-COUNT-6: llvm.store
// CHECK-NOT: llvm.atomicrmw
// CHECK: llvm.store {{.*}} : i32, !llvm.ptr<1>
// CHECK-NEXT: }
// CHECK-NEXT: gpu.return
module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx950">] {
    gpu.func @straight(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "a"
      fly_ktrace.mark "b"
      gpu.return
    }
  }
}

// -----

// A loop carries the counter, and the wave's largest count leaves the loop.
// CHECK-LABEL: gpu.func @loop
// CHECK: scf.for {{.*}} iter_args(%[[CNT:.*]] = {{.*}}) -> (i32)
// CHECK-NOT: llvm.atomicrmw
// CHECK: arith.addi %[[CNT]]
// CHECK: scf.yield
// CHECK: llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umax"
module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx950">] {
    gpu.func @loop(%buf: !fly.ptr<i8, global>) kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        fly_ktrace.mark "tick"
      }
      gpu.return
    }
  }
}

// -----

// scf.while carries the counter through both regions.
// CHECK-LABEL: gpu.func @while_loop
// CHECK: scf.while ({{.*}}) : (index, i32) -> (index, i32)
// CHECK: scf.condition
// CHECK-NOT: llvm.atomicrmw
// CHECK: llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"
// CHECK-NOT: llvm.atomicrmw
// CHECK: scf.yield {{.*}} : index, i32
// CHECK: llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umax"
module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx950">] {
    gpu.func @while_loop(%buf: !fly.ptr<i8, global>) kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %r = scf.while (%i = %c0) : (index) -> index {
        %more = arith.cmpi slt, %i, %c4 : index
        scf.condition(%more) %i : index
      } do {
      ^bb0(%j: index):
        fly_ktrace.mark "tick"
        %n = arith.addi %j, %c1 : index
        scf.yield %n : index
      }
      gpu.return
    }
  }
}

// -----

// An scf.if gives each branch a static range of indices: the then branch starts at the count,
// the else branch after the then branch's events, and the count advances past both.
// CHECK-LABEL: gpu.func @under_if
// CHECK: llvm.atomicrmw add {{.*}} syncscope("agent")
// CHECK: rocdl.readfirstlane
// CHECK-NOT: llvm.atomicrmw
// CHECK: %[[ELSE:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT: scf.if
// CHECK: llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"
// CHECK: } else {
// CHECK: llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"
// CHECK: arith.addi %{{.*}}, %[[ELSE]] : i32
// CHECK-NOT: llvm.atomicrmw
// CHECK-NOT: llvm.amdgcn.wave.reduce.umax
module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx950">] {
    gpu.func @under_if(%c: i1, %buf: !fly.ptr<i8, global>) kernel {
      scf.if %c {
        fly_ktrace.mark "then"
      } else {
        fly_ktrace.mark "else"
      }
      gpu.return
    }
  }
}

// -----

// A loop with events inside a branch has no static count, so the kernel keeps the per-event
// atomic counter.
// CHECK-LABEL: gpu.func @loop_under_if
// CHECK: llvm.atomicrmw add {{.*}} syncscope("agent")
// CHECK: llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"
// CHECK: llvm.atomicrmw add {{.*}} : !llvm.ptr<1>, i32
module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx950">] {
    gpu.func @loop_under_if(%c: i1, %buf: !fly.ptr<i8, global>) kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.if %c {
        scf.for %i = %c0 to %c4 step %c1 {
          fly_ktrace.mark "tick"
        }
      }
      gpu.return
    }
  }
}
