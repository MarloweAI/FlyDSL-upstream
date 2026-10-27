// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    // CHECK-LABEL: gpu.func @probe
    gpu.func @probe(%buf: !fly.ptr<i8, global>) kernel {
      // CHECK: fly.to_llvm_ptr

      // CHECK: llvm.mlir.constant(63492 : i32)
      // CHECK: llvm.call_intrinsic "llvm.amdgcn.s.getreg"
      // CHECK: llvm.mlir.constant(6164 : i32)
      // CHECK: llvm.call_intrinsic "llvm.amdgcn.s.getreg"
      // CHECK: rocdl.ballot
      // CHECK: scf.if
      // CHECK-NOT: atomic monotonic
      // CHECK: llvm.atomicrmw add {{.*}} syncscope("agent") monotonic : !llvm.ptr<1>, i64
      // CHECK: arith.cmpi ule, {{.*}} : i64
      // CHECK: scf.if
      // CHECK: gpu.subgroup_id
      // CHECK: arith.select
      // CHECK: rocdl.readfirstlane

      // CHECK: rocdl.ballot
      // CHECK: llvm.intr.cttz
      // CHECK: arith.cmpi eq
      // CHECK: scf.if
      // CHECK: arith.cmpi ult
      // CHECK: scf.if
      // CHECK: llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"
      // CHECK: llvm.atomicrmw add

      // CHECK: arith.cmpi ult
      // CHECK: arith.cmpi ult
      // CHECK: scf.if
      // CHECK-COUNT-6: llvm.store
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}
