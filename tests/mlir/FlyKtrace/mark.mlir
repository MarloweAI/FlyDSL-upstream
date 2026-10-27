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

      // The event is wave-uniform: no leader-lane branch, no atomic, a record that does not
      // fit is redirected by a select instead of a branch, and scalar stores write it from
      // SGPRs. The wave writes back its scalar cache before it returns.
      // CHECK: llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"
      // CHECK-NOT: scf.if
      // CHECK-NOT: llvm.atomicrmw
      // CHECK: arith.cmpi ult
      // CHECK: arith.cmpi ult
      // CHECK: arith.cmpi ult
      // CHECK: llvm.mlir.constant(2 : i32)
      // CHECK: arith.select
      // CHECK-NOT: scf.if
      // CHECK: %[[RECORD:.*]] = llvm.shufflevector %{{.*}}, %{{.*}} [0, 1, 2, 3] : vector<2xi32>
      // CHECK-COUNT-4: llvm.insertelement
      // CHECK: llvm.inline_asm has_side_effects "s_store_dwordx4 $0, $2, 0x0\0A\09s_store_dwordx4 $1, $2, 0x10\0A\09s_waitcnt lgkmcnt(0)", "s,s,s" %[[RECORD]], %{{.*}}, %{{.*}} : (vector<4xi32>, vector<4xi32>, !llvm.ptr<1>) -> ()
      // CHECK: llvm.inline_asm has_side_effects "s_dcache_wb\0A\09s_waitcnt lgkmcnt(0)"
      // CHECK-NEXT: gpu.return
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}
