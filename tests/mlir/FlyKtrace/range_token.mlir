// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    // CHECK-LABEL: gpu.func @probe
    gpu.func @probe(%buf: !fly.ptr<i8, global>) kernel {
      // CHECK-NOT: !fly_ktrace.token

      // CHECK: rocdl.readfirstlane
      // CHECK-NOT: scf.if
      // CHECK: llvm.mlir.constant(-1 : i32)
      // CHECK: %[[SLOT:.*]] = arith.select {{.*}} : i32

      // CHECK: llvm.mlir.constant(50331649 : i32)
      %t = fly_ktrace.range_start "phase" -> !fly_ktrace.token

      // CHECK: llvm.mlir.constant(67108865 : i32)

      // CHECK: %[[LAST:.*]] = llvm.mlir.constant(3 : i32)
      // CHECK: %[[IDS:.*]] = llvm.insertelement %[[SLOT]], %{{.*}}[%[[LAST]] : i32] : vector<4xi32>
      // CHECK: llvm.inline_asm has_side_effects "s_store_dwordx4 $0, $2, 0x0\0A\09s_store_dwordx4 $1, $2, 0x10\0A\09s_waitcnt lgkmcnt(0)", "s,s,s" %{{.*}}, %[[IDS]], %{{.*}}
      fly_ktrace.range_stop %t, "phase" : !fly_ktrace.token
      gpu.return
    }
  }
}
