// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl --split-input-file --verify-diagnostics | FileCheck %s

// 64 partitions of a 64 MiB buffer: 65 cursor lines of 8 slots (520 slots), then 2096632 slots
// of wave regions, 8190 of them with a short last one. Wave k of partition p takes region
// k * 64 + p.
// CHECK-LABEL: gpu.func @partitioned
// CHECK: %[[TOP_BITS:.*]] = llvm.mlir.constant(26 : i32)
// CHECK-NEXT: %[[PART:.*]] = arith.shrui %{{.*}}, %[[TOP_BITS]] : i32
// CHECK: scf.if
// CHECK-DAG: %[[STRIDE:.*]] = llvm.mlir.constant(256 : i32)
// CHECK-DAG: %[[NEXT:.*]] = arith.addi %[[PART]], %{{.*}} : i32
// CHECK: arith.muli %[[NEXT]], %[[STRIDE]]
// CHECK: llvm.atomicrmw add
// CHECK: %[[WAVE:.*]] = rocdl.readfirstlane
// CHECK-DAG: %[[REGIONS:.*]] = llvm.mlir.constant(8190 : i32)
// CHECK-DAG: %[[P:.*]] = llvm.mlir.constant(64 : i32)
// CHECK: %[[K:.*]] = arith.minui %[[WAVE]], %[[REGIONS]]
// CHECK: %[[KP:.*]] = arith.muli %[[K]], %[[P]]
// CHECK: %[[INTERLEAVED:.*]] = arith.addi %[[KP]], %[[PART]]
// CHECK: arith.minui %[[INTERLEAVED]], %{{.*}}
// CHECK-DAG: llvm.mlir.constant(2096632 : i32)
// CHECK-DAG: llvm.mlir.constant(520 : i32)
// CHECK: llvm.ptrtoint %{{.*}} : !llvm.ptr<1> to i64
module attributes {fly_ktrace.capacity_slots = 2097151 : i32, fly_ktrace.partitions = 64 : i32, gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx950">] {
    gpu.func @partitioned(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}

// -----

module attributes {fly_ktrace.partitions = 48 : i32, gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx950">] {
    // expected-error @below {{fly_ktrace.partitions must be a power of two, got 48}}
    gpu.func @not_a_power_of_two(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}

// -----

module attributes {fly_ktrace.capacity_slots = 127 : i32, fly_ktrace.partitions = 64 : i32, gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx950">] {
    // expected-error @below {{a 4096-byte trace buffer cannot hold 64 partitions}}
    gpu.func @too_small(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}
