// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

// RUN: %fly-opt %s --split-input-file --convert-fly-ktrace-to-rocdl | FileCheck %s

// Waves claim from one of 256 partition cursors keyed by their workgroup and wave, so waves that
// start together do not serialize on one address. The key needs no memory load, and there is no
// cursor pre-load.
// CHECK-LABEL: gpu.func @known_block
// CHECK-NOT: llvm.load
// CHECK: gpu.subgroup_id
// CHECK: llvm.mlir.constant(255 : i32)
// CHECK: arith.andi
// CHECK-NOT: llvm.load
// CHECK: llvm.mlir.constant(128 : i32)
// CHECK: llvm.atomicrmw add {{.*}} syncscope("agent") monotonic : !llvm.ptr<1>, i64
// CHECK: llvm.mlir.constant(7680 : i64)
// CHECK: arith.cmpi ule
// CHECK: %[[CLAIMED:.*]] = llvm.trunc
// CHECK: arith.muli %[[CLAIMED]]
// CHECK: llvm.mlir.constant(1025 : i32)
// CHECK: arith.select
module attributes {gpu.container_module, fly_ktrace.capacity_slots = 2097151 : i32,
                   fly_ktrace.partitions = 256 : i32, fly_ktrace.partition_slots = 7936 : i32} {
  gpu.module @kernels [#rocdl.target<chip = "gfx950">] {
    gpu.func @known_block(%buf: !fly.ptr<i8, global>) kernel
        attributes {known_block_size = array<i32: 256, 1, 1>} {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}

// -----

// One partition (the default without layout attributes) needs no key.
// CHECK-LABEL: gpu.func @one_partition
// CHECK-NOT: gpu.subgroup_id
// CHECK: llvm.atomicrmw add {{.*}} syncscope("agent") monotonic : !llvm.ptr<1>, i64
module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx950">] {
    gpu.func @one_partition(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}

// -----

// A 4096-byte buffer holds no 256-slot region, so no claim is in bounds and no wave writes its
// header. The cursor still advances, so the host reports the overflow.
// CHECK-LABEL: gpu.func @smaller_than_one_wave
// CHECK: llvm.atomicrmw add {{.*}} syncscope("agent") monotonic : !llvm.ptr<1>, i64
// CHECK-NOT: arith.cmpi ule
// CHECK: %[[NONE:.*]] = arith.constant false
// CHECK: scf.if %[[NONE]] {
// CHECK: arith.select %[[NONE]]
module attributes {gpu.container_module, fly_ktrace.capacity_slots = 127 : i32,
                   fly_ktrace.partitions = 1 : i32, fly_ktrace.partition_slots = 0 : i32} {
  gpu.module @kernels [#rocdl.target<chip = "gfx950">] {
    gpu.func @smaller_than_one_wave(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}
