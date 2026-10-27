// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

// RUN: %fly-opt %s --split-input-file --convert-fly-ktrace-to-rocdl | FileCheck %s
// RUN: %fly-opt %s --split-input-file --convert-fly-ktrace-to-rocdl | FileCheck %s --check-prefix=NOCLAIM

// With static slots, wave i of a launch owns region i. The index comes from the workgroup counts
// in the hidden kernel arguments, so a wave records without a claim atomic. The first thread
// stores the launch's wave count for the host, and a wave past the buffer only counts itself.
// CHECK-LABEL: gpu.func @static_known_block
// CHECK: llvm.call_intrinsic "llvm.amdgcn.implicitarg.ptr"
// CHECK-COUNT-3: llvm.load {{.*}} invariant {{.*}} : !llvm.ptr<4> -> i32
// CHECK: llvm.mlir.constant(4 : i32)
// CHECK: gpu.subgroup_id
// CHECK: llvm.atomicrmw add {{.*}} monotonic : !llvm.ptr<1>, i32
// CHECK: llvm.mlir.constant(8 : i32)
// CHECK: llvm.store {{.*}} : i32, !llvm.ptr<1>
// CHECK: %[[REGIONS:.*]] = llvm.mlir.constant(8187 : i32)
// CHECK: %[[FITS:.*]] = arith.cmpi ult, %[[WAVE:.*]], %[[REGIONS]]
// CHECK: arith.muli %[[WAVE]]
// CHECK: %[[DROPPED:.*]] = arith.cmpi uge, %[[WAVE]]
// CHECK: scf.if %[[DROPPED]] {
// CHECK: llvm.atomicrmw add {{.*}} syncscope("agent") monotonic : !llvm.ptr<1>, i32
// CHECK: llvm.mlir.constant(1025 : i32)
// CHECK: arith.select %[[FITS]]
// NOCLAIM-LABEL: gpu.func @static_known_block
// NOCLAIM-NOT: llvm.atomicrmw {{.*}}, i64
// NOCLAIM: gpu.return
module attributes {gpu.container_module, fly_ktrace.capacity_slots = 2097151 : i32,
                   fly_ktrace.partitions = 256 : i32, fly_ktrace.partition_slots = 7936 : i32,
                   fly_ktrace.slots = "static"} {
  gpu.module @kernels [#rocdl.target<chip = "gfx950">] {
    gpu.func @static_known_block(%buf: !fly.ptr<i8, global>) kernel
        attributes {known_block_size = array<i32: 256, 1, 1>} {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}

// -----

// Without a known block size, the waves per workgroup come from the hidden group sizes.
// CHECK-LABEL: gpu.func @static_dynamic_block
// CHECK: llvm.call_intrinsic "llvm.amdgcn.implicitarg.ptr"
// CHECK-COUNT-3: llvm.load {{.*}} : !llvm.ptr<4> -> i16
// CHECK: arith.ceildivui
// NOCLAIM-LABEL: gpu.func @static_dynamic_block
// NOCLAIM-NOT: llvm.atomicrmw {{.*}}, i64
// NOCLAIM: gpu.return
module attributes {gpu.container_module, fly_ktrace.capacity_slots = 2097151 : i32,
                   fly_ktrace.partitions = 256 : i32, fly_ktrace.partition_slots = 7936 : i32,
                   fly_ktrace.slots = "static"} {
  gpu.module @kernels [#rocdl.target<chip = "gfx950">] {
    gpu.func @static_dynamic_block(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}
