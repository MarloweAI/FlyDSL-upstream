// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-flatten-kernel-args | FileCheck %s

// Struct kernel arguments are passed field by field; the kernel rebuilds the struct.

module attributes {gpu.container_module} {

gpu.module @kernel_mod {
  // CHECK-LABEL: gpu.func @kernel
  // CHECK-SAME: (%[[P:.*]]: !fly.ptr<f32, global>, %[[A:.*]]: i32, %[[B:.*]]: i32, %[[N:.*]]: i32) kernel
  gpu.func @kernel(%p: !fly.ptr<f32, global>, %t: !fly.int_tuple<(?,?)>, %n: i32) kernel {
    // CHECK: %[[U:.*]] = llvm.mlir.undef : !llvm.struct<packed (i32, i32)>
    // CHECK: %[[S0:.*]] = llvm.insertvalue %[[A]], %[[U]][0]
    // CHECK: llvm.insertvalue %[[B]], %[[S0]][1]
    %d = fly.get(%t) {mode = array<i32: 0>} : (!fly.int_tuple<(?,?)>) -> !fly.int_tuple<?>
    gpu.return
  }

  // A layout packs as a struct of structs; its leaves are passed in field order.
  // CHECK-LABEL: gpu.func @layout_kernel
  // CHECK-SAME: (%{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32) kernel
  // CHECK-NOT: !llvm.struct{{.*}}) kernel
  gpu.func @layout_kernel(%l: !fly.layout<(?,?):(?,1)>) kernel {
    %d = fly.get_shape(%l) : (!fly.layout<(?,?):(?,1)>) -> !fly.int_tuple<(?,?)>
    gpu.return
  }
}

// CHECK-LABEL: @launch
func.func @launch(%p: !fly.ptr<f32, global>, %a: i32, %b: i32, %n: i32) {
  %t = fly.make_int_tuple(%a, %b) : (i32, i32) -> !fly.int_tuple<(?,?)>
  %c1 = arith.constant 1 : index
  // CHECK: %[[S:.*]] = llvm.insertvalue %{{.*}}[1]
  // CHECK: %[[F0:.*]] = llvm.extractvalue %[[S]][0]
  // CHECK: %[[F1:.*]] = llvm.extractvalue %[[S]][1]
  // CHECK: gpu.launch_func @kernel_mod::@kernel
  // CHECK-SAME: args(%{{.*}} : !fly.ptr<f32, global>, %[[F0]] : i32, %[[F1]] : i32, %{{.*}} : i32)
  gpu.launch_func @kernel_mod::@kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%p : !fly.ptr<f32, global>, %t : !fly.int_tuple<(?,?)>, %n : i32)
  return
}

}
