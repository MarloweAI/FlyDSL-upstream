// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl --verify-diagnostics

// 256 partitions of 256 slots need more than a 4095-slot buffer.
module attributes {gpu.container_module, fly_ktrace.capacity_slots = 4095 : i32,
                   fly_ktrace.partitions = 256 : i32, fly_ktrace.partition_slots = 256 : i32} {
  gpu.module @kernels [#rocdl.target<chip = "gfx950">] {
    // expected-error @below {{do not fit the trace buffer of 4095 slots}}
    gpu.func @too_many_partitions(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "m"
      gpu.return
    }
  }
}
