target datalayout = "m:e-e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @reassoc_repro_k_0(i32 %0) {
  br label %2

2:                                                ; preds = %1
  %3 = icmp sgt i32 0, 0
  %4 = and i1 false, false
  %5 = sub i32 %0, %0
  %6 = add i32 %5, 1
  %7 = tail call i32 @llvm.smin.i32(i32 0, i32 %6)
  %8 = or i32 0, %7
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #0

attributes #0 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
