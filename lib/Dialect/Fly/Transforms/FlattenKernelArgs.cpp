// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

using namespace mlir;

namespace mlir {
namespace fly {
#define GEN_PASS_DEF_FLYFLATTENKERNELARGSPASS
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"
} // namespace fly
} // namespace mlir

namespace {

void appendLeafTypes(Type type, SmallVectorImpl<Type> &leaves) {
  if (auto structTy = dyn_cast<LLVM::LLVMStructType>(type)) {
    for (Type field : structTy.getBody())
      appendLeafTypes(field, leaves);
    return;
  }
  leaves.push_back(type);
}

Value buildStruct(OpBuilder &b, Location loc, LLVM::LLVMStructType structTy, ArrayRef<Value> leaves,
                  unsigned &next) {
  Value result = LLVM::UndefOp::create(b, loc, structTy);
  for (auto [index, field] : llvm::enumerate(structTy.getBody())) {
    Value value = isa<LLVM::LLVMStructType>(field)
                      ? buildStruct(b, loc, cast<LLVM::LLVMStructType>(field), leaves, next)
                      : leaves[next++];
    result = LLVM::InsertValueOp::create(b, loc, result, value,
                                         b.getDenseI64ArrayAttr({static_cast<int64_t>(index)}));
  }
  return result;
}

void appendLeafValues(OpBuilder &b, Location loc, Value value, SmallVectorImpl<Value> &leaves) {
  auto structTy = dyn_cast<LLVM::LLVMStructType>(value.getType());
  if (!structTy) {
    leaves.push_back(value);
    return;
  }
  for (auto [index, field] : llvm::enumerate(structTy.getBody())) {
    Value fieldValue = LLVM::ExtractValueOp::create(
        b, loc, field, value, b.getDenseI64ArrayAttr({static_cast<int64_t>(index)}));
    appendLeafValues(b, loc, fieldValue, leaves);
  }
}

LogicalResult flattenKernel(gpu::GPUFuncOp func) {
  // Back to front, so the indices of the arguments still to visit do not move.
  for (int argIndex = static_cast<int>(func.getNumArguments()) - 1; argIndex >= 0; --argIndex) {
    auto structTy = dyn_cast<LLVM::LLVMStructType>(func.getArgumentTypes()[argIndex]);
    if (!structTy)
      continue;
    Location loc = func.getArgument(argIndex).getLoc();
    SmallVector<Type> leafTypes;
    appendLeafTypes(structTy, leafTypes);
    SmallVector<unsigned> indices(leafTypes.size(), argIndex + 1);
    SmallVector<DictionaryAttr> attrs(leafTypes.size(), DictionaryAttr::get(func.getContext()));
    SmallVector<Location> locs(leafTypes.size(), loc);
    if (failed(func.insertArguments(indices, leafTypes, attrs, locs)))
      return func.emitOpError("cannot insert the fields of struct argument ") << argIndex;

    OpBuilder b = OpBuilder::atBlockBegin(&func.front());
    SmallVector<Value> leaves;
    for (unsigned i = 0; i < leafTypes.size(); ++i)
      leaves.push_back(func.getArgument(argIndex + 1 + i));
    unsigned next = 0;
    Value rebuilt = buildStruct(b, loc, structTy, leaves, next);
    func.getArgument(argIndex).replaceAllUsesWith(rebuilt);
    if (failed(func.eraseArgument(argIndex)))
      return func.emitOpError("cannot erase struct argument ") << argIndex;
  }
  return success();
}

class FlattenKernelArgsPass
    : public mlir::fly::impl::FlyFlattenKernelArgsPassBase<FlattenKernelArgsPass> {
public:
  using mlir::fly::impl::FlyFlattenKernelArgsPassBase<
      FlattenKernelArgsPass>::FlyFlattenKernelArgsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    WalkResult kernels = module.walk([](gpu::GPUFuncOp func) {
      if (func.isKernel() && failed(flattenKernel(func)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (kernels.wasInterrupted())
      return signalPassFailure();

    // Every kernel argument of struct type was flattened above, so flattening every struct operand
    // keeps each call site in step with its callee.
    module.walk([](gpu::LaunchFuncOp launch) {
      if (llvm::none_of(launch.getKernelOperands().getTypes(), llvm::IsaPred<LLVM::LLVMStructType>))
        return;
      OpBuilder b(launch);
      SmallVector<Value> operands;
      for (Value operand : launch.getKernelOperands())
        appendLeafValues(b, launch.getLoc(), operand, operands);
      launch.getKernelOperandsMutable().assign(operands);
    });
  }
};

} // namespace
