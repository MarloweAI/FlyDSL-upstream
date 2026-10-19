# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
add_mlir_public_c_api_library(MLIRCPILLVMOption
  llvm.cc
  LINK_LIBS PUBLIC
  LLVMSupport
  LLVMScalarOpts
)
if(LLVM_VERSION_MAJOR GREATER_EQUAL 24)
  set(FLYDSL_LLVM_SOURCE_DIR "$ENV{FLYDSL_LLVM_SOURCE_DIR}" CACHE PATH "Pinned LLVM source tree")
  set(FLYDSL_LLVM_BUILD_DIR "$ENV{FLYDSL_LLVM_BUILD_DIR}" CACHE PATH "Pinned LLVM build tree")
  if(NOT EXISTS "${FLYDSL_LLVM_SOURCE_DIR}/llvm/lib/Transforms/Scalar/ScalarOptions.h"
     OR NOT EXISTS "${FLYDSL_LLVM_BUILD_DIR}/lib/Transforms/Scalar/ScalarOptions.inc")
    message(FATAL_ERROR "LLVM 24 requires its matching ScalarOptions source/generated headers")
  endif()
  target_include_directories(obj.MLIRCPILLVMOption PRIVATE
    "${FLYDSL_LLVM_SOURCE_DIR}/llvm/lib/Transforms/Scalar"
    "${FLYDSL_LLVM_BUILD_DIR}/lib/Transforms/Scalar")
  target_compile_definitions(obj.MLIRCPILLVMOption PRIVATE FLYDSL_LLVM24_SCALAR_OPTIONS=1)
endif()
