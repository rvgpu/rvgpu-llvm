//===-- TargetInfo/RVGPUTargetInfo.cpp - TargetInfo for RVGPU -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
//
//===----------------------------------------------------------------------===//

#include "TargetInfo/RVGPUTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;
/// The target for GCN GPUs.
Target &llvm::getTheRVGPUTarget() {
  static Target TheRVGPUTarget;
  return TheRVGPUTarget;
}

/// Extern function to initialize the targets for the RVGPU backend
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeRVGPUTargetInfo() {
  RegisterTarget<Triple::rvgpu, false> RVGPU(getTheRVGPUTarget(), "rvgpu",
                                            "Sietium RISC GPUs", "RVGPU");
}
