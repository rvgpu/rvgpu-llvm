//===-- RVGPUTargetInfo.cpp - RVGPU Target Implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetInfo/RVGPUTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
using namespace llvm;

Target &llvm::getTheRVGPUTarget32() {
  static Target TheRVGPUTarget32;
  return TheRVGPUTarget32;
}
Target &llvm::getTheRVGPUTarget64() {
  static Target TheRVGPUTarget64;
  return TheRVGPUTarget64;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeRVGPUTargetInfo() {
  RegisterTarget<Triple::nvptx> X(getTheRVGPUTarget32(), "nvptx",
                                  "NVIDIA PTX 32-bit", "RVGPU");
  RegisterTarget<Triple::nvptx64> Y(getTheRVGPUTarget64(), "nvptx64",
                                    "NVIDIA PTX 64-bit", "RVGPU");
}
