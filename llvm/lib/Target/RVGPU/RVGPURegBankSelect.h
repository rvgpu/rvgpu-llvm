//===- RVGPURegBankSelect.h -------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RVGPU_RVGPUREGBANKSELECT_H
#define LLVM_LIB_TARGET_RVGPU_RVGPUREGBANKSELECT_H

#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"

namespace llvm {

class RVGPURegBankSelect final : public RegBankSelect {
public:
  static char ID;

  RVGPURegBankSelect(Mode RunningMode = Fast);

  StringRef getPassName() const override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // namespace llvm
#endif
