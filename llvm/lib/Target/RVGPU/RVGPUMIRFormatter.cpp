//===- RVGPUMIRFormatter.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Implementation of RVGPU overrides of MIRFormatter.
//
//===----------------------------------------------------------------------===//

#include "RVGPUMIRFormatter.h"
#include "GCNSubtarget.h"
#include "RVMachineFunctionInfo.h"

using namespace llvm;

bool RVGPUMIRFormatter::parseCustomPseudoSourceValue(
    StringRef Src, MachineFunction &MF, PerFunctionMIParsingState &PFS,
    const PseudoSourceValue *&PSV, ErrorCallbackType ErrorCallback) const {
  RVMachineFunctionInfo *MFI = MF.getInfo<RVMachineFunctionInfo>();
  const RVGPUTargetMachine &TM =
      static_cast<const RVGPUTargetMachine &>(MF.getTarget());
  if (Src == "GWSResource") {
    PSV = MFI->getGWSPSV(TM);
    return false;
  }
  llvm_unreachable("unknown MIR custom pseudo source value");
}
