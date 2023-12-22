//===--- RVGPUMacroFusion.cpp - RVGPU Macro Fusion ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains the RVGPU implementation of the DAG scheduling
///  mutation to pair instructions back to back.
//
//===----------------------------------------------------------------------===//

#include "RVGPUMacroFusion.h"
#include "MCTargetDesc/RVGPUMCTargetDesc.h"
#include "RVInstrInfo.h"
#include "llvm/CodeGen/MacroFusion.h"

using namespace llvm;

namespace {

/// Check if the instr pair, FirstMI and SecondMI, should be fused
/// together. Given SecondMI, when FirstMI is unspecified, then check if
/// SecondMI may be part of a fused pair at all.
static bool shouldScheduleAdjacent(const TargetInstrInfo &TII_,
                                   const TargetSubtargetInfo &TSI,
                                   const MachineInstr *FirstMI,
                                   const MachineInstr &SecondMI) {
  const RVInstrInfo &TII = static_cast<const RVInstrInfo&>(TII_);

  switch (SecondMI.getOpcode()) {
  case RVGPU::V_ADDC_U32_e64:
  case RVGPU::V_SUBB_U32_e64:
  case RVGPU::V_SUBBREV_U32_e64:
  case RVGPU::V_CNDMASK_B32_e64: {
    // Try to cluster defs of condition registers to their uses. This improves
    // the chance VCC will be available which will allow shrinking to VOP2
    // encodings.
    if (!FirstMI)
      return true;

    const MachineBasicBlock &MBB = *FirstMI->getParent();
    const MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
    const TargetRegisterInfo *TRI = MRI.getTargetRegisterInfo();
    const MachineOperand *Src2 = TII.getNamedOperand(SecondMI,
                                                     RVGPU::OpName::src2);
    return FirstMI->definesRegister(Src2->getReg(), TRI);
  }
  default:
    return false;
  }

  return false;
}

} // end namespace


namespace llvm {

std::unique_ptr<ScheduleDAGMutation> createRVGPUMacroFusionDAGMutation() {
  return createMacroFusionDAGMutation(shouldScheduleAdjacent);
}

} // end namespace llvm
