//=======- RVGPUFrameLowering.cpp - RVGPU Frame Information ---*- C++ -*-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the RVGPU implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "RVGPUFrameLowering.h"
#include "RVGPU.h"
#include "RVGPURegisterInfo.h"
#include "RVGPUSubtarget.h"
#include "RVGPUTargetMachine.h"
#include "MCTargetDesc/RVGPUMCTargetDesc.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/MC/MachineLocation.h"

using namespace llvm;

RVGPUFrameLowering::RVGPUFrameLowering()
    : TargetFrameLowering(TargetFrameLowering::StackGrowsUp, Align(8), 0) {}

bool RVGPUFrameLowering::hasFP(const MachineFunction &MF) const { return true; }

void RVGPUFrameLowering::emitPrologue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {
  if (MF.getFrameInfo().hasStackObjects()) {
    assert(&MF.front() == &MBB && "Shrink-wrapping not yet supported");
    MachineBasicBlock::iterator MBBI = MBB.begin();
    MachineRegisterInfo &MR = MF.getRegInfo();

    const RVGPURegisterInfo *NRI =
        MF.getSubtarget<RVGPUSubtarget>().getRegisterInfo();

    // This instruction really occurs before first instruction
    // in the BB, so giving it no debug location.
    DebugLoc dl = DebugLoc();

    // Emits
    //   mov %SPL, %depot;
    //   cvta.local %SP, %SPL;
    // for local address accesses in MF.
    bool Is64Bit =
        static_cast<const RVGPUTargetMachine &>(MF.getTarget()).is64Bit();
    unsigned CvtaLocalOpcode =
        (Is64Bit ? RVGPU::cvta_local_yes_64 : RVGPU::cvta_local_yes);
    unsigned MovDepotOpcode =
        (Is64Bit ? RVGPU::MOV_DEPOT_ADDR_64 : RVGPU::MOV_DEPOT_ADDR);
    if (!MR.use_empty(NRI->getFrameRegister(MF))) {
      // If %SP is not used, do not bother emitting "cvta.local %SP, %SPL".
      MBBI = BuildMI(MBB, MBBI, dl,
                     MF.getSubtarget().getInstrInfo()->get(CvtaLocalOpcode),
                     NRI->getFrameRegister(MF))
                 .addReg(NRI->getFrameLocalRegister(MF));
    }
    BuildMI(MBB, MBBI, dl,
            MF.getSubtarget().getInstrInfo()->get(MovDepotOpcode),
            NRI->getFrameLocalRegister(MF))
        .addImm(MF.getFunctionNumber());
  }
}

StackOffset
RVGPUFrameLowering::getFrameIndexReference(const MachineFunction &MF, int FI,
                                           Register &FrameReg) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const RVGPURegisterInfo *RI = MF.getSubtarget<RVGPUSubtarget>().getRegisterInfo();
  FrameReg = RI->getFrameRegister(MF);
  return StackOffset::getFixed(MFI.getObjectOffset(FI) -
                               getOffsetOfLocalArea());
}

void RVGPUFrameLowering::emitEpilogue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {}

// This function eliminates ADJCALLSTACKDOWN,
// ADJCALLSTACKUP pseudo instructions
MachineBasicBlock::iterator RVGPUFrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator I) const {
  // Simply discard ADJCALLSTACKDOWN,
  // ADJCALLSTACKUP instructions.
  return MBB.erase(I);
}

TargetFrameLowering::DwarfFrameBase
RVGPUFrameLowering::getDwarfFrameBase(const MachineFunction &MF) const {
  return {DwarfFrameBase::CFA, {0}};
}
