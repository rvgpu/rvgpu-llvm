//===- RVGPURegisterInfo.cpp - RVGPU Register Information -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the RVGPU implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "RVGPURegisterInfo.h"
#include "RVGPU.h"
#include "RVGPUSubtarget.h"
#include "RVGPUTargetMachine.h"
#include "MCTargetDesc/RVGPUMCTargetDesc.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/MC/MachineLocation.h"

using namespace llvm;

#define DEBUG_TYPE "rvgpu-reg-info"

namespace llvm {
std::string getRVGPURegClassName(TargetRegisterClass const *RC) {
  if (RC == &RVGPU::GPR16RegClass)
    return ".b16";
  if (RC == &RVGPU::GPR32RegClass)
    return ".b32";
  if (RC == &RVGPU::GPR64RegClass)
    return ".b64";
  if (RC == &RVGPU::Int1RegsRegClass)
    return ".pred";
  if (RC == &RVGPU::SpecialRegsRegClass)
    return "!Special!";
  return "INTERNAL";
}

std::string getRVGPURegClassStr(TargetRegisterClass const *RC) {
  if (RC == &RVGPU::GPR64RegClass)
    return "%rd";
  if (RC == &RVGPU::GPR32RegClass)
    return "%r";
  if (RC == &RVGPU::GPR16RegClass)
    return "%rs";
  if (RC == &RVGPU::Int1RegsRegClass)
    return "%p";
  if (RC == &RVGPU::SpecialRegsRegClass)
    return "!Special!";
  return "INTERNAL";
}
}

RVGPURegisterInfo::RVGPURegisterInfo()
    : RVGPUGenRegisterInfo(0), StrPool(StrAlloc) {}

#define GET_REGINFO_TARGET_DESC
#include "RVGPUGenRegisterInfo.inc"

/// RVGPU Callee Saved Registers
const MCPhysReg *
RVGPURegisterInfo::getCalleeSavedRegs(const MachineFunction *) const {
  static const MCPhysReg CalleeSavedRegs[] = { 0 };
  return CalleeSavedRegs;
}

BitVector RVGPURegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
#if 0 
  for (unsigned Reg = RVGPU::ENVREG0; Reg <= RVGPU::ENVREG31; ++Reg) {
    markSuperRegs(Reserved, Reg);
  }
  markSuperRegs(Reserved, RVGPU::VRFrame32);
  markSuperRegs(Reserved, RVGPU::VRFrameLocal32);
  markSuperRegs(Reserved, RVGPU::VRFrame64);
  markSuperRegs(Reserved, RVGPU::VRFrameLocal64);
  markSuperRegs(Reserved, RVGPU::VRDepot);
#endif   
  return Reserved;
}

bool RVGPURegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                            int SPAdj, unsigned FIOperandNum,
                                            RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected");

  MachineInstr &MI = *II;
  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();

  MachineFunction &MF = *MI.getParent()->getParent();
  int Offset = MF.getFrameInfo().getObjectOffset(FrameIndex) +
               MI.getOperand(FIOperandNum + 1).getImm();

  // Using I0 as the frame pointer
  MI.getOperand(FIOperandNum).ChangeToRegister(getFrameRegister(MF), false);
  MI.getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
  return false;
}

Register RVGPURegisterInfo::getFrameRegister(const MachineFunction &MF) const {
#if 0
  const RVGPUTargetMachine &TM =
      static_cast<const RVGPUTargetMachine &>(MF.getTarget());
  return TM.is64Bit() ? RVGPU::VRFrame64 : RVGPU::VRFrame32;
#endif
  return RVGPU::SP_REG;
}

Register
RVGPURegisterInfo::getFrameLocalRegister(const MachineFunction &MF) const {
  return RVGPU::SP_REG;
}
