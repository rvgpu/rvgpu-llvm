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
  if (RC == &RVGPU::Float32RegsRegClass)
    return ".f32";
  if (RC == &RVGPU::Float64RegsRegClass)
    return ".f64";
  if (RC == &RVGPU::Int64RegsRegClass)
    // We use untyped (.b) integer registers here as NVCC does.
    // Correctness of generated code does not depend on register type,
    // but using .s/.u registers runs into ptxas bug that prevents
    // assembly of otherwise valid PTX into SASS. Despite PTX ISA
    // specifying only argument size for fp16 instructions, ptxas does
    // not allow using .s16 or .u16 arguments for .fp16
    // instructions. At the same time it allows using .s32/.u32
    // arguments for .fp16v2 instructions:
    //
    //   .reg .b16 rb16
    //   .reg .s16 rs16
    //   add.f16 rb16,rb16,rb16; // OK
    //   add.f16 rs16,rs16,rs16; // Arguments mismatch for instruction 'add'
    // but:
    //   .reg .b32 rb32
    //   .reg .s32 rs32
    //   add.f16v2 rb32,rb32,rb32; // OK
    //   add.f16v2 rs32,rs32,rs32; // OK
    return ".b64";
  if (RC == &RVGPU::Int32RegsRegClass)
    return ".b32";
  if (RC == &RVGPU::Int16RegsRegClass)
    return ".b16";
  if (RC == &RVGPU::Int1RegsRegClass)
    return ".pred";
  if (RC == &RVGPU::SpecialRegsRegClass)
    return "!Special!";
  return "INTERNAL";
}

std::string getRVGPURegClassStr(TargetRegisterClass const *RC) {
  if (RC == &RVGPU::Float32RegsRegClass)
    return "%f";
  if (RC == &RVGPU::Float64RegsRegClass)
    return "%fd";
  if (RC == &RVGPU::Int64RegsRegClass)
    return "%rd";
  if (RC == &RVGPU::Int32RegsRegClass)
    return "%r";
  if (RC == &RVGPU::Int16RegsRegClass)
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
  for (unsigned Reg = RVGPU::ENVREG0; Reg <= RVGPU::ENVREG31; ++Reg) {
    markSuperRegs(Reserved, Reg);
  }
  markSuperRegs(Reserved, RVGPU::VRFrame32);
  markSuperRegs(Reserved, RVGPU::VRFrameLocal32);
  markSuperRegs(Reserved, RVGPU::VRFrame64);
  markSuperRegs(Reserved, RVGPU::VRFrameLocal64);
  markSuperRegs(Reserved, RVGPU::VRDepot);
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
  const RVGPUTargetMachine &TM =
      static_cast<const RVGPUTargetMachine &>(MF.getTarget());
  return TM.is64Bit() ? RVGPU::VRFrame64 : RVGPU::VRFrame32;
}

Register
RVGPURegisterInfo::getFrameLocalRegister(const MachineFunction &MF) const {
  const RVGPUTargetMachine &TM =
      static_cast<const RVGPUTargetMachine &>(MF.getTarget());
  return TM.is64Bit() ? RVGPU::VRFrameLocal64 : RVGPU::VRFrameLocal32;
}
