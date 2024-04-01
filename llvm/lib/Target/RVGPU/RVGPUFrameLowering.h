//===--- RVGPUFrameLowering.h - Define frame lowering for RVGPU -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RVGPU_RVGPUFRAMELOWERING_H
#define LLVM_LIB_TARGET_RVGPU_RVGPUFRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/Support/TypeSize.h"

namespace llvm {

class RVGPUFrameLowering : public TargetFrameLowering {
public:
  explicit RVGPUFrameLowering();

  bool hasFP(const MachineFunction &MF) const override;
  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  StackOffset getFrameIndexReference(const MachineFunction &MF, int FI,
                                     Register &FrameReg) const override;

  MachineBasicBlock::iterator
  eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator I) const override;
  DwarfFrameBase getDwarfFrameBase(const MachineFunction &MF) const override;
};

} // End llvm namespace

#endif
