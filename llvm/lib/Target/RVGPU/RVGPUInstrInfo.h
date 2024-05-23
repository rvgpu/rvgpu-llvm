//===- RVGPUInstrInfo.h - RVGPU Instruction Information----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the RVGPU implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RVGPU_RVGPUINSTRINFO_H
#define LLVM_LIB_TARGET_RVGPU_RVGPUINSTRINFO_H

#include "RVGPU.h"
#include "RVGPURegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "RVGPUGenInstrInfo.inc"

namespace llvm {

class RVGPUInstrInfo : public RVGPUGenInstrInfo {
  const RVGPURegisterInfo RegInfo;
  virtual void anchor();
public:
   enum TargetOperandFlags {
    MO_MASK = 0xf,

    MO_NONE = 0,
    // MO_GOTPCREL -> symbol@GOTPCREL -> R_AMDGPU_GOTPCREL.
    MO_GOTPCREL = 1,
    // MO_GOTPCREL32_LO -> symbol@gotpcrel32@lo -> R_AMDGPU_GOTPCREL32_LO.
    MO_GOTPCREL32 = 2,
    MO_GOTPCREL32_LO = 2,
    // MO_GOTPCREL32_HI -> symbol@gotpcrel32@hi -> R_AMDGPU_GOTPCREL32_HI.
    MO_GOTPCREL32_HI = 3,
    // MO_REL32_LO -> symbol@rel32@lo -> R_AMDGPU_REL32_LO.
    MO_REL32 = 4,
    MO_REL32_LO = 4,
    // MO_REL32_HI -> symbol@rel32@hi -> R_AMDGPU_REL32_HI.
    MO_REL32_HI = 5,

    MO_FAR_BRANCH_OFFSET = 6,

    MO_ABS32_LO = 8,
    MO_ABS32_HI = 9,
  };
  explicit RVGPUInstrInfo();

  const RVGPURegisterInfo &getRegisterInfo() const { return RegInfo; }

  /* The following virtual functions are used in register allocation.
   * They are not implemented because the existing interface and the logic
   * at the caller side do not work for the elementized vector load and store.
   *
   * virtual unsigned isLoadFromStackSlot(const MachineInstr *MI,
   *                                  int &FrameIndex) const;
   * virtual unsigned isStoreToStackSlot(const MachineInstr *MI,
   *                                 int &FrameIndex) const;
   * virtual void storeRegToStackSlot(MachineBasicBlock &MBB,
   *                              MachineBasicBlock::iterator MBBI,
   *                             unsigned SrcReg, bool isKill, int FrameIndex,
   *                              const TargetRegisterClass *RC,
   *                              Register VReg) const;
   * virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
   *                               MachineBasicBlock::iterator MBBI,
   *                               unsigned DestReg, int FrameIndex,
   *                               const TargetRegisterClass *RC,
   *                               const TargetRegisterInfo *TRI,
   *                               Register VReg) const;
   */

  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                   const DebugLoc &DL, MCRegister DestReg, MCRegister SrcReg,
                   bool KillSrc) const override;

  // Branch analysis.
  bool analyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                     MachineBasicBlock *&FBB,
                     SmallVectorImpl<MachineOperand> &Cond,
                     bool AllowModify) const override;
  unsigned removeBranch(MachineBasicBlock &MBB,
                        int *BytesRemoved = nullptr) const override;
  unsigned insertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                        MachineBasicBlock *FBB, ArrayRef<MachineOperand> Cond,
                        const DebugLoc &DL,
                        int *BytesAdded = nullptr) const override;
    
  LLVM_READONLY
  MachineOperand* getNamedOperand(MachineInstr &MI, unsigned OperandName) const;

  LLVM_READONLY
  const MachineOperand *getNamedOperand(const MachineInstr &MI,
                                        unsigned OpName) const {
    return getNamedOperand(const_cast<MachineInstr &>(MI), OpName);
  }
};

} // namespace llvm

#endif
