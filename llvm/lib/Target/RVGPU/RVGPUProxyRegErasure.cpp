//===- RVGPUProxyRegErasure.cpp - RVGPU Proxy Register Instruction Erasure -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pass is needed to remove ProxyReg instructions and restore related
// registers. The instructions were needed at instruction selection stage to
// make sure that callseq_end nodes won't be removed as "dead nodes". This can
// happen when we expand instructions into libcalls and the call site doesn't
// care about the libcall chain. Call site cares about data flow only, and the
// latest data flow node happens to be before callseq_end. Therefore the node
// becomes dangling and "dead". The ProxyReg acts like an additional data flow
// node *after* the callseq_end in the chain and ensures that everything will be
// preserved.
//
//===----------------------------------------------------------------------===//

#include "RVGPU.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

using namespace llvm;

namespace llvm {
void initializeRVGPUProxyRegErasurePass(PassRegistry &);
}

namespace {

struct RVGPUProxyRegErasure : public MachineFunctionPass {
public:
  static char ID;
  RVGPUProxyRegErasure() : MachineFunctionPass(ID) {
    initializeRVGPUProxyRegErasurePass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "RVGPU Proxy Register Instruction Erasure";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  void replaceMachineInstructionUsage(MachineFunction &MF, MachineInstr &MI);

  void replaceRegisterUsage(MachineInstr &Instr, MachineOperand &From,
                            MachineOperand &To);
};

} // namespace

char RVGPUProxyRegErasure::ID = 0;

INITIALIZE_PASS(RVGPUProxyRegErasure, "rvgpu-proxyreg-erasure", "RVGPU ProxyReg Erasure", false, false)

bool RVGPUProxyRegErasure::runOnMachineFunction(MachineFunction &MF) {
  SmallVector<MachineInstr *, 16> RemoveList;

  for (auto &BB : MF) {
    for (auto &MI : BB) {
      switch (MI.getOpcode()) {
      case RVGPU::ProxyRegI1:
      case RVGPU::ProxyRegI16:
      case RVGPU::ProxyRegI32:
      case RVGPU::ProxyRegI64:
      case RVGPU::ProxyRegF32:
      case RVGPU::ProxyRegF64:
        replaceMachineInstructionUsage(MF, MI);
        RemoveList.push_back(&MI);
        break;
      }
    }
  }

  for (auto *MI : RemoveList) {
    MI->eraseFromParent();
  }

  return !RemoveList.empty();
}

void RVGPUProxyRegErasure::replaceMachineInstructionUsage(MachineFunction &MF,
                                                          MachineInstr &MI) {
  auto &InOp = *MI.uses().begin();
  auto &OutOp = *MI.defs().begin();

  assert(InOp.isReg() && "ProxyReg input operand should be a register.");
  assert(OutOp.isReg() && "ProxyReg output operand should be a register.");

  for (auto &BB : MF) {
    for (auto &I : BB) {
      replaceRegisterUsage(I, OutOp, InOp);
    }
  }
}

void RVGPUProxyRegErasure::replaceRegisterUsage(MachineInstr &Instr,
                                                MachineOperand &From,
                                                MachineOperand &To) {
  for (auto &Op : Instr.uses()) {
    if (Op.isReg() && Op.getReg() == From.getReg()) {
      Op.setReg(To.getReg());
    }
  }
}

MachineFunctionPass *llvm::createRVGPUProxyRegErasurePass() {
  return new RVGPUProxyRegErasure();
}
