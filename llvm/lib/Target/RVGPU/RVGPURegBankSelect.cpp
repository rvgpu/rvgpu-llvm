//===- RVGPURegBankSelect.cpp -----------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Use MachineUniformityAnalysis as the primary basis for making SGPR vs. VGPR
// register bank selection. Use/def analysis as in the default RegBankSelect can
// be useful in narrower circumstances (e.g. choosing AGPR vs. VGPR for gfx908).
//
//===----------------------------------------------------------------------===//

#include "RVGPURegBankSelect.h"
#include "RVGPU.h"
#include "RVSubtarget.h"
#include "llvm/CodeGen/MachineUniformityAnalysis.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "regbankselect"

using namespace llvm;

RVGPURegBankSelect::RVGPURegBankSelect(Mode RunningMode)
    : RegBankSelect(RVGPURegBankSelect::ID, RunningMode) {}

char RVGPURegBankSelect::ID = 0;

StringRef RVGPURegBankSelect::getPassName() const {
  return "RVGPURegBankSelect";
}

void RVGPURegBankSelect::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineCycleInfoWrapperPass>();
  AU.addRequired<MachineDominatorTree>();
  // TODO: Preserve DomTree
  RegBankSelect::getAnalysisUsage(AU);
}

INITIALIZE_PASS_BEGIN(RVGPURegBankSelect, "rvgpu-" DEBUG_TYPE,
                      "RVGPU Register Bank Select", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineCycleInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_END(RVGPURegBankSelect, "rvgpu-" DEBUG_TYPE,
                    "RVGPU Register Bank Select", false, false)

bool RVGPURegBankSelect::runOnMachineFunction(MachineFunction &MF) {
  // If the ISel pipeline failed, do not bother running that pass.
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;

  LLVM_DEBUG(dbgs() << "Assign register banks for: " << MF.getName() << '\n');
  const Function &F = MF.getFunction();
  Mode SaveOptMode = OptMode;
  if (F.hasOptNone())
    OptMode = Mode::Fast;
  init(MF);

  assert(checkFunctionIsLegal(MF));

  const RVSubtarget &ST = MF.getSubtarget<RVSubtarget>();
  MachineCycleInfo &CycleInfo =
      getAnalysis<MachineCycleInfoWrapperPass>().getCycleInfo();
  MachineDominatorTree &DomTree = getAnalysis<MachineDominatorTree>();

  MachineUniformityInfo Uniformity =
      computeMachineUniformityInfo(MF, CycleInfo, DomTree.getBase(),
                                   !ST.isSingleLaneExecution(F));
  (void)Uniformity; // TODO: Use this

  assignRegisterBanks(MF);

  OptMode = SaveOptMode;
  return false;
}
