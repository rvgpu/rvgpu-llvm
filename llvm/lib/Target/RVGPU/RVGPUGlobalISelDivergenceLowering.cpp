//===-- RVGPUGlobalISelDivergenceLowering.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// GlobalISel pass that selects divergent i1 phis as lane mask phis.
/// Lane mask merging uses same algorithm as SDAG in SILowerI1Copies.
/// Handles all cases of temporal divergence.
/// For divergent non-phi i1 and uniform i1 uses outside of the cycle this pass
/// currently depends on LCSSA to insert phis with one incoming.
//
//===----------------------------------------------------------------------===//

#include "RVGPU.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

#define DEBUG_TYPE "rvgpu-global-isel-divergence-lowering"

using namespace llvm;

namespace {

class RVGPUGlobalISelDivergenceLowering : public MachineFunctionPass {
public:
  static char ID;

public:
  RVGPUGlobalISelDivergenceLowering() : MachineFunctionPass(ID) {
    initializeRVGPUGlobalISelDivergenceLoweringPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "RVGPU GlobalISel divergence lowering";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(RVGPUGlobalISelDivergenceLowering, DEBUG_TYPE,
                      "RVGPU GlobalISel divergence lowering", false, false)
INITIALIZE_PASS_END(RVGPUGlobalISelDivergenceLowering, DEBUG_TYPE,
                    "RVGPU GlobalISel divergence lowering", false, false)

char RVGPUGlobalISelDivergenceLowering::ID = 0;

char &llvm::RVGPUGlobalISelDivergenceLoweringID =
    RVGPUGlobalISelDivergenceLowering::ID;

FunctionPass *llvm::createRVGPUGlobalISelDivergenceLoweringPass() {
  return new RVGPUGlobalISelDivergenceLowering();
}

bool RVGPUGlobalISelDivergenceLowering::runOnMachineFunction(
    MachineFunction &MF) {
  return false;
}
