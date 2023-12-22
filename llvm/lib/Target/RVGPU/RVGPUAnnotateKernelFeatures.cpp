//===- RVGPUAnnotateKernelFeaturesPass.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass propagates the uniform-work-group-size attribute from
/// kernels to leaf functions when possible. It also adds additional attributes
/// to hint ABI lowering optimizations later.
//
//===----------------------------------------------------------------------===//

#include "RVGPU.h"
#include "RVSubtarget.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IntrinsicsRVGPU.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "rvgpu-annotate-kernel-features"

using namespace llvm;

namespace {
class RVGPUAnnotateKernelFeatures : public CallGraphSCCPass {
private:
  const TargetMachine *TM = nullptr;

  bool addFeatureAttributes(Function &F);

public:
  static char ID;

  RVGPUAnnotateKernelFeatures() : CallGraphSCCPass(ID) {}

  bool doInitialization(CallGraph &CG) override;
  bool runOnSCC(CallGraphSCC &SCC) override;

  StringRef getPassName() const override {
    return "RVGPU Annotate Kernel Features";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    CallGraphSCCPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char RVGPUAnnotateKernelFeatures::ID = 0;

char &llvm::RVGPUAnnotateKernelFeaturesID = RVGPUAnnotateKernelFeatures::ID;

INITIALIZE_PASS(RVGPUAnnotateKernelFeatures, DEBUG_TYPE,
                "Add RVGPU function attributes", false, false)

bool RVGPUAnnotateKernelFeatures::addFeatureAttributes(Function &F) {
  bool HaveStackObjects = false;
  bool Changed = false;
  bool HaveCall = false;
  bool IsFunc = !RVGPU::isEntryFunctionCC(F.getCallingConv());

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (isa<AllocaInst>(I)) {
        HaveStackObjects = true;
        continue;
      }

      if (auto *CB = dyn_cast<CallBase>(&I)) {
        const Function *Callee =
            dyn_cast<Function>(CB->getCalledOperand()->stripPointerCasts());

        // Note the occurrence of indirect call.
        if (!Callee) {
          if (!CB->isInlineAsm())
            HaveCall = true;

          continue;
        }

        Intrinsic::ID IID = Callee->getIntrinsicID();
        if (IID == Intrinsic::not_intrinsic) {
          HaveCall = true;
          Changed = true;
        }
      }
    }
  }

  // TODO: We could refine this to captured pointers that could possibly be
  // accessed by flat instructions. For now this is mostly a poor way of
  // estimating whether there are calls before argument lowering.
  if (!IsFunc && HaveCall) {
    F.addFnAttr("rvgpu-calls");
    Changed = true;
  }

  if (HaveStackObjects) {
    F.addFnAttr("rvgpu-stack-objects");
    Changed = true;
  }

  return Changed;
}

bool RVGPUAnnotateKernelFeatures::runOnSCC(CallGraphSCC &SCC) {
  bool Changed = false;

  for (CallGraphNode *I : SCC) {
    Function *F = I->getFunction();
    // Ignore functions with graphics calling conventions, these are currently
    // not allowed to have kernel arguments.
    if (!F || F->isDeclaration() || RVGPU::isGraphics(F->getCallingConv()))
      continue;
    // Add feature attributes
    Changed |= addFeatureAttributes(*F);
  }

  return Changed;
}

bool RVGPUAnnotateKernelFeatures::doInitialization(CallGraph &CG) {
  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC)
    report_fatal_error("TargetMachine is required");

  TM = &TPC->getTM<TargetMachine>();
  return false;
}

Pass *llvm::createRVGPUAnnotateKernelFeaturesPass() {
  return new RVGPUAnnotateKernelFeatures();
}
