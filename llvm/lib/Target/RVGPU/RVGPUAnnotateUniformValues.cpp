//===-- RVGPUAnnotateUniformValues.cpp - ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass adds rvgpu.uniform metadata to IR values so this information
/// can be used during instruction selection.
//
//===----------------------------------------------------------------------===//

#include "RVGPU.h"
#include "Utils/RVGPUBaseInfo.h"
#include "Utils/RVGPUMemoryUtils.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "rvgpu-annotate-uniform"

using namespace llvm;

namespace {

class RVGPUAnnotateUniformValues : public FunctionPass,
                       public InstVisitor<RVGPUAnnotateUniformValues> {
  UniformityInfo *UA;
  MemorySSA *MSSA;
  AliasAnalysis *AA;
  bool isEntryFunc;
  bool Changed;

  void setUniformMetadata(Instruction *I) {
    I->setMetadata("rvgpu.uniform", MDNode::get(I->getContext(), {}));
    Changed = true;
  }

  void setNoClobberMetadata(Instruction *I) {
    I->setMetadata("rvgpu.noclobber", MDNode::get(I->getContext(), {}));
    Changed = true;
  }

public:
  static char ID;
  RVGPUAnnotateUniformValues() :
    FunctionPass(ID) { }
  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;
  StringRef getPassName() const override {
    return "RVGPU Annotate Uniform Values";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<UniformityInfoWrapperPass>();
    AU.addRequired<MemorySSAWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.setPreservesAll();
 }

  void visitBranchInst(BranchInst &I);
  void visitLoadInst(LoadInst &I);
};

} // End anonymous namespace

INITIALIZE_PASS_BEGIN(RVGPUAnnotateUniformValues, DEBUG_TYPE,
                      "Add RVGPU uniform metadata", false, false)
INITIALIZE_PASS_DEPENDENCY(UniformityInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MemorySSAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(RVGPUAnnotateUniformValues, DEBUG_TYPE,
                    "Add RVGPU uniform metadata", false, false)

char RVGPUAnnotateUniformValues::ID = 0;

void RVGPUAnnotateUniformValues::visitBranchInst(BranchInst &I) {
  if (UA->isUniform(&I))
    setUniformMetadata(&I);
}

void RVGPUAnnotateUniformValues::visitLoadInst(LoadInst &I) {
  Value *Ptr = I.getPointerOperand();
  if (!UA->isUniform(Ptr))
    return;
  Instruction *PtrI = dyn_cast<Instruction>(Ptr);
  if (PtrI)
    setUniformMetadata(PtrI);

  // We're tracking up to the Function boundaries, and cannot go beyond because
  // of FunctionPass restrictions. We can ensure that is memory not clobbered
  // for memory operations that are live in to entry points only.
  if (!isEntryFunc)
    return;
  bool GlobalLoad = I.getPointerAddressSpace() == RVGPUAS::GLOBAL_ADDRESS;
  if (GlobalLoad && !RVGPU::isClobberedInFunction(&I, MSSA, AA))
    setNoClobberMetadata(&I);
}

bool RVGPUAnnotateUniformValues::doInitialization(Module &M) {
  return false;
}

bool RVGPUAnnotateUniformValues::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  UA = &getAnalysis<UniformityInfoWrapperPass>().getUniformityInfo();
  MSSA = &getAnalysis<MemorySSAWrapperPass>().getMSSA();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  isEntryFunc = RVGPU::isEntryFunctionCC(F.getCallingConv());

  Changed = false;
  visit(F);
  return Changed;
}

FunctionPass *
llvm::createRVGPUAnnotateUniformValues() {
  return new RVGPUAnnotateUniformValues();
}
