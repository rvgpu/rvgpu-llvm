//===-- RVGPUPromoteKernelArguments.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass recursively promotes generic pointer arguments of a kernel
/// into the global address space.
///
/// The pass walks kernel's pointer arguments, then loads from them. If a loaded
/// value is a pointer and loaded pointer is unmodified in the kernel before the
/// load, then promote loaded pointer to global. Then recursively continue.
//
//===----------------------------------------------------------------------===//

#include "RVGPU.h"
#include "Utils/RVGPUMemoryUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "rvgpu-promote-kernel-arguments"

using namespace llvm;

namespace {

class RVGPUPromoteKernelArguments : public FunctionPass {
  MemorySSA *MSSA;

  AliasAnalysis *AA;

  Instruction *ArgCastInsertPt;

  SmallVector<Value *> Ptrs;

  void enqueueUsers(Value *Ptr);

  bool promotePointer(Value *Ptr);

  bool promoteLoad(LoadInst *LI);

public:
  static char ID;

  RVGPUPromoteKernelArguments() : FunctionPass(ID) {}

  bool run(Function &F, MemorySSA &MSSA, AliasAnalysis &AA);

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<MemorySSAWrapperPass>();
    AU.setPreservesAll();
  }
};

} // end anonymous namespace

void RVGPUPromoteKernelArguments::enqueueUsers(Value *Ptr) {
  SmallVector<User *> PtrUsers(Ptr->users());

  while (!PtrUsers.empty()) {
    Instruction *U = dyn_cast<Instruction>(PtrUsers.pop_back_val());
    if (!U)
      continue;

    switch (U->getOpcode()) {
    default:
      break;
    case Instruction::Load: {
      LoadInst *LD = cast<LoadInst>(U);
      if (LD->getPointerOperand()->stripInBoundsOffsets() == Ptr &&
          !RVGPU::isClobberedInFunction(LD, MSSA, AA))
        Ptrs.push_back(LD);

      break;
    }
    case Instruction::GetElementPtr:
    case Instruction::AddrSpaceCast:
    case Instruction::BitCast:
      if (U->getOperand(0)->stripInBoundsOffsets() == Ptr)
        PtrUsers.append(U->user_begin(), U->user_end());
      break;
    }
  }
}

bool RVGPUPromoteKernelArguments::promotePointer(Value *Ptr) {
  bool Changed = false;

  LoadInst *LI = dyn_cast<LoadInst>(Ptr);
  if (LI)
    Changed |= promoteLoad(LI);

  PointerType *PT = dyn_cast<PointerType>(Ptr->getType());
  if (!PT)
    return Changed;

  if (PT->getAddressSpace() == RVGPUAS::FLAT_ADDRESS ||
      PT->getAddressSpace() == RVGPUAS::GLOBAL_ADDRESS ||
      PT->getAddressSpace() == RVGPUAS::CONSTANT_ADDRESS)
    enqueueUsers(Ptr);

  if (PT->getAddressSpace() != RVGPUAS::FLAT_ADDRESS)
    return Changed;

  IRBuilder<> B(LI ? &*std::next(cast<Instruction>(Ptr)->getIterator())
                   : ArgCastInsertPt);

  // Cast pointer to global address space and back to flat and let
  // Infer Address Spaces pass to do all necessary rewriting.
  PointerType *NewPT =
      PointerType::get(PT->getContext(), RVGPUAS::GLOBAL_ADDRESS);
  Value *Cast =
      B.CreateAddrSpaceCast(Ptr, NewPT, Twine(Ptr->getName(), ".global"));
  Value *CastBack =
      B.CreateAddrSpaceCast(Cast, PT, Twine(Ptr->getName(), ".flat"));
  Ptr->replaceUsesWithIf(CastBack,
                         [Cast](Use &U) { return U.getUser() != Cast; });

  return true;
}

bool RVGPUPromoteKernelArguments::promoteLoad(LoadInst *LI) {
  if (!LI->isSimple())
    return false;

  LI->setMetadata("rvgpu.noclobber", MDNode::get(LI->getContext(), {}));
  return true;
}

// skip allocas
static BasicBlock::iterator getInsertPt(BasicBlock &BB) {
  BasicBlock::iterator InsPt = BB.getFirstInsertionPt();
  for (BasicBlock::iterator E = BB.end(); InsPt != E; ++InsPt) {
    AllocaInst *AI = dyn_cast<AllocaInst>(&*InsPt);

    // If this is a dynamic alloca, the value may depend on the loaded kernargs,
    // so loads will need to be inserted before it.
    if (!AI || !AI->isStaticAlloca())
      break;
  }

  return InsPt;
}

bool RVGPUPromoteKernelArguments::run(Function &F, MemorySSA &MSSA,
                                       AliasAnalysis &AA) {
  if (skipFunction(F))
    return false;

  CallingConv::ID CC = F.getCallingConv();
  if (CC != CallingConv::RVGPU_KERNEL || F.arg_empty())
    return false;

  ArgCastInsertPt = &*getInsertPt(*F.begin());
  this->MSSA = &MSSA;
  this->AA = &AA;

  for (Argument &Arg : F.args()) {
    if (Arg.use_empty())
      continue;

    PointerType *PT = dyn_cast<PointerType>(Arg.getType());
    if (!PT || (PT->getAddressSpace() != RVGPUAS::FLAT_ADDRESS &&
                PT->getAddressSpace() != RVGPUAS::GLOBAL_ADDRESS &&
                PT->getAddressSpace() != RVGPUAS::CONSTANT_ADDRESS))
      continue;

    Ptrs.push_back(&Arg);
  }

  bool Changed = false;
  while (!Ptrs.empty()) {
    Value *Ptr = Ptrs.pop_back_val();
    Changed |= promotePointer(Ptr);
  }

  return Changed;
}

bool RVGPUPromoteKernelArguments::runOnFunction(Function &F) {
  MemorySSA &MSSA = getAnalysis<MemorySSAWrapperPass>().getMSSA();
  AliasAnalysis &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
  return run(F, MSSA, AA);
}

INITIALIZE_PASS_BEGIN(RVGPUPromoteKernelArguments, DEBUG_TYPE,
                      "RVGPU Promote Kernel Arguments", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MemorySSAWrapperPass)
INITIALIZE_PASS_END(RVGPUPromoteKernelArguments, DEBUG_TYPE,
                    "RVGPU Promote Kernel Arguments", false, false)

char RVGPUPromoteKernelArguments::ID = 0;

FunctionPass *llvm::createRVGPUPromoteKernelArgumentsPass() {
  return new RVGPUPromoteKernelArguments();
}

PreservedAnalyses
RVGPUPromoteKernelArgumentsPass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
  MemorySSA &MSSA = AM.getResult<MemorySSAAnalysis>(F).getMSSA();
  AliasAnalysis &AA = AM.getResult<AAManager>(F);
  if (RVGPUPromoteKernelArguments().run(F, MSSA, AA)) {
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    PA.preserve<MemorySSAAnalysis>();
    return PA;
  }
  return PreservedAnalyses::all();
}
