//===-- RVGPUAlwaysInlinePass.cpp - Promote Allocas ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass marks all internal functions as always_inline and creates
/// duplicates of all other functions and marks the duplicates as always_inline.
//
//===----------------------------------------------------------------------===//

#include "RVGPU.h"
#include "RVGPUTargetMachine.h"
#include "Utils/RVGPUBaseInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

namespace {

static cl::opt<bool> StressCalls(
  "rvgpu-stress-function-calls",
  cl::Hidden,
  cl::desc("Force all functions to be noinline"),
  cl::init(false));

class RVGPUAlwaysInline : public ModulePass {
  bool GlobalOpt;

public:
  static char ID;

  RVGPUAlwaysInline(bool GlobalOpt = false) :
    ModulePass(ID), GlobalOpt(GlobalOpt) { }
  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
 }
};

} // End anonymous namespace

INITIALIZE_PASS(RVGPUAlwaysInline, "rvgpu-always-inline",
                "RVGPU Inline All Functions", false, false)

char RVGPUAlwaysInline::ID = 0;

static void
recursivelyVisitUsers(GlobalValue &GV,
                      SmallPtrSetImpl<Function *> &FuncsToAlwaysInline) {
  SmallVector<User *, 16> Stack(GV.users());

  SmallPtrSet<const Value *, 8> Visited;

  while (!Stack.empty()) {
    User *U = Stack.pop_back_val();
    if (!Visited.insert(U).second)
      continue;

    if (Instruction *I = dyn_cast<Instruction>(U)) {
      Function *F = I->getParent()->getParent();
      if (!RVGPU::isEntryFunctionCC(F->getCallingConv())) {
        // FIXME: This is a horrible hack. We should always respect noinline,
        // and just let us hit the error when we can't handle this.
        //
        // Unfortunately, clang adds noinline to all functions at -O0. We have
        // to override this here until that's fixed.
        F->removeFnAttr(Attribute::NoInline);

        FuncsToAlwaysInline.insert(F);
        Stack.push_back(F);
      }

      // No need to look at further users, but we do need to inline any callers.
      continue;
    }

    append_range(Stack, U->users());
  }
}

static bool alwaysInlineImpl(Module &M, bool GlobalOpt) {
  std::vector<GlobalAlias*> AliasesToRemove;

  SmallPtrSet<Function *, 8> FuncsToAlwaysInline;
  SmallPtrSet<Function *, 8> FuncsToNoInline;
  Triple TT(M.getTargetTriple());

  for (GlobalAlias &A : M.aliases()) {
    if (Function* F = dyn_cast<Function>(A.getAliasee())) {
      if (TT.getArch() == Triple::rvgpu &&
          A.getLinkage() != GlobalValue::InternalLinkage)
        continue;
      A.replaceAllUsesWith(F);
      AliasesToRemove.push_back(&A);
    }

    // FIXME: If the aliasee isn't a function, it's some kind of constant expr
    // cast that won't be inlined through.
  }

  if (GlobalOpt) {
    for (GlobalAlias* A : AliasesToRemove) {
      A->eraseFromParent();
    }
  }

  // Always force inlining of any function that uses an LDS global address. This
  // is something of a workaround because we don't have a way of supporting LDS
  // objects defined in functions. LDS is always allocated by a kernel, and it
  // is difficult to manage LDS usage if a function may be used by multiple
  // kernels.
  //
  // OpenCL doesn't allow declaring LDS in non-kernels, so in practice this
  // should only appear when IPO passes manages to move LDs defined in a kernel
  // into a single user function.

  for (GlobalVariable &GV : M.globals()) {
    // TODO: Region address
    unsigned AS = GV.getAddressSpace();
    if ((AS == RVGPUAS::REGION_ADDRESS) ||
        (AS == RVGPUAS::LOCAL_ADDRESS &&
         (!RVGPUTargetMachine::EnableLowerModuleLDS)))
      recursivelyVisitUsers(GV, FuncsToAlwaysInline);
  }

  if (!RVGPUTargetMachine::EnableFunctionCalls || StressCalls) {
    auto IncompatAttr
      = StressCalls ? Attribute::AlwaysInline : Attribute::NoInline;

    for (Function &F : M) {
      if (!F.isDeclaration() && !F.use_empty() &&
          !F.hasFnAttribute(IncompatAttr)) {
        if (StressCalls) {
          if (!FuncsToAlwaysInline.count(&F))
            FuncsToNoInline.insert(&F);
        } else
          FuncsToAlwaysInline.insert(&F);
      }
    }
  }

  for (Function *F : FuncsToAlwaysInline)
    F->addFnAttr(Attribute::AlwaysInline);

  for (Function *F : FuncsToNoInline)
    F->addFnAttr(Attribute::NoInline);

  return !FuncsToAlwaysInline.empty() || !FuncsToNoInline.empty();
}

bool RVGPUAlwaysInline::runOnModule(Module &M) {
  return alwaysInlineImpl(M, GlobalOpt);
}

ModulePass *llvm::createRVGPUAlwaysInlinePass(bool GlobalOpt) {
  return new RVGPUAlwaysInline(GlobalOpt);
}

PreservedAnalyses RVGPUAlwaysInlinePass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  alwaysInlineImpl(M, GlobalOpt);
  return PreservedAnalyses::all();
}
