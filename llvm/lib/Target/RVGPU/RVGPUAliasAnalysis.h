//===-------------------- RVGPUAliasAnalysis.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This is the RVGPU address space based alias analysis pass.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RVGPU_RVGPUALIASANALYSIS_H
#define LLVM_LIB_TARGET_RVGPU_RVGPUALIASANALYSIS_H

#include "llvm/Analysis/AliasAnalysis.h"

namespace llvm {

class MemoryLocation;

class RVGPUAAResult : public AAResultBase {
public:
  RVGPUAAResult() {}
  RVGPUAAResult(RVGPUAAResult &&Arg) : AAResultBase(std::move(Arg)) {}

  /// Handle invalidation events from the new pass manager.
  ///
  /// By definition, this result is stateless and so remains valid.
  bool invalidate(Function &, const PreservedAnalyses &,
                  FunctionAnalysisManager::Invalidator &Inv) {
    return false;
  }

  AliasResult alias(const MemoryLocation &LocA, const MemoryLocation &LocB,
                    AAQueryInfo &AAQI, const Instruction *CtxI = nullptr);

  ModRefInfo getModRefInfoMask(const MemoryLocation &Loc, AAQueryInfo &AAQI,
                               bool IgnoreLocals);
};

/// Analysis pass providing a never-invalidated alias analysis result.
class RVGPUAA : public AnalysisInfoMixin<RVGPUAA> {
  friend AnalysisInfoMixin<RVGPUAA>;

  static AnalysisKey Key;

public:
  using Result = RVGPUAAResult;

  RVGPUAAResult run(Function &F, AnalysisManager<Function> &AM) {
    return RVGPUAAResult();
  }
};

/// Legacy wrapper pass to provide the RVGPUAAResult object.
class RVGPUAAWrapperPass : public ImmutablePass {
  std::unique_ptr<RVGPUAAResult> Result;

public:
  static char ID;

  RVGPUAAWrapperPass();

  RVGPUAAResult &getResult() { return *Result; }
  const RVGPUAAResult &getResult() const { return *Result; }

  bool doInitialization(Module &M) override {
    Result.reset(new RVGPUAAResult());
    return false;
  }

  bool doFinalization(Module &M) override {
    Result.reset();
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

// Wrapper around ExternalAAWrapperPass so that the default
// constructor gets the callback.
class RVGPUExternalAAWrapper : public ExternalAAWrapperPass {
public:
  static char ID;

  RVGPUExternalAAWrapper()
      : ExternalAAWrapperPass([](Pass &P, Function &, AAResults &AAR) {
          if (auto *WrapperPass =
                  P.getAnalysisIfAvailable<RVGPUAAWrapperPass>())
            AAR.addAAResult(WrapperPass->getResult());
        }) {}
};

ImmutablePass *createRVGPUAAWrapperPass();
void initializeRVGPUAAWrapperPassPass(PassRegistry &);
ImmutablePass *createRVGPUExternalAAWrapperPass();
void initializeRVGPUExternalAAWrapperPass(PassRegistry &);

} // end namespace llvm

#endif // LLVM_LIB_TARGET_RVGPU_RVGPUALIASANALYSIS_H
