//===-- RVGPUTargetMachine.h - Define TargetMachine for RVGPU ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the RVGPU specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RVGPU_RVGPUTARGETMACHINE_H
#define LLVM_LIB_TARGET_RVGPU_RVGPUTARGETMACHINE_H

#include "RVGPUSubtarget.h"
#include "llvm/Target/TargetMachine.h"
#include <optional>
#include <utility>

namespace llvm {

/// RVGPUTargetMachine
///
class RVGPUTargetMachine : public LLVMTargetMachine {
  bool is64bit;
  // Use 32-bit pointers for accessing const/local/short AS.
  bool UseShortPointers;
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  RVGPU::DrvInterface drvInterface;
  RVGPUSubtarget Subtarget;

  // Hold Strings that can be free'd all together with RVGPUTargetMachine
  BumpPtrAllocator StrAlloc;
  UniqueStringSaver StrPool;

public:
  RVGPUTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                     StringRef FS, const TargetOptions &Options,
                     std::optional<Reloc::Model> RM,
                     std::optional<CodeModel::Model> CM, CodeGenOptLevel OP,
                     bool is64bit);
  ~RVGPUTargetMachine() override;
  const RVGPUSubtarget *getSubtargetImpl(const Function &) const override {
    return &Subtarget;
  }
  const RVGPUSubtarget *getSubtargetImpl() const { return &Subtarget; }
  bool is64Bit() const { return is64bit; }
  bool useShortPointers() const { return UseShortPointers; }
  RVGPU::DrvInterface getDrvInterface() const { return drvInterface; }
  UniqueStringSaver &getStrPool() const {
    return const_cast<UniqueStringSaver &>(StrPool);
  }

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  // Emission of machine code through MCJIT is not supported.
  bool addPassesToEmitMC(PassManagerBase &, MCContext *&, raw_pwrite_stream &,
                         bool = true) override {
    return true;
  }
  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }

  MachineFunctionInfo *
  createMachineFunctionInfo(BumpPtrAllocator &Allocator, const Function &F,
                            const TargetSubtargetInfo *STI) const override;

  void registerDefaultAliasAnalyses(AAManager &AAM) override;

  void registerPassBuilderCallbacks(PassBuilder &PB) override;

  TargetTransformInfo getTargetTransformInfo(const Function &F) const override;

  bool isMachineVerifierClean() const override {
    return false;
  }

  std::pair<const Value *, unsigned>
  getPredicatedAddrSpace(const Value *V) const override;
}; // RVGPUTargetMachine.

class RVGPUTargetMachine32 : public RVGPUTargetMachine {
  virtual void anchor();

public:
  RVGPUTargetMachine32(const Target &T, const Triple &TT, StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       std::optional<Reloc::Model> RM,
                       std::optional<CodeModel::Model> CM, CodeGenOptLevel OL,
                       bool JIT);
};

class RVGPUTargetMachine64 : public RVGPUTargetMachine {
  virtual void anchor();

public:
  RVGPUTargetMachine64(const Target &T, const Triple &TT, StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       std::optional<Reloc::Model> RM,
                       std::optional<CodeModel::Model> CM, CodeGenOptLevel OL,
                       bool JIT);
};

} // end namespace llvm

#endif
