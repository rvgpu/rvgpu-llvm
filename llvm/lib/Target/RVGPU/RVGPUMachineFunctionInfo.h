//===-- RVGPUMachineFunctionInfo.h - RVGPU-specific Function Info  --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class is attached to a MachineFunction instance and tracks target-
// dependent information
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RVGPU_RVGPUMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_RVGPU_RVGPUMACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"

#include "RVModeRegisterDefaults.h"
#include "Utils/RVGPUBaseInfo.h"

namespace llvm {
class RVGPUMachineFunctionInfo : public MachineFunctionInfo {
private:
  /// Stores a mapping from index to symbol name for removing image handles
  /// on Fermi.
  SmallVector<std::string, 8> ImageHandleList;
// Kernels + shaders. i.e. functions called by the hardware and not called
  // by other functions.
  bool IsEntryFunction = false;
  bool IsModuleEntryFunction = false;
  // Feature bits required for inputs passed in system SGPRs.
  bool WorkGroupIDX : 1; // Always initialized.
  bool WorkGroupIDY : 1;
  bool WorkGroupIDZ : 1;
  bool WorkGroupInfo : 1;
  bool LDSKernelId : 1;
  bool PrivateSegmentWaveByteOffset : 1;

  bool WorkItemIDX : 1; // Always initialized.
  bool WorkItemIDY : 1;
  bool WorkItemIDZ : 1;

  // State of MODE register, assumed FP mode.
  RVModeRegisterDefaults Mode;
public:
  RVGPUMachineFunctionInfo(const Function &F, const TargetSubtargetInfo *STI)
  :IsEntryFunction(RVGPU::isEntryFunctionCC(F.getCallingConv())),
   IsModuleEntryFunction(
          RVGPU::isModuleEntryFunctionCC(F.getCallingConv())),
  Mode(F, static_cast<const RVGPUSubtarget &>(*STI)){}

  MachineFunctionInfo *
  clone(BumpPtrAllocator &Allocator, MachineFunction &DestMF,
        const DenseMap<MachineBasicBlock *, MachineBasicBlock *> &Src2DstMBB)
      const override {
    return DestMF.cloneInfo<RVGPUMachineFunctionInfo>(*this);
  }

  /// Returns the index for the symbol \p Symbol. If the symbol was previously,
  /// added, the same index is returned. Otherwise, the symbol is added and the
  /// new index is returned.
  unsigned getImageHandleSymbolIndex(const char *Symbol) {
    // Is the symbol already present?
    for (unsigned i = 0, e = ImageHandleList.size(); i != e; ++i)
      if (ImageHandleList[i] == std::string(Symbol))
        return i;
    // Nope, insert it
    ImageHandleList.push_back(Symbol);
    return ImageHandleList.size()-1;
  }

  /// Returns the symbol name at the given index.
  const char *getImageHandleSymbol(unsigned Idx) const {
    assert(ImageHandleList.size() > Idx && "Bad index");
    return ImageHandleList[Idx].c_str();
  }

  bool isEntryFunction() const {
    //return IsEntryFunction;
    return true;
  }
  /// \returns Default/requested maximum number of waves per execution unit.
  unsigned getMaxWavesPerEU() const {
    return 16;
  }
  
  RVModeRegisterDefaults getMode() const { return Mode; }
  bool isModuleEntryFunction() const { return IsModuleEntryFunction; }

  bool hasWorkGroupIDX() const {
    return WorkGroupIDX;
  }

  bool hasWorkGroupIDY() const {
    return WorkGroupIDY;
  }

  bool hasWorkGroupIDZ() const {
    return WorkGroupIDZ;
  }

  bool hasWorkGroupInfo() const {
    return WorkGroupInfo;
  }

  bool hasPrivateSegmentWaveByteOffset() const {
    return PrivateSegmentWaveByteOffset;
  }

  bool hasWorkItemIDX() const {
    return WorkItemIDX;
  }

  bool hasWorkItemIDY() const {
    return WorkItemIDY;
  }

  bool hasWorkItemIDZ() const {
    return WorkItemIDZ;
  }
  static std::optional<uint32_t> getLDSAbsoluteAddress(const GlobalValue &GV) {
    if (GV.getAddressSpace() != RVGPUAS::LOCAL_ADDRESS)
    return {};

    std::optional<ConstantRange> AbsSymRange = GV.getAbsoluteSymbolRange();
    if (!AbsSymRange)
    return {};

    if (const APInt *V = AbsSymRange->getSingleElement()) {
        std::optional<uint64_t> ZExt = V->tryZExtValue();
        if (ZExt && (*ZExt <= UINT32_MAX)) {
            return *ZExt;
        }
    }

    return {};
  } 
};
}

#endif
