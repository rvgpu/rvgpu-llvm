//===- RVGPUSubtarget.cpp - RVGPU Subtarget Information -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RVGPU specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "RVGPUSubtarget.h"
#include "RVGPUTargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "rvgpu-subtarget"

#define GET_SUBTARGETINFO_ENUM
#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "RVGPUGenSubtargetInfo.inc"

static cl::opt<bool>
    NoF16Math("rvgpu-no-f16-math", cl::Hidden,
              cl::desc("RVGPU Specific: Disable generation of f16 math ops."),
              cl::init(false));
// Pin the vtable to this file.
void RVGPUSubtarget::anchor() {}

RVGPUSubtarget &RVGPUSubtarget::initializeSubtargetDependencies(StringRef CPU,
                                                                StringRef FS) {
    // Provide the default CPU if we don't have one.
    TargetName = std::string(CPU.empty() ? "sm_30" : CPU);

    ParseSubtargetFeatures(TargetName, /*TuneCPU*/ TargetName, FS);

    // Re-map SM version numbers, SmVersion carries the regular SMs which do
    // have relative order, while FullSmVersion allows distinguishing sm_90 from
    // sm_90a, which would *not* be a subset of sm_91.
    SmVersion = getSmVersion();

    // Set default to PTX 6.0 (CUDA 9.0)
    if (PTXVersion == 0) {
      PTXVersion = 60;
  }

  return *this;
}

RVGPUSubtarget::RVGPUSubtarget(const Triple &TT, const std::string &CPU,
                               const std::string &FS,
                               const RVGPUTargetMachine &TM)
    : RVGPUGenSubtargetInfo(TT, CPU, /*TuneCPU*/ CPU, FS), PTXVersion(0),
      FullSmVersion(200), SmVersion(getSmVersion()), TM(TM),
      TLInfo(TM, initializeSubtargetDependencies(CPU, FS)) {}

bool RVGPUSubtarget::hasImageHandles() const {
  // Enable handles for Kepler+, where CUDA supports indirect surfaces and
  // textures
  if (TM.getDrvInterface() == RVGPU::CUDA)
    return (SmVersion >= 30);

  // Disabled, otherwise
  return false;
}

bool RVGPUSubtarget::allowFP16Math() const {
  return hasFP16Math() && NoF16Math == false;
}

uint64_t RVGPUSubtarget::getExplicitKernArgSize(const Function &F,
                                                 Align &MaxAlign) const {
  const DataLayout &DL = F.getParent()->getDataLayout();
  uint64_t ExplicitArgBytes = 0;
  MaxAlign = Align(1);

  for (const Argument &Arg : F.args()) {
    const bool IsByRef = Arg.hasByRefAttr();
    Type *ArgTy = IsByRef ? Arg.getParamByRefType() : Arg.getType();
    Align Alignment = DL.getValueOrABITypeAlignment(
        IsByRef ? Arg.getParamAlign() : std::nullopt, ArgTy);
    uint64_t AllocSize = DL.getTypeAllocSize(ArgTy);
    ExplicitArgBytes = alignTo(ExplicitArgBytes, Alignment) + AllocSize;
    MaxAlign = std::max(MaxAlign, Alignment);
  }

  return ExplicitArgBytes;
}

unsigned RVGPUSubtarget::getImplicitArgNumBytes(const Function &F) const {

  // We don't allocate the segment if we know the implicit arguments weren't
  // used, even if the ABI implies we need them.
  if (F.hasFnAttribute("rvgpu-no-implicitarg-ptr"))
    return 0;

  // Assume all implicit inputs are used by default
  const Module *M = F.getParent();
  unsigned NBytes = 256;
  return F.getFnAttributeAsParsedInteger("rvgpu-implicitarg-num-bytes",
                                         NBytes);
}

unsigned RVGPUSubtarget::getKernArgSegmentSize(const Function &F,
                                                Align &MaxAlign) const {
  uint64_t ExplicitArgBytes = getExplicitKernArgSize(F, MaxAlign);
  unsigned ExplicitOffset = 0;

  uint64_t TotalSize = ExplicitOffset + ExplicitArgBytes;
  unsigned ImplicitBytes = getImplicitArgNumBytes(F);
  if (ImplicitBytes != 0) {
    const Align Alignment = getAlignmentForImplicitArgPtr();
    TotalSize = alignTo(ExplicitArgBytes, Alignment) + ImplicitBytes;
    MaxAlign = std::max(MaxAlign, Alignment);
  }

  // Being able to dereference past the end is useful for emitting scalar loads.
  return alignTo(TotalSize, 4);
}
