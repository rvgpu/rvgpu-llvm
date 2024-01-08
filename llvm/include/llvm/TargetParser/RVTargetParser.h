//===-- TargetParser - Parser for target features ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise hardware features such as
// FPU/CPU/ARCH names as well as specific support such as HDIV, etc.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_RVTARGETPARSER_H
#define LLVM_TARGETPARSER_RVTARGETPARSER_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

template <typename T> class SmallVectorImpl;
class Triple;

// Target specific information in their own namespaces.
// (ARM/AArch64/X86 are declared in ARM/AArch64/X86TargetParser.h)
// These should be generated from TableGen because the information is already
// there, and there is where new information about targets will be added.
// FIXME: To TableGen this we need to make some table generated files available
// even if the back-end is not compiled with LLVM, plus we need to create a new
// back-end to TableGen to create these clean tables.
namespace RVGPU {

/// GPU kinds supported by the RVGPU target.
enum GPUKind : uint32_t {
  // Not specified processor.
  GK_NONE = 0,

  GK_R1000 = 1,
  GK_RVGPU_FIRST = GK_R1000,
  GK_RVGPU_LAST ,
};

/// Instruction set architecture version.
struct IsaVersion {
  unsigned Major;
  unsigned Minor;
  unsigned Stepping;
};

// This isn't comprehensive for now, just things that are needed from the
// frontend driver.
enum ArchFeatureKind : uint32_t {
  FEATURE_NONE = 0,

  // These features only exist for r600, and are implied true for rvgpu.
  FEATURE_FMA = 1 << 1,
  FEATURE_LDEXP = 1 << 2,
  FEATURE_FP64 = 1 << 3,

  // Common features.
  FEATURE_FAST_FMA_F32 = 1 << 4,
  FEATURE_FAST_DENORMAL_F32 = 1 << 5,

  // Wavefront 32 is available.
  FEATURE_WAVE32 = 1 << 6,

  // Xnack is available.
  FEATURE_XNACK = 1 << 7,

  // Sram-ecc is available.
  FEATURE_SRAMECC = 1 << 8,

  // WGP mode is supported.
  FEATURE_WGP = 1 << 9,
};

StringRef getArchNameRVGPU(GPUKind AK);
StringRef getCanonicalArchName(const Triple &T, StringRef Arch);
GPUKind parseArchRVGPU(StringRef CPU);
unsigned getArchAttrRVGPU(GPUKind AK);

void fillValidArchListRVGPU(SmallVectorImpl<StringRef> &Values);

IsaVersion getIsaVersion(StringRef GPU);

/// Fills Features map with default values for given target GPU
void fillRVGPUFeatureMap(StringRef GPU, const Triple &T,
                          StringMap<bool> &Features);

/// Inserts wave size feature for given GPU into features map
bool insertWaveSizeFeature(StringRef GPU, const Triple &T,
                           StringMap<bool> &Features, std::string &ErrorMsg);

} // namespace RVGPU
} // namespace llvm

#endif
