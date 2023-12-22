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

#include "llvm/TargetParser/RVTargetParser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace RVGPU;

namespace {

struct GPUInfo {
  StringLiteral Name;
  StringLiteral CanonicalName;
  RVGPU::GPUKind Kind;
  unsigned Features;
};

// This table should be sorted by the value of GPUKind
// Don't bother listing the implicitly true features
constexpr GPUInfo RVGPUGPUs[] = {

    {{"ss1000"},   {"ss1000"}, GK_SS1000, FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_WAVE32|FEATURE_WGP},
    // clang-format on
};

const GPUInfo *getArchEntry(RVGPU::GPUKind AK, ArrayRef<GPUInfo> Table) {
  GPUInfo Search = { {""}, {""}, AK, RVGPU::FEATURE_NONE };

  auto I =
      llvm::lower_bound(Table, Search, [](const GPUInfo &A, const GPUInfo &B) {
        return A.Kind < B.Kind;
      });

  if (I == Table.end() || I->Kind != Search.Kind)
    return nullptr;
  return I;
}

} // namespace

StringRef llvm::RVGPU::getArchNameRVGPU(GPUKind AK) {
  if (const auto *Entry = getArchEntry(AK, RVGPUGPUs))
    return Entry->CanonicalName;
  return "";
}

RVGPU::GPUKind llvm::RVGPU::parseArchRVGPU(StringRef CPU) {
  for (const auto &C : RVGPUGPUs) {
    if (CPU == C.Name)
      return C.Kind;
  }

  return RVGPU::GPUKind::GK_NONE;
}

unsigned RVGPU::getArchAttrRVGPU(GPUKind AK) {
  if (const auto *Entry = getArchEntry(AK, RVGPUGPUs))
    return Entry->Features;
  return FEATURE_NONE;
}

void RVGPU::fillValidArchListRVGPU(SmallVectorImpl<StringRef> &Values) {
  // XXX: Should this only report unique canonical names?
  for (const auto &C : RVGPUGPUs)
    Values.push_back(C.Name);
}

RVGPU::IsaVersion RVGPU::getIsaVersion(StringRef GPU) {
  return {11, 0, 0};
}

StringRef RVGPU::getCanonicalArchName(const Triple &T, StringRef Arch) {
  assert(T.isRVGPU());
  auto ProcKind = parseArchRVGPU(Arch);
  if (ProcKind == GK_NONE)
    return StringRef();

  return getArchNameRVGPU(ProcKind);
}

void RVGPU::fillRVGPUFeatureMap(StringRef GPU, const Triple &T,
                                  StringMap<bool> &Features) {
  // XXX - What does the member GPU mean if device name string passed here?
    switch (parseArchRVGPU(GPU)) {
    case GK_SS1000:
      Features["ci-insts"] = true;
      Features["dot5-insts"] = true;
      Features["dot7-insts"] = true;
      Features["dot8-insts"] = true;
      Features["dot9-insts"] = true;
      Features["dot10-insts"] = true;
      Features["dl-insts"] = true;
      Features["16-bit-insts"] = true;
      Features["dpp"] = true;
      Features["gfx8-insts"] = true;
      Features["gfx9-insts"] = true;
      Features["gfx10-insts"] = true;
      Features["gfx10-3-insts"] = true;
      Features["gfx11-insts"] = true;
      Features["atomic-fadd-rtn-insts"] = true;
      Features["image-insts"] = true;
      Features["gws"] = true;
      break;
    case GK_NONE:
      break;
    default:
      llvm_unreachable("Unhandled GPU!");
    }
}

bool RVGPU::insertWaveSizeFeature(StringRef GPU, const Triple &T,
                                   StringMap<bool> &Features,
                                   std::string &ErrorMsg) {
    StringRef DefaultWaveSizeFeature = "wavefrontsize32";
    Features.insert(std::make_pair(DefaultWaveSizeFeature, true));
    return true;
}
