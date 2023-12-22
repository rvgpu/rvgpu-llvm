//===-- RVGPUFixupKinds.h - RVGPU Specific Fixup Entries ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RVGPU_MCTARGETDESC_RVGPUFIXUPKINDS_H
#define LLVM_LIB_TARGET_RVGPU_MCTARGETDESC_RVGPUFIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace RVGPU {
enum Fixups {
  /// 16-bit PC relative fixup for SOPP branch instructions.
  fixup_si_sopp_br = FirstTargetFixupKind,

  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
};
}
}

#endif
