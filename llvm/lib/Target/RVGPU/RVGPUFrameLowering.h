//===--------------------- RVGPUFrameLowering.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Interface to describe a layout of a stack frame on an RVGPU target.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RVGPU_RVGPUFRAMELOWERING_H
#define LLVM_LIB_TARGET_RVGPU_RVGPUFRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {

/// Information about the stack frame layout on the RVGPU targets.
///
/// It holds the direction of the stack growth, the known stack alignment on
/// entry to each function, and the offset to the locals area.
/// See TargetFrameInfo for more comments.
class RVGPUFrameLowering : public TargetFrameLowering {
public:
  RVGPUFrameLowering(StackDirection D, Align StackAl, int LAO,
                      Align TransAl = Align(1));
  ~RVGPUFrameLowering() override;

  /// \returns The number of 32-bit sub-registers that are used when storing
  /// values to the stack.
  unsigned getStackWidth(const MachineFunction &MF) const;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_RVGPU_RVGPUFRAMELOWERING_H
