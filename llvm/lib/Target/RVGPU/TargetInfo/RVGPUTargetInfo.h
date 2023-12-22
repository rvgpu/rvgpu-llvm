//===-- TargetInfo/RVGPUTargetInfo.h - TargetInfo for RVGPU ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RVGPU_TARGETINFO_RVGPUTARGETINFO_H
#define LLVM_LIB_TARGET_RVGPU_TARGETINFO_RVGPUTARGETINFO_H

namespace llvm {

class Target;

/// The target for GCN GPUs.
Target &getTheRVGPUTarget();

}

#endif // LLVM_LIB_TARGET_RVGPU_TARGETINFO_RVGPUTARGETINFO_H
