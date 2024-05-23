//===-- RVProgramInfo.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// The RVProgramInfo tracks resource usage and hardware flags for kernels and
/// entry functions.
//
//===----------------------------------------------------------------------===//
//

#include "RVProgramInfo.h"
#include "RVGPUSubtarget.h"
#include "RVDefines.h"
//#include "Utils/AMDGPUBaseInfo.h"

using namespace llvm;

uint64_t RVProgramInfo::getComputePGMRSrc1(const RVGPUSubtarget &ST) const {
  return 0;
}

uint64_t RVProgramInfo::getPGMRSrc1(CallingConv::ID CC,
                                    const RVGPUSubtarget &ST) const {
  return getComputePGMRSrc1(ST);
}

uint64_t RVProgramInfo::getComputePGMRSrc2() const {
  return 0;
}

uint64_t RVProgramInfo::getPGMRSrc2(CallingConv::ID CC) const {
  return getComputePGMRSrc2();
}
