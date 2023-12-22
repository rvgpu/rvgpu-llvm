//===- RVGPUMacroFusion.h - RVGPU Macro Fusion ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RVGPU_RVGPUMACROFUSION_H
#define LLVM_LIB_TARGET_RVGPU_RVGPUMACROFUSION_H

#include "llvm/CodeGen/ScheduleDAGMutation.h"
#include <memory>

namespace llvm {

/// Note that you have to add:
///   DAG.addMutation(createRVGPUMacroFusionDAGMutation());
/// to RVGPUPassConfig::createMachineScheduler() to have an effect.
std::unique_ptr<ScheduleDAGMutation> createRVGPUMacroFusionDAGMutation();

} // llvm

#endif // LLVM_LIB_TARGET_RVGPU_RVGPUMACROFUSION_H
