//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RVGPUArgumentUsageInfo.h"
#include "RVGPU.h"
#include "RVGPUTargetMachine.h"
#include "MCTargetDesc/RVGPUMCTargetDesc.h"
#include "RVRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/NativeFormatting.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "rvgpu-argument-reg-usage-info"

INITIALIZE_PASS(RVGPUArgumentUsageInfo, DEBUG_TYPE,
                "Argument Register Usage Information Storage", false, true)

void RvArgDescriptor::print(raw_ostream &OS,
                          const TargetRegisterInfo *TRI) const {
  if (!isSet()) {
    OS << "<not set>\n";
    return;
  }

  if (isRegister())
    OS << "Reg " << printReg(getRegister(), TRI);
  else
    OS << "Stack offset " << getStackOffset();

  if (isMasked()) {
    OS << " & ";
    llvm::write_hex(OS, Mask, llvm::HexPrintStyle::PrefixLower);
  }

  OS << '\n';
}

char RVGPUArgumentUsageInfo::ID = 0;

const RVGPUFunctionArgInfo RVGPUArgumentUsageInfo::ExternFunctionInfo{};

// Hardcoded registers from fixed function ABI
const RVGPUFunctionArgInfo RVGPUArgumentUsageInfo::FixedABIFunctionInfo
  = RVGPUFunctionArgInfo::fixedABILayout();

bool RVGPUArgumentUsageInfo::doInitialization(Module &M) {
  return false;
}

bool RVGPUArgumentUsageInfo::doFinalization(Module &M) {
  ArgInfoMap.clear();
  return false;
}

// TODO: Print preload kernargs?
void RVGPUArgumentUsageInfo::print(raw_ostream &OS, const Module *M) const {
  for (const auto &FI : ArgInfoMap) {
    OS << "Arguments for " << FI.first->getName() << '\n'
       << "  PrivateSegmentBuffer: " << FI.second.PrivateSegmentBuffer
       << "  DispatchPtr: " << FI.second.DispatchPtr
       << "  QueuePtr: " << FI.second.QueuePtr
       << "  KernargSegmentPtr: " << FI.second.KernargSegmentPtr
       << "  DispatchID: " << FI.second.DispatchID
       << "  FlatScratchInit: " << FI.second.FlatScratchInit
       << "  PrivateSegmentSize: " << FI.second.PrivateSegmentSize
       << "  WorkGroupIDX: " << FI.second.WorkGroupIDX
       << "  WorkGroupIDY: " << FI.second.WorkGroupIDY
       << "  WorkGroupIDZ: " << FI.second.WorkGroupIDZ
       << "  WorkGroupInfo: " << FI.second.WorkGroupInfo
       << "  LDSKernelId: " << FI.second.LDSKernelId
       << "  PrivateSegmentWaveByteOffset: "
          << FI.second.PrivateSegmentWaveByteOffset
       << "  ImplicitBufferPtr: " << FI.second.ImplicitBufferPtr
       << "  ImplicitArgPtr: " << FI.second.ImplicitArgPtr
       << "  WorkItemIDX " << FI.second.WorkItemIDX
       << "  WorkItemIDY " << FI.second.WorkItemIDY
       << "  WorkItemIDZ " << FI.second.WorkItemIDZ
       << '\n';
  }
}

std::tuple<const RvArgDescriptor *, const TargetRegisterClass *, LLT>
RVGPUFunctionArgInfo::getPreloadedValue(
    RVGPUFunctionArgInfo::PreloadedValue Value) const {
  switch (Value) {
  case RVGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER: {
    return std::tuple(PrivateSegmentBuffer ? &PrivateSegmentBuffer : nullptr,
                      &RVGPU::SGPR_128RegClass, LLT::fixed_vector(4, 32));
  }
  case RVGPUFunctionArgInfo::IMPLICIT_BUFFER_PTR:
    return std::tuple(ImplicitBufferPtr ? &ImplicitBufferPtr : nullptr,
                      &RVGPU::SGPR_64RegClass,
                      LLT::pointer(RVGPUAS::CONSTANT_ADDRESS, 64));
  case RVGPUFunctionArgInfo::WORKGROUP_ID_X:
    return std::tuple(WorkGroupIDX ? &WorkGroupIDX : nullptr,
                      &RVGPU::SGPR_32RegClass, LLT::scalar(32));
  case RVGPUFunctionArgInfo::WORKGROUP_ID_Y:
    return std::tuple(WorkGroupIDY ? &WorkGroupIDY : nullptr,
                      &RVGPU::SGPR_32RegClass, LLT::scalar(32));
  case RVGPUFunctionArgInfo::WORKGROUP_ID_Z:
    return std::tuple(WorkGroupIDZ ? &WorkGroupIDZ : nullptr,
                      &RVGPU::SGPR_32RegClass, LLT::scalar(32));
  case RVGPUFunctionArgInfo::LDS_KERNEL_ID:
    return std::tuple(LDSKernelId ? &LDSKernelId : nullptr,
                      &RVGPU::SGPR_32RegClass, LLT::scalar(32));
  case RVGPUFunctionArgInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET:
    return std::tuple(
        PrivateSegmentWaveByteOffset ? &PrivateSegmentWaveByteOffset : nullptr,
        &RVGPU::SGPR_32RegClass, LLT::scalar(32));
  case RVGPUFunctionArgInfo::KERNARG_SEGMENT_PTR:
    return std::tuple(KernargSegmentPtr ? &KernargSegmentPtr : nullptr,
                      &RVGPU::SGPR_64RegClass,
                      LLT::pointer(RVGPUAS::CONSTANT_ADDRESS, 64));
  case RVGPUFunctionArgInfo::IMPLICIT_ARG_PTR:
    return std::tuple(ImplicitArgPtr ? &ImplicitArgPtr : nullptr,
                      &RVGPU::SGPR_64RegClass,
                      LLT::pointer(RVGPUAS::CONSTANT_ADDRESS, 64));
  case RVGPUFunctionArgInfo::DISPATCH_ID:
    return std::tuple(DispatchID ? &DispatchID : nullptr,
                      &RVGPU::SGPR_64RegClass, LLT::scalar(64));
  case RVGPUFunctionArgInfo::FLAT_SCRATCH_INIT:
    return std::tuple(FlatScratchInit ? &FlatScratchInit : nullptr,
                      &RVGPU::SGPR_64RegClass, LLT::scalar(64));
  case RVGPUFunctionArgInfo::DISPATCH_PTR:
    return std::tuple(DispatchPtr ? &DispatchPtr : nullptr,
                      &RVGPU::SGPR_64RegClass,
                      LLT::pointer(RVGPUAS::CONSTANT_ADDRESS, 64));
  case RVGPUFunctionArgInfo::QUEUE_PTR:
    return std::tuple(QueuePtr ? &QueuePtr : nullptr, &RVGPU::SGPR_64RegClass,
                      LLT::pointer(RVGPUAS::CONSTANT_ADDRESS, 64));
  case RVGPUFunctionArgInfo::WORKITEM_ID_X:
    return std::tuple(WorkItemIDX ? &WorkItemIDX : nullptr,
                      &RVGPU::VGPR_32RegClass, LLT::scalar(32));
  case RVGPUFunctionArgInfo::WORKITEM_ID_Y:
    return std::tuple(WorkItemIDY ? &WorkItemIDY : nullptr,
                      &RVGPU::VGPR_32RegClass, LLT::scalar(32));
  case RVGPUFunctionArgInfo::WORKITEM_ID_Z:
    return std::tuple(WorkItemIDZ ? &WorkItemIDZ : nullptr,
                      &RVGPU::VGPR_32RegClass, LLT::scalar(32));
  }
  llvm_unreachable("unexpected preloaded value type");
}

RVGPUFunctionArgInfo RVGPUFunctionArgInfo::fixedABILayout() {
  RVGPUFunctionArgInfo AI;
  AI.PrivateSegmentBuffer
    = RvArgDescriptor::createRegister(RVGPU::SGPR0_SGPR1_SGPR2_SGPR3);
  AI.DispatchPtr = RvArgDescriptor::createRegister(RVGPU::SGPR4_SGPR5);
  AI.QueuePtr = RvArgDescriptor::createRegister(RVGPU::SGPR6_SGPR7);

  // Do not pass kernarg segment pointer, only pass increment version in its
  // place.
  AI.ImplicitArgPtr = RvArgDescriptor::createRegister(RVGPU::SGPR8_SGPR9);
  AI.DispatchID = RvArgDescriptor::createRegister(RVGPU::SGPR10_SGPR11);

  // Skip FlatScratchInit/PrivateSegmentSize
  AI.WorkGroupIDX = RvArgDescriptor::createRegister(RVGPU::SGPR12);
  AI.WorkGroupIDY = RvArgDescriptor::createRegister(RVGPU::SGPR13);
  AI.WorkGroupIDZ = RvArgDescriptor::createRegister(RVGPU::SGPR14);
  AI.LDSKernelId = RvArgDescriptor::createRegister(RVGPU::SGPR15);

  const unsigned Mask = 0x3ff;
  AI.WorkItemIDX = RvArgDescriptor::createRegister(RVGPU::VGPR31, Mask);
  AI.WorkItemIDY = RvArgDescriptor::createRegister(RVGPU::VGPR31, Mask << 10);
  AI.WorkItemIDZ = RvArgDescriptor::createRegister(RVGPU::VGPR31, Mask << 20);
  return AI;
}

const RVGPUFunctionArgInfo &
RVGPUArgumentUsageInfo::lookupFuncArgInfo(const Function &F) const {
  auto I = ArgInfoMap.find(&F);
  if (I == ArgInfoMap.end())
    return FixedABIFunctionInfo;
  return I->second;
}
