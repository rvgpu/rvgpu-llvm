//===-- RVGPUISelDAGToDAG.cpp - A dag to dag inst selector for RVGPU ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the RVGPU target.
//
//===----------------------------------------------------------------------===//

#include "RVGPUISelDAGToDAG.h"
#include "MCTargetDesc/RVGPUBaseInfo.h"
#include "RVGPUUtilities.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsRVGPU.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetIntrinsicInfo.h"

using namespace llvm;

#define DEBUG_TYPE "rvgpu-isel"
#define PASS_NAME "RVGPU DAG->DAG Pattern Instruction Selection"

/// createRVGPUISelDag - This pass converts a legalized DAG into a
/// RVGPU-specific DAG, ready for instruction scheduling.
FunctionPass *llvm::createRVGPUISelDag(RVGPUTargetMachine &TM,
                                       llvm::CodeGenOptLevel OptLevel) {
  return new RVGPUDAGToDAGISel(TM, OptLevel);
}

char RVGPUDAGToDAGISel::ID = 0;

INITIALIZE_PASS(RVGPUDAGToDAGISel, DEBUG_TYPE, PASS_NAME, false, false)

RVGPUDAGToDAGISel::RVGPUDAGToDAGISel(RVGPUTargetMachine &tm,
                                     CodeGenOptLevel OptLevel)
    : SelectionDAGISel(ID, tm, OptLevel), TM(tm) {
  doMulWide = (OptLevel > CodeGenOptLevel::None);
}

bool RVGPUDAGToDAGISel::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<RVGPUSubtarget>();
  return SelectionDAGISel::runOnMachineFunction(MF);
}

int RVGPUDAGToDAGISel::getDivF32Level() const {
  return Subtarget->getTargetLowering()->getDivF32Level();
}

bool RVGPUDAGToDAGISel::usePrecSqrtF32() const {
  return Subtarget->getTargetLowering()->usePrecSqrtF32();
}

bool RVGPUDAGToDAGISel::useF32FTZ() const {
  return Subtarget->getTargetLowering()->useF32FTZ(*MF);
}

bool RVGPUDAGToDAGISel::allowFMA() const {
  const RVGPUTargetLowering *TL = Subtarget->getTargetLowering();
  return TL->allowFMA(*MF, OptLevel);
}

bool RVGPUDAGToDAGISel::allowUnsafeFPMath() const {
  const RVGPUTargetLowering *TL = Subtarget->getTargetLowering();
  return TL->allowUnsafeFPMath(*MF);
}

bool RVGPUDAGToDAGISel::useShortPointers() const {
  return TM.useShortPointers();
}

/// Select - Select instructions not customized! Used for
/// expanded, promoted and normal instructions.
void RVGPUDAGToDAGISel::Select(SDNode *N) {

  if (N->isMachineOpcode()) {
    N->setNodeId(-1);
    return; // Already selected.
  }

  switch (N->getOpcode()) {
  case ISD::LOAD:
  case ISD::ATOMIC_LOAD:
    if (tryLoad(N))
      return;
    break;
  case ISD::STORE:
  case ISD::ATOMIC_STORE:
    if (tryStore(N))
      return;
    break;
  case ISD::EXTRACT_VECTOR_ELT:
    if (tryEXTRACT_VECTOR_ELEMENT(N))
      return;
    break;
  case RVGPUISD::SETP_F16X2:
    SelectSETP_F16X2(N);
    return;
  case RVGPUISD::SETP_BF16X2:
    SelectSETP_BF16X2(N);
    return;
  case RVGPUISD::LoadV2:
  case RVGPUISD::LoadV4:
    if (tryLoadVector(N))
      return;
    break;
  case RVGPUISD::LDGV2:
  case RVGPUISD::LDGV4:
  case RVGPUISD::LDUV2:
  case RVGPUISD::LDUV4:
    if (tryLDGLDU(N))
      return;
    break;
  case RVGPUISD::StoreV2:
  case RVGPUISD::StoreV4:
    if (tryStoreVector(N))
      return;
    break;
  case RVGPUISD::LoadParam:
  case RVGPUISD::LoadParamV2:
  case RVGPUISD::LoadParamV4:
    if (tryLoadParam(N))
      return;
    break;
  case RVGPUISD::StoreRetval:
  case RVGPUISD::StoreRetvalV2:
  case RVGPUISD::StoreRetvalV4:
    if (tryStoreRetval(N))
      return;
    break;
  case RVGPUISD::StoreParam:
  case RVGPUISD::StoreParamV2:
  case RVGPUISD::StoreParamV4:
  case RVGPUISD::StoreParamS32:
  case RVGPUISD::StoreParamU32:
    if (tryStoreParam(N))
      return;
    break;
  case ISD::INTRINSIC_WO_CHAIN:
    if (tryIntrinsicNoChain(N))
      return;
    break;
  case ISD::INTRINSIC_W_CHAIN:
    if (tryIntrinsicChain(N))
      return;
    break;
  case RVGPUISD::Tex1DFloatS32:
  case RVGPUISD::Tex1DFloatFloat:
  case RVGPUISD::Tex1DFloatFloatLevel:
  case RVGPUISD::Tex1DFloatFloatGrad:
  case RVGPUISD::Tex1DS32S32:
  case RVGPUISD::Tex1DS32Float:
  case RVGPUISD::Tex1DS32FloatLevel:
  case RVGPUISD::Tex1DS32FloatGrad:
  case RVGPUISD::Tex1DU32S32:
  case RVGPUISD::Tex1DU32Float:
  case RVGPUISD::Tex1DU32FloatLevel:
  case RVGPUISD::Tex1DU32FloatGrad:
  case RVGPUISD::Tex1DArrayFloatS32:
  case RVGPUISD::Tex1DArrayFloatFloat:
  case RVGPUISD::Tex1DArrayFloatFloatLevel:
  case RVGPUISD::Tex1DArrayFloatFloatGrad:
  case RVGPUISD::Tex1DArrayS32S32:
  case RVGPUISD::Tex1DArrayS32Float:
  case RVGPUISD::Tex1DArrayS32FloatLevel:
  case RVGPUISD::Tex1DArrayS32FloatGrad:
  case RVGPUISD::Tex1DArrayU32S32:
  case RVGPUISD::Tex1DArrayU32Float:
  case RVGPUISD::Tex1DArrayU32FloatLevel:
  case RVGPUISD::Tex1DArrayU32FloatGrad:
  case RVGPUISD::Tex2DFloatS32:
  case RVGPUISD::Tex2DFloatFloat:
  case RVGPUISD::Tex2DFloatFloatLevel:
  case RVGPUISD::Tex2DFloatFloatGrad:
  case RVGPUISD::Tex2DS32S32:
  case RVGPUISD::Tex2DS32Float:
  case RVGPUISD::Tex2DS32FloatLevel:
  case RVGPUISD::Tex2DS32FloatGrad:
  case RVGPUISD::Tex2DU32S32:
  case RVGPUISD::Tex2DU32Float:
  case RVGPUISD::Tex2DU32FloatLevel:
  case RVGPUISD::Tex2DU32FloatGrad:
  case RVGPUISD::Tex2DArrayFloatS32:
  case RVGPUISD::Tex2DArrayFloatFloat:
  case RVGPUISD::Tex2DArrayFloatFloatLevel:
  case RVGPUISD::Tex2DArrayFloatFloatGrad:
  case RVGPUISD::Tex2DArrayS32S32:
  case RVGPUISD::Tex2DArrayS32Float:
  case RVGPUISD::Tex2DArrayS32FloatLevel:
  case RVGPUISD::Tex2DArrayS32FloatGrad:
  case RVGPUISD::Tex2DArrayU32S32:
  case RVGPUISD::Tex2DArrayU32Float:
  case RVGPUISD::Tex2DArrayU32FloatLevel:
  case RVGPUISD::Tex2DArrayU32FloatGrad:
  case RVGPUISD::Tex3DFloatS32:
  case RVGPUISD::Tex3DFloatFloat:
  case RVGPUISD::Tex3DFloatFloatLevel:
  case RVGPUISD::Tex3DFloatFloatGrad:
  case RVGPUISD::Tex3DS32S32:
  case RVGPUISD::Tex3DS32Float:
  case RVGPUISD::Tex3DS32FloatLevel:
  case RVGPUISD::Tex3DS32FloatGrad:
  case RVGPUISD::Tex3DU32S32:
  case RVGPUISD::Tex3DU32Float:
  case RVGPUISD::Tex3DU32FloatLevel:
  case RVGPUISD::Tex3DU32FloatGrad:
  case RVGPUISD::TexCubeFloatFloat:
  case RVGPUISD::TexCubeFloatFloatLevel:
  case RVGPUISD::TexCubeS32Float:
  case RVGPUISD::TexCubeS32FloatLevel:
  case RVGPUISD::TexCubeU32Float:
  case RVGPUISD::TexCubeU32FloatLevel:
  case RVGPUISD::TexCubeArrayFloatFloat:
  case RVGPUISD::TexCubeArrayFloatFloatLevel:
  case RVGPUISD::TexCubeArrayS32Float:
  case RVGPUISD::TexCubeArrayS32FloatLevel:
  case RVGPUISD::TexCubeArrayU32Float:
  case RVGPUISD::TexCubeArrayU32FloatLevel:
  case RVGPUISD::Tld4R2DFloatFloat:
  case RVGPUISD::Tld4G2DFloatFloat:
  case RVGPUISD::Tld4B2DFloatFloat:
  case RVGPUISD::Tld4A2DFloatFloat:
  case RVGPUISD::Tld4R2DS64Float:
  case RVGPUISD::Tld4G2DS64Float:
  case RVGPUISD::Tld4B2DS64Float:
  case RVGPUISD::Tld4A2DS64Float:
  case RVGPUISD::Tld4R2DU64Float:
  case RVGPUISD::Tld4G2DU64Float:
  case RVGPUISD::Tld4B2DU64Float:
  case RVGPUISD::Tld4A2DU64Float:
  case RVGPUISD::TexUnified1DFloatS32:
  case RVGPUISD::TexUnified1DFloatFloat:
  case RVGPUISD::TexUnified1DFloatFloatLevel:
  case RVGPUISD::TexUnified1DFloatFloatGrad:
  case RVGPUISD::TexUnified1DS32S32:
  case RVGPUISD::TexUnified1DS32Float:
  case RVGPUISD::TexUnified1DS32FloatLevel:
  case RVGPUISD::TexUnified1DS32FloatGrad:
  case RVGPUISD::TexUnified1DU32S32:
  case RVGPUISD::TexUnified1DU32Float:
  case RVGPUISD::TexUnified1DU32FloatLevel:
  case RVGPUISD::TexUnified1DU32FloatGrad:
  case RVGPUISD::TexUnified1DArrayFloatS32:
  case RVGPUISD::TexUnified1DArrayFloatFloat:
  case RVGPUISD::TexUnified1DArrayFloatFloatLevel:
  case RVGPUISD::TexUnified1DArrayFloatFloatGrad:
  case RVGPUISD::TexUnified1DArrayS32S32:
  case RVGPUISD::TexUnified1DArrayS32Float:
  case RVGPUISD::TexUnified1DArrayS32FloatLevel:
  case RVGPUISD::TexUnified1DArrayS32FloatGrad:
  case RVGPUISD::TexUnified1DArrayU32S32:
  case RVGPUISD::TexUnified1DArrayU32Float:
  case RVGPUISD::TexUnified1DArrayU32FloatLevel:
  case RVGPUISD::TexUnified1DArrayU32FloatGrad:
  case RVGPUISD::TexUnified2DFloatS32:
  case RVGPUISD::TexUnified2DFloatFloat:
  case RVGPUISD::TexUnified2DFloatFloatLevel:
  case RVGPUISD::TexUnified2DFloatFloatGrad:
  case RVGPUISD::TexUnified2DS32S32:
  case RVGPUISD::TexUnified2DS32Float:
  case RVGPUISD::TexUnified2DS32FloatLevel:
  case RVGPUISD::TexUnified2DS32FloatGrad:
  case RVGPUISD::TexUnified2DU32S32:
  case RVGPUISD::TexUnified2DU32Float:
  case RVGPUISD::TexUnified2DU32FloatLevel:
  case RVGPUISD::TexUnified2DU32FloatGrad:
  case RVGPUISD::TexUnified2DArrayFloatS32:
  case RVGPUISD::TexUnified2DArrayFloatFloat:
  case RVGPUISD::TexUnified2DArrayFloatFloatLevel:
  case RVGPUISD::TexUnified2DArrayFloatFloatGrad:
  case RVGPUISD::TexUnified2DArrayS32S32:
  case RVGPUISD::TexUnified2DArrayS32Float:
  case RVGPUISD::TexUnified2DArrayS32FloatLevel:
  case RVGPUISD::TexUnified2DArrayS32FloatGrad:
  case RVGPUISD::TexUnified2DArrayU32S32:
  case RVGPUISD::TexUnified2DArrayU32Float:
  case RVGPUISD::TexUnified2DArrayU32FloatLevel:
  case RVGPUISD::TexUnified2DArrayU32FloatGrad:
  case RVGPUISD::TexUnified3DFloatS32:
  case RVGPUISD::TexUnified3DFloatFloat:
  case RVGPUISD::TexUnified3DFloatFloatLevel:
  case RVGPUISD::TexUnified3DFloatFloatGrad:
  case RVGPUISD::TexUnified3DS32S32:
  case RVGPUISD::TexUnified3DS32Float:
  case RVGPUISD::TexUnified3DS32FloatLevel:
  case RVGPUISD::TexUnified3DS32FloatGrad:
  case RVGPUISD::TexUnified3DU32S32:
  case RVGPUISD::TexUnified3DU32Float:
  case RVGPUISD::TexUnified3DU32FloatLevel:
  case RVGPUISD::TexUnified3DU32FloatGrad:
  case RVGPUISD::TexUnifiedCubeFloatFloat:
  case RVGPUISD::TexUnifiedCubeFloatFloatLevel:
  case RVGPUISD::TexUnifiedCubeS32Float:
  case RVGPUISD::TexUnifiedCubeS32FloatLevel:
  case RVGPUISD::TexUnifiedCubeU32Float:
  case RVGPUISD::TexUnifiedCubeU32FloatLevel:
  case RVGPUISD::TexUnifiedCubeArrayFloatFloat:
  case RVGPUISD::TexUnifiedCubeArrayFloatFloatLevel:
  case RVGPUISD::TexUnifiedCubeArrayS32Float:
  case RVGPUISD::TexUnifiedCubeArrayS32FloatLevel:
  case RVGPUISD::TexUnifiedCubeArrayU32Float:
  case RVGPUISD::TexUnifiedCubeArrayU32FloatLevel:
  case RVGPUISD::Tld4UnifiedR2DFloatFloat:
  case RVGPUISD::Tld4UnifiedG2DFloatFloat:
  case RVGPUISD::Tld4UnifiedB2DFloatFloat:
  case RVGPUISD::Tld4UnifiedA2DFloatFloat:
  case RVGPUISD::Tld4UnifiedR2DS64Float:
  case RVGPUISD::Tld4UnifiedG2DS64Float:
  case RVGPUISD::Tld4UnifiedB2DS64Float:
  case RVGPUISD::Tld4UnifiedA2DS64Float:
  case RVGPUISD::Tld4UnifiedR2DU64Float:
  case RVGPUISD::Tld4UnifiedG2DU64Float:
  case RVGPUISD::Tld4UnifiedB2DU64Float:
  case RVGPUISD::Tld4UnifiedA2DU64Float:
    if (tryTextureIntrinsic(N))
      return;
    break;
  case RVGPUISD::Suld1DI8Clamp:
  case RVGPUISD::Suld1DI16Clamp:
  case RVGPUISD::Suld1DI32Clamp:
  case RVGPUISD::Suld1DI64Clamp:
  case RVGPUISD::Suld1DV2I8Clamp:
  case RVGPUISD::Suld1DV2I16Clamp:
  case RVGPUISD::Suld1DV2I32Clamp:
  case RVGPUISD::Suld1DV2I64Clamp:
  case RVGPUISD::Suld1DV4I8Clamp:
  case RVGPUISD::Suld1DV4I16Clamp:
  case RVGPUISD::Suld1DV4I32Clamp:
  case RVGPUISD::Suld1DArrayI8Clamp:
  case RVGPUISD::Suld1DArrayI16Clamp:
  case RVGPUISD::Suld1DArrayI32Clamp:
  case RVGPUISD::Suld1DArrayI64Clamp:
  case RVGPUISD::Suld1DArrayV2I8Clamp:
  case RVGPUISD::Suld1DArrayV2I16Clamp:
  case RVGPUISD::Suld1DArrayV2I32Clamp:
  case RVGPUISD::Suld1DArrayV2I64Clamp:
  case RVGPUISD::Suld1DArrayV4I8Clamp:
  case RVGPUISD::Suld1DArrayV4I16Clamp:
  case RVGPUISD::Suld1DArrayV4I32Clamp:
  case RVGPUISD::Suld2DI8Clamp:
  case RVGPUISD::Suld2DI16Clamp:
  case RVGPUISD::Suld2DI32Clamp:
  case RVGPUISD::Suld2DI64Clamp:
  case RVGPUISD::Suld2DV2I8Clamp:
  case RVGPUISD::Suld2DV2I16Clamp:
  case RVGPUISD::Suld2DV2I32Clamp:
  case RVGPUISD::Suld2DV2I64Clamp:
  case RVGPUISD::Suld2DV4I8Clamp:
  case RVGPUISD::Suld2DV4I16Clamp:
  case RVGPUISD::Suld2DV4I32Clamp:
  case RVGPUISD::Suld2DArrayI8Clamp:
  case RVGPUISD::Suld2DArrayI16Clamp:
  case RVGPUISD::Suld2DArrayI32Clamp:
  case RVGPUISD::Suld2DArrayI64Clamp:
  case RVGPUISD::Suld2DArrayV2I8Clamp:
  case RVGPUISD::Suld2DArrayV2I16Clamp:
  case RVGPUISD::Suld2DArrayV2I32Clamp:
  case RVGPUISD::Suld2DArrayV2I64Clamp:
  case RVGPUISD::Suld2DArrayV4I8Clamp:
  case RVGPUISD::Suld2DArrayV4I16Clamp:
  case RVGPUISD::Suld2DArrayV4I32Clamp:
  case RVGPUISD::Suld3DI8Clamp:
  case RVGPUISD::Suld3DI16Clamp:
  case RVGPUISD::Suld3DI32Clamp:
  case RVGPUISD::Suld3DI64Clamp:
  case RVGPUISD::Suld3DV2I8Clamp:
  case RVGPUISD::Suld3DV2I16Clamp:
  case RVGPUISD::Suld3DV2I32Clamp:
  case RVGPUISD::Suld3DV2I64Clamp:
  case RVGPUISD::Suld3DV4I8Clamp:
  case RVGPUISD::Suld3DV4I16Clamp:
  case RVGPUISD::Suld3DV4I32Clamp:
  case RVGPUISD::Suld1DI8Trap:
  case RVGPUISD::Suld1DI16Trap:
  case RVGPUISD::Suld1DI32Trap:
  case RVGPUISD::Suld1DI64Trap:
  case RVGPUISD::Suld1DV2I8Trap:
  case RVGPUISD::Suld1DV2I16Trap:
  case RVGPUISD::Suld1DV2I32Trap:
  case RVGPUISD::Suld1DV2I64Trap:
  case RVGPUISD::Suld1DV4I8Trap:
  case RVGPUISD::Suld1DV4I16Trap:
  case RVGPUISD::Suld1DV4I32Trap:
  case RVGPUISD::Suld1DArrayI8Trap:
  case RVGPUISD::Suld1DArrayI16Trap:
  case RVGPUISD::Suld1DArrayI32Trap:
  case RVGPUISD::Suld1DArrayI64Trap:
  case RVGPUISD::Suld1DArrayV2I8Trap:
  case RVGPUISD::Suld1DArrayV2I16Trap:
  case RVGPUISD::Suld1DArrayV2I32Trap:
  case RVGPUISD::Suld1DArrayV2I64Trap:
  case RVGPUISD::Suld1DArrayV4I8Trap:
  case RVGPUISD::Suld1DArrayV4I16Trap:
  case RVGPUISD::Suld1DArrayV4I32Trap:
  case RVGPUISD::Suld2DI8Trap:
  case RVGPUISD::Suld2DI16Trap:
  case RVGPUISD::Suld2DI32Trap:
  case RVGPUISD::Suld2DI64Trap:
  case RVGPUISD::Suld2DV2I8Trap:
  case RVGPUISD::Suld2DV2I16Trap:
  case RVGPUISD::Suld2DV2I32Trap:
  case RVGPUISD::Suld2DV2I64Trap:
  case RVGPUISD::Suld2DV4I8Trap:
  case RVGPUISD::Suld2DV4I16Trap:
  case RVGPUISD::Suld2DV4I32Trap:
  case RVGPUISD::Suld2DArrayI8Trap:
  case RVGPUISD::Suld2DArrayI16Trap:
  case RVGPUISD::Suld2DArrayI32Trap:
  case RVGPUISD::Suld2DArrayI64Trap:
  case RVGPUISD::Suld2DArrayV2I8Trap:
  case RVGPUISD::Suld2DArrayV2I16Trap:
  case RVGPUISD::Suld2DArrayV2I32Trap:
  case RVGPUISD::Suld2DArrayV2I64Trap:
  case RVGPUISD::Suld2DArrayV4I8Trap:
  case RVGPUISD::Suld2DArrayV4I16Trap:
  case RVGPUISD::Suld2DArrayV4I32Trap:
  case RVGPUISD::Suld3DI8Trap:
  case RVGPUISD::Suld3DI16Trap:
  case RVGPUISD::Suld3DI32Trap:
  case RVGPUISD::Suld3DI64Trap:
  case RVGPUISD::Suld3DV2I8Trap:
  case RVGPUISD::Suld3DV2I16Trap:
  case RVGPUISD::Suld3DV2I32Trap:
  case RVGPUISD::Suld3DV2I64Trap:
  case RVGPUISD::Suld3DV4I8Trap:
  case RVGPUISD::Suld3DV4I16Trap:
  case RVGPUISD::Suld3DV4I32Trap:
  case RVGPUISD::Suld1DI8Zero:
  case RVGPUISD::Suld1DI16Zero:
  case RVGPUISD::Suld1DI32Zero:
  case RVGPUISD::Suld1DI64Zero:
  case RVGPUISD::Suld1DV2I8Zero:
  case RVGPUISD::Suld1DV2I16Zero:
  case RVGPUISD::Suld1DV2I32Zero:
  case RVGPUISD::Suld1DV2I64Zero:
  case RVGPUISD::Suld1DV4I8Zero:
  case RVGPUISD::Suld1DV4I16Zero:
  case RVGPUISD::Suld1DV4I32Zero:
  case RVGPUISD::Suld1DArrayI8Zero:
  case RVGPUISD::Suld1DArrayI16Zero:
  case RVGPUISD::Suld1DArrayI32Zero:
  case RVGPUISD::Suld1DArrayI64Zero:
  case RVGPUISD::Suld1DArrayV2I8Zero:
  case RVGPUISD::Suld1DArrayV2I16Zero:
  case RVGPUISD::Suld1DArrayV2I32Zero:
  case RVGPUISD::Suld1DArrayV2I64Zero:
  case RVGPUISD::Suld1DArrayV4I8Zero:
  case RVGPUISD::Suld1DArrayV4I16Zero:
  case RVGPUISD::Suld1DArrayV4I32Zero:
  case RVGPUISD::Suld2DI8Zero:
  case RVGPUISD::Suld2DI16Zero:
  case RVGPUISD::Suld2DI32Zero:
  case RVGPUISD::Suld2DI64Zero:
  case RVGPUISD::Suld2DV2I8Zero:
  case RVGPUISD::Suld2DV2I16Zero:
  case RVGPUISD::Suld2DV2I32Zero:
  case RVGPUISD::Suld2DV2I64Zero:
  case RVGPUISD::Suld2DV4I8Zero:
  case RVGPUISD::Suld2DV4I16Zero:
  case RVGPUISD::Suld2DV4I32Zero:
  case RVGPUISD::Suld2DArrayI8Zero:
  case RVGPUISD::Suld2DArrayI16Zero:
  case RVGPUISD::Suld2DArrayI32Zero:
  case RVGPUISD::Suld2DArrayI64Zero:
  case RVGPUISD::Suld2DArrayV2I8Zero:
  case RVGPUISD::Suld2DArrayV2I16Zero:
  case RVGPUISD::Suld2DArrayV2I32Zero:
  case RVGPUISD::Suld2DArrayV2I64Zero:
  case RVGPUISD::Suld2DArrayV4I8Zero:
  case RVGPUISD::Suld2DArrayV4I16Zero:
  case RVGPUISD::Suld2DArrayV4I32Zero:
  case RVGPUISD::Suld3DI8Zero:
  case RVGPUISD::Suld3DI16Zero:
  case RVGPUISD::Suld3DI32Zero:
  case RVGPUISD::Suld3DI64Zero:
  case RVGPUISD::Suld3DV2I8Zero:
  case RVGPUISD::Suld3DV2I16Zero:
  case RVGPUISD::Suld3DV2I32Zero:
  case RVGPUISD::Suld3DV2I64Zero:
  case RVGPUISD::Suld3DV4I8Zero:
  case RVGPUISD::Suld3DV4I16Zero:
  case RVGPUISD::Suld3DV4I32Zero:
    if (trySurfaceIntrinsic(N))
      return;
    break;
  case ISD::AND:
  case ISD::SRA:
  case ISD::SRL:
    // Try to select BFE
    if (tryBFE(N))
      return;
    break;
  case ISD::ADDRSPACECAST:
    SelectAddrSpaceCast(N);
    return;
  case ISD::ConstantFP:
    if (tryConstantFP(N))
      return;
    break;
  default:
    break;
  }
  SelectCode(N);
}

bool RVGPUDAGToDAGISel::tryIntrinsicChain(SDNode *N) {
  unsigned IID = cast<ConstantSDNode>(N->getOperand(1))->getZExtValue();
  switch (IID) {
  default:
    return false;
  case Intrinsic::rvgpu_ldg_global_f:
  case Intrinsic::rvgpu_ldg_global_i:
  case Intrinsic::rvgpu_ldg_global_p:
  case Intrinsic::rvgpu_ldu_global_f:
  case Intrinsic::rvgpu_ldu_global_i:
  case Intrinsic::rvgpu_ldu_global_p:
    return tryLDGLDU(N);
  }
}

// There's no way to specify FP16 and BF16 immediates in .(b)f16 ops, so we
// have to load them into an .(b)f16 register first.
bool RVGPUDAGToDAGISel::tryConstantFP(SDNode *N) {
  if (N->getValueType(0) != MVT::f16 && N->getValueType(0) != MVT::bf16)
    return false;
  SDValue Val = CurDAG->getTargetConstantFP(
      cast<ConstantFPSDNode>(N)->getValueAPF(), SDLoc(N), N->getValueType(0));
  SDNode *LoadConstF16 = CurDAG->getMachineNode(
      (N->getValueType(0) == MVT::f16 ? RVGPU::LOAD_CONST_F16
                                      : RVGPU::LOAD_CONST_BF16),
      SDLoc(N), N->getValueType(0), Val);
  ReplaceNode(N, LoadConstF16);
  return true;
}

// Map ISD:CONDCODE value to appropriate CmpMode expected by
// RVGPUInstPrinter::printCmpMode()
static unsigned getPTXCmpMode(const CondCodeSDNode &CondCode, bool FTZ) {
  using RVGPU::PTXCmpMode::CmpMode;
  unsigned PTXCmpMode = [](ISD::CondCode CC) {
    switch (CC) {
    default:
      llvm_unreachable("Unexpected condition code.");
    case ISD::SETOEQ:
      return CmpMode::EQ;
    case ISD::SETOGT:
      return CmpMode::GT;
    case ISD::SETOGE:
      return CmpMode::GE;
    case ISD::SETOLT:
      return CmpMode::LT;
    case ISD::SETOLE:
      return CmpMode::LE;
    case ISD::SETONE:
      return CmpMode::NE;
    case ISD::SETO:
      return CmpMode::NUM;
    case ISD::SETUO:
      return CmpMode::NotANumber;
    case ISD::SETUEQ:
      return CmpMode::EQU;
    case ISD::SETUGT:
      return CmpMode::GTU;
    case ISD::SETUGE:
      return CmpMode::GEU;
    case ISD::SETULT:
      return CmpMode::LTU;
    case ISD::SETULE:
      return CmpMode::LEU;
    case ISD::SETUNE:
      return CmpMode::NEU;
    case ISD::SETEQ:
      return CmpMode::EQ;
    case ISD::SETGT:
      return CmpMode::GT;
    case ISD::SETGE:
      return CmpMode::GE;
    case ISD::SETLT:
      return CmpMode::LT;
    case ISD::SETLE:
      return CmpMode::LE;
    case ISD::SETNE:
      return CmpMode::NE;
    }
  }(CondCode.get());

  if (FTZ)
    PTXCmpMode |= RVGPU::PTXCmpMode::FTZ_FLAG;

  return PTXCmpMode;
}

bool RVGPUDAGToDAGISel::SelectSETP_F16X2(SDNode *N) {
  unsigned PTXCmpMode =
      getPTXCmpMode(*cast<CondCodeSDNode>(N->getOperand(2)), useF32FTZ());
  SDLoc DL(N);
  SDNode *SetP = CurDAG->getMachineNode(
      RVGPU::SETP_f16x2rr, DL, MVT::i1, MVT::i1, N->getOperand(0),
      N->getOperand(1), CurDAG->getTargetConstant(PTXCmpMode, DL, MVT::i32));
  ReplaceNode(N, SetP);
  return true;
}

bool RVGPUDAGToDAGISel::SelectSETP_BF16X2(SDNode *N) {
  unsigned PTXCmpMode =
      getPTXCmpMode(*cast<CondCodeSDNode>(N->getOperand(2)), useF32FTZ());
  SDLoc DL(N);
  SDNode *SetP = CurDAG->getMachineNode(
      RVGPU::SETP_bf16x2rr, DL, MVT::i1, MVT::i1, N->getOperand(0),
      N->getOperand(1), CurDAG->getTargetConstant(PTXCmpMode, DL, MVT::i32));
  ReplaceNode(N, SetP);
  return true;
}

// Find all instances of extract_vector_elt that use this v2f16 vector
// and coalesce them into a scattering move instruction.
bool RVGPUDAGToDAGISel::tryEXTRACT_VECTOR_ELEMENT(SDNode *N) {
  SDValue Vector = N->getOperand(0);

  // We only care about 16x2 as it's the only real vector type we
  // need to deal with.
  MVT VT = Vector.getSimpleValueType();
  if (!Isv2x16VT(VT))
    return false;
  // Find and record all uses of this vector that extract element 0 or 1.
  SmallVector<SDNode *, 4> E0, E1;
  for (auto *U : Vector.getNode()->uses()) {
    if (U->getOpcode() != ISD::EXTRACT_VECTOR_ELT)
      continue;
    if (U->getOperand(0) != Vector)
      continue;
    if (const ConstantSDNode *IdxConst =
            dyn_cast<ConstantSDNode>(U->getOperand(1))) {
      if (IdxConst->getZExtValue() == 0)
        E0.push_back(U);
      else if (IdxConst->getZExtValue() == 1)
        E1.push_back(U);
      else
        llvm_unreachable("Invalid vector index.");
    }
  }

  // There's no point scattering f16x2 if we only ever access one
  // element of it.
  if (E0.empty() || E1.empty())
    return false;

  // Merge (f16 extractelt(V, 0), f16 extractelt(V,1))
  // into f16,f16 SplitF16x2(V)
  MVT EltVT = VT.getVectorElementType();
  SDNode *ScatterOp =
      CurDAG->getMachineNode(RVGPU::I32toV2I16, SDLoc(N), EltVT, EltVT, Vector);
  for (auto *Node : E0)
    ReplaceUses(SDValue(Node, 0), SDValue(ScatterOp, 0));
  for (auto *Node : E1)
    ReplaceUses(SDValue(Node, 0), SDValue(ScatterOp, 1));

  return true;
}

static unsigned int getCodeAddrSpace(MemSDNode *N) {
  const Value *Src = N->getMemOperand()->getValue();

  if (!Src)
    return RVGPU::PTXLdStInstCode::GENERIC;

  if (auto *PT = dyn_cast<PointerType>(Src->getType())) {
    switch (PT->getAddressSpace()) {
    case llvm::ADDRESS_SPACE_LOCAL: return RVGPU::PTXLdStInstCode::LOCAL;
    case llvm::ADDRESS_SPACE_GLOBAL: return RVGPU::PTXLdStInstCode::GLOBAL;
    case llvm::ADDRESS_SPACE_SHARED: return RVGPU::PTXLdStInstCode::SHARED;
    case llvm::ADDRESS_SPACE_GENERIC: return RVGPU::PTXLdStInstCode::GENERIC;
    case llvm::ADDRESS_SPACE_PARAM: return RVGPU::PTXLdStInstCode::PARAM;
    case llvm::ADDRESS_SPACE_CONST: return RVGPU::PTXLdStInstCode::CONSTANT;
    default: break;
    }
  }
  return RVGPU::PTXLdStInstCode::GENERIC;
}

static bool canLowerToLDG(MemSDNode *N, const RVGPUSubtarget &Subtarget,
                          unsigned CodeAddrSpace, MachineFunction *F) {
  // We use ldg (i.e. ld.global.nc) for invariant loads from the global address
  // space.
  //
  // We have two ways of identifying invariant loads: Loads may be explicitly
  // marked as invariant, or we may infer them to be invariant.
  //
  // We currently infer invariance for loads from
  //  - constant global variables, and
  //  - kernel function pointer params that are noalias (i.e. __restrict) and
  //    never written to.
  //
  // TODO: Perform a more powerful invariance analysis (ideally IPO, and ideally
  // not during the SelectionDAG phase).
  //
  // TODO: Infer invariance only at -O2.  We still want to use ldg at -O0 for
  // explicitly invariant loads because these are how clang tells us to use ldg
  // when the user uses a builtin.
  if (!Subtarget.hasLDG() || CodeAddrSpace != RVGPU::PTXLdStInstCode::GLOBAL)
    return false;

  if (N->isInvariant())
    return true;

  bool IsKernelFn = isKernelFunction(F->getFunction());

  // We use getUnderlyingObjects() here instead of getUnderlyingObject() mainly
  // because the former looks through phi nodes while the latter does not. We
  // need to look through phi nodes to handle pointer induction variables.
  SmallVector<const Value *, 8> Objs;
  getUnderlyingObjects(N->getMemOperand()->getValue(), Objs);

  return all_of(Objs, [&](const Value *V) {
    if (auto *A = dyn_cast<const Argument>(V))
      return IsKernelFn && A->onlyReadsMemory() && A->hasNoAliasAttr();
    if (auto *GV = dyn_cast<const GlobalVariable>(V))
      return GV->isConstant();
    return false;
  });
}

bool RVGPUDAGToDAGISel::tryIntrinsicNoChain(SDNode *N) {
  unsigned IID = cast<ConstantSDNode>(N->getOperand(0))->getZExtValue();
  switch (IID) {
  default:
    return false;
  case Intrinsic::rvgpu_texsurf_handle_internal:
    SelectTexSurfHandle(N);
    return true;
  }
}

void RVGPUDAGToDAGISel::SelectTexSurfHandle(SDNode *N) {
  // Op 0 is the intrinsic ID
  SDValue Wrapper = N->getOperand(1);
  SDValue GlobalVal = Wrapper.getOperand(0);
  ReplaceNode(N, CurDAG->getMachineNode(RVGPU::texsurf_handles, SDLoc(N),
                                        MVT::i64, GlobalVal));
}

void RVGPUDAGToDAGISel::SelectAddrSpaceCast(SDNode *N) {
  SDValue Src = N->getOperand(0);
  AddrSpaceCastSDNode *CastN = cast<AddrSpaceCastSDNode>(N);
  unsigned SrcAddrSpace = CastN->getSrcAddressSpace();
  unsigned DstAddrSpace = CastN->getDestAddressSpace();
  assert(SrcAddrSpace != DstAddrSpace &&
         "addrspacecast must be between different address spaces");

  if (DstAddrSpace == ADDRESS_SPACE_GENERIC) {
    // Specific to generic
    unsigned Opc;
    switch (SrcAddrSpace) {
    default: report_fatal_error("Bad address space in addrspacecast");
    case ADDRESS_SPACE_GLOBAL:
      Opc = TM.is64Bit() ? RVGPU::cvta_global_yes_64 : RVGPU::cvta_global_yes;
      break;
    case ADDRESS_SPACE_SHARED:
      Opc = TM.is64Bit() ? (useShortPointers() ? RVGPU::cvta_shared_yes_6432
                                               : RVGPU::cvta_shared_yes_64)
                         : RVGPU::cvta_shared_yes;
      break;
    case ADDRESS_SPACE_CONST:
      Opc = TM.is64Bit() ? (useShortPointers() ? RVGPU::cvta_const_yes_6432
                                               : RVGPU::cvta_const_yes_64)
                         : RVGPU::cvta_const_yes;
      break;
    case ADDRESS_SPACE_LOCAL:
      Opc = TM.is64Bit() ? (useShortPointers() ? RVGPU::cvta_local_yes_6432
                                               : RVGPU::cvta_local_yes_64)
                         : RVGPU::cvta_local_yes;
      break;
    }
    ReplaceNode(N, CurDAG->getMachineNode(Opc, SDLoc(N), N->getValueType(0),
                                          Src));
    return;
  } else {
    // Generic to specific
    if (SrcAddrSpace != 0)
      report_fatal_error("Cannot cast between two non-generic address spaces");
    unsigned Opc;
    switch (DstAddrSpace) {
    default: report_fatal_error("Bad address space in addrspacecast");
    case ADDRESS_SPACE_GLOBAL:
      Opc = TM.is64Bit() ? RVGPU::cvta_to_global_yes_64
                         : RVGPU::cvta_to_global_yes;
      break;
    case ADDRESS_SPACE_SHARED:
      Opc = TM.is64Bit() ? (useShortPointers() ? RVGPU::cvta_to_shared_yes_3264
                                                : RVGPU::cvta_to_shared_yes_64)
                         : RVGPU::cvta_to_shared_yes;
      break;
    case ADDRESS_SPACE_CONST:
      Opc = TM.is64Bit() ? (useShortPointers() ? RVGPU::cvta_to_const_yes_3264
                                             : RVGPU::cvta_to_const_yes_64)
                         : RVGPU::cvta_to_const_yes;
      break;
    case ADDRESS_SPACE_LOCAL:
      Opc = TM.is64Bit() ? (useShortPointers() ? RVGPU::cvta_to_local_yes_3264
                                               : RVGPU::cvta_to_local_yes_64)
                         : RVGPU::cvta_to_local_yes;
      break;
    case ADDRESS_SPACE_PARAM:
      Opc = TM.is64Bit() ? RVGPU::nvvm_ptr_gen_to_param_64
                         : RVGPU::nvvm_ptr_gen_to_param;
      break;
    }
    ReplaceNode(N, CurDAG->getMachineNode(Opc, SDLoc(N), N->getValueType(0),
                                          Src));
    return;
  }
}

// Helper function template to reduce amount of boilerplate code for
// opcode selection.
static std::optional<unsigned>
pickOpcodeForVT(MVT::SimpleValueType VT, unsigned Opcode_i8,
                unsigned Opcode_i16, unsigned Opcode_i32,
                std::optional<unsigned> Opcode_i64, unsigned Opcode_f32,
                std::optional<unsigned> Opcode_f64) {
  switch (VT) {
  case MVT::i1:
  case MVT::i8:
    return Opcode_i8;
  case MVT::i16:
    return Opcode_i16;
  case MVT::i32:
    return Opcode_i32;
  case MVT::i64:
    return Opcode_i64;
  case MVT::f16:
  case MVT::bf16:
    return Opcode_i16;
  case MVT::v2f16:
  case MVT::v2bf16:
  case MVT::v2i16:
  case MVT::v4i8:
    return Opcode_i32;
  case MVT::f32:
    return Opcode_f32;
  case MVT::f64:
    return Opcode_f64;
  default:
    return std::nullopt;
  }
}

static int getLdStRegType(EVT VT) {
  if (VT.isFloatingPoint())
    switch (VT.getSimpleVT().SimpleTy) {
    case MVT::f16:
    case MVT::bf16:
    case MVT::v2f16:
    case MVT::v2bf16:
      return RVGPU::PTXLdStInstCode::Untyped;
    default:
      return RVGPU::PTXLdStInstCode::Float;
    }
  else
    return RVGPU::PTXLdStInstCode::Unsigned;
}

bool RVGPUDAGToDAGISel::tryLoad(SDNode *N) {
  SDLoc dl(N);
  MemSDNode *LD = cast<MemSDNode>(N);
  assert(LD->readMem() && "Expected load");
  LoadSDNode *PlainLoad = dyn_cast<LoadSDNode>(N);
  EVT LoadedVT = LD->getMemoryVT();
  SDNode *RVGPULD = nullptr;

  // do not support pre/post inc/dec
  if (PlainLoad && PlainLoad->isIndexed())
    return false;

  if (!LoadedVT.isSimple())
    return false;

  AtomicOrdering Ordering = LD->getSuccessOrdering();
  // In order to lower atomic loads with stronger guarantees we would need to
  // use load.acquire or insert fences. However these features were only added
  // with PTX ISA 6.0 / sm_70.
  // TODO: Check if we can actually use the new instructions and implement them.
  if (isStrongerThanMonotonic(Ordering))
    return false;

  // Address Space Setting
  unsigned int CodeAddrSpace = getCodeAddrSpace(LD);
  if (canLowerToLDG(LD, *Subtarget, CodeAddrSpace, MF)) {
    return tryLDGLDU(N);
  }

  unsigned int PointerSize =
      CurDAG->getDataLayout().getPointerSizeInBits(LD->getAddressSpace());

  // Volatile Setting
  // - .volatile is only available for .global and .shared
  // - .volatile has the same memory synchronization semantics as .relaxed.sys
  bool isVolatile = LD->isVolatile() || Ordering == AtomicOrdering::Monotonic;
  if (CodeAddrSpace != RVGPU::PTXLdStInstCode::GLOBAL &&
      CodeAddrSpace != RVGPU::PTXLdStInstCode::SHARED &&
      CodeAddrSpace != RVGPU::PTXLdStInstCode::GENERIC)
    isVolatile = false;

  // Type Setting: fromType + fromTypeWidth
  //
  // Sign   : ISD::SEXTLOAD
  // Unsign : ISD::ZEXTLOAD, ISD::NON_EXTLOAD or ISD::EXTLOAD and the
  //          type is integer
  // Float  : ISD::NON_EXTLOAD or ISD::EXTLOAD and the type is float
  MVT SimpleVT = LoadedVT.getSimpleVT();
  MVT ScalarVT = SimpleVT.getScalarType();
  // Read at least 8 bits (predicates are stored as 8-bit values)
  unsigned fromTypeWidth = std::max(8U, (unsigned)ScalarVT.getSizeInBits());
  unsigned int fromType;

  // Vector Setting
  unsigned vecType = RVGPU::PTXLdStInstCode::Scalar;
  if (SimpleVT.isVector()) {
    assert((Isv2x16VT(LoadedVT) || LoadedVT == MVT::v4i8) &&
           "Unexpected vector type");
    // v2f16/v2bf16/v2i16 is loaded using ld.b32
    fromTypeWidth = 32;
  }

  if (PlainLoad && (PlainLoad->getExtensionType() == ISD::SEXTLOAD))
    fromType = RVGPU::PTXLdStInstCode::Signed;
  else
    fromType = getLdStRegType(ScalarVT);

  // Create the machine instruction DAG
  SDValue Chain = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue Addr;
  SDValue Offset, Base;
  std::optional<unsigned> Opcode;
  MVT::SimpleValueType TargetVT = LD->getSimpleValueType(0).SimpleTy;

  if (SelectDirectAddr(N1, Addr)) {
    Opcode = pickOpcodeForVT(TargetVT, RVGPU::LD_i8_avar, RVGPU::LD_i16_avar,
                             RVGPU::LD_i32_avar, RVGPU::LD_i64_avar,
                             RVGPU::LD_f32_avar, RVGPU::LD_f64_avar);
    if (!Opcode)
      return false;
    SDValue Ops[] = { getI32Imm(isVolatile, dl), getI32Imm(CodeAddrSpace, dl),
                      getI32Imm(vecType, dl), getI32Imm(fromType, dl),
                      getI32Imm(fromTypeWidth, dl), Addr, Chain };
    RVGPULD = CurDAG->getMachineNode(*Opcode, dl, TargetVT, MVT::Other, Ops);
  } else if (PointerSize == 64 ? SelectADDRsi64(N1.getNode(), N1, Base, Offset)
                               : SelectADDRsi(N1.getNode(), N1, Base, Offset)) {
    Opcode = pickOpcodeForVT(TargetVT, RVGPU::LD_i8_asi, RVGPU::LD_i16_asi,
                             RVGPU::LD_i32_asi, RVGPU::LD_i64_asi,
                             RVGPU::LD_f32_asi, RVGPU::LD_f64_asi);
    if (!Opcode)
      return false;
    SDValue Ops[] = { getI32Imm(isVolatile, dl), getI32Imm(CodeAddrSpace, dl),
                      getI32Imm(vecType, dl), getI32Imm(fromType, dl),
                      getI32Imm(fromTypeWidth, dl), Base, Offset, Chain };
    RVGPULD = CurDAG->getMachineNode(*Opcode, dl, TargetVT, MVT::Other, Ops);
  } else if (PointerSize == 64 ? SelectADDRri64(N1.getNode(), N1, Base, Offset)
                               : SelectADDRri(N1.getNode(), N1, Base, Offset)) {
    if (PointerSize == 64)
      Opcode =
          pickOpcodeForVT(TargetVT, RVGPU::LD_i8_ari_64, RVGPU::LD_i16_ari_64,
                          RVGPU::LD_i32_ari_64, RVGPU::LD_i64_ari_64,
                          RVGPU::LD_f32_ari_64, RVGPU::LD_f64_ari_64);
    else
      Opcode = pickOpcodeForVT(TargetVT, RVGPU::LD_i8_ari, RVGPU::LD_i16_ari,
                               RVGPU::LD_i32_ari, RVGPU::LD_i64_ari,
                               RVGPU::LD_f32_ari, RVGPU::LD_f64_ari);
    if (!Opcode)
      return false;
    SDValue Ops[] = { getI32Imm(isVolatile, dl), getI32Imm(CodeAddrSpace, dl),
                      getI32Imm(vecType, dl), getI32Imm(fromType, dl),
                      getI32Imm(fromTypeWidth, dl), Base, Offset, Chain };
    RVGPULD = CurDAG->getMachineNode(*Opcode, dl, TargetVT, MVT::Other, Ops);
  } else {
    if (PointerSize == 64)
      Opcode =
          pickOpcodeForVT(TargetVT, RVGPU::LD_i8_areg_64, RVGPU::LD_i16_areg_64,
                          RVGPU::LD_i32_areg_64, RVGPU::LD_i64_areg_64,
                          RVGPU::LD_f32_areg_64, RVGPU::LD_f64_areg_64);
    else
      Opcode = pickOpcodeForVT(TargetVT, RVGPU::LD_i8_areg, RVGPU::LD_i16_areg,
                               RVGPU::LD_i32_areg, RVGPU::LD_i64_areg,
                               RVGPU::LD_f32_areg, RVGPU::LD_f64_areg);
    if (!Opcode)
      return false;
    SDValue Ops[] = { getI32Imm(isVolatile, dl), getI32Imm(CodeAddrSpace, dl),
                      getI32Imm(vecType, dl), getI32Imm(fromType, dl),
                      getI32Imm(fromTypeWidth, dl), N1, Chain };
    RVGPULD = CurDAG->getMachineNode(*Opcode, dl, TargetVT, MVT::Other, Ops);
  }

  if (!RVGPULD)
    return false;

  MachineMemOperand *MemRef = cast<MemSDNode>(N)->getMemOperand();
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(RVGPULD), {MemRef});

  ReplaceNode(N, RVGPULD);
  return true;
}

bool RVGPUDAGToDAGISel::tryLoadVector(SDNode *N) {

  SDValue Chain = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);
  SDValue Addr, Offset, Base;
  std::optional<unsigned> Opcode;
  SDLoc DL(N);
  SDNode *LD;
  MemSDNode *MemSD = cast<MemSDNode>(N);
  EVT LoadedVT = MemSD->getMemoryVT();

  if (!LoadedVT.isSimple())
    return false;

  // Address Space Setting
  unsigned int CodeAddrSpace = getCodeAddrSpace(MemSD);
  if (canLowerToLDG(MemSD, *Subtarget, CodeAddrSpace, MF)) {
    return tryLDGLDU(N);
  }

  unsigned int PointerSize =
      CurDAG->getDataLayout().getPointerSizeInBits(MemSD->getAddressSpace());

  // Volatile Setting
  // - .volatile is only availalble for .global and .shared
  bool IsVolatile = MemSD->isVolatile();
  if (CodeAddrSpace != RVGPU::PTXLdStInstCode::GLOBAL &&
      CodeAddrSpace != RVGPU::PTXLdStInstCode::SHARED &&
      CodeAddrSpace != RVGPU::PTXLdStInstCode::GENERIC)
    IsVolatile = false;

  // Vector Setting
  MVT SimpleVT = LoadedVT.getSimpleVT();

  // Type Setting: fromType + fromTypeWidth
  //
  // Sign   : ISD::SEXTLOAD
  // Unsign : ISD::ZEXTLOAD, ISD::NON_EXTLOAD or ISD::EXTLOAD and the
  //          type is integer
  // Float  : ISD::NON_EXTLOAD or ISD::EXTLOAD and the type is float
  MVT ScalarVT = SimpleVT.getScalarType();
  // Read at least 8 bits (predicates are stored as 8-bit values)
  unsigned FromTypeWidth = std::max(8U, (unsigned)ScalarVT.getSizeInBits());
  unsigned int FromType;
  // The last operand holds the original LoadSDNode::getExtensionType() value
  unsigned ExtensionType = cast<ConstantSDNode>(
      N->getOperand(N->getNumOperands() - 1))->getZExtValue();
  if (ExtensionType == ISD::SEXTLOAD)
    FromType = RVGPU::PTXLdStInstCode::Signed;
  else
    FromType = getLdStRegType(ScalarVT);

  unsigned VecType;

  switch (N->getOpcode()) {
  case RVGPUISD::LoadV2:
    VecType = RVGPU::PTXLdStInstCode::V2;
    break;
  case RVGPUISD::LoadV4:
    VecType = RVGPU::PTXLdStInstCode::V4;
    break;
  default:
    return false;
  }

  EVT EltVT = N->getValueType(0);

  // v8x16 is a special case. PTX doesn't have ld.v8.16
  // instruction. Instead, we split the vector into v2x16 chunks and
  // load them with ld.v4.b32.
  if (Isv2x16VT(EltVT)) {
    assert(N->getOpcode() == RVGPUISD::LoadV4 && "Unexpected load opcode.");
    EltVT = MVT::i32;
    FromType = RVGPU::PTXLdStInstCode::Untyped;
    FromTypeWidth = 32;
  }

  if (SelectDirectAddr(Op1, Addr)) {
    switch (N->getOpcode()) {
    default:
      return false;
    case RVGPUISD::LoadV2:
      Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                               RVGPU::LDV_i8_v2_avar, RVGPU::LDV_i16_v2_avar,
                               RVGPU::LDV_i32_v2_avar, RVGPU::LDV_i64_v2_avar,
                               RVGPU::LDV_f32_v2_avar, RVGPU::LDV_f64_v2_avar);
      break;
    case RVGPUISD::LoadV4:
      Opcode =
          pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy, RVGPU::LDV_i8_v4_avar,
                          RVGPU::LDV_i16_v4_avar, RVGPU::LDV_i32_v4_avar,
                          std::nullopt, RVGPU::LDV_f32_v4_avar, std::nullopt);
      break;
    }
    if (!Opcode)
      return false;
    SDValue Ops[] = { getI32Imm(IsVolatile, DL), getI32Imm(CodeAddrSpace, DL),
                      getI32Imm(VecType, DL), getI32Imm(FromType, DL),
                      getI32Imm(FromTypeWidth, DL), Addr, Chain };
    LD = CurDAG->getMachineNode(*Opcode, DL, N->getVTList(), Ops);
  } else if (PointerSize == 64
                 ? SelectADDRsi64(Op1.getNode(), Op1, Base, Offset)
                 : SelectADDRsi(Op1.getNode(), Op1, Base, Offset)) {
    switch (N->getOpcode()) {
    default:
      return false;
    case RVGPUISD::LoadV2:
      Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                               RVGPU::LDV_i8_v2_asi, RVGPU::LDV_i16_v2_asi,
                               RVGPU::LDV_i32_v2_asi, RVGPU::LDV_i64_v2_asi,
                               RVGPU::LDV_f32_v2_asi, RVGPU::LDV_f64_v2_asi);
      break;
    case RVGPUISD::LoadV4:
      Opcode =
          pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy, RVGPU::LDV_i8_v4_asi,
                          RVGPU::LDV_i16_v4_asi, RVGPU::LDV_i32_v4_asi,
                          std::nullopt, RVGPU::LDV_f32_v4_asi, std::nullopt);
      break;
    }
    if (!Opcode)
      return false;
    SDValue Ops[] = { getI32Imm(IsVolatile, DL), getI32Imm(CodeAddrSpace, DL),
                      getI32Imm(VecType, DL), getI32Imm(FromType, DL),
                      getI32Imm(FromTypeWidth, DL), Base, Offset, Chain };
    LD = CurDAG->getMachineNode(*Opcode, DL, N->getVTList(), Ops);
  } else if (PointerSize == 64
                 ? SelectADDRri64(Op1.getNode(), Op1, Base, Offset)
                 : SelectADDRri(Op1.getNode(), Op1, Base, Offset)) {
    if (PointerSize == 64) {
      switch (N->getOpcode()) {
      default:
        return false;
      case RVGPUISD::LoadV2:
        Opcode =
            pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                            RVGPU::LDV_i8_v2_ari_64, RVGPU::LDV_i16_v2_ari_64,
                            RVGPU::LDV_i32_v2_ari_64, RVGPU::LDV_i64_v2_ari_64,
                            RVGPU::LDV_f32_v2_ari_64, RVGPU::LDV_f64_v2_ari_64);
        break;
      case RVGPUISD::LoadV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, RVGPU::LDV_i8_v4_ari_64,
            RVGPU::LDV_i16_v4_ari_64, RVGPU::LDV_i32_v4_ari_64, std::nullopt,
            RVGPU::LDV_f32_v4_ari_64, std::nullopt);
        break;
      }
    } else {
      switch (N->getOpcode()) {
      default:
        return false;
      case RVGPUISD::LoadV2:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 RVGPU::LDV_i8_v2_ari, RVGPU::LDV_i16_v2_ari,
                                 RVGPU::LDV_i32_v2_ari, RVGPU::LDV_i64_v2_ari,
                                 RVGPU::LDV_f32_v2_ari, RVGPU::LDV_f64_v2_ari);
        break;
      case RVGPUISD::LoadV4:
        Opcode =
            pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy, RVGPU::LDV_i8_v4_ari,
                            RVGPU::LDV_i16_v4_ari, RVGPU::LDV_i32_v4_ari,
                            std::nullopt, RVGPU::LDV_f32_v4_ari, std::nullopt);
        break;
      }
    }
    if (!Opcode)
      return false;
    SDValue Ops[] = { getI32Imm(IsVolatile, DL), getI32Imm(CodeAddrSpace, DL),
                      getI32Imm(VecType, DL), getI32Imm(FromType, DL),
                      getI32Imm(FromTypeWidth, DL), Base, Offset, Chain };

    LD = CurDAG->getMachineNode(*Opcode, DL, N->getVTList(), Ops);
  } else {
    if (PointerSize == 64) {
      switch (N->getOpcode()) {
      default:
        return false;
      case RVGPUISD::LoadV2:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, RVGPU::LDV_i8_v2_areg_64,
            RVGPU::LDV_i16_v2_areg_64, RVGPU::LDV_i32_v2_areg_64,
            RVGPU::LDV_i64_v2_areg_64, RVGPU::LDV_f32_v2_areg_64,
            RVGPU::LDV_f64_v2_areg_64);
        break;
      case RVGPUISD::LoadV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, RVGPU::LDV_i8_v4_areg_64,
            RVGPU::LDV_i16_v4_areg_64, RVGPU::LDV_i32_v4_areg_64, std::nullopt,
            RVGPU::LDV_f32_v4_areg_64, std::nullopt);
        break;
      }
    } else {
      switch (N->getOpcode()) {
      default:
        return false;
      case RVGPUISD::LoadV2:
        Opcode =
            pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy, RVGPU::LDV_i8_v2_areg,
                            RVGPU::LDV_i16_v2_areg, RVGPU::LDV_i32_v2_areg,
                            RVGPU::LDV_i64_v2_areg, RVGPU::LDV_f32_v2_areg,
                            RVGPU::LDV_f64_v2_areg);
        break;
      case RVGPUISD::LoadV4:
        Opcode =
            pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy, RVGPU::LDV_i8_v4_areg,
                            RVGPU::LDV_i16_v4_areg, RVGPU::LDV_i32_v4_areg,
                            std::nullopt, RVGPU::LDV_f32_v4_areg, std::nullopt);
        break;
      }
    }
    if (!Opcode)
      return false;
    SDValue Ops[] = { getI32Imm(IsVolatile, DL), getI32Imm(CodeAddrSpace, DL),
                      getI32Imm(VecType, DL), getI32Imm(FromType, DL),
                      getI32Imm(FromTypeWidth, DL), Op1, Chain };
    LD = CurDAG->getMachineNode(*Opcode, DL, N->getVTList(), Ops);
  }

  MachineMemOperand *MemRef = cast<MemSDNode>(N)->getMemOperand();
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(LD), {MemRef});

  ReplaceNode(N, LD);
  return true;
}

bool RVGPUDAGToDAGISel::tryLDGLDU(SDNode *N) {

  SDValue Chain = N->getOperand(0);
  SDValue Op1;
  MemSDNode *Mem;
  bool IsLDG = true;

  // If this is an LDG intrinsic, the address is the third operand. If its an
  // LDG/LDU SD node (from custom vector handling), then its the second operand
  if (N->getOpcode() == ISD::INTRINSIC_W_CHAIN) {
    Op1 = N->getOperand(2);
    Mem = cast<MemIntrinsicSDNode>(N);
    unsigned IID = cast<ConstantSDNode>(N->getOperand(1))->getZExtValue();
    switch (IID) {
    default:
      return false;
    case Intrinsic::rvgpu_ldg_global_f:
    case Intrinsic::rvgpu_ldg_global_i:
    case Intrinsic::rvgpu_ldg_global_p:
      IsLDG = true;
      break;
    case Intrinsic::rvgpu_ldu_global_f:
    case Intrinsic::rvgpu_ldu_global_i:
    case Intrinsic::rvgpu_ldu_global_p:
      IsLDG = false;
      break;
    }
  } else {
    Op1 = N->getOperand(1);
    Mem = cast<MemSDNode>(N);
  }

  std::optional<unsigned> Opcode;
  SDLoc DL(N);
  SDNode *LD;
  SDValue Base, Offset, Addr;
  EVT OrigType = N->getValueType(0);

  EVT EltVT = Mem->getMemoryVT();
  unsigned NumElts = 1;
  if (EltVT.isVector()) {
    NumElts = EltVT.getVectorNumElements();
    EltVT = EltVT.getVectorElementType();
    // vectors of 16bits type are loaded/stored as multiples of v2x16 elements.
    if ((EltVT == MVT::f16 && OrigType == MVT::v2f16) ||
        (EltVT == MVT::bf16 && OrigType == MVT::v2bf16) ||
        (EltVT == MVT::i16 && OrigType == MVT::v2i16)) {
      assert(NumElts % 2 == 0 && "Vector must have even number of elements");
      EltVT = OrigType;
      NumElts /= 2;
    } else if (OrigType == MVT::v4i8) {
      EltVT = OrigType;
      NumElts = 1;
    }
  }

  // Build the "promoted" result VTList for the load. If we are really loading
  // i8s, then the return type will be promoted to i16 since we do not expose
  // 8-bit registers in RVGPU.
  EVT NodeVT = (EltVT == MVT::i8) ? MVT::i16 : EltVT;
  SmallVector<EVT, 5> InstVTs;
  for (unsigned i = 0; i != NumElts; ++i) {
    InstVTs.push_back(NodeVT);
  }
  InstVTs.push_back(MVT::Other);
  SDVTList InstVTList = CurDAG->getVTList(InstVTs);

  if (SelectDirectAddr(Op1, Addr)) {
    switch (N->getOpcode()) {
    default:
      return false;
    case ISD::LOAD:
    case ISD::INTRINSIC_W_CHAIN:
      if (IsLDG)
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 RVGPU::INT_PTX_LDG_GLOBAL_i8avar,
                                 RVGPU::INT_PTX_LDG_GLOBAL_i16avar,
                                 RVGPU::INT_PTX_LDG_GLOBAL_i32avar,
                                 RVGPU::INT_PTX_LDG_GLOBAL_i64avar,
                                 RVGPU::INT_PTX_LDG_GLOBAL_f32avar,
                                 RVGPU::INT_PTX_LDG_GLOBAL_f64avar);
      else
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 RVGPU::INT_PTX_LDU_GLOBAL_i8avar,
                                 RVGPU::INT_PTX_LDU_GLOBAL_i16avar,
                                 RVGPU::INT_PTX_LDU_GLOBAL_i32avar,
                                 RVGPU::INT_PTX_LDU_GLOBAL_i64avar,
                                 RVGPU::INT_PTX_LDU_GLOBAL_f32avar,
                                 RVGPU::INT_PTX_LDU_GLOBAL_f64avar);
      break;
    case RVGPUISD::LoadV2:
    case RVGPUISD::LDGV2:
      Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                               RVGPU::INT_PTX_LDG_G_v2i8_ELE_avar,
                               RVGPU::INT_PTX_LDG_G_v2i16_ELE_avar,
                               RVGPU::INT_PTX_LDG_G_v2i32_ELE_avar,
                               RVGPU::INT_PTX_LDG_G_v2i64_ELE_avar,
                               RVGPU::INT_PTX_LDG_G_v2f32_ELE_avar,
                               RVGPU::INT_PTX_LDG_G_v2f64_ELE_avar);
      break;
    case RVGPUISD::LDUV2:
      Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                               RVGPU::INT_PTX_LDU_G_v2i8_ELE_avar,
                               RVGPU::INT_PTX_LDU_G_v2i16_ELE_avar,
                               RVGPU::INT_PTX_LDU_G_v2i32_ELE_avar,
                               RVGPU::INT_PTX_LDU_G_v2i64_ELE_avar,
                               RVGPU::INT_PTX_LDU_G_v2f32_ELE_avar,
                               RVGPU::INT_PTX_LDU_G_v2f64_ELE_avar);
      break;
    case RVGPUISD::LoadV4:
    case RVGPUISD::LDGV4:
      Opcode = pickOpcodeForVT(
          EltVT.getSimpleVT().SimpleTy, RVGPU::INT_PTX_LDG_G_v4i8_ELE_avar,
          RVGPU::INT_PTX_LDG_G_v4i16_ELE_avar,
          RVGPU::INT_PTX_LDG_G_v4i32_ELE_avar, std::nullopt,
          RVGPU::INT_PTX_LDG_G_v4f32_ELE_avar, std::nullopt);
      break;
    case RVGPUISD::LDUV4:
      Opcode = pickOpcodeForVT(
          EltVT.getSimpleVT().SimpleTy, RVGPU::INT_PTX_LDU_G_v4i8_ELE_avar,
          RVGPU::INT_PTX_LDU_G_v4i16_ELE_avar,
          RVGPU::INT_PTX_LDU_G_v4i32_ELE_avar, std::nullopt,
          RVGPU::INT_PTX_LDU_G_v4f32_ELE_avar, std::nullopt);
      break;
    }
    if (!Opcode)
      return false;
    SDValue Ops[] = { Addr, Chain };
    LD = CurDAG->getMachineNode(*Opcode, DL, InstVTList, Ops);
  } else if (TM.is64Bit() ? SelectADDRri64(Op1.getNode(), Op1, Base, Offset)
                          : SelectADDRri(Op1.getNode(), Op1, Base, Offset)) {
    if (TM.is64Bit()) {
      switch (N->getOpcode()) {
      default:
        return false;
      case ISD::LOAD:
      case ISD::INTRINSIC_W_CHAIN:
        if (IsLDG)
          Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                       RVGPU::INT_PTX_LDG_GLOBAL_i8ari64,
                                       RVGPU::INT_PTX_LDG_GLOBAL_i16ari64,
                                       RVGPU::INT_PTX_LDG_GLOBAL_i32ari64,
                                       RVGPU::INT_PTX_LDG_GLOBAL_i64ari64,
                                       RVGPU::INT_PTX_LDG_GLOBAL_f32ari64,
                                       RVGPU::INT_PTX_LDG_GLOBAL_f64ari64);
        else
          Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                       RVGPU::INT_PTX_LDU_GLOBAL_i8ari64,
                                       RVGPU::INT_PTX_LDU_GLOBAL_i16ari64,
                                       RVGPU::INT_PTX_LDU_GLOBAL_i32ari64,
                                       RVGPU::INT_PTX_LDU_GLOBAL_i64ari64,
                                       RVGPU::INT_PTX_LDU_GLOBAL_f32ari64,
                                       RVGPU::INT_PTX_LDU_GLOBAL_f64ari64);
        break;
      case RVGPUISD::LoadV2:
      case RVGPUISD::LDGV2:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                     RVGPU::INT_PTX_LDG_G_v2i8_ELE_ari64,
                                     RVGPU::INT_PTX_LDG_G_v2i16_ELE_ari64,
                                     RVGPU::INT_PTX_LDG_G_v2i32_ELE_ari64,
                                     RVGPU::INT_PTX_LDG_G_v2i64_ELE_ari64,
                                     RVGPU::INT_PTX_LDG_G_v2f32_ELE_ari64,
                                     RVGPU::INT_PTX_LDG_G_v2f64_ELE_ari64);
        break;
      case RVGPUISD::LDUV2:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                     RVGPU::INT_PTX_LDU_G_v2i8_ELE_ari64,
                                     RVGPU::INT_PTX_LDU_G_v2i16_ELE_ari64,
                                     RVGPU::INT_PTX_LDU_G_v2i32_ELE_ari64,
                                     RVGPU::INT_PTX_LDU_G_v2i64_ELE_ari64,
                                     RVGPU::INT_PTX_LDU_G_v2f32_ELE_ari64,
                                     RVGPU::INT_PTX_LDU_G_v2f64_ELE_ari64);
        break;
      case RVGPUISD::LoadV4:
      case RVGPUISD::LDGV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, RVGPU::INT_PTX_LDG_G_v4i8_ELE_ari64,
            RVGPU::INT_PTX_LDG_G_v4i16_ELE_ari64,
            RVGPU::INT_PTX_LDG_G_v4i32_ELE_ari64, std::nullopt,
            RVGPU::INT_PTX_LDG_G_v4f32_ELE_ari64, std::nullopt);
        break;
      case RVGPUISD::LDUV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, RVGPU::INT_PTX_LDU_G_v4i8_ELE_ari64,
            RVGPU::INT_PTX_LDU_G_v4i16_ELE_ari64,
            RVGPU::INT_PTX_LDU_G_v4i32_ELE_ari64, std::nullopt,
            RVGPU::INT_PTX_LDU_G_v4f32_ELE_ari64, std::nullopt);
        break;
      }
    } else {
      switch (N->getOpcode()) {
      default:
        return false;
      case ISD::LOAD:
      case ISD::INTRINSIC_W_CHAIN:
        if (IsLDG)
          Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                   RVGPU::INT_PTX_LDG_GLOBAL_i8ari,
                                   RVGPU::INT_PTX_LDG_GLOBAL_i16ari,
                                   RVGPU::INT_PTX_LDG_GLOBAL_i32ari,
                                   RVGPU::INT_PTX_LDG_GLOBAL_i64ari,
                                   RVGPU::INT_PTX_LDG_GLOBAL_f32ari,
                                   RVGPU::INT_PTX_LDG_GLOBAL_f64ari);
        else
          Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                   RVGPU::INT_PTX_LDU_GLOBAL_i8ari,
                                   RVGPU::INT_PTX_LDU_GLOBAL_i16ari,
                                   RVGPU::INT_PTX_LDU_GLOBAL_i32ari,
                                   RVGPU::INT_PTX_LDU_GLOBAL_i64ari,
                                   RVGPU::INT_PTX_LDU_GLOBAL_f32ari,
                                   RVGPU::INT_PTX_LDU_GLOBAL_f64ari);
        break;
      case RVGPUISD::LoadV2:
      case RVGPUISD::LDGV2:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 RVGPU::INT_PTX_LDG_G_v2i8_ELE_ari32,
                                 RVGPU::INT_PTX_LDG_G_v2i16_ELE_ari32,
                                 RVGPU::INT_PTX_LDG_G_v2i32_ELE_ari32,
                                 RVGPU::INT_PTX_LDG_G_v2i64_ELE_ari32,
                                 RVGPU::INT_PTX_LDG_G_v2f32_ELE_ari32,
                                 RVGPU::INT_PTX_LDG_G_v2f64_ELE_ari32);
        break;
      case RVGPUISD::LDUV2:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 RVGPU::INT_PTX_LDU_G_v2i8_ELE_ari32,
                                 RVGPU::INT_PTX_LDU_G_v2i16_ELE_ari32,
                                 RVGPU::INT_PTX_LDU_G_v2i32_ELE_ari32,
                                 RVGPU::INT_PTX_LDU_G_v2i64_ELE_ari32,
                                 RVGPU::INT_PTX_LDU_G_v2f32_ELE_ari32,
                                 RVGPU::INT_PTX_LDU_G_v2f64_ELE_ari32);
        break;
      case RVGPUISD::LoadV4:
      case RVGPUISD::LDGV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, RVGPU::INT_PTX_LDG_G_v4i8_ELE_ari32,
            RVGPU::INT_PTX_LDG_G_v4i16_ELE_ari32,
            RVGPU::INT_PTX_LDG_G_v4i32_ELE_ari32, std::nullopt,
            RVGPU::INT_PTX_LDG_G_v4f32_ELE_ari32, std::nullopt);
        break;
      case RVGPUISD::LDUV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, RVGPU::INT_PTX_LDU_G_v4i8_ELE_ari32,
            RVGPU::INT_PTX_LDU_G_v4i16_ELE_ari32,
            RVGPU::INT_PTX_LDU_G_v4i32_ELE_ari32, std::nullopt,
            RVGPU::INT_PTX_LDU_G_v4f32_ELE_ari32, std::nullopt);
        break;
      }
    }
    if (!Opcode)
      return false;
    SDValue Ops[] = {Base, Offset, Chain};
    LD = CurDAG->getMachineNode(*Opcode, DL, InstVTList, Ops);
  } else {
    if (TM.is64Bit()) {
      switch (N->getOpcode()) {
      default:
        return false;
      case ISD::LOAD:
      case ISD::INTRINSIC_W_CHAIN:
        if (IsLDG)
          Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                       RVGPU::INT_PTX_LDG_GLOBAL_i8areg64,
                                       RVGPU::INT_PTX_LDG_GLOBAL_i16areg64,
                                       RVGPU::INT_PTX_LDG_GLOBAL_i32areg64,
                                       RVGPU::INT_PTX_LDG_GLOBAL_i64areg64,
                                       RVGPU::INT_PTX_LDG_GLOBAL_f32areg64,
                                       RVGPU::INT_PTX_LDG_GLOBAL_f64areg64);
        else
          Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                       RVGPU::INT_PTX_LDU_GLOBAL_i8areg64,
                                       RVGPU::INT_PTX_LDU_GLOBAL_i16areg64,
                                       RVGPU::INT_PTX_LDU_GLOBAL_i32areg64,
                                       RVGPU::INT_PTX_LDU_GLOBAL_i64areg64,
                                       RVGPU::INT_PTX_LDU_GLOBAL_f32areg64,
                                       RVGPU::INT_PTX_LDU_GLOBAL_f64areg64);
        break;
      case RVGPUISD::LoadV2:
      case RVGPUISD::LDGV2:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                     RVGPU::INT_PTX_LDG_G_v2i8_ELE_areg64,
                                     RVGPU::INT_PTX_LDG_G_v2i16_ELE_areg64,
                                     RVGPU::INT_PTX_LDG_G_v2i32_ELE_areg64,
                                     RVGPU::INT_PTX_LDG_G_v2i64_ELE_areg64,
                                     RVGPU::INT_PTX_LDG_G_v2f32_ELE_areg64,
                                     RVGPU::INT_PTX_LDG_G_v2f64_ELE_areg64);
        break;
      case RVGPUISD::LDUV2:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                     RVGPU::INT_PTX_LDU_G_v2i8_ELE_areg64,
                                     RVGPU::INT_PTX_LDU_G_v2i16_ELE_areg64,
                                     RVGPU::INT_PTX_LDU_G_v2i32_ELE_areg64,
                                     RVGPU::INT_PTX_LDU_G_v2i64_ELE_areg64,
                                     RVGPU::INT_PTX_LDU_G_v2f32_ELE_areg64,
                                     RVGPU::INT_PTX_LDU_G_v2f64_ELE_areg64);
        break;
      case RVGPUISD::LoadV4:
      case RVGPUISD::LDGV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, RVGPU::INT_PTX_LDG_G_v4i8_ELE_areg64,
            RVGPU::INT_PTX_LDG_G_v4i16_ELE_areg64,
            RVGPU::INT_PTX_LDG_G_v4i32_ELE_areg64, std::nullopt,
            RVGPU::INT_PTX_LDG_G_v4f32_ELE_areg64, std::nullopt);
        break;
      case RVGPUISD::LDUV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, RVGPU::INT_PTX_LDU_G_v4i8_ELE_areg64,
            RVGPU::INT_PTX_LDU_G_v4i16_ELE_areg64,
            RVGPU::INT_PTX_LDU_G_v4i32_ELE_areg64, std::nullopt,
            RVGPU::INT_PTX_LDU_G_v4f32_ELE_areg64, std::nullopt);
        break;
      }
    } else {
      switch (N->getOpcode()) {
      default:
        return false;
      case ISD::LOAD:
      case ISD::INTRINSIC_W_CHAIN:
        if (IsLDG)
          Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                   RVGPU::INT_PTX_LDG_GLOBAL_i8areg,
                                   RVGPU::INT_PTX_LDG_GLOBAL_i16areg,
                                   RVGPU::INT_PTX_LDG_GLOBAL_i32areg,
                                   RVGPU::INT_PTX_LDG_GLOBAL_i64areg,
                                   RVGPU::INT_PTX_LDG_GLOBAL_f32areg,
                                   RVGPU::INT_PTX_LDG_GLOBAL_f64areg);
        else
          Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                   RVGPU::INT_PTX_LDU_GLOBAL_i8areg,
                                   RVGPU::INT_PTX_LDU_GLOBAL_i16areg,
                                   RVGPU::INT_PTX_LDU_GLOBAL_i32areg,
                                   RVGPU::INT_PTX_LDU_GLOBAL_i64areg,
                                   RVGPU::INT_PTX_LDU_GLOBAL_f32areg,
                                   RVGPU::INT_PTX_LDU_GLOBAL_f64areg);
        break;
      case RVGPUISD::LoadV2:
      case RVGPUISD::LDGV2:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 RVGPU::INT_PTX_LDG_G_v2i8_ELE_areg32,
                                 RVGPU::INT_PTX_LDG_G_v2i16_ELE_areg32,
                                 RVGPU::INT_PTX_LDG_G_v2i32_ELE_areg32,
                                 RVGPU::INT_PTX_LDG_G_v2i64_ELE_areg32,
                                 RVGPU::INT_PTX_LDG_G_v2f32_ELE_areg32,
                                 RVGPU::INT_PTX_LDG_G_v2f64_ELE_areg32);
        break;
      case RVGPUISD::LDUV2:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 RVGPU::INT_PTX_LDU_G_v2i8_ELE_areg32,
                                 RVGPU::INT_PTX_LDU_G_v2i16_ELE_areg32,
                                 RVGPU::INT_PTX_LDU_G_v2i32_ELE_areg32,
                                 RVGPU::INT_PTX_LDU_G_v2i64_ELE_areg32,
                                 RVGPU::INT_PTX_LDU_G_v2f32_ELE_areg32,
                                 RVGPU::INT_PTX_LDU_G_v2f64_ELE_areg32);
        break;
      case RVGPUISD::LoadV4:
      case RVGPUISD::LDGV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, RVGPU::INT_PTX_LDG_G_v4i8_ELE_areg32,
            RVGPU::INT_PTX_LDG_G_v4i16_ELE_areg32,
            RVGPU::INT_PTX_LDG_G_v4i32_ELE_areg32, std::nullopt,
            RVGPU::INT_PTX_LDG_G_v4f32_ELE_areg32, std::nullopt);
        break;
      case RVGPUISD::LDUV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, RVGPU::INT_PTX_LDU_G_v4i8_ELE_areg32,
            RVGPU::INT_PTX_LDU_G_v4i16_ELE_areg32,
            RVGPU::INT_PTX_LDU_G_v4i32_ELE_areg32, std::nullopt,
            RVGPU::INT_PTX_LDU_G_v4f32_ELE_areg32, std::nullopt);
        break;
      }
    }
    if (!Opcode)
      return false;
    SDValue Ops[] = { Op1, Chain };
    LD = CurDAG->getMachineNode(*Opcode, DL, InstVTList, Ops);
  }

  // For automatic generation of LDG (through SelectLoad[Vector], not the
  // intrinsics), we may have an extending load like:
  //
  //   i32,ch = load<LD1[%data1(addrspace=1)], zext from i8> t0, t7, undef:i64
  //
  // In this case, the matching logic above will select a load for the original
  // memory type (in this case, i8) and our types will not match (the node needs
  // to return an i32 in this case). Our LDG/LDU nodes do not support the
  // concept of sign-/zero-extension, so emulate it here by adding an explicit
  // CVT instruction. Ptxas should clean up any redundancies here.

  LoadSDNode *LdNode = dyn_cast<LoadSDNode>(N);

  if (OrigType != EltVT &&
      (LdNode || (OrigType.isFloatingPoint() && EltVT.isFloatingPoint()))) {
    // We have an extending-load. The instruction we selected operates on the
    // smaller type, but the SDNode we are replacing has the larger type. We
    // need to emit a CVT to make the types match.
    unsigned CvtOpc =
        GetConvertOpcode(OrigType.getSimpleVT(), EltVT.getSimpleVT(), LdNode);

    // For each output value, apply the manual sign/zero-extension and make sure
    // all users of the load go through that CVT.
    for (unsigned i = 0; i != NumElts; ++i) {
      SDValue Res(LD, i);
      SDValue OrigVal(N, i);

      SDNode *CvtNode =
        CurDAG->getMachineNode(CvtOpc, DL, OrigType, Res,
                               CurDAG->getTargetConstant(RVGPU::PTXCvtMode::NONE,
                                                         DL, MVT::i32));
      ReplaceUses(OrigVal, SDValue(CvtNode, 0));
    }
  }

  ReplaceNode(N, LD);
  return true;
}

bool RVGPUDAGToDAGISel::tryStore(SDNode *N) {
  SDLoc dl(N);
  MemSDNode *ST = cast<MemSDNode>(N);
  assert(ST->writeMem() && "Expected store");
  StoreSDNode *PlainStore = dyn_cast<StoreSDNode>(N);
  AtomicSDNode *AtomicStore = dyn_cast<AtomicSDNode>(N);
  assert((PlainStore || AtomicStore) && "Expected store");
  EVT StoreVT = ST->getMemoryVT();
  SDNode *RVGPUST = nullptr;

  // do not support pre/post inc/dec
  if (PlainStore && PlainStore->isIndexed())
    return false;

  if (!StoreVT.isSimple())
    return false;

  AtomicOrdering Ordering = ST->getSuccessOrdering();
  // In order to lower atomic loads with stronger guarantees we would need to
  // use store.release or insert fences. However these features were only added
  // with PTX ISA 6.0 / sm_70.
  // TODO: Check if we can actually use the new instructions and implement them.
  if (isStrongerThanMonotonic(Ordering))
    return false;

  // Address Space Setting
  unsigned int CodeAddrSpace = getCodeAddrSpace(ST);
  unsigned int PointerSize =
      CurDAG->getDataLayout().getPointerSizeInBits(ST->getAddressSpace());

  // Volatile Setting
  // - .volatile is only available for .global and .shared
  // - .volatile has the same memory synchronization semantics as .relaxed.sys
  bool isVolatile = ST->isVolatile() || Ordering == AtomicOrdering::Monotonic;
  if (CodeAddrSpace != RVGPU::PTXLdStInstCode::GLOBAL &&
      CodeAddrSpace != RVGPU::PTXLdStInstCode::SHARED &&
      CodeAddrSpace != RVGPU::PTXLdStInstCode::GENERIC)
    isVolatile = false;

  // Vector Setting
  MVT SimpleVT = StoreVT.getSimpleVT();
  unsigned vecType = RVGPU::PTXLdStInstCode::Scalar;

  // Type Setting: toType + toTypeWidth
  // - for integer type, always use 'u'
  //
  MVT ScalarVT = SimpleVT.getScalarType();
  unsigned toTypeWidth = ScalarVT.getSizeInBits();
  if (SimpleVT.isVector()) {
    assert((Isv2x16VT(StoreVT) || StoreVT == MVT::v4i8) &&
           "Unexpected vector type");
    // v2x16 is stored using st.b32
    toTypeWidth = 32;
  }

  unsigned int toType = getLdStRegType(ScalarVT);

  // Create the machine instruction DAG
  SDValue Chain = ST->getChain();
  SDValue Value = PlainStore ? PlainStore->getValue() : AtomicStore->getVal();
  SDValue BasePtr = ST->getBasePtr();
  SDValue Addr;
  SDValue Offset, Base;
  std::optional<unsigned> Opcode;
  MVT::SimpleValueType SourceVT =
      Value.getNode()->getSimpleValueType(0).SimpleTy;

  if (SelectDirectAddr(BasePtr, Addr)) {
    Opcode = pickOpcodeForVT(SourceVT, RVGPU::ST_i8_avar, RVGPU::ST_i16_avar,
                             RVGPU::ST_i32_avar, RVGPU::ST_i64_avar,
                             RVGPU::ST_f32_avar, RVGPU::ST_f64_avar);
    if (!Opcode)
      return false;
    SDValue Ops[] = {Value,
                     getI32Imm(isVolatile, dl),
                     getI32Imm(CodeAddrSpace, dl),
                     getI32Imm(vecType, dl),
                     getI32Imm(toType, dl),
                     getI32Imm(toTypeWidth, dl),
                     Addr,
                     Chain};
    RVGPUST = CurDAG->getMachineNode(*Opcode, dl, MVT::Other, Ops);
  } else if (PointerSize == 64
                 ? SelectADDRsi64(BasePtr.getNode(), BasePtr, Base, Offset)
                 : SelectADDRsi(BasePtr.getNode(), BasePtr, Base, Offset)) {
    Opcode = pickOpcodeForVT(SourceVT, RVGPU::ST_i8_asi, RVGPU::ST_i16_asi,
                             RVGPU::ST_i32_asi, RVGPU::ST_i64_asi,
                             RVGPU::ST_f32_asi, RVGPU::ST_f64_asi);
    if (!Opcode)
      return false;
    SDValue Ops[] = {Value,
                     getI32Imm(isVolatile, dl),
                     getI32Imm(CodeAddrSpace, dl),
                     getI32Imm(vecType, dl),
                     getI32Imm(toType, dl),
                     getI32Imm(toTypeWidth, dl),
                     Base,
                     Offset,
                     Chain};
    RVGPUST = CurDAG->getMachineNode(*Opcode, dl, MVT::Other, Ops);
  } else if (PointerSize == 64
                 ? SelectADDRri64(BasePtr.getNode(), BasePtr, Base, Offset)
                 : SelectADDRri(BasePtr.getNode(), BasePtr, Base, Offset)) {
    if (PointerSize == 64)
      Opcode =
          pickOpcodeForVT(SourceVT, RVGPU::ST_i8_ari_64, RVGPU::ST_i16_ari_64,
                          RVGPU::ST_i32_ari_64, RVGPU::ST_i64_ari_64,
                          RVGPU::ST_f32_ari_64, RVGPU::ST_f64_ari_64);
    else
      Opcode = pickOpcodeForVT(SourceVT, RVGPU::ST_i8_ari, RVGPU::ST_i16_ari,
                               RVGPU::ST_i32_ari, RVGPU::ST_i64_ari,
                               RVGPU::ST_f32_ari, RVGPU::ST_f64_ari);
    if (!Opcode)
      return false;

    SDValue Ops[] = {Value,
                     getI32Imm(isVolatile, dl),
                     getI32Imm(CodeAddrSpace, dl),
                     getI32Imm(vecType, dl),
                     getI32Imm(toType, dl),
                     getI32Imm(toTypeWidth, dl),
                     Base,
                     Offset,
                     Chain};
    RVGPUST = CurDAG->getMachineNode(*Opcode, dl, MVT::Other, Ops);
  } else {
    if (PointerSize == 64)
      Opcode =
          pickOpcodeForVT(SourceVT, RVGPU::ST_i8_areg_64, RVGPU::ST_i16_areg_64,
                          RVGPU::ST_i32_areg_64, RVGPU::ST_i64_areg_64,
                          RVGPU::ST_f32_areg_64, RVGPU::ST_f64_areg_64);
    else
      Opcode = pickOpcodeForVT(SourceVT, RVGPU::ST_i8_areg, RVGPU::ST_i16_areg,
                               RVGPU::ST_i32_areg, RVGPU::ST_i64_areg,
                               RVGPU::ST_f32_areg, RVGPU::ST_f64_areg);
    if (!Opcode)
      return false;
    SDValue Ops[] = {Value,
                     getI32Imm(isVolatile, dl),
                     getI32Imm(CodeAddrSpace, dl),
                     getI32Imm(vecType, dl),
                     getI32Imm(toType, dl),
                     getI32Imm(toTypeWidth, dl),
                     BasePtr,
                     Chain};
    RVGPUST = CurDAG->getMachineNode(*Opcode, dl, MVT::Other, Ops);
  }

  if (!RVGPUST)
    return false;

  MachineMemOperand *MemRef = cast<MemSDNode>(N)->getMemOperand();
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(RVGPUST), {MemRef});
  ReplaceNode(N, RVGPUST);
  return true;
}

bool RVGPUDAGToDAGISel::tryStoreVector(SDNode *N) {
  SDValue Chain = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);
  SDValue Addr, Offset, Base;
  std::optional<unsigned> Opcode;
  SDLoc DL(N);
  SDNode *ST;
  EVT EltVT = Op1.getValueType();
  MemSDNode *MemSD = cast<MemSDNode>(N);
  EVT StoreVT = MemSD->getMemoryVT();

  // Address Space Setting
  unsigned CodeAddrSpace = getCodeAddrSpace(MemSD);
  if (CodeAddrSpace == RVGPU::PTXLdStInstCode::CONSTANT) {
    report_fatal_error("Cannot store to pointer that points to constant "
                       "memory space");
  }
  unsigned int PointerSize =
      CurDAG->getDataLayout().getPointerSizeInBits(MemSD->getAddressSpace());

  // Volatile Setting
  // - .volatile is only availalble for .global and .shared
  bool IsVolatile = MemSD->isVolatile();
  if (CodeAddrSpace != RVGPU::PTXLdStInstCode::GLOBAL &&
      CodeAddrSpace != RVGPU::PTXLdStInstCode::SHARED &&
      CodeAddrSpace != RVGPU::PTXLdStInstCode::GENERIC)
    IsVolatile = false;

  // Type Setting: toType + toTypeWidth
  // - for integer type, always use 'u'
  assert(StoreVT.isSimple() && "Store value is not simple");
  MVT ScalarVT = StoreVT.getSimpleVT().getScalarType();
  unsigned ToTypeWidth = ScalarVT.getSizeInBits();
  unsigned ToType = getLdStRegType(ScalarVT);

  SmallVector<SDValue, 12> StOps;
  SDValue N2;
  unsigned VecType;

  switch (N->getOpcode()) {
  case RVGPUISD::StoreV2:
    VecType = RVGPU::PTXLdStInstCode::V2;
    StOps.push_back(N->getOperand(1));
    StOps.push_back(N->getOperand(2));
    N2 = N->getOperand(3);
    break;
  case RVGPUISD::StoreV4:
    VecType = RVGPU::PTXLdStInstCode::V4;
    StOps.push_back(N->getOperand(1));
    StOps.push_back(N->getOperand(2));
    StOps.push_back(N->getOperand(3));
    StOps.push_back(N->getOperand(4));
    N2 = N->getOperand(5);
    break;
  default:
    return false;
  }

  // v8x16 is a special case. PTX doesn't have st.v8.x16
  // instruction. Instead, we split the vector into v2x16 chunks and
  // store them with st.v4.b32.
  if (Isv2x16VT(EltVT)) {
    assert(N->getOpcode() == RVGPUISD::StoreV4 && "Unexpected load opcode.");
    EltVT = MVT::i32;
    ToType = RVGPU::PTXLdStInstCode::Untyped;
    ToTypeWidth = 32;
  }

  StOps.push_back(getI32Imm(IsVolatile, DL));
  StOps.push_back(getI32Imm(CodeAddrSpace, DL));
  StOps.push_back(getI32Imm(VecType, DL));
  StOps.push_back(getI32Imm(ToType, DL));
  StOps.push_back(getI32Imm(ToTypeWidth, DL));

  if (SelectDirectAddr(N2, Addr)) {
    switch (N->getOpcode()) {
    default:
      return false;
    case RVGPUISD::StoreV2:
      Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                               RVGPU::STV_i8_v2_avar, RVGPU::STV_i16_v2_avar,
                               RVGPU::STV_i32_v2_avar, RVGPU::STV_i64_v2_avar,
                               RVGPU::STV_f32_v2_avar, RVGPU::STV_f64_v2_avar);
      break;
    case RVGPUISD::StoreV4:
      Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                               RVGPU::STV_i8_v4_avar, RVGPU::STV_i16_v4_avar,
                               RVGPU::STV_i32_v4_avar, std::nullopt,
                               RVGPU::STV_f32_v4_avar, std::nullopt);
      break;
    }
    StOps.push_back(Addr);
  } else if (PointerSize == 64 ? SelectADDRsi64(N2.getNode(), N2, Base, Offset)
                               : SelectADDRsi(N2.getNode(), N2, Base, Offset)) {
    switch (N->getOpcode()) {
    default:
      return false;
    case RVGPUISD::StoreV2:
      Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                               RVGPU::STV_i8_v2_asi, RVGPU::STV_i16_v2_asi,
                               RVGPU::STV_i32_v2_asi, RVGPU::STV_i64_v2_asi,
                               RVGPU::STV_f32_v2_asi, RVGPU::STV_f64_v2_asi);
      break;
    case RVGPUISD::StoreV4:
      Opcode =
          pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy, RVGPU::STV_i8_v4_asi,
                          RVGPU::STV_i16_v4_asi, RVGPU::STV_i32_v4_asi,
                          std::nullopt, RVGPU::STV_f32_v4_asi, std::nullopt);
      break;
    }
    StOps.push_back(Base);
    StOps.push_back(Offset);
  } else if (PointerSize == 64 ? SelectADDRri64(N2.getNode(), N2, Base, Offset)
                               : SelectADDRri(N2.getNode(), N2, Base, Offset)) {
    if (PointerSize == 64) {
      switch (N->getOpcode()) {
      default:
        return false;
      case RVGPUISD::StoreV2:
        Opcode =
            pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                            RVGPU::STV_i8_v2_ari_64, RVGPU::STV_i16_v2_ari_64,
                            RVGPU::STV_i32_v2_ari_64, RVGPU::STV_i64_v2_ari_64,
                            RVGPU::STV_f32_v2_ari_64, RVGPU::STV_f64_v2_ari_64);
        break;
      case RVGPUISD::StoreV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, RVGPU::STV_i8_v4_ari_64,
            RVGPU::STV_i16_v4_ari_64, RVGPU::STV_i32_v4_ari_64, std::nullopt,
            RVGPU::STV_f32_v4_ari_64, std::nullopt);
        break;
      }
    } else {
      switch (N->getOpcode()) {
      default:
        return false;
      case RVGPUISD::StoreV2:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 RVGPU::STV_i8_v2_ari, RVGPU::STV_i16_v2_ari,
                                 RVGPU::STV_i32_v2_ari, RVGPU::STV_i64_v2_ari,
                                 RVGPU::STV_f32_v2_ari, RVGPU::STV_f64_v2_ari);
        break;
      case RVGPUISD::StoreV4:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 RVGPU::STV_i8_v4_ari, RVGPU::STV_i16_v4_ari,
                                 RVGPU::STV_i32_v4_ari, std::nullopt,
                                 RVGPU::STV_f32_v4_ari, std::nullopt);
        break;
      }
    }
    StOps.push_back(Base);
    StOps.push_back(Offset);
  } else {
    if (PointerSize == 64) {
      switch (N->getOpcode()) {
      default:
        return false;
      case RVGPUISD::StoreV2:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, RVGPU::STV_i8_v2_areg_64,
            RVGPU::STV_i16_v2_areg_64, RVGPU::STV_i32_v2_areg_64,
            RVGPU::STV_i64_v2_areg_64, RVGPU::STV_f32_v2_areg_64,
            RVGPU::STV_f64_v2_areg_64);
        break;
      case RVGPUISD::StoreV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, RVGPU::STV_i8_v4_areg_64,
            RVGPU::STV_i16_v4_areg_64, RVGPU::STV_i32_v4_areg_64, std::nullopt,
            RVGPU::STV_f32_v4_areg_64, std::nullopt);
        break;
      }
    } else {
      switch (N->getOpcode()) {
      default:
        return false;
      case RVGPUISD::StoreV2:
        Opcode =
            pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy, RVGPU::STV_i8_v2_areg,
                            RVGPU::STV_i16_v2_areg, RVGPU::STV_i32_v2_areg,
                            RVGPU::STV_i64_v2_areg, RVGPU::STV_f32_v2_areg,
                            RVGPU::STV_f64_v2_areg);
        break;
      case RVGPUISD::StoreV4:
        Opcode =
            pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy, RVGPU::STV_i8_v4_areg,
                            RVGPU::STV_i16_v4_areg, RVGPU::STV_i32_v4_areg,
                            std::nullopt, RVGPU::STV_f32_v4_areg, std::nullopt);
        break;
      }
    }
    StOps.push_back(N2);
  }

  if (!Opcode)
    return false;

  StOps.push_back(Chain);

  ST = CurDAG->getMachineNode(*Opcode, DL, MVT::Other, StOps);

  MachineMemOperand *MemRef = cast<MemSDNode>(N)->getMemOperand();
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(ST), {MemRef});

  ReplaceNode(N, ST);
  return true;
}

bool RVGPUDAGToDAGISel::tryLoadParam(SDNode *Node) {
  SDValue Chain = Node->getOperand(0);
  SDValue Offset = Node->getOperand(2);
  SDValue Glue = Node->getOperand(3);
  SDLoc DL(Node);
  MemSDNode *Mem = cast<MemSDNode>(Node);

  unsigned VecSize;
  switch (Node->getOpcode()) {
  default:
    return false;
  case RVGPUISD::LoadParam:
    VecSize = 1;
    break;
  case RVGPUISD::LoadParamV2:
    VecSize = 2;
    break;
  case RVGPUISD::LoadParamV4:
    VecSize = 4;
    break;
  }

  EVT EltVT = Node->getValueType(0);
  EVT MemVT = Mem->getMemoryVT();

  std::optional<unsigned> Opcode;

  switch (VecSize) {
  default:
    return false;
  case 1:
    Opcode = pickOpcodeForVT(MemVT.getSimpleVT().SimpleTy,
                             RVGPU::LoadParamMemI8, RVGPU::LoadParamMemI16,
                             RVGPU::LoadParamMemI32, RVGPU::LoadParamMemI64,
                             RVGPU::LoadParamMemF32, RVGPU::LoadParamMemF64);
    break;
  case 2:
    Opcode =
        pickOpcodeForVT(MemVT.getSimpleVT().SimpleTy, RVGPU::LoadParamMemV2I8,
                        RVGPU::LoadParamMemV2I16, RVGPU::LoadParamMemV2I32,
                        RVGPU::LoadParamMemV2I64, RVGPU::LoadParamMemV2F32,
                        RVGPU::LoadParamMemV2F64);
    break;
  case 4:
    Opcode =
        pickOpcodeForVT(MemVT.getSimpleVT().SimpleTy, RVGPU::LoadParamMemV4I8,
                        RVGPU::LoadParamMemV4I16, RVGPU::LoadParamMemV4I32,
                        std::nullopt, RVGPU::LoadParamMemV4F32, std::nullopt);
    break;
  }
  if (!Opcode)
    return false;

  SDVTList VTs;
  if (VecSize == 1) {
    VTs = CurDAG->getVTList(EltVT, MVT::Other, MVT::Glue);
  } else if (VecSize == 2) {
    VTs = CurDAG->getVTList(EltVT, EltVT, MVT::Other, MVT::Glue);
  } else {
    EVT EVTs[] = { EltVT, EltVT, EltVT, EltVT, MVT::Other, MVT::Glue };
    VTs = CurDAG->getVTList(EVTs);
  }

  unsigned OffsetVal = cast<ConstantSDNode>(Offset)->getZExtValue();

  SmallVector<SDValue, 2> Ops;
  Ops.push_back(CurDAG->getTargetConstant(OffsetVal, DL, MVT::i32));
  Ops.push_back(Chain);
  Ops.push_back(Glue);

  ReplaceNode(Node, CurDAG->getMachineNode(*Opcode, DL, VTs, Ops));
  return true;
}

bool RVGPUDAGToDAGISel::tryStoreRetval(SDNode *N) {
  SDLoc DL(N);
  SDValue Chain = N->getOperand(0);
  SDValue Offset = N->getOperand(1);
  unsigned OffsetVal = cast<ConstantSDNode>(Offset)->getZExtValue();
  MemSDNode *Mem = cast<MemSDNode>(N);

  // How many elements do we have?
  unsigned NumElts = 1;
  switch (N->getOpcode()) {
  default:
    return false;
  case RVGPUISD::StoreRetval:
    NumElts = 1;
    break;
  case RVGPUISD::StoreRetvalV2:
    NumElts = 2;
    break;
  case RVGPUISD::StoreRetvalV4:
    NumElts = 4;
    break;
  }

  // Build vector of operands
  SmallVector<SDValue, 6> Ops;
  for (unsigned i = 0; i < NumElts; ++i)
    Ops.push_back(N->getOperand(i + 2));
  Ops.push_back(CurDAG->getTargetConstant(OffsetVal, DL, MVT::i32));
  Ops.push_back(Chain);

  // Determine target opcode
  // If we have an i1, use an 8-bit store. The lowering code in
  // RVGPUISelLowering will have already emitted an upcast.
  std::optional<unsigned> Opcode = 0;
  switch (NumElts) {
  default:
    return false;
  case 1:
    Opcode = pickOpcodeForVT(Mem->getMemoryVT().getSimpleVT().SimpleTy,
                             RVGPU::StoreRetvalI8, RVGPU::StoreRetvalI16,
                             RVGPU::StoreRetvalI32, RVGPU::StoreRetvalI64,
                             RVGPU::StoreRetvalF32, RVGPU::StoreRetvalF64);
    break;
  case 2:
    Opcode = pickOpcodeForVT(Mem->getMemoryVT().getSimpleVT().SimpleTy,
                             RVGPU::StoreRetvalV2I8, RVGPU::StoreRetvalV2I16,
                             RVGPU::StoreRetvalV2I32, RVGPU::StoreRetvalV2I64,
                             RVGPU::StoreRetvalV2F32, RVGPU::StoreRetvalV2F64);
    break;
  case 4:
    Opcode = pickOpcodeForVT(Mem->getMemoryVT().getSimpleVT().SimpleTy,
                             RVGPU::StoreRetvalV4I8, RVGPU::StoreRetvalV4I16,
                             RVGPU::StoreRetvalV4I32, std::nullopt,
                             RVGPU::StoreRetvalV4F32, std::nullopt);
    break;
  }
  if (!Opcode)
    return false;

  SDNode *Ret = CurDAG->getMachineNode(*Opcode, DL, MVT::Other, Ops);
  MachineMemOperand *MemRef = cast<MemSDNode>(N)->getMemOperand();
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(Ret), {MemRef});

  ReplaceNode(N, Ret);
  return true;
}

bool RVGPUDAGToDAGISel::tryStoreParam(SDNode *N) {
  SDLoc DL(N);
  SDValue Chain = N->getOperand(0);
  SDValue Param = N->getOperand(1);
  unsigned ParamVal = cast<ConstantSDNode>(Param)->getZExtValue();
  SDValue Offset = N->getOperand(2);
  unsigned OffsetVal = cast<ConstantSDNode>(Offset)->getZExtValue();
  MemSDNode *Mem = cast<MemSDNode>(N);
  SDValue Glue = N->getOperand(N->getNumOperands() - 1);

  // How many elements do we have?
  unsigned NumElts = 1;
  switch (N->getOpcode()) {
  default:
    return false;
  case RVGPUISD::StoreParamU32:
  case RVGPUISD::StoreParamS32:
  case RVGPUISD::StoreParam:
    NumElts = 1;
    break;
  case RVGPUISD::StoreParamV2:
    NumElts = 2;
    break;
  case RVGPUISD::StoreParamV4:
    NumElts = 4;
    break;
  }

  // Build vector of operands
  SmallVector<SDValue, 8> Ops;
  for (unsigned i = 0; i < NumElts; ++i)
    Ops.push_back(N->getOperand(i + 3));
  Ops.push_back(CurDAG->getTargetConstant(ParamVal, DL, MVT::i32));
  Ops.push_back(CurDAG->getTargetConstant(OffsetVal, DL, MVT::i32));
  Ops.push_back(Chain);
  Ops.push_back(Glue);

  // Determine target opcode
  // If we have an i1, use an 8-bit store. The lowering code in
  // RVGPUISelLowering will have already emitted an upcast.
  std::optional<unsigned> Opcode = 0;
  switch (N->getOpcode()) {
  default:
    switch (NumElts) {
    default:
      return false;
    case 1:
      Opcode = pickOpcodeForVT(Mem->getMemoryVT().getSimpleVT().SimpleTy,
                               RVGPU::StoreParamI8, RVGPU::StoreParamI16,
                               RVGPU::StoreParamI32, RVGPU::StoreParamI64,
                               RVGPU::StoreParamF32, RVGPU::StoreParamF64);
      break;
    case 2:
      Opcode = pickOpcodeForVT(Mem->getMemoryVT().getSimpleVT().SimpleTy,
                               RVGPU::StoreParamV2I8, RVGPU::StoreParamV2I16,
                               RVGPU::StoreParamV2I32, RVGPU::StoreParamV2I64,
                               RVGPU::StoreParamV2F32, RVGPU::StoreParamV2F64);
      break;
    case 4:
      Opcode = pickOpcodeForVT(Mem->getMemoryVT().getSimpleVT().SimpleTy,
                               RVGPU::StoreParamV4I8, RVGPU::StoreParamV4I16,
                               RVGPU::StoreParamV4I32, std::nullopt,
                               RVGPU::StoreParamV4F32, std::nullopt);
      break;
    }
    if (!Opcode)
      return false;
    break;
  // Special case: if we have a sign-extend/zero-extend node, insert the
  // conversion instruction first, and use that as the value operand to
  // the selected StoreParam node.
  case RVGPUISD::StoreParamU32: {
    Opcode = RVGPU::StoreParamI32;
    SDValue CvtNone = CurDAG->getTargetConstant(RVGPU::PTXCvtMode::NONE, DL,
                                                MVT::i32);
    SDNode *Cvt = CurDAG->getMachineNode(RVGPU::CVT_u32_u16, DL,
                                         MVT::i32, Ops[0], CvtNone);
    Ops[0] = SDValue(Cvt, 0);
    break;
  }
  case RVGPUISD::StoreParamS32: {
    Opcode = RVGPU::StoreParamI32;
    SDValue CvtNone = CurDAG->getTargetConstant(RVGPU::PTXCvtMode::NONE, DL,
                                                MVT::i32);
    SDNode *Cvt = CurDAG->getMachineNode(RVGPU::CVT_s32_s16, DL,
                                         MVT::i32, Ops[0], CvtNone);
    Ops[0] = SDValue(Cvt, 0);
    break;
  }
  }

  SDVTList RetVTs = CurDAG->getVTList(MVT::Other, MVT::Glue);
  SDNode *Ret = CurDAG->getMachineNode(*Opcode, DL, RetVTs, Ops);
  MachineMemOperand *MemRef = cast<MemSDNode>(N)->getMemOperand();
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(Ret), {MemRef});

  ReplaceNode(N, Ret);
  return true;
}

bool RVGPUDAGToDAGISel::tryTextureIntrinsic(SDNode *N) {
  unsigned Opc = 0;

  switch (N->getOpcode()) {
  default: return false;
  case RVGPUISD::Tex1DFloatS32:
    Opc = RVGPU::TEX_1D_F32_S32_RR;
    break;
  case RVGPUISD::Tex1DFloatFloat:
    Opc = RVGPU::TEX_1D_F32_F32_RR;
    break;
  case RVGPUISD::Tex1DFloatFloatLevel:
    Opc = RVGPU::TEX_1D_F32_F32_LEVEL_RR;
    break;
  case RVGPUISD::Tex1DFloatFloatGrad:
    Opc = RVGPU::TEX_1D_F32_F32_GRAD_RR;
    break;
  case RVGPUISD::Tex1DS32S32:
    Opc = RVGPU::TEX_1D_S32_S32_RR;
    break;
  case RVGPUISD::Tex1DS32Float:
    Opc = RVGPU::TEX_1D_S32_F32_RR;
    break;
  case RVGPUISD::Tex1DS32FloatLevel:
    Opc = RVGPU::TEX_1D_S32_F32_LEVEL_RR;
    break;
  case RVGPUISD::Tex1DS32FloatGrad:
    Opc = RVGPU::TEX_1D_S32_F32_GRAD_RR;
    break;
  case RVGPUISD::Tex1DU32S32:
    Opc = RVGPU::TEX_1D_U32_S32_RR;
    break;
  case RVGPUISD::Tex1DU32Float:
    Opc = RVGPU::TEX_1D_U32_F32_RR;
    break;
  case RVGPUISD::Tex1DU32FloatLevel:
    Opc = RVGPU::TEX_1D_U32_F32_LEVEL_RR;
    break;
  case RVGPUISD::Tex1DU32FloatGrad:
    Opc = RVGPU::TEX_1D_U32_F32_GRAD_RR;
    break;
  case RVGPUISD::Tex1DArrayFloatS32:
    Opc = RVGPU::TEX_1D_ARRAY_F32_S32_RR;
    break;
  case RVGPUISD::Tex1DArrayFloatFloat:
    Opc = RVGPU::TEX_1D_ARRAY_F32_F32_RR;
    break;
  case RVGPUISD::Tex1DArrayFloatFloatLevel:
    Opc = RVGPU::TEX_1D_ARRAY_F32_F32_LEVEL_RR;
    break;
  case RVGPUISD::Tex1DArrayFloatFloatGrad:
    Opc = RVGPU::TEX_1D_ARRAY_F32_F32_GRAD_RR;
    break;
  case RVGPUISD::Tex1DArrayS32S32:
    Opc = RVGPU::TEX_1D_ARRAY_S32_S32_RR;
    break;
  case RVGPUISD::Tex1DArrayS32Float:
    Opc = RVGPU::TEX_1D_ARRAY_S32_F32_RR;
    break;
  case RVGPUISD::Tex1DArrayS32FloatLevel:
    Opc = RVGPU::TEX_1D_ARRAY_S32_F32_LEVEL_RR;
    break;
  case RVGPUISD::Tex1DArrayS32FloatGrad:
    Opc = RVGPU::TEX_1D_ARRAY_S32_F32_GRAD_RR;
    break;
  case RVGPUISD::Tex1DArrayU32S32:
    Opc = RVGPU::TEX_1D_ARRAY_U32_S32_RR;
    break;
  case RVGPUISD::Tex1DArrayU32Float:
    Opc = RVGPU::TEX_1D_ARRAY_U32_F32_RR;
    break;
  case RVGPUISD::Tex1DArrayU32FloatLevel:
    Opc = RVGPU::TEX_1D_ARRAY_U32_F32_LEVEL_RR;
    break;
  case RVGPUISD::Tex1DArrayU32FloatGrad:
    Opc = RVGPU::TEX_1D_ARRAY_U32_F32_GRAD_RR;
    break;
  case RVGPUISD::Tex2DFloatS32:
    Opc = RVGPU::TEX_2D_F32_S32_RR;
    break;
  case RVGPUISD::Tex2DFloatFloat:
    Opc = RVGPU::TEX_2D_F32_F32_RR;
    break;
  case RVGPUISD::Tex2DFloatFloatLevel:
    Opc = RVGPU::TEX_2D_F32_F32_LEVEL_RR;
    break;
  case RVGPUISD::Tex2DFloatFloatGrad:
    Opc = RVGPU::TEX_2D_F32_F32_GRAD_RR;
    break;
  case RVGPUISD::Tex2DS32S32:
    Opc = RVGPU::TEX_2D_S32_S32_RR;
    break;
  case RVGPUISD::Tex2DS32Float:
    Opc = RVGPU::TEX_2D_S32_F32_RR;
    break;
  case RVGPUISD::Tex2DS32FloatLevel:
    Opc = RVGPU::TEX_2D_S32_F32_LEVEL_RR;
    break;
  case RVGPUISD::Tex2DS32FloatGrad:
    Opc = RVGPU::TEX_2D_S32_F32_GRAD_RR;
    break;
  case RVGPUISD::Tex2DU32S32:
    Opc = RVGPU::TEX_2D_U32_S32_RR;
    break;
  case RVGPUISD::Tex2DU32Float:
    Opc = RVGPU::TEX_2D_U32_F32_RR;
    break;
  case RVGPUISD::Tex2DU32FloatLevel:
    Opc = RVGPU::TEX_2D_U32_F32_LEVEL_RR;
    break;
  case RVGPUISD::Tex2DU32FloatGrad:
    Opc = RVGPU::TEX_2D_U32_F32_GRAD_RR;
    break;
  case RVGPUISD::Tex2DArrayFloatS32:
    Opc = RVGPU::TEX_2D_ARRAY_F32_S32_RR;
    break;
  case RVGPUISD::Tex2DArrayFloatFloat:
    Opc = RVGPU::TEX_2D_ARRAY_F32_F32_RR;
    break;
  case RVGPUISD::Tex2DArrayFloatFloatLevel:
    Opc = RVGPU::TEX_2D_ARRAY_F32_F32_LEVEL_RR;
    break;
  case RVGPUISD::Tex2DArrayFloatFloatGrad:
    Opc = RVGPU::TEX_2D_ARRAY_F32_F32_GRAD_RR;
    break;
  case RVGPUISD::Tex2DArrayS32S32:
    Opc = RVGPU::TEX_2D_ARRAY_S32_S32_RR;
    break;
  case RVGPUISD::Tex2DArrayS32Float:
    Opc = RVGPU::TEX_2D_ARRAY_S32_F32_RR;
    break;
  case RVGPUISD::Tex2DArrayS32FloatLevel:
    Opc = RVGPU::TEX_2D_ARRAY_S32_F32_LEVEL_RR;
    break;
  case RVGPUISD::Tex2DArrayS32FloatGrad:
    Opc = RVGPU::TEX_2D_ARRAY_S32_F32_GRAD_RR;
    break;
  case RVGPUISD::Tex2DArrayU32S32:
    Opc = RVGPU::TEX_2D_ARRAY_U32_S32_RR;
    break;
  case RVGPUISD::Tex2DArrayU32Float:
    Opc = RVGPU::TEX_2D_ARRAY_U32_F32_RR;
    break;
  case RVGPUISD::Tex2DArrayU32FloatLevel:
    Opc = RVGPU::TEX_2D_ARRAY_U32_F32_LEVEL_RR;
    break;
  case RVGPUISD::Tex2DArrayU32FloatGrad:
    Opc = RVGPU::TEX_2D_ARRAY_U32_F32_GRAD_RR;
    break;
  case RVGPUISD::Tex3DFloatS32:
    Opc = RVGPU::TEX_3D_F32_S32_RR;
    break;
  case RVGPUISD::Tex3DFloatFloat:
    Opc = RVGPU::TEX_3D_F32_F32_RR;
    break;
  case RVGPUISD::Tex3DFloatFloatLevel:
    Opc = RVGPU::TEX_3D_F32_F32_LEVEL_RR;
    break;
  case RVGPUISD::Tex3DFloatFloatGrad:
    Opc = RVGPU::TEX_3D_F32_F32_GRAD_RR;
    break;
  case RVGPUISD::Tex3DS32S32:
    Opc = RVGPU::TEX_3D_S32_S32_RR;
    break;
  case RVGPUISD::Tex3DS32Float:
    Opc = RVGPU::TEX_3D_S32_F32_RR;
    break;
  case RVGPUISD::Tex3DS32FloatLevel:
    Opc = RVGPU::TEX_3D_S32_F32_LEVEL_RR;
    break;
  case RVGPUISD::Tex3DS32FloatGrad:
    Opc = RVGPU::TEX_3D_S32_F32_GRAD_RR;
    break;
  case RVGPUISD::Tex3DU32S32:
    Opc = RVGPU::TEX_3D_U32_S32_RR;
    break;
  case RVGPUISD::Tex3DU32Float:
    Opc = RVGPU::TEX_3D_U32_F32_RR;
    break;
  case RVGPUISD::Tex3DU32FloatLevel:
    Opc = RVGPU::TEX_3D_U32_F32_LEVEL_RR;
    break;
  case RVGPUISD::Tex3DU32FloatGrad:
    Opc = RVGPU::TEX_3D_U32_F32_GRAD_RR;
    break;
  case RVGPUISD::TexCubeFloatFloat:
    Opc = RVGPU::TEX_CUBE_F32_F32_RR;
    break;
  case RVGPUISD::TexCubeFloatFloatLevel:
    Opc = RVGPU::TEX_CUBE_F32_F32_LEVEL_RR;
    break;
  case RVGPUISD::TexCubeS32Float:
    Opc = RVGPU::TEX_CUBE_S32_F32_RR;
    break;
  case RVGPUISD::TexCubeS32FloatLevel:
    Opc = RVGPU::TEX_CUBE_S32_F32_LEVEL_RR;
    break;
  case RVGPUISD::TexCubeU32Float:
    Opc = RVGPU::TEX_CUBE_U32_F32_RR;
    break;
  case RVGPUISD::TexCubeU32FloatLevel:
    Opc = RVGPU::TEX_CUBE_U32_F32_LEVEL_RR;
    break;
  case RVGPUISD::TexCubeArrayFloatFloat:
    Opc = RVGPU::TEX_CUBE_ARRAY_F32_F32_RR;
    break;
  case RVGPUISD::TexCubeArrayFloatFloatLevel:
    Opc = RVGPU::TEX_CUBE_ARRAY_F32_F32_LEVEL_RR;
    break;
  case RVGPUISD::TexCubeArrayS32Float:
    Opc = RVGPU::TEX_CUBE_ARRAY_S32_F32_RR;
    break;
  case RVGPUISD::TexCubeArrayS32FloatLevel:
    Opc = RVGPU::TEX_CUBE_ARRAY_S32_F32_LEVEL_RR;
    break;
  case RVGPUISD::TexCubeArrayU32Float:
    Opc = RVGPU::TEX_CUBE_ARRAY_U32_F32_RR;
    break;
  case RVGPUISD::TexCubeArrayU32FloatLevel:
    Opc = RVGPU::TEX_CUBE_ARRAY_U32_F32_LEVEL_RR;
    break;
  case RVGPUISD::Tld4R2DFloatFloat:
    Opc = RVGPU::TLD4_R_2D_F32_F32_RR;
    break;
  case RVGPUISD::Tld4G2DFloatFloat:
    Opc = RVGPU::TLD4_G_2D_F32_F32_RR;
    break;
  case RVGPUISD::Tld4B2DFloatFloat:
    Opc = RVGPU::TLD4_B_2D_F32_F32_RR;
    break;
  case RVGPUISD::Tld4A2DFloatFloat:
    Opc = RVGPU::TLD4_A_2D_F32_F32_RR;
    break;
  case RVGPUISD::Tld4R2DS64Float:
    Opc = RVGPU::TLD4_R_2D_S32_F32_RR;
    break;
  case RVGPUISD::Tld4G2DS64Float:
    Opc = RVGPU::TLD4_G_2D_S32_F32_RR;
    break;
  case RVGPUISD::Tld4B2DS64Float:
    Opc = RVGPU::TLD4_B_2D_S32_F32_RR;
    break;
  case RVGPUISD::Tld4A2DS64Float:
    Opc = RVGPU::TLD4_A_2D_S32_F32_RR;
    break;
  case RVGPUISD::Tld4R2DU64Float:
    Opc = RVGPU::TLD4_R_2D_U32_F32_RR;
    break;
  case RVGPUISD::Tld4G2DU64Float:
    Opc = RVGPU::TLD4_G_2D_U32_F32_RR;
    break;
  case RVGPUISD::Tld4B2DU64Float:
    Opc = RVGPU::TLD4_B_2D_U32_F32_RR;
    break;
  case RVGPUISD::Tld4A2DU64Float:
    Opc = RVGPU::TLD4_A_2D_U32_F32_RR;
    break;
  case RVGPUISD::TexUnified1DFloatS32:
    Opc = RVGPU::TEX_UNIFIED_1D_F32_S32_R;
    break;
  case RVGPUISD::TexUnified1DFloatFloat:
    Opc = RVGPU::TEX_UNIFIED_1D_F32_F32_R;
    break;
  case RVGPUISD::TexUnified1DFloatFloatLevel:
    Opc = RVGPU::TEX_UNIFIED_1D_F32_F32_LEVEL_R;
    break;
  case RVGPUISD::TexUnified1DFloatFloatGrad:
    Opc = RVGPU::TEX_UNIFIED_1D_F32_F32_GRAD_R;
    break;
  case RVGPUISD::TexUnified1DS32S32:
    Opc = RVGPU::TEX_UNIFIED_1D_S32_S32_R;
    break;
  case RVGPUISD::TexUnified1DS32Float:
    Opc = RVGPU::TEX_UNIFIED_1D_S32_F32_R;
    break;
  case RVGPUISD::TexUnified1DS32FloatLevel:
    Opc = RVGPU::TEX_UNIFIED_1D_S32_F32_LEVEL_R;
    break;
  case RVGPUISD::TexUnified1DS32FloatGrad:
    Opc = RVGPU::TEX_UNIFIED_1D_S32_F32_GRAD_R;
    break;
  case RVGPUISD::TexUnified1DU32S32:
    Opc = RVGPU::TEX_UNIFIED_1D_U32_S32_R;
    break;
  case RVGPUISD::TexUnified1DU32Float:
    Opc = RVGPU::TEX_UNIFIED_1D_U32_F32_R;
    break;
  case RVGPUISD::TexUnified1DU32FloatLevel:
    Opc = RVGPU::TEX_UNIFIED_1D_U32_F32_LEVEL_R;
    break;
  case RVGPUISD::TexUnified1DU32FloatGrad:
    Opc = RVGPU::TEX_UNIFIED_1D_U32_F32_GRAD_R;
    break;
  case RVGPUISD::TexUnified1DArrayFloatS32:
    Opc = RVGPU::TEX_UNIFIED_1D_ARRAY_F32_S32_R;
    break;
  case RVGPUISD::TexUnified1DArrayFloatFloat:
    Opc = RVGPU::TEX_UNIFIED_1D_ARRAY_F32_F32_R;
    break;
  case RVGPUISD::TexUnified1DArrayFloatFloatLevel:
    Opc = RVGPU::TEX_UNIFIED_1D_ARRAY_F32_F32_LEVEL_R;
    break;
  case RVGPUISD::TexUnified1DArrayFloatFloatGrad:
    Opc = RVGPU::TEX_UNIFIED_1D_ARRAY_F32_F32_GRAD_R;
    break;
  case RVGPUISD::TexUnified1DArrayS32S32:
    Opc = RVGPU::TEX_UNIFIED_1D_ARRAY_S32_S32_R;
    break;
  case RVGPUISD::TexUnified1DArrayS32Float:
    Opc = RVGPU::TEX_UNIFIED_1D_ARRAY_S32_F32_R;
    break;
  case RVGPUISD::TexUnified1DArrayS32FloatLevel:
    Opc = RVGPU::TEX_UNIFIED_1D_ARRAY_S32_F32_LEVEL_R;
    break;
  case RVGPUISD::TexUnified1DArrayS32FloatGrad:
    Opc = RVGPU::TEX_UNIFIED_1D_ARRAY_S32_F32_GRAD_R;
    break;
  case RVGPUISD::TexUnified1DArrayU32S32:
    Opc = RVGPU::TEX_UNIFIED_1D_ARRAY_U32_S32_R;
    break;
  case RVGPUISD::TexUnified1DArrayU32Float:
    Opc = RVGPU::TEX_UNIFIED_1D_ARRAY_U32_F32_R;
    break;
  case RVGPUISD::TexUnified1DArrayU32FloatLevel:
    Opc = RVGPU::TEX_UNIFIED_1D_ARRAY_U32_F32_LEVEL_R;
    break;
  case RVGPUISD::TexUnified1DArrayU32FloatGrad:
    Opc = RVGPU::TEX_UNIFIED_1D_ARRAY_U32_F32_GRAD_R;
    break;
  case RVGPUISD::TexUnified2DFloatS32:
    Opc = RVGPU::TEX_UNIFIED_2D_F32_S32_R;
    break;
  case RVGPUISD::TexUnified2DFloatFloat:
    Opc = RVGPU::TEX_UNIFIED_2D_F32_F32_R;
    break;
  case RVGPUISD::TexUnified2DFloatFloatLevel:
    Opc = RVGPU::TEX_UNIFIED_2D_F32_F32_LEVEL_R;
    break;
  case RVGPUISD::TexUnified2DFloatFloatGrad:
    Opc = RVGPU::TEX_UNIFIED_2D_F32_F32_GRAD_R;
    break;
  case RVGPUISD::TexUnified2DS32S32:
    Opc = RVGPU::TEX_UNIFIED_2D_S32_S32_R;
    break;
  case RVGPUISD::TexUnified2DS32Float:
    Opc = RVGPU::TEX_UNIFIED_2D_S32_F32_R;
    break;
  case RVGPUISD::TexUnified2DS32FloatLevel:
    Opc = RVGPU::TEX_UNIFIED_2D_S32_F32_LEVEL_R;
    break;
  case RVGPUISD::TexUnified2DS32FloatGrad:
    Opc = RVGPU::TEX_UNIFIED_2D_S32_F32_GRAD_R;
    break;
  case RVGPUISD::TexUnified2DU32S32:
    Opc = RVGPU::TEX_UNIFIED_2D_U32_S32_R;
    break;
  case RVGPUISD::TexUnified2DU32Float:
    Opc = RVGPU::TEX_UNIFIED_2D_U32_F32_R;
    break;
  case RVGPUISD::TexUnified2DU32FloatLevel:
    Opc = RVGPU::TEX_UNIFIED_2D_U32_F32_LEVEL_R;
    break;
  case RVGPUISD::TexUnified2DU32FloatGrad:
    Opc = RVGPU::TEX_UNIFIED_2D_U32_F32_GRAD_R;
    break;
  case RVGPUISD::TexUnified2DArrayFloatS32:
    Opc = RVGPU::TEX_UNIFIED_2D_ARRAY_F32_S32_R;
    break;
  case RVGPUISD::TexUnified2DArrayFloatFloat:
    Opc = RVGPU::TEX_UNIFIED_2D_ARRAY_F32_F32_R;
    break;
  case RVGPUISD::TexUnified2DArrayFloatFloatLevel:
    Opc = RVGPU::TEX_UNIFIED_2D_ARRAY_F32_F32_LEVEL_R;
    break;
  case RVGPUISD::TexUnified2DArrayFloatFloatGrad:
    Opc = RVGPU::TEX_UNIFIED_2D_ARRAY_F32_F32_GRAD_R;
    break;
  case RVGPUISD::TexUnified2DArrayS32S32:
    Opc = RVGPU::TEX_UNIFIED_2D_ARRAY_S32_S32_R;
    break;
  case RVGPUISD::TexUnified2DArrayS32Float:
    Opc = RVGPU::TEX_UNIFIED_2D_ARRAY_S32_F32_R;
    break;
  case RVGPUISD::TexUnified2DArrayS32FloatLevel:
    Opc = RVGPU::TEX_UNIFIED_2D_ARRAY_S32_F32_LEVEL_R;
    break;
  case RVGPUISD::TexUnified2DArrayS32FloatGrad:
    Opc = RVGPU::TEX_UNIFIED_2D_ARRAY_S32_F32_GRAD_R;
    break;
  case RVGPUISD::TexUnified2DArrayU32S32:
    Opc = RVGPU::TEX_UNIFIED_2D_ARRAY_U32_S32_R;
    break;
  case RVGPUISD::TexUnified2DArrayU32Float:
    Opc = RVGPU::TEX_UNIFIED_2D_ARRAY_U32_F32_R;
    break;
  case RVGPUISD::TexUnified2DArrayU32FloatLevel:
    Opc = RVGPU::TEX_UNIFIED_2D_ARRAY_U32_F32_LEVEL_R;
    break;
  case RVGPUISD::TexUnified2DArrayU32FloatGrad:
    Opc = RVGPU::TEX_UNIFIED_2D_ARRAY_U32_F32_GRAD_R;
    break;
  case RVGPUISD::TexUnified3DFloatS32:
    Opc = RVGPU::TEX_UNIFIED_3D_F32_S32_R;
    break;
  case RVGPUISD::TexUnified3DFloatFloat:
    Opc = RVGPU::TEX_UNIFIED_3D_F32_F32_R;
    break;
  case RVGPUISD::TexUnified3DFloatFloatLevel:
    Opc = RVGPU::TEX_UNIFIED_3D_F32_F32_LEVEL_R;
    break;
  case RVGPUISD::TexUnified3DFloatFloatGrad:
    Opc = RVGPU::TEX_UNIFIED_3D_F32_F32_GRAD_R;
    break;
  case RVGPUISD::TexUnified3DS32S32:
    Opc = RVGPU::TEX_UNIFIED_3D_S32_S32_R;
    break;
  case RVGPUISD::TexUnified3DS32Float:
    Opc = RVGPU::TEX_UNIFIED_3D_S32_F32_R;
    break;
  case RVGPUISD::TexUnified3DS32FloatLevel:
    Opc = RVGPU::TEX_UNIFIED_3D_S32_F32_LEVEL_R;
    break;
  case RVGPUISD::TexUnified3DS32FloatGrad:
    Opc = RVGPU::TEX_UNIFIED_3D_S32_F32_GRAD_R;
    break;
  case RVGPUISD::TexUnified3DU32S32:
    Opc = RVGPU::TEX_UNIFIED_3D_U32_S32_R;
    break;
  case RVGPUISD::TexUnified3DU32Float:
    Opc = RVGPU::TEX_UNIFIED_3D_U32_F32_R;
    break;
  case RVGPUISD::TexUnified3DU32FloatLevel:
    Opc = RVGPU::TEX_UNIFIED_3D_U32_F32_LEVEL_R;
    break;
  case RVGPUISD::TexUnified3DU32FloatGrad:
    Opc = RVGPU::TEX_UNIFIED_3D_U32_F32_GRAD_R;
    break;
  case RVGPUISD::TexUnifiedCubeFloatFloat:
    Opc = RVGPU::TEX_UNIFIED_CUBE_F32_F32_R;
    break;
  case RVGPUISD::TexUnifiedCubeFloatFloatLevel:
    Opc = RVGPU::TEX_UNIFIED_CUBE_F32_F32_LEVEL_R;
    break;
  case RVGPUISD::TexUnifiedCubeS32Float:
    Opc = RVGPU::TEX_UNIFIED_CUBE_S32_F32_R;
    break;
  case RVGPUISD::TexUnifiedCubeS32FloatLevel:
    Opc = RVGPU::TEX_UNIFIED_CUBE_S32_F32_LEVEL_R;
    break;
  case RVGPUISD::TexUnifiedCubeU32Float:
    Opc = RVGPU::TEX_UNIFIED_CUBE_U32_F32_R;
    break;
  case RVGPUISD::TexUnifiedCubeU32FloatLevel:
    Opc = RVGPU::TEX_UNIFIED_CUBE_U32_F32_LEVEL_R;
    break;
  case RVGPUISD::TexUnifiedCubeArrayFloatFloat:
    Opc = RVGPU::TEX_UNIFIED_CUBE_ARRAY_F32_F32_R;
    break;
  case RVGPUISD::TexUnifiedCubeArrayFloatFloatLevel:
    Opc = RVGPU::TEX_UNIFIED_CUBE_ARRAY_F32_F32_LEVEL_R;
    break;
  case RVGPUISD::TexUnifiedCubeArrayS32Float:
    Opc = RVGPU::TEX_UNIFIED_CUBE_ARRAY_S32_F32_R;
    break;
  case RVGPUISD::TexUnifiedCubeArrayS32FloatLevel:
    Opc = RVGPU::TEX_UNIFIED_CUBE_ARRAY_S32_F32_LEVEL_R;
    break;
  case RVGPUISD::TexUnifiedCubeArrayU32Float:
    Opc = RVGPU::TEX_UNIFIED_CUBE_ARRAY_U32_F32_R;
    break;
  case RVGPUISD::TexUnifiedCubeArrayU32FloatLevel:
    Opc = RVGPU::TEX_UNIFIED_CUBE_ARRAY_U32_F32_LEVEL_R;
    break;
  case RVGPUISD::Tld4UnifiedR2DFloatFloat:
    Opc = RVGPU::TLD4_UNIFIED_R_2D_F32_F32_R;
    break;
  case RVGPUISD::Tld4UnifiedG2DFloatFloat:
    Opc = RVGPU::TLD4_UNIFIED_G_2D_F32_F32_R;
    break;
  case RVGPUISD::Tld4UnifiedB2DFloatFloat:
    Opc = RVGPU::TLD4_UNIFIED_B_2D_F32_F32_R;
    break;
  case RVGPUISD::Tld4UnifiedA2DFloatFloat:
    Opc = RVGPU::TLD4_UNIFIED_A_2D_F32_F32_R;
    break;
  case RVGPUISD::Tld4UnifiedR2DS64Float:
    Opc = RVGPU::TLD4_UNIFIED_R_2D_S32_F32_R;
    break;
  case RVGPUISD::Tld4UnifiedG2DS64Float:
    Opc = RVGPU::TLD4_UNIFIED_G_2D_S32_F32_R;
    break;
  case RVGPUISD::Tld4UnifiedB2DS64Float:
    Opc = RVGPU::TLD4_UNIFIED_B_2D_S32_F32_R;
    break;
  case RVGPUISD::Tld4UnifiedA2DS64Float:
    Opc = RVGPU::TLD4_UNIFIED_A_2D_S32_F32_R;
    break;
  case RVGPUISD::Tld4UnifiedR2DU64Float:
    Opc = RVGPU::TLD4_UNIFIED_R_2D_U32_F32_R;
    break;
  case RVGPUISD::Tld4UnifiedG2DU64Float:
    Opc = RVGPU::TLD4_UNIFIED_G_2D_U32_F32_R;
    break;
  case RVGPUISD::Tld4UnifiedB2DU64Float:
    Opc = RVGPU::TLD4_UNIFIED_B_2D_U32_F32_R;
    break;
  case RVGPUISD::Tld4UnifiedA2DU64Float:
    Opc = RVGPU::TLD4_UNIFIED_A_2D_U32_F32_R;
    break;
  }

  // Copy over operands
  SmallVector<SDValue, 8> Ops(drop_begin(N->ops()));
  Ops.push_back(N->getOperand(0)); // Move chain to the back.

  ReplaceNode(N, CurDAG->getMachineNode(Opc, SDLoc(N), N->getVTList(), Ops));
  return true;
}

bool RVGPUDAGToDAGISel::trySurfaceIntrinsic(SDNode *N) {
  unsigned Opc = 0;
  switch (N->getOpcode()) {
  default: return false;
  case RVGPUISD::Suld1DI8Clamp:
    Opc = RVGPU::SULD_1D_I8_CLAMP_R;
    break;
  case RVGPUISD::Suld1DI16Clamp:
    Opc = RVGPU::SULD_1D_I16_CLAMP_R;
    break;
  case RVGPUISD::Suld1DI32Clamp:
    Opc = RVGPU::SULD_1D_I32_CLAMP_R;
    break;
  case RVGPUISD::Suld1DI64Clamp:
    Opc = RVGPU::SULD_1D_I64_CLAMP_R;
    break;
  case RVGPUISD::Suld1DV2I8Clamp:
    Opc = RVGPU::SULD_1D_V2I8_CLAMP_R;
    break;
  case RVGPUISD::Suld1DV2I16Clamp:
    Opc = RVGPU::SULD_1D_V2I16_CLAMP_R;
    break;
  case RVGPUISD::Suld1DV2I32Clamp:
    Opc = RVGPU::SULD_1D_V2I32_CLAMP_R;
    break;
  case RVGPUISD::Suld1DV2I64Clamp:
    Opc = RVGPU::SULD_1D_V2I64_CLAMP_R;
    break;
  case RVGPUISD::Suld1DV4I8Clamp:
    Opc = RVGPU::SULD_1D_V4I8_CLAMP_R;
    break;
  case RVGPUISD::Suld1DV4I16Clamp:
    Opc = RVGPU::SULD_1D_V4I16_CLAMP_R;
    break;
  case RVGPUISD::Suld1DV4I32Clamp:
    Opc = RVGPU::SULD_1D_V4I32_CLAMP_R;
    break;
  case RVGPUISD::Suld1DArrayI8Clamp:
    Opc = RVGPU::SULD_1D_ARRAY_I8_CLAMP_R;
    break;
  case RVGPUISD::Suld1DArrayI16Clamp:
    Opc = RVGPU::SULD_1D_ARRAY_I16_CLAMP_R;
    break;
  case RVGPUISD::Suld1DArrayI32Clamp:
    Opc = RVGPU::SULD_1D_ARRAY_I32_CLAMP_R;
    break;
  case RVGPUISD::Suld1DArrayI64Clamp:
    Opc = RVGPU::SULD_1D_ARRAY_I64_CLAMP_R;
    break;
  case RVGPUISD::Suld1DArrayV2I8Clamp:
    Opc = RVGPU::SULD_1D_ARRAY_V2I8_CLAMP_R;
    break;
  case RVGPUISD::Suld1DArrayV2I16Clamp:
    Opc = RVGPU::SULD_1D_ARRAY_V2I16_CLAMP_R;
    break;
  case RVGPUISD::Suld1DArrayV2I32Clamp:
    Opc = RVGPU::SULD_1D_ARRAY_V2I32_CLAMP_R;
    break;
  case RVGPUISD::Suld1DArrayV2I64Clamp:
    Opc = RVGPU::SULD_1D_ARRAY_V2I64_CLAMP_R;
    break;
  case RVGPUISD::Suld1DArrayV4I8Clamp:
    Opc = RVGPU::SULD_1D_ARRAY_V4I8_CLAMP_R;
    break;
  case RVGPUISD::Suld1DArrayV4I16Clamp:
    Opc = RVGPU::SULD_1D_ARRAY_V4I16_CLAMP_R;
    break;
  case RVGPUISD::Suld1DArrayV4I32Clamp:
    Opc = RVGPU::SULD_1D_ARRAY_V4I32_CLAMP_R;
    break;
  case RVGPUISD::Suld2DI8Clamp:
    Opc = RVGPU::SULD_2D_I8_CLAMP_R;
    break;
  case RVGPUISD::Suld2DI16Clamp:
    Opc = RVGPU::SULD_2D_I16_CLAMP_R;
    break;
  case RVGPUISD::Suld2DI32Clamp:
    Opc = RVGPU::SULD_2D_I32_CLAMP_R;
    break;
  case RVGPUISD::Suld2DI64Clamp:
    Opc = RVGPU::SULD_2D_I64_CLAMP_R;
    break;
  case RVGPUISD::Suld2DV2I8Clamp:
    Opc = RVGPU::SULD_2D_V2I8_CLAMP_R;
    break;
  case RVGPUISD::Suld2DV2I16Clamp:
    Opc = RVGPU::SULD_2D_V2I16_CLAMP_R;
    break;
  case RVGPUISD::Suld2DV2I32Clamp:
    Opc = RVGPU::SULD_2D_V2I32_CLAMP_R;
    break;
  case RVGPUISD::Suld2DV2I64Clamp:
    Opc = RVGPU::SULD_2D_V2I64_CLAMP_R;
    break;
  case RVGPUISD::Suld2DV4I8Clamp:
    Opc = RVGPU::SULD_2D_V4I8_CLAMP_R;
    break;
  case RVGPUISD::Suld2DV4I16Clamp:
    Opc = RVGPU::SULD_2D_V4I16_CLAMP_R;
    break;
  case RVGPUISD::Suld2DV4I32Clamp:
    Opc = RVGPU::SULD_2D_V4I32_CLAMP_R;
    break;
  case RVGPUISD::Suld2DArrayI8Clamp:
    Opc = RVGPU::SULD_2D_ARRAY_I8_CLAMP_R;
    break;
  case RVGPUISD::Suld2DArrayI16Clamp:
    Opc = RVGPU::SULD_2D_ARRAY_I16_CLAMP_R;
    break;
  case RVGPUISD::Suld2DArrayI32Clamp:
    Opc = RVGPU::SULD_2D_ARRAY_I32_CLAMP_R;
    break;
  case RVGPUISD::Suld2DArrayI64Clamp:
    Opc = RVGPU::SULD_2D_ARRAY_I64_CLAMP_R;
    break;
  case RVGPUISD::Suld2DArrayV2I8Clamp:
    Opc = RVGPU::SULD_2D_ARRAY_V2I8_CLAMP_R;
    break;
  case RVGPUISD::Suld2DArrayV2I16Clamp:
    Opc = RVGPU::SULD_2D_ARRAY_V2I16_CLAMP_R;
    break;
  case RVGPUISD::Suld2DArrayV2I32Clamp:
    Opc = RVGPU::SULD_2D_ARRAY_V2I32_CLAMP_R;
    break;
  case RVGPUISD::Suld2DArrayV2I64Clamp:
    Opc = RVGPU::SULD_2D_ARRAY_V2I64_CLAMP_R;
    break;
  case RVGPUISD::Suld2DArrayV4I8Clamp:
    Opc = RVGPU::SULD_2D_ARRAY_V4I8_CLAMP_R;
    break;
  case RVGPUISD::Suld2DArrayV4I16Clamp:
    Opc = RVGPU::SULD_2D_ARRAY_V4I16_CLAMP_R;
    break;
  case RVGPUISD::Suld2DArrayV4I32Clamp:
    Opc = RVGPU::SULD_2D_ARRAY_V4I32_CLAMP_R;
    break;
  case RVGPUISD::Suld3DI8Clamp:
    Opc = RVGPU::SULD_3D_I8_CLAMP_R;
    break;
  case RVGPUISD::Suld3DI16Clamp:
    Opc = RVGPU::SULD_3D_I16_CLAMP_R;
    break;
  case RVGPUISD::Suld3DI32Clamp:
    Opc = RVGPU::SULD_3D_I32_CLAMP_R;
    break;
  case RVGPUISD::Suld3DI64Clamp:
    Opc = RVGPU::SULD_3D_I64_CLAMP_R;
    break;
  case RVGPUISD::Suld3DV2I8Clamp:
    Opc = RVGPU::SULD_3D_V2I8_CLAMP_R;
    break;
  case RVGPUISD::Suld3DV2I16Clamp:
    Opc = RVGPU::SULD_3D_V2I16_CLAMP_R;
    break;
  case RVGPUISD::Suld3DV2I32Clamp:
    Opc = RVGPU::SULD_3D_V2I32_CLAMP_R;
    break;
  case RVGPUISD::Suld3DV2I64Clamp:
    Opc = RVGPU::SULD_3D_V2I64_CLAMP_R;
    break;
  case RVGPUISD::Suld3DV4I8Clamp:
    Opc = RVGPU::SULD_3D_V4I8_CLAMP_R;
    break;
  case RVGPUISD::Suld3DV4I16Clamp:
    Opc = RVGPU::SULD_3D_V4I16_CLAMP_R;
    break;
  case RVGPUISD::Suld3DV4I32Clamp:
    Opc = RVGPU::SULD_3D_V4I32_CLAMP_R;
    break;
  case RVGPUISD::Suld1DI8Trap:
    Opc = RVGPU::SULD_1D_I8_TRAP_R;
    break;
  case RVGPUISD::Suld1DI16Trap:
    Opc = RVGPU::SULD_1D_I16_TRAP_R;
    break;
  case RVGPUISD::Suld1DI32Trap:
    Opc = RVGPU::SULD_1D_I32_TRAP_R;
    break;
  case RVGPUISD::Suld1DI64Trap:
    Opc = RVGPU::SULD_1D_I64_TRAP_R;
    break;
  case RVGPUISD::Suld1DV2I8Trap:
    Opc = RVGPU::SULD_1D_V2I8_TRAP_R;
    break;
  case RVGPUISD::Suld1DV2I16Trap:
    Opc = RVGPU::SULD_1D_V2I16_TRAP_R;
    break;
  case RVGPUISD::Suld1DV2I32Trap:
    Opc = RVGPU::SULD_1D_V2I32_TRAP_R;
    break;
  case RVGPUISD::Suld1DV2I64Trap:
    Opc = RVGPU::SULD_1D_V2I64_TRAP_R;
    break;
  case RVGPUISD::Suld1DV4I8Trap:
    Opc = RVGPU::SULD_1D_V4I8_TRAP_R;
    break;
  case RVGPUISD::Suld1DV4I16Trap:
    Opc = RVGPU::SULD_1D_V4I16_TRAP_R;
    break;
  case RVGPUISD::Suld1DV4I32Trap:
    Opc = RVGPU::SULD_1D_V4I32_TRAP_R;
    break;
  case RVGPUISD::Suld1DArrayI8Trap:
    Opc = RVGPU::SULD_1D_ARRAY_I8_TRAP_R;
    break;
  case RVGPUISD::Suld1DArrayI16Trap:
    Opc = RVGPU::SULD_1D_ARRAY_I16_TRAP_R;
    break;
  case RVGPUISD::Suld1DArrayI32Trap:
    Opc = RVGPU::SULD_1D_ARRAY_I32_TRAP_R;
    break;
  case RVGPUISD::Suld1DArrayI64Trap:
    Opc = RVGPU::SULD_1D_ARRAY_I64_TRAP_R;
    break;
  case RVGPUISD::Suld1DArrayV2I8Trap:
    Opc = RVGPU::SULD_1D_ARRAY_V2I8_TRAP_R;
    break;
  case RVGPUISD::Suld1DArrayV2I16Trap:
    Opc = RVGPU::SULD_1D_ARRAY_V2I16_TRAP_R;
    break;
  case RVGPUISD::Suld1DArrayV2I32Trap:
    Opc = RVGPU::SULD_1D_ARRAY_V2I32_TRAP_R;
    break;
  case RVGPUISD::Suld1DArrayV2I64Trap:
    Opc = RVGPU::SULD_1D_ARRAY_V2I64_TRAP_R;
    break;
  case RVGPUISD::Suld1DArrayV4I8Trap:
    Opc = RVGPU::SULD_1D_ARRAY_V4I8_TRAP_R;
    break;
  case RVGPUISD::Suld1DArrayV4I16Trap:
    Opc = RVGPU::SULD_1D_ARRAY_V4I16_TRAP_R;
    break;
  case RVGPUISD::Suld1DArrayV4I32Trap:
    Opc = RVGPU::SULD_1D_ARRAY_V4I32_TRAP_R;
    break;
  case RVGPUISD::Suld2DI8Trap:
    Opc = RVGPU::SULD_2D_I8_TRAP_R;
    break;
  case RVGPUISD::Suld2DI16Trap:
    Opc = RVGPU::SULD_2D_I16_TRAP_R;
    break;
  case RVGPUISD::Suld2DI32Trap:
    Opc = RVGPU::SULD_2D_I32_TRAP_R;
    break;
  case RVGPUISD::Suld2DI64Trap:
    Opc = RVGPU::SULD_2D_I64_TRAP_R;
    break;
  case RVGPUISD::Suld2DV2I8Trap:
    Opc = RVGPU::SULD_2D_V2I8_TRAP_R;
    break;
  case RVGPUISD::Suld2DV2I16Trap:
    Opc = RVGPU::SULD_2D_V2I16_TRAP_R;
    break;
  case RVGPUISD::Suld2DV2I32Trap:
    Opc = RVGPU::SULD_2D_V2I32_TRAP_R;
    break;
  case RVGPUISD::Suld2DV2I64Trap:
    Opc = RVGPU::SULD_2D_V2I64_TRAP_R;
    break;
  case RVGPUISD::Suld2DV4I8Trap:
    Opc = RVGPU::SULD_2D_V4I8_TRAP_R;
    break;
  case RVGPUISD::Suld2DV4I16Trap:
    Opc = RVGPU::SULD_2D_V4I16_TRAP_R;
    break;
  case RVGPUISD::Suld2DV4I32Trap:
    Opc = RVGPU::SULD_2D_V4I32_TRAP_R;
    break;
  case RVGPUISD::Suld2DArrayI8Trap:
    Opc = RVGPU::SULD_2D_ARRAY_I8_TRAP_R;
    break;
  case RVGPUISD::Suld2DArrayI16Trap:
    Opc = RVGPU::SULD_2D_ARRAY_I16_TRAP_R;
    break;
  case RVGPUISD::Suld2DArrayI32Trap:
    Opc = RVGPU::SULD_2D_ARRAY_I32_TRAP_R;
    break;
  case RVGPUISD::Suld2DArrayI64Trap:
    Opc = RVGPU::SULD_2D_ARRAY_I64_TRAP_R;
    break;
  case RVGPUISD::Suld2DArrayV2I8Trap:
    Opc = RVGPU::SULD_2D_ARRAY_V2I8_TRAP_R;
    break;
  case RVGPUISD::Suld2DArrayV2I16Trap:
    Opc = RVGPU::SULD_2D_ARRAY_V2I16_TRAP_R;
    break;
  case RVGPUISD::Suld2DArrayV2I32Trap:
    Opc = RVGPU::SULD_2D_ARRAY_V2I32_TRAP_R;
    break;
  case RVGPUISD::Suld2DArrayV2I64Trap:
    Opc = RVGPU::SULD_2D_ARRAY_V2I64_TRAP_R;
    break;
  case RVGPUISD::Suld2DArrayV4I8Trap:
    Opc = RVGPU::SULD_2D_ARRAY_V4I8_TRAP_R;
    break;
  case RVGPUISD::Suld2DArrayV4I16Trap:
    Opc = RVGPU::SULD_2D_ARRAY_V4I16_TRAP_R;
    break;
  case RVGPUISD::Suld2DArrayV4I32Trap:
    Opc = RVGPU::SULD_2D_ARRAY_V4I32_TRAP_R;
    break;
  case RVGPUISD::Suld3DI8Trap:
    Opc = RVGPU::SULD_3D_I8_TRAP_R;
    break;
  case RVGPUISD::Suld3DI16Trap:
    Opc = RVGPU::SULD_3D_I16_TRAP_R;
    break;
  case RVGPUISD::Suld3DI32Trap:
    Opc = RVGPU::SULD_3D_I32_TRAP_R;
    break;
  case RVGPUISD::Suld3DI64Trap:
    Opc = RVGPU::SULD_3D_I64_TRAP_R;
    break;
  case RVGPUISD::Suld3DV2I8Trap:
    Opc = RVGPU::SULD_3D_V2I8_TRAP_R;
    break;
  case RVGPUISD::Suld3DV2I16Trap:
    Opc = RVGPU::SULD_3D_V2I16_TRAP_R;
    break;
  case RVGPUISD::Suld3DV2I32Trap:
    Opc = RVGPU::SULD_3D_V2I32_TRAP_R;
    break;
  case RVGPUISD::Suld3DV2I64Trap:
    Opc = RVGPU::SULD_3D_V2I64_TRAP_R;
    break;
  case RVGPUISD::Suld3DV4I8Trap:
    Opc = RVGPU::SULD_3D_V4I8_TRAP_R;
    break;
  case RVGPUISD::Suld3DV4I16Trap:
    Opc = RVGPU::SULD_3D_V4I16_TRAP_R;
    break;
  case RVGPUISD::Suld3DV4I32Trap:
    Opc = RVGPU::SULD_3D_V4I32_TRAP_R;
    break;
  case RVGPUISD::Suld1DI8Zero:
    Opc = RVGPU::SULD_1D_I8_ZERO_R;
    break;
  case RVGPUISD::Suld1DI16Zero:
    Opc = RVGPU::SULD_1D_I16_ZERO_R;
    break;
  case RVGPUISD::Suld1DI32Zero:
    Opc = RVGPU::SULD_1D_I32_ZERO_R;
    break;
  case RVGPUISD::Suld1DI64Zero:
    Opc = RVGPU::SULD_1D_I64_ZERO_R;
    break;
  case RVGPUISD::Suld1DV2I8Zero:
    Opc = RVGPU::SULD_1D_V2I8_ZERO_R;
    break;
  case RVGPUISD::Suld1DV2I16Zero:
    Opc = RVGPU::SULD_1D_V2I16_ZERO_R;
    break;
  case RVGPUISD::Suld1DV2I32Zero:
    Opc = RVGPU::SULD_1D_V2I32_ZERO_R;
    break;
  case RVGPUISD::Suld1DV2I64Zero:
    Opc = RVGPU::SULD_1D_V2I64_ZERO_R;
    break;
  case RVGPUISD::Suld1DV4I8Zero:
    Opc = RVGPU::SULD_1D_V4I8_ZERO_R;
    break;
  case RVGPUISD::Suld1DV4I16Zero:
    Opc = RVGPU::SULD_1D_V4I16_ZERO_R;
    break;
  case RVGPUISD::Suld1DV4I32Zero:
    Opc = RVGPU::SULD_1D_V4I32_ZERO_R;
    break;
  case RVGPUISD::Suld1DArrayI8Zero:
    Opc = RVGPU::SULD_1D_ARRAY_I8_ZERO_R;
    break;
  case RVGPUISD::Suld1DArrayI16Zero:
    Opc = RVGPU::SULD_1D_ARRAY_I16_ZERO_R;
    break;
  case RVGPUISD::Suld1DArrayI32Zero:
    Opc = RVGPU::SULD_1D_ARRAY_I32_ZERO_R;
    break;
  case RVGPUISD::Suld1DArrayI64Zero:
    Opc = RVGPU::SULD_1D_ARRAY_I64_ZERO_R;
    break;
  case RVGPUISD::Suld1DArrayV2I8Zero:
    Opc = RVGPU::SULD_1D_ARRAY_V2I8_ZERO_R;
    break;
  case RVGPUISD::Suld1DArrayV2I16Zero:
    Opc = RVGPU::SULD_1D_ARRAY_V2I16_ZERO_R;
    break;
  case RVGPUISD::Suld1DArrayV2I32Zero:
    Opc = RVGPU::SULD_1D_ARRAY_V2I32_ZERO_R;
    break;
  case RVGPUISD::Suld1DArrayV2I64Zero:
    Opc = RVGPU::SULD_1D_ARRAY_V2I64_ZERO_R;
    break;
  case RVGPUISD::Suld1DArrayV4I8Zero:
    Opc = RVGPU::SULD_1D_ARRAY_V4I8_ZERO_R;
    break;
  case RVGPUISD::Suld1DArrayV4I16Zero:
    Opc = RVGPU::SULD_1D_ARRAY_V4I16_ZERO_R;
    break;
  case RVGPUISD::Suld1DArrayV4I32Zero:
    Opc = RVGPU::SULD_1D_ARRAY_V4I32_ZERO_R;
    break;
  case RVGPUISD::Suld2DI8Zero:
    Opc = RVGPU::SULD_2D_I8_ZERO_R;
    break;
  case RVGPUISD::Suld2DI16Zero:
    Opc = RVGPU::SULD_2D_I16_ZERO_R;
    break;
  case RVGPUISD::Suld2DI32Zero:
    Opc = RVGPU::SULD_2D_I32_ZERO_R;
    break;
  case RVGPUISD::Suld2DI64Zero:
    Opc = RVGPU::SULD_2D_I64_ZERO_R;
    break;
  case RVGPUISD::Suld2DV2I8Zero:
    Opc = RVGPU::SULD_2D_V2I8_ZERO_R;
    break;
  case RVGPUISD::Suld2DV2I16Zero:
    Opc = RVGPU::SULD_2D_V2I16_ZERO_R;
    break;
  case RVGPUISD::Suld2DV2I32Zero:
    Opc = RVGPU::SULD_2D_V2I32_ZERO_R;
    break;
  case RVGPUISD::Suld2DV2I64Zero:
    Opc = RVGPU::SULD_2D_V2I64_ZERO_R;
    break;
  case RVGPUISD::Suld2DV4I8Zero:
    Opc = RVGPU::SULD_2D_V4I8_ZERO_R;
    break;
  case RVGPUISD::Suld2DV4I16Zero:
    Opc = RVGPU::SULD_2D_V4I16_ZERO_R;
    break;
  case RVGPUISD::Suld2DV4I32Zero:
    Opc = RVGPU::SULD_2D_V4I32_ZERO_R;
    break;
  case RVGPUISD::Suld2DArrayI8Zero:
    Opc = RVGPU::SULD_2D_ARRAY_I8_ZERO_R;
    break;
  case RVGPUISD::Suld2DArrayI16Zero:
    Opc = RVGPU::SULD_2D_ARRAY_I16_ZERO_R;
    break;
  case RVGPUISD::Suld2DArrayI32Zero:
    Opc = RVGPU::SULD_2D_ARRAY_I32_ZERO_R;
    break;
  case RVGPUISD::Suld2DArrayI64Zero:
    Opc = RVGPU::SULD_2D_ARRAY_I64_ZERO_R;
    break;
  case RVGPUISD::Suld2DArrayV2I8Zero:
    Opc = RVGPU::SULD_2D_ARRAY_V2I8_ZERO_R;
    break;
  case RVGPUISD::Suld2DArrayV2I16Zero:
    Opc = RVGPU::SULD_2D_ARRAY_V2I16_ZERO_R;
    break;
  case RVGPUISD::Suld2DArrayV2I32Zero:
    Opc = RVGPU::SULD_2D_ARRAY_V2I32_ZERO_R;
    break;
  case RVGPUISD::Suld2DArrayV2I64Zero:
    Opc = RVGPU::SULD_2D_ARRAY_V2I64_ZERO_R;
    break;
  case RVGPUISD::Suld2DArrayV4I8Zero:
    Opc = RVGPU::SULD_2D_ARRAY_V4I8_ZERO_R;
    break;
  case RVGPUISD::Suld2DArrayV4I16Zero:
    Opc = RVGPU::SULD_2D_ARRAY_V4I16_ZERO_R;
    break;
  case RVGPUISD::Suld2DArrayV4I32Zero:
    Opc = RVGPU::SULD_2D_ARRAY_V4I32_ZERO_R;
    break;
  case RVGPUISD::Suld3DI8Zero:
    Opc = RVGPU::SULD_3D_I8_ZERO_R;
    break;
  case RVGPUISD::Suld3DI16Zero:
    Opc = RVGPU::SULD_3D_I16_ZERO_R;
    break;
  case RVGPUISD::Suld3DI32Zero:
    Opc = RVGPU::SULD_3D_I32_ZERO_R;
    break;
  case RVGPUISD::Suld3DI64Zero:
    Opc = RVGPU::SULD_3D_I64_ZERO_R;
    break;
  case RVGPUISD::Suld3DV2I8Zero:
    Opc = RVGPU::SULD_3D_V2I8_ZERO_R;
    break;
  case RVGPUISD::Suld3DV2I16Zero:
    Opc = RVGPU::SULD_3D_V2I16_ZERO_R;
    break;
  case RVGPUISD::Suld3DV2I32Zero:
    Opc = RVGPU::SULD_3D_V2I32_ZERO_R;
    break;
  case RVGPUISD::Suld3DV2I64Zero:
    Opc = RVGPU::SULD_3D_V2I64_ZERO_R;
    break;
  case RVGPUISD::Suld3DV4I8Zero:
    Opc = RVGPU::SULD_3D_V4I8_ZERO_R;
    break;
  case RVGPUISD::Suld3DV4I16Zero:
    Opc = RVGPU::SULD_3D_V4I16_ZERO_R;
    break;
  case RVGPUISD::Suld3DV4I32Zero:
    Opc = RVGPU::SULD_3D_V4I32_ZERO_R;
    break;
  }

  // Copy over operands
  SmallVector<SDValue, 8> Ops(drop_begin(N->ops()));
  Ops.push_back(N->getOperand(0)); // Move chain to the back.

  ReplaceNode(N, CurDAG->getMachineNode(Opc, SDLoc(N), N->getVTList(), Ops));
  return true;
}


/// SelectBFE - Look for instruction sequences that can be made more efficient
/// by using the 'bfe' (bit-field extract) PTX instruction
bool RVGPUDAGToDAGISel::tryBFE(SDNode *N) {
  SDLoc DL(N);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  SDValue Len;
  SDValue Start;
  SDValue Val;
  bool IsSigned = false;

  if (N->getOpcode() == ISD::AND) {
    // Canonicalize the operands
    // We want 'and %val, %mask'
    if (isa<ConstantSDNode>(LHS) && !isa<ConstantSDNode>(RHS)) {
      std::swap(LHS, RHS);
    }

    ConstantSDNode *Mask = dyn_cast<ConstantSDNode>(RHS);
    if (!Mask) {
      // We need a constant mask on the RHS of the AND
      return false;
    }

    // Extract the mask bits
    uint64_t MaskVal = Mask->getZExtValue();
    if (!isMask_64(MaskVal)) {
      // We *could* handle shifted masks here, but doing so would require an
      // 'and' operation to fix up the low-order bits so we would trade
      // shr+and for bfe+and, which has the same throughput
      return false;
    }

    // How many bits are in our mask?
    int64_t NumBits = countr_one(MaskVal);
    Len = CurDAG->getTargetConstant(NumBits, DL, MVT::i32);

    if (LHS.getOpcode() == ISD::SRL || LHS.getOpcode() == ISD::SRA) {
      // We have a 'srl/and' pair, extract the effective start bit and length
      Val = LHS.getNode()->getOperand(0);
      Start = LHS.getNode()->getOperand(1);
      ConstantSDNode *StartConst = dyn_cast<ConstantSDNode>(Start);
      if (StartConst) {
        uint64_t StartVal = StartConst->getZExtValue();
        // How many "good" bits do we have left?  "good" is defined here as bits
        // that exist in the original value, not shifted in.
        int64_t GoodBits = Start.getValueSizeInBits() - StartVal;
        if (NumBits > GoodBits) {
          // Do not handle the case where bits have been shifted in. In theory
          // we could handle this, but the cost is likely higher than just
          // emitting the srl/and pair.
          return false;
        }
        Start = CurDAG->getTargetConstant(StartVal, DL, MVT::i32);
      } else {
        // Do not handle the case where the shift amount (can be zero if no srl
        // was found) is not constant. We could handle this case, but it would
        // require run-time logic that would be more expensive than just
        // emitting the srl/and pair.
        return false;
      }
    } else {
      // Do not handle the case where the LHS of the and is not a shift. While
      // it would be trivial to handle this case, it would just transform
      // 'and' -> 'bfe', but 'and' has higher-throughput.
      return false;
    }
  } else if (N->getOpcode() == ISD::SRL || N->getOpcode() == ISD::SRA) {
    if (LHS->getOpcode() == ISD::AND) {
      ConstantSDNode *ShiftCnst = dyn_cast<ConstantSDNode>(RHS);
      if (!ShiftCnst) {
        // Shift amount must be constant
        return false;
      }

      uint64_t ShiftAmt = ShiftCnst->getZExtValue();

      SDValue AndLHS = LHS->getOperand(0);
      SDValue AndRHS = LHS->getOperand(1);

      // Canonicalize the AND to have the mask on the RHS
      if (isa<ConstantSDNode>(AndLHS)) {
        std::swap(AndLHS, AndRHS);
      }

      ConstantSDNode *MaskCnst = dyn_cast<ConstantSDNode>(AndRHS);
      if (!MaskCnst) {
        // Mask must be constant
        return false;
      }

      uint64_t MaskVal = MaskCnst->getZExtValue();
      uint64_t NumZeros;
      uint64_t NumBits;
      if (isMask_64(MaskVal)) {
        NumZeros = 0;
        // The number of bits in the result bitfield will be the number of
        // trailing ones (the AND) minus the number of bits we shift off
        NumBits = llvm::countr_one(MaskVal) - ShiftAmt;
      } else if (isShiftedMask_64(MaskVal)) {
        NumZeros = llvm::countr_zero(MaskVal);
        unsigned NumOnes = llvm::countr_one(MaskVal >> NumZeros);
        // The number of bits in the result bitfield will be the number of
        // trailing zeros plus the number of set bits in the mask minus the
        // number of bits we shift off
        NumBits = NumZeros + NumOnes - ShiftAmt;
      } else {
        // This is not a mask we can handle
        return false;
      }

      if (ShiftAmt < NumZeros) {
        // Handling this case would require extra logic that would make this
        // transformation non-profitable
        return false;
      }

      Val = AndLHS;
      Start = CurDAG->getTargetConstant(ShiftAmt, DL, MVT::i32);
      Len = CurDAG->getTargetConstant(NumBits, DL, MVT::i32);
    } else if (LHS->getOpcode() == ISD::SHL) {
      // Here, we have a pattern like:
      //
      // (sra (shl val, NN), MM)
      // or
      // (srl (shl val, NN), MM)
      //
      // If MM >= NN, we can efficiently optimize this with bfe
      Val = LHS->getOperand(0);

      SDValue ShlRHS = LHS->getOperand(1);
      ConstantSDNode *ShlCnst = dyn_cast<ConstantSDNode>(ShlRHS);
      if (!ShlCnst) {
        // Shift amount must be constant
        return false;
      }
      uint64_t InnerShiftAmt = ShlCnst->getZExtValue();

      SDValue ShrRHS = RHS;
      ConstantSDNode *ShrCnst = dyn_cast<ConstantSDNode>(ShrRHS);
      if (!ShrCnst) {
        // Shift amount must be constant
        return false;
      }
      uint64_t OuterShiftAmt = ShrCnst->getZExtValue();

      // To avoid extra codegen and be profitable, we need Outer >= Inner
      if (OuterShiftAmt < InnerShiftAmt) {
        return false;
      }

      // If the outer shift is more than the type size, we have no bitfield to
      // extract (since we also check that the inner shift is <= the outer shift
      // then this also implies that the inner shift is < the type size)
      if (OuterShiftAmt >= Val.getValueSizeInBits()) {
        return false;
      }

      Start = CurDAG->getTargetConstant(OuterShiftAmt - InnerShiftAmt, DL,
                                        MVT::i32);
      Len = CurDAG->getTargetConstant(Val.getValueSizeInBits() - OuterShiftAmt,
                                      DL, MVT::i32);

      if (N->getOpcode() == ISD::SRA) {
        // If we have a arithmetic right shift, we need to use the signed bfe
        // variant
        IsSigned = true;
      }
    } else {
      // No can do...
      return false;
    }
  } else {
    // No can do...
    return false;
  }


  unsigned Opc;
  // For the BFE operations we form here from "and" and "srl", always use the
  // unsigned variants.
  if (Val.getValueType() == MVT::i32) {
    if (IsSigned) {
      Opc = RVGPU::BFE_S32rii;
    } else {
      Opc = RVGPU::BFE_U32rii;
    }
  } else if (Val.getValueType() == MVT::i64) {
    if (IsSigned) {
      Opc = RVGPU::BFE_S64rii;
    } else {
      Opc = RVGPU::BFE_U64rii;
    }
  } else {
    // We cannot handle this type
    return false;
  }

  SDValue Ops[] = {
    Val, Start, Len
  };

  ReplaceNode(N, CurDAG->getMachineNode(Opc, DL, N->getVTList(), Ops));
  return true;
}

// SelectDirectAddr - Match a direct address for DAG.
// A direct address could be a globaladdress or externalsymbol.
bool RVGPUDAGToDAGISel::SelectDirectAddr(SDValue N, SDValue &Address) {
  // Return true if TGA or ES.
  if (N.getOpcode() == ISD::TargetGlobalAddress ||
      N.getOpcode() == ISD::TargetExternalSymbol) {
    Address = N;
    return true;
  }
  if (N.getOpcode() == RVGPUISD::Wrapper) {
    Address = N.getOperand(0);
    return true;
  }
  // addrspacecast(MoveParam(arg_symbol) to addrspace(PARAM)) -> arg_symbol
  if (AddrSpaceCastSDNode *CastN = dyn_cast<AddrSpaceCastSDNode>(N)) {
    if (CastN->getSrcAddressSpace() == ADDRESS_SPACE_GENERIC &&
        CastN->getDestAddressSpace() == ADDRESS_SPACE_PARAM &&
        CastN->getOperand(0).getOpcode() == RVGPUISD::MoveParam)
      return SelectDirectAddr(CastN->getOperand(0).getOperand(0), Address);
  }
  return false;
}

// symbol+offset
bool RVGPUDAGToDAGISel::SelectADDRsi_imp(
    SDNode *OpNode, SDValue Addr, SDValue &Base, SDValue &Offset, MVT mvt) {
  if (Addr.getOpcode() == ISD::ADD) {
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Addr.getOperand(1))) {
      SDValue base = Addr.getOperand(0);
      if (SelectDirectAddr(base, Base)) {
        Offset = CurDAG->getTargetConstant(CN->getZExtValue(), SDLoc(OpNode),
                                           mvt);
        return true;
      }
    }
  }
  return false;
}

// symbol+offset
bool RVGPUDAGToDAGISel::SelectADDRsi(SDNode *OpNode, SDValue Addr,
                                     SDValue &Base, SDValue &Offset) {
  return SelectADDRsi_imp(OpNode, Addr, Base, Offset, MVT::i32);
}

// symbol+offset
bool RVGPUDAGToDAGISel::SelectADDRsi64(SDNode *OpNode, SDValue Addr,
                                       SDValue &Base, SDValue &Offset) {
  return SelectADDRsi_imp(OpNode, Addr, Base, Offset, MVT::i64);
}

// register+offset
bool RVGPUDAGToDAGISel::SelectADDRri_imp(
    SDNode *OpNode, SDValue Addr, SDValue &Base, SDValue &Offset, MVT mvt) {
  if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), mvt);
    Offset = CurDAG->getTargetConstant(0, SDLoc(OpNode), mvt);
    return true;
  }
  if (Addr.getOpcode() == ISD::TargetExternalSymbol ||
      Addr.getOpcode() == ISD::TargetGlobalAddress)
    return false; // direct calls.

  if (Addr.getOpcode() == ISD::ADD) {
    if (SelectDirectAddr(Addr.getOperand(0), Addr)) {
      return false;
    }
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Addr.getOperand(1))) {
      if (FrameIndexSDNode *FIN =
              dyn_cast<FrameIndexSDNode>(Addr.getOperand(0)))
        // Constant offset from frame ref.
        Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), mvt);
      else
        Base = Addr.getOperand(0);
      Offset = CurDAG->getTargetConstant(CN->getZExtValue(), SDLoc(OpNode),
                                         mvt);
      return true;
    }
  }
  return false;
}

// register+offset
bool RVGPUDAGToDAGISel::SelectADDRri(SDNode *OpNode, SDValue Addr,
                                     SDValue &Base, SDValue &Offset) {
  return SelectADDRri_imp(OpNode, Addr, Base, Offset, MVT::i32);
}

// register+offset
bool RVGPUDAGToDAGISel::SelectADDRri64(SDNode *OpNode, SDValue Addr,
                                       SDValue &Base, SDValue &Offset) {
  return SelectADDRri_imp(OpNode, Addr, Base, Offset, MVT::i64);
}

bool RVGPUDAGToDAGISel::ChkMemSDNodeAddressSpace(SDNode *N,
                                                 unsigned int spN) const {
  const Value *Src = nullptr;
  if (MemSDNode *mN = dyn_cast<MemSDNode>(N)) {
    if (spN == 0 && mN->getMemOperand()->getPseudoValue())
      return true;
    Src = mN->getMemOperand()->getValue();
  }
  if (!Src)
    return false;
  if (auto *PT = dyn_cast<PointerType>(Src->getType()))
    return (PT->getAddressSpace() == spN);
  return false;
}

/// SelectInlineAsmMemoryOperand - Implement addressing mode selection for
/// inline asm expressions.
bool RVGPUDAGToDAGISel::SelectInlineAsmMemoryOperand(
    const SDValue &Op, InlineAsm::ConstraintCode ConstraintID,
    std::vector<SDValue> &OutOps) {
  SDValue Op0, Op1;
  switch (ConstraintID) {
  default:
    return true;
  case InlineAsm::ConstraintCode::m: // memory
    if (SelectDirectAddr(Op, Op0)) {
      OutOps.push_back(Op0);
      OutOps.push_back(CurDAG->getTargetConstant(0, SDLoc(Op), MVT::i32));
      return false;
    }
    if (SelectADDRri(Op.getNode(), Op, Op0, Op1)) {
      OutOps.push_back(Op0);
      OutOps.push_back(Op1);
      return false;
    }
    break;
  }
  return true;
}

/// GetConvertOpcode - Returns the CVT_ instruction opcode that implements a
/// conversion from \p SrcTy to \p DestTy.
unsigned RVGPUDAGToDAGISel::GetConvertOpcode(MVT DestTy, MVT SrcTy,
                                             LoadSDNode *LdNode) {
  bool IsSigned = LdNode && LdNode->getExtensionType() == ISD::SEXTLOAD;
  switch (SrcTy.SimpleTy) {
  default:
    llvm_unreachable("Unhandled source type");
  case MVT::i8:
    switch (DestTy.SimpleTy) {
    default:
      llvm_unreachable("Unhandled dest type");
    case MVT::i16:
      return IsSigned ? RVGPU::CVT_s16_s8 : RVGPU::CVT_u16_u8;
    case MVT::i32:
      return IsSigned ? RVGPU::CVT_s32_s8 : RVGPU::CVT_u32_u8;
    case MVT::i64:
      return IsSigned ? RVGPU::CVT_s64_s8 : RVGPU::CVT_u64_u8;
    }
  case MVT::i16:
    switch (DestTy.SimpleTy) {
    default:
      llvm_unreachable("Unhandled dest type");
    case MVT::i8:
      return IsSigned ? RVGPU::CVT_s8_s16 : RVGPU::CVT_u8_u16;
    case MVT::i32:
      return IsSigned ? RVGPU::CVT_s32_s16 : RVGPU::CVT_u32_u16;
    case MVT::i64:
      return IsSigned ? RVGPU::CVT_s64_s16 : RVGPU::CVT_u64_u16;
    }
  case MVT::i32:
    switch (DestTy.SimpleTy) {
    default:
      llvm_unreachable("Unhandled dest type");
    case MVT::i8:
      return IsSigned ? RVGPU::CVT_s8_s32 : RVGPU::CVT_u8_u32;
    case MVT::i16:
      return IsSigned ? RVGPU::CVT_s16_s32 : RVGPU::CVT_u16_u32;
    case MVT::i64:
      return IsSigned ? RVGPU::CVT_s64_s32 : RVGPU::CVT_u64_u32;
    }
  case MVT::i64:
    switch (DestTy.SimpleTy) {
    default:
      llvm_unreachable("Unhandled dest type");
    case MVT::i8:
      return IsSigned ? RVGPU::CVT_s8_s64 : RVGPU::CVT_u8_u64;
    case MVT::i16:
      return IsSigned ? RVGPU::CVT_s16_s64 : RVGPU::CVT_u16_u64;
    case MVT::i32:
      return IsSigned ? RVGPU::CVT_s32_s64 : RVGPU::CVT_u32_u64;
    }
  case MVT::f16:
    switch (DestTy.SimpleTy) {
    default:
      llvm_unreachable("Unhandled dest type");
    case MVT::f32:
      return RVGPU::CVT_f32_f16;
    case MVT::f64:
      return RVGPU::CVT_f64_f16;
    }
  }
}
