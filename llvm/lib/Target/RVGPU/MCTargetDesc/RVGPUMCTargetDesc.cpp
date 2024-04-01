//===-- RVGPUMCTargetDesc.cpp - RVGPU Target Descriptions -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides RVGPU specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "RVGPUMCTargetDesc.h"
#include "RVGPUInstPrinter.h"
#include "RVGPUMCAsmInfo.h"
#include "RVGPUTargetStreamer.h"
#include "TargetInfo/RVGPUTargetInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#define ENABLE_INSTR_PREDICATE_VERIFIER
#include "RVGPUGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "RVGPUGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "RVGPUGenRegisterInfo.inc"

static MCInstrInfo *createRVGPUMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitRVGPUMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createRVGPUMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  // PTX does not have a return address register.
  InitRVGPUMCRegisterInfo(X, 0);
  return X;
}

static MCSubtargetInfo *
createRVGPUMCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS) {
  return createRVGPUMCSubtargetInfoImpl(TT, CPU, /*TuneCPU*/ CPU, FS);
}

static MCInstPrinter *createRVGPUMCInstPrinter(const Triple &T,
                                               unsigned SyntaxVariant,
                                               const MCAsmInfo &MAI,
                                               const MCInstrInfo &MII,
                                               const MCRegisterInfo &MRI) {
  if (SyntaxVariant == 0)
    return new RVGPUInstPrinter(MAI, MII, MRI);
  return nullptr;
}

static MCTargetStreamer *createTargetAsmStreamer(MCStreamer &S,
                                                 formatted_raw_ostream &,
                                                 MCInstPrinter *, bool) {
  return new RVGPUAsmTargetStreamer(S);
}

static MCTargetStreamer *createNullTargetStreamer(MCStreamer &S) {
  return new RVGPUTargetStreamer(S);
}

// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeRVGPUTargetMC() {
  for (Target *T : {&getTheRVGPUTarget32(), &getTheRVGPUTarget64()}) {
    // Register the MC asm info.
    RegisterMCAsmInfo<RVGPUMCAsmInfo> X(*T);

    // Register the MC instruction info.
    TargetRegistry::RegisterMCInstrInfo(*T, createRVGPUMCInstrInfo);

    // Register the MC register info.
    TargetRegistry::RegisterMCRegInfo(*T, createRVGPUMCRegisterInfo);

    // Register the MC subtarget info.
    TargetRegistry::RegisterMCSubtargetInfo(*T, createRVGPUMCSubtargetInfo);

    // Register the MCInstPrinter.
    TargetRegistry::RegisterMCInstPrinter(*T, createRVGPUMCInstPrinter);

    // Register the MCTargetStreamer.
    TargetRegistry::RegisterAsmTargetStreamer(*T, createTargetAsmStreamer);

    // Register the MCTargetStreamer.
    TargetRegistry::RegisterNullTargetStreamer(*T, createNullTargetStreamer);
  }
}
