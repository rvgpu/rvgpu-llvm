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
#include "RVGPUELFStreamer.h"
#include "RVGPUInstPrinter.h"
#include "RVGPUMCAsmInfo.h"
#include "RVGPUTargetStreamer.h"
#include "TargetInfo/RVGPUTargetInfo.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
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
    return new RVGPUInstPrinter(MAI, MII, MRI);
}
#if 0
static MCTargetStreamer *createTargetAsmStreamer(MCStreamer &S,
                                                 formatted_raw_ostream &,
                                                 MCInstPrinter *, bool) {
  return new RVGPUAsmTargetStreamer(S);
}

static MCTargetStreamer *createNullTargetStreamer(MCStreamer &S) {
  return new RVGPUTargetStreamer(S);
}
static MCTargetStreamer * createRVGPUObjectTargetStreamer(
                                                   MCStreamer &S,
                                                   const MCSubtargetInfo &STI) {
  return new RVGPUTargetELFStreamer(S, STI);
}
#endif 
static MCTargetStreamer *createRVGPUAsmTargetStreamer(MCStreamer &S,
                                                      formatted_raw_ostream &OS,
                                                      MCInstPrinter *InstPrint,
                                                      bool isVerboseAsm) {
  return new RVGPUTargetAsmStreamer(S, OS);
}

static MCTargetStreamer * createRVGPUObjectTargetStreamer(
                                                   MCStreamer &S,
                                                   const MCSubtargetInfo &STI) {
  return new RVGPUTargetELFStreamer(S, STI);
}

static MCTargetStreamer *createRVGPUNullTargetStreamer(MCStreamer &S) {
  return new RVGPUTargetStreamer(S);
}

static MCStreamer *createMCStreamer(const Triple &T, MCContext &Context,
                                    std::unique_ptr<MCAsmBackend> &&MAB,
                                    std::unique_ptr<MCObjectWriter> &&OW,
                                    std::unique_ptr<MCCodeEmitter> &&Emitter,
                                    bool RelaxAll) {
  return createRVGPUELFStreamer(T, Context, std::move(MAB), std::move(OW),
                                 std::move(Emitter), RelaxAll);
}
// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeRVGPUTargetMC() {
  //for (Target *T : {&getTheRVGPUTarget64()}) {
  Target *T = &getTheRVGPUTarget64();
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
    
    TargetRegistry::RegisterMCAsmBackend(*T, createRVGPUAsmBackend);
    TargetRegistry::RegisterELFStreamer(*T, createMCStreamer);
  
    TargetRegistry::RegisterMCCodeEmitter(*T,
                                          createRVGPUMCCodeEmitter);

    TargetRegistry::RegisterAsmTargetStreamer(*T,
                                              createRVGPUAsmTargetStreamer);
    TargetRegistry::RegisterObjectTargetStreamer(*T, 
                                                 createRVGPUObjectTargetStreamer);
    TargetRegistry::RegisterNullTargetStreamer(*T,
                                               createRVGPUNullTargetStreamer);
#if 0
    // Register the MCTargetStreamer.
    TargetRegistry::RegisterAsmTargetStreamer(*T, createTargetAsmStreamer);

    // Register the MCTargetStreamer.
    TargetRegistry::RegisterNullTargetStreamer(*T, createNullTargetStreamer);
#endif     
}
