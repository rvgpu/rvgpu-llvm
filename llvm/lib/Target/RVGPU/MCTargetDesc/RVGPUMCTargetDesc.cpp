//===-- RVGPUMCTargetDesc.cpp - RVGPU Target Descriptions ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file provides RVGPU specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "RVGPUMCTargetDesc.h"
#include "RVGPUELFStreamer.h"
#include "RVGPUInstPrinter.h"
#include "RVGPUMCAsmInfo.h"
#include "RVGPUTargetStreamer.h"
#include "TargetInfo/RVGPUTargetInfo.h"
#include "llvm/MC/LaneBitmask.h"
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
  InitRVGPUMCRegisterInfo(X, RVGPU::PC_REG);
  return X;
}

MCRegisterInfo *llvm::createGCNMCRegisterInfo(RVGPUDwarfFlavour DwarfFlavour) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitRVGPUMCRegisterInfo(X, RVGPU::PC_REG, DwarfFlavour);
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

static MCStreamer *createRVGPUMCStreamer(const Triple &T, MCContext &Context,
                                    std::unique_ptr<MCAsmBackend> &&MAB,
                                    std::unique_ptr<MCObjectWriter> &&OW,
                                    std::unique_ptr<MCCodeEmitter> &&Emitter,
                                    bool RelaxAll) {
  return createRVGPUELFStreamer(T, Context, std::move(MAB), std::move(OW),
                                 std::move(Emitter), RelaxAll);
}

namespace {

class RVGPUMCInstrAnalysis : public MCInstrAnalysis {
public:
  explicit RVGPUMCInstrAnalysis(const MCInstrInfo *Info)
      : MCInstrAnalysis(Info) {}

  bool evaluateBranch(const MCInst &Inst, uint64_t Addr, uint64_t Size,
                      uint64_t &Target) const override {
    if (Inst.getNumOperands() == 0 || !Inst.getOperand(0).isImm() ||
        Info->get(Inst.getOpcode()).operands()[0].OperandType !=
            MCOI::OPERAND_PCREL)
      return false;

    int64_t Imm = Inst.getOperand(0).getImm();
    // Our branches take a simm16, but we need two extra bits to account for
    // the factor of 4.
    APInt SignedOffset(18, Imm * 4, true);
    Target = (SignedOffset.sext(64) + Addr + Size).getZExtValue();
    return true;
  }
};

} // end anonymous namespace

static MCInstrAnalysis *createRVGPUMCInstrAnalysis(const MCInstrInfo *Info) {
  return new RVGPUMCInstrAnalysis(Info);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeRVGPUTargetMC() {

  TargetRegistry::RegisterMCInstrInfo(getTheRVGPUTarget(), createRVGPUMCInstrInfo);
  for (Target *T : {&getTheRVGPUTarget()}) {
    RegisterMCAsmInfo<RVGPUMCAsmInfo> X(*T);

    TargetRegistry::RegisterMCRegInfo(*T, createRVGPUMCRegisterInfo);
    TargetRegistry::RegisterMCSubtargetInfo(*T, createRVGPUMCSubtargetInfo);
    TargetRegistry::RegisterMCInstPrinter(*T, createRVGPUMCInstPrinter);
    TargetRegistry::RegisterMCInstrAnalysis(*T, createRVGPUMCInstrAnalysis);
    TargetRegistry::RegisterMCAsmBackend(*T, createRVGPUAsmBackend);
    TargetRegistry::RegisterELFStreamer(*T, createRVGPUMCStreamer);
  }

  // GCN specific registration
  TargetRegistry::RegisterMCCodeEmitter(getTheRVGPUTarget(),
                                        createRVGPUMCCodeEmitter);

  TargetRegistry::RegisterAsmTargetStreamer(getTheRVGPUTarget(),
                                            createRVGPUAsmTargetStreamer);
  TargetRegistry::RegisterObjectTargetStreamer(
      getTheRVGPUTarget(), createRVGPUObjectTargetStreamer);
  TargetRegistry::RegisterNullTargetStreamer(getTheRVGPUTarget(),
                                             createRVGPUNullTargetStreamer);
}
