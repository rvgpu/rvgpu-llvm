//===-- RVGPUMCTargetDesc.h - RVGPU Target Descriptions ---------*- C++ -*-===//
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

#ifndef LLVM_LIB_TARGET_RVGPU_MCTARGETDESC_RVGPUMCTARGETDESC_H
#define LLVM_LIB_TARGET_RVGPU_MCTARGETDESC_RVGPUMCTARGETDESC_H

#include <stdint.h>
#include <memory>

namespace llvm {
class Target;
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectTargetWriter;
class MCRegisterInfo;
class MCSubtargetInfo;
class MCTargetOptions;

enum RVGPUDwarfFlavour : unsigned { Wave64 = 0, Wave32 = 1 };

MCCodeEmitter *createRVGPUMCCodeEmitter(const MCInstrInfo &MCII,
                                         MCContext &Ctx);

MCAsmBackend *createRVGPUAsmBackend(const Target &T,
                                     const MCSubtargetInfo &STI,
                                     const MCRegisterInfo &MRI,
                                     const MCTargetOptions &Options);

std::unique_ptr<MCObjectTargetWriter>
createRVGPUELFObjectWriter(bool Is64Bit, uint8_t OSABI,
                            bool HasRelocationAddend, uint8_t ABIVersion);
} // End llvm namespace

// Defines symbolic names for PTX registers.
#define GET_REGINFO_ENUM
#include "RVGPUGenRegisterInfo.inc"

// Defines symbolic names for the PTX instructions.
#define GET_INSTRINFO_ENUM
#define GET_INSTRINFO_OPERAND_ENUM
#define GET_INSTRINFO_MC_HELPER_DECLS
#include "RVGPUGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "RVGPUGenSubtargetInfo.inc"

#endif
