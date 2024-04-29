//===-- RVGPUAsmBackend.cpp - RVGPU Assembler Backend -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
/// \file
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RVGPUFixupKinds.h"
#include "MCTargetDesc/RVGPUMCTargetDesc.h"
//#include "Utils/RVGPUBaseInfo.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/TargetParser/TargetParser.h"

using namespace llvm;
using namespace llvm::RVGPU;

namespace {

class RVGPUAsmBackend : public MCAsmBackend {
public:
  RVGPUAsmBackend(const Target &T) : MCAsmBackend(llvm::endianness::little) {}

  unsigned getNumFixupKinds() const override { return RVGPU::NumTargetFixupKinds; };

  void applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                  const MCValue &Target, MutableArrayRef<char> Data,
                  uint64_t Value, bool IsResolved,
                  const MCSubtargetInfo *STI) const override;
  bool fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                            const MCRelaxableFragment *DF,
                            const MCAsmLayout &Layout) const override;

//  void relaxInstruction(MCInst &Inst,
 //                       const MCSubtargetInfo &STI) const override;

  bool mayNeedRelaxation(const MCInst &Inst,
                         const MCSubtargetInfo &STI) const override;

  unsigned getMinimumNopSize() const override;
  bool writeNopData(raw_ostream &OS, uint64_t Count,
                    const MCSubtargetInfo *STI) const override;

  std::optional<MCFixupKind> getFixupKind(StringRef Name) const override;
  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const override;
  bool shouldForceRelocation(const MCAssembler &Asm, const MCFixup &Fixup,
                             const MCValue &Target,
                             const MCSubtargetInfo *STI) override;
};

} //End anonymous namespace

#if 0                                          
void RVGPUAsmBackend::relaxInstruction(MCInst &Inst,
                                        const MCSubtargetInfo &STI) const {
  MCInst Res;
  unsigned RelaxedOpcode = RVGPU::getSOPPWithRelaxation(Inst.getOpcode());
  Res.setOpcode(RelaxedOpcode);
  Res.addOperand(Inst.getOperand(0));
  Inst = std::move(Res);
}
#endif                                           

bool RVGPUAsmBackend::fixupNeedsRelaxation(const MCFixup &Fixup,
                                            uint64_t Value,
                                            const MCRelaxableFragment *DF,
                                            const MCAsmLayout &Layout) const {
  // if the branch target has an offset of x3f this needs to be relaxed to
  // add a s_nop 0 immediately after branch to effectively increment offset
  // for hardware workaround in gfx1010
  return (((int64_t(Value)/4)-1) == 0x3f);
}

bool RVGPUAsmBackend::mayNeedRelaxation(const MCInst &Inst,
                       const MCSubtargetInfo &STI) const {
  return false;
}

static unsigned getFixupKindNumBytes(unsigned Kind) {
  switch (Kind) {
  case RVGPU::fixup_si_sopp_br:
    return 2;
  case FK_SecRel_1:
  case FK_Data_1:
    return 1;
  case FK_SecRel_2:
  case FK_Data_2:
    return 2;
  case FK_SecRel_4:
  case FK_Data_4:
  case FK_PCRel_4:
    return 4;
  case FK_SecRel_8:
  case FK_Data_8:
    return 8;
  default:
    llvm_unreachable("Unknown fixup kind!");
  }
}

static uint64_t adjustFixupValue(const MCFixup &Fixup, uint64_t Value,
                                 MCContext *Ctx) {
  int64_t SignedValue = static_cast<int64_t>(Value);

  switch (Fixup.getTargetKind()) {
  case RVGPU::fixup_si_sopp_br: {
    int64_t BrImm = (SignedValue - 4) / 4;

    if (Ctx && !isInt<16>(BrImm))
      Ctx->reportError(Fixup.getLoc(), "branch size exceeds simm16");

    return BrImm;
  }
  case FK_Data_1:
  case FK_Data_2:
  case FK_Data_4:
  case FK_Data_8:
  case FK_PCRel_4:
  case FK_SecRel_4:
    return Value;
  default:
    llvm_unreachable("unhandled fixup kind");
  }
}

void RVGPUAsmBackend::applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                                  const MCValue &Target,
                                  MutableArrayRef<char> Data, uint64_t Value,
                                  bool IsResolved,
                                  const MCSubtargetInfo *STI) const {
  if (Fixup.getKind() >= FirstLiteralRelocationKind)
    return;

  Value = adjustFixupValue(Fixup, Value, &Asm.getContext());
  if (!Value)
    return; // Doesn't change encoding.

  MCFixupKindInfo Info = getFixupKindInfo(Fixup.getKind());

  // Shift the value into position.
  Value <<= Info.TargetOffset;

  unsigned NumBytes = getFixupKindNumBytes(Fixup.getKind());
  uint32_t Offset = Fixup.getOffset();
  assert(Offset + NumBytes <= Data.size() && "Invalid fixup offset!");

  // For each byte of the fragment that the fixup touches, mask in the bits from
  // the fixup value.
  for (unsigned i = 0; i != NumBytes; ++i)
    Data[Offset + i] |= static_cast<uint8_t>((Value >> (i * 8)) & 0xff);
}

std::optional<MCFixupKind>
RVGPUAsmBackend::getFixupKind(StringRef Name) const {
  return StringSwitch<std::optional<MCFixupKind>>(Name)
#define ELF_RELOC(Name, Value)                                                 \
  .Case(#Name, MCFixupKind(FirstLiteralRelocationKind + Value))
#include "llvm/BinaryFormat/ELFRelocs/RVGPU.def"
#undef ELF_RELOC
      .Default(std::nullopt);
}

const MCFixupKindInfo &RVGPUAsmBackend::getFixupKindInfo(
                                                       MCFixupKind Kind) const {
  const static MCFixupKindInfo Infos[RVGPU::NumTargetFixupKinds] = {
    // name                   offset bits  flags
    { "fixup_si_sopp_br",     0,     16,   MCFixupKindInfo::FKF_IsPCRel },
  };

  if (Kind >= FirstLiteralRelocationKind)
    return MCAsmBackend::getFixupKindInfo(FK_NONE);

  if (Kind < FirstTargetFixupKind)
    return MCAsmBackend::getFixupKindInfo(Kind);

  assert(unsigned(Kind - FirstTargetFixupKind) < getNumFixupKinds() &&
         "Invalid kind!");
  return Infos[Kind - FirstTargetFixupKind];
}

bool RVGPUAsmBackend::shouldForceRelocation(const MCAssembler &,
                                             const MCFixup &Fixup,
                                             const MCValue &,
                                             const MCSubtargetInfo *STI) {
  return Fixup.getKind() >= FirstLiteralRelocationKind;
}

unsigned RVGPUAsmBackend::getMinimumNopSize() const {
  return 4;
}

bool RVGPUAsmBackend::writeNopData(raw_ostream &OS, uint64_t Count,
                                    const MCSubtargetInfo *STI) const {
  // If the count is not 4-byte aligned, we must be writing data into the text
  // section (otherwise we have unaligned instructions, and thus have far
  // bigger problems), so just write zeros instead.
  OS.write_zeros(Count % 4);

  // We are properly aligned, so write NOPs as requested.
  Count /= 4;

  const uint32_t Encoded_NOP_0 = 0x0;

  for (uint64_t I = 0; I != Count; ++I)
    support::endian::write<uint32_t>(OS, Encoded_NOP_0, Endian);

  return true;
}

//===----------------------------------------------------------------------===//
// ELFRVGPUAsmBackend class
//===----------------------------------------------------------------------===//

namespace {

class ELFRVGPUAsmBackend : public RVGPUAsmBackend {
  bool Is64Bit;
  bool HasRelocationAddend;
  uint8_t OSABI = ELF::ELFOSABI_NONE;
  uint8_t ABIVersion = 0;

public:
  ELFRVGPUAsmBackend(const Target &T, const Triple &TT, uint8_t ABIVersion) :
      RVGPUAsmBackend(T), Is64Bit(true),
      HasRelocationAddend(true),
      ABIVersion(ABIVersion) {
    OSABI = ELF::ELFOSABI_RVGPU;
  }

  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override {
    return createRVGPUELFObjectWriter(Is64Bit, OSABI, HasRelocationAddend,
                                       ABIVersion);
  }
};

} // end anonymous namespace

MCAsmBackend *llvm::createRVGPUAsmBackend(const Target &T,
                                           const MCSubtargetInfo &STI,
                                           const MCRegisterInfo &MRI,
                                           const MCTargetOptions &Options) {
  return new ELFRVGPUAsmBackend(T, STI.getTargetTriple(),
                                0);
                                 //getHsaAbiVersion(&STI).value_or(0));
}
