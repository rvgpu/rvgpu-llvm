//===-- RVGPUMCExpr.cpp - RVGPU specific MC expression classes ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RVGPUMCExpr.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

#define DEBUG_TYPE "rvgpu-mcexpr"

const RVGPUFloatMCExpr *
RVGPUFloatMCExpr::create(VariantKind Kind, const APFloat &Flt, MCContext &Ctx) {
  return new (Ctx) RVGPUFloatMCExpr(Kind, Flt);
}

void RVGPUFloatMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  bool Ignored;
  unsigned NumHex;
  APFloat APF = getAPFloat();

  switch (Kind) {
  default: llvm_unreachable("Invalid kind!");
  case VK_RVGPU_HALF_PREC_FLOAT:
    // ptxas does not have a way to specify half-precision floats.
    // Instead we have to print and load fp16 constants as .b16
    OS << "0x";
    NumHex = 4;
    APF.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &Ignored);
    break;
  case VK_RVGPU_BFLOAT_PREC_FLOAT:
    OS << "0x";
    NumHex = 4;
    APF.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven, &Ignored);
    break;
  case VK_RVGPU_SINGLE_PREC_FLOAT:
    OS << "0f";
    NumHex = 8;
    APF.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &Ignored);
    break;
  case VK_RVGPU_DOUBLE_PREC_FLOAT:
    OS << "0d";
    NumHex = 16;
    APF.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &Ignored);
    break;
  }

  APInt API = APF.bitcastToAPInt();
  OS << format_hex_no_prefix(API.getZExtValue(), NumHex, /*Upper=*/true);
}

const RVGPUGenericMCSymbolRefExpr*
RVGPUGenericMCSymbolRefExpr::create(const MCSymbolRefExpr *SymExpr,
                                    MCContext &Ctx) {
  return new (Ctx) RVGPUGenericMCSymbolRefExpr(SymExpr);
}

void RVGPUGenericMCSymbolRefExpr::printImpl(raw_ostream &OS,
                                            const MCAsmInfo *MAI) const {
  OS << "generic(";
  SymExpr->print(OS, MAI);
  OS << ")";
}

const RVGPUMCExpr *RVGPUMCExpr::create(const MCExpr *Expr, VariantKind Kind,
                                       MCContext &Ctx) {
  return new (Ctx) RVGPUMCExpr(Expr, Kind);
}

void RVGPUMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  VariantKind Kind = getKind();
  bool HasVariant = ((Kind != VK_RVGPU_None) && (Kind != VK_RVGPU_CALL) &&
                     (Kind != VK_RVGPU_CALL_PLT));

  if (HasVariant)
    OS << '%' << getVariantKindName(getKind()) << '(';
  Expr->print(OS, MAI);
  if (Kind == VK_RVGPU_CALL_PLT)
    OS << "@plt";
  if (HasVariant)
    OS << ')';
}

const MCFixup *RVGPUMCExpr::getPCRelHiFixup(const MCFragment **DFOut) const {
  MCValue AUIPCLoc;
  if (!getSubExpr()->evaluateAsRelocatable(AUIPCLoc, nullptr, nullptr))
    return nullptr;

  const MCSymbolRefExpr *AUIPCSRE = AUIPCLoc.getSymA();
  if (!AUIPCSRE)
    return nullptr;

  const MCSymbol *AUIPCSymbol = &AUIPCSRE->getSymbol();
  const auto *DF = dyn_cast_or_null<MCDataFragment>(AUIPCSymbol->getFragment());

  if (!DF)
    return nullptr;

  uint64_t Offset = AUIPCSymbol->getOffset();
  if (DF->getContents().size() == Offset) {
    DF = dyn_cast_or_null<MCDataFragment>(DF->getNextNode());
    if (!DF)
      return nullptr;
    Offset = 0;
  }

  for (const MCFixup &F : DF->getFixups()) {
    if (F.getOffset() != Offset)
      continue;

    switch ((unsigned)F.getKind()) {
    default:
      continue;
/*
    case RVGPU::fixup_riscv_got_hi20:
    case RVGPU::fixup_riscv_tls_got_hi20:
    case RVGPU::fixup_riscv_tls_gd_hi20:
    case RVGPU::fixup_riscv_pcrel_hi20:
    */
      if (DFOut)
        *DFOut = DF;
      return &F;
    }
  }

  return nullptr;
}

bool RVGPUMCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                            const MCAsmLayout *Layout,
                                            const MCFixup *Fixup) const {
  // Explicitly drop the layout and assembler to prevent any symbolic folding in
  // the expression handling.  This is required to preserve symbolic difference
  // expressions to emit the paired relocations.
  if (!getSubExpr()->evaluateAsRelocatable(Res, nullptr, nullptr))
    return false;

  Res =
      MCValue::get(Res.getSymA(), Res.getSymB(), Res.getConstant(), getKind());
  // Custom fixup types are not valid with symbol difference expressions.
  return Res.getSymB() ? getKind() == VK_RVGPU_None : true;
}

void RVGPUMCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}

RVGPUMCExpr::VariantKind RVGPUMCExpr::getVariantKindForName(StringRef name) {
  return StringSwitch<RVGPUMCExpr::VariantKind>(name)
      .Case("lo", VK_RVGPU_LO)
      .Case("hi", VK_RVGPU_HI)
      .Case("pcrel_lo", VK_RVGPU_PCREL_LO)
      .Case("pcrel_hi", VK_RVGPU_PCREL_HI)
      .Case("got_pcrel_hi", VK_RVGPU_GOT_HI)
      .Case("tprel_lo", VK_RVGPU_TPREL_LO)
      .Case("tprel_hi", VK_RVGPU_TPREL_HI)
      .Case("tprel_add", VK_RVGPU_TPREL_ADD)
      .Case("tls_ie_pcrel_hi", VK_RVGPU_TLS_GOT_HI)
      .Case("tls_gd_pcrel_hi", VK_RVGPU_TLS_GD_HI)
      .Default(VK_RVGPU_Invalid);
}

StringRef RVGPUMCExpr::getVariantKindName(VariantKind Kind) {
  switch (Kind) {
  case VK_RVGPU_Invalid:
  case VK_RVGPU_None:
    llvm_unreachable("Invalid ELF symbol kind");
  case VK_RVGPU_LO:
    return "lo";
  case VK_RVGPU_HI:
    return "hi";
  case VK_RVGPU_PCREL_LO:
    return "pcrel_lo";
  case VK_RVGPU_PCREL_HI:
    return "pcrel_hi";
  case VK_RVGPU_GOT_HI:
    return "got_pcrel_hi";
  case VK_RVGPU_TPREL_LO:
    return "tprel_lo";
  case VK_RVGPU_TPREL_HI:
    return "tprel_hi";
  case VK_RVGPU_TPREL_ADD:
    return "tprel_add";
  case VK_RVGPU_TLS_GOT_HI:
    return "tls_ie_pcrel_hi";
  case VK_RVGPU_TLS_GD_HI:
    return "tls_gd_pcrel_hi";
  case VK_RVGPU_CALL:
    return "call";
  case VK_RVGPU_CALL_PLT:
    return "call_plt";
  case VK_RVGPU_32_PCREL:
    return "32_pcrel";
  }
  llvm_unreachable("Invalid ELF symbol kind");
}

static void fixELFSymbolsInTLSFixupsImpl(const MCExpr *Expr, MCAssembler &Asm) {
  switch (Expr->getKind()) {
  case MCExpr::Target:
    llvm_unreachable("Can't handle nested target expression");
    break;
  case MCExpr::Constant:
    break;

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(Expr);
    fixELFSymbolsInTLSFixupsImpl(BE->getLHS(), Asm);
    fixELFSymbolsInTLSFixupsImpl(BE->getRHS(), Asm);
    break;
  }

  case MCExpr::SymbolRef: {
    // We're known to be under a TLS fixup, so any symbol should be
    // modified. There should be only one.
    const MCSymbolRefExpr &SymRef = *cast<MCSymbolRefExpr>(Expr);
    cast<MCSymbolELF>(SymRef.getSymbol()).setType(ELF::STT_TLS);
    break;
  }

  case MCExpr::Unary:
    fixELFSymbolsInTLSFixupsImpl(cast<MCUnaryExpr>(Expr)->getSubExpr(), Asm);
    break;
  }
}

void RVGPUMCExpr::fixELFSymbolsInTLSFixups(MCAssembler &Asm) const {
  switch (getKind()) {
  default:
    return;
  case VK_RVGPU_TPREL_HI:
  case VK_RVGPU_TLS_GOT_HI:
  case VK_RVGPU_TLS_GD_HI:
    break;
  }

  fixELFSymbolsInTLSFixupsImpl(getSubExpr(), Asm);
}

bool RVGPUMCExpr::evaluateAsConstant(int64_t &Res) const {
  MCValue Value;

  if (Kind == VK_RVGPU_PCREL_HI || Kind == VK_RVGPU_PCREL_LO ||
      Kind == VK_RVGPU_GOT_HI || Kind == VK_RVGPU_TPREL_HI ||
      Kind == VK_RVGPU_TPREL_LO || Kind == VK_RVGPU_TPREL_ADD ||
      Kind == VK_RVGPU_TLS_GOT_HI || Kind == VK_RVGPU_TLS_GD_HI ||
      Kind == VK_RVGPU_CALL || Kind == VK_RVGPU_CALL_PLT)
    return false;

  if (!getSubExpr()->evaluateAsRelocatable(Value, nullptr, nullptr))
    return false;

  if (!Value.isAbsolute())
    return false;

  Res = evaluateAsInt64(Value.getConstant());
  return true;
}

int64_t RVGPUMCExpr::evaluateAsInt64(int64_t Value) const {
  switch (Kind) {
  default:
    llvm_unreachable("Invalid kind");
  case VK_RVGPU_LO:
    return SignExtend64<12>(Value);
  case VK_RVGPU_HI:
    // Add 1 if bit 11 is 1, to compensate for low 12 bits being negative.
    return ((Value + 0x800) >> 12) & 0xfffff;
  }
}
