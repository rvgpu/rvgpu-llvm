//===-- RVGPUMCCodeEmitter.cpp - RVGPU Code Emitter ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// The RVGPU code emitter produces machine code that can be executed
/// directly on the GPU device.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RVGPUFixupKinds.h"
#include "MCTargetDesc/RVGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "Utils/RVGPUBaseInfo.h"
#include "llvm/ADT/APInt.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include <optional>

using namespace llvm;

namespace {

class RVGPUMCCodeEmitter : public MCCodeEmitter {
  const MCRegisterInfo &MRI;
  const MCInstrInfo &MCII;

public:
  RVGPUMCCodeEmitter(const MCInstrInfo &MCII, const MCRegisterInfo &MRI)
      : MRI(MRI), MCII(MCII) {}

  /// Encode the instruction and write it to the OS.
  void encodeInstruction(const MCInst &MI, SmallVectorImpl<char> &CB,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;

  void getMachineOpValue(const MCInst &MI, const MCOperand &MO, APInt &Op,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const;

  void getMachineOpValueT16(const MCInst &MI, unsigned OpNo, APInt &Op,
                            SmallVectorImpl<MCFixup> &Fixups,
                            const MCSubtargetInfo &STI) const;

  void getMachineOpValueT16Lo128(const MCInst &MI, unsigned OpNo, APInt &Op,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const;

  /// Use a fixup to encode the simm16 field for SOPP branch
  ///        instructions.
  void getSOPPBrEncoding(const MCInst &MI, unsigned OpNo, APInt &Op,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const;

  void getSMEMOffsetEncoding(const MCInst &MI, unsigned OpNo, APInt &Op,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;

  void getSDWASrcEncoding(const MCInst &MI, unsigned OpNo, APInt &Op,
                          SmallVectorImpl<MCFixup> &Fixups,
                          const MCSubtargetInfo &STI) const;

  void getSDWAVopcDstEncoding(const MCInst &MI, unsigned OpNo, APInt &Op,
                              SmallVectorImpl<MCFixup> &Fixups,
                              const MCSubtargetInfo &STI) const;

  void getAVOperandEncoding(const MCInst &MI, unsigned OpNo, APInt &Op,
                            SmallVectorImpl<MCFixup> &Fixups,
                            const MCSubtargetInfo &STI) const;

private:
  uint64_t getImplicitOpSelHiEncoding(int Opcode) const;
  void getMachineOpValueCommon(const MCInst &MI, const MCOperand &MO,
                               unsigned OpNo, APInt &Op,
                               SmallVectorImpl<MCFixup> &Fixups,
                               const MCSubtargetInfo &STI) const;

  /// Encode an fp or int literal.
  std::optional<uint32_t> getLitEncoding(const MCOperand &MO,
                                         const MCOperandInfo &OpInfo,
                                         const MCSubtargetInfo &STI) const;

  void getBinaryCodeForInstr(const MCInst &MI, SmallVectorImpl<MCFixup> &Fixups,
                             APInt &Inst, APInt &Scratch,
                             const MCSubtargetInfo &STI) const;
};

} // end anonymous namespace

MCCodeEmitter *llvm::createRVGPUMCCodeEmitter(const MCInstrInfo &MCII,
                                               MCContext &Ctx) {
  return new RVGPUMCCodeEmitter(MCII, *Ctx.getRegisterInfo());
}

// Returns the encoding value to use if the given integer is an integer inline
// immediate value, or 0 if it is not.
template <typename IntTy>
static uint32_t getIntInlineImmEncoding(IntTy Imm) {
  if (Imm >= 0 && Imm <= 64)
    return 128 + Imm;

  if (Imm >= -16 && Imm <= -1)
    return 192 + std::abs(Imm);

  return 0;
}

static uint32_t getLit16IntEncoding(uint16_t Val, const MCSubtargetInfo &STI) {
  uint16_t IntImm = getIntInlineImmEncoding(static_cast<int16_t>(Val));
  return IntImm == 0 ? 255 : IntImm;
}

static uint32_t getLit16Encoding(uint16_t Val, const MCSubtargetInfo &STI) {
  uint16_t IntImm = getIntInlineImmEncoding(static_cast<int16_t>(Val));
  if (IntImm != 0)
    return IntImm;

  if (Val == 0x3800) // 0.5
    return 240;

  if (Val == 0xB800) // -0.5
    return 241;

  if (Val == 0x3C00) // 1.0
    return 242;

  if (Val == 0xBC00) // -1.0
    return 243;

  if (Val == 0x4000) // 2.0
    return 244;

  if (Val == 0xC000) // -2.0
    return 245;

  if (Val == 0x4400) // 4.0
    return 246;

  if (Val == 0xC400) // -4.0
    return 247;

  if (Val == 0x3118 && // 1.0 / (2.0 * pi)
      STI.hasFeature(RVGPU::FeatureInv2PiInlineImm))
    return 248;

  return 255;
}

static uint32_t getLit32Encoding(uint32_t Val, const MCSubtargetInfo &STI) {
  uint32_t IntImm = getIntInlineImmEncoding(static_cast<int32_t>(Val));
  if (IntImm != 0)
    return IntImm;

  if (Val == llvm::bit_cast<uint32_t>(0.5f))
    return 240;

  if (Val == llvm::bit_cast<uint32_t>(-0.5f))
    return 241;

  if (Val == llvm::bit_cast<uint32_t>(1.0f))
    return 242;

  if (Val == llvm::bit_cast<uint32_t>(-1.0f))
    return 243;

  if (Val == llvm::bit_cast<uint32_t>(2.0f))
    return 244;

  if (Val == llvm::bit_cast<uint32_t>(-2.0f))
    return 245;

  if (Val == llvm::bit_cast<uint32_t>(4.0f))
    return 246;

  if (Val == llvm::bit_cast<uint32_t>(-4.0f))
    return 247;

  if (Val == 0x3e22f983 && // 1.0 / (2.0 * pi)
      STI.hasFeature(RVGPU::FeatureInv2PiInlineImm))
    return 248;

  return 255;
}

static uint32_t getLit64Encoding(uint64_t Val, const MCSubtargetInfo &STI) {
  uint32_t IntImm = getIntInlineImmEncoding(static_cast<int64_t>(Val));
  if (IntImm != 0)
    return IntImm;

  if (Val == llvm::bit_cast<uint64_t>(0.5))
    return 240;

  if (Val == llvm::bit_cast<uint64_t>(-0.5))
    return 241;

  if (Val == llvm::bit_cast<uint64_t>(1.0))
    return 242;

  if (Val == llvm::bit_cast<uint64_t>(-1.0))
    return 243;

  if (Val == llvm::bit_cast<uint64_t>(2.0))
    return 244;

  if (Val == llvm::bit_cast<uint64_t>(-2.0))
    return 245;

  if (Val == llvm::bit_cast<uint64_t>(4.0))
    return 246;

  if (Val == llvm::bit_cast<uint64_t>(-4.0))
    return 247;

  if (Val == 0x3fc45f306dc9c882 && // 1.0 / (2.0 * pi)
      STI.hasFeature(RVGPU::FeatureInv2PiInlineImm))
    return 248;

  return 255;
}

std::optional<uint32_t>
RVGPUMCCodeEmitter::getLitEncoding(const MCOperand &MO,
                                    const MCOperandInfo &OpInfo,
                                    const MCSubtargetInfo &STI) const {
  int64_t Imm;
  if (MO.isExpr()) {
    const auto *C = dyn_cast<MCConstantExpr>(MO.getExpr());
    if (!C)
      return 255;

    Imm = C->getValue();
  } else {

    assert(!MO.isDFPImm());

    if (!MO.isImm())
      return {};

    Imm = MO.getImm();
  }

  switch (OpInfo.OperandType) {
  case RVGPU::OPERAND_REG_IMM_INT32:
  case RVGPU::OPERAND_REG_IMM_FP32:
  case RVGPU::OPERAND_REG_IMM_FP32_DEFERRED:
  case RVGPU::OPERAND_REG_INLINE_C_INT32:
  case RVGPU::OPERAND_REG_INLINE_C_FP32:
  case RVGPU::OPERAND_REG_INLINE_AC_INT32:
  case RVGPU::OPERAND_REG_INLINE_AC_FP32:
  case RVGPU::OPERAND_REG_IMM_V2INT32:
  case RVGPU::OPERAND_REG_IMM_V2FP32:
  case RVGPU::OPERAND_REG_INLINE_C_V2INT32:
  case RVGPU::OPERAND_REG_INLINE_C_V2FP32:
  case RVGPU::OPERAND_INLINE_SPLIT_BARRIER_INT32:
    return getLit32Encoding(static_cast<uint32_t>(Imm), STI);

  case RVGPU::OPERAND_REG_IMM_INT64:
  case RVGPU::OPERAND_REG_IMM_FP64:
  case RVGPU::OPERAND_REG_INLINE_C_INT64:
  case RVGPU::OPERAND_REG_INLINE_C_FP64:
  case RVGPU::OPERAND_REG_INLINE_AC_FP64:
    return getLit64Encoding(static_cast<uint64_t>(Imm), STI);

  case RVGPU::OPERAND_REG_IMM_INT16:
  case RVGPU::OPERAND_REG_INLINE_C_INT16:
  case RVGPU::OPERAND_REG_INLINE_AC_INT16:
    return getLit16IntEncoding(static_cast<uint16_t>(Imm), STI);
  case RVGPU::OPERAND_REG_IMM_FP16:
  case RVGPU::OPERAND_REG_IMM_FP16_DEFERRED:
  case RVGPU::OPERAND_REG_INLINE_C_FP16:
  case RVGPU::OPERAND_REG_INLINE_AC_FP16:
    // FIXME Is this correct? What do inline immediates do on SI for f16 src
    // which does not have f16 support?
    return getLit16Encoding(static_cast<uint16_t>(Imm), STI);
  case RVGPU::OPERAND_REG_IMM_V2INT16:
  case RVGPU::OPERAND_REG_IMM_V2FP16: {
    if (!isUInt<16>(Imm) && STI.hasFeature(RVGPU::FeatureVOP3Literal))
      return getLit32Encoding(static_cast<uint32_t>(Imm), STI);
    if (OpInfo.OperandType == RVGPU::OPERAND_REG_IMM_V2FP16)
      return getLit16Encoding(static_cast<uint16_t>(Imm), STI);
    [[fallthrough]];
  }
  case RVGPU::OPERAND_REG_INLINE_C_V2INT16:
  case RVGPU::OPERAND_REG_INLINE_AC_V2INT16:
    return getLit16IntEncoding(static_cast<uint16_t>(Imm), STI);
  case RVGPU::OPERAND_REG_INLINE_C_V2FP16:
  case RVGPU::OPERAND_REG_INLINE_AC_V2FP16: {
    uint16_t Lo16 = static_cast<uint16_t>(Imm);
    uint32_t Encoding = getLit16Encoding(Lo16, STI);
    return Encoding;
  }
  case RVGPU::OPERAND_KIMM32:
  case RVGPU::OPERAND_KIMM16:
    return MO.getImm();
  default:
    llvm_unreachable("invalid operand size");
  }
}

uint64_t RVGPUMCCodeEmitter::getImplicitOpSelHiEncoding(int Opcode) const {
  using namespace RVGPU::VOP3PEncoding;
  using namespace RVGPU::OpName;

  if (RVGPU::hasNamedOperand(Opcode, op_sel_hi)) {
    if (RVGPU::hasNamedOperand(Opcode, src2))
      return 0;
    if (RVGPU::hasNamedOperand(Opcode, src1))
      return OP_SEL_HI_2;
    if (RVGPU::hasNamedOperand(Opcode, src0))
      return OP_SEL_HI_1 | OP_SEL_HI_2;
  }
  return OP_SEL_HI_0 | OP_SEL_HI_1 | OP_SEL_HI_2;
}

static bool isVCMPX64(const MCInstrDesc &Desc) {
  return (Desc.TSFlags & SIInstrFlags::VOP3) &&
         Desc.hasImplicitDefOfPhysReg(RVGPU::EXEC);
}

void RVGPUMCCodeEmitter::encodeInstruction(const MCInst &MI,
                                            SmallVectorImpl<char> &CB,
                                            SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {
  int Opcode = MI.getOpcode();
  APInt Encoding, Scratch;
  getBinaryCodeForInstr(MI, Fixups, Encoding, Scratch,  STI);
  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  unsigned bytes = Desc.getSize();

  // Set unused op_sel_hi bits to 1 for VOP3P and MAI instructions.
  // Note that accvgpr_read/write are MAI, have src0, but do not use op_sel.
  if ((Desc.TSFlags & SIInstrFlags::VOP3P) ||
      Opcode == RVGPU::V_ACCVGPR_READ_B32_vi ||
      Opcode == RVGPU::V_ACCVGPR_WRITE_B32_vi) {
    Encoding |= getImplicitOpSelHiEncoding(Opcode);
  }

  // GFX10+ v_cmpx opcodes promoted to VOP3 have implied dst=EXEC.
  // Documentation requires dst to be encoded as EXEC (0x7E),
  // but it looks like the actual value encoded for dst operand
  // is ignored by HW. It was decided to define dst as "do not care"
  // in td files to allow disassembler accept any dst value.
  // However, dst is encoded as EXEC for compatibility with SP3.
  if (RVGPU::isGFX10Plus(STI) && isVCMPX64(Desc)) {
    assert((Encoding & 0xFF) == 0);
    Encoding |= MRI.getEncodingValue(RVGPU::EXEC_LO) &
                RVGPU::HWEncoding::REG_IDX_MASK;
  }

  for (unsigned i = 0; i < bytes; i++) {
    CB.push_back((uint8_t)Encoding.extractBitsAsZExtValue(8, 8 * i));
  }

  // NSA encoding.
  if (RVGPU::isGFX10Plus(STI) && Desc.TSFlags & SIInstrFlags::MIMG) {
    int vaddr0 = RVGPU::getNamedOperandIdx(MI.getOpcode(),
                                            RVGPU::OpName::vaddr0);
    int srsrc = RVGPU::getNamedOperandIdx(MI.getOpcode(),
                                           RVGPU::OpName::srsrc);
    assert(vaddr0 >= 0 && srsrc > vaddr0);
    unsigned NumExtraAddrs = srsrc - vaddr0 - 1;
    unsigned NumPadding = (-NumExtraAddrs) & 3;

    for (unsigned i = 0; i < NumExtraAddrs; ++i) {
      getMachineOpValue(MI, MI.getOperand(vaddr0 + 1 + i), Encoding, Fixups,
                        STI);
      CB.push_back((uint8_t)Encoding.getLimitedValue());
    }
    CB.append(NumPadding, 0);
  }

  if ((bytes > 8 && STI.hasFeature(RVGPU::FeatureVOP3Literal)) ||
      (bytes > 4 && !STI.hasFeature(RVGPU::FeatureVOP3Literal)))
    return;

  // Do not print literals from SISrc Operands for insts with mandatory literals
  if (RVGPU::hasNamedOperand(MI.getOpcode(), RVGPU::OpName::imm))
    return;

  // Check for additional literals
  for (unsigned i = 0, e = Desc.getNumOperands(); i < e; ++i) {

    // Check if this operand should be encoded as [SV]Src
    if (!RVGPU::isSISrcOperand(Desc, i))
      continue;

    // Is this operand a literal immediate?
    const MCOperand &Op = MI.getOperand(i);
    auto Enc = getLitEncoding(Op, Desc.operands()[i], STI);
    if (!Enc || *Enc != 255)
      continue;

    // Yes! Encode it
    int64_t Imm = 0;

    if (Op.isImm())
      Imm = Op.getImm();
    else if (Op.isExpr()) {
      if (const auto *C = dyn_cast<MCConstantExpr>(Op.getExpr()))
        Imm = C->getValue();

    } else if (!Op.isExpr()) // Exprs will be replaced with a fixup value.
      llvm_unreachable("Must be immediate or expr");

    if (Desc.operands()[i].OperandType == RVGPU::OPERAND_REG_IMM_FP64)
      Imm = Hi_32(Imm);

    support::endian::write<uint32_t>(CB, Imm, llvm::endianness::little);

    // Only one literal value allowed
    break;
  }
}

void RVGPUMCCodeEmitter::getSOPPBrEncoding(const MCInst &MI, unsigned OpNo,
                                            APInt &Op,
                                            SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);

  if (MO.isExpr()) {
    const MCExpr *Expr = MO.getExpr();
    MCFixupKind Kind = (MCFixupKind)RVGPU::fixup_si_sopp_br;
    Fixups.push_back(MCFixup::create(0, Expr, Kind, MI.getLoc()));
    Op = APInt::getZero(96);
  } else {
    getMachineOpValue(MI, MO, Op, Fixups, STI);
  }
}

void RVGPUMCCodeEmitter::getSMEMOffsetEncoding(
    const MCInst &MI, unsigned OpNo, APInt &Op,
    SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const {
  auto Offset = MI.getOperand(OpNo).getImm();
  // VI only supports 20-bit unsigned offsets.
  assert(!RVGPU::isVI(STI) || isUInt<20>(Offset));
  Op = Offset;
}

void RVGPUMCCodeEmitter::getSDWASrcEncoding(const MCInst &MI, unsigned OpNo,
                                             APInt &Op,
                                             SmallVectorImpl<MCFixup> &Fixups,
                                             const MCSubtargetInfo &STI) const {
  using namespace RVGPU::SDWA;

  uint64_t RegEnc = 0;

  const MCOperand &MO = MI.getOperand(OpNo);

  if (MO.isReg()) {
    unsigned Reg = MO.getReg();
    RegEnc |= MRI.getEncodingValue(Reg);
    RegEnc &= SDWA9EncValues::SRC_VGPR_MASK;
    if (RVGPU::isSGPR(RVGPU::mc2PseudoReg(Reg), &MRI)) {
      RegEnc |= SDWA9EncValues::SRC_SGPR_MASK;
    }
    Op = RegEnc;
    return;
  } else {
    const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
    auto Enc = getLitEncoding(MO, Desc.operands()[OpNo], STI);
    if (Enc && *Enc != 255) {
      Op = *Enc | SDWA9EncValues::SRC_SGPR_MASK;
      return;
    }
  }

  llvm_unreachable("Unsupported operand kind");
}

void RVGPUMCCodeEmitter::getSDWAVopcDstEncoding(
    const MCInst &MI, unsigned OpNo, APInt &Op,
    SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const {
  using namespace RVGPU::SDWA;

  uint64_t RegEnc = 0;

  const MCOperand &MO = MI.getOperand(OpNo);

  unsigned Reg = MO.getReg();
  if (Reg != RVGPU::VCC && Reg != RVGPU::VCC_LO) {
    RegEnc |= MRI.getEncodingValue(Reg);
    RegEnc &= SDWA9EncValues::VOPC_DST_SGPR_MASK;
    RegEnc |= SDWA9EncValues::VOPC_DST_VCC_MASK;
  }
  Op = RegEnc;
}

void RVGPUMCCodeEmitter::getAVOperandEncoding(
    const MCInst &MI, unsigned OpNo, APInt &Op,
    SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const {
  unsigned Reg = MI.getOperand(OpNo).getReg();
  unsigned Enc = MRI.getEncodingValue(Reg);
  unsigned Idx = Enc & RVGPU::HWEncoding::REG_IDX_MASK;
  bool IsVGPROrAGPR = Enc & RVGPU::HWEncoding::IS_VGPR_OR_AGPR;

  // VGPR and AGPR have the same encoding, but SrcA and SrcB operands of mfma
  // instructions use acc[0:1] modifier bits to distinguish. These bits are
  // encoded as a virtual 9th bit of the register for these operands.
  bool IsAGPR = false;
  if (MRI.getRegClass(RVGPU::AGPR_32RegClassID).contains(Reg) ||
      MRI.getRegClass(RVGPU::AReg_64RegClassID).contains(Reg) ||
      MRI.getRegClass(RVGPU::AReg_96RegClassID).contains(Reg) ||
      MRI.getRegClass(RVGPU::AReg_128RegClassID).contains(Reg) ||
      MRI.getRegClass(RVGPU::AReg_160RegClassID).contains(Reg) ||
      MRI.getRegClass(RVGPU::AReg_192RegClassID).contains(Reg) ||
      MRI.getRegClass(RVGPU::AReg_224RegClassID).contains(Reg) ||
      MRI.getRegClass(RVGPU::AReg_256RegClassID).contains(Reg) ||
      MRI.getRegClass(RVGPU::AReg_288RegClassID).contains(Reg) ||
      MRI.getRegClass(RVGPU::AReg_320RegClassID).contains(Reg) ||
      MRI.getRegClass(RVGPU::AReg_352RegClassID).contains(Reg) ||
      MRI.getRegClass(RVGPU::AReg_384RegClassID).contains(Reg) ||
      MRI.getRegClass(RVGPU::AReg_512RegClassID).contains(Reg) ||
      MRI.getRegClass(RVGPU::AGPR_LO16RegClassID).contains(Reg))
    IsAGPR = true;

  Op = Idx | (IsVGPROrAGPR << 8) | (IsAGPR << 9);
}

static bool needsPCRel(const MCExpr *Expr) {
  switch (Expr->getKind()) {
  case MCExpr::SymbolRef: {
    auto *SE = cast<MCSymbolRefExpr>(Expr);
    MCSymbolRefExpr::VariantKind Kind = SE->getKind();
    return Kind != MCSymbolRefExpr::VK_RVGPU_ABS32_LO &&
           Kind != MCSymbolRefExpr::VK_RVGPU_ABS32_HI;
  }
  case MCExpr::Binary: {
    auto *BE = cast<MCBinaryExpr>(Expr);
    if (BE->getOpcode() == MCBinaryExpr::Sub)
      return false;
    return needsPCRel(BE->getLHS()) || needsPCRel(BE->getRHS());
  }
  case MCExpr::Unary:
    return needsPCRel(cast<MCUnaryExpr>(Expr)->getSubExpr());
  case MCExpr::Target:
  case MCExpr::Constant:
    return false;
  }
  llvm_unreachable("invalid kind");
}

void RVGPUMCCodeEmitter::getMachineOpValue(const MCInst &MI,
                                            const MCOperand &MO, APInt &Op,
                                            SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {
  if (MO.isReg()){
    unsigned Enc = MRI.getEncodingValue(MO.getReg());
    unsigned Idx = Enc & RVGPU::HWEncoding::REG_IDX_MASK;
    bool IsVGPR = Enc & RVGPU::HWEncoding::IS_VGPR_OR_AGPR;
    Op = Idx | (IsVGPR << 8);
    return;
  }
  unsigned OpNo = &MO - MI.begin();
  getMachineOpValueCommon(MI, MO, OpNo, Op, Fixups, STI);
}

void RVGPUMCCodeEmitter::getMachineOpValueT16(
    const MCInst &MI, unsigned OpNo, APInt &Op,
    SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const {
  llvm_unreachable("TODO: Implement getMachineOpValueT16().");
}

void RVGPUMCCodeEmitter::getMachineOpValueT16Lo128(
    const MCInst &MI, unsigned OpNo, APInt &Op,
    SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  if (MO.isReg()) {
    uint16_t Encoding = MRI.getEncodingValue(MO.getReg());
    unsigned RegIdx = Encoding & RVGPU::HWEncoding::REG_IDX_MASK;
    bool IsHi = Encoding & RVGPU::HWEncoding::IS_HI;
    bool IsVGPR = Encoding & RVGPU::HWEncoding::IS_VGPR_OR_AGPR;
    assert((!IsVGPR || isUInt<7>(RegIdx)) && "VGPR0-VGPR127 expected!");
    Op = (IsVGPR ? 0x100 : 0) | (IsHi ? 0x80 : 0) | RegIdx;
    return;
  }
  getMachineOpValueCommon(MI, MO, OpNo, Op, Fixups, STI);
}

void RVGPUMCCodeEmitter::getMachineOpValueCommon(
    const MCInst &MI, const MCOperand &MO, unsigned OpNo, APInt &Op,
    SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const {

  if (MO.isExpr() && MO.getExpr()->getKind() != MCExpr::Constant) {
    // FIXME: If this is expression is PCRel or not should not depend on what
    // the expression looks like. Given that this is just a general expression,
    // it should probably be FK_Data_4 and whatever is producing
    //
    //    s_add_u32 s2, s2, (extern_const_addrspace+16
    //
    // And expecting a PCRel should instead produce
    //
    // .Ltmp1:
    //   s_add_u32 s2, s2, (extern_const_addrspace+16)-.Ltmp1
    MCFixupKind Kind;
    if (needsPCRel(MO.getExpr()))
      Kind = FK_PCRel_4;
    else
      Kind = FK_Data_4;

    const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
    uint32_t Offset = Desc.getSize();
    assert(Offset == 4 || Offset == 8);

    Fixups.push_back(MCFixup::create(Offset, MO.getExpr(), Kind, MI.getLoc()));
  }

  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  if (RVGPU::isSISrcOperand(Desc, OpNo)) {
    if (auto Enc = getLitEncoding(MO, Desc.operands()[OpNo], STI)) {
      Op = *Enc;
      return;
    }
  } else if (MO.isImm()) {
    Op = MO.getImm();
    return;
  }

  llvm_unreachable("Encoding of this operand type is not supported yet.");
}

#include "RVGPUGenMCCodeEmitter.inc"
