//===- RVGPUAsmParser.cpp - Parse SI asm to MCInst instructions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RVKernelCodeT.h"
#include "MCTargetDesc/RVGPUMCTargetDesc.h"
#include "MCTargetDesc/RVGPUTargetStreamer.h"
#include "RVDefines.h"
#include "RVGPUInstrInfo.h"
#include "RVGPURegisterInfo.h"
#include "TargetInfo/RVGPUTargetInfo.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/MachineValueType.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/TargetParser/TargetParser.h"
#include <optional>

using namespace llvm;
using namespace llvm::RVGPU;
//using namespace llvm::amdhsa;

namespace {

class RVGPUAsmParser;

enum RegisterKind { IS_UNKNOWN, IS_VGPR, IS_SPECIAL };

//===----------------------------------------------------------------------===//
// Operand
//===----------------------------------------------------------------------===//

class RVGPUOperand : public MCParsedAsmOperand {
  enum KindTy {
    Token,
    Immediate,
    Register,
    Expression
  } Kind;

  SMLoc StartLoc, EndLoc;
  const RVGPUAsmParser *AsmParser;

public:
  RVGPUOperand(KindTy Kind_, const RVGPUAsmParser *AsmParser_)
      : Kind(Kind_), AsmParser(AsmParser_) {}

  using Ptr = std::unique_ptr<RVGPUOperand>;

  struct Modifiers {
    bool Abs = false;
    bool Neg = false;
    bool Sext = false;
    bool Lit = false;

    friend raw_ostream &operator <<(raw_ostream &OS, RVGPUOperand::Modifiers Mods);
  };

  enum ImmTy {
    ImmTyNone,
    ImmTyGDS,
    ImmTyLDS,
    ImmTyOffen,
    ImmTyIdxen,
    ImmTyAddr64,
    ImmTyOffset,
    ImmTyInstOffset,
    ImmTyOffset0,
    ImmTyOffset1,
    ImmTySMEMOffsetMod,
    ImmTyCPol,
    ImmTyTFE,
    ImmTyD16,
    ImmTyClampSI,
    ImmTyOModSI,
    ImmTyDMask,
    ImmTyDim,
    ImmTyUNorm,
    ImmTyDA,
    ImmTyR128A16,
    ImmTyA16,
    ImmTyLWE,
    ImmTyExpTgt,
    ImmTyExpCompr,
    ImmTyExpVM,
    ImmTyFORMAT,
    ImmTyHwreg,
    ImmTyOff,
    ImmTySendMsg,
    ImmTyInterpSlot,
    ImmTyInterpAttr,
    ImmTyInterpAttrChan,
    ImmTyOpSel,
    ImmTyOpSelHi,
    ImmTyNegLo,
    ImmTyNegHi,
    ImmTyDPP8,
    ImmTyDppCtrl,
    ImmTyDppRowMask,
    ImmTyDppBankMask,
    ImmTyDppBoundCtrl,
    ImmTyDppFI,
    ImmTySwizzle,
    ImmTyGprIdxMode,
    ImmTyHigh,
    ImmTyBLGP,
    ImmTyCBSZ,
    ImmTyABID,
    ImmTyEndpgm,
    ImmTyWaitVDST,
    ImmTyWaitEXP,
  };

  // Immediate operand kind.
  // It helps to identify the location of an offending operand after an error.
  // Note that regular literals and mandatory literals (KImm) must be handled
  // differently. When looking for an offending operand, we should usually
  // ignore mandatory literals because they are part of the instruction and
  // cannot be changed. Report location of mandatory operands only for VOPD,
  // when both OpX and OpY have a KImm and there are no other literals.
  enum ImmKindTy {
    ImmKindTyNone,
    ImmKindTyLiteral,
    ImmKindTyMandatoryLiteral,
    ImmKindTyConst,
  };

private:
  struct TokOp {
    const char *Data;
    unsigned Length;
  };

  struct ImmOp {
    int64_t Val;
    ImmTy Type;
    bool IsFPImm;
    mutable ImmKindTy Kind;
    Modifiers Mods;
  };

  struct RegOp {
    unsigned RegNo;
    Modifiers Mods;
  };

  union {
    TokOp Tok;
    ImmOp Imm;
    RegOp Reg;
    const MCExpr *Expr;
  };

public:
  bool isToken() const override { return Kind == Token; }

  bool isSymbolRefExpr() const {
    return isExpr() && Expr && isa<MCSymbolRefExpr>(Expr);
  }

  bool isImm() const override {
    return Kind == Immediate;
  }

  void setImmKindNone() const {
    assert(isImm());
    Imm.Kind = ImmKindTyNone;
  }

  bool isLiteralImm(MVT type) const;

  bool isRegKind() const {
    return Kind == Register;
  }

  bool isReg() const override {
    return isRegKind();
  }

  bool isRegOrInline(unsigned RCID, MVT type) const {
    return isRegClass(RCID);
  }

  bool isCvtMode() const {
    return false;
  }
  
  void addCvtModeOperands(MCInst &Inst, unsigned N) const {

  }

  bool isT16VRegWithInputMods() const;

  bool isImmTy(ImmTy ImmT) const {
    return isImm() && Imm.Type == ImmT;
  }

  template <ImmTy Ty> bool isImmTy() const { return isImmTy(Ty); }

  bool isRegClass(unsigned RCID) const;

  bool isRegOrInlineNoMods(unsigned RCID, MVT type) const {
    return isRegOrInline(RCID, type);
  }

  bool isVCSrcB64() const {
      return isRegOrInlineNoMods(RVGPU::VS_64RegClassID, MVT::i64);
  }

  bool isVCSrcTB16() const {
      return isRegOrInlineNoMods(RVGPU::VS_16RegClassID, MVT::i16);
  }

  bool isVCSrcTB16_Lo128() const {
      return isRegOrInlineNoMods(RVGPU::VS_16_Lo128RegClassID, MVT::i16);
  }

  bool isVCSrcFake16B16_Lo128() const {
      return isRegOrInlineNoMods(RVGPU::VS_32_Lo128RegClassID, MVT::i16);
  }

  bool isVCSrcB16() const {
      return isRegOrInlineNoMods(RVGPU::VS_32RegClassID, MVT::i16);
  }

  bool isVCSrcV2B16() const {
      return isVCSrcB16();
  }

  bool isVCSrcF32() const {
      return isRegOrInlineNoMods(RVGPU::VS_32RegClassID, MVT::f32);
  }

  bool isVCSrcF64() const {
      return isRegOrInlineNoMods(RVGPU::VS_64RegClassID, MVT::f64);
  }

  bool isVCSrcTF16() const {
      return isRegOrInlineNoMods(RVGPU::VS_16RegClassID, MVT::f16);
  }

  bool isVCSrcTF16_Lo128() const {
      return isRegOrInlineNoMods(RVGPU::VS_16_Lo128RegClassID, MVT::f16);
  }

  bool isVCSrcFake16F16_Lo128() const {
      return isRegOrInlineNoMods(RVGPU::VS_32_Lo128RegClassID, MVT::f16);
  }

  bool isVCSrcF16() const {
      return isRegOrInlineNoMods(RVGPU::VS_32RegClassID, MVT::f16);
  }

  bool isVCSrcV2F16() const {
      return isVCSrcF16();
  }

  bool isVSrcB32() const {
      return isVCSrcF32() || isLiteralImm(MVT::i32) || isExpr();
  }

  bool isVSrcB64() const {
      return isVCSrcF64() || isLiteralImm(MVT::i64);
  }

  bool isVSrcTB16() const { return isVCSrcTB16() || isLiteralImm(MVT::i16); }

  bool isVSrcTB16_Lo128() const {
      return isVCSrcTB16_Lo128() || isLiteralImm(MVT::i16);
  }

  bool isVSrcFake16B16_Lo128() const {
      return isVCSrcFake16B16_Lo128() || isLiteralImm(MVT::i16);
  }

  bool isVSrcB16() const {
      return isVCSrcB16() || isLiteralImm(MVT::i16);
  }

  bool isVSrcV2B16() const {
      return isVSrcB16() || isLiteralImm(MVT::v2i16);
  }

  bool isVCSrcV2FP32() const {
      return isVCSrcF64();
  }

  bool isVSrcV2FP32() const {
      return isVSrcF64() || isLiteralImm(MVT::v2f32);
  }

  bool isVCSrcV2INT32() const {
      return isVCSrcB64();
  }

  bool isVSrcV2INT32() const {
      return isVSrcB64() || isLiteralImm(MVT::v2i32);
  }

  bool isVSrcF32() const {
      return isVCSrcF32() || isLiteralImm(MVT::f32) || isExpr();
  }

  bool isVSrcF64() const {
      return isVCSrcF64() || isLiteralImm(MVT::f64);
  }

  bool isVSrcTF16() const { return isVCSrcTF16() || isLiteralImm(MVT::f16); }

  bool isVSrcTF16_Lo128() const {
      return isVCSrcTF16_Lo128() || isLiteralImm(MVT::f16);
  }

  bool isVSrcFake16F16_Lo128() const {
      return isVCSrcFake16F16_Lo128() || isLiteralImm(MVT::f16);
  }

  bool isVSrcF16() const {
      return isVCSrcF16() || isLiteralImm(MVT::f16);
  }

  bool isVSrcV2F16() const {
      return isVSrcF16() || isLiteralImm(MVT::v2f16);
  }

  bool isVISrcB32() const {
      return isRegOrInlineNoMods(RVGPU::GPR32RegClassID, MVT::i32);
  }

  bool isVISrcB16() const {
      return isRegOrInlineNoMods(RVGPU::GPR32RegClassID, MVT::i16);
  }

  bool isVISrcV2B16() const {
      return isVISrcB16();
  }

  bool isVISrcF32() const {
      return isRegOrInlineNoMods(RVGPU::GPR32RegClassID, MVT::f32);
  }

  bool isVISrcF16() const {
      return isRegOrInlineNoMods(RVGPU::GPR32RegClassID, MVT::f16);
  }

  bool isVISrc_64B64() const {
      return isRegOrInlineNoMods(RVGPU::GPR64RegClassID, MVT::i64);
  }

  bool isVISrc_64F64() const {
      return isRegOrInlineNoMods(RVGPU::GPR64RegClassID, MVT::f64);
  }

  bool isVISrc_64V2FP32() const {
      return isRegOrInlineNoMods(RVGPU::GPR64RegClassID, MVT::f32);
  }

  bool isVISrc_64V2INT32() const {
      return isRegOrInlineNoMods(RVGPU::GPR64RegClassID, MVT::i32);
  }

  bool isVISrc_256B64() const {
      return isRegOrInlineNoMods(RVGPU::GPR256RegClassID, MVT::i64);
  }

  bool isVISrc_256F64() const {
      return isRegOrInlineNoMods(RVGPU::GPR256RegClassID, MVT::f64);
  }

  bool isVISrc_128B16() const {
      return isRegOrInlineNoMods(RVGPU::GPR128RegClassID, MVT::i16);
  }

  bool isVISrc_128V2B16() const {
      return isVISrc_128B16();
  }

  bool isVISrc_128B32() const {
      return isRegOrInlineNoMods(RVGPU::GPR128RegClassID, MVT::i32);
  }

  bool isVISrc_128F32() const {
      return isRegOrInlineNoMods(RVGPU::GPR128RegClassID, MVT::f32);
  }

  bool isVISrc_256V2FP32() const {
      return isRegOrInlineNoMods(RVGPU::GPR256RegClassID, MVT::f32);
  }

  bool isVISrc_256V2INT32() const {
      return isRegOrInlineNoMods(RVGPU::GPR256RegClassID, MVT::i32);
  }

  bool isVISrc_512B32() const {
      return isRegOrInlineNoMods(RVGPU::GPR512RegClassID, MVT::i32);
  }

  bool isVISrc_512B16() const {
      return isRegOrInlineNoMods(RVGPU::GPR512RegClassID, MVT::i16);
  }

  bool isVISrc_512V2B16() const {
      return isVISrc_512B16();
  }

  bool isVISrc_512F32() const {
      return isRegOrInlineNoMods(RVGPU::GPR512RegClassID, MVT::f32);
  }

  bool isVISrc_512F16() const {
      return isRegOrInlineNoMods(RVGPU::GPR512RegClassID, MVT::f16);
  }

  bool isVISrc_512V2F16() const {
      return isVISrc_512F16() || isVISrc_512B32();
  }

  bool isVISrc_1024B32() const {
      return isRegOrInlineNoMods(RVGPU::GPR1024RegClassID, MVT::i32);
  }

  bool isVISrc_1024B16() const {
      return isRegOrInlineNoMods(RVGPU::GPR1024RegClassID, MVT::i16);
  }

  bool isVISrc_1024V2B16() const {
      return isVISrc_1024B16();
  }

  bool isVISrc_1024F32() const {
      return isRegOrInlineNoMods(RVGPU::GPR1024RegClassID, MVT::f32);
  }

  bool isVISrc_1024F16() const {
      return isRegOrInlineNoMods(RVGPU::GPR1024RegClassID, MVT::f16);
  }

  bool isVISrc_1024V2F16() const {
      return isVISrc_1024F16() || isVISrc_1024B32();
  }

  bool isVISrc_128F16() const {
      return isRegOrInlineNoMods(RVGPU::GPR128RegClassID, MVT::f16);
  }

  bool isVISrc_128V2F16() const {
      return isVISrc_128F16() || isVISrc_128B32();
  }

  bool isKImmFP32() const {
      return isLiteralImm(MVT::f32);
  }

  bool isKImmFP16() const {
      return isLiteralImm(MVT::f16);
  }

  bool isMem() const override {
      return false;
  }

  bool isExpr() const {
      return Kind == Expression;
  }

  auto getPredicate(std::function<bool(const RVGPUOperand &Op)> P) const {
      return std::bind(P, *this);
  }

  StringRef getToken() const {
      assert(isToken());
      return StringRef(Tok.Data, Tok.Length);
  }

  int64_t getImm() const {
      assert(isImm());
      return Imm.Val;
  }

  void setImm(int64_t Val) {
      assert(isImm());
      Imm.Val = Val;
  }

  ImmTy getImmTy() const {
      assert(isImm());
      return Imm.Type;
  }

  unsigned getReg() const override {
      assert(isRegKind());
      return Reg.RegNo;
  }

  SMLoc getStartLoc() const override {
      return StartLoc;
  }

  SMLoc getEndLoc() const override {
      return EndLoc;
  }

  SMRange getLocRange() const {
      return SMRange(StartLoc, EndLoc);
  }

  Modifiers getModifiers() const {
      assert(isRegKind() || isImmTy(ImmTyNone));
      return isRegKind() ? Reg.Mods : Imm.Mods;
  }

  void setModifiers(Modifiers Mods) {
      assert(isRegKind() || isImmTy(ImmTyNone));
      if (isRegKind())
          Reg.Mods = Mods;
      else
          Imm.Mods = Mods;
  }

  uint64_t applyInputFPModifiers(uint64_t Val, unsigned Size) const;

  void addImmOperands(MCInst &Inst, unsigned N, bool ApplyModifiers = true) const;

  void addLiteralImmOperand(MCInst &Inst, int64_t Val, bool ApplyModifiers) const;

  void addRegOperands(MCInst &Inst, unsigned N) const;

  void addRegOrImmOperands(MCInst &Inst, unsigned N) const {
      if (isRegKind())
          addRegOperands(Inst, N);
      else
          addImmOperands(Inst, N);
  }

  static void printImmTy(raw_ostream& OS, ImmTy Type) {
      switch (Type) {
          case ImmTyNone: OS << "None"; break;
          case ImmTyGDS: OS << "GDS"; break;
          case ImmTyLDS: OS << "LDS"; break;
          case ImmTyOffen: OS << "Offen"; break;
          case ImmTyIdxen: OS << "Idxen"; break;
          case ImmTyAddr64: OS << "Addr64"; break;
          case ImmTyOffset: OS << "Offset"; break;
          case ImmTyInstOffset: OS << "InstOffset"; break;
          case ImmTyOffset0: OS << "Offset0"; break;
          case ImmTyOffset1: OS << "Offset1"; break;
          case ImmTySMEMOffsetMod: OS << "SMEMOffsetMod"; break;
          case ImmTyCPol: OS << "CPol"; break;
          case ImmTyTFE: OS << "TFE"; break;
          case ImmTyD16: OS << "D16"; break;
          case ImmTyFORMAT: OS << "FORMAT"; break;
          case ImmTyClampSI: OS << "ClampSI"; break;
          case ImmTyOModSI: OS << "OModSI"; break;
          case ImmTyDPP8: OS << "DPP8"; break;
          case ImmTyDppCtrl: OS << "DppCtrl"; break;
          case ImmTyDppRowMask: OS << "DppRowMask"; break;
          case ImmTyDppBankMask: OS << "DppBankMask"; break;
          case ImmTyDppBoundCtrl: OS << "DppBoundCtrl"; break;
          case ImmTyDppFI: OS << "DppFI"; break;
          case ImmTyDMask: OS << "DMask"; break;
          case ImmTyDim: OS << "Dim"; break;
          case ImmTyUNorm: OS << "UNorm"; break;
          case ImmTyDA: OS << "DA"; break;
          case ImmTyR128A16: OS << "R128A16"; break;
          case ImmTyA16: OS << "A16"; break;
          case ImmTyLWE: OS << "LWE"; break;
          case ImmTyOff: OS << "Off"; break;
          case ImmTyExpTgt: OS << "ExpTgt"; break;
          case ImmTyExpCompr: OS << "ExpCompr"; break;
          case ImmTyExpVM: OS << "ExpVM"; break;
          case ImmTyHwreg: OS << "Hwreg"; break;
          case ImmTySendMsg: OS << "SendMsg"; break;
          case ImmTyInterpSlot: OS << "InterpSlot"; break;
          case ImmTyInterpAttr: OS << "InterpAttr"; break;
          case ImmTyInterpAttrChan: OS << "InterpAttrChan"; break;
          case ImmTyOpSel: OS << "OpSel"; break;
          case ImmTyOpSelHi: OS << "OpSelHi"; break;
          case ImmTyNegLo: OS << "NegLo"; break;
          case ImmTyNegHi: OS << "NegHi"; break;
          case ImmTySwizzle: OS << "Swizzle"; break;
          case ImmTyGprIdxMode: OS << "GprIdxMode"; break;
          case ImmTyHigh: OS << "High"; break;
          case ImmTyBLGP: OS << "BLGP"; break;
          case ImmTyCBSZ: OS << "CBSZ"; break;
          case ImmTyABID: OS << "ABID"; break;
          case ImmTyEndpgm: OS << "Endpgm"; break;
          case ImmTyWaitVDST: OS << "WaitVDST"; break;
          case ImmTyWaitEXP: OS << "WaitEXP"; break;
      }
  }

  void print(raw_ostream &OS) const override {
      switch (Kind) {
          case Register:
              OS << "<register " << getReg() << " mods: " << Reg.Mods << '>';
              break;
          case Immediate:
              OS << '<' << getImm();
              if (getImmTy() != ImmTyNone) {
                  OS << " type: "; printImmTy(OS, getImmTy());
              }
              OS << " mods: " << Imm.Mods << '>';
              break;
          case Token:
              OS << '\'' << getToken() << '\'';
              break;
          case Expression:
              OS << "<expr " << *Expr << '>';
              break;
      }
  }

  static RVGPUOperand::Ptr CreateImm(const RVGPUAsmParser *AsmParser,
          int64_t Val, SMLoc Loc,
          ImmTy Type = ImmTyNone,
          bool IsFPImm = false) {
      auto Op = std::make_unique<RVGPUOperand>(Immediate, AsmParser);
      Op->Imm.Val = Val;
      Op->Imm.IsFPImm = IsFPImm;
      Op->Imm.Kind = ImmKindTyNone;
      Op->Imm.Type = Type;
      Op->Imm.Mods = Modifiers();
      Op->StartLoc = Loc;
      Op->EndLoc = Loc;
      return Op;
  }

  static RVGPUOperand::Ptr CreateToken(const RVGPUAsmParser *AsmParser, StringRef Str, SMLoc Loc) {
      auto Res = std::make_unique<RVGPUOperand>(Token, AsmParser);
      Res->Tok.Data = Str.data();
      Res->Tok.Length = Str.size();
      Res->StartLoc = Loc;
      Res->EndLoc = Loc;
      return Res;
  }

  static RVGPUOperand::Ptr CreateReg(const RVGPUAsmParser *AsmParser,
          unsigned RegNo, SMLoc S,
          SMLoc E) {
      auto Op = std::make_unique<RVGPUOperand>(Register, AsmParser);
      Op->Reg.RegNo = RegNo;
      Op->Reg.Mods = Modifiers();
      Op->StartLoc = S;
      Op->EndLoc = E;
      return Op;
  }

  static RVGPUOperand::Ptr CreateExpr(const RVGPUAsmParser *AsmParser,
          const class MCExpr *Expr, SMLoc S) {
      auto Op = std::make_unique<RVGPUOperand>(Expression, AsmParser);
      Op->Expr = Expr;
      Op->StartLoc = S;
      Op->EndLoc = S;
      return Op;
  }
};

raw_ostream &operator <<(raw_ostream &OS, RVGPUOperand::Modifiers Mods) {
    OS << "abs:" << Mods.Abs << " neg: " << Mods.Neg << " sext:" << Mods.Sext;
    return OS;
}

//===----------------------------------------------------------------------===//
// AsmParser
//===----------------------------------------------------------------------===//

class RVGPUAsmParser : public MCTargetAsmParser {
    MCAsmParser &Parser;

    unsigned ForcedEncodingSize = 0;
    bool ForcedDPP = false;

    /// @name Auto-generated Match Functions
    /// {

#define GET_ASSEMBLER_HEADER
#include "RVGPUGenAsmMatcher.inc"

    /// }

    private:

    bool AddNextRegisterToList(unsigned& Reg, unsigned& RegWidth,
            RegisterKind RegKind, unsigned Reg1, SMLoc Loc);
    bool ParseRVGPURegister(RegisterKind &RegKind, unsigned &Reg,
            unsigned &RegNum, unsigned &RegWidth,
            bool RestoreOnFailure = false);
    bool ParseRVGPURegister(RegisterKind &RegKind, unsigned &Reg,
            unsigned &RegNum, unsigned &RegWidth,
            SmallVectorImpl<AsmToken> &Tokens);
    unsigned ParseRegularReg(RegisterKind &RegKind, unsigned &RegNum,
            unsigned &RegWidth,
            SmallVectorImpl<AsmToken> &Tokens);
    unsigned ParseSpecialReg(RegisterKind &RegKind, unsigned &RegNum,
            unsigned &RegWidth,
            SmallVectorImpl<AsmToken> &Tokens);
    unsigned ParseRegList(RegisterKind &RegKind, unsigned &RegNum,
            unsigned &RegWidth, SmallVectorImpl<AsmToken> &Tokens);
    bool ParseRegRange(unsigned& Num, unsigned& Width);
    unsigned getRegularReg(RegisterKind RegKind,
            unsigned RegNum,
            unsigned RegWidth,
            SMLoc Loc);

    bool isRegister();
    bool isRegister(const AsmToken &Token, const AsmToken &NextToken) const;
    std::optional<StringRef> getGprCountSymbolName(RegisterKind RegKind);
    void initializeGprCountSymbol(RegisterKind RegKind);
    bool updateGprCountSymbols(RegisterKind RegKind, unsigned DwordRegIndex,
            unsigned RegWidth);

    public:

    using OptionalImmIndexMap = std::map<RVGPUOperand::ImmTy, unsigned>;

    RVGPUAsmParser(const MCSubtargetInfo &STI, MCAsmParser &_Parser,
                   const MCInstrInfo &MII,
                   const MCTargetOptions &Options)
                   : MCTargetAsmParser(Options, STI, MII), Parser(_Parser) 
    {
      MCAsmParserExtension::Initialize(Parser);

      if (getFeatureBits().none()) {
        // Set default features.
        copySTI().ToggleFeature("southern-islands");
      }

      setAvailableFeatures(ComputeAvailableFeatures(getFeatureBits()));

      {
        // TODO: make those pre-defined variables read-only.
        // Currently there is none suitable machinery in the core llvm-mc for this.
        // MCSymbol::isRedefinable is intended for another purpose, and
        // AsmParser::parseDirectiveSet() cannot be specialized for specific target.
        initializeGprCountSymbol(IS_VGPR);
      }
    }

    RVGPUTargetStreamer &getTargetStreamer() {
        MCTargetStreamer &TS = *getParser().getStreamer().getTargetStreamer();
        return static_cast<RVGPUTargetStreamer &>(TS);
    }

    const MCRegisterInfo *getMRI() const {
        // We need this const_cast because for some reason getContext() is not const
        // in MCAsmParser.
        return const_cast<RVGPUAsmParser*>(this)->getContext().getRegisterInfo();
    }

    const MCInstrInfo *getMII() const {
        return &MII;
    }

    const FeatureBitset &getFeatureBits() const {
        return getSTI().getFeatureBits();
    }

    StringRef getMatchedVariantName() const;

    std::unique_ptr<RVGPUOperand> parseRegister(bool RestoreOnFailure = false);
    bool ParseRegister(MCRegister &RegNo, SMLoc &StartLoc, SMLoc &EndLoc,
            bool RestoreOnFailure);
    bool parseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) override;
    ParseStatus tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
            SMLoc &EndLoc) override;
    unsigned validateTargetOperandClass(MCParsedAsmOperand &Op,
            unsigned Kind) override;
    bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
            OperandVector &Operands, MCStreamer &Out,
            uint64_t &ErrorInfo,
            bool MatchingInlineAsm) override;
    bool ParseDirective(AsmToken DirectiveID) override;
    ParseStatus parseOperand(OperandVector &Operands, StringRef Mnemonic);
    bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name, SMLoc NameLoc, OperandVector &Operands) override;

    ParseStatus parseImm(OperandVector &Operands, bool HasSP3AbsModifier = false, bool HasLit = false);
    ParseStatus parseReg(OperandVector &Operands);
    ParseStatus parseRegOrImm(OperandVector &Operands, bool HasSP3AbsMod = false, bool HasLit = false);

    private:

    bool validateInstruction(const MCInst &Inst, const SMLoc &IDLoc, const OperandVector &Operands);
    bool validateOffset(const MCInst &Inst, const OperandVector &Operands);
    bool checkUnsupportedInstruction(StringRef Name, const SMLoc &IDLoc);

    bool isId(const StringRef Id) const;
    bool isId(const AsmToken &Token, const StringRef Id) const;
    bool isToken(const AsmToken::TokenKind Kind) const;
    StringRef getId() const;
    bool trySkipId(const StringRef Id);
    bool trySkipId(const StringRef Pref, const StringRef Id);
    bool trySkipId(const StringRef Id, const AsmToken::TokenKind Kind);
    bool trySkipToken(const AsmToken::TokenKind Kind);
    bool skipToken(const AsmToken::TokenKind Kind, const StringRef ErrMsg);
    bool parseString(StringRef &Val, const StringRef ErrMsg = "expected a string");
    bool parseId(StringRef &Val, const StringRef ErrMsg = "");

    AsmToken::TokenKind getTokenKind() const;
    bool parseExpr(int64_t &Imm, StringRef Expected = "");
    bool parseExpr(OperandVector &Operands);
    StringRef getTokenStr() const;
    AsmToken peekToken(bool ShouldSkipSpace = true);
    AsmToken getToken() const;
    SMLoc getLoc() const;
    void lex();

    public:
    void onBeginOfFile() override;

    ParseStatus parseCvtModeOperand(OperandVector &Operands);
};

} // end anonymous namespace

// May be called with integer type with equivalent bitwidth.
static const fltSemantics *getFltSemantics(unsigned Size) {
    switch (Size) {
        case 4:
            return &APFloat::IEEEsingle();
        case 8:
            return &APFloat::IEEEdouble();
        case 2:
            return &APFloat::IEEEhalf();
        default:
            llvm_unreachable("unsupported fp type");
    }
}

static const fltSemantics *getFltSemantics(MVT VT) {
    return getFltSemantics(VT.getSizeInBits() / 8);
}

//===----------------------------------------------------------------------===//
// Operand
//===----------------------------------------------------------------------===//

static bool canLosslesslyConvertToFPType(APFloat &FPLiteral, MVT VT) {
    bool Lost;

    // Convert literal to single precision
    APFloat::opStatus Status = FPLiteral.convert(*getFltSemantics(VT),
            APFloat::rmNearestTiesToEven,
            &Lost);
    // We allow precision lost but not overflow or underflow
    if (Status != APFloat::opOK &&
            Lost &&
            ((Status & APFloat::opOverflow)  != 0 ||
             (Status & APFloat::opUnderflow) != 0)) {
        return false;
    }

    return true;
}

static bool isSafeTruncation(int64_t Val, unsigned Size) {
    return isUIntN(Size, Val) || isIntN(Size, Val);
}

bool RVGPUOperand::isLiteralImm(MVT type) const {
    // Check that this immediate can be added as literal
    if (!isImmTy(ImmTyNone)) {
        return false;
    }

    if (!Imm.IsFPImm) {
        // We got int literal token.

        if (type == MVT::f64) {
            // Cannot apply fp modifiers to int literals preserving the same semantics
            // for VOP1/2/C and VOP3 because of integer truncation. To avoid ambiguity,
            // disable these cases.
            return false;
        }

        unsigned Size = type.getSizeInBits();
        if (Size == 64)
            Size = 32;

        // FIXME: 64-bit operands can zero extend, sign extend, or pad zeroes for FP
        // types.
        return isSafeTruncation(Imm.Val, Size);
    }

    // We got fp literal token
    if (type == MVT::f64) { // Expected 64-bit fp operand
                            // We would set low 64-bits of literal to zeroes but we accept this literals
        return true;
    }

    if (type == MVT::i64) { // Expected 64-bit int operand
                            // We don't allow fp literals in 64-bit integer instructions. It is
                            // unclear how we should encode them.
        return false;
    }

    // We allow fp literals with f16x2 operands assuming that the specified
    // literal goes into the lower half and the upper half is zero. We also
    // require that the literal may be losslessly converted to f16.
    MVT ExpectedType = (type == MVT::v2f16)? MVT::f16 :
        (type == MVT::v2i16)? MVT::i16 :
        (type == MVT::v2f32)? MVT::f32 : type;

    APFloat FPLiteral(APFloat::IEEEdouble(), APInt(64, Imm.Val));
    return canLosslesslyConvertToFPType(FPLiteral, ExpectedType);
}

bool RVGPUOperand::isRegClass(unsigned RCID) const {
    return isRegKind() && AsmParser->getMRI()->getRegClass(RCID).contains(getReg());
}

void RVGPUOperand::addImmOperands(MCInst &Inst, unsigned N, bool ApplyModifiers) const {
  if (isExpr()) {
    Inst.addOperand(MCOperand::createExpr(Expr));
    return;
  }

  {
    assert(!isImmTy(ImmTyNone));
    Inst.addOperand(MCOperand::createImm(Imm.Val));
    setImmKindNone();
  }
}

void RVGPUOperand::addRegOperands(MCInst &Inst, unsigned N) const {
  //Inst.addOperand(MCOperand::createReg(RVGPU::getMCReg(getReg(), AsmParser->getSTI())));
  Inst.addOperand(MCOperand::createReg(getReg()));
}

//===----------------------------------------------------------------------===//
// AsmParser
//===----------------------------------------------------------------------===//

static int getRegClass(RegisterKind Is, unsigned RegWidth) {
  if (Is == IS_VGPR) {
    switch (RegWidth) {
      default: return -1;
      case 32:
        return RVGPU::GPR32RegClassID;
      case 64:
        return RVGPU::GPR64RegClassID;
      case 96:
        return RVGPU::GPR96RegClassID;
      case 128:
        return RVGPU::GPR128RegClassID;
      case 160:
        return RVGPU::GPR160RegClassID;
      case 192:
        return RVGPU::GPR192RegClassID;
      case 224:
        return RVGPU::GPR224RegClassID;
      case 256:
        return RVGPU::GPR256RegClassID;
      case 288:
        return RVGPU::GPR288RegClassID;
      case 320:
        return RVGPU::GPR320RegClassID;
      case 352:
        return RVGPU::GPR352RegClassID;
      case 384:
        return RVGPU::GPR384RegClassID;
      case 512:
        return RVGPU::GPR512RegClassID;
      case 1024:
        return RVGPU::GPR1024RegClassID;
    }
  } 
  return -1;
}

static unsigned getSpecialRegForName(StringRef RegName) {
  return StringSwitch<unsigned>(RegName)
    .Case("exec", RVGPU::EXEC)
    .Case("vcc", RVGPU::VCC)
    /*.Case("shared_base", RVGPU::SRC_SHARED_BASE)
    .Case("src_shared_base", RVGPU::SRC_SHARED_BASE)
    .Case("shared_limit", RVGPU::SRC_SHARED_LIMIT)
    .Case("src_shared_limit", RVGPU::SRC_SHARED_LIMIT)
    .Case("private_base", RVGPU::SRC_PRIVATE_BASE)
    .Case("src_private_base", RVGPU::SRC_PRIVATE_BASE)
    .Case("private_limit", RVGPU::SRC_PRIVATE_LIMIT)
    .Case("src_private_limit", RVGPU::SRC_PRIVATE_LIMIT)
    .Case("pops_exiting_wave_id", RVGPU::SRC_POPS_EXITING_WAVE_ID)
    .Case("src_pops_exiting_wave_id", RVGPU::SRC_POPS_EXITING_WAVE_ID)
    .Case("lds_direct", RVGPU::LDS_DIRECT)
    .Case("src_lds_direct", RVGPU::LDS_DIRECT)
    .Case("m0", RVGPU::M0)
    .Case("vccz", RVGPU::SRC_VCCZ)
    .Case("src_vccz", RVGPU::SRC_VCCZ)
    .Case("execz", RVGPU::SRC_EXECZ)
    .Case("src_execz", RVGPU::SRC_EXECZ)
    .Case("scc", RVGPU::SRC_SCC)
    .Case("src_scc", RVGPU::SRC_SCC)
    .Case("tba", RVGPU::TBA)
    .Case("tma", RVGPU::TMA)
    .Case("flat_scratch_lo", RVGPU::FLAT_SCR_LO)
    .Case("flat_scratch_hi", RVGPU::FLAT_SCR_HI)
    .Case("xnack_mask_lo", RVGPU::XNACK_MASK_LO)
    .Case("xnack_mask_hi", RVGPU::XNACK_MASK_HI)
    .Case("vcc_lo", RVGPU::VCC_LO)
    .Case("vcc_hi", RVGPU::VCC_HI)
    .Case("exec_lo", RVGPU::EXEC_LO)
    .Case("exec_hi", RVGPU::EXEC_HI)
    .Case("tma_lo", RVGPU::TMA_LO)
    .Case("tma_hi", RVGPU::TMA_HI)
    .Case("tba_lo", RVGPU::TBA_LO)
    .Case("tba_hi", RVGPU::TBA_HI)
    .Case("pc", RVGPU::PC_REG)
    .Case("null", RVGPU::SGPR_NULL)*/
    .Default(RVGPU::NoRegister);
}

bool RVGPUAsmParser::ParseRegister(MCRegister &RegNo, SMLoc &StartLoc,
                                    SMLoc &EndLoc, bool RestoreOnFailure) {
  auto R = parseRegister();
  if (!R) return true;
  assert(R->isReg());
  RegNo = R->getReg();
  StartLoc = R->getStartLoc();
  EndLoc = R->getEndLoc();
  return false;
}

bool RVGPUAsmParser::parseRegister(MCRegister &Reg, SMLoc &StartLoc,
                                    SMLoc &EndLoc) {
  return ParseRegister(Reg, StartLoc, EndLoc, /*RestoreOnFailure=*/false);
}

ParseStatus RVGPUAsmParser::tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
                                              SMLoc &EndLoc) {
  bool Result = ParseRegister(Reg, StartLoc, EndLoc, /*RestoreOnFailure=*/true);
  bool PendingErrors = getParser().hasPendingError();
  getParser().clearPendingErrors();
  if (PendingErrors)
    return ParseStatus::Failure;
  if (Result)
    return ParseStatus::NoMatch;
  return ParseStatus::Success;
}

bool RVGPUAsmParser::AddNextRegisterToList(unsigned &Reg, unsigned &RegWidth,
                                            RegisterKind RegKind, unsigned Reg1,
                                            SMLoc Loc) {
  switch (RegKind) {
  case IS_SPECIAL:
    if (Reg == RVGPU::EXEC_LO && Reg1 == RVGPU::EXEC_HI) {
      Reg = RVGPU::EXEC;
      RegWidth = 64;
      return true;
    }
    if (Reg == RVGPU::VCC_LO && Reg1 == RVGPU::VCC_HI) {
      Reg = RVGPU::VCC;
      RegWidth = 64;
      return true;
    }
  case IS_VGPR:
    if (Reg1 != Reg + RegWidth / 32) {
      Error(Loc, "registers in a list must have consecutive indices");
      return false;
    }
    RegWidth += 32;
    return true;
  default:
    llvm_unreachable("unexpected register kind");
  }
}

struct RegInfo {
  StringLiteral Name;
  RegisterKind Kind;
};

static constexpr RegInfo RegularRegisters[] = {
  {{"v"},    IS_VGPR},
};

static bool isRegularReg(RegisterKind Kind) {
  return Kind == IS_VGPR;
}

static const RegInfo* getRegularRegInfo(StringRef Str) {
  for (const RegInfo &Reg : RegularRegisters)
    if (Str.starts_with(Reg.Name))
      return &Reg;
  return nullptr;
}

static bool getRegNum(StringRef Str, unsigned& Num) {
  return !Str.getAsInteger(10, Num);
}

bool
RVGPUAsmParser::isRegister(const AsmToken &Token,
                            const AsmToken &NextToken) const {

  // A list of consecutive registers: [s0,s1,s2,s3]
  if (Token.is(AsmToken::LBrac))
    return true;

  if (!Token.is(AsmToken::Identifier))
    return false;

  // A single register like s0 or a range of registers like s[0:1]

  StringRef Str = Token.getString();
  const RegInfo *Reg = getRegularRegInfo(Str);
  if (Reg) {
    StringRef RegName = Reg->Name;
    StringRef RegSuffix = Str.substr(RegName.size());
    if (!RegSuffix.empty()) {
      unsigned Num;
      // A single register with an index: rXX
      if (getRegNum(RegSuffix, Num))
        return true;
    } else {
      // A range of registers: r[XX:YY].
      if (NextToken.is(AsmToken::LBrac))
        return true;
    }
  }

  return getSpecialRegForName(Str) != RVGPU::NoRegister;
}

bool
RVGPUAsmParser::isRegister()
{
  return isRegister(getToken(), peekToken());
}

unsigned
RVGPUAsmParser::getRegularReg(RegisterKind RegKind,
                               unsigned RegNum,
                               unsigned RegWidth,
                               SMLoc Loc) {

  assert(isRegularReg(RegKind));

  unsigned AlignSize = 1;

  if (RegNum % AlignSize != 0) {
    Error(Loc, "invalid register alignment");
    return RVGPU::NoRegister;
  }

  unsigned RegIdx = RegNum / AlignSize;
  int RCID = getRegClass(RegKind, RegWidth);
  if (RCID == -1) {
    Error(Loc, "invalid or unsupported register size");
    return RVGPU::NoRegister;
  }

  const MCRegisterInfo *TRI = getContext().getRegisterInfo();
  const MCRegisterClass RC = TRI->getRegClass(RCID);
  if (RegIdx >= RC.getNumRegs()) {
    Error(Loc, "register index is out of range");
    return RVGPU::NoRegister;
  }

  return RC.getRegister(RegIdx);
}

bool RVGPUAsmParser::ParseRegRange(unsigned &Num, unsigned &RegWidth) {
  int64_t RegLo, RegHi;
  if (!skipToken(AsmToken::LBrac, "missing register index"))
    return false;

  SMLoc FirstIdxLoc = getLoc();
  SMLoc SecondIdxLoc;

  if (!parseExpr(RegLo))
    return false;

  if (trySkipToken(AsmToken::Colon)) {
    SecondIdxLoc = getLoc();
    if (!parseExpr(RegHi))
      return false;
  } else {
    RegHi = RegLo;
  }

  if (!skipToken(AsmToken::RBrac, "expected a closing square bracket"))
    return false;

  if (!isUInt<32>(RegLo)) {
    Error(FirstIdxLoc, "invalid register index");
    return false;
  }

  if (!isUInt<32>(RegHi)) {
    Error(SecondIdxLoc, "invalid register index");
    return false;
  }

  if (RegLo > RegHi) {
    Error(FirstIdxLoc, "first register index should not exceed second index");
    return false;
  }

  Num = static_cast<unsigned>(RegLo);
  RegWidth = 32 * ((RegHi - RegLo) + 1);
  return true;
}

unsigned RVGPUAsmParser::ParseSpecialReg(RegisterKind &RegKind,
                                          unsigned &RegNum, unsigned &RegWidth,
                                          SmallVectorImpl<AsmToken> &Tokens) {
  assert(isToken(AsmToken::Identifier));
  unsigned Reg = getSpecialRegForName(getTokenStr());
  if (Reg) {
    RegNum = 0;
    RegWidth = 32;
    RegKind = IS_SPECIAL;
    Tokens.push_back(getToken());
    lex(); // skip register name
  }
  return Reg;
}

unsigned RVGPUAsmParser::ParseRegularReg(RegisterKind &RegKind,
                                          unsigned &RegNum, unsigned &RegWidth,
                                          SmallVectorImpl<AsmToken> &Tokens) {
  assert(isToken(AsmToken::Identifier));
  StringRef RegName = getTokenStr();
  auto Loc = getLoc();

  const RegInfo *RI = getRegularRegInfo(RegName);
  if (!RI) {
    Error(Loc, "invalid register name");
    return RVGPU::NoRegister;
  }

  Tokens.push_back(getToken());
  lex(); // skip register name

  RegKind = RI->Kind;
  StringRef RegSuffix = RegName.substr(RI->Name.size());
  if (!RegSuffix.empty()) {
    // Single 32-bit register: vXX.
    if (!getRegNum(RegSuffix, RegNum)) {
      Error(Loc, "invalid register index");
      return RVGPU::NoRegister;
    }
    RegWidth = 32;
  } else {
    // Range of registers: v[XX:YY]. ":YY" is optional.
    if (!ParseRegRange(RegNum, RegWidth))
      return RVGPU::NoRegister;
  }

  return getRegularReg(RegKind, RegNum, RegWidth, Loc);
}

unsigned RVGPUAsmParser::ParseRegList(RegisterKind &RegKind, unsigned &RegNum,
                                       unsigned &RegWidth,
                                       SmallVectorImpl<AsmToken> &Tokens) {
  unsigned Reg = RVGPU::NoRegister;
  auto ListLoc = getLoc();

  if (!skipToken(AsmToken::LBrac,
                 "expected a register or a list of registers")) {
    return RVGPU::NoRegister;
  }

  // List of consecutive registers, e.g.: [s0,s1,s2,s3]

  auto Loc = getLoc();
  if (!ParseRVGPURegister(RegKind, Reg, RegNum, RegWidth))
    return RVGPU::NoRegister;
  if (RegWidth != 32) {
    Error(Loc, "expected a single 32-bit register");
    return RVGPU::NoRegister;
  }

  for (; trySkipToken(AsmToken::Comma); ) {
    RegisterKind NextRegKind;
    unsigned NextReg, NextRegNum, NextRegWidth;
    Loc = getLoc();

    if (!ParseRVGPURegister(NextRegKind, NextReg,
                             NextRegNum, NextRegWidth,
                             Tokens)) {
      return RVGPU::NoRegister;
    }
    if (NextRegWidth != 32) {
      Error(Loc, "expected a single 32-bit register");
      return RVGPU::NoRegister;
    }
    if (NextRegKind != RegKind) {
      Error(Loc, "registers in a list must be of the same kind");
      return RVGPU::NoRegister;
    }
    if (!AddNextRegisterToList(Reg, RegWidth, RegKind, NextReg, Loc))
      return RVGPU::NoRegister;
  }

  if (!skipToken(AsmToken::RBrac,
                 "expected a comma or a closing square bracket")) {
    return RVGPU::NoRegister;
  }

  if (isRegularReg(RegKind))
    Reg = getRegularReg(RegKind, RegNum, RegWidth, ListLoc);

  return Reg;
}

bool RVGPUAsmParser::ParseRVGPURegister(RegisterKind &RegKind, unsigned &Reg,
                                          unsigned &RegNum, unsigned &RegWidth,
                                          SmallVectorImpl<AsmToken> &Tokens) {
  Reg = RVGPU::NoRegister;

  if (isToken(AsmToken::Identifier)) {
    Reg = ParseSpecialReg(RegKind, RegNum, RegWidth, Tokens);
    if (Reg == RVGPU::NoRegister)
      Reg = ParseRegularReg(RegKind, RegNum, RegWidth, Tokens);
  } else {
    Reg = ParseRegList(RegKind, RegNum, RegWidth, Tokens);
  }

  if (Reg == RVGPU::NoRegister) {
    assert(Parser.hasPendingError());
    return false;
  }

  return true;
}

bool RVGPUAsmParser::ParseRVGPURegister(RegisterKind &RegKind, unsigned &Reg,
                                          unsigned &RegNum, unsigned &RegWidth,
                                          bool RestoreOnFailure /*=false*/) {
  Reg = RVGPU::NoRegister;

  SmallVector<AsmToken, 1> Tokens;
  if (ParseRVGPURegister(RegKind, Reg, RegNum, RegWidth, Tokens)) {
    if (RestoreOnFailure) {
      while (!Tokens.empty()) {
        getLexer().UnLex(Tokens.pop_back_val());
      }
    }
    return true;
  }
  return false;
}

std::optional<StringRef>
RVGPUAsmParser::getGprCountSymbolName(RegisterKind RegKind) {
  switch (RegKind) {
  case IS_VGPR:
    return StringRef(".rvgpu.next_free_vgpr");
  default:
    return std::nullopt;
  }
}

void RVGPUAsmParser::initializeGprCountSymbol(RegisterKind RegKind) {
  auto SymbolName = getGprCountSymbolName(RegKind);
  assert(SymbolName && "initializing invalid register kind");
  MCSymbol *Sym = getContext().getOrCreateSymbol(*SymbolName);
  Sym->setVariableValue(MCConstantExpr::create(0, getContext()));
}

bool RVGPUAsmParser::updateGprCountSymbols(RegisterKind RegKind,
                                            unsigned DwordRegIndex,
                                            unsigned RegWidth) {

  auto SymbolName = getGprCountSymbolName(RegKind);
  if (!SymbolName)
    return true;
  MCSymbol *Sym = getContext().getOrCreateSymbol(*SymbolName);

  int64_t NewMax = DwordRegIndex + divideCeil(RegWidth, 32) - 1;
  int64_t OldCount;

  if (!Sym->isVariable())
    return !Error(getLoc(),
                  ".rvgpu.next_free_{v,s}gpr symbols must be variable");
  if (!Sym->getVariableValue(false)->evaluateAsAbsolute(OldCount))
    return !Error(
        getLoc(),
        ".rvgpu.next_free_{v,s}gpr symbols must be absolute expressions");

  if (OldCount <= NewMax)
    Sym->setVariableValue(MCConstantExpr::create(NewMax + 1, getContext()));

  return true;
}

std::unique_ptr<RVGPUOperand>
RVGPUAsmParser::parseRegister(bool RestoreOnFailure) {
  const auto &Tok = getToken();
  SMLoc StartLoc = Tok.getLoc();
  SMLoc EndLoc = Tok.getEndLoc();
  RegisterKind RegKind;
  unsigned Reg, RegNum, RegWidth;

  if (!ParseRVGPURegister(RegKind, Reg, RegNum, RegWidth)) {
    return nullptr;
  }
  if (!updateGprCountSymbols(RegKind, RegNum, RegWidth))
    return nullptr;
  return RVGPUOperand::CreateReg(this, Reg, StartLoc, EndLoc);
}

ParseStatus RVGPUAsmParser::parseImm(OperandVector &Operands,
                                      bool HasSP3AbsModifier, bool HasLit) {
  // TODO: add syntactic sugar for 1/(2*PI)

  if (isRegister())
    return ParseStatus::NoMatch;

  if (!HasLit) {
    HasLit = trySkipId("lit");
    if (HasLit) {
      if (!skipToken(AsmToken::LParen, "expected left paren after lit"))
        return ParseStatus::Failure;
      ParseStatus S = parseImm(Operands, HasSP3AbsModifier, HasLit);
      if (S.isSuccess() &&
          !skipToken(AsmToken::RParen, "expected closing parentheses"))
        return ParseStatus::Failure;
      return S;
    }
  }

  const auto& Tok = getToken();
  const auto& NextTok = peekToken();
  bool IsReal = Tok.is(AsmToken::Real);
  SMLoc S = getLoc();
  bool Negate = false;

  if (!IsReal && Tok.is(AsmToken::Minus) && NextTok.is(AsmToken::Real)) {
    lex();
    IsReal = true;
    Negate = true;
  }

  RVGPUOperand::Modifiers Mods;
  Mods.Lit = HasLit;

  if (IsReal) {
    // Floating-point expressions are not supported.
    // Can only allow floating-point literals with an
    // optional sign.

    StringRef Num = getTokenStr();
    lex();

    APFloat RealVal(APFloat::IEEEdouble());
    auto roundMode = APFloat::rmNearestTiesToEven;
    if (errorToBool(RealVal.convertFromString(Num, roundMode).takeError()))
      return ParseStatus::Failure;
    if (Negate)
      RealVal.changeSign();

    Operands.push_back(
      RVGPUOperand::CreateImm(this, RealVal.bitcastToAPInt().getZExtValue(), S,
                               RVGPUOperand::ImmTyNone, true));
    RVGPUOperand &Op = static_cast<RVGPUOperand &>(*Operands.back());
    Op.setModifiers(Mods);

    return ParseStatus::Success;

  } else {
    int64_t IntVal;
    const MCExpr *Expr;
    SMLoc S = getLoc();

    if (HasSP3AbsModifier) {
      // This is a workaround for handling expressions
      // as arguments of SP3 'abs' modifier, for example:
      //     |1.0|
      //     |-1|
      //     |1+x|
      // This syntax is not compatible with syntax of standard
      // MC expressions (due to the trailing '|').
      SMLoc EndLoc;
      if (getParser().parsePrimaryExpr(Expr, EndLoc, nullptr))
        return ParseStatus::Failure;
    } else {
      if (Parser.parseExpression(Expr))
        return ParseStatus::Failure;
    }

    if (Expr->evaluateAsAbsolute(IntVal)) {
      Operands.push_back(RVGPUOperand::CreateImm(this, IntVal, S));
      RVGPUOperand &Op = static_cast<RVGPUOperand &>(*Operands.back());
      Op.setModifiers(Mods);
    } else {
      if (HasLit)
        return ParseStatus::NoMatch;
      Operands.push_back(RVGPUOperand::CreateExpr(this, Expr, S));
    }

    return ParseStatus::Success;
  }

  return ParseStatus::NoMatch;
}

ParseStatus RVGPUAsmParser::parseReg(OperandVector &Operands) {
  if (!isRegister())
    return ParseStatus::NoMatch;

  if (auto R = parseRegister()) {
    assert(R->isReg());
    Operands.push_back(std::move(R));
    return ParseStatus::Success;
  }
  return ParseStatus::Failure;
}

ParseStatus RVGPUAsmParser::parseRegOrImm(OperandVector &Operands,
                                           bool HasSP3AbsMod, bool HasLit) {
  ParseStatus Res = parseReg(Operands);
  if (!Res.isNoMatch())
    return Res;
  return parseImm(Operands, HasSP3AbsMod, HasLit);
}

StringRef RVGPUAsmParser::getMatchedVariantName() const {
  return "";
}

constexpr unsigned MAX_SRC_OPERANDS_NUM = 6;
using OperandIndices = SmallVector<int16_t, MAX_SRC_OPERANDS_NUM>;

bool RVGPUAsmParser::validateInstruction(const MCInst &Inst,
                                          const SMLoc &IDLoc,
                                          const OperandVector &Operands) {
  return true;
}

bool RVGPUAsmParser::checkUnsupportedInstruction(StringRef Mnemo,
                                                 const SMLoc &IDLoc) {
  FeatureBitset FBS = ComputeAvailableFeatures(getFeatureBits());

  // This instruction is not supported.
  // Clear any other pending errors because they are no longer relevant.
  getParser().clearPendingErrors();

  // Requested instruction variant is not supported.
  // Check if any other variants are supported.
  StringRef VariantName = getMatchedVariantName();
  if (!VariantName.empty()) {
    return Error(IDLoc,
                 Twine(VariantName,
                       " variant of this instruction is not supported"));
  }

  return Error(IDLoc, "invalid instruction");
}

bool RVGPUAsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                              OperandVector &Operands,
                                              MCStreamer &Out,
                                              uint64_t &ErrorInfo,
                                              bool MatchingInlineAsm) {
  MCInst Inst;
  unsigned Result = Match_Success;
  uint64_t EI;
  auto R = MatchInstructionImpl(Operands, Inst, EI, MatchingInlineAsm, MatchingInlineAsm);
  // We order match statuses from least to most specific. We use most specific
  // status as resulting
  // Match_MnemonicFail < Match_InvalidOperand < Match_MissingFeature < Match_PreferE32
  if (R == Match_Success) {
    Result = R;
    ErrorInfo = EI;
  }

  if (Result == Match_Success) {
    if (!validateInstruction(Inst, IDLoc, Operands)) {
      return true;
    }
    Inst.setLoc(IDLoc);
    Out.emitInstruction(Inst, getSTI());
    return false;
  }

  StringRef Mnemo = ((RVGPUOperand &)*Operands[0]).getToken();
  if (checkUnsupportedInstruction(Mnemo, IDLoc)) {
    return true;
  }

  switch (Result) {
  default: break;
  case Match_MissingFeature:
    // It has been verified that the specified instruction
    // mnemonic is valid. A match was found but it requires
    // features which are not supported on this GPU.
    return Error(IDLoc, "operands are not valid for this GPU or mode");

  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0ULL) {
      if (ErrorInfo >= Operands.size()) {
        return Error(IDLoc, "too few operands for instruction");
      }
      ErrorLoc = ((RVGPUOperand &)*Operands[ErrorInfo]).getStartLoc();
      if (ErrorLoc == SMLoc())
        ErrorLoc = IDLoc;

    }
    return Error(ErrorLoc, "invalid operand for instruction");
  }

  case Match_MnemonicFail:
    llvm_unreachable("Invalid instructions should have been handled already");
  }
  llvm_unreachable("Implement any new match types added!");
}

bool RVGPUAsmParser::ParseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getString();

  printf("ParseDirective: %s\n", IDVal.data());
  return true;
}

ParseStatus RVGPUAsmParser::parseOperand(OperandVector &Operands, StringRef Mnemonic) {
  // Check if the current operand has a custom associated parser, if so, try to
  // custom parse the operand, or fallback to the general approach.
  ParseStatus Result = MatchOperandParserImpl(Operands, Mnemonic, /*ParseForAllFeatures=*/true);
  
  // ParseStatus Res;

  // If we successfully parsed the operand or if there as an error parsing,
  // we are done.
  //
  // If we are parsing after we reach EndOfStatement then this means we
  // are appending default values to the Operands list.  This is only done
  // by custom parser, so we shouldn't continue on to the generic parsing.
  //  if (Res.isSuccess() || Res.isFailure() || isToken(AsmToken::EndOfStatement))
  //    return Res;

  SMLoc RBraceLoc;
  SMLoc LBraceLoc = getLoc();
  if (trySkipToken(AsmToken::LBrac)) {
    unsigned Prefix = Operands.size();

    for (;;) {
      auto Loc = getLoc();
      Result = parseReg(Operands);
      if (Result.isNoMatch())
        Error(Loc, "expected a register");
      if (!Result.isSuccess())
        return ParseStatus::Failure;

      RBraceLoc = getLoc();
      if (trySkipToken(AsmToken::RBrac))
        break;

      if (!skipToken(AsmToken::Comma,
                     "expected a comma or a closing square bracket"))
        return ParseStatus::Failure;
    }

    if (Operands.size() - Prefix > 1) {
      Operands.insert(Operands.begin() + Prefix,
                      RVGPUOperand::CreateToken(this, "[", LBraceLoc));
      Operands.push_back(RVGPUOperand::CreateToken(this, "]", RBraceLoc));
    }

    return ParseStatus::Success;
  }

  return parseRegOrImm(Operands);
}

bool RVGPUAsmParser::ParseInstruction(ParseInstructionInfo &Info,
                                       StringRef Name,
                                       SMLoc NameLoc, OperandVector &Operands) {
  // First operand is token for instruction
  Operands.push_back(RVGPUOperand::CreateToken(this, Name, NameLoc));

  while (!trySkipToken(AsmToken::EndOfStatement)) {
    ParseStatus Res = parseOperand(Operands, Name);

    if (!Res.isSuccess()) {
      checkUnsupportedInstruction(Name, NameLoc);
      if (!Parser.hasPendingError()) {
        // FIXME: use real operand location rather than the current location.
        StringRef Msg = Res.isFailure() ? "failed parsing operand."
                                        : "not a valid operand.";
        Error(getLoc(), Msg);
      }
      while (!trySkipToken(AsmToken::EndOfStatement)) {
        lex();
      }
      return true;
    }

    // Eat the comma or space if there is one.
    trySkipToken(AsmToken::Comma);
  }

  return false;
}

//===----------------------------------------------------------------------===//
// parser helpers
//===----------------------------------------------------------------------===//

bool
RVGPUAsmParser::isId(const AsmToken &Token, const StringRef Id) const {
  return Token.is(AsmToken::Identifier) && Token.getString() == Id;
}

bool
RVGPUAsmParser::isId(const StringRef Id) const {
  return isId(getToken(), Id);
}

bool
RVGPUAsmParser::isToken(const AsmToken::TokenKind Kind) const {
  return getTokenKind() == Kind;
}

bool
RVGPUAsmParser::trySkipId(const StringRef Id) {
  if (isId(Id)) {
    lex();
    return true;
  }
  return false;
}

bool
RVGPUAsmParser::trySkipToken(const AsmToken::TokenKind Kind) {
  if (isToken(Kind)) {
    lex();
    return true;
  }
  return false;
}

bool
RVGPUAsmParser::skipToken(const AsmToken::TokenKind Kind,
                           const StringRef ErrMsg) {
  if (!trySkipToken(Kind)) {
    Error(getLoc(), ErrMsg);
    return false;
  }
  return true;
}

void RVGPUAsmParser::onBeginOfFile() {
    getTargetStreamer().EmitDirectiveRVGPUTarget();
}

bool
RVGPUAsmParser::parseExpr(int64_t &Imm, StringRef Expected) {
  SMLoc S = getLoc();

  const MCExpr *Expr;
  if (Parser.parseExpression(Expr))
    return false;

  if (Expr->evaluateAsAbsolute(Imm))
    return true;

  if (Expected.empty()) {
    Error(S, "expected absolute expression");
  } else {
    Error(S, Twine("expected ", Expected) +
             Twine(" or an absolute expression"));
  }
  return false;
}

AsmToken
RVGPUAsmParser::getToken() const {
  return Parser.getTok();
}

AsmToken RVGPUAsmParser::peekToken(bool ShouldSkipSpace) {
  return isToken(AsmToken::EndOfStatement)
             ? getToken()
             : getLexer().peekTok(ShouldSkipSpace);
}

AsmToken::TokenKind
RVGPUAsmParser::getTokenKind() const {
  return getLexer().getKind();
}

SMLoc
RVGPUAsmParser::getLoc() const {
  return getToken().getLoc();
}

StringRef
RVGPUAsmParser::getTokenStr() const {
  return getToken().getString();
}

void
RVGPUAsmParser::lex() {
  Parser.Lex();
}

/// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeRVGPUAsmParser() {
  RegisterMCAsmParser<RVGPUAsmParser> B(getTheRVGPUTarget64());
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#define GET_MNEMONIC_SPELL_CHECKER
#define GET_MNEMONIC_CHECKER
#include "RVGPUGenAsmMatcher.inc"

// This function should be defined after auto-generated include so that we have
// MatchClassKind enum defined
unsigned RVGPUAsmParser::validateTargetOperandClass(MCParsedAsmOperand &Op,
                                                     unsigned Kind) {
  return Match_Success;                                                         
  // Tokens like "glc" would be parsed as immediate operands in ParseOperand().
  // But MatchInstructionImpl() expects to meet token and fails to validate
  // operand. This method checks if we are given immediate operand but expect to
  // get corresponding token.
}

ParseStatus RVGPUAsmParser::parseCvtModeOperand(OperandVector &Operands) {
  return ParseStatus::Failure;
}