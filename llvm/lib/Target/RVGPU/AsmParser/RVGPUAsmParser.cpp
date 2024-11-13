//===- RVGPUAsmParser.cpp - Parse SI asm to MCInst instructions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <optional>

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

using namespace llvm;
using namespace llvm::RVGPU;
//using namespace llvm::amdhsa;

namespace {

class RVGPUAsmParser;

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

  enum ImmTy {
    ImmTyNone,
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
  };

  struct RegOp {
    unsigned RegNo;
  };

  union {
    TokOp Tok;
    ImmOp Imm;
    RegOp Reg;
    const MCExpr *Expr;
  };

public:
  bool isToken() const override { return Kind == Token; }
  bool isImm() const override { return Kind == Immediate; }
  bool isReg() const override { return Kind == Register; }

  bool isRegOrInline(unsigned RCID, MVT type) const {
    return isRegClass(RCID);
  }

  bool isT16VRegWithInputMods() const;

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
      return isVCSrcF32() || isExpr();
  }

  bool isVSrcB64() const {
      return isVCSrcF64();
  }

  bool isVSrcTB16() const { return isVCSrcTB16(); }

  bool isVSrcTB16_Lo128() const {
      return isVCSrcTB16_Lo128();
  }

  bool isVSrcFake16B16_Lo128() const {
      return isVCSrcFake16B16_Lo128();
  }

  bool isVSrcB16() const {
      return isVCSrcB16();
  }

  bool isVSrcV2B16() const {
      return isVSrcB16();
  }

  bool isVCSrcV2FP32() const {
      return isVCSrcF64();
  }

  bool isVSrcV2FP32() const {
      return isVSrcF64();
  }

  bool isVCSrcV2INT32() const {
      return isVCSrcB64();
  }

  bool isVSrcV2INT32() const {
      return isVSrcB64();
  }

  bool isVSrcF32() const {
      return isVCSrcF32() || isExpr();
  }

  bool isVSrcF64() const {
      return isVCSrcF64();
  }

  bool isVSrcTF16() const { return isVCSrcTF16(); }

  bool isVSrcTF16_Lo128() const {
      return isVCSrcTF16_Lo128();
  }

  bool isVSrcFake16F16_Lo128() const {
      return isVCSrcFake16F16_Lo128();
  }

  bool isVSrcF16() const {
      return isVCSrcF16();
  }

  bool isVSrcV2F16() const {
      return isVSrcF16();
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

  bool isExpr() const {
      return Kind == Expression;
  }

  StringRef getToken() const {
      assert(isToken());
      return StringRef(Tok.Data, Tok.Length);
  }

  bool isMem() const override {
      return false;
  }

  unsigned getReg() const override {
      assert(isReg());
      return Reg.RegNo;
  }

  SMLoc getStartLoc() const override {
      return StartLoc;
  }

  SMLoc getEndLoc() const override {
      return EndLoc;
  }

  void print(raw_ostream &OS) const override {
    OS << "TODO\n";
  }

  static RVGPUOperand::Ptr CreateToken(const RVGPUAsmParser *AsmParser, StringRef Str, SMLoc Loc) {
      auto Res = std::make_unique<RVGPUOperand>(Token, AsmParser);
      Res->Tok.Data = Str.data();
      Res->Tok.Length = Str.size();
      Res->StartLoc = Loc;
      Res->EndLoc = Loc;
      return Res;
  }

  void addImmOperands(MCInst &Inst, unsigned N, bool ApplyModifiers = true) const;
  void addRegOperands(MCInst &Inst, unsigned N) const;

  bool isCvtMode() const {
    return false;
  }
  
  void addCvtModeOperands(MCInst &Inst, unsigned N) const {

  }
};

//===----------------------------------------------------------------------===//
// AsmParser
//===----------------------------------------------------------------------===//

class RVGPUAsmParser : public MCTargetAsmParser {
  private:
    MCAsmParser &Parser;

    bool isToken(const AsmToken::TokenKind Kind) const;
    bool trySkipToken(const AsmToken::TokenKind Kind);

    AsmToken::TokenKind getTokenKind() const;
    void lex();

  public:
    RVGPUAsmParser(const MCSubtargetInfo &STI, MCAsmParser &_Parser,
                   const MCInstrInfo &MII,
                   const MCTargetOptions &Options)
                   : MCTargetAsmParser(Options, STI, MII), Parser(_Parser) 
    {
      MCAsmParserExtension::Initialize(Parser);

      setAvailableFeatures(ComputeAvailableFeatures(getFeatureBits()));
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

    bool parseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) override;
    ParseStatus tryParseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) override;
    unsigned validateTargetOperandClass(MCParsedAsmOperand &Op, unsigned Kind) override;
    bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode, OperandVector &Operands, MCStreamer &Out, uint64_t &ErrorInfo, bool MatchingInlineAsm) override;
    bool ParseDirective(AsmToken DirectiveID) override;
    ParseStatus parseOperand(OperandVector &Operands, StringRef Mnemonic);
    bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name, SMLoc NameLoc, OperandVector &Operands) override;

    void onBeginOfFile() override;

    /// Auto-generated Match Functions
#define GET_ASSEMBLER_HEADER
#include "RVGPUGenAsmMatcher.inc"

    ParseStatus parseCvtModeOperand(OperandVector &Operands);
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Operand
//===----------------------------------------------------------------------===//

bool RVGPUOperand::isRegClass(unsigned RCID) const {
  return false;
}

void RVGPUOperand::addImmOperands(MCInst &Inst, unsigned N, bool ApplyModifiers) const {
  return;
}

void RVGPUOperand::addRegOperands(MCInst &Inst, unsigned N) const {
  //Inst.addOperand(MCOperand::createReg(RVGPU::getMCReg(getReg(), AsmParser->getSTI())));
  Inst.addOperand(MCOperand::createReg(getReg()));
}

//===----------------------------------------------------------------------===//
// AsmParser
//===----------------------------------------------------------------------===//

bool RVGPUAsmParser::parseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) {
  return false;
}

ParseStatus RVGPUAsmParser::tryParseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) {
  return ParseStatus::Failure;
}

constexpr unsigned MAX_SRC_OPERANDS_NUM = 6;
using OperandIndices = SmallVector<int16_t, MAX_SRC_OPERANDS_NUM>;

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

  llvm_unreachable("Implement any new match types added!");

  return (Result == Match_Success);
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

  return Result;
}

bool RVGPUAsmParser::ParseInstruction(ParseInstructionInfo &Info,
                                      StringRef Name,
                                      SMLoc NameLoc, OperandVector &Operands) {
  // First operand is token for instruction
  Operands.push_back(RVGPUOperand::CreateToken(this, Name, NameLoc));

  while (!trySkipToken(AsmToken::EndOfStatement)) {
    ParseStatus Res = parseOperand(Operands, Name);

    // Eat the comma or space if there is one.
    trySkipToken(AsmToken::Comma);

    if (Res.isSuccess()) {
      break;
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// parser helpers
//===----------------------------------------------------------------------===//

bool
RVGPUAsmParser::isToken(const AsmToken::TokenKind Kind) const {
  return getTokenKind() == Kind;
}

bool
RVGPUAsmParser::trySkipToken(const AsmToken::TokenKind Kind) {
  if (isToken(Kind)) {
    lex();
    return true;
  }
  return false;
}

void RVGPUAsmParser::onBeginOfFile() {
    getTargetStreamer().EmitDirectiveRVGPUTarget();
}

AsmToken::TokenKind
RVGPUAsmParser::getTokenKind() const {
  return getLexer().getKind();
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