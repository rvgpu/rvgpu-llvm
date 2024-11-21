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
#include "MCTargetDesc/RVGPUInstPrinter.h"
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
private:
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

  enum ImmTy {
    ImmTyNone,
  };

  enum KindTy {
    Token,
    Immediate,
    Register,
    Expression,
    Modifier
  } Kind;

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

  struct ModOp {
    uint32_t ModNo;
  };

  union {
    TokOp Tok;
    ImmOp Imm;
    RegOp Reg;
    ModOp Mod;
    const MCExpr *Expr;
  };

  SMLoc StartLoc, EndLoc;
  const RVGPUAsmParser *AsmParser;

public:
  RVGPUOperand(KindTy Kind_, const RVGPUAsmParser *AsmParser_)
      : Kind(Kind_), AsmParser(AsmParser_) {}

  using Ptr = std::unique_ptr<RVGPUOperand>;

  bool isToken() const override { return Kind == KindTy::Token; }
  bool isImm() const override { return Kind == KindTy::Immediate; }
  bool isReg() const override { return Kind == KindTy::Register; }
  bool isCvtMode() const { return Kind == KindTy::Modifier; }

  bool isRegOrInline(unsigned RCID, MVT type) const {
    return isRegClass(RCID);
  }

  bool isT16VRegWithInputMods() const;

  bool isRegClass(unsigned RCID) const;

  bool isRegOrInlineNoMods(unsigned RCID, MVT type) const {
    return isRegOrInline(RCID, type);
  }

  bool isRVSrcB32() const {
      return isRegOrInlineNoMods(RVGPU::GPR32RegClassID, MVT::i32);
  }

  bool isRVSrcB16() const {
      return isRegOrInlineNoMods(RVGPU::GPR32RegClassID, MVT::i16);
  }

  bool isRVSrcV2B16() const {
      return isRVSrcB16();
  }

  bool isRVSrcF32() const {
      return isRegOrInlineNoMods(RVGPU::GPR32RegClassID, MVT::f32);
  }

  bool isRVSrcF16() const {
      return isRegOrInlineNoMods(RVGPU::GPR32RegClassID, MVT::f16);
  }

  bool isRVSrc_64B64() const {
      return isRegOrInlineNoMods(RVGPU::GPR64RegClassID, MVT::i64);
  }

  bool isRVSrc_64F64() const {
      return isRegOrInlineNoMods(RVGPU::GPR64RegClassID, MVT::f64);
  }

  bool isRVSrc_64V2FP32() const {
      return isRegOrInlineNoMods(RVGPU::GPR64RegClassID, MVT::f32);
  }

  bool isRVSrc_64V2INT32() const {
      return isRegOrInlineNoMods(RVGPU::GPR64RegClassID, MVT::i32);
  }

  bool isRVSrc_128B16() const {
      return isRegOrInlineNoMods(RVGPU::GPR128RegClassID, MVT::i16);
  }

  bool isRVSrc_128V2B16() const {
      return isRVSrc_128B16();
  }

  bool isRVSrc_128B32() const {
      return isRegOrInlineNoMods(RVGPU::GPR128RegClassID, MVT::i32);
  }

  bool isRVSrc_128F32() const {
      return isRegOrInlineNoMods(RVGPU::GPR128RegClassID, MVT::f32);
  }

  bool isRVSrc_128F16() const {
      return isRegOrInlineNoMods(RVGPU::GPR128RegClassID, MVT::f16);
  }

  bool isRVSrc_128V2F16() const {
      return isRVSrc_128F16() || isRVSrc_128B32();
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
    switch (Kind)
    {
    case KindTy::Token:
      OS << "<token '" << getToken() << "'";
      break;
    case KindTy::Register: {
      auto RegName = RVGPUInstPrinter::getRegisterName(getReg());
      OS << "<register " << RegName << ">";
      break;
    }
    case KindTy::Modifier:
      OS << "<modifier " << StartLoc.getPointer() << ">";
      break;
    default:
      OS << "RVGPUOperand Print TODO";
      break;
    }
  }

  static RVGPUOperand::Ptr CreateToken(const RVGPUAsmParser *AsmParser, StringRef Str, SMLoc Loc) {
    auto Res = std::make_unique<RVGPUOperand>(KindTy::Token, AsmParser);
    Res->Tok.Data = Str.data();
    Res->Tok.Length = Str.size();
    Res->StartLoc = Loc;
    Res->EndLoc = Loc;
    return Res;
  }

  static RVGPUOperand::Ptr CreateReg(const RVGPUAsmParser *AsmParser, unsigned RegNo, SMLoc SLoc, SMLoc ELoc) {
    auto Res = std::make_unique<RVGPUOperand>(KindTy::Register, AsmParser);
    Res->Reg.RegNo = RegNo;
    Res->StartLoc = SLoc;
    Res->EndLoc = ELoc;
    return Res;
  }

  static RVGPUOperand::Ptr CreateMode(const RVGPUAsmParser *AsmParser, unsigned ModNo, SMLoc SLoc, SMLoc ELoc) {
    auto Res = std::make_unique<RVGPUOperand>(KindTy::Modifier, AsmParser);
    Res->Mod.ModNo = ModNo;
    Res->StartLoc = SLoc;
    Res->EndLoc = ELoc;
    return Res;
  }

  void addImmOperands(MCInst &Inst, unsigned N, bool ApplyModifiers = true) const;
  void addRegOperands(MCInst &Inst, unsigned N) const;
  
  void addCvtModeOperands(MCInst &Inst, unsigned N) const {
  }
};

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

class RVGPUAsmParser : public MCTargetAsmParser {
  private:
    MCAsmParser &Parser;
    StringRef ModifierStr;

    AsmToken::TokenKind getTokenKind() const;

    StringRef parseMnemonicSuffix(StringRef Name);
    bool parseRegister(OperandVector &Operands);

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
    bool parseOperand(OperandVector &Operands, StringRef Mnemonic);
    bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name, SMLoc NameLoc, OperandVector &Operands) override;

    void onBeginOfFile() override;

    /// Auto-generated Match Functions
#define GET_ASSEMBLER_HEADER
#include "RVGPUGenAsmMatcher.inc"

    ParseStatus parseCvtModeOperand(OperandVector &Operands);
};

} // end anonymous namespace

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#define GET_MNEMONIC_SPELL_CHECKER
#define GET_MNEMONIC_CHECKER
#include "RVGPUGenAsmMatcher.inc"

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

  int size = Operands.size();
  auto R = MatchInstructionImpl(Operands, Inst, EI, MatchingInlineAsm, MatchingInlineAsm);

  return (Result == Match_Success);
}

bool RVGPUAsmParser::ParseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getString();

  printf("ParseDirective: %s\n", IDVal.data());
  return true;
}

bool RVGPUAsmParser::parseRegister(OperandVector &Operands) {
  bool ret = true;

  if (getLexer().getKind() == AsmToken::Identifier) {
    StringRef Name = getLexer().getTok().getIdentifier();
    MCRegister RegNo = MatchRegisterName(Name);
    if (RegNo == 0) {
      // 没有匹配到寄存器号
      return false;
    }

    SMLoc S = getParser().getTok().getLoc();
    SMLoc E = SMLoc::getFromPointer(S.getPointer() + Name.size());
    getLexer().Lex();
    Operands.push_back(RVGPUOperand::CreateReg(this, RegNo, S, E));
  } else {
    ret = false;
  }

  return ret;
}

bool RVGPUAsmParser::parseOperand(OperandVector &Operands, StringRef Mnemonic) {
  // Check if the current operand has a custom associated parser, if so, try to
  // custom parse the operand, or fallback to the general approach.
  ParseStatus Result = MatchOperandParserImpl(Operands, Mnemonic, /*ParseForAllFeatures=*/true);
  if (Result.isFailure()) {
    return false;
  }

  if (Result.isSuccess()) {
    return true;
  }

  if (parseRegister(Operands)) {
    return true;
  }

  return false;
}

StringRef RVGPUAsmParser::parseMnemonicSuffix(StringRef Name) {
  if (Name.ends_with(".rn")) {
    ModifierStr = Name.substr(Name.size()-3, Name.size());
    return Name.substr(0, Name.size() - 3);
  }

  return Name;
}

bool RVGPUAsmParser::ParseInstruction(ParseInstructionInfo &Info, StringRef Name, SMLoc NameLoc, OperandVector &Operands) {
  // 处理CVT指令的后缀
  Name = parseMnemonicSuffix(Name);

  // First operand is token for instruction
  Operands.push_back(RVGPUOperand::CreateToken(this, Name, NameLoc));

  // If there are no more operands, then finish
  if (getLexer().is(AsmToken::EndOfStatement)) {
    // Consume the EndOfStatement.
    getParser().Lex();
    return false;
  }

  // Parse first operand
  if (parseOperand(Operands, Name) == false) {
    return true;
  }

  while (parseOptionalToken(AsmToken::Comma)) {
    // Parse next operand
    if (parseOperand(Operands, Name) == false) {
      return true;
    } 
  }

  if (getParser().parseEOL("unexpected token")) {
    getParser().eatToEndOfStatement();
    return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// parser helpers
//===----------------------------------------------------------------------===//

void RVGPUAsmParser::onBeginOfFile() {
    getTargetStreamer().EmitDirectiveRVGPUTarget();
}

AsmToken::TokenKind RVGPUAsmParser::getTokenKind() const {
  return getLexer().getKind();
}

/// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeRVGPUAsmParser() {
  RegisterMCAsmParser<RVGPUAsmParser> B(getTheRVGPUTarget64());
}

// This function should be defined after auto-generated include so that we have
// MatchClassKind enum defined
unsigned RVGPUAsmParser::validateTargetOperandClass(MCParsedAsmOperand &Op, unsigned Kind) {
  return Match_Success;                                                         
  // Tokens like "glc" would be parsed as immediate operands in ParseOperand().
  // But MatchInstructionImpl() expects to meet token and fails to validate
  // operand. This method checks if we are given immediate operand but expect to
  // get corresponding token.
}

ParseStatus RVGPUAsmParser::parseCvtModeOperand(OperandVector &Operands) {
  // CVT.mod dst, rs0

  // 处理modifier
  SMLoc S = SMLoc::getFromPointer(ModifierStr.data());
  SMLoc E = SMLoc::getFromPointer(S.getPointer() + ModifierStr.size());
  Operands.push_back(RVGPUOperand::CreateMode(this, 0, S, E));

  // 处理第一个目的寄存器
  if (parseRegister(Operands) == false) {
    return ParseStatus::Failure;
  }

  // 处理 ','
  if (parseOptionalToken(AsmToken::Comma) == false) {
    return ParseStatus::Failure;
  }

  // 处理第二个源寄存器
  if (parseRegister(Operands) == false) {
    return ParseStatus::Failure;
  }
  
  return ParseStatus::Success;
}