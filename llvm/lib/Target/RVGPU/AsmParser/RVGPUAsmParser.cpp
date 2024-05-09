//===-- RVGPUAsmParser.cpp - Parse RISC-V assembly to MCInst instructions -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//#include "MCTargetDesc/RVGPUAsmBackend.h"
//#include "MCTargetDesc/RVGPUBaseInfo.h"
#include "MCTargetDesc/RVGPUInstPrinter.h"
//#include "MCTargetDesc/RVGPUMCExpr.h"
#include "MCTargetDesc/RVGPUMCTargetDesc.h"
//#include "MCTargetDesc/RVGPUMatInt.h"
#include "MCTargetDesc/RVGPUTargetStreamer.h"
#include "TargetInfo/RVGPUTargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
//#include "llvm/Support/RVGPUAttributes.h"
//#include "llvm/Support/RVGPUISAInfo.h"

#include <limits>

using namespace llvm;

#define DEBUG_TYPE "rvgpu-asm-parser"

STATISTIC(RVGPUNumInstrsCompressed,
          "Number of RVGPU Compressed instructions emitted");

static cl::opt<bool> AddBuildAttributes("riscv-add-build-attributes",
                                        cl::init(false));

namespace llvm {
extern const SubtargetFeatureKV RVGPUFeatureKV[RVGPU::NumSubtargetFeatures];
} // namespace llvm

namespace {
struct RVGPUOperand;

struct ParserOptionsSet {
  bool IsPicEnabled;
};

class RVGPUAsmParser : public MCTargetAsmParser {
  // This tracks the parsing of the 4 operands that make up the vtype portion
  // of vset(i)vli instructions which are separated by commas. The state names
  // represent the next expected operand with Done meaning no other operands are
  // expected.
  enum VTypeState {
    VTypeState_SEW,
    VTypeState_LMUL,
    VTypeState_TailPolicy,
    VTypeState_MaskPolicy,
    VTypeState_Done,
  };

  SmallVector<FeatureBitset, 4> FeatureBitStack;

  SmallVector<ParserOptionsSet, 4> ParserOptionsStack;
  ParserOptionsSet ParserOptions;

  SMLoc getLoc() const { return getParser().getTok().getLoc(); }
  bool isRV64() const { return true; }
  bool isRVE() const { return false; }

  RVGPUTargetStreamer &getTargetStreamer() {
    assert(getParser().getStreamer().getTargetStreamer() &&
           "do not have a target streamer");
    MCTargetStreamer &TS = *getParser().getStreamer().getTargetStreamer();
    return static_cast<RVGPUTargetStreamer &>(TS);
  }

  unsigned validateTargetOperandClass(MCParsedAsmOperand &Op,
                                      unsigned Kind) override;
  unsigned checkTargetMatchPredicate(MCInst &Inst) override;

  bool generateImmOutOfRangeError(OperandVector &Operands, uint64_t ErrorInfo,
                                  int64_t Lower, int64_t Upper,
                                  const Twine &Msg);
  bool generateImmOutOfRangeError(SMLoc ErrorLoc, int64_t Lower, int64_t Upper,
                                  const Twine &Msg);

  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;

  bool parseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) override;
  ParseStatus tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
                               SMLoc &EndLoc) override;

  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

  ParseStatus parseDirective(AsmToken DirectiveID) override;

  bool generateVTypeError(SMLoc ErrorLoc);

  // Helper to actually emit an instruction to the MCStreamer. Also, when
  // possible, compression of the instruction is performed.
  void emitToStreamer(MCStreamer &S, const MCInst &Inst);

  // Check instruction constraints.
  bool validateInstruction(MCInst &Inst, OperandVector &Operands);

  /// Helper for processing MC instructions that have been successfully matched
  /// by MatchAndEmitInstruction. Modifications to the emitted instructions,
  /// like the expansion of pseudo instructions (e.g., "li"), can be performed
  /// in this method.
  bool processInstruction(MCInst &Inst, SMLoc IDLoc, OperandVector &Operands,
                          MCStreamer &Out);

// Auto-generated instruction matching functions
#define GET_ASSEMBLER_HEADER
#include "RVGPUGenAsmMatcher.inc"
  ParseStatus parseFPImm(OperandVector &Operands);
  ParseStatus parseImmediate(OperandVector &Operands);
  ParseStatus parseRegister(OperandVector &Operands, bool AllowParens = false);
  ParseStatus parseMemOpBaseReg(OperandVector &Operands);
  ParseStatus parseOperandWithModifier(OperandVector &Operands);
  ParseStatus parseBareSymbol(OperandVector &Operands);
  ParseStatus parseCallSymbol(OperandVector &Operands);
  bool parseOperand(OperandVector &Operands, StringRef Mnemonic);

public:
  enum RVGPUMatchResultTy {
    Match_Dummy = FIRST_TARGET_MATCH_RESULT_TY,
    Match_RequiresEvenGPRs,
#define GET_OPERAND_DIAGNOSTIC_TYPES
#include "RVGPUGenAsmMatcher.inc"
#undef GET_OPERAND_DIAGNOSTIC_TYPES
  };

  static bool classifySymbolRef(const MCExpr *Expr,
                                RVGPUMCExpr::VariantKind &Kind);
  static bool isSymbolDiff(const MCExpr *Expr);

  RVGPUAsmParser(const MCSubtargetInfo &STI, MCAsmParser &Parser,
                 const MCInstrInfo &MII, const MCTargetOptions &Options)
      : MCTargetAsmParser(Options, STI, MII) {
    MCAsmParserExtension::Initialize(Parser);

    Parser.addAliasForDirective(".half", ".2byte");
    Parser.addAliasForDirective(".hword", ".2byte");
    Parser.addAliasForDirective(".word", ".4byte");
    Parser.addAliasForDirective(".dword", ".8byte");
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));

    auto ABIName = StringRef(Options.ABIName);
    if (ABIName.ends_with("f") && !getSTI().hasFeature(RVGPU::FeatureStdExtF)) {
      errs() << "Hard-float 'f' ABI can't be used for a target that "
                "doesn't support the F instruction set extension (ignoring "
                "target-abi)\n";
    } else if (ABIName.ends_with("d") &&
               !getSTI().hasFeature(RVGPU::FeatureStdExtD)) {
      errs() << "Hard-float 'd' ABI can't be used for a target that "
                "doesn't support the D instruction set extension (ignoring "
                "target-abi)\n";
    }

    // Use computeTargetABI to check if ABIName is valid. If invalid, output
    // error message.
    RVGPUABI::computeTargetABI(STI.getTargetTriple(), STI.getFeatureBits(),
                               ABIName);

    const MCObjectFileInfo *MOFI = Parser.getContext().getObjectFileInfo();
    ParserOptions.IsPicEnabled = MOFI->isPositionIndependent();

    if (AddBuildAttributes)
      getTargetStreamer().emitTargetAttributes(STI, /*EmitStackAlign*/ false);
  }
};

/// RVGPUOperand - Instances of this class represent a parsed machine
/// instruction
struct RVGPUOperand final : public MCParsedAsmOperand {

  enum class KindTy {
    Token,
    Register,
    Immediate,
    FPImmediate,
    SystemRegister,
    VType,
    FRM,
    Fence,
    Rlist,
    Spimm,
    RegReg,
  } Kind;

  struct RegOp {
    MCRegister RegNum;
    bool IsGPRAsFPR;
  };

  struct ImmOp {
    const MCExpr *Val;
    bool IsRV64;
  };

  struct FPImmOp {
    uint64_t Val;
  };

  struct SysRegOp {
    const char *Data;
    unsigned Length;
    unsigned Encoding;
    // FIXME: Add the Encoding parsed fields as needed for checks,
    // e.g.: read/write or user/supervisor/machine privileges.
  };

  struct VTypeOp {
    unsigned Val;
  };

  struct FRMOp {
    RVGPUFPRndMode::RoundingMode FRM;
  };

  struct FenceOp {
    unsigned Val;
  };

  struct RlistOp {
    unsigned Val;
  };

  struct SpimmOp {
    unsigned Val;
  };

  struct RegRegOp {
    MCRegister Reg1;
    MCRegister Reg2;
  };

  SMLoc StartLoc, EndLoc;
  union {
    StringRef Tok;
    RegOp Reg;
    ImmOp Imm;
    FPImmOp FPImm;
    struct SysRegOp SysReg;
    struct VTypeOp VType;
    struct FRMOp FRM;
    struct FenceOp Fence;
    struct RlistOp Rlist;
    struct SpimmOp Spimm;
    struct RegRegOp RegReg;
  };

  RVGPUOperand(KindTy K) : Kind(K) {}

public:
  RVGPUOperand(const RVGPUOperand &o) : MCParsedAsmOperand() {
    Kind = o.Kind;
    StartLoc = o.StartLoc;
    EndLoc = o.EndLoc;
    switch (Kind) {
    case KindTy::Register:
      Reg = o.Reg;
      break;
    case KindTy::Immediate:
      Imm = o.Imm;
      break;
    case KindTy::FPImmediate:
      FPImm = o.FPImm;
      break;
    case KindTy::Token:
      Tok = o.Tok;
      break;
    case KindTy::SystemRegister:
      SysReg = o.SysReg;
      break;
    case KindTy::VType:
      VType = o.VType;
      break;
    case KindTy::FRM:
      FRM = o.FRM;
      break;
    case KindTy::Fence:
      Fence = o.Fence;
      break;
    case KindTy::Rlist:
      Rlist = o.Rlist;
      break;
    case KindTy::Spimm:
      Spimm = o.Spimm;
      break;
    case KindTy::RegReg:
      RegReg = o.RegReg;
      break;
    }
  }

  bool isToken() const override { return Kind == KindTy::Token; }
  bool isReg() const override { return Kind == KindTy::Register; }
  bool isImm() const override { return Kind == KindTy::Immediate; }
  bool isMem() const override { return false; }
  bool isSystemRegister() const { return Kind == KindTy::SystemRegister; }
  bool isRegReg() const { return Kind == KindTy::RegReg; }
  bool isRlist() const { return Kind == KindTy::Rlist; }
  bool isSpimm() const { return Kind == KindTy::Spimm; }

  bool isGPR() const {
    return Kind == KindTy::Register &&
           RVGPUMCRegisterClasses[RVGPU::GPRRegClassID].contains(Reg.RegNum);
  }


  static bool evaluateConstantImm(const MCExpr *Expr, int64_t &Imm,
                                  RVGPUMCExpr::VariantKind &VK) {
    if (auto *RE = dyn_cast<RVGPUMCExpr>(Expr)) {
      VK = RE->getKind();
      return RE->evaluateAsConstant(Imm);
    }

    if (auto CE = dyn_cast<MCConstantExpr>(Expr)) {
      VK = RVGPUMCExpr::VK_RVGPU_None;
      Imm = CE->getValue();
      return true;
    }

    return false;
  }

  // True if operand is a symbol with no modifiers, or a constant with no
  // modifiers and isShiftedInt<N-1, 1>(Op).
  template <int N> bool isBareSimmNLsb0() const {
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    bool IsValid;
    if (!IsConstantImm)
      IsValid = RVGPUAsmParser::classifySymbolRef(getImm(), VK);
    else
      IsValid = isShiftedInt<N - 1, 1>(Imm);
    return IsValid && VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  // Predicate methods for AsmOperands defined in RVGPUInstrInfo.td

  bool isBareSymbol() const {
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    // Must be of 'immediate' type but not a constant.
    if (!isImm() || evaluateConstantImm(getImm(), Imm, VK))
      return false;
    return RVGPUAsmParser::classifySymbolRef(getImm(), VK) &&
           VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  bool isCallSymbol() const {
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    // Must be of 'immediate' type but not a constant.
    if (!isImm() || evaluateConstantImm(getImm(), Imm, VK))
      return false;
    return RVGPUAsmParser::classifySymbolRef(getImm(), VK) &&
           (VK == RVGPUMCExpr::VK_RVGPU_CALL ||
            VK == RVGPUMCExpr::VK_RVGPU_CALL_PLT);
  }

  bool isPseudoJumpSymbol() const {
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    // Must be of 'immediate' type but not a constant.
    if (!isImm() || evaluateConstantImm(getImm(), Imm, VK))
      return false;
    return RVGPUAsmParser::classifySymbolRef(getImm(), VK) &&
           VK == RVGPUMCExpr::VK_RVGPU_CALL;
  }

  bool isTPRelAddSymbol() const {
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    // Must be of 'immediate' type but not a constant.
    if (!isImm() || evaluateConstantImm(getImm(), Imm, VK))
      return false;
    return RVGPUAsmParser::classifySymbolRef(getImm(), VK) &&
           VK == RVGPUMCExpr::VK_RVGPU_TPREL_ADD;
  }

  bool isCSRSystemRegister() const { return isSystemRegister(); }

  bool isVTypeImm(unsigned N) const {
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isUIntN(N, Imm) && VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  template <unsigned N> bool IsUImm() const {
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isUInt<N>(Imm) && VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  bool isUImm1() const { return IsUImm<1>(); }
  bool isUImm2() const { return IsUImm<2>(); }
  bool isUImm3() const { return IsUImm<3>(); }
  bool isUImm4() const { return IsUImm<4>(); }
  bool isUImm5() const { return IsUImm<5>(); }
  bool isUImm6() const { return IsUImm<6>(); }
  bool isUImm7() const { return IsUImm<7>(); }
  bool isUImm8() const { return IsUImm<8>(); }
  bool isUImm20() const { return IsUImm<20>(); }

  bool isUImm8GE32() const {
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isUInt<8>(Imm) && Imm >= 32 &&
           VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  bool isRnumArg() const {
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && Imm >= INT64_C(0) && Imm <= INT64_C(10) &&
           VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  bool isRnumArg_0_7() const {
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && Imm >= INT64_C(0) && Imm <= INT64_C(7) &&
           VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  bool isRnumArg_1_10() const {
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && Imm >= INT64_C(1) && Imm <= INT64_C(10) &&
           VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  bool isRnumArg_2_14() const {
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && Imm >= INT64_C(2) && Imm <= INT64_C(14) &&
           VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  bool isSImm5() const {
    if (!isImm())
      return false;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    int64_t Imm;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isInt<5>(fixImmediateForRV32(Imm, isRV64Imm())) &&
           VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  bool isSImm6() const {
    if (!isImm())
      return false;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    int64_t Imm;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isInt<6>(fixImmediateForRV32(Imm, isRV64Imm())) &&
           VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  bool isSImm6NonZero() const {
    if (!isImm())
      return false;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    int64_t Imm;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && Imm != 0 &&
           isInt<6>(fixImmediateForRV32(Imm, isRV64Imm())) &&
           VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  bool isCLUIImm() const {
    if (!isImm())
      return false;
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && (Imm != 0) &&
           (isUInt<5>(Imm) || (Imm >= 0xfffe0 && Imm <= 0xfffff)) &&
           VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  bool isUImm2Lsb0() const {
    if (!isImm())
      return false;
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isShiftedUInt<1, 1>(Imm) &&
           VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  bool isUImm7Lsb00() const {
    if (!isImm())
      return false;
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isShiftedUInt<5, 2>(Imm) &&
           VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  bool isUImm8Lsb00() const {
    if (!isImm())
      return false;
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isShiftedUInt<6, 2>(Imm) &&
           VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  bool isUImm8Lsb000() const {
    if (!isImm())
      return false;
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isShiftedUInt<5, 3>(Imm) &&
           VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  bool isSImm9Lsb0() const { return isBareSimmNLsb0<9>(); }

  bool isUImm9Lsb000() const {
    if (!isImm())
      return false;
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isShiftedUInt<6, 3>(Imm) &&
           VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  bool isUImm10Lsb00NonZero() const {
    if (!isImm())
      return false;
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isShiftedUInt<8, 2>(Imm) && (Imm != 0) &&
           VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  // If this a RV32 and the immediate is a uimm32, sign extend it to 32 bits.
  // This allows writing 'addi a0, a0, 0xffffffff'.
  static int64_t fixImmediateForRV32(int64_t Imm, bool IsRV64Imm) {
    if (IsRV64Imm || !isUInt<32>(Imm))
      return Imm;
    return SignExtend64<32>(Imm);
  }

  bool isSImm12() const {
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    int64_t Imm;
    bool IsValid;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    if (!IsConstantImm)
      IsValid = RVGPUAsmParser::classifySymbolRef(getImm(), VK);
    else
      IsValid = isInt<12>(fixImmediateForRV32(Imm, isRV64Imm()));
    return IsValid && ((IsConstantImm && VK == RVGPUMCExpr::VK_RVGPU_None) ||
                       VK == RVGPUMCExpr::VK_RVGPU_LO ||
                       VK == RVGPUMCExpr::VK_RVGPU_PCREL_LO ||
                       VK == RVGPUMCExpr::VK_RVGPU_TPREL_LO);
  }

  bool isSImm12Lsb0() const { return isBareSimmNLsb0<12>(); }

  bool isSImm12Lsb00000() const {
    if (!isImm())
      return false;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    int64_t Imm;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isShiftedInt<7, 5>(Imm) &&
           VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  bool isSImm13Lsb0() const { return isBareSimmNLsb0<13>(); }

  bool isSImm10Lsb0000NonZero() const {
    if (!isImm())
      return false;
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && (Imm != 0) && isShiftedInt<6, 4>(Imm) &&
           VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  bool isUImm20LUI() const {
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    int64_t Imm;
    bool IsValid;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    if (!IsConstantImm) {
      IsValid = RVGPUAsmParser::classifySymbolRef(getImm(), VK);
      return IsValid && (VK == RVGPUMCExpr::VK_RVGPU_HI ||
                         VK == RVGPUMCExpr::VK_RVGPU_TPREL_HI);
    } else {
      return isUInt<20>(Imm) && (VK == RVGPUMCExpr::VK_RVGPU_None ||
                                 VK == RVGPUMCExpr::VK_RVGPU_HI ||
                                 VK == RVGPUMCExpr::VK_RVGPU_TPREL_HI);
    }
  }

  bool isUImm20AUIPC() const {
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    int64_t Imm;
    bool IsValid;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    if (!IsConstantImm) {
      IsValid = RVGPUAsmParser::classifySymbolRef(getImm(), VK);
      return IsValid && (VK == RVGPUMCExpr::VK_RVGPU_PCREL_HI ||
                         VK == RVGPUMCExpr::VK_RVGPU_GOT_HI ||
                         VK == RVGPUMCExpr::VK_RVGPU_TLS_GOT_HI ||
                         VK == RVGPUMCExpr::VK_RVGPU_TLS_GD_HI);
    } else {
      return isUInt<20>(Imm) && (VK == RVGPUMCExpr::VK_RVGPU_None ||
                                 VK == RVGPUMCExpr::VK_RVGPU_PCREL_HI ||
                                 VK == RVGPUMCExpr::VK_RVGPU_GOT_HI ||
                                 VK == RVGPUMCExpr::VK_RVGPU_TLS_GOT_HI ||
                                 VK == RVGPUMCExpr::VK_RVGPU_TLS_GD_HI);
    }
  }

  bool isSImm21Lsb0JAL() const { return isBareSimmNLsb0<21>(); }

  bool isImmZero() const {
    if (!isImm())
      return false;
    int64_t Imm;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && (Imm == 0) && VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  bool isSImm5Plus1() const {
    if (!isImm())
      return false;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    int64_t Imm;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm &&
           isInt<5>(fixImmediateForRV32(Imm, isRV64Imm()) - 1) &&
           VK == RVGPUMCExpr::VK_RVGPU_None;
  }

  /// getStartLoc - Gets location of the first token of this operand
  SMLoc getStartLoc() const override { return StartLoc; }
  /// getEndLoc - Gets location of the last token of this operand
  SMLoc getEndLoc() const override { return EndLoc; }
  /// True if this operand is for an RV64 instruction
  bool isRV64Imm() const {
    assert(Kind == KindTy::Immediate && "Invalid type access!");
    return Imm.IsRV64;
  }

  unsigned getReg() const override {
    assert(Kind == KindTy::Register && "Invalid type access!");
    return Reg.RegNum.id();
  }

  StringRef getSysReg() const {
    assert(Kind == KindTy::SystemRegister && "Invalid type access!");
    return StringRef(SysReg.Data, SysReg.Length);
  }

  const MCExpr *getImm() const {
    assert(Kind == KindTy::Immediate && "Invalid type access!");
    return Imm.Val;
  }

  uint64_t getFPConst() const {
    assert(Kind == KindTy::FPImmediate && "Invalid type access!");
    return FPImm.Val;
  }

  StringRef getToken() const {
    assert(Kind == KindTy::Token && "Invalid type access!");
    return Tok;
  }

  unsigned getVType() const {
    assert(Kind == KindTy::VType && "Invalid type access!");
    return VType.Val;
  }

  RVGPUFPRndMode::RoundingMode getFRM() const {
    assert(Kind == KindTy::FRM && "Invalid type access!");
    return FRM.FRM;
  }

  unsigned getFence() const {
    assert(Kind == KindTy::Fence && "Invalid type access!");
    return Fence.Val;
  }

  void print(raw_ostream &OS) const override {
    auto RegName = [](MCRegister Reg) {
      if (Reg)
        return RVGPUInstPrinter::getRegisterName(Reg);
      else
        return "noreg";
    };

    switch (Kind) {
    case KindTy::Immediate:
      OS << *getImm();
      break;
    case KindTy::FPImmediate:
      break;
    case KindTy::Register:
      OS << "<register " << RegName(getReg()) << ">";
      break;
    case KindTy::Token:
      OS << "'" << getToken() << "'";
      break;
    case KindTy::SystemRegister:
      OS << "<sysreg: " << getSysReg() << '>';
      break;
    case KindTy::VType:
      OS << "<vtype: ";
      RVGPUVType::printVType(getVType(), OS);
      OS << '>';
      break;
    case KindTy::FRM:
      OS << "<frm: ";
      roundingModeToString(getFRM());
      OS << '>';
      break;
    case KindTy::Fence:
      OS << "<fence: ";
      OS << getFence();
      OS << '>';
      break;
    case KindTy::Rlist:
      OS << "<rlist: ";
      RVGPUZC::printRlist(Rlist.Val, OS);
      OS << '>';
      break;
    case KindTy::Spimm:
      OS << "<Spimm: ";
      RVGPUZC::printSpimm(Spimm.Val, OS);
      OS << '>';
      break;
    case KindTy::RegReg:
      OS << "<RegReg:  Reg1 " << RegName(RegReg.Reg1);
      OS << " Reg2 " << RegName(RegReg.Reg2);
      break;
    }
  }

  static std::unique_ptr<RVGPUOperand> createToken(StringRef Str, SMLoc S) {
    auto Op = std::make_unique<RVGPUOperand>(KindTy::Token);
    Op->Tok = Str;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<RVGPUOperand>
  createReg(unsigned RegNo, SMLoc S, SMLoc E, bool IsGPRAsFPR = false) {
    auto Op = std::make_unique<RVGPUOperand>(KindTy::Register);
    Op->Reg.RegNum = RegNo;
    Op->Reg.IsGPRAsFPR = IsGPRAsFPR;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<RVGPUOperand> createImm(const MCExpr *Val, SMLoc S,
                                                 SMLoc E, bool IsRV64) {
    auto Op = std::make_unique<RVGPUOperand>(KindTy::Immediate);
    Op->Imm.Val = Val;
    Op->Imm.IsRV64 = IsRV64;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<RVGPUOperand> createFPImm(uint64_t Val, SMLoc S) {
    auto Op = std::make_unique<RVGPUOperand>(KindTy::FPImmediate);
    Op->FPImm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<RVGPUOperand> createSysReg(StringRef Str, SMLoc S,
                                                    unsigned Encoding) {
    auto Op = std::make_unique<RVGPUOperand>(KindTy::SystemRegister);
    Op->SysReg.Data = Str.data();
    Op->SysReg.Length = Str.size();
    Op->SysReg.Encoding = Encoding;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<RVGPUOperand>
  createFRMArg(RVGPUFPRndMode::RoundingMode FRM, SMLoc S) {
    auto Op = std::make_unique<RVGPUOperand>(KindTy::FRM);
    Op->FRM.FRM = FRM;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<RVGPUOperand> createFenceArg(unsigned Val, SMLoc S) {
    auto Op = std::make_unique<RVGPUOperand>(KindTy::Fence);
    Op->Fence.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<RVGPUOperand> createVType(unsigned VTypeI, SMLoc S) {
    auto Op = std::make_unique<RVGPUOperand>(KindTy::VType);
    Op->VType.Val = VTypeI;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<RVGPUOperand> createRlist(unsigned RlistEncode,
                                                   SMLoc S) {
    auto Op = std::make_unique<RVGPUOperand>(KindTy::Rlist);
    Op->Rlist.Val = RlistEncode;
    Op->StartLoc = S;
    return Op;
  }

  static std::unique_ptr<RVGPUOperand> createRegReg(unsigned Reg1No,
                                                    unsigned Reg2No, SMLoc S) {
    auto Op = std::make_unique<RVGPUOperand>(KindTy::RegReg);
    Op->RegReg.Reg1 = Reg1No;
    Op->RegReg.Reg2 = Reg2No;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<RVGPUOperand> createSpimm(unsigned Spimm, SMLoc S) {
    auto Op = std::make_unique<RVGPUOperand>(KindTy::Spimm);
    Op->Spimm.Val = Spimm;
    Op->StartLoc = S;
    return Op;
  }

  static void addExpr(MCInst &Inst, const MCExpr *Expr, bool IsRV64Imm) {
    assert(Expr && "Expr shouldn't be null!");
    int64_t Imm = 0;
    RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::VK_RVGPU_None;
    bool IsConstant = evaluateConstantImm(Expr, Imm, VK);

    if (IsConstant)
      Inst.addOperand(
          MCOperand::createImm(fixImmediateForRV32(Imm, IsRV64Imm)));
    else
      Inst.addOperand(MCOperand::createExpr(Expr));
  }

  // Used by the TableGen Code
  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getReg()));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm(), isRV64Imm());
  }

  void addFPImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    if (isImm()) {
      addExpr(Inst, getImm(), isRV64Imm());
      return;
    }

    int Imm = RVGPULoadFPImm::getLoadFPImm(
        APFloat(APFloat::IEEEdouble(), APInt(64, getFPConst())));
    Inst.addOperand(MCOperand::createImm(Imm));
  }

  void addCSRSystemRegisterOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createImm(SysReg.Encoding));
  }

  void addRlistOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createImm(Rlist.Val));
  }

  void addRegRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(RegReg.Reg1));
    Inst.addOperand(MCOperand::createReg(RegReg.Reg2));
  }

  void addSpimmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createImm(Spimm.Val));
  }

};
} // end anonymous namespace.

#define GET_REGISTER_MATCHER
#define GET_SUBTARGET_FEATURE_NAME
#define GET_MATCHER_IMPLEMENTATION
#define GET_MNEMONIC_SPELL_CHECKER
#include "RVGPUGenAsmMatcher.inc"

unsigned RVGPUAsmParser::validateTargetOperandClass(MCParsedAsmOperand &AsmOp,
                                                    unsigned Kind) {
  RVGPUOperand &Op = static_cast<RVGPUOperand &>(AsmOp);
  if (!Op.isReg())
    return Match_InvalidOperand;
#if 0
  MCRegister Reg = Op.getReg();
  bool IsRegFPR64 =
      RVGPUMCRegisterClasses[RVGPU::FPR64RegClassID].contains(Reg);
  bool IsRegFPR64C =
      RVGPUMCRegisterClasses[RVGPU::FPR64CRegClassID].contains(Reg);
  bool IsRegVR = RVGPUMCRegisterClasses[RVGPU::VRRegClassID].contains(Reg);

  // As the parser couldn't differentiate an FPR32 from an FPR64, coerce the
  // register from FPR64 to FPR32 or FPR64C to FPR32C if necessary.
  if ((IsRegFPR64 && Kind == MCK_FPR32) ||
      (IsRegFPR64C && Kind == MCK_FPR32C)) {
    Op.Reg.RegNum = convertFPR64ToFPR32(Reg);
    return Match_Success;
  }
  // As the parser couldn't differentiate an FPR16 from an FPR64, coerce the
  // register from FPR64 to FPR16 if necessary.
  if (IsRegFPR64 && Kind == MCK_FPR16) {
    Op.Reg.RegNum = convertFPR64ToFPR16(Reg);
    return Match_Success;
  }
  // As the parser couldn't differentiate an VRM2/VRM4/VRM8 from an VR, coerce
  // the register from VR to VRM2/VRM4/VRM8 if necessary.
  if (IsRegVR && (Kind == MCK_VRM2 || Kind == MCK_VRM4 || Kind == MCK_VRM8)) {
    Op.Reg.RegNum = convertVRToVRMx(*getContext().getRegisterInfo(), Reg, Kind);
    if (Op.Reg.RegNum == 0)
      return Match_InvalidOperand;
    return Match_Success;
  }
  return Match_InvalidOperand;
#endif 
  return Match_Success;
}

unsigned RVGPUAsmParser::checkTargetMatchPredicate(MCInst &Inst) {
  const MCInstrDesc &MCID = MII.get(Inst.getOpcode());

  for (unsigned I = 0; I < MCID.NumOperands; ++I) {
    if (MCID.operands()[I].RegClass == RVGPU::GPRPF64RegClassID) {
      const auto &Op = Inst.getOperand(I);
      assert(Op.isReg());

      MCRegister Reg = Op.getReg();
      if (((Reg.id() - RVGPU::X0) & 1) != 0)
        return Match_RequiresEvenGPRs;
    }
  }

  return Match_Success;
}

bool RVGPUAsmParser::generateImmOutOfRangeError(
    SMLoc ErrorLoc, int64_t Lower, int64_t Upper,
    const Twine &Msg = "immediate must be an integer in the range") {
  return Error(ErrorLoc, Msg + " [" + Twine(Lower) + ", " + Twine(Upper) + "]");
}

bool RVGPUAsmParser::generateImmOutOfRangeError(
    OperandVector &Operands, uint64_t ErrorInfo, int64_t Lower, int64_t Upper,
    const Twine &Msg = "immediate must be an integer in the range") {
  SMLoc ErrorLoc = ((RVGPUOperand &)*Operands[ErrorInfo]).getStartLoc();
  return generateImmOutOfRangeError(ErrorLoc, Lower, Upper, Msg);
}

bool RVGPUAsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                             OperandVector &Operands,
                                             MCStreamer &Out,
                                             uint64_t &ErrorInfo,
                                             bool MatchingInlineAsm) {
  MCInst Inst;
  FeatureBitset MissingFeatures;

  auto Result = MatchInstructionImpl(Operands, Inst, ErrorInfo, MissingFeatures,
                                     MatchingInlineAsm);
  switch (Result) {
  default:
    break;
  case Match_Success:
    if (validateInstruction(Inst, Operands))
      return true;
    return processInstruction(Inst, IDLoc, Operands, Out);
  case Match_MissingFeature: {
    assert(MissingFeatures.any() && "Unknown missing features!");
    bool FirstFeature = true;
    std::string Msg = "instruction requires the following:";
    for (unsigned i = 0, e = MissingFeatures.size(); i != e; ++i) {
      if (MissingFeatures[i]) {
        Msg += FirstFeature ? " " : ", ";
        Msg += getSubtargetFeatureName(i);
        FirstFeature = false;
      }
    }
    return Error(IDLoc, Msg);
  }
  case Match_MnemonicFail: {
    FeatureBitset FBS = ComputeAvailableFeatures(getSTI().getFeatureBits());
    std::string Suggestion = RVGPUMnemonicSpellCheck(
        ((RVGPUOperand &)*Operands[0]).getToken(), FBS, 0);
    return Error(IDLoc, "unrecognized instruction mnemonic" + Suggestion);
  }
  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0ULL) {
      if (ErrorInfo >= Operands.size())
        return Error(ErrorLoc, "too few operands for instruction");

      ErrorLoc = ((RVGPUOperand &)*Operands[ErrorInfo]).getStartLoc();
      if (ErrorLoc == SMLoc())
        ErrorLoc = IDLoc;
    }
    return Error(ErrorLoc, "invalid operand for instruction");
  }
  }

  // Handle the case when the error message is of specific type
  // other than the generic Match_InvalidOperand, and the
  // corresponding operand is missing.
  if (Result > FIRST_TARGET_MATCH_RESULT_TY) {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0ULL && ErrorInfo >= Operands.size())
      return Error(ErrorLoc, "too few operands for instruction");
  }

  switch (Result) {
  default:
    break;
  case Match_RequiresEvenGPRs:
    return Error(IDLoc,
                 "double precision floating point operands must use even "
                 "numbered X register");
  case Match_InvalidImmXLenLI:
      SMLoc ErrorLoc = ((RVGPUOperand &)*Operands[ErrorInfo]).getStartLoc();
      return Error(ErrorLoc, "operand must be a constant 64-bit integer");
  case Match_InvalidImmXLenLI_Restricted:
      SMLoc ErrorLoc = ((RVGPUOperand &)*Operands[ErrorInfo]).getStartLoc();
      return Error(ErrorLoc, "operand either must be a constant 64-bit integer "
                             "or a bare symbol name");
  case Match_InvalidImmZero: {
    SMLoc ErrorLoc = ((RVGPUOperand &)*Operands[ErrorInfo]).getStartLoc();
    return Error(ErrorLoc, "immediate must be zero");
  }
  case Match_InvalidUImmLog2XLen:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 6) - 1);
  case Match_InvalidUImmLog2XLenNonZero:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 1, (1 << 6) - 1);
  case Match_InvalidUImmLog2XLenHalf:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 5) - 1);
  case Match_InvalidUImm1:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 1) - 1);
  case Match_InvalidUImm2:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 2) - 1);
  case Match_InvalidUImm2Lsb0:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, 2,
                                      "immediate must be one of");
  case Match_InvalidUImm3:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 3) - 1);
  case Match_InvalidUImm4:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 4) - 1);
  case Match_InvalidUImm5:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 5) - 1);
  case Match_InvalidUImm6:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 6) - 1);
  case Match_InvalidUImm7:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 7) - 1);
  case Match_InvalidUImm8:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 8) - 1);
  case Match_InvalidUImm8GE32:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 32, (1 << 8) - 1);
  case Match_InvalidSImm5:
    return generateImmOutOfRangeError(Operands, ErrorInfo, -(1 << 4),
                                      (1 << 4) - 1);
  case Match_InvalidSImm6:
    return generateImmOutOfRangeError(Operands, ErrorInfo, -(1 << 5),
                                      (1 << 5) - 1);
  case Match_InvalidSImm6NonZero:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 5), (1 << 5) - 1,
        "immediate must be non-zero in the range");
  case Match_InvalidCLUIImm:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, 1, (1 << 5) - 1,
        "immediate must be in [0xfffe0, 0xfffff] or");
  case Match_InvalidUImm7Lsb00:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, 0, (1 << 7) - 4,
        "immediate must be a multiple of 4 bytes in the range");
  case Match_InvalidUImm8Lsb00:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, 0, (1 << 8) - 4,
        "immediate must be a multiple of 4 bytes in the range");
  case Match_InvalidUImm8Lsb000:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, 0, (1 << 8) - 8,
        "immediate must be a multiple of 8 bytes in the range");
  case Match_InvalidSImm9Lsb0:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 8), (1 << 8) - 2,
        "immediate must be a multiple of 2 bytes in the range");
  case Match_InvalidUImm9Lsb000:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, 0, (1 << 9) - 8,
        "immediate must be a multiple of 8 bytes in the range");
  case Match_InvalidUImm10Lsb00NonZero:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, 4, (1 << 10) - 4,
        "immediate must be a multiple of 4 bytes in the range");
  case Match_InvalidSImm10Lsb0000NonZero:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 9), (1 << 9) - 16,
        "immediate must be a multiple of 16 bytes and non-zero in the range");
  case Match_InvalidSImm12:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 11), (1 << 11) - 1,
        "operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an "
        "integer in the range");
  case Match_InvalidSImm12Lsb0:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 11), (1 << 11) - 2,
        "immediate must be a multiple of 2 bytes in the range");
  case Match_InvalidSImm12Lsb00000:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 11), (1 << 11) - 32,
        "immediate must be a multiple of 32 bytes in the range");
  case Match_InvalidSImm13Lsb0:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 12), (1 << 12) - 2,
        "immediate must be a multiple of 2 bytes in the range");
  case Match_InvalidUImm20LUI:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 20) - 1,
                                      "operand must be a symbol with "
                                      "%hi/%tprel_hi modifier or an integer in "
                                      "the range");
  case Match_InvalidUImm20:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 20) - 1);
  case Match_InvalidUImm20AUIPC:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, 0, (1 << 20) - 1,
        "operand must be a symbol with a "
        "%pcrel_hi/%got_pcrel_hi/%tls_ie_pcrel_hi/%tls_gd_pcrel_hi modifier or "
        "an integer in the range");
  case Match_InvalidSImm21Lsb0JAL:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 20), (1 << 20) - 2,
        "immediate must be a multiple of 2 bytes in the range");
  case Match_InvalidCSRSystemRegister: {
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 12) - 1,
                                      "operand must be a valid system register "
                                      "name or an integer in the range");
  }
  case Match_InvalidLoadFPImm: {
    SMLoc ErrorLoc = ((RVGPUOperand &)*Operands[ErrorInfo]).getStartLoc();
    return Error(ErrorLoc, "operand must be a valid floating-point constant");
  }
  case Match_InvalidBareSymbol: {
    SMLoc ErrorLoc = ((RVGPUOperand &)*Operands[ErrorInfo]).getStartLoc();
    return Error(ErrorLoc, "operand must be a bare symbol name");
  }
  case Match_InvalidPseudoJumpSymbol: {
    SMLoc ErrorLoc = ((RVGPUOperand &)*Operands[ErrorInfo]).getStartLoc();
    return Error(ErrorLoc, "operand must be a valid jump target");
  }
  case Match_InvalidCallSymbol: {
    SMLoc ErrorLoc = ((RVGPUOperand &)*Operands[ErrorInfo]).getStartLoc();
    return Error(ErrorLoc, "operand must be a bare symbol name");
  }
  case Match_InvalidTPRelAddSymbol: {
    SMLoc ErrorLoc = ((RVGPUOperand &)*Operands[ErrorInfo]).getStartLoc();
    return Error(ErrorLoc, "operand must be a symbol with %tprel_add modifier");
  }
  case Match_InvalidRTZArg: {
    SMLoc ErrorLoc = ((RVGPUOperand &)*Operands[ErrorInfo]).getStartLoc();
    return Error(ErrorLoc, "operand must be 'rtz' floating-point rounding mode");
  }
  case Match_InvalidVTypeI: {
    SMLoc ErrorLoc = ((RVGPUOperand &)*Operands[ErrorInfo]).getStartLoc();
    return generateVTypeError(ErrorLoc);
  }
  case Match_InvalidVMaskRegister: {
    SMLoc ErrorLoc = ((RVGPUOperand &)*Operands[ErrorInfo]).getStartLoc();
    return Error(ErrorLoc, "operand must be v0.t");
  }
  case Match_InvalidSImm5Plus1: {
    return generateImmOutOfRangeError(Operands, ErrorInfo, -(1 << 4) + 1,
                                      (1 << 4),
                                      "immediate must be in the range");
  }
  case Match_InvalidRlist: {
    SMLoc ErrorLoc = ((RVGPUOperand &)*Operands[ErrorInfo]).getStartLoc();
    return Error(
        ErrorLoc,
        "operand must be {ra [, s0[-sN]]} or {x1 [, x8[-x9][, x18[-xN]]]}");
  }
  case Match_InvalidSpimm: {
    SMLoc ErrorLoc = ((RVGPUOperand &)*Operands[ErrorInfo]).getStartLoc();
    return Error(
        ErrorLoc,
        "stack adjustment is invalid for this instruction and register list; "
        "refer to Zc spec for a detailed range of stack adjustment");
  }
  case Match_InvalidRnumArg: {
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, 10);
  }
  case Match_InvalidRegReg: {
    SMLoc ErrorLoc = ((RVGPUOperand &)*Operands[ErrorInfo]).getStartLoc();
    return Error(ErrorLoc, "operands must be register and register");
  }
  }

  llvm_unreachable("Unknown match type detected!");
}

// Attempts to match Name as a register (either using the default name or
// alternative ABI names), setting RegNo to the matching register. Upon
// failure, returns a non-valid MCRegister. If IsRVE, then registers x16-x31
// will be rejected.
static MCRegister matchRegisterNameHelper(bool IsRVE, StringRef Name) {
  MCRegister Reg = MatchRegisterName(Name);
  // The 16-/32- and 64-bit FPRs have the same asm name. Check that the initial
  // match always matches the 64-bit variant, and not the 16/32-bit one.
  if (!Reg)
    Reg = MatchRegisterAltName(Name);
  return Reg;
}

bool RVGPUAsmParser::parseRegister(MCRegister &Reg, SMLoc &StartLoc,
                                   SMLoc &EndLoc) {
  if (!tryParseRegister(Reg, StartLoc, EndLoc).isSuccess())
    return Error(StartLoc, "invalid register name");
  return false;
}

ParseStatus RVGPUAsmParser::tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
                                             SMLoc &EndLoc) {
  const AsmToken &Tok = getParser().getTok();
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();
  StringRef Name = getLexer().getTok().getIdentifier();

  Reg = matchRegisterNameHelper(isRVE(), Name);
  if (!Reg)
    return ParseStatus::NoMatch;

  getParser().Lex(); // Eat identifier token.
  return ParseStatus::Success;
}

ParseStatus RVGPUAsmParser::parseRegister(OperandVector &Operands,
                                          bool AllowParens) {
  SMLoc FirstS = getLoc();
  bool HadParens = false;
  AsmToken LParen;

  // If this is an LParen and a parenthesised register name is allowed, parse it
  // atomically.
  if (AllowParens && getLexer().is(AsmToken::LParen)) {
    AsmToken Buf[2];
    size_t ReadCount = getLexer().peekTokens(Buf);
    if (ReadCount == 2 && Buf[1].getKind() == AsmToken::RParen) {
      HadParens = true;
      LParen = getParser().getTok();
      getParser().Lex(); // Eat '('
    }
  }

  switch (getLexer().getKind()) {
  default:
    if (HadParens)
      getLexer().UnLex(LParen);
    return ParseStatus::NoMatch;
  case AsmToken::Identifier:
    StringRef Name = getLexer().getTok().getIdentifier();
    MCRegister RegNo = matchRegisterNameHelper(isRVE(), Name);

    if (!RegNo) {
      if (HadParens)
        getLexer().UnLex(LParen);
      return ParseStatus::NoMatch;
    }
    if (HadParens)
      Operands.push_back(RVGPUOperand::createToken("(", FirstS));
    SMLoc S = getLoc();
    SMLoc E = SMLoc::getFromPointer(S.getPointer() + Name.size());
    getLexer().Lex();
    Operands.push_back(RVGPUOperand::createReg(RegNo, S, E));
  }

  if (HadParens) {
    getParser().Lex(); // Eat ')'
    Operands.push_back(RVGPUOperand::createToken(")", getLoc()));
  }

  return ParseStatus::Success;
}

ParseStatus RVGPUAsmParser::parseFPImm(OperandVector &Operands) {
  SMLoc S = getLoc();

  // Parse special floats (inf/nan/min) representation.
  if (getTok().is(AsmToken::Identifier)) {
    StringRef Identifier = getTok().getIdentifier();
    if (Identifier.compare_insensitive("inf") == 0) {
      Operands.push_back(
          RVGPUOperand::createImm(MCConstantExpr::create(30, getContext()), S,
                                  getTok().getEndLoc(), isRV64()));
    } else if (Identifier.compare_insensitive("nan") == 0) {
      Operands.push_back(
          RVGPUOperand::createImm(MCConstantExpr::create(31, getContext()), S,
                                  getTok().getEndLoc(), isRV64()));
    } else if (Identifier.compare_insensitive("min") == 0) {
      Operands.push_back(
          RVGPUOperand::createImm(MCConstantExpr::create(1, getContext()), S,
                                  getTok().getEndLoc(), isRV64()));
    } else {
      return TokError("invalid floating point literal");
    }

    Lex(); // Eat the token.

    return ParseStatus::Success;
  }

  // Handle negation, as that still comes through as a separate token.
  bool IsNegative = parseOptionalToken(AsmToken::Minus);

  const AsmToken &Tok = getTok();
  if (!Tok.is(AsmToken::Real))
    return TokError("invalid floating point immediate");

  // Parse FP representation.
  APFloat RealVal(APFloat::IEEEdouble());
  auto StatusOrErr =
      RealVal.convertFromString(Tok.getString(), APFloat::rmTowardZero);
  if (errorToBool(StatusOrErr.takeError()))
    return TokError("invalid floating point representation");

  if (IsNegative)
    RealVal.changeSign();

  Operands.push_back(RVGPUOperand::createFPImm(
      RealVal.bitcastToAPInt().getZExtValue(), S));

  Lex(); // Eat the token.

  return ParseStatus::Success;
}

ParseStatus RVGPUAsmParser::parseImmediate(OperandVector &Operands) {
  SMLoc S = getLoc();
  SMLoc E;
  const MCExpr *Res;

  switch (getLexer().getKind()) {
  default:
    return ParseStatus::NoMatch;
  case AsmToken::LParen:
  case AsmToken::Dot:
  case AsmToken::Minus:
  case AsmToken::Plus:
  case AsmToken::Exclaim:
  case AsmToken::Tilde:
  case AsmToken::Integer:
  case AsmToken::String:
  case AsmToken::Identifier:
    if (getParser().parseExpression(Res, E))
      return ParseStatus::Failure;
    break;
  case AsmToken::Percent:
    return parseOperandWithModifier(Operands);
  }

  Operands.push_back(RVGPUOperand::createImm(Res, S, E, isRV64()));
  return ParseStatus::Success;
}

ParseStatus RVGPUAsmParser::parseOperandWithModifier(OperandVector &Operands) {
  SMLoc S = getLoc();
  SMLoc E;

  if (parseToken(AsmToken::Percent, "expected '%' for operand modifier"))
    return ParseStatus::Failure;

  if (getLexer().getKind() != AsmToken::Identifier)
    return Error(getLoc(), "expected valid identifier for operand modifier");
  StringRef Identifier = getParser().getTok().getIdentifier();
  RVGPUMCExpr::VariantKind VK = RVGPUMCExpr::getVariantKindForName(Identifier);
  if (VK == RVGPUMCExpr::VK_RVGPU_Invalid)
    return Error(getLoc(), "unrecognized operand modifier");

  getParser().Lex(); // Eat the identifier
  if (parseToken(AsmToken::LParen, "expected '('"))
    return ParseStatus::Failure;

  const MCExpr *SubExpr;
  if (getParser().parseParenExpression(SubExpr, E))
    return ParseStatus::Failure;

  const MCExpr *ModExpr = RVGPUMCExpr::create(SubExpr, VK, getContext());
  Operands.push_back(RVGPUOperand::createImm(ModExpr, S, E, isRV64()));
  return ParseStatus::Success;
}

ParseStatus RVGPUAsmParser::parseBareSymbol(OperandVector &Operands) {
  SMLoc S = getLoc();
  const MCExpr *Res;

  if (getLexer().getKind() != AsmToken::Identifier)
    return ParseStatus::NoMatch;

  StringRef Identifier;
  AsmToken Tok = getLexer().getTok();

  if (getParser().parseIdentifier(Identifier))
    return ParseStatus::Failure;

  SMLoc E = SMLoc::getFromPointer(S.getPointer() + Identifier.size());

  if (Identifier.consume_back("@plt"))
    return Error(getLoc(), "'@plt' operand not valid for instruction");

  MCSymbol *Sym = getContext().getOrCreateSymbol(Identifier);

  if (Sym->isVariable()) {
    const MCExpr *V = Sym->getVariableValue(/*SetUsed=*/false);
    if (!isa<MCSymbolRefExpr>(V)) {
      getLexer().UnLex(Tok); // Put back if it's not a bare symbol.
      return ParseStatus::NoMatch;
    }
    Res = V;
  } else
    Res = MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_None, getContext());

  MCBinaryExpr::Opcode Opcode;
  switch (getLexer().getKind()) {
  default:
    Operands.push_back(RVGPUOperand::createImm(Res, S, E, isRV64()));
    return ParseStatus::Success;
  case AsmToken::Plus:
    Opcode = MCBinaryExpr::Add;
    getLexer().Lex();
    break;
  case AsmToken::Minus:
    Opcode = MCBinaryExpr::Sub;
    getLexer().Lex();
    break;
  }

  const MCExpr *Expr;
  if (getParser().parseExpression(Expr, E))
    return ParseStatus::Failure;
  Res = MCBinaryExpr::create(Opcode, Res, Expr, getContext());
  Operands.push_back(RVGPUOperand::createImm(Res, S, E, isRV64()));
  return ParseStatus::Success;
}

ParseStatus RVGPUAsmParser::parseCallSymbol(OperandVector &Operands) {
  SMLoc S = getLoc();
  const MCExpr *Res;

  if (getLexer().getKind() != AsmToken::Identifier)
    return ParseStatus::NoMatch;

  // Avoid parsing the register in `call rd, foo` as a call symbol.
  if (getLexer().peekTok().getKind() != AsmToken::EndOfStatement)
    return ParseStatus::NoMatch;

  StringRef Identifier;
  if (getParser().parseIdentifier(Identifier))
    return ParseStatus::Failure;

  SMLoc E = SMLoc::getFromPointer(S.getPointer() + Identifier.size());

  RVGPUMCExpr::VariantKind Kind = RVGPUMCExpr::VK_RVGPU_CALL;
  if (Identifier.consume_back("@plt"))
    Kind = RVGPUMCExpr::VK_RVGPU_CALL_PLT;

  MCSymbol *Sym = getContext().getOrCreateSymbol(Identifier);
  Res = MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_None, getContext());
  Res = RVGPUMCExpr::create(Res, Kind, getContext());
  Operands.push_back(RVGPUOperand::createImm(Res, S, E, isRV64()));
  return ParseStatus::Success;
}

bool RVGPUAsmParser::generateVTypeError(SMLoc ErrorLoc) {
  return Error(
      ErrorLoc,
      "operand must be "
      "e[8|16|32|64|128|256|512|1024],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]");
}

ParseStatus RVGPUAsmParser::parseMemOpBaseReg(OperandVector &Operands) {
  if (parseToken(AsmToken::LParen, "expected '('"))
    return ParseStatus::Failure;
  Operands.push_back(RVGPUOperand::createToken("(", getLoc()));

  if (!parseRegister(Operands).isSuccess())
    return Error(getLoc(), "expected register");

  if (parseToken(AsmToken::RParen, "expected ')'"))
    return ParseStatus::Failure;
  Operands.push_back(RVGPUOperand::createToken(")", getLoc()));

  return ParseStatus::Success;
}

/// Looks at a token type and creates the relevant operand from this
/// information, adding to Operands. If operand was parsed, returns false, else
/// true.
bool RVGPUAsmParser::parseOperand(OperandVector &Operands, StringRef Mnemonic) {
  // Check if the current operand has a custom associated parser, if so, try to
  // custom parse the operand, or fallback to the general approach.
  ParseStatus Result =
      MatchOperandParserImpl(Operands, Mnemonic, /*ParseForAllFeatures=*/true);
  if (Result.isSuccess())
    return false;
  if (Result.isFailure())
    return true;

  // Attempt to parse token as a register.
  if (parseRegister(Operands, true).isSuccess())
    return false;

  // Attempt to parse token as an immediate
  if (parseImmediate(Operands).isSuccess()) {
    // Parse memory base register if present
    if (getLexer().is(AsmToken::LParen))
      return !parseMemOpBaseReg(Operands).isSuccess();
    return false;
  }

  // Finally we have exhausted all options and must declare defeat.
  Error(getLoc(), "unknown operand");
  return true;
}

bool RVGPUAsmParser::ParseInstruction(ParseInstructionInfo &Info,
                                      StringRef Name, SMLoc NameLoc,
                                      OperandVector &Operands) {
  // First operand is token for instruction
  Operands.push_back(RVGPUOperand::createToken(Name, NameLoc));

  // If there are no more operands, then finish
  if (getLexer().is(AsmToken::EndOfStatement)) {
    getParser().Lex(); // Consume the EndOfStatement.
    return false;
  }

  // Parse first operand
  if (parseOperand(Operands, Name))
    return true;

  // Parse until end of statement, consuming commas between operands
  while (parseOptionalToken(AsmToken::Comma)) {
    // Parse next operand
    if (parseOperand(Operands, Name))
      return true;
  }

  if (getParser().parseEOL("unexpected token")) {
    getParser().eatToEndOfStatement();
    return true;
  }
  return false;
}

bool RVGPUAsmParser::classifySymbolRef(const MCExpr *Expr,
                                       RVGPUMCExpr::VariantKind &Kind) {
  Kind = RVGPUMCExpr::VK_RVGPU_None;

  if (const RVGPUMCExpr *RE = dyn_cast<RVGPUMCExpr>(Expr)) {
    Kind = RE->getKind();
    Expr = RE->getSubExpr();
  }

  MCValue Res;
  MCFixup Fixup;
  if (Expr->evaluateAsRelocatable(Res, nullptr, &Fixup))
    return Res.getRefKind() == RVGPUMCExpr::VK_RVGPU_None;
  return false;
}

bool RVGPUAsmParser::isSymbolDiff(const MCExpr *Expr) {
  MCValue Res;
  MCFixup Fixup;
  if (Expr->evaluateAsRelocatable(Res, nullptr, &Fixup)) {
    return Res.getRefKind() == RVGPUMCExpr::VK_RVGPU_None && Res.getSymA() &&
           Res.getSymB();
  }
  return false;
}

ParseStatus RVGPUAsmParser::parseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getString();

  return ParseStatus::NoMatch;
}

void RVGPUAsmParser::emitToStreamer(MCStreamer &S, const MCInst &Inst) {
  S.emitInstruction(Inst, getSTI());
}

bool RVGPUAsmParser::validateInstruction(MCInst &Inst,
                                         OperandVector &Operands) {
  unsigned Opcode = Inst.getOpcode();
  return false;
}

bool RVGPUAsmParser::processInstruction(MCInst &Inst, SMLoc IDLoc,
                                        OperandVector &Operands,
                                        MCStreamer &Out) {
  Inst.setLoc(IDLoc);

  emitToStreamer(Out, Inst);
  return false;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeRVGPUAsmParser() {
  RegisterMCAsmParser<RVGPUAsmParser> X(getTheRVGPUTarget64());
}
