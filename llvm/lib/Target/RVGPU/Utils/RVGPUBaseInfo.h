//===- RVGPUBaseInfo.h - Top level definitions for RVGPU ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RVGPU_UTILS_RVGPUBASEINFO_H
#define LLVM_LIB_TARGET_RVGPU_UTILS_RVGPUBASEINFO_H

#include "RVDefines.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Alignment.h"
#include <array>
#include <functional>
#include <utility>

struct rv_kernel_code_t;

namespace llvm {

struct Align;
class Argument;
class Function;
class GlobalValue;
class MCInstrInfo;
class MCRegisterClass;
class MCRegisterInfo;
class MCSubtargetInfo;
class StringRef;
class Triple;
class raw_ostream;

namespace rvhsa {
struct kernel_descriptor_t;
}

namespace RVGPU {

struct IsaVersion;

enum { RVHSA_COV5 = 5 };

/// \returns The offset of the multigrid_sync_arg argument from implicitarg_ptr
unsigned getMultigridSyncArgImplicitArgPosition(unsigned COV);

/// \returns The offset of the hostcall pointer argument from implicitarg_ptr
unsigned getHostcallImplicitArgPosition(unsigned COV);

unsigned getDefaultQueueImplicitArgPosition(unsigned COV);
unsigned getCompletionActionImplicitArgPosition(unsigned COV);

struct GcnBufferFormatInfo {
  unsigned Format;
  unsigned BitsPerComp;
  unsigned NumComponents;
  unsigned NumFormat;
  unsigned DataFormat;
};

struct MAIInstInfo {
  uint16_t Opcode;
  bool is_dgemm;
  bool is_gfx940_xdl;
};

namespace IsaInfo {

enum {
  // The closed Vulkan driver sets 96, which limits the wave count to 8 but
  // doesn't spill SGPRs as much as when 80 is set.
  FIXED_NUM_SGPRS_FOR_INIT_BUG = 96,
  TRAP_NUM_SGPRS = 16
};

enum class TargetIDSetting {
  Unsupported,
  Any,
  Off,
  On
};

class RVGPUTargetID {
private:
  const MCSubtargetInfo &STI;
  TargetIDSetting XnackSetting;
  TargetIDSetting SramEccSetting;
  unsigned CodeObjectVersion;

public:
  explicit RVGPUTargetID(const MCSubtargetInfo &STI);
  ~RVGPUTargetID() = default;

  void setCodeObjectVersion(unsigned COV) {
    CodeObjectVersion = COV;
  }
  /// \returns String representation of an object.
  std::string toString() const;
};

/// \returns Wavefront size for given subtarget \p STI.
unsigned getWavefrontSize(const MCSubtargetInfo *STI);

/// \returns Local memory size in bytes for given subtarget \p STI.
unsigned getLocalMemorySize(const MCSubtargetInfo *STI);

/// \returns Maximum addressable local memory size in bytes for given subtarget
/// \p STI.
unsigned getAddressableLocalMemorySize(const MCSubtargetInfo *STI);

/// \returns Number of execution units per compute unit for given subtarget \p
/// STI.
unsigned getEUsPerCU(const MCSubtargetInfo *STI);

/// \returns Maximum number of work groups per compute unit for given subtarget
/// \p STI and limited by given \p FlatWorkGroupSize.
unsigned getMaxWorkGroupsPerCU(const MCSubtargetInfo *STI,
                               unsigned FlatWorkGroupSize);

/// \returns Minimum number of waves per execution unit for given subtarget \p
/// STI.
unsigned getMinWavesPerEU(const MCSubtargetInfo *STI);

/// \returns Maximum number of waves per execution unit for given subtarget \p
/// STI without any kind of limitation.
unsigned getMaxWavesPerEU(const MCSubtargetInfo *STI);

/// \returns Number of waves per execution unit required to support the given \p
/// FlatWorkGroupSize.
unsigned getWavesPerEUForWorkGroup(const MCSubtargetInfo *STI,
                                   unsigned FlatWorkGroupSize);

/// \returns Minimum flat work group size for given subtarget \p STI.
unsigned getMinFlatWorkGroupSize(const MCSubtargetInfo *STI);

/// \returns Maximum flat work group size for given subtarget \p STI.
unsigned getMaxFlatWorkGroupSize(const MCSubtargetInfo *STI);

/// \returns Number of waves per work group for given subtarget \p STI and
/// \p FlatWorkGroupSize.
unsigned getWavesPerWorkGroup(const MCSubtargetInfo *STI,
                              unsigned FlatWorkGroupSize);

/// \returns VGPR allocation granularity for given subtarget \p STI.
///
/// For subtargets which support it, \p EnableWavefrontSize32 should match
/// the ENABLE_WAVEFRONT_SIZE32 kernel descriptor field.
unsigned
getVGPRAllocGranule(const MCSubtargetInfo *STI,
                    std::optional<bool> EnableWavefrontSize32 = std::nullopt);

/// \returns VGPR encoding granularity for given subtarget \p STI.
///
/// For subtargets which support it, \p EnableWavefrontSize32 should match
/// the ENABLE_WAVEFRONT_SIZE32 kernel descriptor field.
unsigned getVGPREncodingGranule(
    const MCSubtargetInfo *STI,
    std::optional<bool> EnableWavefrontSize32 = std::nullopt);

/// \returns Total number of VGPRs for given subtarget \p STI.
unsigned getTotalNumVGPRs(const MCSubtargetInfo *STI);

/// \returns Addressable number of VGPRs for given subtarget \p STI.
unsigned getAddressableNumVGPRs(const MCSubtargetInfo *STI);

/// \returns Minimum number of VGPRs that meets given number of waves per
/// execution unit requirement for given subtarget \p STI.
unsigned getMinNumVGPRs(const MCSubtargetInfo *STI, unsigned WavesPerEU);

/// \returns Maximum number of VGPRs that meets given number of waves per
/// execution unit requirement for given subtarget \p STI.
unsigned getMaxNumVGPRs(const MCSubtargetInfo *STI, unsigned WavesPerEU);

/// \returns Number of waves reachable for a given \p NumVGPRs usage for given
/// subtarget \p STI.
unsigned getNumWavesPerEUWithNumVGPRs(const MCSubtargetInfo *STI,
                                      unsigned NumVGPRs);

/// \returns Number of VGPR blocks needed for given subtarget \p STI when
/// \p NumVGPRs are used.
///
/// For subtargets which support it, \p EnableWavefrontSize32 should match the
/// ENABLE_WAVEFRONT_SIZE32 kernel descriptor field.
unsigned
getNumVGPRBlocks(const MCSubtargetInfo *STI, unsigned NumSGPRs,
                 std::optional<bool> EnableWavefrontSize32 = std::nullopt);

} // end namespace IsaInfo

LLVM_READONLY
int16_t getNamedOperandIdx(uint16_t Opcode, uint16_t NamedIdx);

LLVM_READONLY
inline bool hasNamedOperand(uint64_t Opcode, uint64_t NamedIdx) {
  return getNamedOperandIdx(Opcode, NamedIdx) != -1;
}


void initDefaultRVKernelCodeT(rv_kernel_code_t &Header,
                               const MCSubtargetInfo *STI);

rvhsa::kernel_descriptor_t getDefaultRvhsaKernelDescriptor(
    const MCSubtargetInfo *STI);

bool isGroupSegment(const GlobalValue *GV);
bool isGlobalSegment(const GlobalValue *GV);
bool isReadOnlySegment(const GlobalValue *GV);

/// \returns True if constants should be emitted to .text section for given
/// target triple \p TT, false otherwise.
bool shouldEmitConstantsToTextSection(const Triple &TT);

/// \returns Integer value requested using \p F's \p Name attribute.
///
/// \returns \p Default if attribute is not present.
///
/// \returns \p Default and emits error if requested value cannot be converted
/// to integer.
int getIntegerAttribute(const Function &F, StringRef Name, int Default);

/// \returns A pair of integer values requested using \p F's \p Name attribute
/// in "first[,second]" format ("second" is optional unless \p OnlyFirstRequired
/// is false).
///
/// \returns \p Default if attribute is not present.
///
/// \returns \p Default and emits error if one of the requested values cannot be
/// converted to integer, or \p OnlyFirstRequired is false and "second" value is
/// not present.
std::pair<unsigned, unsigned>
getIntegerPairAttribute(const Function &F, StringRef Name,
                        std::pair<unsigned, unsigned> Default,
                        bool OnlyFirstRequired = false);

/// Represents the counter values to wait for in an s_waitcnt instruction.
///
/// Large values (including the maximum possible integer) can be used to
/// represent "don't care" waits.
struct Waitcnt {
  unsigned VmCnt = ~0u;
  unsigned ExpCnt = ~0u;
  unsigned LgkmCnt = ~0u;
  unsigned VsCnt = ~0u;

  Waitcnt() = default;
  Waitcnt(unsigned VmCnt, unsigned ExpCnt, unsigned LgkmCnt, unsigned VsCnt)
      : VmCnt(VmCnt), ExpCnt(ExpCnt), LgkmCnt(LgkmCnt), VsCnt(VsCnt) {}

  static Waitcnt allZero(bool HasVscnt) {
    return Waitcnt(0, 0, 0, HasVscnt ? 0 : ~0u);
  }
  static Waitcnt allZeroExceptVsCnt() { return Waitcnt(0, 0, 0, ~0u); }

  bool hasWait() const {
    return VmCnt != ~0u || ExpCnt != ~0u || LgkmCnt != ~0u || VsCnt != ~0u;
  }

  bool hasWaitExceptVsCnt() const {
    return VmCnt != ~0u || ExpCnt != ~0u || LgkmCnt != ~0u;
  }

  bool hasWaitVsCnt() const {
    return VsCnt != ~0u;
  }

  Waitcnt combined(const Waitcnt &Other) const {
    return Waitcnt(std::min(VmCnt, Other.VmCnt), std::min(ExpCnt, Other.ExpCnt),
                   std::min(LgkmCnt, Other.LgkmCnt),
                   std::min(VsCnt, Other.VsCnt));
  }
};


namespace Hwreg {

LLVM_READONLY
int64_t getHwregId(const StringRef Name, const MCSubtargetInfo &STI);

LLVM_READNONE
bool isValidHwreg(int64_t Id);

LLVM_READNONE
bool isValidHwregOffset(int64_t Offset);

LLVM_READNONE
bool isValidHwregWidth(int64_t Width);

LLVM_READNONE
uint64_t encodeHwreg(uint64_t Id, uint64_t Offset, uint64_t Width);

LLVM_READNONE
StringRef getHwreg(unsigned Id, const MCSubtargetInfo &STI);

void decodeHwreg(unsigned Val, unsigned &Id, unsigned &Offset, unsigned &Width);

} // namespace Hwreg

namespace DepCtr {

int getDefaultDepCtrEncoding(const MCSubtargetInfo &STI);
int encodeDepCtr(const StringRef Name, int64_t Val, unsigned &UsedOprMask,
                 const MCSubtargetInfo &STI);
bool isSymbolicDepCtrEncoding(unsigned Code, bool &HasNonDefaultVal,
                              const MCSubtargetInfo &STI);
bool decodeDepCtr(unsigned Code, int &Id, StringRef &Name, unsigned &Val,
                  bool &IsDefault, const MCSubtargetInfo &STI);

/// \returns Decoded VaVdst from given immediate \p Encoded.
unsigned decodeFieldVaVdst(unsigned Encoded);

/// \returns Decoded VmVsrc from given immediate \p Encoded.
unsigned decodeFieldVmVsrc(unsigned Encoded);

/// \returns Decoded SaSdst from given immediate \p Encoded.
unsigned decodeFieldSaSdst(unsigned Encoded);

/// \returns \p VmVsrc as an encoded Depctr immediate.
unsigned encodeFieldVmVsrc(unsigned VmVsrc);

/// \returns \p Encoded combined with encoded \p VmVsrc.
unsigned encodeFieldVmVsrc(unsigned Encoded, unsigned VmVsrc);

/// \returns \p VaVdst as an encoded Depctr immediate.
unsigned encodeFieldVaVdst(unsigned VaVdst);

/// \returns \p Encoded combined with encoded \p VaVdst.
unsigned encodeFieldVaVdst(unsigned Encoded, unsigned VaVdst);

/// \returns \p SaSdst as an encoded Depctr immediate.
unsigned encodeFieldSaSdst(unsigned SaSdst);

/// \returns \p Encoded combined with encoded \p SaSdst.
unsigned encodeFieldSaSdst(unsigned Encoded, unsigned SaSdst);

} // namespace DepCtr

namespace Exp {

bool getTgtName(unsigned Id, StringRef &Name, int &Index);

LLVM_READONLY
unsigned getTgtId(const StringRef Name);

LLVM_READNONE
bool isSupportedTgtId(unsigned Id, const MCSubtargetInfo &STI);

} // namespace Exp

namespace MTBUFFormat {

LLVM_READNONE
int64_t encodeDfmtNfmt(unsigned Dfmt, unsigned Nfmt);

void decodeDfmtNfmt(unsigned Format, unsigned &Dfmt, unsigned &Nfmt);

int64_t getDfmt(const StringRef Name);

StringRef getDfmtName(unsigned Id);

int64_t getNfmt(const StringRef Name, const MCSubtargetInfo &STI);

StringRef getNfmtName(unsigned Id, const MCSubtargetInfo &STI);

bool isValidDfmtNfmt(unsigned Val, const MCSubtargetInfo &STI);

bool isValidNfmt(unsigned Val, const MCSubtargetInfo &STI);

int64_t getUnifiedFormat(const StringRef Name, const MCSubtargetInfo &STI);

StringRef getUnifiedFormatName(unsigned Id, const MCSubtargetInfo &STI);

bool isValidUnifiedFormat(unsigned Val, const MCSubtargetInfo &STI);

int64_t convertDfmtNfmt2Ufmt(unsigned Dfmt, unsigned Nfmt,
                             const MCSubtargetInfo &STI);

bool isValidFormatEncoding(unsigned Val, const MCSubtargetInfo &STI);

unsigned getDefaultFormatEncoding(const MCSubtargetInfo &STI);

} // namespace MTBUFFormat

namespace SendMsg {

LLVM_READONLY
int64_t getMsgId(const StringRef Name, const MCSubtargetInfo &STI);

LLVM_READONLY
int64_t getMsgOpId(int64_t MsgId, const StringRef Name);

LLVM_READNONE
StringRef getMsgName(int64_t MsgId, const MCSubtargetInfo &STI);

LLVM_READNONE
StringRef getMsgOpName(int64_t MsgId, int64_t OpId, const MCSubtargetInfo &STI);

LLVM_READNONE
bool isValidMsgId(int64_t MsgId, const MCSubtargetInfo &STI);

LLVM_READNONE
bool isValidMsgOp(int64_t MsgId, int64_t OpId, const MCSubtargetInfo &STI,
                  bool Strict = true);

LLVM_READNONE
bool isValidMsgStream(int64_t MsgId, int64_t OpId, int64_t StreamId,
                      const MCSubtargetInfo &STI, bool Strict = true);

LLVM_READNONE
bool msgRequiresOp(int64_t MsgId, const MCSubtargetInfo &STI);

LLVM_READNONE
bool msgSupportsStream(int64_t MsgId, int64_t OpId, const MCSubtargetInfo &STI);

void decodeMsg(unsigned Val, uint16_t &MsgId, uint16_t &OpId,
               uint16_t &StreamId, const MCSubtargetInfo &STI);

LLVM_READNONE
uint64_t encodeMsg(uint64_t MsgId,
                   uint64_t OpId,
                   uint64_t StreamId);

} // namespace SendMsg


unsigned getInitialPSInputAddr(const Function &F);

bool getHasColorExport(const Function &F);

bool getHasDepthExport(const Function &F);

LLVM_READNONE
bool isShader(CallingConv::ID CC);

LLVM_READNONE
bool isGraphics(CallingConv::ID CC);

LLVM_READNONE
bool isCompute(CallingConv::ID CC);

LLVM_READNONE
bool isEntryFunctionCC(CallingConv::ID CC);

// These functions are considered entrypoints into the current module, i.e. they
// are allowed to be called from outside the current module. This is different
// from isEntryFunctionCC, which is only true for functions that are entered by
// the hardware. Module entry points include all entry functions but also
// include functions that can be called from other functions inside or outside
// the current module. Module entry functions are allowed to allocate LDS.
LLVM_READNONE
bool isModuleEntryFunctionCC(CallingConv::ID CC);

LLVM_READNONE
bool isChainCC(CallingConv::ID CC);

bool isKernelCC(const Function *Func);

// FIXME: Remove this when calling conventions cleaned up
LLVM_READNONE
inline bool isKernel(CallingConv::ID CC) {
  switch (CC) {
  case CallingConv::RVGPU_KERNEL:
  case CallingConv::SPIR_KERNEL:
    return true;
  default:
    return false;
  }
}

bool hasMIMG_R128(const MCSubtargetInfo &STI);
bool hasGDS(const MCSubtargetInfo &STI);
unsigned getNSAMaxSize(const MCSubtargetInfo &STI, bool HasSampler = false);
unsigned getMaxNumUserSGPRs(const MCSubtargetInfo &STI);

bool isGFX11(const MCSubtargetInfo &STI);
bool hasArchitectedFlatScratch(const MCSubtargetInfo &STI);
bool hasMAIInsts(const MCSubtargetInfo &STI);
int getTotalNumVGPRs(bool has90AInsts, int32_t ArgNumAGPR, int32_t ArgNumVGPR);
unsigned hasKernargPreload(const MCSubtargetInfo &STI);

/// \returns if \p Reg occupies the high 16-bits of a 32-bit register.
/// The bit indicating isHi is the LSB of the encoding.
bool isHi(unsigned Reg, const MCRegisterInfo &MRI);

/// If \p Reg is a pseudo reg, return the correct hardware register given
/// \p STI otherwise return \p Reg.
unsigned getMCReg(unsigned Reg, const MCSubtargetInfo &STI);

/// Convert hardware register \p Reg to a pseudo register
LLVM_READNONE
unsigned mc2PseudoReg(unsigned Reg);

LLVM_READNONE
bool isInlineValue(unsigned Reg);

/// Is this an RVGPU specific source operand? These include registers,
/// inline constants, literals and mandatory literals (KImm).
bool isSISrcOperand(const MCInstrDesc &Desc, unsigned OpNo);

/// Is this a KImm operand?
bool isKImmOperand(const MCInstrDesc &Desc, unsigned OpNo);

/// Is this floating-point operand?
bool isSISrcFPOperand(const MCInstrDesc &Desc, unsigned OpNo);

/// Does this operand support only inlinable literals?
bool isSISrcInlinableOperand(const MCInstrDesc &Desc, unsigned OpNo);

/// Get the size in bits of a register from the register class \p RC.
unsigned getRegBitWidth(unsigned RCID);

/// Get the size in bits of a register from the register class \p RC.
unsigned getRegBitWidth(const MCRegisterClass &RC);

/// Get size of register operand
unsigned getRegOperandSize(const MCRegisterInfo *MRI, const MCInstrDesc &Desc,
                           unsigned OpNo);

LLVM_READNONE
inline unsigned getOperandSize(const MCOperandInfo &OpInfo) {
#if 0
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
  case RVGPU::OPERAND_KIMM32:
  case RVGPU::OPERAND_KIMM16: // mandatory literal is always size 4
  case RVGPU::OPERAND_INLINE_SPLIT_BARRIER_INT32:
    return 4;

  case RVGPU::OPERAND_REG_IMM_INT64:
  case RVGPU::OPERAND_REG_IMM_FP64:
  case RVGPU::OPERAND_REG_INLINE_C_INT64:
  case RVGPU::OPERAND_REG_INLINE_C_FP64:
  case RVGPU::OPERAND_REG_INLINE_AC_FP64:
    return 8;

  case RVGPU::OPERAND_REG_IMM_INT16:
  case RVGPU::OPERAND_REG_IMM_FP16:
  case RVGPU::OPERAND_REG_IMM_FP16_DEFERRED:
  case RVGPU::OPERAND_REG_INLINE_C_INT16:
  case RVGPU::OPERAND_REG_INLINE_C_FP16:
  case RVGPU::OPERAND_REG_INLINE_C_V2INT16:
  case RVGPU::OPERAND_REG_INLINE_C_V2FP16:
  case RVGPU::OPERAND_REG_INLINE_AC_INT16:
  case RVGPU::OPERAND_REG_INLINE_AC_FP16:
  case RVGPU::OPERAND_REG_INLINE_AC_V2INT16:
  case RVGPU::OPERAND_REG_INLINE_AC_V2FP16:
  case RVGPU::OPERAND_REG_IMM_V2INT16:
  case RVGPU::OPERAND_REG_IMM_V2FP16:
    return 2;
  default:
    llvm_unreachable("unhandled operand type");
  }
#endif 
    llvm_unreachable("unhandled operand type");
    return 0;
}

LLVM_READNONE
inline unsigned getOperandSize(const MCInstrDesc &Desc, unsigned OpNo) {
  return getOperandSize(Desc.operands()[OpNo]);
}

/// Is this literal inlinable, and not one of the values intended for floating
/// point values.
LLVM_READNONE
inline bool isInlinableIntLiteral(int64_t Literal) {
  return Literal >= -16 && Literal <= 64;
}

/// Is this literal inlinable
LLVM_READNONE
bool isInlinableLiteral64(int64_t Literal, bool HasInv2Pi);

LLVM_READNONE
bool isInlinableLiteral32(int32_t Literal, bool HasInv2Pi);

LLVM_READNONE
bool isInlinableLiteral16(int16_t Literal, bool HasInv2Pi);

LLVM_READNONE
bool isInlinableLiteralV216(int32_t Literal, bool HasInv2Pi);

LLVM_READNONE
bool isInlinableIntLiteralV216(int32_t Literal);

LLVM_READNONE
bool isInlinableLiteralV216(int32_t Literal, bool HasInv2Pi, uint8_t OpType);

LLVM_READNONE
bool isFoldableLiteralV216(int32_t Literal, bool HasInv2Pi);

LLVM_READNONE
bool isValid32BitLiteral(uint64_t Val, bool IsFP64);

bool isArgPassedInSGPR(const Argument *Arg);

bool isArgPassedInSGPR(const CallBase *CB, unsigned ArgNo);

LLVM_READONLY
bool isLegalSMRDEncodedUnsignedOffset(const MCSubtargetInfo &ST,
                                      int64_t EncodedOffset);

LLVM_READONLY
bool isLegalSMRDEncodedSignedOffset(const MCSubtargetInfo &ST,
                                    int64_t EncodedOffset,
                                    bool IsBuffer);

/// Convert \p ByteOffset to dwords if the subtarget uses dword SMRD immediate
/// offsets.
uint64_t convertSMRDOffsetUnits(const MCSubtargetInfo &ST, uint64_t ByteOffset);

/// \returns The encoding that will be used for \p ByteOffset in the
/// SMRD offset field, or std::nullopt if it won't fit. On GFX9 and GFX10
/// S_LOAD instructions have a signed offset, on other subtargets it is
/// unsigned. S_BUFFER has an unsigned offset for all subtargets.
std::optional<int64_t> getSMRDEncodedOffset(const MCSubtargetInfo &ST,
                                            int64_t ByteOffset, bool IsBuffer);

/// \return The encoding that can be used for a 32-bit literal offset in an SMRD
/// instruction. This is only useful on CI.s
std::optional<int64_t> getSMRDEncodedLiteralOffset32(const MCSubtargetInfo &ST,
                                                     int64_t ByteOffset);

/// For pre-GFX12 FLAT instructions the offset must be positive;
/// MSB is ignored and forced to zero.
///
/// \return The number of bits available for the signed offset field in flat
/// instructions. Note that some forms of the instruction disallow negative
/// offsets.
unsigned getNumFlatOffsetBits(const MCSubtargetInfo &ST);

/// \returns true if this offset is small enough to fit in the SMRD
/// offset field.  \p ByteOffset should be the offset in bytes and
/// not the encoded offset.
bool isLegalSMRDImmOffset(const MCSubtargetInfo &ST, int64_t ByteOffset);

LLVM_READNONE
inline bool isLegalDPALU_DPPControl(unsigned DC) {
  return DC >= DPP::ROW_NEWBCAST_FIRST && DC <= DPP::ROW_NEWBCAST_LAST;
}

/// \returns true if an instruction may have a 64-bit VGPR operand.
bool hasAny64BitVGPROperands(const MCInstrDesc &OpDesc);

/// \returns true if an instruction is a DP ALU DPP.
bool isDPALU_DPP(const MCInstrDesc &OpDesc);

/// \returns true if the intrinsic is divergent
bool isIntrinsicSourceOfDivergence(unsigned IntrID);

/// \returns true if the intrinsic is uniform
bool isIntrinsicAlwaysUniform(unsigned IntrID);

} // end namespace RVGPU

raw_ostream &operator<<(raw_ostream &OS,
                        const RVGPU::IsaInfo::TargetIDSetting S);

} // end namespace llvm

#endif // LLVM_LIB_TARGET_RVGPU_UTILS_RVGPUBASEINFO_H
