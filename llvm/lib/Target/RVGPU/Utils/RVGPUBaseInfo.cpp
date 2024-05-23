//===- RVGPUBaseInfo.cpp - RVGPU Base encoding information --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RVGPUBaseInfo.h"
#include "RVGPU.h"
//#include "RVGPUAsmUtils.h"
#include "RVKernelCodeT.h"
#include "MCTargetDesc/RVGPUMCTargetDesc.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IntrinsicsRVGPU.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/RVHSAKernelDescriptor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/TargetParser.h"
#include <optional>

#define GET_INSTRINFO_NAMED_OPS
#define GET_INSTRMAP_INFO
#include "RVGPUGenInstrInfo.inc"

namespace {

/// \returns Bit mask for given bit \p Shift and bit \p Width.
unsigned getBitMask(unsigned Shift, unsigned Width) {
  return ((1 << Width) - 1) << Shift;
}

/// Packs \p Src into \p Dst for given bit \p Shift and bit \p Width.
///
/// \returns Packed \p Dst.
unsigned packBits(unsigned Src, unsigned Dst, unsigned Shift, unsigned Width) {
  unsigned Mask = getBitMask(Shift, Width);
  return ((Src << Shift) & Mask) | (Dst & ~Mask);
}

/// Unpacks bits from \p Src for given bit \p Shift and bit \p Width.
///
/// \returns Unpacked bits.
unsigned unpackBits(unsigned Src, unsigned Shift, unsigned Width) {
  return (Src & getBitMask(Shift, Width)) >> Shift;
}
} // end namespace anonymous

namespace llvm {

namespace RVGPU {

unsigned getMultigridSyncArgImplicitArgPosition(unsigned CodeObjectVersion) {
  return RVGPU::ImplicitArg::MULTIGRID_SYNC_ARG_OFFSET;
}


// FIXME: All such magic numbers about the ABI should be in a
// central TD file.
unsigned getHostcallImplicitArgPosition(unsigned CodeObjectVersion) {
  return RVGPU::ImplicitArg::HOSTCALL_PTR_OFFSET;
}

unsigned getDefaultQueueImplicitArgPosition(unsigned CodeObjectVersion) {
  return RVGPU::ImplicitArg::DEFAULT_QUEUE_OFFSET;
}

unsigned getCompletionActionImplicitArgPosition(unsigned CodeObjectVersion) {
  return RVGPU::ImplicitArg::COMPLETION_ACTION_OFFSET;
}

namespace IsaInfo {

RVGPUTargetID::RVGPUTargetID(const MCSubtargetInfo &STI)
    : STI(STI), CodeObjectVersion(0) {
}

static TargetIDSetting
getTargetIDSettingFromFeatureString(StringRef FeatureString) {
  if (FeatureString.ends_with("-"))
    return TargetIDSetting::Off;
  if (FeatureString.ends_with("+"))
    return TargetIDSetting::On;

  llvm_unreachable("Malformed feature string");
}


std::string RVGPUTargetID::toString() const {
  std::string StringRep;
  raw_string_ostream StreamRep(StringRep);

  auto TargetTriple = STI.getTargetTriple();

  StreamRep << TargetTriple.getArchName() << '-'
            << TargetTriple.getVendorName() << '-'
            << TargetTriple.getOSName() << '-'
            << TargetTriple.getEnvironmentName() << '-';

  StreamRep.flush();
  return StringRep;
}

unsigned getWavefrontSize(const MCSubtargetInfo *STI) {
    return 32;
}

unsigned getLocalMemorySize(const MCSubtargetInfo *STI) {
  unsigned BytesPerCU = 65536;

  return BytesPerCU;
}

unsigned getAddressableLocalMemorySize(const MCSubtargetInfo *STI) {
    return 65536;
}

unsigned getEUsPerCU(const MCSubtargetInfo *STI) {
    return 2;
}

unsigned getMaxWorkGroupsPerCU(const MCSubtargetInfo *STI,
                               unsigned FlatWorkGroupSize) {
  unsigned MaxWaves = getMaxWavesPerEU(STI) * getEUsPerCU(STI);
  unsigned N = getWavesPerWorkGroup(STI, FlatWorkGroupSize);
  if (N == 1) {
    // Single-wave workgroups don't consume barrier resources.
    return MaxWaves;
  }

  unsigned MaxBarriers = 32;

  return std::min(MaxWaves / N, MaxBarriers);
}

unsigned getMinWavesPerEU(const MCSubtargetInfo *STI) {
  return 1;
}

unsigned getMaxWavesPerEU(const MCSubtargetInfo *STI) {
    return 16;
}

unsigned getWavesPerEUForWorkGroup(const MCSubtargetInfo *STI,
                                   unsigned FlatWorkGroupSize) {
  return divideCeil(getWavesPerWorkGroup(STI, FlatWorkGroupSize),
                    getEUsPerCU(STI));
}

unsigned getMinFlatWorkGroupSize(const MCSubtargetInfo *STI) {
  return 1;
}

unsigned getMaxFlatWorkGroupSize(const MCSubtargetInfo *STI) {
  // Some subtargets allow encoding 2048, but this isn't tested or supported.
  return 1024;
}

unsigned getWavesPerWorkGroup(const MCSubtargetInfo *STI,
                              unsigned FlatWorkGroupSize) {
  return divideCeil(FlatWorkGroupSize, getWavefrontSize(STI));
}


unsigned getVGPRAllocGranule(const MCSubtargetInfo *STI,
                             std::optional<bool> EnableWavefrontSize32) {
  return 24;
}

unsigned getVGPREncodingGranule(const MCSubtargetInfo *STI,
                                std::optional<bool> EnableWavefrontSize32) {
  return 8;
}

unsigned getTotalNumVGPRs(const MCSubtargetInfo *STI) {
    return 1536;
}

unsigned getAddressableNumVGPRs(const MCSubtargetInfo *STI) {
  return 256;
}

unsigned getNumWavesPerEUWithNumVGPRs(const MCSubtargetInfo *STI,
                                      unsigned NumVGPRs) {
  unsigned MaxWaves = getMaxWavesPerEU(STI);
  unsigned Granule = getVGPRAllocGranule(STI);
  if (NumVGPRs < Granule)
    return MaxWaves;
  unsigned RoundedRegs = alignTo(NumVGPRs, Granule);
  return std::min(std::max(getTotalNumVGPRs(STI) / RoundedRegs, 1u), MaxWaves);
}

unsigned getMinNumVGPRs(const MCSubtargetInfo *STI, unsigned WavesPerEU) {
  assert(WavesPerEU != 0);

  unsigned MaxWavesPerEU = getMaxWavesPerEU(STI);
  if (WavesPerEU >= MaxWavesPerEU)
    return 0;

  unsigned TotNumVGPRs = getTotalNumVGPRs(STI);
  unsigned AddrsableNumVGPRs = getAddressableNumVGPRs(STI);
  unsigned Granule = getVGPRAllocGranule(STI);
  unsigned MaxNumVGPRs = alignDown(TotNumVGPRs / WavesPerEU, Granule);

  if (MaxNumVGPRs == alignDown(TotNumVGPRs / MaxWavesPerEU, Granule))
    return 0;

  unsigned MinWavesPerEU = getNumWavesPerEUWithNumVGPRs(STI, AddrsableNumVGPRs);
  if (WavesPerEU < MinWavesPerEU)
    return getMinNumVGPRs(STI, MinWavesPerEU);

  unsigned MaxNumVGPRsNext = alignDown(TotNumVGPRs / (WavesPerEU + 1), Granule);
  unsigned MinNumVGPRs = 1 + std::min(MaxNumVGPRs - Granule, MaxNumVGPRsNext);
  return std::min(MinNumVGPRs, AddrsableNumVGPRs);
}

unsigned getMaxNumVGPRs(const MCSubtargetInfo *STI, unsigned WavesPerEU) {
  assert(WavesPerEU != 0);

  unsigned MaxNumVGPRs = alignDown(getTotalNumVGPRs(STI) / WavesPerEU,
                                   getVGPRAllocGranule(STI));
  unsigned AddressableNumVGPRs = getAddressableNumVGPRs(STI);
  return std::min(MaxNumVGPRs, AddressableNumVGPRs);
}

unsigned getNumVGPRBlocks(const MCSubtargetInfo *STI, unsigned NumVGPRs,
                          std::optional<bool> EnableWavefrontSize32) {
  NumVGPRs = alignTo(std::max(1u, NumVGPRs),
                     getVGPREncodingGranule(STI, EnableWavefrontSize32));
  // VGPRBlocks is actual number of VGPR blocks minus 1.
  return NumVGPRs / getVGPREncodingGranule(STI, EnableWavefrontSize32) - 1;
}

} // end namespace IsaInfo

void initDefaultRVKernelCodeT(rv_kernel_code_t &Header,
                               const MCSubtargetInfo *STI) {
  memset(&Header, 0, sizeof(Header));

  Header.rv_kernel_code_version_major = 1;
  Header.rv_kernel_code_version_minor = 2;
  Header.rv_machine_kind = 1; // RV_MACHINE_KIND_RVGPU
  Header.rv_machine_version_major = 100;
  Header.rv_machine_version_minor = 0;
  Header.rv_machine_version_stepping = 0;
  Header.kernel_code_entry_byte_offset = sizeof(Header);
  Header.wavefront_size = 6;

  // If the code object does not support indirect functions, then the value must
  // be 0xffffffff.
  Header.call_convention = -1;

  // These alignment values are specified in powers of two, so alignment =
  // 2^n.  The minimum alignment is 2^4 = 16.
  Header.kernarg_segment_alignment = 4;
  Header.group_segment_alignment = 4;
  Header.private_segment_alignment = 4;
}

rvhsa::kernel_descriptor_t getDefaultRvhsaKernelDescriptor(
    const MCSubtargetInfo *STI) {
  rvhsa::kernel_descriptor_t KD;
  memset(&KD, 0, sizeof(KD));

  return KD;
}

bool isGroupSegment(const GlobalValue *GV) {
  return GV->getAddressSpace() == RVGPUAS::LOCAL_ADDRESS;
}

bool isGlobalSegment(const GlobalValue *GV) {
  return GV->getAddressSpace() == RVGPUAS::GLOBAL_ADDRESS;
}

bool isReadOnlySegment(const GlobalValue *GV) {
  unsigned AS = GV->getAddressSpace();
  return AS == RVGPUAS::CONSTANT_ADDRESS ||
         AS == RVGPUAS::CONSTANT_ADDRESS_32BIT;
}

std::pair<unsigned, unsigned>
getIntegerPairAttribute(const Function &F, StringRef Name,
                        std::pair<unsigned, unsigned> Default,
                        bool OnlyFirstRequired) {
  Attribute A = F.getFnAttribute(Name);
  if (!A.isStringAttribute())
    return Default;

  LLVMContext &Ctx = F.getContext();
  std::pair<unsigned, unsigned> Ints = Default;
  std::pair<StringRef, StringRef> Strs = A.getValueAsString().split(',');
  if (Strs.first.trim().getAsInteger(0, Ints.first)) {
    Ctx.emitError("can't parse first integer attribute " + Name);
    return Default;
  }
  if (Strs.second.trim().getAsInteger(0, Ints.second)) {
    if (!OnlyFirstRequired || !Strs.second.trim().empty()) {
      Ctx.emitError("can't parse second integer attribute " + Name);
      return Default;
    }
  }

  return Ints;
}

bool isGraphics(CallingConv::ID cc) {
    return false;
//  return isShader(cc) || cc == CallingConv::RVGPU_Gfx;
}

bool isCompute(CallingConv::ID cc) {
    return true;
  //return !isGraphics(cc) || cc == CallingConv::RVGPU_CS;
}

bool isEntryFunctionCC(CallingConv::ID CC) {
  switch (CC) {
  case CallingConv::RVGPU_KERNEL:
#if 0      
  case CallingConv::SPIR_KERNEL:
  case CallingConv::RVGPU_VS:
  case CallingConv::RVGPU_GS:
  case CallingConv::RVGPU_PS:
  case CallingConv::RVGPU_CS:
  case CallingConv::RVGPU_ES:
  case CallingConv::RVGPU_HS:
  case CallingConv::RVGPU_LS:
#endif
    return true;
  default:
    return false;
  }
}

bool isChainCC(CallingConv::ID CC) {
  return false;
#if 0
  switch (CC) {
  case CallingConv::RVGPU_CS_Chain:
  case CallingConv::RVGPU_CS_ChainPreserve:
    return true;
  default:
    return false;
  }
#endif     
}

bool isModuleEntryFunctionCC(CallingConv::ID CC) {
    return isEntryFunctionCC(CC) || isChainCC(CC);
}

bool isKernelCC(const Function *Func) {
  return RVGPU::isModuleEntryFunctionCC(Func->getCallingConv());
}

bool hasGDS(const MCSubtargetInfo &STI) {
  return true;
}

bool hasMAIInsts(const MCSubtargetInfo &STI) {
    return false;
//  return STI.hasFeature(RVGPU::FeatureMAIInsts);
}

unsigned hasKernargPreload(const MCSubtargetInfo &STI) {
  return true;
//  return STI.hasFeature(RVGPU::FeatureKernargPreload);
}

int32_t getTotalNumVGPRs(int32_t ArgNumAGPR,
                         int32_t ArgNumVGPR) {
  return std::max(ArgNumVGPR, ArgNumAGPR);
}

bool isHi(unsigned Reg, const MCRegisterInfo &MRI) {
  return MRI.getEncodingValue(Reg) & RVGPU::HWEncoding::IS_HI;
}



// Avoid using MCRegisterClass::getSize, since that function will go away
// (move from MC* level to Target* level). Return size in bits.
unsigned getRegBitWidth(unsigned RCID) {
  switch (RCID) {
  case RVGPU::GPR_LO16RegClassID:
  case RVGPU::GPR_HI16RegClassID:
    return 16;
  case RVGPU::GPR32RegClassID:
  case RVGPU::VRegOrLds_32RegClassID:
  case RVGPU::VS_32RegClassID:
    return 32;
  case RVGPU::VS_64RegClassID:
  case RVGPU::GPR64RegClassID:
    return 64;
  case RVGPU::GPR96RegClassID:
    return 96;
  case RVGPU::GPR128RegClassID:
    return 128;
  case RVGPU::GPR160RegClassID:
    return 160;
  case RVGPU::GPR192RegClassID:
    return 192;
  case RVGPU::GPR224RegClassID:
    return 224;
  case RVGPU::GPR256RegClassID:
    return 256;
  case RVGPU::GPR288RegClassID:
    return 288;
  case RVGPU::GPR320RegClassID:
    return 320;
  case RVGPU::GPR352RegClassID:
    return 352;
  case RVGPU::GPR384RegClassID:
    return 384;
  case RVGPU::GPR512RegClassID:
    return 512;
  case RVGPU::GPR1024RegClassID:
    return 1024;
  default:
    llvm_unreachable("Unexpected register class");
  }
}

unsigned getRegBitWidth(const MCRegisterClass &RC) {
  return getRegBitWidth(RC.getID());
}

unsigned getRegOperandSize(const MCRegisterInfo *MRI, const MCInstrDesc &Desc,
                           unsigned OpNo) {
  assert(OpNo < Desc.NumOperands);
  unsigned RCID = Desc.operands()[OpNo].RegClass;
  return getRegBitWidth(RCID) / 8;
}

bool isInlinableLiteral64(int64_t Literal, bool HasInv2Pi) {
  if (isInlinableIntLiteral(Literal))
    return true;

  uint64_t Val = static_cast<uint64_t>(Literal);
  return (Val == llvm::bit_cast<uint64_t>(0.0)) ||
         (Val == llvm::bit_cast<uint64_t>(1.0)) ||
         (Val == llvm::bit_cast<uint64_t>(-1.0)) ||
         (Val == llvm::bit_cast<uint64_t>(0.5)) ||
         (Val == llvm::bit_cast<uint64_t>(-0.5)) ||
         (Val == llvm::bit_cast<uint64_t>(2.0)) ||
         (Val == llvm::bit_cast<uint64_t>(-2.0)) ||
         (Val == llvm::bit_cast<uint64_t>(4.0)) ||
         (Val == llvm::bit_cast<uint64_t>(-4.0)) ||
         (Val == 0x3fc45f306dc9c882 && HasInv2Pi);
}

bool isInlinableLiteral32(int32_t Literal, bool HasInv2Pi) {
  if (isInlinableIntLiteral(Literal))
    return true;

  // The actual type of the operand does not seem to matter as long
  // as the bits match one of the inline immediate values.  For example:
  //
  // -nan has the hexadecimal encoding of 0xfffffffe which is -2 in decimal,
  // so it is a legal inline immediate.
  //
  // 1065353216 has the hexadecimal encoding 0x3f800000 which is 1.0f in
  // floating-point, so it is a legal inline immediate.

  uint32_t Val = static_cast<uint32_t>(Literal);
  return (Val == llvm::bit_cast<uint32_t>(0.0f)) ||
         (Val == llvm::bit_cast<uint32_t>(1.0f)) ||
         (Val == llvm::bit_cast<uint32_t>(-1.0f)) ||
         (Val == llvm::bit_cast<uint32_t>(0.5f)) ||
         (Val == llvm::bit_cast<uint32_t>(-0.5f)) ||
         (Val == llvm::bit_cast<uint32_t>(2.0f)) ||
         (Val == llvm::bit_cast<uint32_t>(-2.0f)) ||
         (Val == llvm::bit_cast<uint32_t>(4.0f)) ||
         (Val == llvm::bit_cast<uint32_t>(-4.0f)) ||
         (Val == 0x3e22f983 && HasInv2Pi);
}

bool isInlinableLiteral16(int16_t Literal, bool HasInv2Pi) {
  if (!HasInv2Pi)
    return false;

  if (isInlinableIntLiteral(Literal))
    return true;

  uint16_t Val = static_cast<uint16_t>(Literal);
  return Val == 0x3C00 || // 1.0
         Val == 0xBC00 || // -1.0
         Val == 0x3800 || // 0.5
         Val == 0xB800 || // -0.5
         Val == 0x4000 || // 2.0
         Val == 0xC000 || // -2.0
         Val == 0x4400 || // 4.0
         Val == 0xC400 || // -4.0
         Val == 0x3118;   // 1/2pi
}

bool isInlinableLiteralV216(int32_t Literal, bool HasInv2Pi) {
  assert(HasInv2Pi);

  if (isInt<16>(Literal) || isUInt<16>(Literal)) {
    int16_t Trunc = static_cast<int16_t>(Literal);
    return RVGPU::isInlinableLiteral16(Trunc, HasInv2Pi);
  }
  if (!(Literal & 0xffff))
    return RVGPU::isInlinableLiteral16(Literal >> 16, HasInv2Pi);

  int16_t Lo16 = static_cast<int16_t>(Literal);
  int16_t Hi16 = static_cast<int16_t>(Literal >> 16);
  return Lo16 == Hi16 && isInlinableLiteral16(Lo16, HasInv2Pi);
}

bool isInlinableIntLiteralV216(int32_t Literal) {
  int16_t Lo16 = static_cast<int16_t>(Literal);
  if (isInt<16>(Literal) || isUInt<16>(Literal))
    return isInlinableIntLiteral(Lo16);

  int16_t Hi16 = static_cast<int16_t>(Literal >> 16);
  if (!(Literal & 0xffff))
    return isInlinableIntLiteral(Hi16);
  return Lo16 == Hi16 && isInlinableIntLiteral(Lo16);
}

bool isInlinableLiteralV216(int32_t Literal, bool HasInv2Pi, uint8_t OpType) {
  switch (OpType) {
  case RVGPU::OPERAND_REG_IMM_V2FP16:
  case RVGPU::OPERAND_REG_INLINE_C_V2FP16:
    return isInlinableLiteralV216(Literal, HasInv2Pi);
  default:
    return isInlinableIntLiteralV216(Literal);
  }
}

bool isFoldableLiteralV216(int32_t Literal, bool HasInv2Pi) {
  assert(HasInv2Pi);

  int16_t Lo16 = static_cast<int16_t>(Literal);
  if (isInt<16>(Literal) || isUInt<16>(Literal))
    return true;

  int16_t Hi16 = static_cast<int16_t>(Literal >> 16);
  if (!(Literal & 0xffff))
    return true;
  return Lo16 == Hi16;
}

bool isValid32BitLiteral(uint64_t Val, bool IsFP64) {
  if (IsFP64)
    return !(Val & 0xffffffffu);

  return isUInt<32>(Val) || isInt<32>(Val);
}

bool isArgPassedInGPR(const Argument *A) {
  return true;
#if 0
  const Function *F = A->getParent();

  // Arguments to compute shaders are never a source of divergence.
  CallingConv::ID CC = F->getCallingConv();
  switch (CC) {
  case CallingConv::RVGPU_KERNEL:
  case CallingConv::SPIR_KERNEL:
    return true;
  case CallingConv::RVGPU_VS:
  case CallingConv::RVGPU_LS:
  case CallingConv::RVGPU_HS:
  case CallingConv::RVGPU_ES:
  case CallingConv::RVGPU_GS:
  case CallingConv::RVGPU_PS:
  case CallingConv::RVGPU_CS:
  case CallingConv::RVGPU_Gfx:
  case CallingConv::RVGPU_CS_Chain:
  case CallingConv::RVGPU_CS_ChainPreserve:
    // For non-compute shaders, SGPR inputs are marked with either inreg or
    // byval. Everything else is in VGPRs.
    return A->hasAttribute(Attribute::InReg) ||
           A->hasAttribute(Attribute::ByVal);
  default:
    // TODO: treat i1 as divergent?
    return A->hasAttribute(Attribute::InReg);
  }
#endif     
}

bool isArgPassedInGPR(const CallBase *CB, unsigned ArgNo) {
  return true;
#if 0    
  // Arguments to compute shaders are never a source of divergence.
  CallingConv::ID CC = CB->getCallingConv();
  switch (CC) {
  case CallingConv::RVGPU_KERNEL:
  case CallingConv::SPIR_KERNEL:
    return true;
  case CallingConv::RVGPU_VS:
  case CallingConv::RVGPU_LS:
  case CallingConv::RVGPU_HS:
  case CallingConv::RVGPU_ES:
  case CallingConv::RVGPU_GS:
  case CallingConv::RVGPU_PS:
  case CallingConv::RVGPU_CS:
  case CallingConv::RVGPU_Gfx:
  case CallingConv::RVGPU_CS_Chain:
  case CallingConv::RVGPU_CS_ChainPreserve:
    // For non-compute shaders, SGPR inputs are marked with either inreg or
    // byval. Everything else is in VGPRs.
    return CB->paramHasAttr(ArgNo, Attribute::InReg) ||
           CB->paramHasAttr(ArgNo, Attribute::ByVal);
  default:
    return CB->paramHasAttr(ArgNo, Attribute::InReg);
  }
#endif     
}

static bool isDwordAligned(uint64_t ByteOffset) {
  return (ByteOffset & 3) == 0;
}

namespace {

struct SourceOfDivergence {
  unsigned Intr;
};

struct AlwaysUniform {
  unsigned Intr;
};

} // end anonymous namespace
} // namespace RVGPU

raw_ostream &operator<<(raw_ostream &OS,
                        const RVGPU::IsaInfo::TargetIDSetting S) {
  switch (S) {
  case (RVGPU::IsaInfo::TargetIDSetting::Unsupported):
    OS << "Unsupported";
    break;
  case (RVGPU::IsaInfo::TargetIDSetting::Any):
    OS << "Any";
    break;
  case (RVGPU::IsaInfo::TargetIDSetting::Off):
    OS << "Off";
    break;
  case (RVGPU::IsaInfo::TargetIDSetting::On):
    OS << "On";
    break;
  }
  return OS;
}

} // namespace llvm
