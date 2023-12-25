//=====-- RVSubtarget.h - Define GCN Subtarget for RVGPU ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==-----------------------------------------------------------------------===//
//
/// \file
/// RV specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RVGPU_GCNSUBTARGET_H
#define LLVM_LIB_TARGET_RVGPU_GCNSUBTARGET_H

#include "RVGPUCallLowering.h"
#include "RVGPURegisterBankInfo.h"
#include "RVGPUSubtarget.h"
#include "RVFrameLowering.h"
#include "RVISelLowering.h"
#include "RVInstrInfo.h"
#include "Utils/RVGPUBaseInfo.h"
#include "llvm/CodeGen/SelectionDAGTargetInfo.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_SUBTARGETINFO_HEADER
#include "RVGPUGenSubtargetInfo.inc"

namespace llvm {

class RVTargetMachine;

class RVSubtarget final : public RVGPUGenSubtargetInfo,
                           public RVGPUSubtarget {
public:
  using RVGPUSubtarget::getMaxWavesPerEU;

  // Following 2 enums are documented at:
  //   - https://llvm.org/docs/RVGPUUsage.html#trap-handler-abi
  enum class TrapHandlerAbi {
    NONE   = 0x00,
    SS = 0x01,
  };

  enum class TrapID {
    LLVMSSTrap      = 0x02,
    LLVMSSDebugTrap = 0x03,
  };

private:
  /// GlobalISel related APIs.
  std::unique_ptr<RVGPUCallLowering> CallLoweringInfo;
  std::unique_ptr<InlineAsmLowering> InlineAsmLoweringInfo;
  std::unique_ptr<InstructionSelector> InstSelector;
  std::unique_ptr<LegalizerInfo> Legalizer;
  std::unique_ptr<RVGPURegisterBankInfo> RegBankInfo;

protected:
  // Basic subtarget description.
  Triple TargetTriple;
  RVGPU::IsaInfo::RVGPUTargetID TargetID;
  unsigned Gen = INVALID;
  InstrItineraryData InstrItins;
  int LDSBankCount = 0;
  unsigned MaxPrivateElementSize = 0;

  // Possibly statically set by tablegen, but may want to be overridden.
  bool FastDenormalF32 = false;
  bool HalfRate64Ops = false;
  bool FullRate64Ops = false;

  // Dynamically set bits that enable features.
  bool FlatForGlobal = false;
  bool AutoWaitcntBeforeBarrier = false;
  bool BackOffBarrier = false;
  bool UnalignedScratchAccess = false;
  bool UnalignedAccessMode = false;
  bool HasApertureRegs = false;
  bool SupportsXNACK = false;
  bool KernargPreload = false;

  // This should not be used directly. 'TargetID' tracks the dynamic settings
  // for XNACK.
  bool EnableXNACK = false;

  bool EnableTgSplit = false;
  bool EnableCuMode = false;
  bool TrapHandler = false;

  // Used as options.
  bool EnableLoadStoreOpt = false;
  bool EnableUnsafeDSOffsetFolding = false;
  bool EnableSIScheduler = false;
  bool EnableDS128 = false;
  bool EnablePRTStrictNull = false;
  bool DumpCode = false;

  // Subtarget statically properties set by tablegen
  bool FP64 = false;
  bool FMA = false;
  bool MIMG_R128 = false;
  bool CIInsts = false;
  bool GFX8Insts = false;
  bool GFX9Insts = false;
  bool GFX90AInsts = false;
  bool GFX940Insts = false;
  bool GFX10Insts = false;
  bool GFX11Insts = false;
  bool GFX12Insts = false;
  bool GFX10_3Insts = false;
  bool GFX7GFX8GFX9Insts = false;
  bool SGPRInitBug = false;
  bool UserSGPRInit16Bug = false;
  bool NegativeScratchOffsetBug = false;
  bool NegativeUnalignedScratchOffsetBug = false;
  bool HasSMemRealTime = false;
  bool HasIntClamp = false;
  bool HasFmaMixInsts = false;
  bool HasMovrel = false;
  bool HasVGPRIndexMode = false;
  bool HasScalarDwordx3Loads = false;
  bool HasScalarStores = false;
  bool HasScalarAtomics = false;
  bool HasSDWAOmod = false;
  bool HasSDWAScalar = false;
  bool HasSDWASdst = false;
  bool HasSDWAMac = false;
  bool HasSDWAOutModsVOPC = false;
  bool HasDPP = false;
  bool HasDPP8 = false;
  bool HasDPALU_DPP = false;
  bool HasDPPSrc1SGPR = false;
  bool HasPackedFP32Ops = false;
  bool HasImageInsts = false;
  bool HasExtendedImageInsts = false;
  bool HasR128A16 = false;
  bool HasA16 = false;
  bool HasG16 = false;
  bool HasNSAEncoding = false;
  bool HasPartialNSAEncoding = false;
  bool GFX10_AEncoding = false;
  bool GFX10_BEncoding = false;
  bool HasDLInsts = false;
  bool HasFmacF64Inst = false;
  bool HasDot1Insts = false;
  bool HasDot2Insts = false;
  bool HasDot3Insts = false;
  bool HasDot4Insts = false;
  bool HasDot5Insts = false;
  bool HasDot6Insts = false;
  bool HasDot7Insts = false;
  bool HasDot8Insts = false;
  bool HasDot9Insts = false;
  bool HasDot10Insts = false;
  bool HasMAIInsts = false;
  bool HasFP8Insts = false;
  bool HasPkFmacF16Inst = false;
  bool HasAtomicDsPkAdd16Insts = false;
  bool HasAtomicFlatPkAdd16Insts = false;
  bool HasAtomicFaddRtnInsts = false;
  bool HasAtomicFaddNoRtnInsts = false;
  bool HasAtomicBufferGlobalPkAddF16NoRtnInsts = false;
  bool HasAtomicBufferGlobalPkAddF16Insts = false;
  bool HasAtomicCSubNoRtnInsts = false;
  bool HasAtomicGlobalPkAddBF16Inst = false;
  bool HasFlatAtomicFaddF32Inst = false;
  bool SupportsSRAMECC = false;

  // This should not be used directly. 'TargetID' tracks the dynamic settings
  // for SRAMECC.
  bool EnableSRAMECC = false;

  bool HasNoSdstCMPX = false;
  bool HasVscnt = false;
  bool HasGetWaveIdInst = false;
  bool HasSMemTimeInst = false;
  bool HasShaderCyclesRegister = false;
  bool HasVOP3Literal = false;
  bool HasNoDataDepHazard = false;
  bool FlatAddressSpace = false;
  bool FlatInstOffsets = false;
  bool FlatGlobalInsts = false;
  bool FlatScratchInsts = false;
  bool ScalarFlatScratchInsts = false;
  bool HasArchitectedFlatScratch = false;
  bool EnableFlatScratch = false;
  bool HasArchitectedSGPRs = false;
  bool HasGDS = false;
  bool HasGWS = false;
  bool AddNoCarryInsts = false;
  bool HasUnpackedD16VMem = false;
  bool LDSMisalignedBug = false;
  bool HasMFMAInlineLiteralBug = false;
  bool UnalignedBufferAccess = false;
  bool UnalignedDSAccess = false;
  bool HasPackedTID = false;
  bool ScalarizeGlobal = false;
  bool HasSALUFloatInsts = false;
  bool HasVGPRSingleUseHintInsts = false;
  bool HasPseudoScalarTrans = false;
  bool HasRestrictedSOffset = false;

  bool HasVcmpxPermlaneHazard = false;
  bool HasVMEMtoScalarWriteHazard = false;
  bool HasSMEMtoVectorWriteHazard = false;
  bool HasInstFwdPrefetchBug = false;
  bool HasVcmpxExecWARHazard = false;
  bool HasLdsBranchVmemWARHazard = false;
  bool HasNSAtoVMEMBug = false;
  bool HasNSAClauseBug = false;
  bool HasOffset3fBug = false;
  bool HasFlatSegmentOffsetBug = false;
  bool HasImageStoreD16Bug = false;
  bool HasImageGather4D16Bug = false;
  bool HasMSAALoadDstSelBug = false;
  bool HasGFX11FullVGPRs = false;
  bool HasMADIntraFwdBug = false;
  bool HasVOPDInsts = false;
  bool HasVALUTransUseHazard = false;
  bool HasForceStoreSC0SC1 = false;

  // Dummy feature to use for assembler in tablegen.
  bool FeatureDisable = false;

  SelectionDAGTargetInfo TSInfo;
private:
  RVInstrInfo InstrInfo;
  RVTargetLowering TLInfo;
  RVFrameLowering FrameLowering;

public:
  RVSubtarget(const Triple &TT, StringRef GPU, StringRef FS,
               const RVTargetMachine &TM);
  ~RVSubtarget() override;

  RVSubtarget &initializeSubtargetDependencies(const Triple &TT,
                                                   StringRef GPU, StringRef FS);

  const RVInstrInfo *getInstrInfo() const override {
    return &InstrInfo;
  }

  const RVFrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }

  const RVTargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }

  const RVRegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }

  const CallLowering *getCallLowering() const override {
    return CallLoweringInfo.get();
  }

  const InlineAsmLowering *getInlineAsmLowering() const override {
    return InlineAsmLoweringInfo.get();
  }

  InstructionSelector *getInstructionSelector() const override {
    return InstSelector.get();
  }

  const LegalizerInfo *getLegalizerInfo() const override {
    return Legalizer.get();
  }

  const RVGPURegisterBankInfo *getRegBankInfo() const override {
    return RegBankInfo.get();
  }

  const RVGPU::IsaInfo::RVGPUTargetID &getTargetID() const {
    return TargetID;
  }

  // Nothing implemented, just prevent crashes on use.
  const SelectionDAGTargetInfo *getSelectionDAGInfo() const override {
    return &TSInfo;
  }

  const InstrItineraryData *getInstrItineraryData() const override {
    return &InstrItins;
  }

  void ParseSubtargetFeatures(StringRef CPU, StringRef TuneCPU, StringRef FS);

  Generation getGeneration() const {
    return (Generation)Gen;
  }

  unsigned getMaxWaveScratchSize() const {
    // 15-bit field in units of 64-dword.
    return (64 * 4) * ((1 << 15) - 1);
  }

  /// Return the number of high bits known to be zero for a frame index.
  unsigned getKnownHighZeroBitsForFrameIndex() const {
    return llvm::countl_zero(getMaxWaveScratchSize()) + getWavefrontSizeLog2();
  }

  int getLDSBankCount() const {
    return LDSBankCount;
  }

  unsigned getMaxPrivateElementSize(bool ForBufferRSrc = false) const {
    return (ForBufferRSrc || !enableFlatScratch()) ? MaxPrivateElementSize : 16;
  }

  unsigned getConstantBusLimit(unsigned Opcode) const;

  /// Returns if the result of this instruction with a 16-bit result returned in
  /// a 32-bit register implicitly zeroes the high 16-bits, rather than preserve
  /// the original value.
  bool zeroesHigh16BitsOfDest(unsigned Opcode) const;

  bool supportsWGP() const { return true; }

  bool hasIntClamp() const {
    return HasIntClamp;
  }

  bool hasFP64() const {
    return FP64;
  }

  bool hasMIMG_R128() const {
    return MIMG_R128;
  }

  bool hasHWFP64() const {
    return FP64;
  }

  bool hasHalfRate64Ops() const {
    return HalfRate64Ops;
  }

  bool hasFullRate64Ops() const {
    return FullRate64Ops;
  }

  bool hasAddr64() const {
    return false;
  }

  bool hasFlat() const {
    return true;
  }

  // Return true if the target only has the reverse operand versions of VALU
  // shift instructions (e.g. v_lshrrev_b32, and no v_lshr_b32).
  bool hasOnlyRevVALUShifts() const {
    return true;
  }

  bool hasFractBug() const {
    return false;
  }

  bool hasBFE() const {
    return true;
  }

  bool hasBFI() const {
    return true;
  }

  bool hasBFM() const {
    return hasBFE();
  }

  bool hasBCNT(unsigned Size) const {
    return true;
  }

  bool hasFFBL() const {
    return true;
  }

  bool hasFFBH() const {
    return true;
  }

  bool hasMed3_16() const {
    return true;
  }

  bool hasMin3Max3_16() const {
    return true;
  }

  bool hasFmaMixInsts() const {
    return HasFmaMixInsts;
  }

  bool hasCARRY() const {
    return true;
  }

  bool hasFMA() const {
    return FMA;
  }

  bool hasSwap() const {
    return false;
  }

  bool hasScalarPackInsts() const {
    return false;
  }

  bool hasScalarMulHiInsts() const {
    return false;
  }

  TrapHandlerAbi getTrapHandlerAbi() const {
    return isRvHsaOS() ? TrapHandlerAbi::SS : TrapHandlerAbi::NONE;
  }

  bool supportsGetDoorbellID() const {
    // The S_GETREG DOORBELL_ID is supported by all GFX9 onward targets.
    return true;
  }

  /// True if the offset field of DS instructions works as expected. On SI, the
  /// offset uses a 16-bit adder and does not always wrap properly.
  bool hasUsableDSOffset() const {
    return true;
  }

  bool unsafeDSOffsetFoldingEnabled() const {
    return EnableUnsafeDSOffsetFolding;
  }

  /// Condition output from div_scale is usable.
  bool hasUsableDivScaleConditionOutput() const {
    return true;
  }

  /// Extra wait hazard is needed in some cases before
  /// s_cbranch_vccnz/s_cbranch_vccz.
  bool hasReadVCCZBug() const {
    return false;
  }

  /// Writes to VCC_LO/VCC_HI update the VCCZ flag.
  bool partialVCCWritesUpdateVCCZ() const {
    return true;
  }

  /// A read of an SGPR by SMRD instruction requires 4 wait states when the SGPR
  /// was written by a VALU instruction.
  bool hasSMRDReadVALUDefHazard() const {
    return false;
  }

  /// A read of an SGPR by a VMEM instruction requires 5 wait states when the
  /// SGPR was written by a VALU Instruction.
  bool hasVMEMReadSGPRVALUDefHazard() const {
    return true;
  }

  bool hasRFEHazards() const {
    return true;
  }

  /// Number of hazard wait states for s_setreg_b32/s_setreg_imm32_b32.
  unsigned getSetRegWaitStates() const {
    return 2;
  }

  bool dumpCode() const {
    return DumpCode;
  }

  /// Return the amount of LDS that can be used that will not restrict the
  /// occupancy lower than WaveCount.
  unsigned getMaxLocalMemSizeWithWaveCount(unsigned WaveCount,
                                           const Function &) const;

  bool supportsMinMaxDenormModes() const {
    return true;
  }

  /// \returns If target supports S_DENORM_MODE.
  bool hasDenormModeInst() const {
    return true;
  }

  bool useFlatForGlobal() const {
    return FlatForGlobal;
  }

  /// \returns If target supports ds_read/write_b128 and user enables generation
  /// of ds_read/write_b128.
  bool useDS128() const {
    return CIInsts && EnableDS128;
  }

  /// \return If target supports ds_read/write_b96/128.
  bool hasDS96AndDS128() const {
    return CIInsts;
  }

  /// Have v_trunc_f64, v_ceil_f64, v_rndne_f64
  bool haveRoundOpsF64() const {
    return CIInsts;
  }

  /// \returns If MUBUF instructions always perform range checking, even for
  /// buffer resources used for private memory access.
  bool privateMemoryResourceIsRangeChecked() const {
    return false;
  }

  /// \returns If target requires PRT Struct NULL support (zero result registers
  /// for sparse texture support).
  bool usePRTStrictNull() const {
    return EnablePRTStrictNull;
  }

  bool hasAutoWaitcntBeforeBarrier() const {
    return AutoWaitcntBeforeBarrier;
  }

  /// \returns true if the target supports backing off of s_barrier instructions
  /// when an exception is raised.
  bool supportsBackOffBarrier() const {
    return BackOffBarrier;
  }

  bool hasUnalignedBufferAccess() const {
    return UnalignedBufferAccess;
  }

  bool hasUnalignedBufferAccessEnabled() const {
    return UnalignedBufferAccess && UnalignedAccessMode;
  }

  bool hasUnalignedDSAccess() const {
    return UnalignedDSAccess;
  }

  bool hasUnalignedDSAccessEnabled() const {
    return UnalignedDSAccess && UnalignedAccessMode;
  }

  bool hasUnalignedScratchAccess() const {
    return UnalignedScratchAccess;
  }

  bool hasUnalignedAccessMode() const {
    return UnalignedAccessMode;
  }

  bool hasApertureRegs() const {
    return HasApertureRegs;
  }

  bool isTrapHandlerEnabled() const {
    return TrapHandler;
  }

  bool isXNACKEnabled() const {
    return TargetID.isXnackOnOrAny();
  }

  bool isTgSplitEnabled() const {
    return EnableTgSplit;
  }

  bool isCuModeEnabled() const {
    return EnableCuMode;
  }

  bool hasFlatAddressSpace() const {
    return FlatAddressSpace;
  }

  bool hasFlatScrRegister() const {
    return hasFlatAddressSpace();
  }

  bool hasFlatInstOffsets() const {
    return FlatInstOffsets;
  }

  bool hasFlatGlobalInsts() const {
    return FlatGlobalInsts;
  }

  bool hasFlatScratchInsts() const {
    return FlatScratchInsts;
  }

  // Check if target supports ST addressing mode with FLAT scratch instructions.
  // The ST addressing mode means no registers are used, either VGPR or SGPR,
  // but only immediate offset is swizzled and added to the FLAT scratch base.
  bool hasFlatScratchSTMode() const {
      return false;
  }

  bool hasFlatScratchSVSMode() const { return true; }

  bool hasScalarFlatScratchInsts() const {
    return ScalarFlatScratchInsts;
  }

  bool enableFlatScratch() const {
    return flatScratchIsArchitected() ||
           (EnableFlatScratch && hasFlatScratchInsts());
  }

  bool hasGlobalAddTidInsts() const {
    return GFX10_BEncoding;
  }

  bool hasAtomicCSub() const {
    return GFX10_BEncoding;
  }

  bool hasMultiDwordFlatScratchAddressing() const {
    return true;
  }

  bool hasFlatSegmentOffsetBug() const {
    return HasFlatSegmentOffsetBug;
  }

  bool hasFlatLgkmVMemCountInOrder() const {
    return true;
  }

  bool hasD16LoadStore() const {
    return true;
  }

  bool d16PreservesUnusedBits() const {
    return hasD16LoadStore() && !TargetID.isSramEccOnOrAny();
  }

  bool hasD16Images() const {
    return true;
  }

  /// Return if most LDS instructions have an m0 use that require m0 to be
  /// initialized.
  bool ldsRequiresM0Init() const {
    return false;
  }

  // True if the hardware rewinds and replays GWS operations if a wave is
  // preempted.
  //
  // If this is false, a GWS operation requires testing if a nack set the
  // MEM_VIOL bit, and repeating if so.
  bool hasGWSAutoReplay() const {
    return true;
  }

  /// \returns if target has ds_gws_sema_release_all instruction.
  bool hasGWSSemaReleaseAll() const {
    return CIInsts;
  }

  /// \returns true if the target has integer add/sub instructions that do not
  /// produce a carry-out. This includes v_add_[iu]32, v_sub_[iu]32,
  /// v_add_[iu]16, and v_sub_[iu]16, all of which support the clamp modifier
  /// for saturation.
  bool hasAddNoCarry() const {
    return AddNoCarryInsts;
  }

  bool hasScalarAddSub64() const { return false; }

  bool hasUnpackedD16VMem() const {
    return HasUnpackedD16VMem;
  }

  // Covers VS/PS/CS graphics shaders
  bool isMesaGfxShader(const Function &F) const {
    return isMesa3DOS() && RVGPU::isShader(F.getCallingConv());
  }

  bool hasMad64_32() const {
    return true;
  }

  bool hasSDWAOmod() const {
    return HasSDWAOmod;
  }

  bool hasSDWAScalar() const {
    return HasSDWAScalar;
  }

  bool hasSDWASdst() const {
    return HasSDWASdst;
  }

  bool hasSDWAMac() const {
    return HasSDWAMac;
  }

  bool hasSDWAOutModsVOPC() const {
    return HasSDWAOutModsVOPC;
  }

  bool hasDLInsts() const {
    return HasDLInsts;
  }

  bool hasFmacF64Inst() const { return HasFmacF64Inst; }

  bool hasDot1Insts() const {
    return HasDot1Insts;
  }

  bool hasDot2Insts() const {
    return HasDot2Insts;
  }

  bool hasDot3Insts() const {
    return HasDot3Insts;
  }

  bool hasDot4Insts() const {
    return HasDot4Insts;
  }

  bool hasDot5Insts() const {
    return HasDot5Insts;
  }

  bool hasDot6Insts() const {
    return HasDot6Insts;
  }

  bool hasDot7Insts() const {
    return HasDot7Insts;
  }

  bool hasDot8Insts() const {
    return HasDot8Insts;
  }

  bool hasDot9Insts() const {
    return HasDot9Insts;
  }

  bool hasDot10Insts() const {
    return HasDot10Insts;
  }

  bool hasMAIInsts() const {
    return HasMAIInsts;
  }

  bool hasFP8Insts() const {
    return HasFP8Insts;
  }

  bool hasPkFmacF16Inst() const {
    return HasPkFmacF16Inst;
  }

  bool hasAtomicDsPkAdd16Insts() const { return HasAtomicDsPkAdd16Insts; }

  bool hasAtomicFlatPkAdd16Insts() const { return HasAtomicFlatPkAdd16Insts; }

  bool hasAtomicFaddInsts() const {
    return HasAtomicFaddRtnInsts || HasAtomicFaddNoRtnInsts;
  }

  bool hasAtomicFaddRtnInsts() const { return HasAtomicFaddRtnInsts; }

  bool hasAtomicFaddNoRtnInsts() const { return HasAtomicFaddNoRtnInsts; }

  bool hasAtomicBufferGlobalPkAddF16NoRtnInsts() const {
    return HasAtomicBufferGlobalPkAddF16NoRtnInsts;
  }

  bool hasAtomicBufferGlobalPkAddF16Insts() const {
    return HasAtomicBufferGlobalPkAddF16Insts;
  }

  bool hasAtomicGlobalPkAddBF16Inst() const {
    return HasAtomicGlobalPkAddBF16Inst;
  }

  bool hasFlatAtomicFaddF32Inst() const { return HasFlatAtomicFaddF32Inst; }

  bool hasNoSdstCMPX() const {
    return HasNoSdstCMPX;
  }

  bool hasVscnt() const {
    return HasVscnt;
  }

  bool hasGetWaveIdInst() const {
    return HasGetWaveIdInst;
  }

  bool hasSMemTimeInst() const {
    return HasSMemTimeInst;
  }

  bool hasShaderCyclesRegister() const {
    return HasShaderCyclesRegister;
  }

  bool hasVOP3Literal() const {
    return HasVOP3Literal;
  }

  bool hasNoDataDepHazard() const {
    return HasNoDataDepHazard;
  }

  bool vmemWriteNeedsExpWaitcnt() const {
    return getGeneration() < SEA_ISLANDS;
  }

  bool hasInstPrefetch() const { return true; }

  bool hasPrefetch() const { return false; }

  // Has s_cmpk_* instructions.
  bool hasSCmpK() const { return true; }

  // Scratch is allocated in 256 dword per wave blocks for the entire
  // wavefront. When viewed from the perspective of an arbitrary workitem, this
  // is 4-byte aligned.
  //
  // Only 4-byte alignment is really needed to access anything. Transformations
  // on the pointer value itself may rely on the alignment / known low bits of
  // the pointer. Set this to something above the minimum to avoid needing
  // dynamic realignment in common cases.
  Align getStackAlignment() const { return Align(16); }

  bool enableMachineScheduler() const override {
    return true;
  }

  bool useAA() const override;

  bool enableSubRegLiveness() const override {
    return true;
  }

  void setScalarizeGlobalBehavior(bool b) { ScalarizeGlobal = b; }
  bool getScalarizeGlobalBehavior() const { return ScalarizeGlobal; }

  // static wrappers
  static bool hasHalfRate64Ops(const TargetSubtargetInfo &STI);

  // XXX - Why is this here if it isn't in the default pass set?
  bool enableEarlyIfConversion() const override {
    return true;
  }

  void overrideSchedPolicy(MachineSchedPolicy &Policy,
                           unsigned NumRegionInstrs) const override;

  unsigned getMaxNumUserSGPRs() const {
    return RVGPU::getMaxNumUserSGPRs(*this);
  }

  bool hasSMemRealTime() const {
    return HasSMemRealTime;
  }

  bool hasMovrel() const {
    return HasMovrel;
  }

  bool hasVGPRIndexMode() const {
    return HasVGPRIndexMode;
  }

  bool useVGPRIndexMode() const;

  bool hasScalarCompareEq64() const {
    return false;
  }

  bool hasScalarDwordx3Loads() const { return HasScalarDwordx3Loads; }

  bool hasScalarStores() const {
    return HasScalarStores;
  }

  bool hasScalarAtomics() const {
    return HasScalarAtomics;
  }

  bool hasLDSFPAtomicAdd() const { return false; }

  /// \returns true if the subtarget has the v_permlanex16_b32 instruction.
  bool hasPermLaneX16() const { return true; }

  /// \returns true if the subtarget has the v_permlane64_b32 instruction.
  bool hasPermLane64() const { return true; }

  bool hasDPP() const {
    return HasDPP;
  }

  bool hasDPPBroadcasts() const {
    return false;
  }

  bool hasDPPWavefrontShifts() const {
    return false;
  }

  bool hasDPP8() const {
    return HasDPP8;
  }

  bool hasDPALU_DPP() const {
    return HasDPALU_DPP;
  }

  bool hasDPPSrc1SGPR() const { return HasDPPSrc1SGPR; }

  bool hasPackedFP32Ops() const {
    return HasPackedFP32Ops;
  }

  // Has V_PK_MOV_B32 opcode
  bool hasPkMovB32() const {
    return false;
  }

  bool hasFmaakFmamkF32Insts() const {
    return false;
  }

  bool hasImageInsts() const {
    return HasImageInsts;
  }

  bool hasExtendedImageInsts() const {
    return HasExtendedImageInsts;
  }

  bool hasR128A16() const {
    return HasR128A16;
  }

  bool hasA16() const { return HasA16; }

  bool hasG16() const { return HasG16; }

  bool hasOffset3fBug() const {
    return HasOffset3fBug;
  }

  bool hasImageStoreD16Bug() const { return HasImageStoreD16Bug; }

  bool hasImageGather4D16Bug() const { return HasImageGather4D16Bug; }

  bool hasMADIntraFwdBug() const { return HasMADIntraFwdBug; }

  bool hasMSAALoadDstSelBug() const { return HasMSAALoadDstSelBug; }

  bool hasNSAEncoding() const { return HasNSAEncoding; }

  bool hasPartialNSAEncoding() const { return HasPartialNSAEncoding; }

  unsigned getNSAMaxSize(bool HasSampler = false) const {
    return RVGPU::getNSAMaxSize(*this, HasSampler);
  }

  bool hasMadF16() const;

  bool enableSIScheduler() const {
    return EnableSIScheduler;
  }

  bool loadStoreOptEnabled() const {
    return EnableLoadStoreOpt;
  }

  bool hasSGPRInitBug() const {
    return SGPRInitBug;
  }

  bool hasUserSGPRInit16Bug() const {
    return UserSGPRInit16Bug && isWave32();
  }

  bool hasNegativeScratchOffsetBug() const { return NegativeScratchOffsetBug; }

  bool hasNegativeUnalignedScratchOffsetBug() const {
    return NegativeUnalignedScratchOffsetBug;
  }

  bool hasMFMAInlineLiteralBug() const {
    return HasMFMAInlineLiteralBug;
  }

  bool has12DWordStoreHazard() const {
    return true;
  }

  // \returns true if the subtarget supports DWORDX3 load/store instructions.
  bool hasDwordx3LoadStores() const {
    return false;
  }

  bool hasReadM0MovRelInterpHazard() const {
    return false;
  }

  bool hasReadM0SendMsgHazard() const {
    return false;
  }

  bool hasReadM0LdsDmaHazard() const {
    return false;
  }

  bool hasReadM0LdsDirectHazard() const {
    return false;
  }

  bool hasVcmpxPermlaneHazard() const {
    return HasVcmpxPermlaneHazard;
  }

  bool hasVMEMtoScalarWriteHazard() const {
    return HasVMEMtoScalarWriteHazard;
  }

  bool hasSMEMtoVectorWriteHazard() const {
    return HasSMEMtoVectorWriteHazard;
  }

  bool hasLDSMisalignedBug() const {
    return LDSMisalignedBug && !EnableCuMode;
  }

  bool hasInstFwdPrefetchBug() const {
    return HasInstFwdPrefetchBug;
  }

  bool hasVcmpxExecWARHazard() const {
    return HasVcmpxExecWARHazard;
  }

  bool hasLdsBranchVmemWARHazard() const {
    return HasLdsBranchVmemWARHazard;
  }

  // Shift amount of a 64 bit shift cannot be a highest allocated register
  // if also at the end of the allocation block.
  bool hasShift64HighRegBug() const {
    return false;
  }

  // Has one cycle hazard on transcendental instruction feeding a
  // non transcendental VALU.
  bool hasTransForwardingHazard() const { return false; }

  // Has one cycle hazard on a VALU instruction partially writing dst with
  // a shift of result bits feeding another VALU instruction.
  bool hasDstSelForwardingHazard() const { return false; }

  // Cannot use op_sel with v_dot instructions.
  bool hasDOTOpSelHazard() const { return false; }

  // Does not have HW interlocs for VALU writing and then reading SGPRs.
  bool hasVDecCoExecHazard() const {
    return GFX940Insts;
  }

  bool hasNSAtoVMEMBug() const {
    return HasNSAtoVMEMBug;
  }

  bool hasNSAClauseBug() const { return HasNSAClauseBug; }

  bool hasHardClauses() const { return true; }

  bool hasGFX90AInsts() const { return GFX90AInsts; }

  bool hasFPAtomicToDenormModeHazard() const {
    return false;
  }

  bool hasVOP3DPP() const { return true; }

  bool hasLdsDirect() const { return true; }

  bool hasVALUPartialForwardingHazard() const {
    return true;
  }

  bool hasVALUTransUseHazard() const { return HasVALUTransUseHazard; }

  bool hasForceStoreSC0SC1() const { return HasForceStoreSC0SC1; }

  bool hasVALUMaskWriteHazard() const { return true; }

  /// Return if operations acting on VGPR tuples require even alignment.
  bool needsAlignedVGPRs() const { return GFX90AInsts; }

  /// Return true if the target has the S_PACK_HL_B32_B16 instruction.
  bool hasSPackHL() const { return GFX11Insts; }

  /// Return true if the target's EXP instruction has the COMPR flag, which
  /// affects the meaning of the EN (enable) bits.
  bool hasCompressedExport() const { return !GFX11Insts; }

  /// Return true if the target's EXP instruction supports the NULL export
  /// target.
  bool hasNullExportTarget() const { return !GFX11Insts; }

  bool hasGFX11FullVGPRs() const { return HasGFX11FullVGPRs; }
  
  bool hasGFX10_AEncoding() const {
    return GFX10_AEncoding;
  }

  bool hasGFX10_BEncoding() const {
    return GFX10_BEncoding;
  }

  bool hasGFX10_3Insts() const {
    return GFX10_3Insts;
  }
  bool hasLshlAddB64() const { return false; }

  bool hasMovB64() const { return false; }
  bool hasVOPDInsts() const { return HasVOPDInsts; }

  bool hasFlatScratchSVSSwizzleBug() const { return true; }

  /// Return true if the target has the S_DELAY_ALU instruction.
  bool hasDelayAlu() const { return GFX11Insts; }

  bool hasPackedTID() const { return HasPackedTID; }

  // GFX940 is a derivation to GFX90A. hasGFX940Insts() being true implies that
  // hasGFX90AInsts is also true.
  bool hasGFX940Insts() const { return GFX940Insts; }

  bool hasSALUFloatInsts() const { return HasSALUFloatInsts; }

  bool hasVGPRSingleUseHintInsts() const { return HasVGPRSingleUseHintInsts; }

  bool hasPseudoScalarTrans() const { return HasPseudoScalarTrans; }

  bool hasRestrictedSOffset() const { return HasRestrictedSOffset; }

  /// Return the maximum number of waves per SIMD for kernels using \p SGPRs
  /// SGPRs
  unsigned getOccupancyWithNumSGPRs(unsigned SGPRs) const;

  /// Return the maximum number of waves per SIMD for kernels using \p VGPRs
  /// VGPRs
  unsigned getOccupancyWithNumVGPRs(unsigned VGPRs) const;

  /// Return occupancy for the given function. Used LDS and a number of
  /// registers if provided.
  /// Note, occupancy can be affected by the scratch allocation as well, but
  /// we do not have enough information to compute it.
  unsigned computeOccupancy(const Function &F, unsigned LDSSize = 0,
                            unsigned NumSGPRs = 0, unsigned NumVGPRs = 0) const;

  /// \returns true if the flat_scratch register should be initialized with the
  /// pointer to the wave's scratch memory rather than a size and offset.
  bool flatScratchIsPointer() const {
    return true;
  }

  /// \returns true if the flat_scratch register is initialized by the HW.
  /// In this case it is readonly.
  bool flatScratchIsArchitected() const { return HasArchitectedFlatScratch; }

  /// \returns true if the architected SGPRs are enabled.
  bool hasArchitectedSGPRs() const { return HasArchitectedSGPRs; }

  /// \returns true if Global Data Share is supported.
  bool hasGDS() const { return HasGDS; }

  /// \returns true if Global Wave Sync is supported.
  bool hasGWS() const { return HasGWS; }

  /// \returns true if the machine has merged shaders in which s0-s7 are
  /// reserved by the hardware and user SGPRs start at s8
  bool hasMergedShaders() const {
    return true;
  }

  // \returns true if the target supports the pre-NGG legacy geometry path.
  bool hasLegacyGeometry() const { return false; }

  // \returns true if preloading kernel arguments is supported.
  bool hasKernargPreload() const { return KernargPreload; }

  // \returns true if we need to generate backwards compatible code when
  // preloading kernel arguments.
  bool needsKernargPreloadBackwardsCompatibility() const {
    return hasKernargPreload() && !hasGFX940Insts();
  }

  // \returns true if the target has split barriers feature
  bool hasSplitBarriers() const { return false; }

  // \returns true if FP8/BF8 VOP1 form of conversion to F32 is unreliable.
  bool hasCvtFP8VOP1Bug() const { return true; }

  // \returns true if CSUB (a.k.a. SUB_CLAMP on GFX12) atomics support a
  // no-return form.
  bool hasAtomicCSubNoRtnInsts() const { return HasAtomicCSubNoRtnInsts; }

  // \returns true if the target has DX10_CLAMP kernel descriptor mode bit
  bool hasDX10ClampMode() const { return true; }

  // \returns true if the target has IEEE kernel descriptor mode bit
  bool hasIEEEMode() const { return true; }

  // \returns true if the target has IEEE fminimum/fmaximum instructions
  bool hasIEEEMinMax() const { return false; }

  // \returns true if the target has WG_RR_MODE kernel descriptor mode bit
  bool hasRrWGMode() const { return false; }

  /// \returns SGPR allocation granularity supported by the subtarget.
  unsigned getSGPRAllocGranule() const {
    return RVGPU::IsaInfo::getSGPRAllocGranule(this);
  }

  /// \returns SGPR encoding granularity supported by the subtarget.
  unsigned getSGPREncodingGranule() const {
    return RVGPU::IsaInfo::getSGPREncodingGranule(this);
  }

  /// \returns Total number of SGPRs supported by the subtarget.
  unsigned getTotalNumSGPRs() const {
    return RVGPU::IsaInfo::getTotalNumSGPRs(this);
  }

  /// \returns Addressable number of SGPRs supported by the subtarget.
  unsigned getAddressableNumSGPRs() const {
    return RVGPU::IsaInfo::getAddressableNumSGPRs(this);
  }

  /// \returns Minimum number of SGPRs that meets the given number of waves per
  /// execution unit requirement supported by the subtarget.
  unsigned getMinNumSGPRs(unsigned WavesPerEU) const {
    return RVGPU::IsaInfo::getMinNumSGPRs(this, WavesPerEU);
  }

  /// \returns Maximum number of SGPRs that meets the given number of waves per
  /// execution unit requirement supported by the subtarget.
  unsigned getMaxNumSGPRs(unsigned WavesPerEU, bool Addressable) const {
    return RVGPU::IsaInfo::getMaxNumSGPRs(this, WavesPerEU, Addressable);
  }

  /// \returns Reserved number of SGPRs. This is common
  /// utility function called by MachineFunction and
  /// Function variants of getReservedNumSGPRs.
  unsigned getBaseReservedNumSGPRs(const bool HasFlatScratch) const;
  /// \returns Reserved number of SGPRs for given machine function \p MF.
  unsigned getReservedNumSGPRs(const MachineFunction &MF) const;

  /// \returns Reserved number of SGPRs for given function \p F.
  unsigned getReservedNumSGPRs(const Function &F) const;

  /// \returns max num SGPRs. This is the common utility
  /// function called by MachineFunction and Function
  /// variants of getMaxNumSGPRs.
  unsigned getBaseMaxNumSGPRs(const Function &F,
                              std::pair<unsigned, unsigned> WavesPerEU,
                              unsigned PreloadedSGPRs,
                              unsigned ReservedNumSGPRs) const;

  /// \returns Maximum number of SGPRs that meets number of waves per execution
  /// unit requirement for function \p MF, or number of SGPRs explicitly
  /// requested using "rvgpu-num-sgpr" attribute attached to function \p MF.
  ///
  /// \returns Value that meets number of waves per execution unit requirement
  /// if explicitly requested value cannot be converted to integer, violates
  /// subtarget's specifications, or does not meet number of waves per execution
  /// unit requirement.
  unsigned getMaxNumSGPRs(const MachineFunction &MF) const;

  /// \returns Maximum number of SGPRs that meets number of waves per execution
  /// unit requirement for function \p F, or number of SGPRs explicitly
  /// requested using "rvgpu-num-sgpr" attribute attached to function \p F.
  ///
  /// \returns Value that meets number of waves per execution unit requirement
  /// if explicitly requested value cannot be converted to integer, violates
  /// subtarget's specifications, or does not meet number of waves per execution
  /// unit requirement.
  unsigned getMaxNumSGPRs(const Function &F) const;

  /// \returns VGPR allocation granularity supported by the subtarget.
  unsigned getVGPRAllocGranule() const {
    return RVGPU::IsaInfo::getVGPRAllocGranule(this);
  }

  /// \returns VGPR encoding granularity supported by the subtarget.
  unsigned getVGPREncodingGranule() const {
    return RVGPU::IsaInfo::getVGPREncodingGranule(this);
  }

  /// \returns Total number of VGPRs supported by the subtarget.
  unsigned getTotalNumVGPRs() const {
    return RVGPU::IsaInfo::getTotalNumVGPRs(this);
  }

  /// \returns Addressable number of VGPRs supported by the subtarget.
  unsigned getAddressableNumVGPRs() const {
    return RVGPU::IsaInfo::getAddressableNumVGPRs(this);
  }

  /// \returns the minimum number of VGPRs that will prevent achieving more than
  /// the specified number of waves \p WavesPerEU.
  unsigned getMinNumVGPRs(unsigned WavesPerEU) const {
    return RVGPU::IsaInfo::getMinNumVGPRs(this, WavesPerEU);
  }

  /// \returns the maximum number of VGPRs that can be used and still achieved
  /// at least the specified number of waves \p WavesPerEU.
  unsigned getMaxNumVGPRs(unsigned WavesPerEU) const {
    return RVGPU::IsaInfo::getMaxNumVGPRs(this, WavesPerEU);
  }

  /// \returns max num VGPRs. This is the common utility function
  /// called by MachineFunction and Function variants of getMaxNumVGPRs.
  unsigned getBaseMaxNumVGPRs(const Function &F,
                              std::pair<unsigned, unsigned> WavesPerEU) const;
  /// \returns Maximum number of VGPRs that meets number of waves per execution
  /// unit requirement for function \p F, or number of VGPRs explicitly
  /// requested using "rvgpu-num-vgpr" attribute attached to function \p F.
  ///
  /// \returns Value that meets number of waves per execution unit requirement
  /// if explicitly requested value cannot be converted to integer, violates
  /// subtarget's specifications, or does not meet number of waves per execution
  /// unit requirement.
  unsigned getMaxNumVGPRs(const Function &F) const;

  unsigned getMaxNumAGPRs(const Function &F) const {
    return getMaxNumVGPRs(F);
  }

  /// \returns Maximum number of VGPRs that meets number of waves per execution
  /// unit requirement for function \p MF, or number of VGPRs explicitly
  /// requested using "rvgpu-num-vgpr" attribute attached to function \p MF.
  ///
  /// \returns Value that meets number of waves per execution unit requirement
  /// if explicitly requested value cannot be converted to integer, violates
  /// subtarget's specifications, or does not meet number of waves per execution
  /// unit requirement.
  unsigned getMaxNumVGPRs(const MachineFunction &MF) const;

  void getPostRAMutations(
      std::vector<std::unique_ptr<ScheduleDAGMutation>> &Mutations)
      const override;

  std::unique_ptr<ScheduleDAGMutation>
  createFillMFMAShadowMutation(const TargetInstrInfo *TII) const;

  bool isWave32() const {
    return getWavefrontSize() == 32;
  }

  bool isWave64() const {
    return getWavefrontSize() == 64;
  }

  const TargetRegisterClass *getBoolRC() const {
    return getRegisterInfo()->getBoolRC();
  }

  /// \returns Maximum number of work groups per compute unit supported by the
  /// subtarget and limited by given \p FlatWorkGroupSize.
  unsigned getMaxWorkGroupsPerCU(unsigned FlatWorkGroupSize) const override {
    return RVGPU::IsaInfo::getMaxWorkGroupsPerCU(this, FlatWorkGroupSize);
  }

  /// \returns Minimum flat work group size supported by the subtarget.
  unsigned getMinFlatWorkGroupSize() const override {
    return RVGPU::IsaInfo::getMinFlatWorkGroupSize(this);
  }

  /// \returns Maximum flat work group size supported by the subtarget.
  unsigned getMaxFlatWorkGroupSize() const override {
    return RVGPU::IsaInfo::getMaxFlatWorkGroupSize(this);
  }

  /// \returns Number of waves per execution unit required to support the given
  /// \p FlatWorkGroupSize.
  unsigned
  getWavesPerEUForWorkGroup(unsigned FlatWorkGroupSize) const override {
    return RVGPU::IsaInfo::getWavesPerEUForWorkGroup(this, FlatWorkGroupSize);
  }

  /// \returns Minimum number of waves per execution unit supported by the
  /// subtarget.
  unsigned getMinWavesPerEU() const override {
    return RVGPU::IsaInfo::getMinWavesPerEU(this);
  }

  void adjustSchedDependency(SUnit *Def, int DefOpIdx, SUnit *Use, int UseOpIdx,
                             SDep &Dep) const override;

  // \returns true if it's beneficial on this subtarget for the scheduler to
  // cluster stores as well as loads.
  bool shouldClusterStores() const { return true; }

  // \returns the number of address arguments from which to enable MIMG NSA
  // on supported architectures.
  unsigned getNSAThreshold(const MachineFunction &MF) const;

  // \returns true if the subtarget has a hazard requiring an "s_nop 0"
  // instruction before "s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)".
  bool requiresNopBeforeDeallocVGPRs() const {
    // Currently all targets that support the dealloc VGPRs message also require
    // the nop.
    return true;
  }
};

class RVUserSGPRUsageInfo {
public:
  bool hasImplicitBufferPtr() const { return ImplicitBufferPtr; }

  bool hasPrivateSegmentBuffer() const { return PrivateSegmentBuffer; }

  bool hasDispatchPtr() const { return DispatchPtr; }

  bool hasQueuePtr() const { return QueuePtr; }

  bool hasKernargSegmentPtr() const { return KernargSegmentPtr; }

  bool hasDispatchID() const { return DispatchID; }

  bool hasFlatScratchInit() const { return FlatScratchInit; }

  unsigned getNumKernargPreloadSGPRs() const { return NumKernargPreloadSGPRs; }

  unsigned getNumUsedUserSGPRs() const { return NumUsedUserSGPRs; }

  unsigned getNumFreeUserSGPRs();

  void allocKernargPreloadSGPRs(unsigned NumSGPRs);

  enum UserSGPRID : unsigned {
    ImplicitBufferPtrID = 0,
    PrivateSegmentBufferID = 1,
    DispatchPtrID = 2,
    QueuePtrID = 3,
    KernargSegmentPtrID = 4,
    DispatchIdID = 5,
    FlatScratchInitID = 6,
    PrivateSegmentSizeID = 7
  };

  // Returns the size in number of SGPRs for preload user SGPR field.
  static unsigned getNumUserSGPRForField(UserSGPRID ID) {
    switch (ID) {
    case ImplicitBufferPtrID:
      return 2;
    case PrivateSegmentBufferID:
      return 4;
    case DispatchPtrID:
      return 2;
    case QueuePtrID:
      return 2;
    case KernargSegmentPtrID:
      return 2;
    case DispatchIdID:
      return 2;
    case FlatScratchInitID:
      return 2;
    case PrivateSegmentSizeID:
      return 1;
    }
    llvm_unreachable("Unknown UserSGPRID.");
  }

  RVUserSGPRUsageInfo(const Function &F, const RVSubtarget &ST);

private:
  const RVSubtarget &ST;

  // Private memory buffer
  // Compute directly in sgpr[0:1]
  // Other shaders indirect 64-bits at sgpr[0:1]
  bool ImplicitBufferPtr = false;

  bool PrivateSegmentBuffer = false;

  bool DispatchPtr = false;

  bool QueuePtr = false;

  bool KernargSegmentPtr = false;

  bool DispatchID = false;

  bool FlatScratchInit = false;

  unsigned NumKernargPreloadSGPRs = 0;

  unsigned NumUsedUserSGPRs = 0;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_RVGPU_GCNSUBTARGET_H
