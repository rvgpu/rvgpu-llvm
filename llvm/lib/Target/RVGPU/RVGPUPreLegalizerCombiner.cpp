//=== lib/CodeGen/GlobalISel/RVGPUPreLegalizerCombiner.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass does combining of machine instructions at the generic MI level,
// before the legalizer.
//
//===----------------------------------------------------------------------===//

#include "RVGPU.h"
#include "RVGPUCombinerHelper.h"
#include "RVGPULegalizerInfo.h"
#include "RVSubtarget.h"
#include "MCTargetDesc/RVGPUMCTargetDesc.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Target/TargetMachine.h"

#define GET_GICOMBINER_DEPS
#include "RVGPUGenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_DEPS

#define DEBUG_TYPE "rvgpu-prelegalizer-combiner"

using namespace llvm;
using namespace MIPatternMatch;
namespace {

#define GET_GICOMBINER_TYPES
#include "RVGPUGenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_TYPES

class RVGPUPreLegalizerCombinerImpl : public Combiner {
protected:
  const RVGPUPreLegalizerCombinerImplRuleConfig &RuleConfig;
  const RVSubtarget &STI;
  // TODO: Make CombinerHelper methods const.
  mutable RVGPUCombinerHelper Helper;

public:
  RVGPUPreLegalizerCombinerImpl(
      MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
      GISelKnownBits &KB, GISelCSEInfo *CSEInfo,
      const RVGPUPreLegalizerCombinerImplRuleConfig &RuleConfig,
      const RVSubtarget &STI, MachineDominatorTree *MDT,
      const LegalizerInfo *LI);

  static const char *getName() { return "RVGPUPreLegalizerCombinerImpl"; }

  bool tryCombineAllImpl(MachineInstr &MI) const;
  bool tryCombineAll(MachineInstr &I) const override;

  struct ClampI64ToI16MatchInfo {
    int64_t Cmp1 = 0;
    int64_t Cmp2 = 0;
    Register Origin;
  };

  bool matchClampI64ToI16(MachineInstr &MI, const MachineRegisterInfo &MRI,
                          const MachineFunction &MF,
                          ClampI64ToI16MatchInfo &MatchInfo) const;

  void applyClampI64ToI16(MachineInstr &MI,
                          const ClampI64ToI16MatchInfo &MatchInfo) const;

private:
#define GET_GICOMBINER_CLASS_MEMBERS
#define RVGPUSubtarget RVSubtarget
#include "RVGPUGenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CLASS_MEMBERS
#undef RVGPUSubtarget
};

#define GET_GICOMBINER_IMPL
#define RVGPUSubtarget RVSubtarget
#include "RVGPUGenPreLegalizeGICombiner.inc"
#undef RVGPUSubtarget
#undef GET_GICOMBINER_IMPL

RVGPUPreLegalizerCombinerImpl::RVGPUPreLegalizerCombinerImpl(
    MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
    GISelKnownBits &KB, GISelCSEInfo *CSEInfo,
    const RVGPUPreLegalizerCombinerImplRuleConfig &RuleConfig,
    const RVSubtarget &STI, MachineDominatorTree *MDT, const LegalizerInfo *LI)
    : Combiner(MF, CInfo, TPC, &KB, CSEInfo), RuleConfig(RuleConfig), STI(STI),
      Helper(Observer, B, /*IsPreLegalize*/ true, &KB, MDT, LI),
#define GET_GICOMBINER_CONSTRUCTOR_INITS
#include "RVGPUGenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CONSTRUCTOR_INITS
{
}

bool RVGPUPreLegalizerCombinerImpl::tryCombineAll(MachineInstr &MI) const {
  if (tryCombineAllImpl(MI))
    return true;

  switch (MI.getOpcode()) {
  case TargetOpcode::G_CONCAT_VECTORS:
    return Helper.tryCombineConcatVectors(MI);
  case TargetOpcode::G_SHUFFLE_VECTOR:
    return Helper.tryCombineShuffleVector(MI);
  }

  return false;
}

bool RVGPUPreLegalizerCombinerImpl::matchClampI64ToI16(
    MachineInstr &MI, const MachineRegisterInfo &MRI, const MachineFunction &MF,
    ClampI64ToI16MatchInfo &MatchInfo) const {
  assert(MI.getOpcode() == TargetOpcode::G_TRUNC && "Invalid instruction!");

  // Try to find a pattern where an i64 value should get clamped to short.
  const LLT SrcType = MRI.getType(MI.getOperand(1).getReg());
  if (SrcType != LLT::scalar(64))
    return false;

  const LLT DstType = MRI.getType(MI.getOperand(0).getReg());
  if (DstType != LLT::scalar(16))
    return false;

  Register Base;

  auto IsApplicableForCombine = [&MatchInfo]() -> bool {
    const auto Cmp1 = MatchInfo.Cmp1;
    const auto Cmp2 = MatchInfo.Cmp2;
    const auto Diff = std::abs(Cmp2 - Cmp1);

    // If the difference between both comparison values is 0 or 1, there is no
    // need to clamp.
    if (Diff == 0 || Diff == 1)
      return false;

    const int64_t Min = std::numeric_limits<int16_t>::min();
    const int64_t Max = std::numeric_limits<int16_t>::max();

    // Check if the comparison values are between SHORT_MIN and SHORT_MAX.
    return ((Cmp2 >= Cmp1 && Cmp1 >= Min && Cmp2 <= Max) ||
            (Cmp1 >= Cmp2 && Cmp1 <= Max && Cmp2 >= Min));
  };

  // Try to match a combination of min / max MIR opcodes.
  if (mi_match(MI.getOperand(1).getReg(), MRI,
               m_GSMin(m_Reg(Base), m_ICst(MatchInfo.Cmp1)))) {
    if (mi_match(Base, MRI,
                 m_GSMax(m_Reg(MatchInfo.Origin), m_ICst(MatchInfo.Cmp2)))) {
      return IsApplicableForCombine();
    }
  }

  if (mi_match(MI.getOperand(1).getReg(), MRI,
               m_GSMax(m_Reg(Base), m_ICst(MatchInfo.Cmp1)))) {
    if (mi_match(Base, MRI,
                 m_GSMin(m_Reg(MatchInfo.Origin), m_ICst(MatchInfo.Cmp2)))) {
      return IsApplicableForCombine();
    }
  }

  return false;
}

// We want to find a combination of instructions that
// gets generated when an i64 gets clamped to i16.
// The corresponding pattern is:
// G_MAX / G_MAX for i16 <= G_TRUNC i64.
// This can be efficiently written as following:
// v_cvt_pk_i16_i32 v0, v0, v1
// v_med3_i32 v0, Clamp_Min, v0, Clamp_Max
void RVGPUPreLegalizerCombinerImpl::applyClampI64ToI16(
    MachineInstr &MI, const ClampI64ToI16MatchInfo &MatchInfo) const {

  Register Src = MatchInfo.Origin;
  assert(MI.getParent()->getParent()->getRegInfo().getType(Src) ==
         LLT::scalar(64));
  const LLT S32 = LLT::scalar(32);

  B.setInstrAndDebugLoc(MI);

  auto Unmerge = B.buildUnmerge(S32, Src);

  assert(MI.getOpcode() != RVGPU::G_RVGPU_CVT_PK_I16_I32);

  const LLT V2S16 = LLT::fixed_vector(2, 16);
  auto CvtPk =
      B.buildInstr(RVGPU::G_RVGPU_CVT_PK_I16_I32, {V2S16},
                   {Unmerge.getReg(0), Unmerge.getReg(1)}, MI.getFlags());

  auto MinBoundary = std::min(MatchInfo.Cmp1, MatchInfo.Cmp2);
  auto MaxBoundary = std::max(MatchInfo.Cmp1, MatchInfo.Cmp2);
  auto MinBoundaryDst = B.buildConstant(S32, MinBoundary);
  auto MaxBoundaryDst = B.buildConstant(S32, MaxBoundary);

  auto Bitcast = B.buildBitcast({S32}, CvtPk);

  auto Med3 = B.buildInstr(
      RVGPU::G_RVGPU_SMED3, {S32},
      {MinBoundaryDst.getReg(0), Bitcast.getReg(0), MaxBoundaryDst.getReg(0)},
      MI.getFlags());

  B.buildTrunc(MI.getOperand(0).getReg(), Med3);

  MI.eraseFromParent();
}

// Pass boilerplate
// ================

class RVGPUPreLegalizerCombiner : public MachineFunctionPass {
public:
  static char ID;

  RVGPUPreLegalizerCombiner(bool IsOptNone = false);

  StringRef getPassName() const override {
    return "RVGPUPreLegalizerCombiner";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  bool IsOptNone;
  RVGPUPreLegalizerCombinerImplRuleConfig RuleConfig;
};
} // end anonymous namespace

void RVGPUPreLegalizerCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  AU.addRequired<GISelKnownBitsAnalysis>();
  AU.addPreserved<GISelKnownBitsAnalysis>();
  if (!IsOptNone) {
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
  }

  AU.addRequired<GISelCSEAnalysisWrapperPass>();
  AU.addPreserved<GISelCSEAnalysisWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

RVGPUPreLegalizerCombiner::RVGPUPreLegalizerCombiner(bool IsOptNone)
    : MachineFunctionPass(ID), IsOptNone(IsOptNone) {
  initializeRVGPUPreLegalizerCombinerPass(*PassRegistry::getPassRegistry());

  if (!RuleConfig.parseCommandLineOption())
    report_fatal_error("Invalid rule identifier");
}

bool RVGPUPreLegalizerCombiner::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  auto *TPC = &getAnalysis<TargetPassConfig>();
  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOptLevel::None && !skipFunction(F);
  GISelKnownBits *KB = &getAnalysis<GISelKnownBitsAnalysis>().get(MF);

  // Enable CSE.
  GISelCSEAnalysisWrapper &Wrapper =
      getAnalysis<GISelCSEAnalysisWrapperPass>().getCSEWrapper();
  auto *CSEInfo = &Wrapper.get(TPC->getCSEConfig());

  const RVSubtarget &STI = MF.getSubtarget<RVSubtarget>();
  MachineDominatorTree *MDT =
      IsOptNone ? nullptr : &getAnalysis<MachineDominatorTree>();
  CombinerInfo CInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     nullptr, EnableOpt, F.hasOptSize(), F.hasMinSize());
  RVGPUPreLegalizerCombinerImpl Impl(MF, CInfo, TPC, *KB, CSEInfo, RuleConfig,
                                      STI, MDT, STI.getLegalizerInfo());
  return Impl.combineMachineInstrs();
}

char RVGPUPreLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(RVGPUPreLegalizerCombiner, DEBUG_TYPE,
                      "Combine RVGPU machine instrs before legalization",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelKnownBitsAnalysis)
INITIALIZE_PASS_END(RVGPUPreLegalizerCombiner, DEBUG_TYPE,
                    "Combine RVGPU machine instrs before legalization", false,
                    false)

namespace llvm {
FunctionPass *createRVGPUPreLegalizeCombiner(bool IsOptNone) {
  return new RVGPUPreLegalizerCombiner(IsOptNone);
}
} // end namespace llvm
