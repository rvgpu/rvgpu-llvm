//===- RVGPURegisterBankInfo.cpp -------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the targeting of the RegisterBankInfo class for
/// RVGPU.
///
/// \par
///
/// RVGPU has unique register bank constraints that require special high level
/// strategies to deal with. There are two main true physical register banks
/// VGPR (vector), and SGPR (scalar). Additionally the VCC register bank is a
/// sort of pseudo-register bank needed to represent SGPRs used in a vector
/// boolean context. There is also the AGPR bank, which is a special purpose
/// physical register bank present on some subtargets.
///
/// Copying from VGPR to SGPR is generally illegal, unless the value is known to
/// be uniform. It is generally not valid to legalize operands by inserting
/// copies as on other targets. Operations which require uniform, SGPR operands
/// generally require scalarization by repeatedly executing the instruction,
/// activating each set of lanes using a unique set of input values. This is
/// referred to as a waterfall loop.
///
/// \par Booleans
///
/// Booleans (s1 values) requires special consideration. A vector compare result
/// is naturally a bitmask with one bit per lane, in a 32 or 64-bit
/// register. These are represented with the VCC bank. During selection, we need
/// to be able to unambiguously go back from a register class to a register
/// bank. To distinguish whether an SGPR should use the SGPR or VCC register
/// bank, we need to know the use context type. An SGPR s1 value always means a
/// VCC bank value, otherwise it will be the SGPR bank. A scalar compare sets
/// SCC, which is a 1-bit unaddressable register. This will need to be copied to
/// a 32-bit virtual register. Taken together, this means we need to adjust the
/// type of boolean operations to be regbank legal. All SALU booleans need to be
/// widened to 32-bits, and all VALU booleans need to be s1 values.
///
/// A noteworthy exception to the s1-means-vcc rule is for legalization artifact
/// casts. G_TRUNC s1 results, and G_SEXT/G_ZEXT/G_ANYEXT sources are never vcc
/// bank. A non-boolean source (such as a truncate from a 1-bit load from
/// memory) will require a copy to the VCC bank which will require clearing the
/// high bits and inserting a compare.
///
/// \par Constant bus restriction
///
/// VALU instructions have a limitation known as the constant bus
/// restriction. Most VALU instructions can use SGPR operands, but may read at
/// most 1 SGPR or constant literal value (this to 2 in gfx10 for most
/// instructions). This is one unique SGPR, so the same SGPR may be used for
/// multiple operands. From a register bank perspective, any combination of
/// operands should be legal as an SGPR, but this is contextually dependent on
/// the SGPR operands all being the same register. There is therefore optimal to
/// choose the SGPR with the most uses to minimize the number of copies.
///
/// We avoid trying to solve this problem in RegBankSelect. Any VALU G_*
/// operation should have its source operands all mapped to VGPRs (except for
/// VCC), inserting copies from any SGPR operands. This the most trivial legal
/// mapping. Anything beyond the simplest 1:1 instruction selection would be too
/// complicated to solve here. Every optimization pattern or instruction
/// selected to multiple outputs would have to enforce this rule, and there
/// would be additional complexity in tracking this rule for every G_*
/// operation. By forcing all inputs to VGPRs, it also simplifies the task of
/// picking the optimal operand combination from a post-isel optimization pass.
///
//===----------------------------------------------------------------------===//

#include "RVGPURegisterBankInfo.h"

#include "RVGPU.h"
#include "RVGPUGlobalISelUtils.h"
#include "RVGPUInstrInfo.h"
#include "RVSubtarget.h"
#include "RVMachineFunctionInfo.h"
#include "RVRegisterInfo.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/RegisterBank.h"
#include "llvm/IR/IntrinsicsRVGPU.h"

#define GET_TARGET_REGBANK_IMPL
#include "RVGPUGenRegisterBank.inc"

// This file will be TableGen'ed at some point.
#include "RVGPUGenRegisterBankInfo.def"

using namespace llvm;
using namespace MIPatternMatch;

namespace {

// Observer to apply a register bank to new registers created by LegalizerHelper.
class ApplyRegBankMapping final : public GISelChangeObserver {
private:
  MachineIRBuilder &B;
  const RVGPURegisterBankInfo &RBI;
  MachineRegisterInfo &MRI;
  const RegisterBank *NewBank;
  SmallVector<MachineInstr *, 4> NewInsts;

public:
  ApplyRegBankMapping(MachineIRBuilder &B, const RVGPURegisterBankInfo &RBI_,
                      MachineRegisterInfo &MRI_, const RegisterBank *RB)
      : B(B), RBI(RBI_), MRI(MRI_), NewBank(RB) {
    assert(!B.isObservingChanges());
    B.setChangeObserver(*this);
  }

  ~ApplyRegBankMapping() {
    for (MachineInstr *MI : NewInsts)
      applyBank(*MI);

    B.stopObservingChanges();
  }

  /// Set any registers that don't have a set register class or bank to SALU.
  void applyBank(MachineInstr &MI) {
    const unsigned Opc = MI.getOpcode();
    if (Opc == RVGPU::G_ANYEXT || Opc == RVGPU::G_ZEXT ||
        Opc == RVGPU::G_SEXT) {
      // LegalizerHelper wants to use the basic legalization artifacts when
      // widening etc. We don't handle selection with vcc in artifact sources,
      // so we need to use a select instead to handle these properly.
      Register DstReg = MI.getOperand(0).getReg();
      Register SrcReg = MI.getOperand(1).getReg();
      const RegisterBank *SrcBank = RBI.getRegBank(SrcReg, MRI, *RBI.TRI);
      if (SrcBank == &RVGPU::VCCRegBank) {
        const LLT S32 = LLT::scalar(32);
        assert(MRI.getType(SrcReg) == LLT::scalar(1));
        assert(MRI.getType(DstReg) == S32);
        assert(NewBank == &RVGPU::VGPRRegBank);

        // Replace the extension with a select, which really uses the boolean
        // source.
        B.setInsertPt(*MI.getParent(), MI);

        auto True = B.buildConstant(S32, Opc == RVGPU::G_SEXT ? -1 : 1);
        auto False = B.buildConstant(S32, 0);
        B.buildSelect(DstReg, SrcReg, True, False);
        MRI.setRegBank(True.getReg(0), *NewBank);
        MRI.setRegBank(False.getReg(0), *NewBank);
        MI.eraseFromParent();
      }

      assert(!MRI.getRegClassOrRegBank(DstReg));
      MRI.setRegBank(DstReg, *NewBank);
      return;
    }

#ifndef NDEBUG
    if (Opc == RVGPU::G_TRUNC) {
      Register DstReg = MI.getOperand(0).getReg();
      const RegisterBank *DstBank = RBI.getRegBank(DstReg, MRI, *RBI.TRI);
      assert(DstBank != &RVGPU::VCCRegBank);
    }
#endif

    for (MachineOperand &Op : MI.operands()) {
      if (!Op.isReg())
        continue;

      // We may see physical registers if building a real MI
      Register Reg = Op.getReg();
      if (Reg.isPhysical() || MRI.getRegClassOrRegBank(Reg))
        continue;

      const RegisterBank *RB = NewBank;
      if (MRI.getType(Reg) == LLT::scalar(1)) {
        assert(NewBank == &RVGPU::VGPRRegBank &&
               "s1 operands should only be used for vector bools");
        assert((MI.getOpcode() != RVGPU::G_TRUNC &&
                MI.getOpcode() != RVGPU::G_ANYEXT) &&
               "not expecting legalization artifacts here");
        RB = &RVGPU::VCCRegBank;
      }

      MRI.setRegBank(Reg, *RB);
    }
  }

  void erasingInstr(MachineInstr &MI) override {}

  void createdInstr(MachineInstr &MI) override {
    // At this point, the instruction was just inserted and has no operands.
    NewInsts.push_back(&MI);
  }

  void changingInstr(MachineInstr &MI) override {}
  void changedInstr(MachineInstr &MI) override {
    // FIXME: In principle we should probably add the instruction to NewInsts,
    // but the way the LegalizerHelper uses the observer, we will always see the
    // registers we need to set the regbank on also referenced in a new
    // instruction.
  }
};

}

RVGPURegisterBankInfo::RVGPURegisterBankInfo(const RVSubtarget &ST)
    : Subtarget(ST), TRI(Subtarget.getRegisterInfo()),
      TII(Subtarget.getInstrInfo()) {

  // HACK: Until this is fully tablegen'd.
  static llvm::once_flag InitializeRegisterBankFlag;

  static auto InitializeRegisterBankOnce = [this]() {
    assert(&getRegBank(RVGPU::SGPRRegBankID) == &RVGPU::SGPRRegBank &&
           &getRegBank(RVGPU::VGPRRegBankID) == &RVGPU::VGPRRegBank &&
           &getRegBank(RVGPU::AGPRRegBankID) == &RVGPU::AGPRRegBank);
    (void)this;
  };

  llvm::call_once(InitializeRegisterBankFlag, InitializeRegisterBankOnce);
}

static bool isVectorRegisterBank(const RegisterBank &Bank) {
  unsigned BankID = Bank.getID();
  return BankID == RVGPU::VGPRRegBankID || BankID == RVGPU::AGPRRegBankID;
}

bool RVGPURegisterBankInfo::isDivergentRegBank(const RegisterBank *RB) const {
  return RB != &RVGPU::SGPRRegBank;
}

unsigned RVGPURegisterBankInfo::copyCost(const RegisterBank &Dst,
                                          const RegisterBank &Src,
                                          TypeSize Size) const {
  // TODO: Should there be a UniformVGPRRegBank which can use readfirstlane?
  if (Dst.getID() == RVGPU::SGPRRegBankID &&
      (isVectorRegisterBank(Src) || Src.getID() == RVGPU::VCCRegBankID)) {
    return std::numeric_limits<unsigned>::max();
  }

  // Bool values are tricky, because the meaning is based on context. The SCC
  // and VCC banks are for the natural scalar and vector conditions produced by
  // a compare.
  //
  // Legalization doesn't know about the necessary context, so an s1 use may
  // have been a truncate from an arbitrary value, in which case a copy (lowered
  // as a compare with 0) needs to be inserted.
  if (Size == 1 &&
      (Dst.getID() == RVGPU::SGPRRegBankID) &&
      (isVectorRegisterBank(Src) ||
       Src.getID() == RVGPU::SGPRRegBankID ||
       Src.getID() == RVGPU::VCCRegBankID))
    return std::numeric_limits<unsigned>::max();

  // There is no direct copy between AGPRs.
  if (Dst.getID() == RVGPU::AGPRRegBankID &&
      Src.getID() == RVGPU::AGPRRegBankID)
    return 4;

  return RegisterBankInfo::copyCost(Dst, Src, Size);
}

unsigned RVGPURegisterBankInfo::getBreakDownCost(
  const ValueMapping &ValMapping,
  const RegisterBank *CurBank) const {
  // Check if this is a breakdown for G_LOAD to move the pointer from SGPR to
  // VGPR.
  // FIXME: Is there a better way to do this?
  if (ValMapping.NumBreakDowns >= 2 || ValMapping.BreakDown[0].Length >= 64)
    return 10; // This is expensive.

  assert(ValMapping.NumBreakDowns == 2 &&
         ValMapping.BreakDown[0].Length == 32 &&
         ValMapping.BreakDown[0].StartIdx == 0 &&
         ValMapping.BreakDown[1].Length == 32 &&
         ValMapping.BreakDown[1].StartIdx == 32 &&
         ValMapping.BreakDown[0].RegBank == ValMapping.BreakDown[1].RegBank);

  // 32-bit extract of a 64-bit value is just access of a subregister, so free.
  // TODO: Cost of 0 hits assert, though it's not clear it's what we really
  // want.

  // TODO: 32-bit insert to a 64-bit SGPR may incur a non-free copy due to SGPR
  // alignment restrictions, but this probably isn't important.
  return 1;
}

const RegisterBank &
RVGPURegisterBankInfo::getRegBankFromRegClass(const TargetRegisterClass &RC,
                                               LLT Ty) const {
  if (&RC == &RVGPU::SReg_1RegClass)
    return RVGPU::VCCRegBank;

  // We promote real scalar booleans to SReg_32. Any SGPR using s1 is really a
  // VCC-like use.
  if (TRI->isSGPRClass(&RC)) {
    // FIXME: This probably came from a copy from a physical register, which
    // should be inferable from the copied to-type. We don't have many boolean
    // physical register constraints so just assume a normal SGPR for now.
    if (!Ty.isValid())
      return RVGPU::SGPRRegBank;

    return Ty == LLT::scalar(1) ? RVGPU::VCCRegBank : RVGPU::SGPRRegBank;
  }

  return TRI->isAGPRClass(&RC) ? RVGPU::AGPRRegBank : RVGPU::VGPRRegBank;
}

template <unsigned NumOps>
RegisterBankInfo::InstructionMappings
RVGPURegisterBankInfo::addMappingFromTable(
    const MachineInstr &MI, const MachineRegisterInfo &MRI,
    const std::array<unsigned, NumOps> RegSrcOpIdx,
    ArrayRef<OpRegBankEntry<NumOps>> Table) const {

  InstructionMappings AltMappings;

  SmallVector<const ValueMapping *, 10> Operands(MI.getNumOperands());

  unsigned Sizes[NumOps];
  for (unsigned I = 0; I < NumOps; ++I) {
    Register Reg = MI.getOperand(RegSrcOpIdx[I]).getReg();
    Sizes[I] = getSizeInBits(Reg, MRI, *TRI);
  }

  for (unsigned I = 0, E = MI.getNumExplicitDefs(); I != E; ++I) {
    unsigned SizeI = getSizeInBits(MI.getOperand(I).getReg(), MRI, *TRI);
    Operands[I] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, SizeI);
  }

  // getInstrMapping's default mapping uses ID 1, so start at 2.
  unsigned MappingID = 2;
  for (const auto &Entry : Table) {
    for (unsigned I = 0; I < NumOps; ++I) {
      int OpIdx = RegSrcOpIdx[I];
      Operands[OpIdx] = RVGPU::getValueMapping(Entry.RegBanks[I], Sizes[I]);
    }

    AltMappings.push_back(&getInstructionMapping(MappingID++, Entry.Cost,
                                                 getOperandsMapping(Operands),
                                                 Operands.size()));
  }

  return AltMappings;
}

RegisterBankInfo::InstructionMappings
RVGPURegisterBankInfo::getInstrAlternativeMappingsIntrinsic(
    const MachineInstr &MI, const MachineRegisterInfo &MRI) const {
  switch (cast<GIntrinsic>(MI).getIntrinsicID()) {
  case Intrinsic::rvgpu_readlane: {
    static const OpRegBankEntry<3> Table[2] = {
      // Perfectly legal.
      { { RVGPU::SGPRRegBankID, RVGPU::VGPRRegBankID, RVGPU::SGPRRegBankID }, 1 },

      // Need a readfirstlane for the index.
      { { RVGPU::SGPRRegBankID, RVGPU::VGPRRegBankID, RVGPU::VGPRRegBankID }, 2 }
    };

    const std::array<unsigned, 3> RegSrcOpIdx = { { 0, 2, 3 } };
    return addMappingFromTable<3>(MI, MRI, RegSrcOpIdx, Table);
  }
  case Intrinsic::rvgpu_writelane: {
    static const OpRegBankEntry<4> Table[4] = {
      // Perfectly legal.
      { { RVGPU::VGPRRegBankID, RVGPU::SGPRRegBankID, RVGPU::SGPRRegBankID, RVGPU::VGPRRegBankID }, 1 },

      // Need readfirstlane of first op
      { { RVGPU::VGPRRegBankID, RVGPU::VGPRRegBankID, RVGPU::SGPRRegBankID, RVGPU::VGPRRegBankID }, 2 },

      // Need readfirstlane of second op
      { { RVGPU::VGPRRegBankID, RVGPU::SGPRRegBankID, RVGPU::VGPRRegBankID, RVGPU::VGPRRegBankID }, 2 },

      // Need readfirstlane of both ops
      { { RVGPU::VGPRRegBankID, RVGPU::VGPRRegBankID, RVGPU::VGPRRegBankID, RVGPU::VGPRRegBankID }, 3 }
    };

    // rsrc, voffset, offset
    const std::array<unsigned, 4> RegSrcOpIdx = { { 0, 2, 3, 4 } };
    return addMappingFromTable<4>(MI, MRI, RegSrcOpIdx, Table);
  }
  default:
    return RegisterBankInfo::getInstrAlternativeMappings(MI);
  }
}

RegisterBankInfo::InstructionMappings
RVGPURegisterBankInfo::getInstrAlternativeMappingsIntrinsicWSideEffects(
    const MachineInstr &MI, const MachineRegisterInfo &MRI) const {

  switch (cast<GIntrinsic>(MI).getIntrinsicID()) {
  case Intrinsic::rvgpu_s_buffer_load: {
    static const OpRegBankEntry<2> Table[4] = {
      // Perfectly legal.
      { { RVGPU::SGPRRegBankID, RVGPU::SGPRRegBankID }, 1 },

      // Only need 1 register in loop
      { { RVGPU::SGPRRegBankID, RVGPU::VGPRRegBankID }, 300 },

      // Have to waterfall the resource.
      { { RVGPU::VGPRRegBankID, RVGPU::SGPRRegBankID }, 1000 },

      // Have to waterfall the resource, and the offset.
      { { RVGPU::VGPRRegBankID, RVGPU::VGPRRegBankID }, 1500 }
    };

    // rsrc, offset
    const std::array<unsigned, 2> RegSrcOpIdx = { { 2, 3 } };
    return addMappingFromTable<2>(MI, MRI, RegSrcOpIdx, Table);
  }
  case Intrinsic::rvgpu_ds_ordered_add:
  case Intrinsic::rvgpu_ds_ordered_swap: {
    // VGPR = M0, VGPR
    static const OpRegBankEntry<3> Table[2] = {
      // Perfectly legal.
      { { RVGPU::VGPRRegBankID, RVGPU::SGPRRegBankID, RVGPU::VGPRRegBankID  }, 1 },

      // Need a readfirstlane for m0
      { { RVGPU::VGPRRegBankID, RVGPU::VGPRRegBankID, RVGPU::VGPRRegBankID }, 2 }
    };

    const std::array<unsigned, 3> RegSrcOpIdx = { { 0, 2, 3 } };
    return addMappingFromTable<3>(MI, MRI, RegSrcOpIdx, Table);
  }
  case Intrinsic::rvgpu_s_sendmsg:
  case Intrinsic::rvgpu_s_sendmsghalt: {
    // FIXME: Should have no register for immediate
    static const OpRegBankEntry<1> Table[2] = {
      // Perfectly legal.
      { { RVGPU::SGPRRegBankID }, 1 },

      // Need readlane
      { { RVGPU::VGPRRegBankID }, 3 }
    };

    const std::array<unsigned, 1> RegSrcOpIdx = { { 2 } };
    return addMappingFromTable<1>(MI, MRI, RegSrcOpIdx, Table);
  }
  default:
    return RegisterBankInfo::getInstrAlternativeMappings(MI);
  }
}

// FIXME: Returns uniform if there's no source value information. This is
// probably wrong.
static bool isScalarLoadLegal(const MachineInstr &MI) {
  if (!MI.hasOneMemOperand())
    return false;

  const MachineMemOperand *MMO = *MI.memoperands_begin();
  const unsigned AS = MMO->getAddrSpace();
  const bool IsConst = AS == RVGPUAS::CONSTANT_ADDRESS ||
                       AS == RVGPUAS::CONSTANT_ADDRESS_32BIT;
  // Require 4-byte alignment.
  return MMO->getAlign() >= Align(4) &&
         // Can't do a scalar atomic load.
         !MMO->isAtomic() &&
         // Don't use scalar loads for volatile accesses to non-constant address
         // spaces.
         (IsConst || !MMO->isVolatile()) &&
         // Memory must be known constant, or not written before this load.
         (IsConst || MMO->isInvariant() || (MMO->getFlags() & MONoClobber)) &&
         RVGPUInstrInfo::isUniformMMO(MMO);
}

RegisterBankInfo::InstructionMappings
RVGPURegisterBankInfo::getInstrAlternativeMappings(
    const MachineInstr &MI) const {

  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();


  InstructionMappings AltMappings;
  switch (MI.getOpcode()) {
  case TargetOpcode::G_CONSTANT:
  case TargetOpcode::G_IMPLICIT_DEF: {
    unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
    if (Size == 1) {
      static const OpRegBankEntry<1> Table[3] = {
        { { RVGPU::VGPRRegBankID }, 1 },
        { { RVGPU::SGPRRegBankID }, 1 },
        { { RVGPU::VCCRegBankID }, 1 }
      };

      return addMappingFromTable<1>(MI, MRI, {{ 0 }}, Table);
    }

    [[fallthrough]];
  }
  case TargetOpcode::G_FCONSTANT:
  case TargetOpcode::G_FRAME_INDEX:
  case TargetOpcode::G_GLOBAL_VALUE: {
    static const OpRegBankEntry<1> Table[2] = {
      { { RVGPU::VGPRRegBankID }, 1 },
      { { RVGPU::SGPRRegBankID }, 1 }
    };

    return addMappingFromTable<1>(MI, MRI, {{ 0 }}, Table);
  }
  case TargetOpcode::G_AND:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR: {
    unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);

    if (Size == 1) {
      // s_{and|or|xor}_b32 set scc when the result of the 32-bit op is not 0.
      const InstructionMapping &SCCMapping = getInstructionMapping(
        1, 1, getOperandsMapping(
          {RVGPU::getValueMapping(RVGPU::SGPRRegBankID, 32),
           RVGPU::getValueMapping(RVGPU::SGPRRegBankID, 32),
           RVGPU::getValueMapping(RVGPU::SGPRRegBankID, 32)}),
        3); // Num Operands
      AltMappings.push_back(&SCCMapping);

      const InstructionMapping &VCCMapping0 = getInstructionMapping(
        2, 1, getOperandsMapping(
          {RVGPU::getValueMapping(RVGPU::VCCRegBankID, Size),
           RVGPU::getValueMapping(RVGPU::VCCRegBankID, Size),
           RVGPU::getValueMapping(RVGPU::VCCRegBankID, Size)}),
        3); // Num Operands
      AltMappings.push_back(&VCCMapping0);
      return AltMappings;
    }

    if (Size != 64)
      break;

    const InstructionMapping &SSMapping = getInstructionMapping(
      1, 1, getOperandsMapping(
        {RVGPU::getValueMapping(RVGPU::SGPRRegBankID, Size),
         RVGPU::getValueMapping(RVGPU::SGPRRegBankID, Size),
         RVGPU::getValueMapping(RVGPU::SGPRRegBankID, Size)}),
      3); // Num Operands
    AltMappings.push_back(&SSMapping);

    const InstructionMapping &VVMapping = getInstructionMapping(
      2, 2, getOperandsMapping(
        {RVGPU::getValueMappingSGPR64Only(RVGPU::VGPRRegBankID, Size),
         RVGPU::getValueMappingSGPR64Only(RVGPU::VGPRRegBankID, Size),
         RVGPU::getValueMappingSGPR64Only(RVGPU::VGPRRegBankID, Size)}),
      3); // Num Operands
    AltMappings.push_back(&VVMapping);
    break;
  }
  case TargetOpcode::G_LOAD:
  case TargetOpcode::G_ZEXTLOAD:
  case TargetOpcode::G_SEXTLOAD: {
    unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
    LLT PtrTy = MRI.getType(MI.getOperand(1).getReg());
    unsigned PtrSize = PtrTy.getSizeInBits();
    unsigned AS = PtrTy.getAddressSpace();

    if ((AS != RVGPUAS::LOCAL_ADDRESS && AS != RVGPUAS::REGION_ADDRESS &&
         AS != RVGPUAS::PRIVATE_ADDRESS) &&
        isScalarLoadLegal(MI)) {
      const InstructionMapping &SSMapping = getInstructionMapping(
          1, 1, getOperandsMapping(
                    {RVGPU::getValueMapping(RVGPU::SGPRRegBankID, Size),
                     RVGPU::getValueMapping(RVGPU::SGPRRegBankID, PtrSize)}),
          2); // Num Operands
      AltMappings.push_back(&SSMapping);
    }

    const InstructionMapping &VVMapping = getInstructionMapping(
        2, 1,
        getOperandsMapping(
            {RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size),
             RVGPU::getValueMapping(RVGPU::VGPRRegBankID, PtrSize)}),
        2); // Num Operands
    AltMappings.push_back(&VVMapping);

    // It may be possible to have a vgpr = load sgpr mapping here, because
    // the mubuf instructions support this kind of load, but probably for only
    // gfx7 and older.  However, the addressing mode matching in the instruction
    // selector should be able to do a better job of detecting and selecting
    // these kinds of loads from the vgpr = load vgpr mapping.

    return AltMappings;

  }
  case TargetOpcode::G_SELECT: {
    unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
    const InstructionMapping &SSMapping = getInstructionMapping(1, 1,
      getOperandsMapping({RVGPU::getValueMapping(RVGPU::SGPRRegBankID, Size),
                          RVGPU::getValueMapping(RVGPU::SGPRRegBankID, 1),
                          RVGPU::getValueMapping(RVGPU::SGPRRegBankID, Size),
                          RVGPU::getValueMapping(RVGPU::SGPRRegBankID, Size)}),
      4); // Num Operands
    AltMappings.push_back(&SSMapping);

    const InstructionMapping &VVMapping = getInstructionMapping(2, 1,
      getOperandsMapping({RVGPU::getValueMappingSGPR64Only(RVGPU::VGPRRegBankID, Size),
                          RVGPU::getValueMapping(RVGPU::VCCRegBankID, 1),
                          RVGPU::getValueMappingSGPR64Only(RVGPU::VGPRRegBankID, Size),
                          RVGPU::getValueMappingSGPR64Only(RVGPU::VGPRRegBankID, Size)}),
      4); // Num Operands
    AltMappings.push_back(&VVMapping);

    return AltMappings;
  }
  case TargetOpcode::G_UADDE:
  case TargetOpcode::G_USUBE:
  case TargetOpcode::G_SADDE:
  case TargetOpcode::G_SSUBE: {
    unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
    const InstructionMapping &SSMapping = getInstructionMapping(1, 1,
      getOperandsMapping(
        {RVGPU::getValueMapping(RVGPU::SGPRRegBankID, Size),
         RVGPU::getValueMapping(RVGPU::SGPRRegBankID, 1),
         RVGPU::getValueMapping(RVGPU::SGPRRegBankID, Size),
         RVGPU::getValueMapping(RVGPU::SGPRRegBankID, Size),
         RVGPU::getValueMapping(RVGPU::SGPRRegBankID, 1)}),
      5); // Num Operands
    AltMappings.push_back(&SSMapping);

    const InstructionMapping &VVMapping = getInstructionMapping(2, 1,
      getOperandsMapping({RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size),
                          RVGPU::getValueMapping(RVGPU::VCCRegBankID, 1),
                          RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size),
                          RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size),
                          RVGPU::getValueMapping(RVGPU::VCCRegBankID, 1)}),
      5); // Num Operands
    AltMappings.push_back(&VVMapping);
    return AltMappings;
  }
  case RVGPU::G_BRCOND: {
    assert(MRI.getType(MI.getOperand(0).getReg()).getSizeInBits() == 1);

    // TODO: Change type to 32 for scalar
    const InstructionMapping &SMapping = getInstructionMapping(
      1, 1, getOperandsMapping(
        {RVGPU::getValueMapping(RVGPU::SGPRRegBankID, 1), nullptr}),
      2); // Num Operands
    AltMappings.push_back(&SMapping);

    const InstructionMapping &VMapping = getInstructionMapping(
      1, 1, getOperandsMapping(
        {RVGPU::getValueMapping(RVGPU::VCCRegBankID, 1), nullptr }),
      2); // Num Operands
    AltMappings.push_back(&VMapping);
    return AltMappings;
  }
  case RVGPU::G_INTRINSIC:
  case RVGPU::G_INTRINSIC_CONVERGENT:
    return getInstrAlternativeMappingsIntrinsic(MI, MRI);
  case RVGPU::G_INTRINSIC_W_SIDE_EFFECTS:
  case RVGPU::G_INTRINSIC_CONVERGENT_W_SIDE_EFFECTS:
    return getInstrAlternativeMappingsIntrinsicWSideEffects(MI, MRI);
  default:
    break;
  }
  return RegisterBankInfo::getInstrAlternativeMappings(MI);
}

void RVGPURegisterBankInfo::split64BitValueForMapping(
  MachineIRBuilder &B,
  SmallVector<Register, 2> &Regs,
  LLT HalfTy,
  Register Reg) const {
  assert(HalfTy.getSizeInBits() == 32);
  MachineRegisterInfo *MRI = B.getMRI();
  Register LoLHS = MRI->createGenericVirtualRegister(HalfTy);
  Register HiLHS = MRI->createGenericVirtualRegister(HalfTy);
  const RegisterBank *Bank = getRegBank(Reg, *MRI, *TRI);
  MRI->setRegBank(LoLHS, *Bank);
  MRI->setRegBank(HiLHS, *Bank);

  Regs.push_back(LoLHS);
  Regs.push_back(HiLHS);

  B.buildInstr(RVGPU::G_UNMERGE_VALUES)
    .addDef(LoLHS)
    .addDef(HiLHS)
    .addUse(Reg);
}

/// Replace the current type each register in \p Regs has with \p NewTy
static void setRegsToType(MachineRegisterInfo &MRI, ArrayRef<Register> Regs,
                          LLT NewTy) {
  for (Register Reg : Regs) {
    assert(MRI.getType(Reg).getSizeInBits() == NewTy.getSizeInBits());
    MRI.setType(Reg, NewTy);
  }
}

static LLT getHalfSizedType(LLT Ty) {
  if (Ty.isVector()) {
    assert(Ty.getElementCount().isKnownMultipleOf(2));
    return LLT::scalarOrVector(Ty.getElementCount().divideCoefficientBy(2),
                               Ty.getElementType());
  }

  assert(Ty.getScalarSizeInBits() % 2 == 0);
  return LLT::scalar(Ty.getScalarSizeInBits() / 2);
}

// Build one or more V_READFIRSTLANE_B32 instructions to move the given vector
// source value into a scalar register.
Register RVGPURegisterBankInfo::buildReadFirstLane(MachineIRBuilder &B,
                                                    MachineRegisterInfo &MRI,
                                                    Register Src) const {
  LLT Ty = MRI.getType(Src);
  const RegisterBank *Bank = getRegBank(Src, MRI, *TRI);

  if (Bank == &RVGPU::SGPRRegBank)
    return Src;

  unsigned Bits = Ty.getSizeInBits();
  assert(Bits % 32 == 0);

  if (Bank != &RVGPU::VGPRRegBank) {
    // We need to copy from AGPR to VGPR
    Src = B.buildCopy(Ty, Src).getReg(0);
    MRI.setRegBank(Src, RVGPU::VGPRRegBank);
  }

  LLT S32 = LLT::scalar(32);
  unsigned NumParts = Bits / 32;
  SmallVector<Register, 8> SrcParts;
  SmallVector<Register, 8> DstParts;

  if (Bits == 32) {
    SrcParts.push_back(Src);
  } else {
    auto Unmerge = B.buildUnmerge(S32, Src);
    for (unsigned i = 0; i < NumParts; ++i)
      SrcParts.push_back(Unmerge.getReg(i));
  }

  for (unsigned i = 0; i < NumParts; ++i) {
    Register SrcPart = SrcParts[i];
    Register DstPart = MRI.createVirtualRegister(&RVGPU::SReg_32RegClass);
    MRI.setType(DstPart, NumParts == 1 ? Ty : S32);

    const TargetRegisterClass *Constrained =
        constrainGenericRegister(SrcPart, RVGPU::VGPR_32RegClass, MRI);
    (void)Constrained;
    assert(Constrained && "Failed to constrain readfirstlane src reg");

    B.buildInstr(RVGPU::V_READFIRSTLANE_B32, {DstPart}, {SrcPart});

    DstParts.push_back(DstPart);
  }

  if (Bits == 32)
    return DstParts[0];

  Register Dst = B.buildMergeLikeInstr(Ty, DstParts).getReg(0);
  MRI.setRegBank(Dst, RVGPU::SGPRRegBank);
  return Dst;
}

/// Legalize instruction \p MI where operands in \p OpIndices must be SGPRs. If
/// any of the required SGPR operands are VGPRs, perform a waterfall loop to
/// execute the instruction for each unique combination of values in all lanes
/// in the wave. The block will be split such that rest of the instructions are
/// moved to a new block.
///
/// Essentially performs this loop:
//
/// Save Execution Mask
/// For (Lane : Wavefront) {
///   Enable Lane, Disable all other lanes
///   SGPR = read SGPR value for current lane from VGPR
///   VGPRResult[Lane] = use_op SGPR
/// }
/// Restore Execution Mask
///
/// There is additional complexity to try for compare values to identify the
/// unique values used.
bool RVGPURegisterBankInfo::executeInWaterfallLoop(
    MachineIRBuilder &B, iterator_range<MachineBasicBlock::iterator> Range,
    SmallSet<Register, 4> &SGPROperandRegs) const {
  // Track use registers which have already been expanded with a readfirstlane
  // sequence. This may have multiple uses if moving a sequence.
  DenseMap<Register, Register> WaterfalledRegMap;

  MachineBasicBlock &MBB = B.getMBB();
  MachineFunction *MF = &B.getMF();

  const TargetRegisterClass *WaveRC = TRI->getWaveMaskRegClass();
  const unsigned MovExecOpc =
      Subtarget.isWave32() ? RVGPU::S_MOV_B32 : RVGPU::S_MOV_B64;
  const unsigned MovExecTermOpc =
      Subtarget.isWave32() ? RVGPU::S_MOV_B32_term : RVGPU::S_MOV_B64_term;

  const unsigned XorTermOpc = Subtarget.isWave32() ?
    RVGPU::S_XOR_B32_term : RVGPU::S_XOR_B64_term;
  const unsigned AndSaveExecOpc =  Subtarget.isWave32() ?
    RVGPU::S_AND_SAVEEXEC_B32 : RVGPU::S_AND_SAVEEXEC_B64;
  const unsigned ExecReg =  Subtarget.isWave32() ?
    RVGPU::EXEC_LO : RVGPU::EXEC;

#ifndef NDEBUG
  const int OrigRangeSize = std::distance(Range.begin(), Range.end());
#endif

  MachineRegisterInfo &MRI = *B.getMRI();
  Register SaveExecReg = MRI.createVirtualRegister(WaveRC);
  Register InitSaveExecReg = MRI.createVirtualRegister(WaveRC);

  // Don't bother using generic instructions/registers for the exec mask.
  B.buildInstr(TargetOpcode::IMPLICIT_DEF)
    .addDef(InitSaveExecReg);

  Register PhiExec = MRI.createVirtualRegister(WaveRC);
  Register NewExec = MRI.createVirtualRegister(WaveRC);

  // To insert the loop we need to split the block. Move everything before this
  // point to a new block, and insert a new empty block before this instruction.
  MachineBasicBlock *LoopBB = MF->CreateMachineBasicBlock();
  MachineBasicBlock *BodyBB = MF->CreateMachineBasicBlock();
  MachineBasicBlock *RemainderBB = MF->CreateMachineBasicBlock();
  MachineBasicBlock *RestoreExecBB = MF->CreateMachineBasicBlock();
  MachineFunction::iterator MBBI(MBB);
  ++MBBI;
  MF->insert(MBBI, LoopBB);
  MF->insert(MBBI, BodyBB);
  MF->insert(MBBI, RestoreExecBB);
  MF->insert(MBBI, RemainderBB);

  LoopBB->addSuccessor(BodyBB);
  BodyBB->addSuccessor(RestoreExecBB);
  BodyBB->addSuccessor(LoopBB);

  // Move the rest of the block into a new block.
  RemainderBB->transferSuccessorsAndUpdatePHIs(&MBB);
  RemainderBB->splice(RemainderBB->begin(), &MBB, Range.end(), MBB.end());

  MBB.addSuccessor(LoopBB);
  RestoreExecBB->addSuccessor(RemainderBB);

  B.setInsertPt(*LoopBB, LoopBB->end());

  B.buildInstr(TargetOpcode::PHI)
      .addDef(PhiExec)
      .addReg(InitSaveExecReg)
      .addMBB(&MBB)
      .addReg(NewExec)
      .addMBB(BodyBB);

  const DebugLoc &DL = B.getDL();

  MachineInstr &FirstInst = *Range.begin();

  // Move the instruction into the loop body. Note we moved everything after
  // Range.end() already into a new block, so Range.end() is no longer valid.
  BodyBB->splice(BodyBB->end(), &MBB, Range.begin(), MBB.end());

  // Figure out the iterator range after splicing the instructions.
  MachineBasicBlock::iterator NewBegin = FirstInst.getIterator();
  auto NewEnd = BodyBB->end();

  B.setMBB(*LoopBB);

  LLT S1 = LLT::scalar(1);
  Register CondReg;

  assert(std::distance(NewBegin, NewEnd) == OrigRangeSize);

  for (MachineInstr &MI : make_range(NewBegin, NewEnd)) {
    for (MachineOperand &Op : MI.all_uses()) {
      Register OldReg = Op.getReg();
      if (!SGPROperandRegs.count(OldReg))
        continue;

      // See if we already processed this register in another instruction in the
      // sequence.
      auto OldVal = WaterfalledRegMap.find(OldReg);
      if (OldVal != WaterfalledRegMap.end()) {
        Op.setReg(OldVal->second);
        continue;
      }

      Register OpReg = Op.getReg();
      LLT OpTy = MRI.getType(OpReg);

      const RegisterBank *OpBank = getRegBank(OpReg, MRI, *TRI);
      if (OpBank != &RVGPU::VGPRRegBank) {
        // Insert copy from AGPR to VGPR before the loop.
        B.setMBB(MBB);
        OpReg = B.buildCopy(OpTy, OpReg).getReg(0);
        MRI.setRegBank(OpReg, RVGPU::VGPRRegBank);
        B.setMBB(*LoopBB);
      }

      Register CurrentLaneReg = buildReadFirstLane(B, MRI, OpReg);

      // Build the comparison(s).
      unsigned OpSize = OpTy.getSizeInBits();
      bool Is64 = OpSize % 64 == 0;
      unsigned PartSize = Is64 ? 64 : 32;
      LLT PartTy = LLT::scalar(PartSize);
      unsigned NumParts = OpSize / PartSize;
      SmallVector<Register, 8> OpParts;
      SmallVector<Register, 8> CurrentLaneParts;

      if (NumParts == 1) {
        OpParts.push_back(OpReg);
        CurrentLaneParts.push_back(CurrentLaneReg);
      } else {
        auto UnmergeOp = B.buildUnmerge(PartTy, OpReg);
        auto UnmergeCurrentLane = B.buildUnmerge(PartTy, CurrentLaneReg);
        for (unsigned i = 0; i < NumParts; ++i) {
          OpParts.push_back(UnmergeOp.getReg(i));
          CurrentLaneParts.push_back(UnmergeCurrentLane.getReg(i));
          MRI.setRegBank(OpParts[i], RVGPU::VGPRRegBank);
          MRI.setRegBank(CurrentLaneParts[i], RVGPU::SGPRRegBank);
        }
      }

      for (unsigned i = 0; i < NumParts; ++i) {
        auto CmpReg = B.buildICmp(CmpInst::ICMP_EQ, S1, CurrentLaneParts[i],
                                  OpParts[i]).getReg(0);
        MRI.setRegBank(CmpReg, RVGPU::VCCRegBank);

        if (!CondReg) {
          CondReg = CmpReg;
        } else {
          CondReg = B.buildAnd(S1, CondReg, CmpReg).getReg(0);
          MRI.setRegBank(CondReg, RVGPU::VCCRegBank);
        }
      }

      Op.setReg(CurrentLaneReg);

      // Make sure we don't re-process this register again.
      WaterfalledRegMap.insert(std::pair(OldReg, Op.getReg()));
    }
  }

  // The ballot becomes a no-op during instruction selection.
  CondReg = B.buildIntrinsic(Intrinsic::rvgpu_ballot,
                             {LLT::scalar(Subtarget.isWave32() ? 32 : 64)})
                .addReg(CondReg)
                .getReg(0);
  MRI.setRegClass(CondReg, WaveRC);

  // Update EXEC, save the original EXEC value to VCC.
  B.buildInstr(AndSaveExecOpc)
    .addDef(NewExec)
    .addReg(CondReg, RegState::Kill);

  MRI.setSimpleHint(NewExec, CondReg);

  B.setInsertPt(*BodyBB, BodyBB->end());

  // Update EXEC, switch all done bits to 0 and all todo bits to 1.
  B.buildInstr(XorTermOpc)
    .addDef(ExecReg)
    .addReg(ExecReg)
    .addReg(NewExec);

  // XXX - s_xor_b64 sets scc to 1 if the result is nonzero, so can we use
  // s_cbranch_scc0?

  // Loop back to V_READFIRSTLANE_B32 if there are still variants to cover.
  B.buildInstr(RVGPU::SI_WATERFALL_LOOP).addMBB(LoopBB);

  // Save the EXEC mask before the loop.
  BuildMI(MBB, MBB.end(), DL, TII->get(MovExecOpc), SaveExecReg)
    .addReg(ExecReg);

  // Restore the EXEC mask after the loop.
  B.setMBB(*RestoreExecBB);
  B.buildInstr(MovExecTermOpc)
    .addDef(ExecReg)
    .addReg(SaveExecReg);

  // Set the insert point after the original instruction, so any new
  // instructions will be in the remainder.
  B.setInsertPt(*RemainderBB, RemainderBB->begin());

  return true;
}

// Return any unique registers used by \p MI at \p OpIndices that need to be
// handled in a waterfall loop. Returns these registers in \p
// SGPROperandRegs. Returns true if there are any operands to handle and a
// waterfall loop is necessary.
bool RVGPURegisterBankInfo::collectWaterfallOperands(
  SmallSet<Register, 4> &SGPROperandRegs, MachineInstr &MI,
  MachineRegisterInfo &MRI, ArrayRef<unsigned> OpIndices) const {
  for (unsigned Op : OpIndices) {
    assert(MI.getOperand(Op).isUse());
    Register Reg = MI.getOperand(Op).getReg();
    const RegisterBank *OpBank = getRegBank(Reg, MRI, *TRI);
    if (OpBank->getID() != RVGPU::SGPRRegBankID)
      SGPROperandRegs.insert(Reg);
  }

  // No operands need to be replaced, so no need to loop.
  return !SGPROperandRegs.empty();
}

bool RVGPURegisterBankInfo::executeInWaterfallLoop(
    MachineIRBuilder &B, MachineInstr &MI, ArrayRef<unsigned> OpIndices) const {
  // Use a set to avoid extra readfirstlanes in the case where multiple operands
  // are the same register.
  SmallSet<Register, 4> SGPROperandRegs;

  if (!collectWaterfallOperands(SGPROperandRegs, MI, *B.getMRI(), OpIndices))
    return false;

  MachineBasicBlock::iterator I = MI.getIterator();
  return executeInWaterfallLoop(B, make_range(I, std::next(I)),
                                SGPROperandRegs);
}

// Legalize an operand that must be an SGPR by inserting a readfirstlane.
void RVGPURegisterBankInfo::constrainOpWithReadfirstlane(
    MachineIRBuilder &B, MachineInstr &MI, unsigned OpIdx) const {
  Register Reg = MI.getOperand(OpIdx).getReg();
  MachineRegisterInfo &MRI = *B.getMRI();
  const RegisterBank *Bank = getRegBank(Reg, MRI, *TRI);
  if (Bank == &RVGPU::SGPRRegBank)
    return;

  Reg = buildReadFirstLane(B, MRI, Reg);
  MI.getOperand(OpIdx).setReg(Reg);
}

/// Split \p Ty into 2 pieces. The first will have \p FirstSize bits, and the
/// rest will be in the remainder.
static std::pair<LLT, LLT> splitUnequalType(LLT Ty, unsigned FirstSize) {
  unsigned TotalSize = Ty.getSizeInBits();
  if (!Ty.isVector())
    return {LLT::scalar(FirstSize), LLT::scalar(TotalSize - FirstSize)};

  LLT EltTy = Ty.getElementType();
  unsigned EltSize = EltTy.getSizeInBits();
  assert(FirstSize % EltSize == 0);

  unsigned FirstPartNumElts = FirstSize / EltSize;
  unsigned RemainderElts = (TotalSize - FirstSize) / EltSize;

  return {LLT::scalarOrVector(ElementCount::getFixed(FirstPartNumElts), EltTy),
          LLT::scalarOrVector(ElementCount::getFixed(RemainderElts), EltTy)};
}

static LLT widen96To128(LLT Ty) {
  if (!Ty.isVector())
    return LLT::scalar(128);

  LLT EltTy = Ty.getElementType();
  assert(128 % EltTy.getSizeInBits() == 0);
  return LLT::fixed_vector(128 / EltTy.getSizeInBits(), EltTy);
}

bool RVGPURegisterBankInfo::applyMappingLoad(
    MachineIRBuilder &B,
    const RVGPURegisterBankInfo::OperandsMapper &OpdMapper,
    MachineInstr &MI) const {
  MachineRegisterInfo &MRI = *B.getMRI();
  Register DstReg = MI.getOperand(0).getReg();
  const LLT LoadTy = MRI.getType(DstReg);
  unsigned LoadSize = LoadTy.getSizeInBits();
  const unsigned MaxNonSmrdLoadSize = 128;

  const RegisterBank *DstBank =
      OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
  if (DstBank == &RVGPU::SGPRRegBank) {
    // There are some special cases that we need to look at for 32 bit and 96
    // bit SGPR loads otherwise we have nothing to do.
    if (LoadSize != 32 && (LoadSize != 96 || Subtarget.hasScalarDwordx3Loads()))
      return false;

    MachineMemOperand *MMO = *MI.memoperands_begin();
    const unsigned MemSize = 8 * MMO->getSize();
    // Scalar loads of size 8 or 16 bit with proper alignment may be widened to
    // 32 bit. Check to see if we need to widen the memory access, 8 or 16 bit
    // scalar loads should have a load size of 32 but memory access size of less
    // than 32.
    if (LoadSize == 32 &&
        (MemSize == 32 || LoadTy.isVector() || !isScalarLoadLegal(MI)))
      return false;

    Register PtrReg = MI.getOperand(1).getReg();

    ApplyRegBankMapping ApplyBank(B, *this, MRI, DstBank);

    if (LoadSize == 32) {
      // This is an extending load from a sub-dword size. Widen the memory
      // access size to 4 bytes and clear the extra high bits appropriately
      const LLT S32 = LLT::scalar(32);
      if (MI.getOpcode() == RVGPU::G_SEXTLOAD) {
        // Must extend the sign bit into higher bits for a G_SEXTLOAD
        auto WideLoad = B.buildLoadFromOffset(S32, PtrReg, *MMO, 0);
        B.buildSExtInReg(MI.getOperand(0), WideLoad, MemSize);
      } else if (MI.getOpcode() == RVGPU::G_ZEXTLOAD) {
        // Must extend zero into higher bits with an AND for a G_ZEXTLOAD
        auto WideLoad = B.buildLoadFromOffset(S32, PtrReg, *MMO, 0);
        B.buildZExtInReg(MI.getOperand(0), WideLoad, MemSize);
      } else
        // We do not need to touch the higher bits for regular loads.
        B.buildLoadFromOffset(MI.getOperand(0), PtrReg, *MMO, 0);
    } else {
      // 96-bit loads are only available for vector loads. We need to split this
      // into a 64-bit part, and 32 (unless we can widen to a 128-bit load).
      if (MMO->getAlign() < Align(16)) {
        LegalizerHelper Helper(B.getMF(), ApplyBank, B);
        LLT Part64, Part32;
        std::tie(Part64, Part32) = splitUnequalType(LoadTy, 64);
        if (Helper.reduceLoadStoreWidth(cast<GAnyLoad>(MI), 0, Part64) !=
            LegalizerHelper::Legalized)
          return false;
        return true;
      } else {
        LLT WiderTy = widen96To128(LoadTy);
        auto WideLoad = B.buildLoadFromOffset(WiderTy, PtrReg, *MMO, 0);
        if (WiderTy.isScalar())
          B.buildTrunc(MI.getOperand(0), WideLoad);
        else {
          B.buildDeleteTrailingVectorElements(MI.getOperand(0).getReg(),
                                              WideLoad);
        }
      }
    }

    MI.eraseFromParent();
    return true;
  }

  // 128-bit loads are supported for all instruction types.
  if (LoadSize <= MaxNonSmrdLoadSize)
    return false;

  SmallVector<Register, 16> DefRegs(OpdMapper.getVRegs(0));
  SmallVector<Register, 1> SrcRegs(OpdMapper.getVRegs(1));

  if (SrcRegs.empty())
    SrcRegs.push_back(MI.getOperand(1).getReg());

  assert(LoadSize % MaxNonSmrdLoadSize == 0);

  // RegBankSelect only emits scalar types, so we need to reset the pointer
  // operand to a pointer type.
  Register BasePtrReg = SrcRegs[0];
  LLT PtrTy = MRI.getType(MI.getOperand(1).getReg());
  MRI.setType(BasePtrReg, PtrTy);

  unsigned NumSplitParts = LoadTy.getSizeInBits() / MaxNonSmrdLoadSize;
  const LLT LoadSplitTy = LoadTy.divide(NumSplitParts);
  ApplyRegBankMapping O(B, *this, MRI, &RVGPU::VGPRRegBank);
  LegalizerHelper Helper(B.getMF(), O, B);

  if (LoadTy.isVector()) {
    if (Helper.fewerElementsVector(MI, 0, LoadSplitTy) != LegalizerHelper::Legalized)
      return false;
  } else {
    if (Helper.narrowScalar(MI, 0, LoadSplitTy) != LegalizerHelper::Legalized)
      return false;
  }

  MRI.setRegBank(DstReg, RVGPU::VGPRRegBank);
  return true;
}

bool RVGPURegisterBankInfo::applyMappingDynStackAlloc(
    MachineIRBuilder &B,
    const RVGPURegisterBankInfo::OperandsMapper &OpdMapper,
    MachineInstr &MI) const {
  MachineRegisterInfo &MRI = *B.getMRI();
  const MachineFunction &MF = B.getMF();
  const RVSubtarget &ST = MF.getSubtarget<RVSubtarget>();
  const auto &TFI = *ST.getFrameLowering();

  // Guard in case the stack growth direction ever changes with scratch
  // instructions.
  if (TFI.getStackGrowthDirection() == TargetFrameLowering::StackGrowsDown)
    return false;

  Register Dst = MI.getOperand(0).getReg();
  Register AllocSize = MI.getOperand(1).getReg();
  Align Alignment = assumeAligned(MI.getOperand(2).getImm());

  const RegisterBank *SizeBank = getRegBank(AllocSize, MRI, *TRI);

  // TODO: Need to emit a wave reduction to get the maximum size.
  if (SizeBank != &RVGPU::SGPRRegBank)
    return false;

  LLT PtrTy = MRI.getType(Dst);
  LLT IntPtrTy = LLT::scalar(PtrTy.getSizeInBits());

  const RVMachineFunctionInfo *Info = MF.getInfo<RVMachineFunctionInfo>();
  Register SPReg = Info->getStackPtrOffsetReg();
  ApplyRegBankMapping ApplyBank(B, *this, MRI, &RVGPU::SGPRRegBank);

  auto WaveSize = B.buildConstant(LLT::scalar(32), ST.getWavefrontSizeLog2());
  auto ScaledSize = B.buildShl(IntPtrTy, AllocSize, WaveSize);

  auto SPCopy = B.buildCopy(PtrTy, SPReg);
  if (Alignment > TFI.getStackAlign()) {
    auto PtrAdd = B.buildPtrAdd(PtrTy, SPCopy, ScaledSize);
    B.buildMaskLowPtrBits(Dst, PtrAdd,
                          Log2(Alignment) + ST.getWavefrontSizeLog2());
  } else {
    B.buildPtrAdd(Dst, SPCopy, ScaledSize);
  }

  MI.eraseFromParent();
  return true;
}

bool RVGPURegisterBankInfo::applyMappingImage(
    MachineIRBuilder &B, MachineInstr &MI,
    const RVGPURegisterBankInfo::OperandsMapper &OpdMapper,
    int RsrcIdx) const {
  const int NumDefs = MI.getNumExplicitDefs();

  // The reported argument index is relative to the IR intrinsic call arguments,
  // so we need to shift by the number of defs and the intrinsic ID.
  RsrcIdx += NumDefs + 1;

  // Insert copies to VGPR arguments.
  applyDefaultMapping(OpdMapper);

  // Fixup any SGPR arguments.
  SmallVector<unsigned, 4> SGPRIndexes;
  for (int I = NumDefs, NumOps = MI.getNumOperands(); I != NumOps; ++I) {
    if (!MI.getOperand(I).isReg())
      continue;

    // If this intrinsic has a sampler, it immediately follows rsrc.
    if (I == RsrcIdx || I == RsrcIdx + 1)
      SGPRIndexes.push_back(I);
  }

  executeInWaterfallLoop(B, MI, SGPRIndexes);
  return true;
}

// Analyze a combined offset from an llvm.rvgpu.s.buffer intrinsic and store
// the three offsets (voffset, soffset and instoffset)
unsigned RVGPURegisterBankInfo::setBufferOffsets(
    MachineIRBuilder &B, Register CombinedOffset, Register &VOffsetReg,
    Register &SOffsetReg, int64_t &InstOffsetVal, Align Alignment) const {
  const LLT S32 = LLT::scalar(32);
  MachineRegisterInfo *MRI = B.getMRI();

  if (std::optional<int64_t> Imm =
          getIConstantVRegSExtVal(CombinedOffset, *MRI)) {
    uint32_t SOffset, ImmOffset;
    if (TII->splitMUBUFOffset(*Imm, SOffset, ImmOffset, Alignment)) {
      VOffsetReg = B.buildConstant(S32, 0).getReg(0);
      SOffsetReg = B.buildConstant(S32, SOffset).getReg(0);
      InstOffsetVal = ImmOffset;

      B.getMRI()->setRegBank(VOffsetReg, RVGPU::VGPRRegBank);
      B.getMRI()->setRegBank(SOffsetReg, RVGPU::SGPRRegBank);
      return SOffset + ImmOffset;
    }
  }

  Register Base;
  unsigned Offset;

  std::tie(Base, Offset) =
      RVGPU::getBaseWithConstantOffset(*MRI, CombinedOffset);

  uint32_t SOffset, ImmOffset;
  if ((int)Offset > 0 &&
      TII->splitMUBUFOffset(Offset, SOffset, ImmOffset, Alignment)) {
    if (getRegBank(Base, *MRI, *TRI) == &RVGPU::VGPRRegBank) {
      VOffsetReg = Base;
      SOffsetReg = B.buildConstant(S32, SOffset).getReg(0);
      B.getMRI()->setRegBank(SOffsetReg, RVGPU::SGPRRegBank);
      InstOffsetVal = ImmOffset;
      return 0; // XXX - Why is this 0?
    }

    // If we have SGPR base, we can use it for soffset.
    if (SOffset == 0) {
      VOffsetReg = B.buildConstant(S32, 0).getReg(0);
      B.getMRI()->setRegBank(VOffsetReg, RVGPU::VGPRRegBank);
      SOffsetReg = Base;
      InstOffsetVal = ImmOffset;
      return 0; // XXX - Why is this 0?
    }
  }

  // Handle the variable sgpr + vgpr case.
  MachineInstr *Add = getOpcodeDef(RVGPU::G_ADD, CombinedOffset, *MRI);
  if (Add && (int)Offset >= 0) {
    Register Src0 = getSrcRegIgnoringCopies(Add->getOperand(1).getReg(), *MRI);
    Register Src1 = getSrcRegIgnoringCopies(Add->getOperand(2).getReg(), *MRI);

    const RegisterBank *Src0Bank = getRegBank(Src0, *MRI, *TRI);
    const RegisterBank *Src1Bank = getRegBank(Src1, *MRI, *TRI);

    if (Src0Bank == &RVGPU::VGPRRegBank && Src1Bank == &RVGPU::SGPRRegBank) {
      VOffsetReg = Src0;
      SOffsetReg = Src1;
      return 0;
    }

    if (Src0Bank == &RVGPU::SGPRRegBank && Src1Bank == &RVGPU::VGPRRegBank) {
      VOffsetReg = Src1;
      SOffsetReg = Src0;
      return 0;
    }
  }

  // Ensure we have a VGPR for the combined offset. This could be an issue if we
  // have an SGPR offset and a VGPR resource.
  if (getRegBank(CombinedOffset, *MRI, *TRI) == &RVGPU::VGPRRegBank) {
    VOffsetReg = CombinedOffset;
  } else {
    VOffsetReg = B.buildCopy(S32, CombinedOffset).getReg(0);
    B.getMRI()->setRegBank(VOffsetReg, RVGPU::VGPRRegBank);
  }

  SOffsetReg = B.buildConstant(S32, 0).getReg(0);
  B.getMRI()->setRegBank(SOffsetReg, RVGPU::SGPRRegBank);
  return 0;
}

bool RVGPURegisterBankInfo::applyMappingSBufferLoad(
    MachineIRBuilder &B, const OperandsMapper &OpdMapper) const {
  MachineInstr &MI = OpdMapper.getMI();
  MachineRegisterInfo &MRI = OpdMapper.getMRI();

  const LLT S32 = LLT::scalar(32);
  Register Dst = MI.getOperand(0).getReg();
  LLT Ty = MRI.getType(Dst);

  const RegisterBank *RSrcBank =
    OpdMapper.getInstrMapping().getOperandMapping(1).BreakDown[0].RegBank;
  const RegisterBank *OffsetBank =
    OpdMapper.getInstrMapping().getOperandMapping(2).BreakDown[0].RegBank;
  if (RSrcBank == &RVGPU::SGPRRegBank &&
      OffsetBank == &RVGPU::SGPRRegBank)
    return true; // Legal mapping

  // FIXME: 96-bit case was widened during legalize. We need to narrow it back
  // here but don't have an MMO.

  unsigned LoadSize = Ty.getSizeInBits();
  int NumLoads = 1;
  if (LoadSize == 256 || LoadSize == 512) {
    NumLoads = LoadSize / 128;
    Ty = Ty.divide(NumLoads);
  }

  // Use the alignment to ensure that the required offsets will fit into the
  // immediate offsets.
  const Align Alignment = NumLoads > 1 ? Align(16 * NumLoads) : Align(1);

  MachineFunction &MF = B.getMF();

  Register SOffset;
  Register VOffset;
  int64_t ImmOffset = 0;

  unsigned MMOOffset = setBufferOffsets(B, MI.getOperand(2).getReg(), VOffset,
                                        SOffset, ImmOffset, Alignment);

  // TODO: 96-bit loads were widened to 128-bit results. Shrink the result if we
  // can, but we need to track an MMO for that.
  const unsigned MemSize = (Ty.getSizeInBits() + 7) / 8;
  const Align MemAlign(4); // FIXME: ABI type alignment?
  MachineMemOperand *BaseMMO = MF.getMachineMemOperand(
    MachinePointerInfo(),
    MachineMemOperand::MOLoad | MachineMemOperand::MODereferenceable |
    MachineMemOperand::MOInvariant,
    MemSize, MemAlign);
  if (MMOOffset != 0)
    BaseMMO = MF.getMachineMemOperand(BaseMMO, MMOOffset, MemSize);

  // If only the offset is divergent, emit a MUBUF buffer load instead. We can
  // assume that the buffer is unswizzled.

  Register RSrc = MI.getOperand(1).getReg();
  Register VIndex = B.buildConstant(S32, 0).getReg(0);
  B.getMRI()->setRegBank(VIndex, RVGPU::VGPRRegBank);

  SmallVector<Register, 4> LoadParts(NumLoads);

  MachineBasicBlock::iterator MII = MI.getIterator();
  MachineInstrSpan Span(MII, &B.getMBB());

  for (int i = 0; i < NumLoads; ++i) {
    if (NumLoads == 1) {
      LoadParts[i] = Dst;
    } else {
      LoadParts[i] = MRI.createGenericVirtualRegister(Ty);
      MRI.setRegBank(LoadParts[i], RVGPU::VGPRRegBank);
    }

    MachineMemOperand *MMO = BaseMMO;
    if (i != 0)
      BaseMMO = MF.getMachineMemOperand(BaseMMO, MMOOffset + 16 * i, MemSize);

    B.buildInstr(RVGPU::G_RVGPU_BUFFER_LOAD)
      .addDef(LoadParts[i])       // vdata
      .addUse(RSrc)               // rsrc
      .addUse(VIndex)             // vindex
      .addUse(VOffset)            // voffset
      .addUse(SOffset)            // soffset
      .addImm(ImmOffset + 16 * i) // offset(imm)
      .addImm(0)                  // cachepolicy, swizzled buffer(imm)
      .addImm(0)                  // idxen(imm)
      .addMemOperand(MMO);
  }

  // TODO: If only the resource is a VGPR, it may be better to execute the
  // scalar load in the waterfall loop if the resource is expected to frequently
  // be dynamically uniform.
  if (RSrcBank != &RVGPU::SGPRRegBank) {
    // Remove the original instruction to avoid potentially confusing the
    // waterfall loop logic.
    B.setInstr(*Span.begin());
    MI.eraseFromParent();

    SmallSet<Register, 4> OpsToWaterfall;

    OpsToWaterfall.insert(RSrc);
    executeInWaterfallLoop(B, make_range(Span.begin(), Span.end()),
                           OpsToWaterfall);
  }

  if (NumLoads != 1) {
    if (Ty.isVector())
      B.buildConcatVectors(Dst, LoadParts);
    else
      B.buildMergeLikeInstr(Dst, LoadParts);
  }

  // We removed the instruction earlier with a waterfall loop.
  if (RSrcBank == &RVGPU::SGPRRegBank)
    MI.eraseFromParent();

  return true;
}

bool RVGPURegisterBankInfo::applyMappingBFE(MachineIRBuilder &B,
                                             const OperandsMapper &OpdMapper,
                                             bool Signed) const {
  MachineInstr &MI = OpdMapper.getMI();
  MachineRegisterInfo &MRI = OpdMapper.getMRI();

  // Insert basic copies
  applyDefaultMapping(OpdMapper);

  Register DstReg = MI.getOperand(0).getReg();
  LLT Ty = MRI.getType(DstReg);

  const LLT S32 = LLT::scalar(32);

  unsigned FirstOpnd = isa<GIntrinsic>(MI) ? 2 : 1;
  Register SrcReg = MI.getOperand(FirstOpnd).getReg();
  Register OffsetReg = MI.getOperand(FirstOpnd + 1).getReg();
  Register WidthReg = MI.getOperand(FirstOpnd + 2).getReg();

  const RegisterBank *DstBank =
    OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
  if (DstBank == &RVGPU::VGPRRegBank) {
    if (Ty == S32)
      return true;

    // There is no 64-bit vgpr bitfield extract instructions so the operation
    // is expanded to a sequence of instructions that implement the operation.
    ApplyRegBankMapping ApplyBank(B, *this, MRI, &RVGPU::VGPRRegBank);

    const LLT S64 = LLT::scalar(64);
    // Shift the source operand so that extracted bits start at bit 0.
    auto ShiftOffset = Signed ? B.buildAShr(S64, SrcReg, OffsetReg)
                              : B.buildLShr(S64, SrcReg, OffsetReg);
    auto UnmergeSOffset = B.buildUnmerge({S32, S32}, ShiftOffset);

    // A 64-bit bitfield extract uses the 32-bit bitfield extract instructions
    // if the width is a constant.
    if (auto ConstWidth = getIConstantVRegValWithLookThrough(WidthReg, MRI)) {
      // Use the 32-bit bitfield extract instruction if the width is a constant.
      // Depending on the width size, use either the low or high 32-bits.
      auto Zero = B.buildConstant(S32, 0);
      auto WidthImm = ConstWidth->Value.getZExtValue();
      if (WidthImm <= 32) {
        // Use bitfield extract on the lower 32-bit source, and then sign-extend
        // or clear the upper 32-bits.
        auto Extract =
            Signed ? B.buildSbfx(S32, UnmergeSOffset.getReg(0), Zero, WidthReg)
                   : B.buildUbfx(S32, UnmergeSOffset.getReg(0), Zero, WidthReg);
        auto Extend =
            Signed ? B.buildAShr(S32, Extract, B.buildConstant(S32, 31)) : Zero;
        B.buildMergeLikeInstr(DstReg, {Extract, Extend});
      } else {
        // Use bitfield extract on upper 32-bit source, and combine with lower
        // 32-bit source.
        auto UpperWidth = B.buildConstant(S32, WidthImm - 32);
        auto Extract =
            Signed
                ? B.buildSbfx(S32, UnmergeSOffset.getReg(1), Zero, UpperWidth)
                : B.buildUbfx(S32, UnmergeSOffset.getReg(1), Zero, UpperWidth);
        B.buildMergeLikeInstr(DstReg, {UnmergeSOffset.getReg(0), Extract});
      }
      MI.eraseFromParent();
      return true;
    }

    // Expand to Src >> Offset << (64 - Width) >> (64 - Width) using 64-bit
    // operations.
    auto ExtShift = B.buildSub(S32, B.buildConstant(S32, 64), WidthReg);
    auto SignBit = B.buildShl(S64, ShiftOffset, ExtShift);
    if (Signed)
      B.buildAShr(S64, SignBit, ExtShift);
    else
      B.buildLShr(S64, SignBit, ExtShift);
    MI.eraseFromParent();
    return true;
  }

  // The scalar form packs the offset and width in a single operand.

  ApplyRegBankMapping ApplyBank(B, *this, MRI, &RVGPU::SGPRRegBank);

  // Ensure the high bits are clear to insert the offset.
  auto OffsetMask = B.buildConstant(S32, maskTrailingOnes<unsigned>(6));
  auto ClampOffset = B.buildAnd(S32, OffsetReg, OffsetMask);

  // Zeros out the low bits, so don't bother clamping the input value.
  auto ShiftWidth = B.buildShl(S32, WidthReg, B.buildConstant(S32, 16));

  // Transformation function, pack the offset and width of a BFE into
  // the format expected by the S_BFE_I32 / S_BFE_U32. In the second
  // source, bits [5:0] contain the offset and bits [22:16] the width.
  auto MergedInputs = B.buildOr(S32, ClampOffset, ShiftWidth);

  // TODO: It might be worth using a pseudo here to avoid scc clobber and
  // register class constraints.
  unsigned Opc = Ty == S32 ? (Signed ? RVGPU::S_BFE_I32 : RVGPU::S_BFE_U32) :
                             (Signed ? RVGPU::S_BFE_I64 : RVGPU::S_BFE_U64);

  auto MIB = B.buildInstr(Opc, {DstReg}, {SrcReg, MergedInputs});
  if (!constrainSelectedInstRegOperands(*MIB, *TII, *TRI, *this))
    llvm_unreachable("failed to constrain BFE");

  MI.eraseFromParent();
  return true;
}

bool RVGPURegisterBankInfo::applyMappingMAD_64_32(
    MachineIRBuilder &B, const OperandsMapper &OpdMapper) const {
  MachineInstr &MI = OpdMapper.getMI();
  MachineRegisterInfo &MRI = OpdMapper.getMRI();

  // Insert basic copies.
  applyDefaultMapping(OpdMapper);

  Register Dst0 = MI.getOperand(0).getReg();
  Register Dst1 = MI.getOperand(1).getReg();
  Register Src0 = MI.getOperand(2).getReg();
  Register Src1 = MI.getOperand(3).getReg();
  Register Src2 = MI.getOperand(4).getReg();

  if (MRI.getRegBankOrNull(Src0) == &RVGPU::VGPRRegBank)
    return true;

  bool IsUnsigned = MI.getOpcode() == RVGPU::G_RVGPU_MAD_U64_U32;
  LLT S1 = LLT::scalar(1);
  LLT S32 = LLT::scalar(32);

  bool DstOnValu = MRI.getRegBankOrNull(Src2) == &RVGPU::VGPRRegBank;
  bool Accumulate = true;

  if (!DstOnValu) {
    if (mi_match(Src2, MRI, m_ZeroInt()))
      Accumulate = false;
  }

  // Keep the multiplication on the SALU.
  Register DstHi;
  Register DstLo = B.buildMul(S32, Src0, Src1).getReg(0);
  bool MulHiInVgpr = false;

  MRI.setRegBank(DstLo, RVGPU::SGPRRegBank);

  if (Subtarget.hasSMulHi()) {
    DstHi = IsUnsigned ? B.buildUMulH(S32, Src0, Src1).getReg(0)
                       : B.buildSMulH(S32, Src0, Src1).getReg(0);
    MRI.setRegBank(DstHi, RVGPU::SGPRRegBank);
  } else {
    Register VSrc0 = B.buildCopy(S32, Src0).getReg(0);
    Register VSrc1 = B.buildCopy(S32, Src1).getReg(0);

    MRI.setRegBank(VSrc0, RVGPU::VGPRRegBank);
    MRI.setRegBank(VSrc1, RVGPU::VGPRRegBank);

    DstHi = IsUnsigned ? B.buildUMulH(S32, VSrc0, VSrc1).getReg(0)
                       : B.buildSMulH(S32, VSrc0, VSrc1).getReg(0);
    MRI.setRegBank(DstHi, RVGPU::VGPRRegBank);

    if (!DstOnValu) {
      DstHi = buildReadFirstLane(B, MRI, DstHi);
    } else {
      MulHiInVgpr = true;
    }
  }

  // Accumulate and produce the "carry-out" bit.
  //
  // The "carry-out" is defined as bit 64 of the result when computed as a
  // big integer. For unsigned multiply-add, this matches the usual definition
  // of carry-out. For signed multiply-add, bit 64 is the sign bit of the
  // result, which is determined as:
  //   sign(Src0 * Src1) + sign(Src2) + carry-out from unsigned 64-bit add
  LLT CarryType = DstOnValu ? S1 : S32;
  const RegisterBank &CarryBank =
      DstOnValu ? RVGPU::VCCRegBank : RVGPU::SGPRRegBank;
  const RegisterBank &DstBank =
      DstOnValu ? RVGPU::VGPRRegBank : RVGPU::SGPRRegBank;
  Register Carry;
  Register Zero;

  if (!IsUnsigned) {
    Zero = B.buildConstant(S32, 0).getReg(0);
    MRI.setRegBank(Zero,
                   MulHiInVgpr ? RVGPU::VGPRRegBank : RVGPU::SGPRRegBank);

    Carry = B.buildICmp(CmpInst::ICMP_SLT, MulHiInVgpr ? S1 : S32, DstHi, Zero)
                .getReg(0);
    MRI.setRegBank(Carry, MulHiInVgpr ? RVGPU::VCCRegBank
                                      : RVGPU::SGPRRegBank);

    if (DstOnValu && !MulHiInVgpr) {
      Carry = B.buildTrunc(S1, Carry).getReg(0);
      MRI.setRegBank(Carry, RVGPU::VCCRegBank);
    }
  }

  if (Accumulate) {
    if (DstOnValu) {
      DstLo = B.buildCopy(S32, DstLo).getReg(0);
      DstHi = B.buildCopy(S32, DstHi).getReg(0);
      MRI.setRegBank(DstLo, RVGPU::VGPRRegBank);
      MRI.setRegBank(DstHi, RVGPU::VGPRRegBank);
    }

    auto Unmerge = B.buildUnmerge(S32, Src2);
    Register Src2Lo = Unmerge.getReg(0);
    Register Src2Hi = Unmerge.getReg(1);
    MRI.setRegBank(Src2Lo, DstBank);
    MRI.setRegBank(Src2Hi, DstBank);

    if (!IsUnsigned) {
      auto Src2Sign = B.buildICmp(CmpInst::ICMP_SLT, CarryType, Src2Hi, Zero);
      MRI.setRegBank(Src2Sign.getReg(0), CarryBank);

      Carry = B.buildXor(CarryType, Carry, Src2Sign).getReg(0);
      MRI.setRegBank(Carry, CarryBank);
    }

    auto AddLo = B.buildUAddo(S32, CarryType, DstLo, Src2Lo);
    DstLo = AddLo.getReg(0);
    Register CarryLo = AddLo.getReg(1);
    MRI.setRegBank(DstLo, DstBank);
    MRI.setRegBank(CarryLo, CarryBank);

    auto AddHi = B.buildUAdde(S32, CarryType, DstHi, Src2Hi, CarryLo);
    DstHi = AddHi.getReg(0);
    MRI.setRegBank(DstHi, DstBank);

    Register CarryHi = AddHi.getReg(1);
    MRI.setRegBank(CarryHi, CarryBank);

    if (IsUnsigned) {
      Carry = CarryHi;
    } else {
      Carry = B.buildXor(CarryType, Carry, CarryHi).getReg(0);
      MRI.setRegBank(Carry, CarryBank);
    }
  } else {
    if (IsUnsigned) {
      Carry = B.buildConstant(CarryType, 0).getReg(0);
      MRI.setRegBank(Carry, CarryBank);
    }
  }

  B.buildMergeLikeInstr(Dst0, {DstLo, DstHi});

  if (DstOnValu) {
    B.buildCopy(Dst1, Carry);
  } else {
    B.buildTrunc(Dst1, Carry);
  }

  MI.eraseFromParent();
  return true;
}

// Return a suitable opcode for extending the operands of Opc when widening.
static unsigned getExtendOp(unsigned Opc) {
  switch (Opc) {
  case TargetOpcode::G_ASHR:
  case TargetOpcode::G_SMIN:
  case TargetOpcode::G_SMAX:
    return TargetOpcode::G_SEXT;
  case TargetOpcode::G_LSHR:
  case TargetOpcode::G_UMIN:
  case TargetOpcode::G_UMAX:
    return TargetOpcode::G_ZEXT;
  default:
    return TargetOpcode::G_ANYEXT;
  }
}

// Emit a legalized extension from <2 x s16> to 2 32-bit components, avoiding
// any illegal vector extend or unmerge operations.
static std::pair<Register, Register>
unpackV2S16ToS32(MachineIRBuilder &B, Register Src, unsigned ExtOpcode) {
  const LLT S32 = LLT::scalar(32);
  auto Bitcast = B.buildBitcast(S32, Src);

  if (ExtOpcode == TargetOpcode::G_SEXT) {
    auto ExtLo = B.buildSExtInReg(S32, Bitcast, 16);
    auto ShiftHi = B.buildAShr(S32, Bitcast, B.buildConstant(S32, 16));
    return std::pair(ExtLo.getReg(0), ShiftHi.getReg(0));
  }

  auto ShiftHi = B.buildLShr(S32, Bitcast, B.buildConstant(S32, 16));
  if (ExtOpcode == TargetOpcode::G_ZEXT) {
    auto ExtLo = B.buildAnd(S32, Bitcast, B.buildConstant(S32, 0xffff));
    return std::pair(ExtLo.getReg(0), ShiftHi.getReg(0));
  }

  assert(ExtOpcode == TargetOpcode::G_ANYEXT);
  return std::pair(Bitcast.getReg(0), ShiftHi.getReg(0));
}

// For cases where only a single copy is inserted for matching register banks.
// Replace the register in the instruction operand
static bool substituteSimpleCopyRegs(
  const RVGPURegisterBankInfo::OperandsMapper &OpdMapper, unsigned OpIdx) {
  SmallVector<unsigned, 1> SrcReg(OpdMapper.getVRegs(OpIdx));
  if (!SrcReg.empty()) {
    assert(SrcReg.size() == 1);
    OpdMapper.getMI().getOperand(OpIdx).setReg(SrcReg[0]);
    return true;
  }

  return false;
}

/// Handle register layout difference for f16 images for some subtargets.
Register RVGPURegisterBankInfo::handleD16VData(MachineIRBuilder &B,
                                                MachineRegisterInfo &MRI,
                                                Register Reg) const {
  if (!Subtarget.hasUnpackedD16VMem())
    return Reg;

  const LLT S16 = LLT::scalar(16);
  LLT StoreVT = MRI.getType(Reg);
  if (!StoreVT.isVector() || StoreVT.getElementType() != S16)
    return Reg;

  auto Unmerge = B.buildUnmerge(S16, Reg);


  SmallVector<Register, 4> WideRegs;
  for (int I = 0, E = Unmerge->getNumOperands() - 1; I != E; ++I)
    WideRegs.push_back(Unmerge.getReg(I));

  const LLT S32 = LLT::scalar(32);
  int NumElts = StoreVT.getNumElements();

  return B.buildMergeLikeInstr(LLT::fixed_vector(NumElts, S32), WideRegs)
      .getReg(0);
}

static std::pair<Register, unsigned>
getBaseWithConstantOffset(MachineRegisterInfo &MRI, Register Reg) {
  int64_t Const;
  if (mi_match(Reg, MRI, m_ICst(Const)))
    return std::pair(Register(), Const);

  Register Base;
  if (mi_match(Reg, MRI, m_GAdd(m_Reg(Base), m_ICst(Const))))
    return std::pair(Base, Const);

  // TODO: Handle G_OR used for add case
  return std::pair(Reg, 0);
}

std::pair<Register, unsigned>
RVGPURegisterBankInfo::splitBufferOffsets(MachineIRBuilder &B,
                                           Register OrigOffset) const {
  const unsigned MaxImm = RVInstrInfo::getMaxMUBUFImmOffset(Subtarget);
  Register BaseReg;
  unsigned ImmOffset;
  const LLT S32 = LLT::scalar(32);

  // TODO: Use RVGPU::getBaseWithConstantOffset() instead.
  std::tie(BaseReg, ImmOffset) = getBaseWithConstantOffset(*B.getMRI(),
                                                           OrigOffset);

  unsigned C1 = 0;
  if (ImmOffset != 0) {
    // If the immediate value is too big for the immoffset field, put only bits
    // that would normally fit in the immoffset field. The remaining value that
    // is copied/added for the voffset field is a large power of 2, and it
    // stands more chance of being CSEd with the copy/add for another similar
    // load/store.
    // However, do not do that rounding down if that is a negative
    // number, as it appears to be illegal to have a negative offset in the
    // vgpr, even if adding the immediate offset makes it positive.
    unsigned Overflow = ImmOffset & ~MaxImm;
    ImmOffset -= Overflow;
    if ((int32_t)Overflow < 0) {
      Overflow += ImmOffset;
      ImmOffset = 0;
    }

    C1 = ImmOffset;
    if (Overflow != 0) {
      if (!BaseReg)
        BaseReg = B.buildConstant(S32, Overflow).getReg(0);
      else {
        auto OverflowVal = B.buildConstant(S32, Overflow);
        BaseReg = B.buildAdd(S32, BaseReg, OverflowVal).getReg(0);
      }
    }
  }

  if (!BaseReg)
    BaseReg = B.buildConstant(S32, 0).getReg(0);

  return {BaseReg, C1};
}

bool RVGPURegisterBankInfo::buildVCopy(MachineIRBuilder &B, Register DstReg,
                                        Register SrcReg) const {
  MachineRegisterInfo &MRI = *B.getMRI();
  LLT SrcTy = MRI.getType(SrcReg);
  if (SrcTy.getSizeInBits() == 32) {
    // Use a v_mov_b32 here to make the exec dependency explicit.
    B.buildInstr(RVGPU::V_MOV_B32_e32)
      .addDef(DstReg)
      .addUse(SrcReg);
    return constrainGenericRegister(DstReg, RVGPU::VGPR_32RegClass, MRI) &&
           constrainGenericRegister(SrcReg, RVGPU::SReg_32RegClass, MRI);
  }

  Register TmpReg0 = MRI.createVirtualRegister(&RVGPU::VGPR_32RegClass);
  Register TmpReg1 = MRI.createVirtualRegister(&RVGPU::VGPR_32RegClass);

  B.buildInstr(RVGPU::V_MOV_B32_e32)
    .addDef(TmpReg0)
    .addUse(SrcReg, 0, RVGPU::sub0);
  B.buildInstr(RVGPU::V_MOV_B32_e32)
    .addDef(TmpReg1)
    .addUse(SrcReg, 0, RVGPU::sub1);
  B.buildInstr(RVGPU::REG_SEQUENCE)
    .addDef(DstReg)
    .addUse(TmpReg0)
    .addImm(RVGPU::sub0)
    .addUse(TmpReg1)
    .addImm(RVGPU::sub1);

  return constrainGenericRegister(SrcReg, RVGPU::SReg_64RegClass, MRI) &&
         constrainGenericRegister(DstReg, RVGPU::VReg_64RegClass, MRI);
}

/// Utility function for pushing dynamic vector indexes with a constant offset
/// into waterfall loops.
static void reinsertVectorIndexAdd(MachineIRBuilder &B,
                                   MachineInstr &IdxUseInstr,
                                   unsigned OpIdx,
                                   unsigned ConstOffset) {
  MachineRegisterInfo &MRI = *B.getMRI();
  const LLT S32 = LLT::scalar(32);
  Register WaterfallIdx = IdxUseInstr.getOperand(OpIdx).getReg();
  B.setInsertPt(*IdxUseInstr.getParent(), IdxUseInstr.getIterator());

  auto MaterializedOffset = B.buildConstant(S32, ConstOffset);

  auto Add = B.buildAdd(S32, WaterfallIdx, MaterializedOffset);
  MRI.setRegBank(MaterializedOffset.getReg(0), RVGPU::SGPRRegBank);
  MRI.setRegBank(Add.getReg(0), RVGPU::SGPRRegBank);
  IdxUseInstr.getOperand(OpIdx).setReg(Add.getReg(0));
}

/// Implement extending a 32-bit value to a 64-bit value. \p Lo32Reg is the
/// original 32-bit source value (to be inserted in the low part of the combined
/// 64-bit result), and \p Hi32Reg is the high half of the combined 64-bit
/// value.
static void extendLow32IntoHigh32(MachineIRBuilder &B,
                                  Register Hi32Reg, Register Lo32Reg,
                                  unsigned ExtOpc,
                                  const RegisterBank &RegBank,
                                  bool IsBooleanSrc = false) {
  if (ExtOpc == RVGPU::G_ZEXT) {
    B.buildConstant(Hi32Reg, 0);
  } else if (ExtOpc == RVGPU::G_SEXT) {
    if (IsBooleanSrc) {
      // If we know the original source was an s1, the high half is the same as
      // the low.
      B.buildCopy(Hi32Reg, Lo32Reg);
    } else {
      // Replicate sign bit from 32-bit extended part.
      auto ShiftAmt = B.buildConstant(LLT::scalar(32), 31);
      B.getMRI()->setRegBank(ShiftAmt.getReg(0), RegBank);
      B.buildAShr(Hi32Reg, Lo32Reg, ShiftAmt);
    }
  } else {
    assert(ExtOpc == RVGPU::G_ANYEXT && "not an integer extension");
    B.buildUndef(Hi32Reg);
  }
}

bool RVGPURegisterBankInfo::foldExtractEltToCmpSelect(
    MachineIRBuilder &B, MachineInstr &MI,
    const OperandsMapper &OpdMapper) const {
  MachineRegisterInfo &MRI = *B.getMRI();

  Register VecReg = MI.getOperand(1).getReg();
  Register Idx = MI.getOperand(2).getReg();

  const RegisterBank &IdxBank =
    *OpdMapper.getInstrMapping().getOperandMapping(2).BreakDown[0].RegBank;

  bool IsDivergentIdx = IdxBank != RVGPU::SGPRRegBank;

  LLT VecTy = MRI.getType(VecReg);
  unsigned EltSize = VecTy.getScalarSizeInBits();
  unsigned NumElem = VecTy.getNumElements();

  if (!RVTargetLowering::shouldExpandVectorDynExt(EltSize, NumElem,
                                                  IsDivergentIdx, &Subtarget))
    return false;

  LLT S32 = LLT::scalar(32);

  const RegisterBank &DstBank =
    *OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
  const RegisterBank &SrcBank =
    *OpdMapper.getInstrMapping().getOperandMapping(1).BreakDown[0].RegBank;

  const RegisterBank &CCBank =
    (DstBank == RVGPU::SGPRRegBank &&
     SrcBank == RVGPU::SGPRRegBank &&
     IdxBank == RVGPU::SGPRRegBank) ? RVGPU::SGPRRegBank
                                     : RVGPU::VCCRegBank;
  LLT CCTy = (CCBank == RVGPU::SGPRRegBank) ? S32 : LLT::scalar(1);

  if (CCBank == RVGPU::VCCRegBank && IdxBank == RVGPU::SGPRRegBank) {
    Idx = B.buildCopy(S32, Idx)->getOperand(0).getReg();
    MRI.setRegBank(Idx, RVGPU::VGPRRegBank);
  }

  LLT EltTy = VecTy.getScalarType();
  SmallVector<Register, 2> DstRegs(OpdMapper.getVRegs(0));
  unsigned NumLanes = DstRegs.size();
  if (!NumLanes)
    NumLanes = 1;
  else
    EltTy = MRI.getType(DstRegs[0]);

  auto UnmergeToEltTy = B.buildUnmerge(EltTy, VecReg);
  SmallVector<Register, 2> Res(NumLanes);
  for (unsigned L = 0; L < NumLanes; ++L)
    Res[L] = UnmergeToEltTy.getReg(L);

  for (unsigned I = 1; I < NumElem; ++I) {
    auto IC = B.buildConstant(S32, I);
    MRI.setRegBank(IC->getOperand(0).getReg(), RVGPU::SGPRRegBank);
    auto Cmp = B.buildICmp(CmpInst::ICMP_EQ, CCTy, Idx, IC);
    MRI.setRegBank(Cmp->getOperand(0).getReg(), CCBank);

    for (unsigned L = 0; L < NumLanes; ++L) {
      auto S = B.buildSelect(EltTy, Cmp,
                             UnmergeToEltTy.getReg(I * NumLanes + L), Res[L]);

      for (unsigned N : { 0, 2, 3 })
        MRI.setRegBank(S->getOperand(N).getReg(), DstBank);

      Res[L] = S->getOperand(0).getReg();
    }
  }

  for (unsigned L = 0; L < NumLanes; ++L) {
    Register DstReg = (NumLanes == 1) ? MI.getOperand(0).getReg() : DstRegs[L];
    B.buildCopy(DstReg, Res[L]);
    MRI.setRegBank(DstReg, DstBank);
  }

  MRI.setRegBank(MI.getOperand(0).getReg(), DstBank);
  MI.eraseFromParent();

  return true;
}

// Insert a cross regbank copy for a register if it already has a bank that
// differs from the one we want to set.
static Register constrainRegToBank(MachineRegisterInfo &MRI,
                                   MachineIRBuilder &B, Register &Reg,
                                   const RegisterBank &Bank) {
  const RegisterBank *CurrBank = MRI.getRegBankOrNull(Reg);
  if (CurrBank && *CurrBank != Bank) {
    Register Copy = B.buildCopy(MRI.getType(Reg), Reg).getReg(0);
    MRI.setRegBank(Copy, Bank);
    return Copy;
  }

  MRI.setRegBank(Reg, Bank);
  return Reg;
}

bool RVGPURegisterBankInfo::foldInsertEltToCmpSelect(
    MachineIRBuilder &B, MachineInstr &MI,
    const OperandsMapper &OpdMapper) const {

  MachineRegisterInfo &MRI = *B.getMRI();
  Register VecReg = MI.getOperand(1).getReg();
  Register Idx = MI.getOperand(3).getReg();

  const RegisterBank &IdxBank =
    *OpdMapper.getInstrMapping().getOperandMapping(3).BreakDown[0].RegBank;

  bool IsDivergentIdx = IdxBank != RVGPU::SGPRRegBank;

  LLT VecTy = MRI.getType(VecReg);
  unsigned EltSize = VecTy.getScalarSizeInBits();
  unsigned NumElem = VecTy.getNumElements();

  if (!RVTargetLowering::shouldExpandVectorDynExt(EltSize, NumElem,
                                                  IsDivergentIdx, &Subtarget))
    return false;

  LLT S32 = LLT::scalar(32);

  const RegisterBank &DstBank =
    *OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
  const RegisterBank &SrcBank =
    *OpdMapper.getInstrMapping().getOperandMapping(1).BreakDown[0].RegBank;
  const RegisterBank &InsBank =
    *OpdMapper.getInstrMapping().getOperandMapping(2).BreakDown[0].RegBank;

  const RegisterBank &CCBank =
    (DstBank == RVGPU::SGPRRegBank &&
     SrcBank == RVGPU::SGPRRegBank &&
     InsBank == RVGPU::SGPRRegBank &&
     IdxBank == RVGPU::SGPRRegBank) ? RVGPU::SGPRRegBank
                                     : RVGPU::VCCRegBank;
  LLT CCTy = (CCBank == RVGPU::SGPRRegBank) ? S32 : LLT::scalar(1);

  if (CCBank == RVGPU::VCCRegBank && IdxBank == RVGPU::SGPRRegBank) {
    Idx = B.buildCopy(S32, Idx)->getOperand(0).getReg();
    MRI.setRegBank(Idx, RVGPU::VGPRRegBank);
  }

  LLT EltTy = VecTy.getScalarType();
  SmallVector<Register, 2> InsRegs(OpdMapper.getVRegs(2));
  unsigned NumLanes = InsRegs.size();
  if (!NumLanes) {
    NumLanes = 1;
    InsRegs.push_back(MI.getOperand(2).getReg());
  } else {
    EltTy = MRI.getType(InsRegs[0]);
  }

  auto UnmergeToEltTy = B.buildUnmerge(EltTy, VecReg);
  SmallVector<Register, 16> Ops(NumElem * NumLanes);

  for (unsigned I = 0; I < NumElem; ++I) {
    auto IC = B.buildConstant(S32, I);
    MRI.setRegBank(IC->getOperand(0).getReg(), RVGPU::SGPRRegBank);
    auto Cmp = B.buildICmp(CmpInst::ICMP_EQ, CCTy, Idx, IC);
    MRI.setRegBank(Cmp->getOperand(0).getReg(), CCBank);

    for (unsigned L = 0; L < NumLanes; ++L) {
      Register Op0 = constrainRegToBank(MRI, B, InsRegs[L], DstBank);
      Register Op1 = UnmergeToEltTy.getReg(I * NumLanes + L);
      Op1 = constrainRegToBank(MRI, B, Op1, DstBank);

      Register Select = B.buildSelect(EltTy, Cmp, Op0, Op1).getReg(0);
      MRI.setRegBank(Select, DstBank);

      Ops[I * NumLanes + L] = Select;
    }
  }

  LLT MergeTy = LLT::fixed_vector(Ops.size(), EltTy);
  if (MergeTy == MRI.getType(MI.getOperand(0).getReg())) {
    B.buildBuildVector(MI.getOperand(0), Ops);
  } else {
    auto Vec = B.buildBuildVector(MergeTy, Ops);
    MRI.setRegBank(Vec->getOperand(0).getReg(), DstBank);
    B.buildBitcast(MI.getOperand(0).getReg(), Vec);
  }

  MRI.setRegBank(MI.getOperand(0).getReg(), DstBank);
  MI.eraseFromParent();

  return true;
}

void RVGPURegisterBankInfo::applyMappingImpl(
    MachineIRBuilder &B, const OperandsMapper &OpdMapper) const {
  MachineInstr &MI = OpdMapper.getMI();
  B.setInstrAndDebugLoc(MI);
  unsigned Opc = MI.getOpcode();
  MachineRegisterInfo &MRI = OpdMapper.getMRI();
  switch (Opc) {
  case RVGPU::G_CONSTANT:
  case RVGPU::G_IMPLICIT_DEF: {
    Register DstReg = MI.getOperand(0).getReg();
    LLT DstTy = MRI.getType(DstReg);
    if (DstTy != LLT::scalar(1))
      break;

    const RegisterBank *DstBank =
        OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
    if (DstBank == &RVGPU::VCCRegBank)
      break;
    SmallVector<Register, 1> DefRegs(OpdMapper.getVRegs(0));
    if (DefRegs.empty())
      DefRegs.push_back(DstReg);

    B.setInsertPt(*MI.getParent(), ++MI.getIterator());

    Register NewDstReg = MRI.createGenericVirtualRegister(LLT::scalar(32));
    LLVMContext &Ctx = B.getMF().getFunction().getContext();

    MI.getOperand(0).setReg(NewDstReg);
    if (Opc != RVGPU::G_IMPLICIT_DEF) {
      uint64_t ConstVal = MI.getOperand(1).getCImm()->getZExtValue();
      MI.getOperand(1).setCImm(
          ConstantInt::get(IntegerType::getInt32Ty(Ctx), ConstVal));
    }

    MRI.setRegBank(NewDstReg, *DstBank);
    B.buildTrunc(DefRegs[0], NewDstReg);
    return;
  }
  case RVGPU::G_PHI: {
    Register DstReg = MI.getOperand(0).getReg();
    LLT DstTy = MRI.getType(DstReg);
    if (DstTy != LLT::scalar(1))
      break;

    const LLT S32 = LLT::scalar(32);
    const RegisterBank *DstBank =
      OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
    if (DstBank == &RVGPU::VCCRegBank) {
      applyDefaultMapping(OpdMapper);
      // The standard handling only considers the result register bank for
      // phis. For VCC, blindly inserting a copy when the phi is lowered will
      // produce an invalid copy. We can only copy with some kind of compare to
      // get a vector boolean result. Insert a register bank copy that will be
      // correctly lowered to a compare.
      for (unsigned I = 1, E = MI.getNumOperands(); I != E; I += 2) {
        Register SrcReg = MI.getOperand(I).getReg();
        const RegisterBank *SrcBank = getRegBank(SrcReg, MRI, *TRI);

        if (SrcBank != &RVGPU::VCCRegBank) {
          MachineBasicBlock *SrcMBB = MI.getOperand(I + 1).getMBB();
          B.setInsertPt(*SrcMBB, SrcMBB->getFirstTerminator());

          auto Copy = B.buildCopy(LLT::scalar(1), SrcReg);
          MRI.setRegBank(Copy.getReg(0), RVGPU::VCCRegBank);
          MI.getOperand(I).setReg(Copy.getReg(0));
        }
      }

      return;
    }

    // Phi handling is strange and only considers the bank of the destination.
    substituteSimpleCopyRegs(OpdMapper, 0);

    // Promote SGPR/VGPR booleans to s32
    ApplyRegBankMapping ApplyBank(B, *this, MRI, DstBank);
    B.setInsertPt(B.getMBB(), MI);
    LegalizerHelper Helper(B.getMF(), ApplyBank, B);

    if (Helper.widenScalar(MI, 0, S32) != LegalizerHelper::Legalized)
      llvm_unreachable("widen scalar should have succeeded");

    return;
  }
  case RVGPU::G_FCMP:
    if (!Subtarget.hasSALUFloatInsts())
      break;
    LLVM_FALLTHROUGH;
  case RVGPU::G_ICMP:
  case RVGPU::G_UADDO:
  case RVGPU::G_USUBO:
  case RVGPU::G_UADDE:
  case RVGPU::G_SADDE:
  case RVGPU::G_USUBE:
  case RVGPU::G_SSUBE: {
    unsigned BoolDstOp =
        (Opc == RVGPU::G_ICMP || Opc == RVGPU::G_FCMP) ? 0 : 1;
    Register DstReg = MI.getOperand(BoolDstOp).getReg();

    const RegisterBank *DstBank =
      OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
    if (DstBank != &RVGPU::SGPRRegBank)
      break;

    const bool HasCarryIn = MI.getNumOperands() == 5;

    // If this is a scalar compare, promote the result to s32, as the selection
    // will end up using a copy to a 32-bit vreg.
    const LLT S32 = LLT::scalar(32);
    Register NewDstReg = MRI.createGenericVirtualRegister(S32);
    MRI.setRegBank(NewDstReg, RVGPU::SGPRRegBank);
    MI.getOperand(BoolDstOp).setReg(NewDstReg);

    if (HasCarryIn) {
      Register NewSrcReg = MRI.createGenericVirtualRegister(S32);
      MRI.setRegBank(NewSrcReg, RVGPU::SGPRRegBank);
      B.buildZExt(NewSrcReg, MI.getOperand(4).getReg());
      MI.getOperand(4).setReg(NewSrcReg);
    }

    MachineBasicBlock *MBB = MI.getParent();
    B.setInsertPt(*MBB, std::next(MI.getIterator()));

    // If we had a constrained VCC result register, a copy was inserted to VCC
    // from SGPR.
    SmallVector<Register, 1> DefRegs(OpdMapper.getVRegs(0));
    if (DefRegs.empty())
      DefRegs.push_back(DstReg);
    B.buildTrunc(DefRegs[0], NewDstReg);
    return;
  }
  case RVGPU::G_SELECT: {
    Register DstReg = MI.getOperand(0).getReg();
    LLT DstTy = MRI.getType(DstReg);

    SmallVector<Register, 1> CondRegs(OpdMapper.getVRegs(1));
    if (CondRegs.empty())
      CondRegs.push_back(MI.getOperand(1).getReg());
    else {
      assert(CondRegs.size() == 1);
    }

    const RegisterBank *CondBank = getRegBank(CondRegs[0], MRI, *TRI);
    if (CondBank == &RVGPU::SGPRRegBank) {
      const LLT S32 = LLT::scalar(32);
      Register NewCondReg = MRI.createGenericVirtualRegister(S32);
      MRI.setRegBank(NewCondReg, RVGPU::SGPRRegBank);

      MI.getOperand(1).setReg(NewCondReg);
      B.buildZExt(NewCondReg, CondRegs[0]);
    }

    if (DstTy.getSizeInBits() != 64)
      break;

    LLT HalfTy = getHalfSizedType(DstTy);

    SmallVector<Register, 2> DefRegs(OpdMapper.getVRegs(0));
    SmallVector<Register, 2> Src1Regs(OpdMapper.getVRegs(2));
    SmallVector<Register, 2> Src2Regs(OpdMapper.getVRegs(3));

    // All inputs are SGPRs, nothing special to do.
    if (DefRegs.empty()) {
      assert(Src1Regs.empty() && Src2Regs.empty());
      break;
    }

    if (Src1Regs.empty())
      split64BitValueForMapping(B, Src1Regs, HalfTy, MI.getOperand(2).getReg());
    else {
      setRegsToType(MRI, Src1Regs, HalfTy);
    }

    if (Src2Regs.empty())
      split64BitValueForMapping(B, Src2Regs, HalfTy, MI.getOperand(3).getReg());
    else
      setRegsToType(MRI, Src2Regs, HalfTy);

    setRegsToType(MRI, DefRegs, HalfTy);

    B.buildSelect(DefRegs[0], CondRegs[0], Src1Regs[0], Src2Regs[0]);
    B.buildSelect(DefRegs[1], CondRegs[0], Src1Regs[1], Src2Regs[1]);

    MRI.setRegBank(DstReg, RVGPU::VGPRRegBank);
    MI.eraseFromParent();
    return;
  }
  case RVGPU::G_BRCOND: {
    Register CondReg = MI.getOperand(0).getReg();
    // FIXME: Should use legalizer helper, but should change bool ext type.
    const RegisterBank *CondBank =
      OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;

    if (CondBank == &RVGPU::SGPRRegBank) {
      const LLT S32 = LLT::scalar(32);
      Register NewCondReg = MRI.createGenericVirtualRegister(S32);
      MRI.setRegBank(NewCondReg, RVGPU::SGPRRegBank);

      MI.getOperand(0).setReg(NewCondReg);
      B.buildZExt(NewCondReg, CondReg);
      return;
    }

    break;
  }
  case RVGPU::G_AND:
  case RVGPU::G_OR:
  case RVGPU::G_XOR: {
    // 64-bit and is only available on the SALU, so split into 2 32-bit ops if
    // there is a VGPR input.
    Register DstReg = MI.getOperand(0).getReg();
    LLT DstTy = MRI.getType(DstReg);

    if (DstTy.getSizeInBits() == 1) {
      const RegisterBank *DstBank =
        OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
      if (DstBank == &RVGPU::VCCRegBank)
        break;

      MachineFunction *MF = MI.getParent()->getParent();
      ApplyRegBankMapping ApplyBank(B, *this, MRI, DstBank);
      LegalizerHelper Helper(*MF, ApplyBank, B);

      if (Helper.widenScalar(MI, 0, LLT::scalar(32)) !=
          LegalizerHelper::Legalized)
        llvm_unreachable("widen scalar should have succeeded");
      return;
    }

    if (DstTy.getSizeInBits() != 64)
      break;

    LLT HalfTy = getHalfSizedType(DstTy);
    SmallVector<Register, 2> DefRegs(OpdMapper.getVRegs(0));
    SmallVector<Register, 2> Src0Regs(OpdMapper.getVRegs(1));
    SmallVector<Register, 2> Src1Regs(OpdMapper.getVRegs(2));

    // All inputs are SGPRs, nothing special to do.
    if (DefRegs.empty()) {
      assert(Src0Regs.empty() && Src1Regs.empty());
      break;
    }

    assert(DefRegs.size() == 2);
    assert(Src0Regs.size() == Src1Regs.size() &&
           (Src0Regs.empty() || Src0Regs.size() == 2));

    // Depending on where the source registers came from, the generic code may
    // have decided to split the inputs already or not. If not, we still need to
    // extract the values.

    if (Src0Regs.empty())
      split64BitValueForMapping(B, Src0Regs, HalfTy, MI.getOperand(1).getReg());
    else
      setRegsToType(MRI, Src0Regs, HalfTy);

    if (Src1Regs.empty())
      split64BitValueForMapping(B, Src1Regs, HalfTy, MI.getOperand(2).getReg());
    else
      setRegsToType(MRI, Src1Regs, HalfTy);

    setRegsToType(MRI, DefRegs, HalfTy);

    B.buildInstr(Opc, {DefRegs[0]}, {Src0Regs[0], Src1Regs[0]});
    B.buildInstr(Opc, {DefRegs[1]}, {Src0Regs[1], Src1Regs[1]});

    MRI.setRegBank(DstReg, RVGPU::VGPRRegBank);
    MI.eraseFromParent();
    return;
  }
  case RVGPU::G_ABS: {
    Register SrcReg = MI.getOperand(1).getReg();
    const RegisterBank *SrcBank = MRI.getRegBankOrNull(SrcReg);

    // There is no VALU abs instruction so we need to replace it with a sub and
    // max combination.
    if (SrcBank && SrcBank == &RVGPU::VGPRRegBank) {
      MachineFunction *MF = MI.getParent()->getParent();
      ApplyRegBankMapping Apply(B, *this, MRI, &RVGPU::VGPRRegBank);
      LegalizerHelper Helper(*MF, Apply, B);

      if (Helper.lowerAbsToMaxNeg(MI) != LegalizerHelper::Legalized)
        llvm_unreachable("lowerAbsToMaxNeg should have succeeded");
      return;
    }
    [[fallthrough]];
  }
  case RVGPU::G_ADD:
  case RVGPU::G_SUB:
  case RVGPU::G_MUL:
  case RVGPU::G_SHL:
  case RVGPU::G_LSHR:
  case RVGPU::G_ASHR:
  case RVGPU::G_SMIN:
  case RVGPU::G_SMAX:
  case RVGPU::G_UMIN:
  case RVGPU::G_UMAX: {
    Register DstReg = MI.getOperand(0).getReg();
    LLT DstTy = MRI.getType(DstReg);

    // 16-bit operations are VALU only, but can be promoted to 32-bit SALU.
    // Packed 16-bit operations need to be scalarized and promoted.
    if (DstTy != LLT::scalar(16) && DstTy != LLT::fixed_vector(2, 16))
      break;

    const RegisterBank *DstBank =
      OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
    if (DstBank == &RVGPU::VGPRRegBank)
      break;

    const LLT S32 = LLT::scalar(32);
    MachineBasicBlock *MBB = MI.getParent();
    MachineFunction *MF = MBB->getParent();
    ApplyRegBankMapping ApplySALU(B, *this, MRI, &RVGPU::SGPRRegBank);

    if (DstTy.isVector() && Opc == RVGPU::G_ABS) {
      Register WideSrcLo, WideSrcHi;

      std::tie(WideSrcLo, WideSrcHi) =
          unpackV2S16ToS32(B, MI.getOperand(1).getReg(), TargetOpcode::G_SEXT);
      auto Lo = B.buildInstr(RVGPU::G_ABS, {S32}, {WideSrcLo});
      auto Hi = B.buildInstr(RVGPU::G_ABS, {S32}, {WideSrcHi});
      B.buildBuildVectorTrunc(DstReg, {Lo.getReg(0), Hi.getReg(0)});
      MI.eraseFromParent();
      return;
    }

    if (DstTy.isVector()) {
      Register WideSrc0Lo, WideSrc0Hi;
      Register WideSrc1Lo, WideSrc1Hi;

      unsigned ExtendOp = getExtendOp(MI.getOpcode());
      std::tie(WideSrc0Lo, WideSrc0Hi)
        = unpackV2S16ToS32(B, MI.getOperand(1).getReg(), ExtendOp);
      std::tie(WideSrc1Lo, WideSrc1Hi)
        = unpackV2S16ToS32(B, MI.getOperand(2).getReg(), ExtendOp);
      auto Lo = B.buildInstr(MI.getOpcode(), {S32}, {WideSrc0Lo, WideSrc1Lo});
      auto Hi = B.buildInstr(MI.getOpcode(), {S32}, {WideSrc0Hi, WideSrc1Hi});
      B.buildBuildVectorTrunc(DstReg, {Lo.getReg(0), Hi.getReg(0)});
      MI.eraseFromParent();
    } else {
      LegalizerHelper Helper(*MF, ApplySALU, B);

      if (Helper.widenScalar(MI, 0, S32) != LegalizerHelper::Legalized)
        llvm_unreachable("widen scalar should have succeeded");

      // FIXME: s16 shift amounts should be legal.
      if (Opc == RVGPU::G_SHL || Opc == RVGPU::G_LSHR ||
          Opc == RVGPU::G_ASHR) {
        B.setInsertPt(*MBB, MI.getIterator());
        if (Helper.widenScalar(MI, 1, S32) != LegalizerHelper::Legalized)
          llvm_unreachable("widen scalar should have succeeded");
      }
    }

    return;
  }
  case RVGPU::G_SEXT_INREG: {
    SmallVector<Register, 2> SrcRegs(OpdMapper.getVRegs(1));
    if (SrcRegs.empty())
      break; // Nothing to repair

    const LLT S32 = LLT::scalar(32);
    ApplyRegBankMapping O(B, *this, MRI, &RVGPU::VGPRRegBank);

    // Don't use LegalizerHelper's narrowScalar. It produces unwanted G_SEXTs
    // we would need to further expand, and doesn't let us directly set the
    // result registers.
    SmallVector<Register, 2> DstRegs(OpdMapper.getVRegs(0));

    int Amt = MI.getOperand(2).getImm();
    if (Amt <= 32) {
      // Downstream users have expectations for the high bit behavior, so freeze
      // incoming undefined bits.
      if (Amt == 32) {
        // The low bits are unchanged.
        B.buildFreeze(DstRegs[0], SrcRegs[0]);
      } else {
        auto Freeze = B.buildFreeze(S32, SrcRegs[0]);
        // Extend in the low bits and propagate the sign bit to the high half.
        B.buildSExtInReg(DstRegs[0], Freeze, Amt);
      }

      B.buildAShr(DstRegs[1], DstRegs[0], B.buildConstant(S32, 31));
    } else {
      // The low bits are unchanged, and extend in the high bits.
      // No freeze required
      B.buildCopy(DstRegs[0], SrcRegs[0]);
      B.buildSExtInReg(DstRegs[1], DstRegs[0], Amt - 32);
    }

    Register DstReg = MI.getOperand(0).getReg();
    MRI.setRegBank(DstReg, RVGPU::VGPRRegBank);
    MI.eraseFromParent();
    return;
  }
  case RVGPU::G_CTPOP:
  case RVGPU::G_BITREVERSE: {
    const RegisterBank *DstBank =
      OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
    if (DstBank == &RVGPU::SGPRRegBank)
      break;

    Register SrcReg = MI.getOperand(1).getReg();
    const LLT S32 = LLT::scalar(32);
    LLT Ty = MRI.getType(SrcReg);
    if (Ty == S32)
      break;

    ApplyRegBankMapping ApplyVALU(B, *this, MRI, &RVGPU::VGPRRegBank);

    MachineFunction &MF = B.getMF();
    LegalizerHelper Helper(MF, ApplyVALU, B);

    if (Helper.narrowScalar(MI, 1, S32) != LegalizerHelper::Legalized)
      llvm_unreachable("narrowScalar should have succeeded");
    return;
  }
  case RVGPU::G_RVGPU_FFBH_U32:
  case RVGPU::G_RVGPU_FFBL_B32:
  case RVGPU::G_CTLZ_ZERO_UNDEF:
  case RVGPU::G_CTTZ_ZERO_UNDEF: {
    const RegisterBank *DstBank =
        OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
    if (DstBank == &RVGPU::SGPRRegBank)
      break;

    Register SrcReg = MI.getOperand(1).getReg();
    const LLT S32 = LLT::scalar(32);
    LLT Ty = MRI.getType(SrcReg);
    if (Ty == S32)
      break;

    // We can narrow this more efficiently than Helper can by using ffbh/ffbl
    // which return -1 when the input is zero:
    // (ctlz_zero_undef hi:lo) -> (umin (ffbh hi), (add (ffbh lo), 32))
    // (cttz_zero_undef hi:lo) -> (umin (add (ffbl hi), 32), (ffbl lo))
    // (ffbh hi:lo) -> (umin (ffbh hi), (uaddsat (ffbh lo), 32))
    // (ffbl hi:lo) -> (umin (uaddsat (ffbh hi), 32), (ffbh lo))
    ApplyRegBankMapping ApplyVALU(B, *this, MRI, &RVGPU::VGPRRegBank);
    SmallVector<Register, 2> SrcRegs(OpdMapper.getVRegs(1));
    unsigned NewOpc = Opc == RVGPU::G_CTLZ_ZERO_UNDEF
                          ? (unsigned)RVGPU::G_RVGPU_FFBH_U32
                          : Opc == RVGPU::G_CTTZ_ZERO_UNDEF
                                ? (unsigned)RVGPU::G_RVGPU_FFBL_B32
                                : Opc;
    unsigned Idx = NewOpc == RVGPU::G_RVGPU_FFBH_U32;
    auto X = B.buildInstr(NewOpc, {S32}, {SrcRegs[Idx]});
    auto Y = B.buildInstr(NewOpc, {S32}, {SrcRegs[Idx ^ 1]});
    unsigned AddOpc =
        Opc == RVGPU::G_CTLZ_ZERO_UNDEF || Opc == RVGPU::G_CTTZ_ZERO_UNDEF
            ? RVGPU::G_ADD
            : RVGPU::G_UADDSAT;
    Y = B.buildInstr(AddOpc, {S32}, {Y, B.buildConstant(S32, 32)});
    Register DstReg = MI.getOperand(0).getReg();
    B.buildUMin(DstReg, X, Y);
    MI.eraseFromParent();
    return;
  }
  case RVGPU::G_SEXT:
  case RVGPU::G_ZEXT:
  case RVGPU::G_ANYEXT: {
    Register SrcReg = MI.getOperand(1).getReg();
    LLT SrcTy = MRI.getType(SrcReg);
    const bool Signed = Opc == RVGPU::G_SEXT;

    assert(OpdMapper.getVRegs(1).empty());

    const RegisterBank *SrcBank =
      OpdMapper.getInstrMapping().getOperandMapping(1).BreakDown[0].RegBank;

    Register DstReg = MI.getOperand(0).getReg();
    LLT DstTy = MRI.getType(DstReg);
    if (DstTy.isScalar() &&
        SrcBank != &RVGPU::SGPRRegBank &&
        SrcBank != &RVGPU::VCCRegBank &&
        // FIXME: Should handle any type that round to s64 when irregular
        // breakdowns supported.
        DstTy.getSizeInBits() == 64 &&
        SrcTy.getSizeInBits() <= 32) {
      SmallVector<Register, 2> DefRegs(OpdMapper.getVRegs(0));

      // Extend to 32-bit, and then extend the low half.
      if (Signed) {
        // TODO: Should really be buildSExtOrCopy
        B.buildSExtOrTrunc(DefRegs[0], SrcReg);
      } else if (Opc == RVGPU::G_ZEXT) {
        B.buildZExtOrTrunc(DefRegs[0], SrcReg);
      } else {
        B.buildAnyExtOrTrunc(DefRegs[0], SrcReg);
      }

      extendLow32IntoHigh32(B, DefRegs[1], DefRegs[0], Opc, *SrcBank);
      MRI.setRegBank(DstReg, *SrcBank);
      MI.eraseFromParent();
      return;
    }

    if (SrcTy != LLT::scalar(1))
      return;

    // It is not legal to have a legalization artifact with a VCC source. Rather
    // than introducing a copy, insert the select we would have to select the
    // copy to.
    if (SrcBank == &RVGPU::VCCRegBank) {
      SmallVector<Register, 2> DefRegs(OpdMapper.getVRegs(0));

      const RegisterBank *DstBank = &RVGPU::VGPRRegBank;

      unsigned DstSize = DstTy.getSizeInBits();
      // 64-bit select is SGPR only
      const bool UseSel64 = DstSize > 32 &&
        SrcBank->getID() == RVGPU::SGPRRegBankID;

      // TODO: Should s16 select be legal?
      LLT SelType = UseSel64 ? LLT::scalar(64) : LLT::scalar(32);
      auto True = B.buildConstant(SelType, Signed ? -1 : 1);
      auto False = B.buildConstant(SelType, 0);

      MRI.setRegBank(True.getReg(0), *DstBank);
      MRI.setRegBank(False.getReg(0), *DstBank);
      MRI.setRegBank(DstReg, *DstBank);

      if (DstSize > 32) {
        B.buildSelect(DefRegs[0], SrcReg, True, False);
        extendLow32IntoHigh32(B, DefRegs[1], DefRegs[0], Opc, *SrcBank, true);
      } else if (DstSize < 32) {
        auto Sel = B.buildSelect(SelType, SrcReg, True, False);
        MRI.setRegBank(Sel.getReg(0), *DstBank);
        B.buildTrunc(DstReg, Sel);
      } else {
        B.buildSelect(DstReg, SrcReg, True, False);
      }

      MI.eraseFromParent();
      return;
    }

    break;
  }
  case RVGPU::G_EXTRACT_VECTOR_ELT: {
    SmallVector<Register, 2> DstRegs(OpdMapper.getVRegs(0));

    assert(OpdMapper.getVRegs(1).empty() && OpdMapper.getVRegs(2).empty());

    Register DstReg = MI.getOperand(0).getReg();
    Register SrcReg = MI.getOperand(1).getReg();

    const LLT S32 = LLT::scalar(32);
    LLT DstTy = MRI.getType(DstReg);
    LLT SrcTy = MRI.getType(SrcReg);

    if (foldExtractEltToCmpSelect(B, MI, OpdMapper))
      return;

    const ValueMapping &DstMapping
      = OpdMapper.getInstrMapping().getOperandMapping(0);
    const RegisterBank *DstBank = DstMapping.BreakDown[0].RegBank;
    const RegisterBank *SrcBank =
      OpdMapper.getInstrMapping().getOperandMapping(1).BreakDown[0].RegBank;
    const RegisterBank *IdxBank =
        OpdMapper.getInstrMapping().getOperandMapping(2).BreakDown[0].RegBank;

    Register BaseIdxReg;
    unsigned ConstOffset;
    std::tie(BaseIdxReg, ConstOffset) =
        RVGPU::getBaseWithConstantOffset(MRI, MI.getOperand(2).getReg());

    // See if the index is an add of a constant which will be foldable by moving
    // the base register of the index later if this is going to be executed in a
    // waterfall loop. This is essentially to reassociate the add of a constant
    // with the readfirstlane.
    bool ShouldMoveIndexIntoLoop = IdxBank != &RVGPU::SGPRRegBank &&
                                   ConstOffset > 0 &&
                                   ConstOffset < SrcTy.getNumElements();

    // Move the base register. We'll re-insert the add later.
    if (ShouldMoveIndexIntoLoop)
      MI.getOperand(2).setReg(BaseIdxReg);

    // If this is a VGPR result only because the index was a VGPR result, the
    // actual indexing will be done on the SGPR source vector, which will
    // produce a scalar result. We need to copy to the VGPR result inside the
    // waterfall loop.
    const bool NeedCopyToVGPR = DstBank == &RVGPU::VGPRRegBank &&
                                SrcBank == &RVGPU::SGPRRegBank;
    if (DstRegs.empty()) {
      applyDefaultMapping(OpdMapper);

      executeInWaterfallLoop(B, MI, {2});

      if (NeedCopyToVGPR) {
        // We don't want a phi for this temporary reg.
        Register TmpReg = MRI.createGenericVirtualRegister(DstTy);
        MRI.setRegBank(TmpReg, RVGPU::SGPRRegBank);
        MI.getOperand(0).setReg(TmpReg);
        B.setInsertPt(*MI.getParent(), ++MI.getIterator());

        // Use a v_mov_b32 here to make the exec dependency explicit.
        buildVCopy(B, DstReg, TmpReg);
      }

      // Re-insert the constant offset add inside the waterfall loop.
      if (ShouldMoveIndexIntoLoop)
        reinsertVectorIndexAdd(B, MI, 2, ConstOffset);

      return;
    }

    assert(DstTy.getSizeInBits() == 64);

    LLT Vec32 = LLT::fixed_vector(2 * SrcTy.getNumElements(), 32);

    auto CastSrc = B.buildBitcast(Vec32, SrcReg);
    auto One = B.buildConstant(S32, 1);

    MachineBasicBlock::iterator MII = MI.getIterator();

    // Split the vector index into 32-bit pieces. Prepare to move all of the
    // new instructions into a waterfall loop if necessary.
    //
    // Don't put the bitcast or constant in the loop.
    MachineInstrSpan Span(MII, &B.getMBB());

    // Compute 32-bit element indices, (2 * OrigIdx, 2 * OrigIdx + 1).
    auto IdxLo = B.buildShl(S32, BaseIdxReg, One);
    auto IdxHi = B.buildAdd(S32, IdxLo, One);

    auto Extract0 = B.buildExtractVectorElement(DstRegs[0], CastSrc, IdxLo);
    auto Extract1 = B.buildExtractVectorElement(DstRegs[1], CastSrc, IdxHi);

    MRI.setRegBank(DstReg, *DstBank);
    MRI.setRegBank(CastSrc.getReg(0), *SrcBank);
    MRI.setRegBank(One.getReg(0), RVGPU::SGPRRegBank);
    MRI.setRegBank(IdxLo.getReg(0), RVGPU::SGPRRegBank);
    MRI.setRegBank(IdxHi.getReg(0), RVGPU::SGPRRegBank);

    SmallSet<Register, 4> OpsToWaterfall;
    if (!collectWaterfallOperands(OpsToWaterfall, MI, MRI, { 2 })) {
      MI.eraseFromParent();
      return;
    }

    // Remove the original instruction to avoid potentially confusing the
    // waterfall loop logic.
    B.setInstr(*Span.begin());
    MI.eraseFromParent();
    executeInWaterfallLoop(B, make_range(Span.begin(), Span.end()),
                           OpsToWaterfall);

    if (NeedCopyToVGPR) {
      MachineBasicBlock *LoopBB = Extract1->getParent();
      Register TmpReg0 = MRI.createGenericVirtualRegister(S32);
      Register TmpReg1 = MRI.createGenericVirtualRegister(S32);
      MRI.setRegBank(TmpReg0, RVGPU::SGPRRegBank);
      MRI.setRegBank(TmpReg1, RVGPU::SGPRRegBank);

      Extract0->getOperand(0).setReg(TmpReg0);
      Extract1->getOperand(0).setReg(TmpReg1);

      B.setInsertPt(*LoopBB, ++Extract1->getIterator());

      buildVCopy(B, DstRegs[0], TmpReg0);
      buildVCopy(B, DstRegs[1], TmpReg1);
    }

    if (ShouldMoveIndexIntoLoop)
      reinsertVectorIndexAdd(B, *IdxLo, 1, ConstOffset);

    return;
  }
  case RVGPU::G_INSERT_VECTOR_ELT: {
    SmallVector<Register, 2> InsRegs(OpdMapper.getVRegs(2));

    Register DstReg = MI.getOperand(0).getReg();
    LLT VecTy = MRI.getType(DstReg);

    assert(OpdMapper.getVRegs(0).empty());
    assert(OpdMapper.getVRegs(3).empty());

    if (substituteSimpleCopyRegs(OpdMapper, 1))
      MRI.setType(MI.getOperand(1).getReg(), VecTy);

    if (foldInsertEltToCmpSelect(B, MI, OpdMapper))
      return;

    const RegisterBank *IdxBank =
      OpdMapper.getInstrMapping().getOperandMapping(3).BreakDown[0].RegBank;

    Register SrcReg = MI.getOperand(1).getReg();
    Register InsReg = MI.getOperand(2).getReg();
    LLT InsTy = MRI.getType(InsReg);
    (void)InsTy;

    Register BaseIdxReg;
    unsigned ConstOffset;
    std::tie(BaseIdxReg, ConstOffset) =
        RVGPU::getBaseWithConstantOffset(MRI, MI.getOperand(3).getReg());

    // See if the index is an add of a constant which will be foldable by moving
    // the base register of the index later if this is going to be executed in a
    // waterfall loop. This is essentially to reassociate the add of a constant
    // with the readfirstlane.
    bool ShouldMoveIndexIntoLoop = IdxBank != &RVGPU::SGPRRegBank &&
      ConstOffset > 0 &&
      ConstOffset < VecTy.getNumElements();

    // Move the base register. We'll re-insert the add later.
    if (ShouldMoveIndexIntoLoop)
      MI.getOperand(3).setReg(BaseIdxReg);


    if (InsRegs.empty()) {
      executeInWaterfallLoop(B, MI, {3});

      // Re-insert the constant offset add inside the waterfall loop.
      if (ShouldMoveIndexIntoLoop) {
        reinsertVectorIndexAdd(B, MI, 3, ConstOffset);
      }

      return;
    }

    assert(InsTy.getSizeInBits() == 64);

    const LLT S32 = LLT::scalar(32);
    LLT Vec32 = LLT::fixed_vector(2 * VecTy.getNumElements(), 32);

    auto CastSrc = B.buildBitcast(Vec32, SrcReg);
    auto One = B.buildConstant(S32, 1);

    // Split the vector index into 32-bit pieces. Prepare to move all of the
    // new instructions into a waterfall loop if necessary.
    //
    // Don't put the bitcast or constant in the loop.
    MachineInstrSpan Span(MachineBasicBlock::iterator(&MI), &B.getMBB());

    // Compute 32-bit element indices, (2 * OrigIdx, 2 * OrigIdx + 1).
    auto IdxLo = B.buildShl(S32, BaseIdxReg, One);
    auto IdxHi = B.buildAdd(S32, IdxLo, One);

    auto InsLo = B.buildInsertVectorElement(Vec32, CastSrc, InsRegs[0], IdxLo);
    auto InsHi = B.buildInsertVectorElement(Vec32, InsLo, InsRegs[1], IdxHi);

    const RegisterBank *DstBank =
      OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
    const RegisterBank *SrcBank =
      OpdMapper.getInstrMapping().getOperandMapping(1).BreakDown[0].RegBank;
    const RegisterBank *InsSrcBank =
      OpdMapper.getInstrMapping().getOperandMapping(2).BreakDown[0].RegBank;

    MRI.setRegBank(InsReg, *InsSrcBank);
    MRI.setRegBank(CastSrc.getReg(0), *SrcBank);
    MRI.setRegBank(InsLo.getReg(0), *DstBank);
    MRI.setRegBank(InsHi.getReg(0), *DstBank);
    MRI.setRegBank(One.getReg(0), RVGPU::SGPRRegBank);
    MRI.setRegBank(IdxLo.getReg(0), RVGPU::SGPRRegBank);
    MRI.setRegBank(IdxHi.getReg(0), RVGPU::SGPRRegBank);


    SmallSet<Register, 4> OpsToWaterfall;
    if (!collectWaterfallOperands(OpsToWaterfall, MI, MRI, { 3 })) {
      B.setInsertPt(B.getMBB(), MI);
      B.buildBitcast(DstReg, InsHi);
      MI.eraseFromParent();
      return;
    }

    B.setInstr(*Span.begin());
    MI.eraseFromParent();

    // Figure out the point after the waterfall loop before mangling the control
    // flow.
    executeInWaterfallLoop(B, make_range(Span.begin(), Span.end()),
                           OpsToWaterfall);

    // The insertion point is now right after the original instruction.
    //
    // Keep the bitcast to the original vector type out of the loop. Doing this
    // saved an extra phi we don't need inside the loop.
    B.buildBitcast(DstReg, InsHi);

    // Re-insert the constant offset add inside the waterfall loop.
    if (ShouldMoveIndexIntoLoop)
      reinsertVectorIndexAdd(B, *IdxLo, 1, ConstOffset);

    return;
  }
  case RVGPU::G_RVGPU_BUFFER_LOAD:
  case RVGPU::G_RVGPU_BUFFER_LOAD_USHORT:
  case RVGPU::G_RVGPU_BUFFER_LOAD_SSHORT:
  case RVGPU::G_RVGPU_BUFFER_LOAD_UBYTE:
  case RVGPU::G_RVGPU_BUFFER_LOAD_SBYTE:
  case RVGPU::G_RVGPU_BUFFER_LOAD_FORMAT:
  case RVGPU::G_RVGPU_BUFFER_LOAD_FORMAT_TFE:
  case RVGPU::G_RVGPU_BUFFER_LOAD_FORMAT_D16:
  case RVGPU::G_RVGPU_TBUFFER_LOAD_FORMAT:
  case RVGPU::G_RVGPU_TBUFFER_LOAD_FORMAT_D16:
  case RVGPU::G_RVGPU_BUFFER_STORE:
  case RVGPU::G_RVGPU_BUFFER_STORE_BYTE:
  case RVGPU::G_RVGPU_BUFFER_STORE_SHORT:
  case RVGPU::G_RVGPU_BUFFER_STORE_FORMAT:
  case RVGPU::G_RVGPU_BUFFER_STORE_FORMAT_D16:
  case RVGPU::G_RVGPU_TBUFFER_STORE_FORMAT:
  case RVGPU::G_RVGPU_TBUFFER_STORE_FORMAT_D16: {
    applyDefaultMapping(OpdMapper);
    executeInWaterfallLoop(B, MI, {1, 4});
    return;
  }
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_SWAP:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_ADD:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_SUB:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_SMIN:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_UMIN:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_SMAX:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_UMAX:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_AND:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_OR:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_XOR:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_INC:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_DEC: {
    applyDefaultMapping(OpdMapper);
    executeInWaterfallLoop(B, MI, {2, 5});
    return;
  }
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_FADD:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_FMIN:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_FMAX: {
    applyDefaultMapping(OpdMapper);
    executeInWaterfallLoop(B, MI, {2, 5});
    return;
  }
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_CMPSWAP: {
    applyDefaultMapping(OpdMapper);
    executeInWaterfallLoop(B, MI, {3, 6});
    return;
  }
  case RVGPU::G_RVGPU_S_BUFFER_LOAD: {
    applyMappingSBufferLoad(B, OpdMapper);
    return;
  }
  case RVGPU::G_INTRINSIC:
  case RVGPU::G_INTRINSIC_CONVERGENT: {
    switch (cast<GIntrinsic>(MI).getIntrinsicID()) {
    case Intrinsic::rvgpu_readlane: {
      substituteSimpleCopyRegs(OpdMapper, 2);

      assert(OpdMapper.getVRegs(0).empty());
      assert(OpdMapper.getVRegs(3).empty());

      // Make sure the index is an SGPR. It doesn't make sense to run this in a
      // waterfall loop, so assume it's a uniform value.
      constrainOpWithReadfirstlane(B, MI, 3); // Index
      return;
    }
    case Intrinsic::rvgpu_writelane: {
      assert(OpdMapper.getVRegs(0).empty());
      assert(OpdMapper.getVRegs(2).empty());
      assert(OpdMapper.getVRegs(3).empty());

      substituteSimpleCopyRegs(OpdMapper, 4); // VGPR input val
      constrainOpWithReadfirstlane(B, MI, 2); // Source value
      constrainOpWithReadfirstlane(B, MI, 3); // Index
      return;
    }
    case Intrinsic::rvgpu_interp_p1:
    case Intrinsic::rvgpu_interp_p2:
    case Intrinsic::rvgpu_interp_mov:
    case Intrinsic::rvgpu_interp_p1_f16:
    case Intrinsic::rvgpu_interp_p2_f16:
    case Intrinsic::rvgpu_lds_param_load: {
      applyDefaultMapping(OpdMapper);

      // Readlane for m0 value, which is always the last operand.
      // FIXME: Should this be a waterfall loop instead?
      constrainOpWithReadfirstlane(B, MI, MI.getNumOperands() - 1); // Index
      return;
    }
    case Intrinsic::rvgpu_interp_inreg_p10:
    case Intrinsic::rvgpu_interp_inreg_p2:
    case Intrinsic::rvgpu_interp_inreg_p10_f16:
    case Intrinsic::rvgpu_interp_inreg_p2_f16:
      applyDefaultMapping(OpdMapper);
      return;
    case Intrinsic::rvgpu_permlane16:
    case Intrinsic::rvgpu_permlanex16: {
      // Doing a waterfall loop over these wouldn't make any sense.
      substituteSimpleCopyRegs(OpdMapper, 2);
      substituteSimpleCopyRegs(OpdMapper, 3);
      constrainOpWithReadfirstlane(B, MI, 4);
      constrainOpWithReadfirstlane(B, MI, 5);
      return;
    }
    case Intrinsic::rvgpu_sbfe:
      applyMappingBFE(B, OpdMapper, true);
      return;
    case Intrinsic::rvgpu_ubfe:
      applyMappingBFE(B, OpdMapper, false);
      return;
    case Intrinsic::rvgpu_inverse_ballot:
    case Intrinsic::rvgpu_s_bitreplicate:
    case Intrinsic::rvgpu_s_quadmask:
    case Intrinsic::rvgpu_s_wqm:
      applyDefaultMapping(OpdMapper);
      constrainOpWithReadfirstlane(B, MI, 2); // Mask
      return;
    case Intrinsic::rvgpu_ballot:
      // Use default handling and insert copy to vcc source.
      break;
    }
    break;
  }
  case RVGPU::G_RVGPU_INTRIN_IMAGE_LOAD:
  case RVGPU::G_RVGPU_INTRIN_IMAGE_LOAD_D16:
  case RVGPU::G_RVGPU_INTRIN_IMAGE_STORE:
  case RVGPU::G_RVGPU_INTRIN_IMAGE_STORE_D16: {
    const RVGPU::RsrcIntrinsic *RSrcIntrin =
        RVGPU::lookupRsrcIntrinsic(RVGPU::getIntrinsicID(MI));
    assert(RSrcIntrin && RSrcIntrin->IsImage);
    // Non-images can have complications from operands that allow both SGPR
    // and VGPR. For now it's too complicated to figure out the final opcode
    // to derive the register bank from the MCInstrDesc.
    applyMappingImage(B, MI, OpdMapper, RSrcIntrin->RsrcArg);
    return;
  }
  case RVGPU::G_RVGPU_INTRIN_BVH_INTERSECT_RAY: {
    unsigned N = MI.getNumExplicitOperands() - 2;
    applyDefaultMapping(OpdMapper);
    executeInWaterfallLoop(B, MI, {N});
    return;
  }
  case RVGPU::G_INTRINSIC_W_SIDE_EFFECTS:
  case RVGPU::G_INTRINSIC_CONVERGENT_W_SIDE_EFFECTS: {
    auto IntrID = cast<GIntrinsic>(MI).getIntrinsicID();
    switch (IntrID) {
    case Intrinsic::rvgpu_ds_ordered_add:
    case Intrinsic::rvgpu_ds_ordered_swap: {
      // This is only allowed to execute with 1 lane, so readfirstlane is safe.
      assert(OpdMapper.getVRegs(0).empty());
      substituteSimpleCopyRegs(OpdMapper, 3);
      constrainOpWithReadfirstlane(B, MI, 2); // M0
      return;
    }
    case Intrinsic::rvgpu_ds_gws_init:
    case Intrinsic::rvgpu_ds_gws_barrier:
    case Intrinsic::rvgpu_ds_gws_sema_br: {
      // Only the first lane is executes, so readfirstlane is safe.
      substituteSimpleCopyRegs(OpdMapper, 1);
      constrainOpWithReadfirstlane(B, MI, 2); // M0
      return;
    }
    case Intrinsic::rvgpu_ds_gws_sema_v:
    case Intrinsic::rvgpu_ds_gws_sema_p:
    case Intrinsic::rvgpu_ds_gws_sema_release_all: {
      // Only the first lane is executes, so readfirstlane is safe.
      constrainOpWithReadfirstlane(B, MI, 1); // M0
      return;
    }
    case Intrinsic::rvgpu_ds_append:
    case Intrinsic::rvgpu_ds_consume: {
      constrainOpWithReadfirstlane(B, MI, 2); // M0
      return;
    }
    case Intrinsic::rvgpu_s_sendmsg:
    case Intrinsic::rvgpu_s_sendmsghalt: {
      // FIXME: Should this use a waterfall loop?
      constrainOpWithReadfirstlane(B, MI, 2); // M0
      return;
    }
    case Intrinsic::rvgpu_s_setreg: {
      constrainOpWithReadfirstlane(B, MI, 2);
      return;
    }
    case Intrinsic::rvgpu_s_ttracedata:
      constrainOpWithReadfirstlane(B, MI, 1); // M0
      return;
    case Intrinsic::rvgpu_raw_buffer_load_lds:
    case Intrinsic::rvgpu_raw_ptr_buffer_load_lds: {
      applyDefaultMapping(OpdMapper);
      constrainOpWithReadfirstlane(B, MI, 1); // rsrc
      constrainOpWithReadfirstlane(B, MI, 2); // M0
      constrainOpWithReadfirstlane(B, MI, 5); // soffset
      return;
    }
    case Intrinsic::rvgpu_struct_buffer_load_lds:
    case Intrinsic::rvgpu_struct_ptr_buffer_load_lds: {
      applyDefaultMapping(OpdMapper);
      constrainOpWithReadfirstlane(B, MI, 1); // rsrc
      constrainOpWithReadfirstlane(B, MI, 2); // M0
      constrainOpWithReadfirstlane(B, MI, 6); // soffset
      return;
    }
    case Intrinsic::rvgpu_global_load_lds: {
      applyDefaultMapping(OpdMapper);
      constrainOpWithReadfirstlane(B, MI, 2);
      return;
    }
    case Intrinsic::rvgpu_lds_direct_load: {
      applyDefaultMapping(OpdMapper);
      // Readlane for m0 value, which is always the last operand.
      constrainOpWithReadfirstlane(B, MI, MI.getNumOperands() - 1); // Index
      return;
    }
    case Intrinsic::rvgpu_exp_row:
      applyDefaultMapping(OpdMapper);
      constrainOpWithReadfirstlane(B, MI, 8); // M0
      return;
    case Intrinsic::rvgpu_s_sleep_var:
      assert(OpdMapper.getVRegs(1).empty());
      constrainOpWithReadfirstlane(B, MI, 1);
      return;
    case Intrinsic::rvgpu_s_barrier_signal_var:
    case Intrinsic::rvgpu_s_barrier_join:
    case Intrinsic::rvgpu_s_wakeup_barrier:
      constrainOpWithReadfirstlane(B, MI, 1);
      return;
    case Intrinsic::rvgpu_s_barrier_signal_isfirst_var:
      constrainOpWithReadfirstlane(B, MI, 2);
      return;
    case Intrinsic::rvgpu_s_barrier_init:
      constrainOpWithReadfirstlane(B, MI, 1);
      constrainOpWithReadfirstlane(B, MI, 2);
      return;
    case Intrinsic::rvgpu_s_get_barrier_state: {
      constrainOpWithReadfirstlane(B, MI, 2);
      return;
    }
    default: {
      if (const RVGPU::RsrcIntrinsic *RSrcIntrin =
              RVGPU::lookupRsrcIntrinsic(IntrID)) {
        // Non-images can have complications from operands that allow both SGPR
        // and VGPR. For now it's too complicated to figure out the final opcode
        // to derive the register bank from the MCInstrDesc.
        if (RSrcIntrin->IsImage) {
          applyMappingImage(B, MI, OpdMapper, RSrcIntrin->RsrcArg);
          return;
        }
      }

      break;
    }
    }
    break;
  }
  case RVGPU::G_SI_CALL: {
    // Use a set to avoid extra readfirstlanes in the case where multiple
    // operands are the same register.
    SmallSet<Register, 4> SGPROperandRegs;

    if (!collectWaterfallOperands(SGPROperandRegs, MI, MRI, {1}))
      break;

    // Move all copies to physical SGPRs that are used by the call instruction
    // into the loop block. Start searching for these copies until the
    // ADJCALLSTACKUP.
    unsigned FrameSetupOpcode = RVGPU::ADJCALLSTACKUP;
    unsigned FrameDestroyOpcode = RVGPU::ADJCALLSTACKDOWN;

    // Move all non-copies before the copies, so that a complete range can be
    // moved into the waterfall loop.
    SmallVector<MachineInstr *, 4> NonCopyInstrs;
    // Count of NonCopyInstrs found until the current LastCopy.
    unsigned NonCopyInstrsLen = 0;
    MachineBasicBlock::iterator Start(&MI);
    MachineBasicBlock::iterator LastCopy = Start;
    MachineBasicBlock *MBB = MI.getParent();
    const RVMachineFunctionInfo *Info =
        MBB->getParent()->getInfo<RVMachineFunctionInfo>();
    while (Start->getOpcode() != FrameSetupOpcode) {
      --Start;
      bool IsCopy = false;
      if (Start->getOpcode() == RVGPU::COPY) {
        auto &Dst = Start->getOperand(0);
        if (Dst.isReg()) {
          Register Reg = Dst.getReg();
          if (Reg.isPhysical() && MI.readsRegister(Reg, TRI)) {
            IsCopy = true;
          } else {
            // Also move the copy from the scratch rsrc descriptor into the loop
            // to allow it to be optimized away.
            auto &Src = Start->getOperand(1);
            if (Src.isReg()) {
              Reg = Src.getReg();
              IsCopy = Info->getScratchRSrcReg() == Reg;
            }
          }
        }
      }

      if (IsCopy) {
        LastCopy = Start;
        NonCopyInstrsLen = NonCopyInstrs.size();
      } else {
        NonCopyInstrs.push_back(&*Start);
      }
    }
    NonCopyInstrs.resize(NonCopyInstrsLen);

    for (auto *NonCopy : reverse(NonCopyInstrs)) {
      MBB->splice(LastCopy, MBB, NonCopy->getIterator());
    }
    Start = LastCopy;

    // Do the same for copies after the loop
    NonCopyInstrs.clear();
    NonCopyInstrsLen = 0;
    MachineBasicBlock::iterator End(&MI);
    LastCopy = End;
    while (End->getOpcode() != FrameDestroyOpcode) {
      ++End;
      bool IsCopy = false;
      if (End->getOpcode() == RVGPU::COPY) {
        auto &Src = End->getOperand(1);
        if (Src.isReg()) {
          Register Reg = Src.getReg();
          IsCopy = Reg.isPhysical() && MI.modifiesRegister(Reg, TRI);
        }
      }

      if (IsCopy) {
        LastCopy = End;
        NonCopyInstrsLen = NonCopyInstrs.size();
      } else {
        NonCopyInstrs.push_back(&*End);
      }
    }
    NonCopyInstrs.resize(NonCopyInstrsLen);

    End = LastCopy;
    ++LastCopy;
    for (auto *NonCopy : reverse(NonCopyInstrs)) {
      MBB->splice(LastCopy, MBB, NonCopy->getIterator());
    }

    ++End;
    B.setInsertPt(B.getMBB(), Start);
    executeInWaterfallLoop(B, make_range(Start, End), SGPROperandRegs);
    break;
  }
  case RVGPU::G_LOAD:
  case RVGPU::G_ZEXTLOAD:
  case RVGPU::G_SEXTLOAD: {
    if (applyMappingLoad(B, OpdMapper, MI))
      return;
    break;
  }
  case RVGPU::G_DYN_STACKALLOC:
    applyMappingDynStackAlloc(B, OpdMapper, MI);
    return;
  case RVGPU::G_STACKRESTORE: {
    applyDefaultMapping(OpdMapper);
    constrainOpWithReadfirstlane(B, MI, 0);
    return;
  }
  case RVGPU::G_SBFX:
    applyMappingBFE(B, OpdMapper, /*Signed*/ true);
    return;
  case RVGPU::G_UBFX:
    applyMappingBFE(B, OpdMapper, /*Signed*/ false);
    return;
  case RVGPU::G_RVGPU_MAD_U64_U32:
  case RVGPU::G_RVGPU_MAD_I64_I32:
    applyMappingMAD_64_32(B, OpdMapper);
    return;
  case RVGPU::G_PREFETCH: {
    if (!Subtarget.hasPrefetch()) {
      MI.eraseFromParent();
      return;
    }
    unsigned PtrBank =
        getRegBankID(MI.getOperand(0).getReg(), MRI, RVGPU::SGPRRegBankID);
    if (PtrBank == RVGPU::VGPRRegBankID) {
      MI.eraseFromParent();
      return;
    }
    // FIXME: There is currently no support for prefetch in global isel.
    // There is no node equivalence and what's worse there is no MMO produced
    // for a prefetch on global isel path.
    // Prefetch does not affect execution so erase it for now.
    MI.eraseFromParent();
    return;
  }
  default:
    break;
  }

  return applyDefaultMapping(OpdMapper);
}

// vgpr, sgpr -> vgpr
// vgpr, agpr -> vgpr
// agpr, agpr -> agpr
// agpr, sgpr -> vgpr
static unsigned regBankUnion(unsigned RB0, unsigned RB1) {
  if (RB0 == RVGPU::InvalidRegBankID)
    return RB1;
  if (RB1 == RVGPU::InvalidRegBankID)
    return RB0;

  if (RB0 == RVGPU::SGPRRegBankID && RB1 == RVGPU::SGPRRegBankID)
    return RVGPU::SGPRRegBankID;

  if (RB0 == RVGPU::AGPRRegBankID && RB1 == RVGPU::AGPRRegBankID)
    return RVGPU::AGPRRegBankID;

  return RVGPU::VGPRRegBankID;
}

static unsigned regBankBoolUnion(unsigned RB0, unsigned RB1) {
  if (RB0 == RVGPU::InvalidRegBankID)
    return RB1;
  if (RB1 == RVGPU::InvalidRegBankID)
    return RB0;

  // vcc, vcc -> vcc
  // vcc, sgpr -> vcc
  // vcc, vgpr -> vcc
  if (RB0 == RVGPU::VCCRegBankID || RB1 == RVGPU::VCCRegBankID)
    return RVGPU::VCCRegBankID;

  // vcc, vgpr -> vgpr
  return regBankUnion(RB0, RB1);
}

unsigned RVGPURegisterBankInfo::getMappingType(const MachineRegisterInfo &MRI,
                                                const MachineInstr &MI) const {
  unsigned RegBank = RVGPU::InvalidRegBankID;

  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg())
      continue;
    Register Reg = MO.getReg();
    if (const RegisterBank *Bank = getRegBank(Reg, MRI, *TRI)) {
      RegBank = regBankUnion(RegBank, Bank->getID());
      if (RegBank == RVGPU::VGPRRegBankID)
        break;
    }
  }

  return RegBank;
}

bool RVGPURegisterBankInfo::isSALUMapping(const MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg())
      continue;
    Register Reg = MO.getReg();
    if (const RegisterBank *Bank = getRegBank(Reg, MRI, *TRI)) {
      if (Bank->getID() != RVGPU::SGPRRegBankID)
        return false;
    }
  }
  return true;
}

const RegisterBankInfo::InstructionMapping &
RVGPURegisterBankInfo::getDefaultMappingSOP(const MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  SmallVector<const ValueMapping*, 8> OpdsMapping(MI.getNumOperands());

  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    const MachineOperand &SrcOp = MI.getOperand(i);
    if (!SrcOp.isReg())
      continue;

    unsigned Size = getSizeInBits(SrcOp.getReg(), MRI, *TRI);
    OpdsMapping[i] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, Size);
  }
  return getInstructionMapping(1, 1, getOperandsMapping(OpdsMapping),
                               MI.getNumOperands());
}

const RegisterBankInfo::InstructionMapping &
RVGPURegisterBankInfo::getDefaultMappingVOP(const MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  SmallVector<const ValueMapping*, 8> OpdsMapping(MI.getNumOperands());

  // Even though we technically could use SGPRs, this would require knowledge of
  // the constant bus restriction. Force all sources to VGPR (except for VCC).
  //
  // TODO: Unary ops are trivially OK, so accept SGPRs?
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    const MachineOperand &Src = MI.getOperand(i);
    if (!Src.isReg())
      continue;

    unsigned Size = getSizeInBits(Src.getReg(), MRI, *TRI);
    unsigned BankID = Size == 1 ? RVGPU::VCCRegBankID : RVGPU::VGPRRegBankID;
    OpdsMapping[i] = RVGPU::getValueMapping(BankID, Size);
  }

  return getInstructionMapping(1, 1, getOperandsMapping(OpdsMapping),
                               MI.getNumOperands());
}

const RegisterBankInfo::InstructionMapping &
RVGPURegisterBankInfo::getDefaultMappingAllVGPR(const MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  SmallVector<const ValueMapping*, 8> OpdsMapping(MI.getNumOperands());

  for (unsigned I = 0, E = MI.getNumOperands(); I != E; ++I) {
    const MachineOperand &Op = MI.getOperand(I);
    if (!Op.isReg())
      continue;

    unsigned Size = getSizeInBits(Op.getReg(), MRI, *TRI);
    OpdsMapping[I] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size);
  }

  return getInstructionMapping(1, 1, getOperandsMapping(OpdsMapping),
                               MI.getNumOperands());
}

const RegisterBankInfo::InstructionMapping &
RVGPURegisterBankInfo::getImageMapping(const MachineRegisterInfo &MRI,
                                        const MachineInstr &MI,
                                        int RsrcIdx) const {
  // The reported argument index is relative to the IR intrinsic call arguments,
  // so we need to shift by the number of defs and the intrinsic ID.
  RsrcIdx += MI.getNumExplicitDefs() + 1;

  const int NumOps = MI.getNumOperands();
  SmallVector<const ValueMapping *, 8> OpdsMapping(NumOps);

  // TODO: Should packed/unpacked D16 difference be reported here as part of
  // the value mapping?
  for (int I = 0; I != NumOps; ++I) {
    if (!MI.getOperand(I).isReg())
      continue;

    Register OpReg = MI.getOperand(I).getReg();
    // We replace some dead address operands with $noreg
    if (!OpReg)
      continue;

    unsigned Size = getSizeInBits(OpReg, MRI, *TRI);

    // FIXME: Probably need a new intrinsic register bank searchable table to
    // handle arbitrary intrinsics easily.
    //
    // If this has a sampler, it immediately follows rsrc.
    const bool MustBeSGPR = I == RsrcIdx || I == RsrcIdx + 1;

    if (MustBeSGPR) {
      // If this must be an SGPR, so we must report whatever it is as legal.
      unsigned NewBank = getRegBankID(OpReg, MRI, RVGPU::SGPRRegBankID);
      OpdsMapping[I] = RVGPU::getValueMapping(NewBank, Size);
    } else {
      // Some operands must be VGPR, and these are easy to copy to.
      OpdsMapping[I] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size);
    }
  }

  return getInstructionMapping(1, 1, getOperandsMapping(OpdsMapping), NumOps);
}

/// Return the mapping for a pointer argument.
const RegisterBankInfo::ValueMapping *
RVGPURegisterBankInfo::getValueMappingForPtr(const MachineRegisterInfo &MRI,
                                              Register PtrReg) const {
  LLT PtrTy = MRI.getType(PtrReg);
  unsigned Size = PtrTy.getSizeInBits();
  if (Subtarget.useFlatForGlobal() ||
      !RVGPU::isFlatGlobalAddrSpace(PtrTy.getAddressSpace()))
    return RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size);

  // If we're using MUBUF instructions for global memory, an SGPR base register
  // is possible. Otherwise this needs to be a VGPR.
  const RegisterBank *PtrBank = getRegBank(PtrReg, MRI, *TRI);
  return RVGPU::getValueMapping(PtrBank->getID(), Size);
}

const RegisterBankInfo::InstructionMapping &
RVGPURegisterBankInfo::getInstrMappingForLoad(const MachineInstr &MI) const {

  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  SmallVector<const ValueMapping*, 2> OpdsMapping(2);
  unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
  Register PtrReg = MI.getOperand(1).getReg();
  LLT PtrTy = MRI.getType(PtrReg);
  unsigned AS = PtrTy.getAddressSpace();
  unsigned PtrSize = PtrTy.getSizeInBits();

  const ValueMapping *ValMapping;
  const ValueMapping *PtrMapping;

  const RegisterBank *PtrBank = getRegBank(PtrReg, MRI, *TRI);

  if (PtrBank == &RVGPU::SGPRRegBank && RVGPU::isFlatGlobalAddrSpace(AS)) {
    if (isScalarLoadLegal(MI)) {
      // We have a uniform instruction so we want to use an SMRD load
      ValMapping = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, Size);
      PtrMapping = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, PtrSize);
    } else {
      ValMapping = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size);

      // If we're using MUBUF instructions for global memory, an SGPR base
      // register is possible. Otherwise this needs to be a VGPR.
      unsigned PtrBankID = Subtarget.useFlatForGlobal() ?
        RVGPU::VGPRRegBankID : RVGPU::SGPRRegBankID;

      PtrMapping = RVGPU::getValueMapping(PtrBankID, PtrSize);
    }
  } else {
    ValMapping = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size);
    PtrMapping = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, PtrSize);
  }

  OpdsMapping[0] = ValMapping;
  OpdsMapping[1] = PtrMapping;
  const RegisterBankInfo::InstructionMapping &Mapping = getInstructionMapping(
      1, 1, getOperandsMapping(OpdsMapping), MI.getNumOperands());
  return Mapping;

  // FIXME: Do we want to add a mapping for FLAT load, or should we just
  // handle that during instruction selection?
}

unsigned
RVGPURegisterBankInfo::getRegBankID(Register Reg,
                                     const MachineRegisterInfo &MRI,
                                     unsigned Default) const {
  const RegisterBank *Bank = getRegBank(Reg, MRI, *TRI);
  return Bank ? Bank->getID() : Default;
}

const RegisterBankInfo::ValueMapping *
RVGPURegisterBankInfo::getSGPROpMapping(Register Reg,
                                         const MachineRegisterInfo &MRI,
                                         const TargetRegisterInfo &TRI) const {
  // Lie and claim anything is legal, even though this needs to be an SGPR
  // applyMapping will have to deal with it as a waterfall loop.
  unsigned Bank = getRegBankID(Reg, MRI, RVGPU::SGPRRegBankID);
  unsigned Size = getSizeInBits(Reg, MRI, TRI);
  return RVGPU::getValueMapping(Bank, Size);
}

const RegisterBankInfo::ValueMapping *
RVGPURegisterBankInfo::getVGPROpMapping(Register Reg,
                                         const MachineRegisterInfo &MRI,
                                         const TargetRegisterInfo &TRI) const {
  unsigned Size = getSizeInBits(Reg, MRI, TRI);
  return RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size);
}

const RegisterBankInfo::ValueMapping *
RVGPURegisterBankInfo::getAGPROpMapping(Register Reg,
                                         const MachineRegisterInfo &MRI,
                                         const TargetRegisterInfo &TRI) const {
  unsigned Size = getSizeInBits(Reg, MRI, TRI);
  return RVGPU::getValueMapping(RVGPU::AGPRRegBankID, Size);
}

///
/// This function must return a legal mapping, because
/// RVGPURegisterBankInfo::getInstrAlternativeMappings() is not called
/// in RegBankSelect::Mode::Fast.  Any mapping that would cause a
/// VGPR to SGPR generated is illegal.
///
// Operands that must be SGPRs must accept potentially divergent VGPRs as
// legal. These will be dealt with in applyMappingImpl.
//
const RegisterBankInfo::InstructionMapping &
RVGPURegisterBankInfo::getInstrMapping(const MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  if (MI.isCopy() || MI.getOpcode() == RVGPU::G_FREEZE) {
    // The default logic bothers to analyze impossible alternative mappings. We
    // want the most straightforward mapping, so just directly handle this.
    const RegisterBank *DstBank = getRegBank(MI.getOperand(0).getReg(), MRI,
                                             *TRI);
    const RegisterBank *SrcBank = getRegBank(MI.getOperand(1).getReg(), MRI,
                                             *TRI);
    assert(SrcBank && "src bank should have been assigned already");
    if (!DstBank)
      DstBank = SrcBank;

    unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
    if (MI.getOpcode() != RVGPU::G_FREEZE &&
        cannotCopy(*DstBank, *SrcBank, TypeSize::getFixed(Size)))
      return getInvalidInstructionMapping();

    const ValueMapping &ValMap = getValueMapping(0, Size, *DstBank);
    unsigned OpdsMappingSize = MI.isCopy() ? 1 : 2;
    SmallVector<const ValueMapping *, 1> OpdsMapping(OpdsMappingSize);
    OpdsMapping[0] = &ValMap;
    if (MI.getOpcode() == RVGPU::G_FREEZE)
      OpdsMapping[1] = &ValMap;

    return getInstructionMapping(
        1, /*Cost*/ 1,
        /*OperandsMapping*/ getOperandsMapping(OpdsMapping), OpdsMappingSize);
  }

  if (MI.isRegSequence()) {
    // If any input is a VGPR, the result must be a VGPR. The default handling
    // assumes any copy between banks is legal.
    unsigned BankID = RVGPU::SGPRRegBankID;

    for (unsigned I = 1, E = MI.getNumOperands(); I != E; I += 2) {
      auto OpBank = getRegBankID(MI.getOperand(I).getReg(), MRI);
      // It doesn't make sense to use vcc or scc banks here, so just ignore
      // them.
      if (OpBank != RVGPU::SGPRRegBankID) {
        BankID = RVGPU::VGPRRegBankID;
        break;
      }
    }
    unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);

    const ValueMapping &ValMap = getValueMapping(0, Size, getRegBank(BankID));
    return getInstructionMapping(
        1, /*Cost*/ 1,
        /*OperandsMapping*/ getOperandsMapping({&ValMap}), 1);
  }

  // The default handling is broken and doesn't handle illegal SGPR->VGPR copies
  // properly.
  //
  // TODO: There are additional exec masking dependencies to analyze.
  if (MI.getOpcode() == TargetOpcode::G_PHI) {
    unsigned ResultBank = RVGPU::InvalidRegBankID;
    Register DstReg = MI.getOperand(0).getReg();

    // Sometimes the result may have already been assigned a bank.
    if (const RegisterBank *DstBank = getRegBank(DstReg, MRI, *TRI))
      ResultBank = DstBank->getID();

    for (unsigned I = 1, E = MI.getNumOperands(); I != E; I += 2) {
      Register Reg = MI.getOperand(I).getReg();
      const RegisterBank *Bank = getRegBank(Reg, MRI, *TRI);

      // FIXME: Assuming VGPR for any undetermined inputs.
      if (!Bank || Bank->getID() == RVGPU::VGPRRegBankID) {
        ResultBank = RVGPU::VGPRRegBankID;
        break;
      }

      // FIXME: Need to promote SGPR case to s32
      unsigned OpBank = Bank->getID();
      ResultBank = regBankBoolUnion(ResultBank, OpBank);
    }

    assert(ResultBank != RVGPU::InvalidRegBankID);

    unsigned Size = MRI.getType(DstReg).getSizeInBits();

    const ValueMapping &ValMap =
        getValueMapping(0, Size, getRegBank(ResultBank));
    return getInstructionMapping(
        1, /*Cost*/ 1,
        /*OperandsMapping*/ getOperandsMapping({&ValMap}), 1);
  }

  const RegisterBankInfo::InstructionMapping &Mapping = getInstrMappingImpl(MI);
  if (Mapping.isValid())
    return Mapping;

  SmallVector<const ValueMapping*, 8> OpdsMapping(MI.getNumOperands());

  switch (MI.getOpcode()) {
  default:
    return getInvalidInstructionMapping();

  case RVGPU::G_AND:
  case RVGPU::G_OR:
  case RVGPU::G_XOR: {
    unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    if (Size == 1) {
      const RegisterBank *DstBank
        = getRegBank(MI.getOperand(0).getReg(), MRI, *TRI);

      unsigned TargetBankID = RVGPU::InvalidRegBankID;
      unsigned BankLHS = RVGPU::InvalidRegBankID;
      unsigned BankRHS = RVGPU::InvalidRegBankID;
      if (DstBank) {
        TargetBankID = DstBank->getID();
        if (DstBank == &RVGPU::VCCRegBank) {
          TargetBankID = RVGPU::VCCRegBankID;
          BankLHS = RVGPU::VCCRegBankID;
          BankRHS = RVGPU::VCCRegBankID;
        } else {
          BankLHS = getRegBankID(MI.getOperand(1).getReg(), MRI,
                                 RVGPU::SGPRRegBankID);
          BankRHS = getRegBankID(MI.getOperand(2).getReg(), MRI,
                                 RVGPU::SGPRRegBankID);
        }
      } else {
        BankLHS = getRegBankID(MI.getOperand(1).getReg(), MRI,
                               RVGPU::VCCRegBankID);
        BankRHS = getRegBankID(MI.getOperand(2).getReg(), MRI,
                               RVGPU::VCCRegBankID);

        // Both inputs should be true booleans to produce a boolean result.
        if (BankLHS == RVGPU::VGPRRegBankID || BankRHS == RVGPU::VGPRRegBankID) {
          TargetBankID = RVGPU::VGPRRegBankID;
        } else if (BankLHS == RVGPU::VCCRegBankID || BankRHS == RVGPU::VCCRegBankID) {
          TargetBankID = RVGPU::VCCRegBankID;
          BankLHS = RVGPU::VCCRegBankID;
          BankRHS = RVGPU::VCCRegBankID;
        } else if (BankLHS == RVGPU::SGPRRegBankID && BankRHS == RVGPU::SGPRRegBankID) {
          TargetBankID = RVGPU::SGPRRegBankID;
        }
      }

      OpdsMapping[0] = RVGPU::getValueMapping(TargetBankID, Size);
      OpdsMapping[1] = RVGPU::getValueMapping(BankLHS, Size);
      OpdsMapping[2] = RVGPU::getValueMapping(BankRHS, Size);
      break;
    }

    if (Size == 64) {

      if (isSALUMapping(MI)) {
        OpdsMapping[0] = getValueMappingSGPR64Only(RVGPU::SGPRRegBankID, Size);
        OpdsMapping[1] = OpdsMapping[2] = OpdsMapping[0];
      } else {
        OpdsMapping[0] = getValueMappingSGPR64Only(RVGPU::VGPRRegBankID, Size);
        unsigned Bank1 = getRegBankID(MI.getOperand(1).getReg(), MRI /*, DefaultBankID*/);
        OpdsMapping[1] = RVGPU::getValueMapping(Bank1, Size);

        unsigned Bank2 = getRegBankID(MI.getOperand(2).getReg(), MRI /*, DefaultBankID*/);
        OpdsMapping[2] = RVGPU::getValueMapping(Bank2, Size);
      }

      break;
    }

    [[fallthrough]];
  }
  case RVGPU::G_PTR_ADD:
  case RVGPU::G_PTRMASK:
  case RVGPU::G_ADD:
  case RVGPU::G_SUB:
  case RVGPU::G_MUL:
  case RVGPU::G_SHL:
  case RVGPU::G_LSHR:
  case RVGPU::G_ASHR:
  case RVGPU::G_UADDO:
  case RVGPU::G_USUBO:
  case RVGPU::G_UADDE:
  case RVGPU::G_SADDE:
  case RVGPU::G_USUBE:
  case RVGPU::G_SSUBE:
  case RVGPU::G_SMIN:
  case RVGPU::G_SMAX:
  case RVGPU::G_UMIN:
  case RVGPU::G_UMAX:
  case RVGPU::G_ABS:
  case RVGPU::G_SHUFFLE_VECTOR:
  case RVGPU::G_SBFX:
  case RVGPU::G_UBFX:
    if (isSALUMapping(MI))
      return getDefaultMappingSOP(MI);
    return getDefaultMappingVOP(MI);
  case RVGPU::G_FADD:
  case RVGPU::G_FSUB:
  case RVGPU::G_FMUL:
  case RVGPU::G_FMA:
  case RVGPU::G_FFLOOR:
  case RVGPU::G_FCEIL:
  case RVGPU::G_INTRINSIC_ROUNDEVEN:
  case RVGPU::G_FMINNUM:
  case RVGPU::G_FMAXNUM:
  case RVGPU::G_FMINIMUM:
  case RVGPU::G_FMAXIMUM:
  case RVGPU::G_INTRINSIC_TRUNC:
  case RVGPU::G_STRICT_FADD:
  case RVGPU::G_STRICT_FSUB:
  case RVGPU::G_STRICT_FMUL:
  case RVGPU::G_STRICT_FMA: {
    LLT Ty = MRI.getType(MI.getOperand(0).getReg());
    unsigned Size = Ty.getSizeInBits();
    if (Subtarget.hasSALUFloatInsts() && Ty.isScalar() &&
        (Size == 32 || Size == 16) && isSALUMapping(MI))
      return getDefaultMappingSOP(MI);
    return getDefaultMappingVOP(MI);
  }
  case RVGPU::G_FPTOSI:
  case RVGPU::G_FPTOUI:
  case RVGPU::G_SITOFP:
  case RVGPU::G_UITOFP: {
    unsigned SizeDst = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    unsigned SizeSrc = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
    if (Subtarget.hasSALUFloatInsts() && SizeDst == 32 && SizeSrc == 32 &&
        isSALUMapping(MI))
      return getDefaultMappingSOP(MI);
    return getDefaultMappingVOP(MI);
  }
  case RVGPU::G_FPTRUNC:
  case RVGPU::G_FPEXT: {
    unsigned SizeDst = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    unsigned SizeSrc = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
    if (Subtarget.hasSALUFloatInsts() && SizeDst != 64 && SizeSrc != 64 &&
        isSALUMapping(MI))
      return getDefaultMappingSOP(MI);
    return getDefaultMappingVOP(MI);
  }
  case RVGPU::G_FSQRT:
  case RVGPU::G_FEXP2:
  case RVGPU::G_FLOG2: {
    unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    if (Subtarget.hasPseudoScalarTrans() && (Size == 16 || Size == 32) &&
        isSALUMapping(MI))
      return getDefaultMappingSOP(MI);
    return getDefaultMappingVOP(MI);
  }
  case RVGPU::G_SADDSAT: // FIXME: Could lower sat ops for SALU
  case RVGPU::G_SSUBSAT:
  case RVGPU::G_UADDSAT:
  case RVGPU::G_USUBSAT:
  case RVGPU::G_FMAD:
  case RVGPU::G_FLDEXP:
  case RVGPU::G_FMINNUM_IEEE:
  case RVGPU::G_FMAXNUM_IEEE:
  case RVGPU::G_FCANONICALIZE:
  case RVGPU::G_STRICT_FLDEXP:
  case RVGPU::G_BSWAP: // TODO: Somehow expand for scalar?
  case RVGPU::G_FSHR: // TODO: Expand for scalar
  case RVGPU::G_RVGPU_FMIN_LEGACY:
  case RVGPU::G_RVGPU_FMAX_LEGACY:
  case RVGPU::G_RVGPU_RCP_IFLAG:
  case RVGPU::G_RVGPU_CVT_F32_UBYTE0:
  case RVGPU::G_RVGPU_CVT_F32_UBYTE1:
  case RVGPU::G_RVGPU_CVT_F32_UBYTE2:
  case RVGPU::G_RVGPU_CVT_F32_UBYTE3:
  case RVGPU::G_RVGPU_CVT_PK_I16_I32:
  case RVGPU::G_RVGPU_SMED3:
  case RVGPU::G_RVGPU_FMED3:
    return getDefaultMappingVOP(MI);
  case RVGPU::G_UMULH:
  case RVGPU::G_SMULH: {
    if (Subtarget.hasScalarMulHiInsts() && isSALUMapping(MI))
      return getDefaultMappingSOP(MI);
    return getDefaultMappingVOP(MI);
  }
  case RVGPU::G_RVGPU_MAD_U64_U32:
  case RVGPU::G_RVGPU_MAD_I64_I32: {
    // Three possible mappings:
    //
    //  - Default SOP
    //  - Default VOP
    //  - Scalar multiply: src0 and src1 are SGPRs, the rest is VOP.
    //
    // This allows instruction selection to keep the multiplication part of the
    // instruction on the SALU.
    bool AllSalu = true;
    bool MulSalu = true;
    for (unsigned i = 0; i < 5; ++i) {
      Register Reg = MI.getOperand(i).getReg();
      if (const RegisterBank *Bank = getRegBank(Reg, MRI, *TRI)) {
        if (Bank->getID() != RVGPU::SGPRRegBankID) {
          AllSalu = false;
          if (i == 2 || i == 3) {
            MulSalu = false;
            break;
          }
        }
      }
    }

    if (AllSalu)
      return getDefaultMappingSOP(MI);

    // If the multiply-add is full-rate in VALU, use that even if the
    // multiplication part is scalar. Accumulating separately on the VALU would
    // take two instructions.
    if (!MulSalu || Subtarget.hasFullRate64Ops())
      return getDefaultMappingVOP(MI);

    // Keep the multiplication on the SALU, then accumulate on the VALU.
    OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, 64);
    OpdsMapping[1] = RVGPU::getValueMapping(RVGPU::VCCRegBankID, 1);
    OpdsMapping[2] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, 32);
    OpdsMapping[3] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, 32);
    OpdsMapping[4] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, 64);
    break;
  }
  case RVGPU::G_IMPLICIT_DEF: {
    unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, Size);
    break;
  }
  case RVGPU::G_FCONSTANT:
  case RVGPU::G_CONSTANT:
  case RVGPU::G_GLOBAL_VALUE:
  case RVGPU::G_BLOCK_ADDR:
  case RVGPU::G_READCYCLECOUNTER: {
    unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, Size);
    break;
  }
  case RVGPU::G_FRAME_INDEX: {
    // TODO: This should be the same as other constants, but eliminateFrameIndex
    // currently assumes VALU uses.
    unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size);
    break;
  }
  case RVGPU::G_DYN_STACKALLOC: {
    // Result is always uniform, and a wave reduction is needed for the source.
    OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, 32);
    unsigned SrcBankID = getRegBankID(MI.getOperand(1).getReg(), MRI);
    OpdsMapping[1] = RVGPU::getValueMapping(SrcBankID, 32);
    break;
  }
  case RVGPU::G_RVGPU_WAVE_ADDRESS: {
    // This case is weird because we expect a physical register in the source,
    // but need to set a bank anyway.
    //
    // TODO: We could select the result to SGPR or VGPR
    OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, 32);
    OpdsMapping[1] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, 32);
    break;
  }
  case RVGPU::G_INSERT: {
    unsigned BankID = getMappingType(MRI, MI);
    unsigned DstSize = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
    unsigned SrcSize = getSizeInBits(MI.getOperand(1).getReg(), MRI, *TRI);
    unsigned EltSize = getSizeInBits(MI.getOperand(2).getReg(), MRI, *TRI);
    OpdsMapping[0] = RVGPU::getValueMapping(BankID, DstSize);
    OpdsMapping[1] = RVGPU::getValueMapping(BankID, SrcSize);
    OpdsMapping[2] = RVGPU::getValueMapping(BankID, EltSize);
    OpdsMapping[3] = nullptr;
    break;
  }
  case RVGPU::G_EXTRACT: {
    unsigned BankID = getRegBankID(MI.getOperand(1).getReg(), MRI);
    unsigned DstSize = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
    unsigned SrcSize = getSizeInBits(MI.getOperand(1).getReg(), MRI, *TRI);
    OpdsMapping[0] = RVGPU::getValueMapping(BankID, DstSize);
    OpdsMapping[1] = RVGPU::getValueMapping(BankID, SrcSize);
    OpdsMapping[2] = nullptr;
    break;
  }
  case RVGPU::G_BUILD_VECTOR:
  case RVGPU::G_BUILD_VECTOR_TRUNC: {
    LLT DstTy = MRI.getType(MI.getOperand(0).getReg());
    if (DstTy == LLT::fixed_vector(2, 16)) {
      unsigned DstSize = DstTy.getSizeInBits();
      unsigned SrcSize = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
      unsigned Src0BankID = getRegBankID(MI.getOperand(1).getReg(), MRI);
      unsigned Src1BankID = getRegBankID(MI.getOperand(2).getReg(), MRI);
      unsigned DstBankID = regBankUnion(Src0BankID, Src1BankID);

      OpdsMapping[0] = RVGPU::getValueMapping(DstBankID, DstSize);
      OpdsMapping[1] = RVGPU::getValueMapping(Src0BankID, SrcSize);
      OpdsMapping[2] = RVGPU::getValueMapping(Src1BankID, SrcSize);
      break;
    }

    [[fallthrough]];
  }
  case RVGPU::G_MERGE_VALUES:
  case RVGPU::G_CONCAT_VECTORS: {
    unsigned Bank = getMappingType(MRI, MI);
    unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    unsigned SrcSize = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();

    OpdsMapping[0] = RVGPU::getValueMapping(Bank, DstSize);
    // Op1 and Dst should use the same register bank.
    for (unsigned i = 1, e = MI.getNumOperands(); i != e; ++i)
      OpdsMapping[i] = RVGPU::getValueMapping(Bank, SrcSize);
    break;
  }
  case RVGPU::G_BITREVERSE:
  case RVGPU::G_BITCAST:
  case RVGPU::G_INTTOPTR:
  case RVGPU::G_PTRTOINT:
  case RVGPU::G_FABS:
  case RVGPU::G_FNEG: {
    unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    unsigned BankID = getRegBankID(MI.getOperand(1).getReg(), MRI);
    OpdsMapping[0] = OpdsMapping[1] = RVGPU::getValueMapping(BankID, Size);
    break;
  }
  case RVGPU::G_RVGPU_FFBH_U32:
  case RVGPU::G_RVGPU_FFBL_B32:
  case RVGPU::G_CTLZ_ZERO_UNDEF:
  case RVGPU::G_CTTZ_ZERO_UNDEF: {
    unsigned Size = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
    unsigned BankID = getRegBankID(MI.getOperand(1).getReg(), MRI);
    OpdsMapping[0] = RVGPU::getValueMapping(BankID, 32);
    OpdsMapping[1] = RVGPU::getValueMappingSGPR64Only(BankID, Size);
    break;
  }
  case RVGPU::G_CTPOP: {
    unsigned Size = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
    unsigned BankID = getRegBankID(MI.getOperand(1).getReg(), MRI);
    OpdsMapping[0] = RVGPU::getValueMapping(BankID, 32);

    // This should really be getValueMappingSGPR64Only, but allowing the generic
    // code to handle the register split just makes using LegalizerHelper more
    // difficult.
    OpdsMapping[1] = RVGPU::getValueMapping(BankID, Size);
    break;
  }
  case RVGPU::G_TRUNC: {
    Register Dst = MI.getOperand(0).getReg();
    Register Src = MI.getOperand(1).getReg();
    unsigned Bank = getRegBankID(Src, MRI);
    unsigned DstSize = getSizeInBits(Dst, MRI, *TRI);
    unsigned SrcSize = getSizeInBits(Src, MRI, *TRI);
    OpdsMapping[0] = RVGPU::getValueMapping(Bank, DstSize);
    OpdsMapping[1] = RVGPU::getValueMapping(Bank, SrcSize);
    break;
  }
  case RVGPU::G_ZEXT:
  case RVGPU::G_SEXT:
  case RVGPU::G_ANYEXT:
  case RVGPU::G_SEXT_INREG: {
    Register Dst = MI.getOperand(0).getReg();
    Register Src = MI.getOperand(1).getReg();
    unsigned DstSize = getSizeInBits(Dst, MRI, *TRI);
    unsigned SrcSize = getSizeInBits(Src, MRI, *TRI);

    unsigned DstBank;
    const RegisterBank *SrcBank = getRegBank(Src, MRI, *TRI);
    assert(SrcBank);
    switch (SrcBank->getID()) {
    case RVGPU::SGPRRegBankID:
      DstBank = RVGPU::SGPRRegBankID;
      break;
    default:
      DstBank = RVGPU::VGPRRegBankID;
      break;
    }

    // Scalar extend can use 64-bit BFE, but VGPRs require extending to
    // 32-bits, and then to 64.
    OpdsMapping[0] = RVGPU::getValueMappingSGPR64Only(DstBank, DstSize);
    OpdsMapping[1] = RVGPU::getValueMappingSGPR64Only(SrcBank->getID(),
                                                       SrcSize);
    break;
  }
  case RVGPU::G_IS_FPCLASS: {
    Register SrcReg = MI.getOperand(1).getReg();
    unsigned SrcSize = MRI.getType(SrcReg).getSizeInBits();
    unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::VCCRegBankID, DstSize);
    OpdsMapping[1] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, SrcSize);
    break;
  }
  case RVGPU::G_STORE: {
    assert(MI.getOperand(0).isReg());
    unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();

    // FIXME: We need to specify a different reg bank once scalar stores are
    // supported.
    const ValueMapping *ValMapping =
        RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size);
    OpdsMapping[0] = ValMapping;
    OpdsMapping[1] = getValueMappingForPtr(MRI, MI.getOperand(1).getReg());
    break;
  }
  case RVGPU::G_ICMP:
  case RVGPU::G_FCMP: {
    unsigned Size = MRI.getType(MI.getOperand(2).getReg()).getSizeInBits();

    // See if the result register has already been constrained to vcc, which may
    // happen due to control flow intrinsic lowering.
    unsigned DstBank = getRegBankID(MI.getOperand(0).getReg(), MRI,
                                    RVGPU::SGPRRegBankID);
    unsigned Op2Bank = getRegBankID(MI.getOperand(2).getReg(), MRI);
    unsigned Op3Bank = getRegBankID(MI.getOperand(3).getReg(), MRI);

    auto canUseSCCICMP = [&]() {
      auto Pred =
          static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate());
      return Size == 32 ||
             (Size == 64 &&
              (Pred == CmpInst::ICMP_EQ || Pred == CmpInst::ICMP_NE) &&
              Subtarget.hasScalarCompareEq64());
    };
    auto canUseSCCFCMP = [&]() {
      return Subtarget.hasSALUFloatInsts() && (Size == 32 || Size == 16);
    };

    bool isICMP = MI.getOpcode() == RVGPU::G_ICMP;
    bool CanUseSCC = DstBank == RVGPU::SGPRRegBankID &&
                     Op2Bank == RVGPU::SGPRRegBankID &&
                     Op3Bank == RVGPU::SGPRRegBankID &&
                     (isICMP ? canUseSCCICMP() : canUseSCCFCMP());

    DstBank = CanUseSCC ? RVGPU::SGPRRegBankID : RVGPU::VCCRegBankID;
    unsigned SrcBank = CanUseSCC ? RVGPU::SGPRRegBankID : RVGPU::VGPRRegBankID;

    // TODO: Use 32-bit for scalar output size.
    // SCC results will need to be copied to a 32-bit SGPR virtual register.
    const unsigned ResultSize = 1;

    OpdsMapping[0] = RVGPU::getValueMapping(DstBank, ResultSize);
    OpdsMapping[1] = nullptr; // Predicate Operand.
    OpdsMapping[2] = RVGPU::getValueMapping(SrcBank, Size);
    OpdsMapping[3] = RVGPU::getValueMapping(SrcBank, Size);
    break;
  }
  case RVGPU::G_EXTRACT_VECTOR_ELT: {
    // VGPR index can be used for waterfall when indexing a SGPR vector.
    unsigned SrcBankID = getRegBankID(MI.getOperand(1).getReg(), MRI);
    unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    unsigned SrcSize = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
    unsigned IdxSize = MRI.getType(MI.getOperand(2).getReg()).getSizeInBits();
    unsigned IdxBank = getRegBankID(MI.getOperand(2).getReg(), MRI);
    unsigned OutputBankID = regBankUnion(SrcBankID, IdxBank);

    OpdsMapping[0] = RVGPU::getValueMappingSGPR64Only(OutputBankID, DstSize);
    OpdsMapping[1] = RVGPU::getValueMapping(SrcBankID, SrcSize);

    // The index can be either if the source vector is VGPR.
    OpdsMapping[2] = RVGPU::getValueMapping(IdxBank, IdxSize);
    break;
  }
  case RVGPU::G_INSERT_VECTOR_ELT: {
    unsigned OutputBankID = isSALUMapping(MI) ?
      RVGPU::SGPRRegBankID : RVGPU::VGPRRegBankID;

    unsigned VecSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    unsigned InsertSize = MRI.getType(MI.getOperand(2).getReg()).getSizeInBits();
    unsigned IdxSize = MRI.getType(MI.getOperand(3).getReg()).getSizeInBits();
    unsigned InsertEltBankID = getRegBankID(MI.getOperand(2).getReg(), MRI);
    unsigned IdxBankID = getRegBankID(MI.getOperand(3).getReg(), MRI);

    OpdsMapping[0] = RVGPU::getValueMapping(OutputBankID, VecSize);
    OpdsMapping[1] = RVGPU::getValueMapping(OutputBankID, VecSize);

    // This is a weird case, because we need to break down the mapping based on
    // the register bank of a different operand.
    if (InsertSize == 64 && OutputBankID == RVGPU::VGPRRegBankID) {
      OpdsMapping[2] = RVGPU::getValueMappingSplit64(InsertEltBankID,
                                                      InsertSize);
    } else {
      assert(InsertSize == 32 || InsertSize == 64);
      OpdsMapping[2] = RVGPU::getValueMapping(InsertEltBankID, InsertSize);
    }

    // The index can be either if the source vector is VGPR.
    OpdsMapping[3] = RVGPU::getValueMapping(IdxBankID, IdxSize);
    break;
  }
  case RVGPU::G_UNMERGE_VALUES: {
    unsigned Bank = getMappingType(MRI, MI);

    // Op1 and Dst should use the same register bank.
    // FIXME: Shouldn't this be the default? Why do we need to handle this?
    for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
      unsigned Size = getSizeInBits(MI.getOperand(i).getReg(), MRI, *TRI);
      OpdsMapping[i] = RVGPU::getValueMapping(Bank, Size);
    }
    break;
  }
  case RVGPU::G_RVGPU_BUFFER_LOAD:
  case RVGPU::G_RVGPU_BUFFER_LOAD_UBYTE:
  case RVGPU::G_RVGPU_BUFFER_LOAD_SBYTE:
  case RVGPU::G_RVGPU_BUFFER_LOAD_USHORT:
  case RVGPU::G_RVGPU_BUFFER_LOAD_SSHORT:
  case RVGPU::G_RVGPU_BUFFER_LOAD_FORMAT:
  case RVGPU::G_RVGPU_BUFFER_LOAD_FORMAT_TFE:
  case RVGPU::G_RVGPU_BUFFER_LOAD_FORMAT_D16:
  case RVGPU::G_RVGPU_TBUFFER_LOAD_FORMAT:
  case RVGPU::G_RVGPU_TBUFFER_LOAD_FORMAT_D16:
  case RVGPU::G_RVGPU_TBUFFER_STORE_FORMAT:
  case RVGPU::G_RVGPU_TBUFFER_STORE_FORMAT_D16:
  case RVGPU::G_RVGPU_BUFFER_STORE:
  case RVGPU::G_RVGPU_BUFFER_STORE_BYTE:
  case RVGPU::G_RVGPU_BUFFER_STORE_SHORT:
  case RVGPU::G_RVGPU_BUFFER_STORE_FORMAT:
  case RVGPU::G_RVGPU_BUFFER_STORE_FORMAT_D16: {
    OpdsMapping[0] = getVGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI);

    // rsrc
    OpdsMapping[1] = getSGPROpMapping(MI.getOperand(1).getReg(), MRI, *TRI);

    // vindex
    OpdsMapping[2] = getVGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);

    // voffset
    OpdsMapping[3] = getVGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);

    // soffset
    OpdsMapping[4] = getSGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);

    // Any remaining operands are immediates and were correctly null
    // initialized.
    break;
  }
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_SWAP:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_ADD:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_SUB:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_SMIN:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_UMIN:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_SMAX:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_UMAX:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_AND:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_OR:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_XOR:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_INC:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_DEC:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_FADD:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_FMIN:
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_FMAX: {
    // vdata_out
    OpdsMapping[0] = getVGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI);

    // vdata_in
    OpdsMapping[1] = getVGPROpMapping(MI.getOperand(1).getReg(), MRI, *TRI);

    // rsrc
    OpdsMapping[2] = getSGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);

    // vindex
    OpdsMapping[3] = getVGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);

    // voffset
    OpdsMapping[4] = getVGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);

    // soffset
    OpdsMapping[5] = getSGPROpMapping(MI.getOperand(5).getReg(), MRI, *TRI);

    // Any remaining operands are immediates and were correctly null
    // initialized.
    break;
  }
  case RVGPU::G_RVGPU_BUFFER_ATOMIC_CMPSWAP: {
    // vdata_out
    OpdsMapping[0] = getVGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI);

    // vdata_in
    OpdsMapping[1] = getVGPROpMapping(MI.getOperand(1).getReg(), MRI, *TRI);

    // cmp
    OpdsMapping[2] = getVGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);

    // rsrc
    OpdsMapping[3] = getSGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);

    // vindex
    OpdsMapping[4] = getVGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);

    // voffset
    OpdsMapping[5] = getVGPROpMapping(MI.getOperand(5).getReg(), MRI, *TRI);

    // soffset
    OpdsMapping[6] = getSGPROpMapping(MI.getOperand(6).getReg(), MRI, *TRI);

    // Any remaining operands are immediates and were correctly null
    // initialized.
    break;
  }
  case RVGPU::G_RVGPU_S_BUFFER_LOAD: {
    // Lie and claim everything is legal, even though some need to be
    // SGPRs. applyMapping will have to deal with it as a waterfall loop.
    OpdsMapping[1] = getSGPROpMapping(MI.getOperand(1).getReg(), MRI, *TRI);
    OpdsMapping[2] = getSGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);

    // We need to convert this to a MUBUF if either the resource of offset is
    // VGPR.
    unsigned RSrcBank = OpdsMapping[1]->BreakDown[0].RegBank->getID();
    unsigned OffsetBank = OpdsMapping[2]->BreakDown[0].RegBank->getID();
    unsigned ResultBank = regBankUnion(RSrcBank, OffsetBank);

    unsigned Size0 = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    OpdsMapping[0] = RVGPU::getValueMapping(ResultBank, Size0);
    break;
  }
  case RVGPU::G_INTRINSIC:
  case RVGPU::G_INTRINSIC_CONVERGENT: {
    switch (cast<GIntrinsic>(MI).getIntrinsicID()) {
    default:
      return getInvalidInstructionMapping();
    case Intrinsic::rvgpu_div_fmas:
    case Intrinsic::rvgpu_div_fixup:
    case Intrinsic::rvgpu_trig_preop:
    case Intrinsic::rvgpu_sin:
    case Intrinsic::rvgpu_cos:
    case Intrinsic::rvgpu_log_clamp:
    case Intrinsic::rvgpu_rcp_legacy:
    case Intrinsic::rvgpu_rsq_legacy:
    case Intrinsic::rvgpu_rsq_clamp:
    case Intrinsic::rvgpu_fmul_legacy:
    case Intrinsic::rvgpu_fma_legacy:
    case Intrinsic::rvgpu_frexp_mant:
    case Intrinsic::rvgpu_frexp_exp:
    case Intrinsic::rvgpu_fract:
    case Intrinsic::rvgpu_cvt_pknorm_i16:
    case Intrinsic::rvgpu_cvt_pknorm_u16:
    case Intrinsic::rvgpu_cvt_pk_i16:
    case Intrinsic::rvgpu_cvt_pk_u16:
    case Intrinsic::rvgpu_fmed3:
    case Intrinsic::rvgpu_cubeid:
    case Intrinsic::rvgpu_cubema:
    case Intrinsic::rvgpu_cubesc:
    case Intrinsic::rvgpu_cubetc:
    case Intrinsic::rvgpu_sffbh:
    case Intrinsic::rvgpu_fmad_ftz:
    case Intrinsic::rvgpu_mbcnt_lo:
    case Intrinsic::rvgpu_mbcnt_hi:
    case Intrinsic::rvgpu_mul_u24:
    case Intrinsic::rvgpu_mul_i24:
    case Intrinsic::rvgpu_mulhi_u24:
    case Intrinsic::rvgpu_mulhi_i24:
    case Intrinsic::rvgpu_lerp:
    case Intrinsic::rvgpu_sad_u8:
    case Intrinsic::rvgpu_msad_u8:
    case Intrinsic::rvgpu_sad_hi_u8:
    case Intrinsic::rvgpu_sad_u16:
    case Intrinsic::rvgpu_qsad_pk_u16_u8:
    case Intrinsic::rvgpu_mqsad_pk_u16_u8:
    case Intrinsic::rvgpu_mqsad_u32_u8:
    case Intrinsic::rvgpu_cvt_pk_u8_f32:
    case Intrinsic::rvgpu_alignbyte:
    case Intrinsic::rvgpu_perm:
    case Intrinsic::rvgpu_fdot2:
    case Intrinsic::rvgpu_sdot2:
    case Intrinsic::rvgpu_udot2:
    case Intrinsic::rvgpu_sdot4:
    case Intrinsic::rvgpu_udot4:
    case Intrinsic::rvgpu_sdot8:
    case Intrinsic::rvgpu_udot8:
    case Intrinsic::rvgpu_fdot2_bf16_bf16:
    case Intrinsic::rvgpu_fdot2_f16_f16:
    case Intrinsic::rvgpu_fdot2_f32_bf16:
    case Intrinsic::rvgpu_sudot4:
    case Intrinsic::rvgpu_sudot8:
    case Intrinsic::rvgpu_wmma_bf16_16x16x16_bf16:
    case Intrinsic::rvgpu_wmma_f16_16x16x16_f16:
    case Intrinsic::rvgpu_wmma_bf16_16x16x16_bf16_tied:
    case Intrinsic::rvgpu_wmma_f16_16x16x16_f16_tied:
    case Intrinsic::rvgpu_wmma_f32_16x16x16_bf16:
    case Intrinsic::rvgpu_wmma_f32_16x16x16_f16:
    case Intrinsic::rvgpu_wmma_i32_16x16x16_iu4:
    case Intrinsic::rvgpu_wmma_i32_16x16x16_iu8:
      return getDefaultMappingVOP(MI);
    case Intrinsic::rvgpu_log:
    case Intrinsic::rvgpu_exp2:
    case Intrinsic::rvgpu_rcp:
    case Intrinsic::rvgpu_rsq:
    case Intrinsic::rvgpu_sqrt: {
      unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      if (Subtarget.hasPseudoScalarTrans() && (Size == 16 || Size == 32) &&
          isSALUMapping(MI))
        return getDefaultMappingSOP(MI);
      return getDefaultMappingVOP(MI);
    }
    case Intrinsic::rvgpu_sbfe:
    case Intrinsic::rvgpu_ubfe:
      if (isSALUMapping(MI))
        return getDefaultMappingSOP(MI);
      return getDefaultMappingVOP(MI);
    case Intrinsic::rvgpu_ds_swizzle:
    case Intrinsic::rvgpu_ds_permute:
    case Intrinsic::rvgpu_ds_bpermute:
    case Intrinsic::rvgpu_update_dpp:
    case Intrinsic::rvgpu_mov_dpp8:
    case Intrinsic::rvgpu_mov_dpp:
    case Intrinsic::rvgpu_strict_wwm:
    case Intrinsic::rvgpu_wwm:
    case Intrinsic::rvgpu_strict_wqm:
    case Intrinsic::rvgpu_wqm:
    case Intrinsic::rvgpu_softwqm:
    case Intrinsic::rvgpu_set_inactive:
    case Intrinsic::rvgpu_set_inactive_chain_arg:
    case Intrinsic::rvgpu_permlane64:
      return getDefaultMappingAllVGPR(MI);
    case Intrinsic::rvgpu_cvt_pkrtz:
      if (Subtarget.hasSALUFloatInsts() && isSALUMapping(MI))
        return getDefaultMappingSOP(MI);
      return getDefaultMappingVOP(MI);
    case Intrinsic::rvgpu_kernarg_segment_ptr:
    case Intrinsic::rvgpu_s_getpc:
    case Intrinsic::rvgpu_groupstaticsize:
    case Intrinsic::rvgpu_reloc_constant:
    case Intrinsic::returnaddress: {
      unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, Size);
      break;
    }
    case Intrinsic::rvgpu_wqm_vote: {
      unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      OpdsMapping[0] = OpdsMapping[2]
        = RVGPU::getValueMapping(RVGPU::VCCRegBankID, Size);
      break;
    }
    case Intrinsic::rvgpu_ps_live: {
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::VCCRegBankID, 1);
      break;
    }
    case Intrinsic::rvgpu_div_scale: {
      unsigned Dst0Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      unsigned Dst1Size = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Dst0Size);
      OpdsMapping[1] = RVGPU::getValueMapping(RVGPU::VCCRegBankID, Dst1Size);

      unsigned SrcSize = MRI.getType(MI.getOperand(3).getReg()).getSizeInBits();
      OpdsMapping[3] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, SrcSize);
      OpdsMapping[4] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, SrcSize);
      break;
    }
    case Intrinsic::rvgpu_class: {
      Register Src0Reg = MI.getOperand(2).getReg();
      Register Src1Reg = MI.getOperand(3).getReg();
      unsigned Src0Size = MRI.getType(Src0Reg).getSizeInBits();
      unsigned Src1Size = MRI.getType(Src1Reg).getSizeInBits();
      unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::VCCRegBankID, DstSize);
      OpdsMapping[2] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Src0Size);
      OpdsMapping[3] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Src1Size);
      break;
    }
    case Intrinsic::rvgpu_icmp:
    case Intrinsic::rvgpu_fcmp: {
      unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      // This is not VCCRegBank because this is not used in boolean contexts.
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, DstSize);
      unsigned OpSize = MRI.getType(MI.getOperand(2).getReg()).getSizeInBits();
      OpdsMapping[2] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, OpSize);
      OpdsMapping[3] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, OpSize);
      break;
    }
    case Intrinsic::rvgpu_readlane: {
      // This must be an SGPR, but accept a VGPR.
      Register IdxReg = MI.getOperand(3).getReg();
      unsigned IdxSize = MRI.getType(IdxReg).getSizeInBits();
      unsigned IdxBank = getRegBankID(IdxReg, MRI, RVGPU::SGPRRegBankID);
      OpdsMapping[3] = RVGPU::getValueMapping(IdxBank, IdxSize);
      [[fallthrough]];
    }
    case Intrinsic::rvgpu_readfirstlane: {
      unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      unsigned SrcSize = MRI.getType(MI.getOperand(2).getReg()).getSizeInBits();
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, DstSize);
      OpdsMapping[2] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, SrcSize);
      break;
    }
    case Intrinsic::rvgpu_writelane: {
      unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      Register SrcReg = MI.getOperand(2).getReg();
      unsigned SrcSize = MRI.getType(SrcReg).getSizeInBits();
      unsigned SrcBank = getRegBankID(SrcReg, MRI, RVGPU::SGPRRegBankID);
      Register IdxReg = MI.getOperand(3).getReg();
      unsigned IdxSize = MRI.getType(IdxReg).getSizeInBits();
      unsigned IdxBank = getRegBankID(IdxReg, MRI, RVGPU::SGPRRegBankID);
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, DstSize);

      // These 2 must be SGPRs, but accept VGPRs. Readfirstlane will be inserted
      // to legalize.
      OpdsMapping[2] = RVGPU::getValueMapping(SrcBank, SrcSize);
      OpdsMapping[3] = RVGPU::getValueMapping(IdxBank, IdxSize);
      OpdsMapping[4] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, SrcSize);
      break;
    }
    case Intrinsic::rvgpu_if_break: {
      unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, Size);
      OpdsMapping[2] = RVGPU::getValueMapping(RVGPU::VCCRegBankID, 1);
      OpdsMapping[3] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, Size);
      break;
    }
    case Intrinsic::rvgpu_permlane16:
    case Intrinsic::rvgpu_permlanex16: {
      unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size);
      OpdsMapping[2] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size);
      OpdsMapping[3] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size);
      OpdsMapping[4] = getSGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);
      OpdsMapping[5] = getSGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);
      break;
    }
    case Intrinsic::rvgpu_permlane16_var:
    case Intrinsic::rvgpu_permlanex16_var: {
      unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size);
      OpdsMapping[2] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size);
      OpdsMapping[3] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size);
      OpdsMapping[4] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size);
      break;
    }
    case Intrinsic::rvgpu_mfma_f32_4x4x1f32:
    case Intrinsic::rvgpu_mfma_f32_4x4x4f16:
    case Intrinsic::rvgpu_mfma_i32_4x4x4i8:
    case Intrinsic::rvgpu_mfma_f32_4x4x2bf16:
    case Intrinsic::rvgpu_mfma_f32_16x16x1f32:
    case Intrinsic::rvgpu_mfma_f32_16x16x4f32:
    case Intrinsic::rvgpu_mfma_f32_16x16x4f16:
    case Intrinsic::rvgpu_mfma_f32_16x16x16f16:
    case Intrinsic::rvgpu_mfma_i32_16x16x4i8:
    case Intrinsic::rvgpu_mfma_i32_16x16x16i8:
    case Intrinsic::rvgpu_mfma_f32_16x16x2bf16:
    case Intrinsic::rvgpu_mfma_f32_16x16x8bf16:
    case Intrinsic::rvgpu_mfma_f32_32x32x1f32:
    case Intrinsic::rvgpu_mfma_f32_32x32x2f32:
    case Intrinsic::rvgpu_mfma_f32_32x32x4f16:
    case Intrinsic::rvgpu_mfma_f32_32x32x8f16:
    case Intrinsic::rvgpu_mfma_i32_32x32x4i8:
    case Intrinsic::rvgpu_mfma_i32_32x32x8i8:
    case Intrinsic::rvgpu_mfma_f32_32x32x2bf16:
    case Intrinsic::rvgpu_mfma_f32_32x32x4bf16:
    case Intrinsic::rvgpu_mfma_f32_32x32x4bf16_1k:
    case Intrinsic::rvgpu_mfma_f32_16x16x4bf16_1k:
    case Intrinsic::rvgpu_mfma_f32_4x4x4bf16_1k:
    case Intrinsic::rvgpu_mfma_f32_32x32x8bf16_1k:
    case Intrinsic::rvgpu_mfma_f32_16x16x16bf16_1k:
    case Intrinsic::rvgpu_mfma_f64_16x16x4f64:
    case Intrinsic::rvgpu_mfma_f64_4x4x4f64:
    case Intrinsic::rvgpu_mfma_i32_16x16x32_i8:
    case Intrinsic::rvgpu_mfma_i32_32x32x16_i8:
    case Intrinsic::rvgpu_mfma_f32_16x16x8_xf32:
    case Intrinsic::rvgpu_mfma_f32_32x32x4_xf32:
    case Intrinsic::rvgpu_mfma_f32_16x16x32_bf8_bf8:
    case Intrinsic::rvgpu_mfma_f32_16x16x32_bf8_fp8:
    case Intrinsic::rvgpu_mfma_f32_16x16x32_fp8_bf8:
    case Intrinsic::rvgpu_mfma_f32_16x16x32_fp8_fp8:
    case Intrinsic::rvgpu_mfma_f32_32x32x16_bf8_bf8:
    case Intrinsic::rvgpu_mfma_f32_32x32x16_bf8_fp8:
    case Intrinsic::rvgpu_mfma_f32_32x32x16_fp8_bf8:
    case Intrinsic::rvgpu_mfma_f32_32x32x16_fp8_fp8: {
      // Default for MAI intrinsics.
      // srcC can also be an immediate which can be folded later.
      // FIXME: Should we eventually add an alternative mapping with AGPR src
      // for srcA/srcB?
      //
      // vdst, srcA, srcB, srcC
      const RVMachineFunctionInfo *Info = MF.getInfo<RVMachineFunctionInfo>();
      OpdsMapping[0] =
          Info->mayNeedAGPRs()
              ? getAGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI)
              : getVGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI);
      OpdsMapping[2] = getVGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
      OpdsMapping[3] = getVGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);
      OpdsMapping[4] =
          Info->mayNeedAGPRs()
              ? getAGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI)
              : getVGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);
      break;
    }
    case Intrinsic::rvgpu_smfmac_f32_16x16x32_f16:
    case Intrinsic::rvgpu_smfmac_f32_32x32x16_f16:
    case Intrinsic::rvgpu_smfmac_f32_16x16x32_bf16:
    case Intrinsic::rvgpu_smfmac_f32_32x32x16_bf16:
    case Intrinsic::rvgpu_smfmac_i32_16x16x64_i8:
    case Intrinsic::rvgpu_smfmac_i32_32x32x32_i8:
    case Intrinsic::rvgpu_smfmac_f32_16x16x64_bf8_bf8:
    case Intrinsic::rvgpu_smfmac_f32_16x16x64_bf8_fp8:
    case Intrinsic::rvgpu_smfmac_f32_16x16x64_fp8_bf8:
    case Intrinsic::rvgpu_smfmac_f32_16x16x64_fp8_fp8:
    case Intrinsic::rvgpu_smfmac_f32_32x32x32_bf8_bf8:
    case Intrinsic::rvgpu_smfmac_f32_32x32x32_bf8_fp8:
    case Intrinsic::rvgpu_smfmac_f32_32x32x32_fp8_bf8:
    case Intrinsic::rvgpu_smfmac_f32_32x32x32_fp8_fp8: {
      // vdst, srcA, srcB, srcC, idx
      OpdsMapping[0] = getAGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI);
      OpdsMapping[2] = getVGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
      OpdsMapping[3] = getVGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);
      OpdsMapping[4] = getAGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);
      OpdsMapping[5] = getVGPROpMapping(MI.getOperand(5).getReg(), MRI, *TRI);
      break;
    }
    case Intrinsic::rvgpu_interp_p1:
    case Intrinsic::rvgpu_interp_p2:
    case Intrinsic::rvgpu_interp_mov:
    case Intrinsic::rvgpu_interp_p1_f16:
    case Intrinsic::rvgpu_interp_p2_f16:
    case Intrinsic::rvgpu_lds_param_load: {
      const int M0Idx = MI.getNumOperands() - 1;
      Register M0Reg = MI.getOperand(M0Idx).getReg();
      unsigned M0Bank = getRegBankID(M0Reg, MRI, RVGPU::SGPRRegBankID);
      unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();

      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, DstSize);
      for (int I = 2; I != M0Idx && MI.getOperand(I).isReg(); ++I)
        OpdsMapping[I] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, 32);

      // Must be SGPR, but we must take whatever the original bank is and fix it
      // later.
      OpdsMapping[M0Idx] = RVGPU::getValueMapping(M0Bank, 32);
      break;
    }
    case Intrinsic::rvgpu_interp_inreg_p10:
    case Intrinsic::rvgpu_interp_inreg_p2:
    case Intrinsic::rvgpu_interp_inreg_p10_f16:
    case Intrinsic::rvgpu_interp_inreg_p2_f16: {
      unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, DstSize);
      OpdsMapping[2] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, 32);
      OpdsMapping[3] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, 32);
      OpdsMapping[4] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, 32);
      break;
    }
    case Intrinsic::rvgpu_ballot: {
      unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      unsigned SrcSize = MRI.getType(MI.getOperand(2).getReg()).getSizeInBits();
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, DstSize);
      OpdsMapping[2] = RVGPU::getValueMapping(RVGPU::VCCRegBankID, SrcSize);
      break;
    }
    case Intrinsic::rvgpu_inverse_ballot: {
      // This must be an SGPR, but accept a VGPR.
      Register MaskReg = MI.getOperand(2).getReg();
      unsigned MaskSize = MRI.getType(MaskReg).getSizeInBits();
      unsigned MaskBank = getRegBankID(MaskReg, MRI, RVGPU::SGPRRegBankID);
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::VCCRegBankID, 1);
      OpdsMapping[2] = RVGPU::getValueMapping(MaskBank, MaskSize);
      break;
    }
    case Intrinsic::rvgpu_s_quadmask:
    case Intrinsic::rvgpu_s_wqm: {
      Register MaskReg = MI.getOperand(2).getReg();
      unsigned MaskSize = MRI.getType(MaskReg).getSizeInBits();
      unsigned MaskBank = getRegBankID(MaskReg, MRI, RVGPU::SGPRRegBankID);
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, MaskSize);
      OpdsMapping[2] = RVGPU::getValueMapping(MaskBank, MaskSize);
      break;
    }
    case Intrinsic::rvgpu_wave_reduce_umin:
    case Intrinsic::rvgpu_wave_reduce_umax: {
      unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, DstSize);
      unsigned OpSize = MRI.getType(MI.getOperand(2).getReg()).getSizeInBits();
      auto regBankID =
          isSALUMapping(MI) ? RVGPU::SGPRRegBankID : RVGPU::VGPRRegBankID;
      OpdsMapping[2] = RVGPU::getValueMapping(regBankID, OpSize);
      break;
    }
    case Intrinsic::rvgpu_s_bitreplicate:
      Register MaskReg = MI.getOperand(2).getReg();
      unsigned MaskBank = getRegBankID(MaskReg, MRI, RVGPU::SGPRRegBankID);
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, 64);
      OpdsMapping[2] = RVGPU::getValueMapping(MaskBank, 32);
    }
    break;
  }
  case RVGPU::G_RVGPU_INTRIN_IMAGE_LOAD:
  case RVGPU::G_RVGPU_INTRIN_IMAGE_LOAD_D16:
  case RVGPU::G_RVGPU_INTRIN_IMAGE_STORE:
  case RVGPU::G_RVGPU_INTRIN_IMAGE_STORE_D16: {
    auto IntrID = RVGPU::getIntrinsicID(MI);
    const RVGPU::RsrcIntrinsic *RSrcIntrin = RVGPU::lookupRsrcIntrinsic(IntrID);
    assert(RSrcIntrin && "missing RsrcIntrinsic for image intrinsic");
    // Non-images can have complications from operands that allow both SGPR
    // and VGPR. For now it's too complicated to figure out the final opcode
    // to derive the register bank from the MCInstrDesc.
    assert(RSrcIntrin->IsImage);
    return getImageMapping(MRI, MI, RSrcIntrin->RsrcArg);
  }
  case RVGPU::G_RVGPU_INTRIN_BVH_INTERSECT_RAY: {
    unsigned N = MI.getNumExplicitOperands() - 2;
    OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, 128);
    OpdsMapping[N] = getSGPROpMapping(MI.getOperand(N).getReg(), MRI, *TRI);
    if (N == 3) {
      // Sequential form: all operands combined into VGPR256/VGPR512
      unsigned Size = MRI.getType(MI.getOperand(2).getReg()).getSizeInBits();
      if (Size > 256)
        Size = 512;
      OpdsMapping[2] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size);
    } else {
      // NSA form
      for (unsigned I = 2; I < N; ++I) {
        unsigned Size = MRI.getType(MI.getOperand(I).getReg()).getSizeInBits();
        OpdsMapping[I] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, Size);
      }
    }
    break;
  }
  case RVGPU::G_INTRINSIC_W_SIDE_EFFECTS:
  case RVGPU::G_INTRINSIC_CONVERGENT_W_SIDE_EFFECTS: {
    auto IntrID = cast<GIntrinsic>(MI).getIntrinsicID();
    switch (IntrID) {
    case Intrinsic::rvgpu_s_getreg:
    case Intrinsic::rvgpu_s_memtime:
    case Intrinsic::rvgpu_s_memrealtime:
    case Intrinsic::rvgpu_s_get_waveid_in_workgroup:
    case Intrinsic::rvgpu_s_sendmsg_rtn: {
      unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, Size);
      break;
    }
    case Intrinsic::rvgpu_global_atomic_fadd:
    case Intrinsic::rvgpu_global_atomic_csub:
    case Intrinsic::rvgpu_global_atomic_fmin:
    case Intrinsic::rvgpu_global_atomic_fmax:
    case Intrinsic::rvgpu_global_atomic_fmin_num:
    case Intrinsic::rvgpu_global_atomic_fmax_num:
    case Intrinsic::rvgpu_flat_atomic_fadd:
    case Intrinsic::rvgpu_flat_atomic_fmin:
    case Intrinsic::rvgpu_flat_atomic_fmax:
    case Intrinsic::rvgpu_flat_atomic_fmin_num:
    case Intrinsic::rvgpu_flat_atomic_fmax_num:
    case Intrinsic::rvgpu_global_atomic_fadd_v2bf16:
    case Intrinsic::rvgpu_flat_atomic_fadd_v2bf16:
      return getDefaultMappingAllVGPR(MI);
    case Intrinsic::rvgpu_ds_ordered_add:
    case Intrinsic::rvgpu_ds_ordered_swap:
    case Intrinsic::rvgpu_ds_fadd_v2bf16: {
      unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, DstSize);
      unsigned M0Bank = getRegBankID(MI.getOperand(2).getReg(), MRI,
                                 RVGPU::SGPRRegBankID);
      OpdsMapping[2] = RVGPU::getValueMapping(M0Bank, 32);
      OpdsMapping[3] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, 32);
      break;
    }
    case Intrinsic::rvgpu_ds_append:
    case Intrinsic::rvgpu_ds_consume: {
      unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, DstSize);
      OpdsMapping[2] = getSGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
      break;
    }
    case Intrinsic::rvgpu_exp_compr:
      OpdsMapping[3] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, 32);
      OpdsMapping[4] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, 32);
      break;
    case Intrinsic::rvgpu_exp:
      // FIXME: Could we support packed types here?
      OpdsMapping[3] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, 32);
      OpdsMapping[4] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, 32);
      OpdsMapping[5] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, 32);
      OpdsMapping[6] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, 32);
      break;
    case Intrinsic::rvgpu_exp_row:
      OpdsMapping[3] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, 32);
      OpdsMapping[4] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, 32);
      OpdsMapping[5] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, 32);
      OpdsMapping[6] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, 32);
      OpdsMapping[8] = getSGPROpMapping(MI.getOperand(8).getReg(), MRI, *TRI);
      break;
    case Intrinsic::rvgpu_s_sendmsg:
    case Intrinsic::rvgpu_s_sendmsghalt: {
      // This must be an SGPR, but accept a VGPR.
      unsigned Bank = getRegBankID(MI.getOperand(2).getReg(), MRI,
                                   RVGPU::SGPRRegBankID);
      OpdsMapping[2] = RVGPU::getValueMapping(Bank, 32);
      break;
    }
    case Intrinsic::rvgpu_s_setreg: {
      // This must be an SGPR, but accept a VGPR.
      unsigned Bank = getRegBankID(MI.getOperand(2).getReg(), MRI,
                                   RVGPU::SGPRRegBankID);
      OpdsMapping[2] = RVGPU::getValueMapping(Bank, 32);
      break;
    }
    case Intrinsic::rvgpu_s_ttracedata: {
      // This must be an SGPR, but accept a VGPR.
      unsigned Bank =
          getRegBankID(MI.getOperand(1).getReg(), MRI, RVGPU::SGPRRegBankID);
      OpdsMapping[1] = RVGPU::getValueMapping(Bank, 32);
      break;
    }
    case Intrinsic::rvgpu_end_cf: {
      unsigned Size = getSizeInBits(MI.getOperand(1).getReg(), MRI, *TRI);
      OpdsMapping[1] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, Size);
      break;
    }
    case Intrinsic::rvgpu_else: {
      unsigned WaveSize = getSizeInBits(MI.getOperand(1).getReg(), MRI, *TRI);
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::VCCRegBankID, 1);
      OpdsMapping[1] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, WaveSize);
      OpdsMapping[3] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, WaveSize);
      break;
    }
    case Intrinsic::rvgpu_live_mask: {
      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::VCCRegBankID, 1);
      break;
    }
    case Intrinsic::rvgpu_wqm_demote:
    case Intrinsic::rvgpu_kill: {
      OpdsMapping[1] = RVGPU::getValueMapping(RVGPU::VCCRegBankID, 1);
      break;
    }
    case Intrinsic::rvgpu_raw_buffer_load:
    case Intrinsic::rvgpu_raw_ptr_buffer_load:
    case Intrinsic::rvgpu_raw_tbuffer_load:
    case Intrinsic::rvgpu_raw_ptr_tbuffer_load: {
      // FIXME: Should make intrinsic ID the last operand of the instruction,
      // then this would be the same as store
      OpdsMapping[0] = getVGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI);
      OpdsMapping[2] = getSGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
      OpdsMapping[3] = getVGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);
      OpdsMapping[4] = getSGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);
      break;
    }
    case Intrinsic::rvgpu_raw_buffer_load_lds:
    case Intrinsic::rvgpu_raw_ptr_buffer_load_lds: {
      OpdsMapping[1] = getSGPROpMapping(MI.getOperand(1).getReg(), MRI, *TRI);
      OpdsMapping[2] = getSGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
      OpdsMapping[4] = getVGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);
      OpdsMapping[5] = getSGPROpMapping(MI.getOperand(5).getReg(), MRI, *TRI);
      break;
    }
    case Intrinsic::rvgpu_raw_buffer_store:
    case Intrinsic::rvgpu_raw_ptr_buffer_store:
    case Intrinsic::rvgpu_raw_buffer_store_format:
    case Intrinsic::rvgpu_raw_ptr_buffer_store_format:
    case Intrinsic::rvgpu_raw_tbuffer_store:
    case Intrinsic::rvgpu_raw_ptr_tbuffer_store: {
      OpdsMapping[1] = getVGPROpMapping(MI.getOperand(1).getReg(), MRI, *TRI);
      OpdsMapping[2] = getSGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
      OpdsMapping[3] = getVGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);
      OpdsMapping[4] = getSGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);
      break;
    }
    case Intrinsic::rvgpu_struct_buffer_load:
    case Intrinsic::rvgpu_struct_ptr_buffer_load:
    case Intrinsic::rvgpu_struct_tbuffer_load:
    case Intrinsic::rvgpu_struct_ptr_tbuffer_load: {
      OpdsMapping[0] = getVGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI);
      OpdsMapping[2] = getSGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
      OpdsMapping[3] = getVGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);
      OpdsMapping[4] = getVGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);
      OpdsMapping[5] = getSGPROpMapping(MI.getOperand(5).getReg(), MRI, *TRI);
      break;
    }
    case Intrinsic::rvgpu_struct_buffer_load_lds:
    case Intrinsic::rvgpu_struct_ptr_buffer_load_lds: {
      OpdsMapping[1] = getSGPROpMapping(MI.getOperand(1).getReg(), MRI, *TRI);
      OpdsMapping[2] = getSGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
      OpdsMapping[4] = getVGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);
      OpdsMapping[5] = getVGPROpMapping(MI.getOperand(5).getReg(), MRI, *TRI);
      OpdsMapping[6] = getSGPROpMapping(MI.getOperand(6).getReg(), MRI, *TRI);
      break;
    }
    case Intrinsic::rvgpu_struct_buffer_store:
    case Intrinsic::rvgpu_struct_ptr_buffer_store:
    case Intrinsic::rvgpu_struct_tbuffer_store:
    case Intrinsic::rvgpu_struct_ptr_tbuffer_store: {
      OpdsMapping[1] = getVGPROpMapping(MI.getOperand(1).getReg(), MRI, *TRI);
      OpdsMapping[2] = getSGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
      OpdsMapping[3] = getVGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);
      OpdsMapping[4] = getVGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);
      OpdsMapping[5] = getSGPROpMapping(MI.getOperand(5).getReg(), MRI, *TRI);
      break;
    }
    case Intrinsic::rvgpu_init_exec_from_input: {
      unsigned Size = getSizeInBits(MI.getOperand(1).getReg(), MRI, *TRI);
      OpdsMapping[1] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, Size);
      break;
    }
    case Intrinsic::rvgpu_ds_gws_init:
    case Intrinsic::rvgpu_ds_gws_barrier:
    case Intrinsic::rvgpu_ds_gws_sema_br: {
      OpdsMapping[1] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, 32);

      // This must be an SGPR, but accept a VGPR.
      unsigned Bank = getRegBankID(MI.getOperand(2).getReg(), MRI,
                                   RVGPU::SGPRRegBankID);
      OpdsMapping[2] = RVGPU::getValueMapping(Bank, 32);
      break;
    }
    case Intrinsic::rvgpu_ds_gws_sema_v:
    case Intrinsic::rvgpu_ds_gws_sema_p:
    case Intrinsic::rvgpu_ds_gws_sema_release_all: {
      // This must be an SGPR, but accept a VGPR.
      unsigned Bank = getRegBankID(MI.getOperand(1).getReg(), MRI,
                                   RVGPU::SGPRRegBankID);
      OpdsMapping[1] = RVGPU::getValueMapping(Bank, 32);
      break;
    }
    case Intrinsic::rvgpu_global_load_lds: {
      OpdsMapping[1] = getVGPROpMapping(MI.getOperand(1).getReg(), MRI, *TRI);
      OpdsMapping[2] = getSGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
      break;
    }
    case Intrinsic::rvgpu_lds_direct_load: {
      const int M0Idx = MI.getNumOperands() - 1;
      Register M0Reg = MI.getOperand(M0Idx).getReg();
      unsigned M0Bank = getRegBankID(M0Reg, MRI, RVGPU::SGPRRegBankID);
      unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();

      OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, DstSize);
      for (int I = 2; I != M0Idx && MI.getOperand(I).isReg(); ++I)
        OpdsMapping[I] = RVGPU::getValueMapping(RVGPU::VGPRRegBankID, 32);

      // Must be SGPR, but we must take whatever the original bank is and fix it
      // later.
      OpdsMapping[M0Idx] = RVGPU::getValueMapping(M0Bank, 32);
      break;
    }
    case Intrinsic::rvgpu_ds_add_gs_reg_rtn:
    case Intrinsic::rvgpu_ds_sub_gs_reg_rtn:
      OpdsMapping[0] = getVGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI);
      OpdsMapping[2] = getVGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
      break;
    case Intrinsic::rvgpu_ds_bvh_stack_rtn: {
      OpdsMapping[0] =
          getVGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI); // %vdst
      OpdsMapping[1] =
          getVGPROpMapping(MI.getOperand(1).getReg(), MRI, *TRI); // %addr
      OpdsMapping[3] =
          getVGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI); // %addr
      OpdsMapping[4] =
          getVGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI); // %data0
      OpdsMapping[5] =
          getVGPROpMapping(MI.getOperand(5).getReg(), MRI, *TRI); // %data1
      break;
    }
    case Intrinsic::rvgpu_s_sleep_var:
      OpdsMapping[1] = getSGPROpMapping(MI.getOperand(1).getReg(), MRI, *TRI);
      break;
    case Intrinsic::rvgpu_s_barrier_signal_var:
    case Intrinsic::rvgpu_s_barrier_join:
    case Intrinsic::rvgpu_s_wakeup_barrier:
      OpdsMapping[1] = getSGPROpMapping(MI.getOperand(1).getReg(), MRI, *TRI);
      break;
    case Intrinsic::rvgpu_s_barrier_init:
      OpdsMapping[1] = getSGPROpMapping(MI.getOperand(1).getReg(), MRI, *TRI);
      OpdsMapping[2] = getSGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
      break;
    case Intrinsic::rvgpu_s_barrier_signal_isfirst_var: {
      const unsigned ResultSize = 1;
      OpdsMapping[0] =
          RVGPU::getValueMapping(RVGPU::SGPRRegBankID, ResultSize);
      OpdsMapping[2] = getSGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
      break;
    }
    case Intrinsic::rvgpu_s_barrier_signal_isfirst:
    case Intrinsic::rvgpu_s_barrier_leave: {
      const unsigned ResultSize = 1;
      OpdsMapping[0] =
          RVGPU::getValueMapping(RVGPU::SGPRRegBankID, ResultSize);
      break;
    }
    case Intrinsic::rvgpu_s_get_barrier_state: {
      OpdsMapping[0] = getSGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI);
      OpdsMapping[2] = getSGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
      break;
    }
    default:
      return getInvalidInstructionMapping();
    }
    break;
  }
  case RVGPU::G_SELECT: {
    unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    unsigned Op2Bank = getRegBankID(MI.getOperand(2).getReg(), MRI,
                                    RVGPU::SGPRRegBankID);
    unsigned Op3Bank = getRegBankID(MI.getOperand(3).getReg(), MRI,
                                    RVGPU::SGPRRegBankID);
    bool SGPRSrcs = Op2Bank == RVGPU::SGPRRegBankID &&
                    Op3Bank == RVGPU::SGPRRegBankID;

    unsigned CondBankDefault = SGPRSrcs ?
      RVGPU::SGPRRegBankID : RVGPU::VCCRegBankID;
    unsigned CondBank = getRegBankID(MI.getOperand(1).getReg(), MRI,
                                     CondBankDefault);
    if (CondBank == RVGPU::SGPRRegBankID)
      CondBank = SGPRSrcs ? RVGPU::SGPRRegBankID : RVGPU::VCCRegBankID;
    else if (CondBank == RVGPU::VGPRRegBankID)
      CondBank = RVGPU::VCCRegBankID;

    unsigned Bank = SGPRSrcs && CondBank == RVGPU::SGPRRegBankID ?
      RVGPU::SGPRRegBankID : RVGPU::VGPRRegBankID;

    assert(CondBank == RVGPU::VCCRegBankID || CondBank == RVGPU::SGPRRegBankID);

    // TODO: Should report 32-bit for scalar condition type.
    if (Size == 64) {
      OpdsMapping[0] = RVGPU::getValueMappingSGPR64Only(Bank, Size);
      OpdsMapping[1] = RVGPU::getValueMapping(CondBank, 1);
      OpdsMapping[2] = RVGPU::getValueMappingSGPR64Only(Bank, Size);
      OpdsMapping[3] = RVGPU::getValueMappingSGPR64Only(Bank, Size);
    } else {
      OpdsMapping[0] = RVGPU::getValueMapping(Bank, Size);
      OpdsMapping[1] = RVGPU::getValueMapping(CondBank, 1);
      OpdsMapping[2] = RVGPU::getValueMapping(Bank, Size);
      OpdsMapping[3] = RVGPU::getValueMapping(Bank, Size);
    }

    break;
  }

  case RVGPU::G_SI_CALL: {
    OpdsMapping[0] = RVGPU::getValueMapping(RVGPU::SGPRRegBankID, 64);
    // Lie and claim everything is legal, even though some need to be
    // SGPRs. applyMapping will have to deal with it as a waterfall loop.
    OpdsMapping[1] = getSGPROpMapping(MI.getOperand(1).getReg(), MRI, *TRI);

    // Allow anything for implicit arguments
    for (unsigned I = 4; I < MI.getNumOperands(); ++I) {
      if (MI.getOperand(I).isReg()) {
        Register Reg = MI.getOperand(I).getReg();
        auto OpBank = getRegBankID(Reg, MRI);
        unsigned Size = getSizeInBits(Reg, MRI, *TRI);
        OpdsMapping[I] = RVGPU::getValueMapping(OpBank, Size);
      }
    }
    break;
  }
  case RVGPU::G_LOAD:
  case RVGPU::G_ZEXTLOAD:
  case RVGPU::G_SEXTLOAD:
    return getInstrMappingForLoad(MI);

  case RVGPU::G_ATOMICRMW_XCHG:
  case RVGPU::G_ATOMICRMW_ADD:
  case RVGPU::G_ATOMICRMW_SUB:
  case RVGPU::G_ATOMICRMW_AND:
  case RVGPU::G_ATOMICRMW_OR:
  case RVGPU::G_ATOMICRMW_XOR:
  case RVGPU::G_ATOMICRMW_MAX:
  case RVGPU::G_ATOMICRMW_MIN:
  case RVGPU::G_ATOMICRMW_UMAX:
  case RVGPU::G_ATOMICRMW_UMIN:
  case RVGPU::G_ATOMICRMW_FADD:
  case RVGPU::G_ATOMICRMW_UINC_WRAP:
  case RVGPU::G_ATOMICRMW_UDEC_WRAP:
  case RVGPU::G_RVGPU_ATOMIC_CMPXCHG:
  case RVGPU::G_RVGPU_ATOMIC_FMIN:
  case RVGPU::G_RVGPU_ATOMIC_FMAX: {
    OpdsMapping[0] = getVGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI);
    OpdsMapping[1] = getValueMappingForPtr(MRI, MI.getOperand(1).getReg());
    OpdsMapping[2] = getVGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
    break;
  }
  case RVGPU::G_ATOMIC_CMPXCHG: {
    OpdsMapping[0] = getVGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI);
    OpdsMapping[1] = getValueMappingForPtr(MRI, MI.getOperand(1).getReg());
    OpdsMapping[2] = getVGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
    OpdsMapping[3] = getVGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);
    break;
  }
  case RVGPU::G_BRCOND: {
    unsigned Bank = getRegBankID(MI.getOperand(0).getReg(), MRI,
                                 RVGPU::SGPRRegBankID);
    assert(MRI.getType(MI.getOperand(0).getReg()).getSizeInBits() == 1);
    if (Bank != RVGPU::SGPRRegBankID)
      Bank = RVGPU::VCCRegBankID;

    OpdsMapping[0] = RVGPU::getValueMapping(Bank, 1);
    break;
  }
  case RVGPU::G_FPTRUNC_ROUND_UPWARD:
  case RVGPU::G_FPTRUNC_ROUND_DOWNWARD:
    return getDefaultMappingVOP(MI);
  case RVGPU::G_PREFETCH:
    OpdsMapping[0] = getSGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI);
    break;
  }

  return getInstructionMapping(/*ID*/1, /*Cost*/1,
                               getOperandsMapping(OpdsMapping),
                               MI.getNumOperands());
}
