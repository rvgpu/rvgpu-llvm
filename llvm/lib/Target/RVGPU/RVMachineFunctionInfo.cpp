//===- RVMachineFunctionInfo.cpp - SI Machine Function Info ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RVMachineFunctionInfo.h"
#include "RVGPUSubtarget.h"
#include "RVGPUTargetMachine.h"
#include "RVSubtarget.h"
#include "MCTargetDesc/RVGPUMCTargetDesc.h"
#include "RVRegisterInfo.h"
#include "Utils/RVGPUBaseInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MIRParser/MIParser.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include <cassert>
#include <optional>
#include <vector>

#define MAX_LANES 64

using namespace llvm;

const RVTargetMachine &getTM(const RVSubtarget *STI) {
  const RVTargetLowering *TLI = STI->getTargetLowering();
  return static_cast<const RVTargetMachine &>(TLI->getTargetMachine());
}

RVMachineFunctionInfo::RVMachineFunctionInfo(const Function &F,
                                             const RVSubtarget *STI)
    : RVGPUMachineFunction(F, *STI), Mode(F, *STI), GWSResourcePSV(getTM(STI)),
      UserSGPRInfo(F, *STI), WorkGroupIDX(false), WorkGroupIDY(false),
      WorkGroupIDZ(false), WorkGroupInfo(false), LDSKernelId(false),
      PrivateSegmentWaveByteOffset(false), WorkItemIDX(false),
      WorkItemIDY(false), WorkItemIDZ(false), ImplicitArgPtr(false),
      GITPtrHigh(0xffffffff), HighBitsOf32BitAddress(0) {
  const RVSubtarget &ST = *static_cast<const RVSubtarget *>(STI);
  FlatWorkGroupSizes = ST.getFlatWorkGroupSizes(F);
  WavesPerEU = ST.getWavesPerEU(F);

  Occupancy = ST.computeOccupancy(F, getLDSSize());
  CallingConv::ID CC = F.getCallingConv();

  VRegFlags.reserve(1024);

  const bool IsKernel = CC == CallingConv::RVGPU_KERNEL ||
                        CC == CallingConv::SPIR_KERNEL;

  if (IsKernel) {
    WorkGroupIDX = true;
    WorkItemIDX = true;
  } else if (CC == CallingConv::RVGPU_PS) {
    PSInputAddr = RVGPU::getInitialPSInputAddr(F);
  }

  MayNeedAGPRs = ST.hasMAIInsts();

  if (RVGPU::isChainCC(CC)) {
    // Chain functions don't receive an SP from their caller, but are free to
    // set one up. For now, we can use s32 to match what rvgpu_gfx functions
    // would use if called, but this can be revisited.
    // FIXME: Only reserve this if we actually need it.
    StackPtrOffsetReg = RVGPU::SGPR32;

    ScratchRSrcReg = RVGPU::SGPR48_SGPR49_SGPR50_SGPR51;

    ArgInfo.PrivateSegmentBuffer =
        RvArgDescriptor::createRegister(ScratchRSrcReg);

    ImplicitArgPtr = false;
  } else if (!isEntryFunction()) {
    if (CC != CallingConv::RVGPU_Gfx)
      ArgInfo = RVGPUArgumentUsageInfo::FixedABIFunctionInfo;

    // TODO: Pick a high register, and shift down, similar to a kernel.
    FrameOffsetReg = RVGPU::SGPR33;
    StackPtrOffsetReg = RVGPU::SGPR32;

    if (!ST.enableFlatScratch()) {
      // Non-entry functions have no special inputs for now, other registers
      // required for scratch access.
      ScratchRSrcReg = RVGPU::SGPR0_SGPR1_SGPR2_SGPR3;

      ArgInfo.PrivateSegmentBuffer =
        RvArgDescriptor::createRegister(ScratchRSrcReg);
    }

    if (!F.hasFnAttribute("rvgpu-no-implicitarg-ptr"))
      ImplicitArgPtr = true;
  } else {
    ImplicitArgPtr = false;
    MaxKernArgAlign = std::max(ST.getAlignmentForImplicitArgPtr(),
                               MaxKernArgAlign);

    if (ST.hasGFX90AInsts() &&
        ST.getMaxNumVGPRs(F) <= RVGPU::VGPR_32RegClass.getNumRegs() &&
        !mayUseAGPRs(F))
      MayNeedAGPRs = false; // We will select all MAI with VGPR operands.
  }

  if (!RVGPU::isGraphics(CC) ||
      (CC == CallingConv::RVGPU_CS && ST.hasArchitectedSGPRs())) {
    if (IsKernel || !F.hasFnAttribute("rvgpu-no-workgroup-id-x"))
      WorkGroupIDX = true;

    if (!F.hasFnAttribute("rvgpu-no-workgroup-id-y"))
      WorkGroupIDY = true;

    if (!F.hasFnAttribute("rvgpu-no-workgroup-id-z"))
      WorkGroupIDZ = true;
  }

  if (!RVGPU::isGraphics(CC)) {
    if (IsKernel || !F.hasFnAttribute("rvgpu-no-workitem-id-x"))
      WorkItemIDX = true;

    if (!F.hasFnAttribute("rvgpu-no-workitem-id-y") &&
        ST.getMaxWorkitemID(F, 1) != 0)
      WorkItemIDY = true;

    if (!F.hasFnAttribute("rvgpu-no-workitem-id-z") &&
        ST.getMaxWorkitemID(F, 2) != 0)
      WorkItemIDZ = true;

    if (!IsKernel && !F.hasFnAttribute("rvgpu-no-lds-kernel-id"))
      LDSKernelId = true;
  }

  if (isEntryFunction()) {
    // X, XY, and XYZ are the only supported combinations, so make sure Y is
    // enabled if Z is.
    if (WorkItemIDZ)
      WorkItemIDY = true;

    if (!ST.flatScratchIsArchitected()) {
      PrivateSegmentWaveByteOffset = true;

      // HS and GS always have the scratch wave offset in SGPR5 on GFX9.
      if (ST.getGeneration() >= RVGPUSubtarget::GFX9 &&
          (CC == CallingConv::RVGPU_HS || CC == CallingConv::RVGPU_GS))
        ArgInfo.PrivateSegmentWaveByteOffset =
            RvArgDescriptor::createRegister(RVGPU::SGPR5);
    }
  }

  Attribute A = F.getFnAttribute("rvgpu-git-ptr-high");
  StringRef S = A.getValueAsString();
  if (!S.empty())
    S.consumeInteger(0, GITPtrHigh);

  A = F.getFnAttribute("rvgpu-32bit-address-high-bits");
  S = A.getValueAsString();
  if (!S.empty())
    S.consumeInteger(0, HighBitsOf32BitAddress);

  // On GFX908, in order to guarantee copying between AGPRs, we need a scratch
  // VGPR available at all times. For now, reserve highest available VGPR. After
  // RA, shift it to the lowest available unused VGPR if the one exist.
  if (ST.hasMAIInsts() && !ST.hasGFX90AInsts()) {
    VGPRForAGPRCopy =
        RVGPU::VGPR_32RegClass.getRegister(ST.getMaxNumVGPRs(F) - 1);
  }
}

MachineFunctionInfo *RVMachineFunctionInfo::clone(
    BumpPtrAllocator &Allocator, MachineFunction &DestMF,
    const DenseMap<MachineBasicBlock *, MachineBasicBlock *> &Src2DstMBB)
    const {
  return DestMF.cloneInfo<RVMachineFunctionInfo>(*this);
}

void RVMachineFunctionInfo::limitOccupancy(const MachineFunction &MF) {
  limitOccupancy(getMaxWavesPerEU());
  const RVSubtarget& ST = MF.getSubtarget<RVSubtarget>();
  limitOccupancy(ST.getOccupancyWithLocalMemSize(getLDSSize(),
                 MF.getFunction()));
}

Register RVMachineFunctionInfo::addPrivateSegmentBuffer(
  const RVRegisterInfo &TRI) {
  ArgInfo.PrivateSegmentBuffer =
    RvArgDescriptor::createRegister(TRI.getMatchingSuperReg(
    getNextUserSGPR(), RVGPU::sub0, &RVGPU::SGPR_128RegClass));
  NumUserSGPRs += 4;
  return ArgInfo.PrivateSegmentBuffer.getRegister();
}

Register RVMachineFunctionInfo::addDispatchPtr(const RVRegisterInfo &TRI) {
  ArgInfo.DispatchPtr = RvArgDescriptor::createRegister(TRI.getMatchingSuperReg(
    getNextUserSGPR(), RVGPU::sub0, &RVGPU::SReg_64RegClass));
  NumUserSGPRs += 2;
  return ArgInfo.DispatchPtr.getRegister();
}

Register RVMachineFunctionInfo::addQueuePtr(const RVRegisterInfo &TRI) {
  ArgInfo.QueuePtr = RvArgDescriptor::createRegister(TRI.getMatchingSuperReg(
    getNextUserSGPR(), RVGPU::sub0, &RVGPU::SReg_64RegClass));
  NumUserSGPRs += 2;
  return ArgInfo.QueuePtr.getRegister();
}

Register RVMachineFunctionInfo::addKernargSegmentPtr(const RVRegisterInfo &TRI) {
  ArgInfo.KernargSegmentPtr
    = RvArgDescriptor::createRegister(TRI.getMatchingSuperReg(
    getNextUserSGPR(), RVGPU::sub0, &RVGPU::SReg_64RegClass));
  NumUserSGPRs += 2;
  return ArgInfo.KernargSegmentPtr.getRegister();
}

Register RVMachineFunctionInfo::addDispatchID(const RVRegisterInfo &TRI) {
  ArgInfo.DispatchID = RvArgDescriptor::createRegister(TRI.getMatchingSuperReg(
    getNextUserSGPR(), RVGPU::sub0, &RVGPU::SReg_64RegClass));
  NumUserSGPRs += 2;
  return ArgInfo.DispatchID.getRegister();
}

Register RVMachineFunctionInfo::addFlatScratchInit(const RVRegisterInfo &TRI) {
  ArgInfo.FlatScratchInit = RvArgDescriptor::createRegister(TRI.getMatchingSuperReg(
    getNextUserSGPR(), RVGPU::sub0, &RVGPU::SReg_64RegClass));
  NumUserSGPRs += 2;
  return ArgInfo.FlatScratchInit.getRegister();
}

Register RVMachineFunctionInfo::addImplicitBufferPtr(const RVRegisterInfo &TRI) {
  ArgInfo.ImplicitBufferPtr = RvArgDescriptor::createRegister(TRI.getMatchingSuperReg(
    getNextUserSGPR(), RVGPU::sub0, &RVGPU::SReg_64RegClass));
  NumUserSGPRs += 2;
  return ArgInfo.ImplicitBufferPtr.getRegister();
}

Register RVMachineFunctionInfo::addLDSKernelId() {
  ArgInfo.LDSKernelId = RvArgDescriptor::createRegister(getNextUserSGPR());
  NumUserSGPRs += 1;
  return ArgInfo.LDSKernelId.getRegister();
}

SmallVectorImpl<MCRegister> *RVMachineFunctionInfo::addPreloadedKernArg(
    const RVRegisterInfo &TRI, const TargetRegisterClass *RC,
    unsigned AllocSizeDWord, int KernArgIdx, int PaddingSGPRs) {
  assert(!ArgInfo.PreloadKernArgs.count(KernArgIdx) &&
         "Preload kernel argument allocated twice.");
  NumUserSGPRs += PaddingSGPRs;
  // If the available register tuples are aligned with the kernarg to be
  // preloaded use that register, otherwise we need to use a set of SGPRs and
  // merge them.
  Register PreloadReg =
      TRI.getMatchingSuperReg(getNextUserSGPR(), RVGPU::sub0, RC);
  if (PreloadReg &&
      (RC == &RVGPU::SReg_32RegClass || RC == &RVGPU::SReg_64RegClass)) {
    ArgInfo.PreloadKernArgs[KernArgIdx].Regs.push_back(PreloadReg);
    NumUserSGPRs += AllocSizeDWord;
  } else {
    for (unsigned I = 0; I < AllocSizeDWord; ++I) {
      ArgInfo.PreloadKernArgs[KernArgIdx].Regs.push_back(getNextUserSGPR());
      NumUserSGPRs++;
    }
  }

  // Track the actual number of SGPRs that HW will preload to.
  UserSGPRInfo.allocKernargPreloadSGPRs(AllocSizeDWord + PaddingSGPRs);
  return &ArgInfo.PreloadKernArgs[KernArgIdx].Regs;
}

void RVMachineFunctionInfo::allocateWWMSpill(MachineFunction &MF, Register VGPR,
                                             uint64_t Size, Align Alignment) {
  // Skip if it is an entry function or the register is already added.
  if (isEntryFunction() || WWMSpills.count(VGPR))
    return;

  // Skip if this is a function with the rvgpu_cs_chain or
  // rvgpu_cs_chain_preserve calling convention and this is a scratch register.
  // We never need to allocate a spill for these because we don't even need to
  // restore the inactive lanes for them (they're scratchier than the usual
  // scratch registers).
  if (isChainFunction() && RVRegisterInfo::isChainScratchRegister(VGPR))
    return;

  WWMSpills.insert(std::make_pair(
      VGPR, MF.getFrameInfo().CreateSpillStackObject(Size, Alignment)));
}

// Separate out the callee-saved and scratch registers.
void RVMachineFunctionInfo::splitWWMSpillRegisters(
    MachineFunction &MF,
    SmallVectorImpl<std::pair<Register, int>> &CalleeSavedRegs,
    SmallVectorImpl<std::pair<Register, int>> &ScratchRegs) const {
  const MCPhysReg *CSRegs = MF.getRegInfo().getCalleeSavedRegs();
  for (auto &Reg : WWMSpills) {
    if (isCalleeSavedReg(CSRegs, Reg.first))
      CalleeSavedRegs.push_back(Reg);
    else
      ScratchRegs.push_back(Reg);
  }
}

bool RVMachineFunctionInfo::isCalleeSavedReg(const MCPhysReg *CSRegs,
                                             MCPhysReg Reg) const {
  for (unsigned I = 0; CSRegs[I]; ++I) {
    if (CSRegs[I] == Reg)
      return true;
  }

  return false;
}

bool RVMachineFunctionInfo::allocateVirtualVGPRForSGPRSpills(
    MachineFunction &MF, int FI, unsigned LaneIndex) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  Register LaneVGPR;
  if (!LaneIndex) {
    LaneVGPR = MRI.createVirtualRegister(&RVGPU::VGPR_32RegClass);
    SpillVGPRs.push_back(LaneVGPR);
  } else {
    LaneVGPR = SpillVGPRs.back();
  }

  SGPRSpillsToVirtualVGPRLanes[FI].push_back(
      RVRegisterInfo::SpilledReg(LaneVGPR, LaneIndex));
  return true;
}

bool RVMachineFunctionInfo::allocatePhysicalVGPRForSGPRSpills(
    MachineFunction &MF, int FI, unsigned LaneIndex) {
  const RVSubtarget &ST = MF.getSubtarget<RVSubtarget>();
  const RVRegisterInfo *TRI = ST.getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  Register LaneVGPR;
  if (!LaneIndex) {
    LaneVGPR = TRI->findUnusedRegister(MRI, &RVGPU::VGPR_32RegClass, MF);
    if (LaneVGPR == RVGPU::NoRegister) {
      // We have no VGPRs left for spilling SGPRs. Reset because we will not
      // partially spill the SGPR to VGPRs.
      SGPRSpillsToPhysicalVGPRLanes.erase(FI);
      return false;
    }

    allocateWWMSpill(MF, LaneVGPR);
    reserveWWMRegister(LaneVGPR);
    for (MachineBasicBlock &MBB : MF) {
      MBB.addLiveIn(LaneVGPR);
      MBB.sortUniqueLiveIns();
    }
    SpillPhysVGPRs.push_back(LaneVGPR);
  } else {
    LaneVGPR = SpillPhysVGPRs.back();
  }

  SGPRSpillsToPhysicalVGPRLanes[FI].push_back(
      RVRegisterInfo::SpilledReg(LaneVGPR, LaneIndex));
  return true;
}

bool RVMachineFunctionInfo::allocateSGPRSpillToVGPRLane(MachineFunction &MF,
                                                        int FI,
                                                        bool IsPrologEpilog) {
  std::vector<RVRegisterInfo::SpilledReg> &SpillLanes =
      IsPrologEpilog ? SGPRSpillsToPhysicalVGPRLanes[FI]
                     : SGPRSpillsToVirtualVGPRLanes[FI];

  // This has already been allocated.
  if (!SpillLanes.empty())
    return true;

  const RVSubtarget &ST = MF.getSubtarget<RVSubtarget>();
  MachineFrameInfo &FrameInfo = MF.getFrameInfo();
  unsigned WaveSize = ST.getWavefrontSize();

  unsigned Size = FrameInfo.getObjectSize(FI);
  unsigned NumLanes = Size / 4;

  if (NumLanes > WaveSize)
    return false;

  assert(Size >= 4 && "invalid sgpr spill size");
  assert(ST.getRegisterInfo()->spillSGPRToVGPR() &&
         "not spilling SGPRs to VGPRs");

  unsigned &NumSpillLanes =
      IsPrologEpilog ? NumPhysicalVGPRSpillLanes : NumVirtualVGPRSpillLanes;

  for (unsigned I = 0; I < NumLanes; ++I, ++NumSpillLanes) {
    unsigned LaneIndex = (NumSpillLanes % WaveSize);

    bool Allocated = IsPrologEpilog
                         ? allocatePhysicalVGPRForSGPRSpills(MF, FI, LaneIndex)
                         : allocateVirtualVGPRForSGPRSpills(MF, FI, LaneIndex);
    if (!Allocated) {
      NumSpillLanes -= I;
      return false;
    }
  }

  return true;
}

/// Reserve AGPRs or VGPRs to support spilling for FrameIndex \p FI.
/// Either AGPR is spilled to VGPR to vice versa.
/// Returns true if a \p FI can be eliminated completely.
bool RVMachineFunctionInfo::allocateVGPRSpillToAGPR(MachineFunction &MF,
                                                    int FI,
                                                    bool isAGPRtoVGPR) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  MachineFrameInfo &FrameInfo = MF.getFrameInfo();
  const RVSubtarget &ST =  MF.getSubtarget<RVSubtarget>();

  assert(ST.hasMAIInsts() && FrameInfo.isSpillSlotObjectIndex(FI));

  auto &Spill = VGPRToAGPRSpills[FI];

  // This has already been allocated.
  if (!Spill.Lanes.empty())
    return Spill.FullyAllocated;

  unsigned Size = FrameInfo.getObjectSize(FI);
  unsigned NumLanes = Size / 4;
  Spill.Lanes.resize(NumLanes, RVGPU::NoRegister);

  const TargetRegisterClass &RC =
      isAGPRtoVGPR ? RVGPU::VGPR_32RegClass : RVGPU::AGPR_32RegClass;
  auto Regs = RC.getRegisters();

  auto &SpillRegs = isAGPRtoVGPR ? SpillAGPR : SpillVGPR;
  const RVRegisterInfo *TRI = ST.getRegisterInfo();
  Spill.FullyAllocated = true;

  // FIXME: Move allocation logic out of MachineFunctionInfo and initialize
  // once.
  BitVector OtherUsedRegs;
  OtherUsedRegs.resize(TRI->getNumRegs());

  const uint32_t *CSRMask =
      TRI->getCallPreservedMask(MF, MF.getFunction().getCallingConv());
  if (CSRMask)
    OtherUsedRegs.setBitsInMask(CSRMask);

  // TODO: Should include register tuples, but doesn't matter with current
  // usage.
  for (MCPhysReg Reg : SpillAGPR)
    OtherUsedRegs.set(Reg);
  for (MCPhysReg Reg : SpillVGPR)
    OtherUsedRegs.set(Reg);

  SmallVectorImpl<MCPhysReg>::const_iterator NextSpillReg = Regs.begin();
  for (int I = NumLanes - 1; I >= 0; --I) {
    NextSpillReg = std::find_if(
        NextSpillReg, Regs.end(), [&MRI, &OtherUsedRegs](MCPhysReg Reg) {
          return MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg) &&
                 !OtherUsedRegs[Reg];
        });

    if (NextSpillReg == Regs.end()) { // Registers exhausted
      Spill.FullyAllocated = false;
      break;
    }

    OtherUsedRegs.set(*NextSpillReg);
    SpillRegs.push_back(*NextSpillReg);
    MRI.reserveReg(*NextSpillReg, TRI);
    Spill.Lanes[I] = *NextSpillReg++;
  }

  return Spill.FullyAllocated;
}

bool RVMachineFunctionInfo::removeDeadFrameIndices(
    MachineFrameInfo &MFI, bool ResetSGPRSpillStackIDs) {
  // Remove dead frame indices from function frame, however keep FP & BP since
  // spills for them haven't been inserted yet. And also make sure to remove the
  // frame indices from `SGPRSpillsToVirtualVGPRLanes` data structure,
  // otherwise, it could result in an unexpected side effect and bug, in case of
  // any re-mapping of freed frame indices by later pass(es) like "stack slot
  // coloring".
  for (auto &R : make_early_inc_range(SGPRSpillsToVirtualVGPRLanes)) {
    MFI.RemoveStackObject(R.first);
    SGPRSpillsToVirtualVGPRLanes.erase(R.first);
  }

  // Remove the dead frame indices of CSR SGPRs which are spilled to physical
  // VGPR lanes during SILowerSGPRSpills pass.
  if (!ResetSGPRSpillStackIDs) {
    for (auto &R : make_early_inc_range(SGPRSpillsToPhysicalVGPRLanes)) {
      MFI.RemoveStackObject(R.first);
      SGPRSpillsToPhysicalVGPRLanes.erase(R.first);
    }
  }
  bool HaveSGPRToMemory = false;

  if (ResetSGPRSpillStackIDs) {
    // All other SGPRs must be allocated on the default stack, so reset the
    // stack ID.
    for (int I = MFI.getObjectIndexBegin(), E = MFI.getObjectIndexEnd(); I != E;
         ++I) {
      if (!checkIndexInPrologEpilogSGPRSpills(I)) {
        if (MFI.getStackID(I) == TargetStackID::SGPRSpill) {
          MFI.setStackID(I, TargetStackID::Default);
          HaveSGPRToMemory = true;
        }
      }
    }
  }

  for (auto &R : VGPRToAGPRSpills) {
    if (R.second.IsDead)
      MFI.RemoveStackObject(R.first);
  }

  return HaveSGPRToMemory;
}

int RVMachineFunctionInfo::getScavengeFI(MachineFrameInfo &MFI,
                                         const RVRegisterInfo &TRI) {
  if (ScavengeFI)
    return *ScavengeFI;
  if (isBottomOfStack()) {
    ScavengeFI = MFI.CreateFixedObject(
        TRI.getSpillSize(RVGPU::SGPR_32RegClass), 0, false);
  } else {
    ScavengeFI = MFI.CreateStackObject(
        TRI.getSpillSize(RVGPU::SGPR_32RegClass),
        TRI.getSpillAlign(RVGPU::SGPR_32RegClass), false);
  }
  return *ScavengeFI;
}

MCPhysReg RVMachineFunctionInfo::getNextUserSGPR() const {
  assert(NumSystemSGPRs == 0 && "System SGPRs must be added after user SGPRs");
  return RVGPU::SGPR0 + NumUserSGPRs;
}

MCPhysReg RVMachineFunctionInfo::getNextSystemSGPR() const {
  return RVGPU::SGPR0 + NumUserSGPRs + NumSystemSGPRs;
}

void RVMachineFunctionInfo::MRI_NoteNewVirtualRegister(Register Reg) {
  VRegFlags.grow(Reg);
}

void RVMachineFunctionInfo::MRI_NoteCloneVirtualRegister(Register NewReg,
                                                         Register SrcReg) {
  VRegFlags.grow(NewReg);
  VRegFlags[NewReg] = VRegFlags[SrcReg];
}

Register
RVMachineFunctionInfo::getGITPtrLoReg(const MachineFunction &MF) const {
  const RVSubtarget &ST = MF.getSubtarget<RVSubtarget>();
  if (!ST.isRvPalOS())
    return Register();
  Register GitPtrLo = RVGPU::SGPR0; // Low GIT address passed in
  if (ST.hasMergedShaders()) {
    switch (MF.getFunction().getCallingConv()) {
    case CallingConv::RVGPU_HS:
    case CallingConv::RVGPU_GS:
      // Low GIT address is passed in s8 rather than s0 for an LS+HS or
      // ES+GS merged shader on gfx9+.
      GitPtrLo = RVGPU::SGPR8;
      return GitPtrLo;
    default:
      return GitPtrLo;
    }
  }
  return GitPtrLo;
}

static yaml::StringValue regToString(Register Reg,
                                     const TargetRegisterInfo &TRI) {
  yaml::StringValue Dest;
  {
    raw_string_ostream OS(Dest.Value);
    OS << printReg(Reg, &TRI);
  }
  return Dest;
}

static std::optional<yaml::SIArgumentInfo>
convertArgumentInfo(const RVGPUFunctionArgInfo &ArgInfo,
                    const TargetRegisterInfo &TRI) {
  yaml::SIArgumentInfo AI;

  auto convertArg = [&](std::optional<yaml::SIArgument> &A,
                        const RvArgDescriptor &Arg) {
    if (!Arg)
      return false;

    // Create a register or stack argument.
    yaml::SIArgument SA = yaml::SIArgument::createArgument(Arg.isRegister());
    if (Arg.isRegister()) {
      raw_string_ostream OS(SA.RegisterName.Value);
      OS << printReg(Arg.getRegister(), &TRI);
    } else
      SA.StackOffset = Arg.getStackOffset();
    // Check and update the optional mask.
    if (Arg.isMasked())
      SA.Mask = Arg.getMask();

    A = SA;
    return true;
  };

  // TODO: Need to serialize kernarg preloads.
  bool Any = false;
  Any |= convertArg(AI.PrivateSegmentBuffer, ArgInfo.PrivateSegmentBuffer);
  Any |= convertArg(AI.DispatchPtr, ArgInfo.DispatchPtr);
  Any |= convertArg(AI.QueuePtr, ArgInfo.QueuePtr);
  Any |= convertArg(AI.KernargSegmentPtr, ArgInfo.KernargSegmentPtr);
  Any |= convertArg(AI.DispatchID, ArgInfo.DispatchID);
  Any |= convertArg(AI.FlatScratchInit, ArgInfo.FlatScratchInit);
  Any |= convertArg(AI.LDSKernelId, ArgInfo.LDSKernelId);
  Any |= convertArg(AI.PrivateSegmentSize, ArgInfo.PrivateSegmentSize);
  Any |= convertArg(AI.WorkGroupIDX, ArgInfo.WorkGroupIDX);
  Any |= convertArg(AI.WorkGroupIDY, ArgInfo.WorkGroupIDY);
  Any |= convertArg(AI.WorkGroupIDZ, ArgInfo.WorkGroupIDZ);
  Any |= convertArg(AI.WorkGroupInfo, ArgInfo.WorkGroupInfo);
  Any |= convertArg(AI.PrivateSegmentWaveByteOffset,
                    ArgInfo.PrivateSegmentWaveByteOffset);
  Any |= convertArg(AI.ImplicitArgPtr, ArgInfo.ImplicitArgPtr);
  Any |= convertArg(AI.ImplicitBufferPtr, ArgInfo.ImplicitBufferPtr);
  Any |= convertArg(AI.WorkItemIDX, ArgInfo.WorkItemIDX);
  Any |= convertArg(AI.WorkItemIDY, ArgInfo.WorkItemIDY);
  Any |= convertArg(AI.WorkItemIDZ, ArgInfo.WorkItemIDZ);

  if (Any)
    return AI;

  return std::nullopt;
}

yaml::RVMachineFunctionInfo::RVMachineFunctionInfo(
    const llvm::RVMachineFunctionInfo &MFI, const TargetRegisterInfo &TRI,
    const llvm::MachineFunction &MF)
    : ExplicitKernArgSize(MFI.getExplicitKernArgSize()),
      MaxKernArgAlign(MFI.getMaxKernArgAlign()), LDSSize(MFI.getLDSSize()),
      GDSSize(MFI.getGDSSize()),
      DynLDSAlign(MFI.getDynLDSAlign()), IsEntryFunction(MFI.isEntryFunction()),
      NoSignedZerosFPMath(MFI.hasNoSignedZerosFPMath()),
      MemoryBound(MFI.isMemoryBound()), WaveLimiter(MFI.needsWaveLimiter()),
      HasSpilledSGPRs(MFI.hasSpilledSGPRs()),
      HasSpilledVGPRs(MFI.hasSpilledVGPRs()),
      HighBitsOf32BitAddress(MFI.get32BitAddressHighBits()),
      Occupancy(MFI.getOccupancy()),
      ScratchRSrcReg(regToString(MFI.getScratchRSrcReg(), TRI)),
      FrameOffsetReg(regToString(MFI.getFrameOffsetReg(), TRI)),
      StackPtrOffsetReg(regToString(MFI.getStackPtrOffsetReg(), TRI)),
      BytesInStackArgArea(MFI.getBytesInStackArgArea()),
      ReturnsVoid(MFI.returnsVoid()),
      ArgInfo(convertArgumentInfo(MFI.getArgInfo(), TRI)),
      PSInputAddr(MFI.getPSInputAddr()),
      PSInputEnable(MFI.getPSInputEnable()),
      Mode(MFI.getMode()) {
  for (Register Reg : MFI.getWWMReservedRegs())
    WWMReservedRegs.push_back(regToString(Reg, TRI));

  if (MFI.getLongBranchReservedReg())
    LongBranchReservedReg = regToString(MFI.getLongBranchReservedReg(), TRI);
  if (MFI.getVGPRForAGPRCopy())
    VGPRForAGPRCopy = regToString(MFI.getVGPRForAGPRCopy(), TRI);

  if (MFI.getSGPRForEXECCopy())
    SGPRForEXECCopy = regToString(MFI.getSGPRForEXECCopy(), TRI);

  auto SFI = MFI.getOptionalScavengeFI();
  if (SFI)
    ScavengeFI = yaml::FrameIndex(*SFI, MF.getFrameInfo());
}

void yaml::RVMachineFunctionInfo::mappingImpl(yaml::IO &YamlIO) {
  MappingTraits<RVMachineFunctionInfo>::mapping(YamlIO, *this);
}

bool RVMachineFunctionInfo::initializeBaseYamlFields(
    const yaml::RVMachineFunctionInfo &YamlMFI, const MachineFunction &MF,
    PerFunctionMIParsingState &PFS, SMDiagnostic &Error, SMRange &SourceRange) {
  ExplicitKernArgSize = YamlMFI.ExplicitKernArgSize;
  MaxKernArgAlign = YamlMFI.MaxKernArgAlign;
  LDSSize = YamlMFI.LDSSize;
  GDSSize = YamlMFI.GDSSize;
  DynLDSAlign = YamlMFI.DynLDSAlign;
  PSInputAddr = YamlMFI.PSInputAddr;
  PSInputEnable = YamlMFI.PSInputEnable;
  HighBitsOf32BitAddress = YamlMFI.HighBitsOf32BitAddress;
  Occupancy = YamlMFI.Occupancy;
  IsEntryFunction = YamlMFI.IsEntryFunction;
  NoSignedZerosFPMath = YamlMFI.NoSignedZerosFPMath;
  MemoryBound = YamlMFI.MemoryBound;
  WaveLimiter = YamlMFI.WaveLimiter;
  HasSpilledSGPRs = YamlMFI.HasSpilledSGPRs;
  HasSpilledVGPRs = YamlMFI.HasSpilledVGPRs;
  BytesInStackArgArea = YamlMFI.BytesInStackArgArea;
  ReturnsVoid = YamlMFI.ReturnsVoid;

  if (YamlMFI.ScavengeFI) {
    auto FIOrErr = YamlMFI.ScavengeFI->getFI(MF.getFrameInfo());
    if (!FIOrErr) {
      // Create a diagnostic for a the frame index.
      const MemoryBuffer &Buffer =
          *PFS.SM->getMemoryBuffer(PFS.SM->getMainFileID());

      Error = SMDiagnostic(*PFS.SM, SMLoc(), Buffer.getBufferIdentifier(), 1, 1,
                           SourceMgr::DK_Error, toString(FIOrErr.takeError()),
                           "", std::nullopt, std::nullopt);
      SourceRange = YamlMFI.ScavengeFI->SourceRange;
      return true;
    }
    ScavengeFI = *FIOrErr;
  } else {
    ScavengeFI = std::nullopt;
  }
  return false;
}

bool RVMachineFunctionInfo::mayUseAGPRs(const Function &F) const {
  for (const BasicBlock &BB : F) {
    for (const Instruction &I : BB) {
      const auto *CB = dyn_cast<CallBase>(&I);
      if (!CB)
        continue;

      if (CB->isInlineAsm()) {
        const InlineAsm *IA = dyn_cast<InlineAsm>(CB->getCalledOperand());
        for (const auto &CI : IA->ParseConstraints()) {
          for (StringRef Code : CI.Codes) {
            Code.consume_front("{");
            if (Code.starts_with("a"))
              return true;
          }
        }
        continue;
      }

      const Function *Callee =
          dyn_cast<Function>(CB->getCalledOperand()->stripPointerCasts());
      if (!Callee)
        return true;

      if (Callee->getIntrinsicID() == Intrinsic::not_intrinsic)
        return true;
    }
  }

  return false;
}

bool RVMachineFunctionInfo::usesAGPRs(const MachineFunction &MF) const {
  if (UsesAGPRs)
    return *UsesAGPRs;

  if (!mayNeedAGPRs()) {
    UsesAGPRs = false;
    return false;
  }

  if (!RVGPU::isEntryFunctionCC(MF.getFunction().getCallingConv()) ||
      MF.getFrameInfo().hasCalls()) {
    UsesAGPRs = true;
    return true;
  }

  const MachineRegisterInfo &MRI = MF.getRegInfo();

  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    const Register Reg = Register::index2VirtReg(I);
    const TargetRegisterClass *RC = MRI.getRegClassOrNull(Reg);
    if (RC && RVRegisterInfo::isAGPRClass(RC)) {
      UsesAGPRs = true;
      return true;
    } else if (!RC && !MRI.use_empty(Reg) && MRI.getType(Reg).isValid()) {
      // Defer caching UsesAGPRs, function might not yet been regbank selected.
      return true;
    }
  }

  for (MCRegister Reg : RVGPU::AGPR_32RegClass) {
    if (MRI.isPhysRegUsed(Reg)) {
      UsesAGPRs = true;
      return true;
    }
  }

  UsesAGPRs = false;
  return false;
}
