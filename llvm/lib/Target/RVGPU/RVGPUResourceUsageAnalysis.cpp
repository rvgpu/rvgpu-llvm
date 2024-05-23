//===- RVGPUResourceUsageAnalysis.h ---- analysis of resources -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Analyzes how many registers and other resources are used by
/// functions.
///
/// The results of this analysis are used to fill the register usage, flat
/// usage, etc. into hardware registers.
///
/// The analysis takes callees into account. E.g. if a function A that needs 10
/// VGPRs calls a function B that needs 20 VGPRs, querying the VGPR usage of A
/// will return 20.
/// It is assumed that an indirect call can go into any function except
/// hardware-entrypoints. Therefore the register usage of functions with
/// indirect calls is estimated as the maximum of all non-entrypoint functions
/// in the module.
///
//===----------------------------------------------------------------------===//

#include "RVGPUResourceUsageAnalysis.h"
#include "RVGPU.h"
#include "RVGPUSubtarget.h"
#include "RVGPUMachineFunctionInfo.h"
#include "MCTargetDesc/RVGPUMCTargetDesc.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;
using namespace llvm::RVGPU;

#define DEBUG_TYPE "rvgpu-resource-usage"

char llvm::RVGPUResourceUsageAnalysis::ID = 0;
char &llvm::RVGPUResourceUsageAnalysisID = RVGPUResourceUsageAnalysis::ID;

// In code object v4 and older, we need to tell the runtime some amount ahead of
// time if we don't know the true stack size. Assume a smaller number if this is
// only due to dynamic / non-entry block allocas.
static cl::opt<uint32_t> AssumedStackSizeForExternalCall(
    "rvgpu-assume-external-call-stack-size",
    cl::desc("Assumed stack use of any external call (in bytes)"), cl::Hidden,
    cl::init(16384));

static cl::opt<uint32_t> AssumedStackSizeForDynamicSizeObjects(
    "rvgpu-assume-dynamic-stack-object-size",
    cl::desc("Assumed extra stack use if there are any "
             "variable sized objects (in bytes)"),
    cl::Hidden, cl::init(4096));

INITIALIZE_PASS(RVGPUResourceUsageAnalysis, DEBUG_TYPE,
                "Function register usage analysis", true, true)

static const Function *getCalleeFunction(const MachineOperand &Op) {
  if (Op.isImm()) {
    assert(Op.getImm() == 0);
    return nullptr;
  }
  if (auto *GA = dyn_cast<GlobalAlias>(Op.getGlobal()))
    return cast<Function>(GA->getOperand(0));
  return cast<Function>(Op.getGlobal());
}


int32_t RVGPUResourceUsageAnalysis::RVFunctionResourceInfo::getTotalNumVGPRs(
    const RVGPUSubtarget &ST, int32_t ArgNumAGPR, int32_t ArgNumVGPR) const {
  return std::max(ArgNumVGPR, ArgNumAGPR);
}

int32_t RVGPUResourceUsageAnalysis::RVFunctionResourceInfo::getTotalNumVGPRs(
    const RVGPUSubtarget &ST) const {
  return getTotalNumVGPRs(ST, NumAGPR, NumVGPR);
}

bool RVGPUResourceUsageAnalysis::runOnModule(Module &M) {
  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC)
    return false;

  MachineModuleInfo &MMI = getAnalysis<MachineModuleInfoWrapperPass>().getMMI();
  const TargetMachine &TM = TPC->getTM<TargetMachine>();
  const MCSubtargetInfo &STI = *TM.getMCSubtargetInfo();
  bool HasIndirectCall = false;

  CallGraph CG = CallGraph(M);
  auto End = po_end(&CG);

  // By default, for code object v5 and later, track only the minimum scratch
  // size
  if (!AssumedStackSizeForDynamicSizeObjects.getNumOccurrences())
    AssumedStackSizeForDynamicSizeObjects = 0;
  if (!AssumedStackSizeForExternalCall.getNumOccurrences())
    AssumedStackSizeForExternalCall = 0;

  for (auto IT = po_begin(&CG); IT != End; ++IT) {
    Function *F = IT->getFunction();
    if (!F || F->isDeclaration())
      continue;

    MachineFunction *MF = MMI.getMachineFunction(*F);
    assert(MF && "function must have been generated already");

    auto CI =
        CallGraphResourceInfo.insert(std::pair(F, RVFunctionResourceInfo()));
    RVFunctionResourceInfo &Info = CI.first->second;
    assert(CI.second && "should only be called once per function");
    Info = analyzeResourceUsage(*MF, TM);
    HasIndirectCall |= Info.HasIndirectCall;
  }

  // It's possible we have unreachable functions in the module which weren't
  // visited by the PO traversal. Make sure we have some resource counts to
  // report.
  for (const auto &IT : CG) {
    const Function *F = IT.first;
    if (!F || F->isDeclaration())
      continue;

    auto CI =
        CallGraphResourceInfo.insert(std::pair(F, RVFunctionResourceInfo()));
    if (!CI.second) // Skip already visited functions
      continue;

    RVFunctionResourceInfo &Info = CI.first->second;
    MachineFunction *MF = MMI.getMachineFunction(*F);
    assert(MF && "function must have been generated already");
    Info = analyzeResourceUsage(*MF, TM);
    HasIndirectCall |= Info.HasIndirectCall;
  }

  if (HasIndirectCall)
    propagateIndirectCallRegisterUsage();

  return false;
}

RVGPUResourceUsageAnalysis::RVFunctionResourceInfo
RVGPUResourceUsageAnalysis::analyzeResourceUsage(
    const MachineFunction &MF, const TargetMachine &TM) const {
  RVFunctionResourceInfo Info;

  const RVGPUMachineFunctionInfo *MFI = MF.getInfo<RVGPUMachineFunctionInfo>();
  const RVGPUSubtarget &ST = MF.getSubtarget<RVGPUSubtarget>();
  const MachineFrameInfo &FrameInfo = MF.getFrameInfo();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const RVGPUInstrInfo *TII = ST.getInstrInfo();
  const RVGPURegisterInfo &TRI = TII->getRegisterInfo();

  Info.PrivateSegmentSize = FrameInfo.getStackSize();

  // Assume a big number if there are any unknown sized objects.
  Info.HasDynamicallySizedStack = FrameInfo.hasVarSizedObjects();
  if (Info.HasDynamicallySizedStack)
    Info.PrivateSegmentSize += AssumedStackSizeForDynamicSizeObjects;

//  if (MFI->isStackRealigned())
    Info.PrivateSegmentSize += FrameInfo.getMaxAlign().value();

  Info.UsesVCC =
      MRI.isPhysRegUsed(RVGPU::VCC_LO) || MRI.isPhysRegUsed(RVGPU::VCC_HI);

  // If there are no calls, MachineRegisterInfo can tell us the used register
  // count easily.
  // A tail call isn't considered a call for MachineFrameInfo's purposes.
  if (!FrameInfo.hasCalls() && !FrameInfo.hasTailCall()) {
    MCPhysReg HighestVGPRReg = RVGPU::NoRegister;
    for (MCPhysReg Reg : reverse(RVGPU::GPR32RegClass.getRegisters())) {
      if (MRI.isPhysRegUsed(Reg)) {
        HighestVGPRReg = Reg;
        break;
      }
    }

    // We found the maximum register index. They start at 0, so add one to get
    // the number of registers.
    Info.NumVGPR = HighestVGPRReg == RVGPU::NoRegister
                       ? 0
                       : TRI.getHWRegIndex(HighestVGPRReg) + 1;

    return Info;
  }

  int32_t MaxVGPR = -1;
  uint64_t CalleeFrameSize = 0;

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      // TODO: Check regmasks? Do they occur anywhere except calls?
      for (const MachineOperand &MO : MI.operands()) {
        unsigned Width = 0;

        if (!MO.isReg())
          continue;

        Register Reg = MO.getReg();
        switch (Reg) {
        case RVGPU::EXEC:
        case RVGPU::EXEC_LO:
        case RVGPU::EXEC_HI:
          continue;
        case RVGPU::NoRegister:
          assert(MI.isDebugInstr() &&
                 "Instruction uses invalid noreg register");
          continue;
        case RVGPU::VCC:
        case RVGPU::VCC_LO:
        case RVGPU::VCC_HI:
        case RVGPU::VCC_LO_LO16:
        case RVGPU::VCC_LO_HI16:
        case RVGPU::VCC_HI_LO16:
        case RVGPU::VCC_HI_HI16:
          Info.UsesVCC = true;
          continue;
        default:
          break;
        }
        if (RVGPU::GPR32RegClass.contains(Reg) ||
                   RVGPU::GPR_LO16RegClass.contains(Reg) ||
                   RVGPU::GPR_HI16RegClass.contains(Reg)) {
          Width = 1;
        } else if (RVGPU::GPR64RegClass.contains(Reg)) {
          Width = 2;
        } else if (RVGPU::GPR96RegClass.contains(Reg)) {
          Width = 3;
        } else if (RVGPU::GPR128RegClass.contains(Reg)) {
          Width = 4;
        } else if (RVGPU::GPR160RegClass.contains(Reg)) {
          Width = 5;
        } else if (RVGPU::GPR192RegClass.contains(Reg)) {
          Width = 6;
        } else if (RVGPU::GPR224RegClass.contains(Reg)) {
          Width = 7;
        } else if (RVGPU::GPR256RegClass.contains(Reg)) {
          Width = 8;
        } else if (RVGPU::GPR288RegClass.contains(Reg)) {
          Width = 9;
        } else if (RVGPU::GPR320RegClass.contains(Reg)) {
          Width = 10;
        } else if (RVGPU::GPR352RegClass.contains(Reg)) {
          Width = 11;
        } else if (RVGPU::GPR384RegClass.contains(Reg)) {
          Width = 12;
        } else if (RVGPU::GPR512RegClass.contains(Reg)) {
          Width = 16;
        } else if (RVGPU::GPR1024RegClass.contains(Reg)) {
          Width = 32;
        } else {
          assert(0 && "Unknown register class");
        }
        unsigned HWReg = TRI.getHWRegIndex(Reg);
        int MaxUsed = HWReg + Width - 1;
          MaxVGPR = MaxUsed > MaxVGPR ? MaxUsed : MaxVGPR;
      }
#if 0
      if (MI.isCall()) {
        // Pseudo used just to encode the underlying global. Is there a better
        // way to track this?

        const MachineOperand *CalleeOp =
            TII->getNamedOperand(MI, RVGPU::OpName::callee);

        const Function *Callee = getCalleeFunction(*CalleeOp);
        DenseMap<const Function *, RVFunctionResourceInfo>::const_iterator I =
            CallGraphResourceInfo.end();

        // Avoid crashing on undefined behavior with an illegal call to a
        // kernel. If a callsite's calling convention doesn't match the
        // function's, it's undefined behavior. If the callsite calling
        // convention does match, that would have errored earlier.
        if (Callee && RVGPU::isEntryFunctionCC(Callee->getCallingConv()))
          report_fatal_error("invalid call to entry function");

        bool IsIndirect = !Callee || Callee->isDeclaration();
        if (!IsIndirect)
          I = CallGraphResourceInfo.find(Callee);

        // FIXME: Call site could have norecurse on it
        if (!Callee || !Callee->doesNotRecurse()) {
          Info.HasRecursion = true;

          // TODO: If we happen to know there is no stack usage in the
          // callgraph, we don't need to assume an infinitely growing stack.
          if (!MI.isReturn()) {
            // We don't need to assume an unknown stack size for tail calls.

            // FIXME: This only benefits in the case where the kernel does not
            // directly call the tail called function. If a kernel directly
            // calls a tail recursive function, we'll assume maximum stack size
            // based on the regular call instruction.
            CalleeFrameSize =
              std::max(CalleeFrameSize,
                       static_cast<uint64_t>(AssumedStackSizeForExternalCall));
          }
        }

        if (IsIndirect || I == CallGraphResourceInfo.end()) {
          CalleeFrameSize =
              std::max(CalleeFrameSize,
                       static_cast<uint64_t>(AssumedStackSizeForExternalCall));

          // Register usage of indirect calls gets handled later
          Info.UsesVCC = true;
          Info.UsesFlatScratch = ST.hasFlatAddressSpace();
          Info.HasDynamicallySizedStack = true;
          Info.HasIndirectCall = true;
        } else {
          // We force CodeGen to run in SCC order, so the callee's register
          // usage etc. should be the cumulative usage of all callees.
          MaxVGPR = std::max(I->second.NumVGPR - 1, MaxVGPR);
          CalleeFrameSize =
              std::max(I->second.PrivateSegmentSize, CalleeFrameSize);
          Info.UsesVCC |= I->second.UsesVCC;
          Info.UsesFlatScratch |= I->second.UsesFlatScratch;
          Info.HasDynamicallySizedStack |= I->second.HasDynamicallySizedStack;
          Info.HasRecursion |= I->second.HasRecursion;
          Info.HasIndirectCall |= I->second.HasIndirectCall;
        }
      }
    #endif 
    }
  }

  Info.NumVGPR = MaxVGPR + 1;
  Info.PrivateSegmentSize += CalleeFrameSize;

  return Info;
}

void RVGPUResourceUsageAnalysis::propagateIndirectCallRegisterUsage() {
  // Collect the maximum number of registers from non-hardware-entrypoints.
  // All these functions are potential targets for indirect calls.
  int32_t NonKernelMaxVGPRs = 0;
#if 0
  for (const auto &I : CallGraphResourceInfo) {
    if (!RVGPU::isEntryFunctionCC(I.getFirst()->getCallingConv())) {
      auto &Info = I.getSecond();
      NonKernelMaxVGPRs = std::max(NonKernelMaxVGPRs, Info.NumVGPR);
    }
  }
#endif 
  // Add register usage for functions with indirect calls.
  // For calls to unknown functions, we assume the maximum register usage of
  // all non-hardware-entrypoints in the current module.
  for (auto &I : CallGraphResourceInfo) {
    auto &Info = I.getSecond();
    if (Info.HasIndirectCall) {
      Info.NumVGPR = std::max(Info.NumVGPR, NonKernelMaxVGPRs);
    }
  }
}
