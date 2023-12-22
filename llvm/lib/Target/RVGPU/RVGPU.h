//===-- RVGPU.h - MachineFunction passes hw codegen --------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
/// \file
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RVGPU_RVGPU_H
#define LLVM_LIB_TARGET_RVGPU_RVGPU_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/RVGPUAddrSpace.h"
#include "llvm/Support/CodeGen.h"

namespace llvm {

class RVGPUTargetMachine;
class TargetMachine;

// GlobalISel passes
void initializeRVGPUPreLegalizerCombinerPass(PassRegistry &);
FunctionPass *createRVGPUPreLegalizeCombiner(bool IsOptNone);
void initializeRVGPUPostLegalizerCombinerPass(PassRegistry &);
FunctionPass *createRVGPUPostLegalizeCombiner(bool IsOptNone);
FunctionPass *createRVGPURegBankCombiner(bool IsOptNone);
void initializeRVGPURegBankCombinerPass(PassRegistry &);

void initializeRVGPURegBankSelectPass(PassRegistry &);

// SI Passes
FunctionPass *createGCNDPPCombinePass();
FunctionPass *createSIAnnotateControlFlowPass();
FunctionPass *createSIFoldOperandsPass();
FunctionPass *createSIPeepholeSDWAPass();
FunctionPass *createSILowerI1CopiesPass();
FunctionPass *createRVGPUGlobalISelDivergenceLoweringPass();
FunctionPass *createSIShrinkInstructionsPass();
FunctionPass *createSILoadStoreOptimizerPass();
FunctionPass *createSIWholeQuadModePass();
FunctionPass *createSIFixControlFlowLiveIntervalsPass();
FunctionPass *createSIOptimizeExecMaskingPreRAPass();
FunctionPass *createSIOptimizeVGPRLiveRangePass();
FunctionPass *createSIFixSGPRCopiesPass();
FunctionPass *createLowerWWMCopiesPass();
FunctionPass *createSIMemoryLegalizerPass();
FunctionPass *createSIInsertWaitcntsPass();
FunctionPass *createSIPreAllocateWWMRegsPass();
FunctionPass *createSIFormMemoryClausesPass();

FunctionPass *createSIPostRABundlerPass();
FunctionPass *createRVGPUImageIntrinsicOptimizerPass(const TargetMachine *);
ModulePass *createRVGPURemoveIncompatibleFunctionsPass(const TargetMachine *);
FunctionPass *createRVGPUCodeGenPreparePass();
FunctionPass *createRVGPULateCodeGenPreparePass();
FunctionPass *createRVGPUMachineCFGStructurizerPass();
FunctionPass *createRVGPURewriteOutArgumentsPass();
ModulePass *
createRVGPULowerModuleLDSLegacyPass(const RVGPUTargetMachine *TM = nullptr);
FunctionPass *createSIModeRegisterPass();
FunctionPass *createGCNPreRAOptimizationsPass();

struct RVGPUSimplifyLibCallsPass : PassInfoMixin<RVGPUSimplifyLibCallsPass> {
  RVGPUSimplifyLibCallsPass() {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

struct RVGPUImageIntrinsicOptimizerPass
    : PassInfoMixin<RVGPUImageIntrinsicOptimizerPass> {
  RVGPUImageIntrinsicOptimizerPass(TargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  TargetMachine &TM;
};

struct RVGPUUseNativeCallsPass : PassInfoMixin<RVGPUUseNativeCallsPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

void initializeRVGPUDAGToDAGISelPass(PassRegistry&);

void initializeRVGPUMachineCFGStructurizerPass(PassRegistry&);
extern char &RVGPUMachineCFGStructurizerID;

void initializeRVGPUAlwaysInlinePass(PassRegistry&);

Pass *createRVGPUAnnotateKernelFeaturesPass();
Pass *createRVGPUAttributorLegacyPass();
void initializeRVGPUAttributorLegacyPass(PassRegistry &);
void initializeRVGPUAnnotateKernelFeaturesPass(PassRegistry &);
extern char &RVGPUAnnotateKernelFeaturesID;

// DPP/Iterative option enables the atomic optimizer with given strategy
// whereas None disables the atomic optimizer.
enum class ScanOptions { DPP, Iterative, None };
FunctionPass *createRVGPUAtomicOptimizerPass(ScanOptions ScanStrategy);
void initializeRVGPUAtomicOptimizerPass(PassRegistry &);
extern char &RVGPUAtomicOptimizerID;

ModulePass *createRVGPUCtorDtorLoweringLegacyPass();
void initializeRVGPUCtorDtorLoweringLegacyPass(PassRegistry &);
extern char &RVGPUCtorDtorLoweringLegacyPassID;

FunctionPass *createRVGPULowerKernelArgumentsPass();
void initializeRVGPULowerKernelArgumentsPass(PassRegistry &);
extern char &RVGPULowerKernelArgumentsID;

FunctionPass *createRVGPUPromoteKernelArgumentsPass();
void initializeRVGPUPromoteKernelArgumentsPass(PassRegistry &);
extern char &RVGPUPromoteKernelArgumentsID;

struct RVGPUPromoteKernelArgumentsPass
    : PassInfoMixin<RVGPUPromoteKernelArgumentsPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

ModulePass *createRVGPULowerKernelAttributesPass();
void initializeRVGPULowerKernelAttributesPass(PassRegistry &);
extern char &RVGPULowerKernelAttributesID;

struct RVGPULowerKernelAttributesPass
    : PassInfoMixin<RVGPULowerKernelAttributesPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

void initializeRVGPULowerModuleLDSLegacyPass(PassRegistry &);
extern char &RVGPULowerModuleLDSLegacyPassID;

struct RVGPULowerModuleLDSPass : PassInfoMixin<RVGPULowerModuleLDSPass> {
  const RVGPUTargetMachine &TM;
  RVGPULowerModuleLDSPass(const RVGPUTargetMachine &TM_) : TM(TM_) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

void initializeRVGPURewriteOutArgumentsPass(PassRegistry &);
extern char &RVGPURewriteOutArgumentsID;

void initializeGCNDPPCombinePass(PassRegistry &);
extern char &GCNDPPCombineID;

void initializeSIFoldOperandsPass(PassRegistry &);
extern char &SIFoldOperandsID;

void initializeSIPeepholeSDWAPass(PassRegistry &);
extern char &SIPeepholeSDWAID;

void initializeSIShrinkInstructionsPass(PassRegistry&);
extern char &SIShrinkInstructionsID;

void initializeSIFixSGPRCopiesPass(PassRegistry &);
extern char &SIFixSGPRCopiesID;

void initializeSIFixVGPRCopiesPass(PassRegistry &);
extern char &SIFixVGPRCopiesID;

void initializeSILowerWWMCopiesPass(PassRegistry &);
extern char &SILowerWWMCopiesID;

void initializeSILowerI1CopiesPass(PassRegistry &);
extern char &SILowerI1CopiesID;

void initializeRVGPUGlobalISelDivergenceLoweringPass(PassRegistry &);
extern char &RVGPUGlobalISelDivergenceLoweringID;

void initializeSILowerSGPRSpillsPass(PassRegistry &);
extern char &SILowerSGPRSpillsID;

void initializeSILoadStoreOptimizerPass(PassRegistry &);
extern char &SILoadStoreOptimizerID;

void initializeSIWholeQuadModePass(PassRegistry &);
extern char &SIWholeQuadModeID;

void initializeSILowerControlFlowPass(PassRegistry &);
extern char &SILowerControlFlowID;

void initializeSIPreEmitPeepholePass(PassRegistry &);
extern char &SIPreEmitPeepholeID;

void initializeSILateBranchLoweringPass(PassRegistry &);
extern char &SILateBranchLoweringPassID;

void initializeSIOptimizeExecMaskingPass(PassRegistry &);
extern char &SIOptimizeExecMaskingID;

void initializeSIPreAllocateWWMRegsPass(PassRegistry &);
extern char &SIPreAllocateWWMRegsID;

void initializeRVGPUImageIntrinsicOptimizerPass(PassRegistry &);
extern char &RVGPUImageIntrinsicOptimizerID;

void initializeRVGPUPerfHintAnalysisPass(PassRegistry &);
extern char &RVGPUPerfHintAnalysisID;

void initializeGCNRegPressurePrinterPass(PassRegistry &);
extern char &GCNRegPressurePrinterID;

// Passes common to R600 and SI
FunctionPass *createRVGPUPromoteAlloca();
void initializeRVGPUPromoteAllocaPass(PassRegistry&);
extern char &RVGPUPromoteAllocaID;

FunctionPass *createRVGPUPromoteAllocaToVector();
void initializeRVGPUPromoteAllocaToVectorPass(PassRegistry&);
extern char &RVGPUPromoteAllocaToVectorID;

struct RVGPUPromoteAllocaPass : PassInfoMixin<RVGPUPromoteAllocaPass> {
  RVGPUPromoteAllocaPass(TargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  TargetMachine &TM;
};

struct RVGPUPromoteAllocaToVectorPass
    : PassInfoMixin<RVGPUPromoteAllocaToVectorPass> {
  RVGPUPromoteAllocaToVectorPass(TargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  TargetMachine &TM;
};

struct RVGPUAtomicOptimizerPass : PassInfoMixin<RVGPUAtomicOptimizerPass> {
  RVGPUAtomicOptimizerPass(TargetMachine &TM, ScanOptions ScanImpl)
      : TM(TM), ScanImpl(ScanImpl) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  TargetMachine &TM;
  ScanOptions ScanImpl;
};

Pass *createRVGPUStructurizeCFGPass();
FunctionPass *createRVGPUISelDag(TargetMachine &TM, CodeGenOptLevel OptLevel);
ModulePass *createRVGPUAlwaysInlinePass(bool GlobalOpt = true);

struct RVGPUAlwaysInlinePass : PassInfoMixin<RVGPUAlwaysInlinePass> {
  RVGPUAlwaysInlinePass(bool GlobalOpt = true) : GlobalOpt(GlobalOpt) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  bool GlobalOpt;
};

class RVGPUCodeGenPreparePass
    : public PassInfoMixin<RVGPUCodeGenPreparePass> {
private:
  TargetMachine &TM;

public:
  RVGPUCodeGenPreparePass(TargetMachine &TM) : TM(TM){};
  PreservedAnalyses run(Function &, FunctionAnalysisManager &);
};

class RVGPULowerKernelArgumentsPass
    : public PassInfoMixin<RVGPULowerKernelArgumentsPass> {
private:
  TargetMachine &TM;

public:
  RVGPULowerKernelArgumentsPass(TargetMachine &TM) : TM(TM){};
  PreservedAnalyses run(Function &, FunctionAnalysisManager &);
};

class RVGPUAttributorPass : public PassInfoMixin<RVGPUAttributorPass> {
private:
  TargetMachine &TM;

public:
  RVGPUAttributorPass(TargetMachine &TM) : TM(TM){};
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

FunctionPass *createRVGPUAnnotateUniformValues();

ModulePass *createRVGPUPrintfRuntimeBinding();
void initializeRVGPUPrintfRuntimeBindingPass(PassRegistry&);
extern char &RVGPUPrintfRuntimeBindingID;

void initializeRVGPUResourceUsageAnalysisPass(PassRegistry &);
extern char &RVGPUResourceUsageAnalysisID;

struct RVGPUPrintfRuntimeBindingPass
    : PassInfoMixin<RVGPUPrintfRuntimeBindingPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass* createRVGPUUnifyMetadataPass();
void initializeRVGPUUnifyMetadataPass(PassRegistry&);
extern char &RVGPUUnifyMetadataID;

struct RVGPUUnifyMetadataPass : PassInfoMixin<RVGPUUnifyMetadataPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

void initializeSIOptimizeExecMaskingPreRAPass(PassRegistry&);
extern char &SIOptimizeExecMaskingPreRAID;

void initializeSIOptimizeVGPRLiveRangePass(PassRegistry &);
extern char &SIOptimizeVGPRLiveRangeID;

void initializeRVGPUAnnotateUniformValuesPass(PassRegistry&);
extern char &RVGPUAnnotateUniformValuesPassID;

void initializeRVGPUCodeGenPreparePass(PassRegistry&);
extern char &RVGPUCodeGenPrepareID;

void initializeRVGPURemoveIncompatibleFunctionsPass(PassRegistry &);
extern char &RVGPURemoveIncompatibleFunctionsID;

void initializeRVGPULateCodeGenPreparePass(PassRegistry &);
extern char &RVGPULateCodeGenPrepareID;

FunctionPass *createRVGPURewriteUndefForPHILegacyPass();
void initializeRVGPURewriteUndefForPHILegacyPass(PassRegistry &);
extern char &RVGPURewriteUndefForPHILegacyPassID;

class RVGPURewriteUndefForPHIPass
    : public PassInfoMixin<RVGPURewriteUndefForPHIPass> {
public:
  RVGPURewriteUndefForPHIPass() = default;
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

void initializeSIAnnotateControlFlowPass(PassRegistry&);
extern char &SIAnnotateControlFlowPassID;

void initializeSIMemoryLegalizerPass(PassRegistry&);
extern char &SIMemoryLegalizerID;

void initializeSIModeRegisterPass(PassRegistry&);
extern char &SIModeRegisterID;

void initializeRVGPUInsertDelayAluPass(PassRegistry &);
extern char &RVGPUInsertDelayAluID;

void initializeRVGPUInsertSingleUseVDSTPass(PassRegistry &);
extern char &RVGPUInsertSingleUseVDSTID;

void initializeSIInsertHardClausesPass(PassRegistry &);
extern char &SIInsertHardClausesID;

void initializeSIInsertWaitcntsPass(PassRegistry&);
extern char &SIInsertWaitcntsID;

void initializeSIFormMemoryClausesPass(PassRegistry&);
extern char &SIFormMemoryClausesID;

void initializeSIPostRABundlerPass(PassRegistry&);
extern char &SIPostRABundlerID;

void initializeGCNCreateVOPDPass(PassRegistry &);
extern char &GCNCreateVOPDID;

void initializeRVGPUUnifyDivergentExitNodesPass(PassRegistry&);
extern char &RVGPUUnifyDivergentExitNodesID;

ImmutablePass *createRVGPUAAWrapperPass();
void initializeRVGPUAAWrapperPassPass(PassRegistry&);
ImmutablePass *createRVGPUExternalAAWrapperPass();
void initializeRVGPUExternalAAWrapperPass(PassRegistry&);

void initializeRVGPUArgumentUsageInfoPass(PassRegistry &);

ModulePass *createRVGPUOpenCLEnqueuedBlockLoweringPass();
void initializeRVGPUOpenCLEnqueuedBlockLoweringPass(PassRegistry &);
extern char &RVGPUOpenCLEnqueuedBlockLoweringID;

void initializeGCNNSAReassignPass(PassRegistry &);
extern char &GCNNSAReassignID;

void initializeGCNPreRALongBranchRegPass(PassRegistry &);
extern char &GCNPreRALongBranchRegID;

void initializeGCNPreRAOptimizationsPass(PassRegistry &);
extern char &GCNPreRAOptimizationsID;

FunctionPass *createRVGPUSetWavePriorityPass();
void initializeRVGPUSetWavePriorityPass(PassRegistry &);

void initializeGCNRewritePartialRegUsesPass(llvm::PassRegistry &);
extern char &GCNRewritePartialRegUsesID;

namespace RVGPU {
enum TargetIndex {
  TI_CONSTDATA_START,
  TI_SCRATCH_RSRC_DWORD0,
  TI_SCRATCH_RSRC_DWORD1,
  TI_SCRATCH_RSRC_DWORD2,
  TI_SCRATCH_RSRC_DWORD3
};

// FIXME: Missing constant_32bit
inline bool isFlatGlobalAddrSpace(unsigned AS) {
  return AS == RVGPUAS::GLOBAL_ADDRESS ||
         AS == RVGPUAS::FLAT_ADDRESS ||
         AS == RVGPUAS::CONSTANT_ADDRESS ||
         AS > RVGPUAS::MAX_RVGPU_ADDRESS;
}

inline bool isExtendedGlobalAddrSpace(unsigned AS) {
  return AS == RVGPUAS::GLOBAL_ADDRESS || AS == RVGPUAS::CONSTANT_ADDRESS ||
         AS == RVGPUAS::CONSTANT_ADDRESS_32BIT ||
         AS > RVGPUAS::MAX_RVGPU_ADDRESS;
}

static inline bool addrspacesMayAlias(unsigned AS1, unsigned AS2) {
  static_assert(RVGPUAS::MAX_RVGPU_ADDRESS <= 9, "Addr space out of range");

  if (AS1 > RVGPUAS::MAX_RVGPU_ADDRESS || AS2 > RVGPUAS::MAX_RVGPU_ADDRESS)
    return true;

  // This array is indexed by address space value enum elements 0 ... to 9
  // clang-format off
  static const bool ASAliasRules[10][10] = {
    /*                       Flat   Global Region  Group Constant Private Const32 BufFatPtr BufRsrc BufStrdPtr */
    /* Flat     */            {true,  true,  false, true,  true,  true,  true,  true,  true,  true},
    /* Global   */            {true,  true,  false, false, true,  false, true,  true,  true,  true},
    /* Region   */            {false, false, true,  false, false, false, false, false, false, false},
    /* Group    */            {true,  false, false, true,  false, false, false, false, false, false},
    /* Constant */            {true,  true,  false, false, false, false, true,  true,  true,  true},
    /* Private  */            {true,  false, false, false, false, true,  false, false, false, false},
    /* Constant 32-bit */     {true,  true,  false, false, true,  false, false, true,  true,  true},
    /* Buffer Fat Ptr  */     {true,  true,  false, false, true,  false, true,  true,  true,  true},
    /* Buffer Resource */     {true,  true,  false, false, true,  false, true,  true,  true,  true},
    /* Buffer Strided Ptr  */ {true,  true,  false, false, true,  false, true,  true,  true,  true},
  };
  // clang-format on

  return ASAliasRules[AS1][AS2];
}

}

} // End namespace llvm

#endif
