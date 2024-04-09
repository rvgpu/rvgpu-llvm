//===-- RVGPUTargetMachine.cpp - Define TargetMachine for RVGPU -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the RVGPU target.
//
//===----------------------------------------------------------------------===//

#include "RVGPUTargetMachine.h"
#include "RVGPU.h"
#include "RVGPUAliasAnalysis.h"
#include "RVGPUAllocaHoisting.h"
#include "RVGPUAtomicLower.h"
#include "RVGPUCtorDtorLowering.h"
#include "RVGPULowerAggrCopies.h"
#include "RVGPUMachineFunctionInfo.h"
#include "RVGPUTargetObjectFile.h"
#include "RVGPUTargetTransformInfo.h"
#include "TargetInfo/RVGPUTargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IntrinsicsRVGPU.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Vectorize/LoadStoreVectorizer.h"
#include <cassert>
#include <optional>
#include <string>

using namespace llvm;

// LSV is still relatively new; this switch lets us turn it off in case we
// encounter (or suspect) a bug.
static cl::opt<bool>
    DisableLoadStoreVectorizer("disable-nvptx-load-store-vectorizer",
                               cl::desc("Disable load/store vectorizer"),
                               cl::init(false), cl::Hidden);

// TODO: Remove this flag when we are confident with no regressions.
static cl::opt<bool> DisableRequireStructuredCFG(
    "disable-nvptx-require-structured-cfg",
    cl::desc("Transitional flag to turn off RVGPU's requirement on preserving "
             "structured CFG. The requirement should be disabled only when "
             "unexpected regressions happen."),
    cl::init(false), cl::Hidden);

static cl::opt<bool> UseShortPointersOpt(
    "nvptx-short-ptr",
    cl::desc(
        "Use 32-bit pointers for accessing const/local/shared address spaces."),
    cl::init(false), cl::Hidden);

namespace llvm {

void initializeGenericToNVVMLegacyPassPass(PassRegistry &);
void initializeRVGPUAllocaHoistingPass(PassRegistry &);
void initializeRVGPUAssignValidGlobalNamesPass(PassRegistry &);
void initializeRVGPUAtomicLowerPass(PassRegistry &);
void initializeRVGPUCtorDtorLoweringLegacyPass(PassRegistry &);
void initializeRVGPULowerAggrCopiesPass(PassRegistry &);
void initializeRVGPULowerAllocaPass(PassRegistry &);
void initializeRVGPULowerUnreachablePass(PassRegistry &);
void initializeRVGPUCtorDtorLoweringLegacyPass(PassRegistry &);
void initializeRVGPULowerArgsPass(PassRegistry &);
void initializeRVGPUProxyRegErasurePass(PassRegistry &);
void initializeNVVMIntrRangePass(PassRegistry &);
void initializeNVVMReflectPass(PassRegistry &);
void initializeRVGPUAAWrapperPassPass(PassRegistry &);
void initializeRVGPUExternalAAWrapperPass(PassRegistry &);

} // end namespace llvm

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeRVGPUTarget() {
  // Register the target.
  RegisterTargetMachine<RVGPUTargetMachine32> X(getTheRVGPUTarget32());
  RegisterTargetMachine<RVGPUTargetMachine64> Y(getTheRVGPUTarget64());

  PassRegistry &PR = *PassRegistry::getPassRegistry();
  // FIXME: This pass is really intended to be invoked during IR optimization,
  // but it's very RVGPU-specific.
  initializeNVVMReflectPass(PR);
  initializeNVVMIntrRangePass(PR);
  initializeGenericToNVVMLegacyPassPass(PR);
  initializeRVGPUAllocaHoistingPass(PR);
  initializeRVGPUAssignValidGlobalNamesPass(PR);
  initializeRVGPUAtomicLowerPass(PR);
  initializeRVGPULowerArgsPass(PR);
  initializeRVGPULowerAllocaPass(PR);
  initializeRVGPULowerUnreachablePass(PR);
  initializeRVGPUCtorDtorLoweringLegacyPass(PR);
  initializeRVGPULowerAggrCopiesPass(PR);
  initializeRVGPUProxyRegErasurePass(PR);
  initializeRVGPUDAGToDAGISelPass(PR);
  initializeRVGPUAAWrapperPassPass(PR);
  initializeRVGPUExternalAAWrapperPass(PR);
}

static std::string computeDataLayout(bool is64Bit, bool UseShortPointers) {
  std::string Ret = "e";

  if (!is64Bit)
    Ret += "-p:32:32";
  else if (UseShortPointers)
    Ret += "-p3:32:32-p4:32:32-p5:32:32";

  Ret += "-i64:64-i128:128-v16:16-v32:32-n16:32:64";

  return Ret;
}

RVGPUTargetMachine::RVGPUTargetMachine(const Target &T, const Triple &TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       std::optional<Reloc::Model> RM,
                                       std::optional<CodeModel::Model> CM,
                                       CodeGenOptLevel OL, bool is64bit)
    // The pic relocation model is used regardless of what the client has
    // specified, as it is the only relocation model currently supported.
    : LLVMTargetMachine(T, computeDataLayout(is64bit, UseShortPointersOpt), TT,
                        CPU, FS, Options, Reloc::PIC_,
                        getEffectiveCodeModel(CM, CodeModel::Small), OL),
      is64bit(is64bit), UseShortPointers(UseShortPointersOpt),
      TLOF(std::make_unique<RVGPUTargetObjectFile>()),
      Subtarget(TT, std::string(CPU), std::string(FS), *this),
      StrPool(StrAlloc) {
  if (TT.getOS() == Triple::NVCL)
    drvInterface = RVGPU::NVCL;
  else
    drvInterface = RVGPU::CUDA;
  if (!DisableRequireStructuredCFG)
    setRequiresStructuredCFG(true);
  initAsmInfo();
}

RVGPUTargetMachine::~RVGPUTargetMachine() = default;

void RVGPUTargetMachine32::anchor() {}

RVGPUTargetMachine32::RVGPUTargetMachine32(const Target &T, const Triple &TT,
                                           StringRef CPU, StringRef FS,
                                           const TargetOptions &Options,
                                           std::optional<Reloc::Model> RM,
                                           std::optional<CodeModel::Model> CM,
                                           CodeGenOptLevel OL, bool JIT)
    : RVGPUTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {}

void RVGPUTargetMachine64::anchor() {}

RVGPUTargetMachine64::RVGPUTargetMachine64(const Target &T, const Triple &TT,
                                           StringRef CPU, StringRef FS,
                                           const TargetOptions &Options,
                                           std::optional<Reloc::Model> RM,
                                           std::optional<CodeModel::Model> CM,
                                           CodeGenOptLevel OL, bool JIT)
    : RVGPUTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true) {}

namespace {

class RVGPUPassConfig : public TargetPassConfig {
public:
  RVGPUPassConfig(RVGPUTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  RVGPUTargetMachine &getRVGPUTargetMachine() const {
    return getTM<RVGPUTargetMachine>();
  }

  void addIRPasses() override;
  bool addInstSelector() override;
  void addPreRegAlloc() override;
  void addPostRegAlloc() override;
  void addMachineSSAOptimization() override;

  FunctionPass *createTargetRegisterAllocator(bool) override;
  void addFastRegAlloc() override;
  void addOptimizedRegAlloc() override;

  bool addRegAssignAndRewriteFast() override {
    llvm_unreachable("should not be used");
  }

  bool addRegAssignAndRewriteOptimized() override {
    llvm_unreachable("should not be used");
  }

private:
  // If the opt level is aggressive, add GVN; otherwise, add EarlyCSE. This
  // function is only called in opt mode.
  void addEarlyCSEOrGVNPass();

  // Add passes that propagate special memory spaces.
  void addAddressSpaceInferencePasses();

  // Add passes that perform straight-line scalar optimizations.
  void addStraightLineScalarOptimizationPasses();
};

} // end anonymous namespace

TargetPassConfig *RVGPUTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new RVGPUPassConfig(*this, PM);
}

MachineFunctionInfo *RVGPUTargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return RVGPUMachineFunctionInfo::create<RVGPUMachineFunctionInfo>(Allocator,
                                                                    F, STI);
}

void RVGPUTargetMachine::registerDefaultAliasAnalyses(AAManager &AAM) {
  AAM.registerFunctionAnalysis<RVGPUAA>();
}

void RVGPUTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
  PB.registerPipelineParsingCallback(
      [](StringRef PassName, FunctionPassManager &PM,
         ArrayRef<PassBuilder::PipelineElement>) {
        if (PassName == "nvvm-reflect") {
          PM.addPass(NVVMReflectPass());
          return true;
        }
        if (PassName == "nvvm-intr-range") {
          PM.addPass(NVVMIntrRangePass());
          return true;
        }
        return false;
      });

  PB.registerAnalysisRegistrationCallback([](FunctionAnalysisManager &FAM) {
    FAM.registerPass([&] { return RVGPUAA(); });
  });

  PB.registerParseAACallback([](StringRef AAName, AAManager &AAM) {
    if (AAName == "nvptx-aa") {
      AAM.registerFunctionAnalysis<RVGPUAA>();
      return true;
    }
    return false;
  });

  PB.registerPipelineParsingCallback(
      [](StringRef PassName, ModulePassManager &PM,
         ArrayRef<PassBuilder::PipelineElement>) {
        if (PassName == "nvptx-lower-ctor-dtor") {
          PM.addPass(RVGPUCtorDtorLoweringPass());
          return true;
        }
        if (PassName == "generic-to-nvvm") {
          PM.addPass(GenericToNVVMPass());
          return true;
        }
        return false;
      });

  PB.registerPipelineStartEPCallback(
      [this](ModulePassManager &PM, OptimizationLevel Level) {
        FunctionPassManager FPM;
        FPM.addPass(NVVMReflectPass(Subtarget.getSmVersion()));
        // FIXME: NVVMIntrRangePass is causing numerical discrepancies,
        // investigate and re-enable.
        // FPM.addPass(NVVMIntrRangePass(Subtarget.getSmVersion()));
        PM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
      });
}

TargetTransformInfo
RVGPUTargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(RVGPUTTIImpl(this, F));
}

std::pair<const Value *, unsigned>
RVGPUTargetMachine::getPredicatedAddrSpace(const Value *V) const {
  if (auto *II = dyn_cast<IntrinsicInst>(V)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::rvgpu_isspacep_const:
      return std::make_pair(II->getArgOperand(0), llvm::ADDRESS_SPACE_CONST);
    case Intrinsic::rvgpu_isspacep_global:
      return std::make_pair(II->getArgOperand(0), llvm::ADDRESS_SPACE_GLOBAL);
    case Intrinsic::rvgpu_isspacep_local:
      return std::make_pair(II->getArgOperand(0), llvm::ADDRESS_SPACE_LOCAL);
    case Intrinsic::rvgpu_isspacep_shared:
    case Intrinsic::rvgpu_isspacep_shared_cluster:
      return std::make_pair(II->getArgOperand(0), llvm::ADDRESS_SPACE_SHARED);
    default:
      break;
    }
  }
  return std::make_pair(nullptr, -1);
}

void RVGPUPassConfig::addEarlyCSEOrGVNPass() {
  if (getOptLevel() == CodeGenOptLevel::Aggressive)
    addPass(createGVNPass());
  else
    addPass(createEarlyCSEPass());
}

void RVGPUPassConfig::addAddressSpaceInferencePasses() {
  // RVGPULowerArgs emits alloca for byval parameters which can often
  // be eliminated by SROA.
  addPass(createSROAPass());
  addPass(createRVGPULowerAllocaPass());
  addPass(createInferAddressSpacesPass());
  addPass(createRVGPUAtomicLowerPass());
}

void RVGPUPassConfig::addStraightLineScalarOptimizationPasses() {
  addPass(createSeparateConstOffsetFromGEPPass());
  addPass(createSpeculativeExecutionPass());
  // ReassociateGEPs exposes more opportunites for SLSR. See
  // the example in reassociate-geps-and-slsr.ll.
  addPass(createStraightLineStrengthReducePass());
  // SeparateConstOffsetFromGEP and SLSR creates common expressions which GVN or
  // EarlyCSE can reuse. GVN generates significantly better code than EarlyCSE
  // for some of our benchmarks.
  addEarlyCSEOrGVNPass();
  // Run NaryReassociate after EarlyCSE/GVN to be more effective.
  addPass(createNaryReassociatePass());
  // NaryReassociate on GEPs creates redundant common expressions, so run
  // EarlyCSE after it.
  addPass(createEarlyCSEPass());
}

void RVGPUPassConfig::addIRPasses() {
  // The following passes are known to not play well with virtual regs hanging
  // around after register allocation (which in our case, is *all* registers).
  // We explicitly disable them here.  We do, however, need some functionality
  // of the PrologEpilogCodeInserter pass, so we emulate that behavior in the
  // RVGPUPrologEpilog pass (see RVGPUPrologEpilogPass.cpp).
  disablePass(&PrologEpilogCodeInserterID);
  disablePass(&MachineLateInstrsCleanupID);
  disablePass(&MachineCopyPropagationID);
  disablePass(&TailDuplicateID);
  disablePass(&StackMapLivenessID);
  disablePass(&LiveDebugValuesID);
  disablePass(&PostRAMachineSinkingID);
  disablePass(&PostRASchedulerID);
  disablePass(&FuncletLayoutID);
  disablePass(&PatchableFunctionID);
  disablePass(&ShrinkWrapID);

  addPass(createRVGPUAAWrapperPass());
  addPass(createExternalAAWrapperPass([](Pass &P, Function &, AAResults &AAR) {
    if (auto *WrapperPass = P.getAnalysisIfAvailable<RVGPUAAWrapperPass>())
      AAR.addAAResult(WrapperPass->getResult());
  }));

  // NVVMReflectPass is added in addEarlyAsPossiblePasses, so hopefully running
  // it here does nothing.  But since we need it for correctness when lowering
  // to RVGPU, run it here too, in case whoever built our pass pipeline didn't
  // call addEarlyAsPossiblePasses.
  const RVGPUSubtarget &ST = *getTM<RVGPUTargetMachine>().getSubtargetImpl();
  addPass(createNVVMReflectPass(ST.getSmVersion()));

  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createRVGPUImageOptimizerPass());
  addPass(createRVGPUAssignValidGlobalNamesPass());
  addPass(createGenericToNVVMLegacyPass());

  // RVGPULowerArgs is required for correctness and should be run right
  // before the address space inference passes.
  addPass(createRVGPULowerArgsPass());
  if (getOptLevel() != CodeGenOptLevel::None) {
    addAddressSpaceInferencePasses();
    addStraightLineScalarOptimizationPasses();
  }

  addPass(createAtomicExpandPass());
  addPass(createRVGPUCtorDtorLoweringLegacyPass());

  // === LSR and other generic IR passes ===
  TargetPassConfig::addIRPasses();
  // EarlyCSE is not always strong enough to clean up what LSR produces. For
  // example, GVN can combine
  //
  //   %0 = add %a, %b
  //   %1 = add %b, %a
  //
  // and
  //
  //   %0 = shl nsw %a, 2
  //   %1 = shl %a, 2
  //
  // but EarlyCSE can do neither of them.
  if (getOptLevel() != CodeGenOptLevel::None) {
    addEarlyCSEOrGVNPass();
    if (!DisableLoadStoreVectorizer)
      addPass(createLoadStoreVectorizerPass());
    addPass(createSROAPass());
  }

  const auto &Options = getRVGPUTargetMachine().Options;
  addPass(createRVGPULowerUnreachablePass(Options.TrapUnreachable,
                                          Options.NoTrapAfterNoreturn));
}

bool RVGPUPassConfig::addInstSelector() {
  const RVGPUSubtarget &ST = *getTM<RVGPUTargetMachine>().getSubtargetImpl();

  addPass(createLowerAggrCopies());
  addPass(createAllocaHoisting());
  addPass(createRVGPUISelDag(getRVGPUTargetMachine(), getOptLevel()));

  if (!ST.hasImageHandles())
    addPass(createRVGPUReplaceImageHandlesPass());

  return false;
}

void RVGPUPassConfig::addPreRegAlloc() {
  // Remove Proxy Register pseudo instructions used to keep `callseq_end` alive.
  addPass(createRVGPUProxyRegErasurePass());
}

void RVGPUPassConfig::addPostRegAlloc() {
  addPass(createRVGPUPrologEpilogPass());
  if (getOptLevel() != CodeGenOptLevel::None) {
    // RVGPUPrologEpilogPass calculates frame object offset and replace frame
    // index with VRFrame register. RVGPUPeephole need to be run after that and
    // will replace VRFrame with VRFrameLocal when possible.
    addPass(createRVGPUPeephole());
  }
}

FunctionPass *RVGPUPassConfig::createTargetRegisterAllocator(bool) {
  return nullptr; // No reg alloc
}

void RVGPUPassConfig::addFastRegAlloc() {
  addPass(&PHIEliminationID);
  addPass(&TwoAddressInstructionPassID);
}

void RVGPUPassConfig::addOptimizedRegAlloc() {
  addPass(&ProcessImplicitDefsID);
  addPass(&LiveVariablesID);
  addPass(&MachineLoopInfoID);
  addPass(&PHIEliminationID);

  addPass(&TwoAddressInstructionPassID);
  addPass(&RegisterCoalescerID);

  // PreRA instruction scheduling.
  if (addPass(&MachineSchedulerID))
    printAndVerify("After Machine Scheduling");

  addPass(&StackSlotColoringID);

  // FIXME: Needs physical registers
  // addPass(&MachineLICMID);

  printAndVerify("After StackSlotColoring");
}

void RVGPUPassConfig::addMachineSSAOptimization() {
  // Pre-ra tail duplication.
  if (addPass(&EarlyTailDuplicateID))
    printAndVerify("After Pre-RegAlloc TailDuplicate");

  // Optimize PHIs before DCE: removing dead PHI cycles may make more
  // instructions dead.
  addPass(&OptimizePHIsID);

  // This pass merges large allocas. StackSlotColoring is a different pass
  // which merges spill slots.
  addPass(&StackColoringID);

  // If the target requests it, assign local variables to stack slots relative
  // to one another and simplify frame index references where possible.
  addPass(&LocalStackSlotAllocationID);

  // With optimization, dead code should already be eliminated. However
  // there is one known exception: lowered code for arguments that are only
  // used by tail calls, where the tail calls reuse the incoming stack
  // arguments directly (see t11 in test/CodeGen/X86/sibcall.ll).
  addPass(&DeadMachineInstructionElimID);
  printAndVerify("After codegen DCE pass");

  // Allow targets to insert passes that improve instruction level parallelism,
  // like if-conversion. Such passes will typically need dominator trees and
  // loop info, just like LICM and CSE below.
  if (addILPOpts())
    printAndVerify("After ILP optimizations");

  addPass(&EarlyMachineLICMID);
  addPass(&MachineCSEID);

  addPass(&MachineSinkingID);
  printAndVerify("After Machine LICM, CSE and Sinking passes");

  addPass(&PeepholeOptimizerID);
  printAndVerify("After codegen peephole optimization pass");
}
