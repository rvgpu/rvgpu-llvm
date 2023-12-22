//===-- RVGPUTargetMachine.cpp - TargetMachine for hw codegen targets-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// The RVGPU target machine contains all of the hardware specific
/// information  needed to emit code for SI+ GPUs.
//
//===----------------------------------------------------------------------===//

#include "RVGPUTargetMachine.h"
#include "RVGPU.h"
#include "RVGPUAliasAnalysis.h"
#include "RVGPUCtorDtorLowering.h"
#include "RVGPUExportClustering.h"
#include "RVGPUIGroupLP.h"
#include "RVGPUMacroFusion.h"
#include "RVGPURegBankSelect.h"
#include "RVGPUTargetObjectFile.h"
#include "RVGPUTargetTransformInfo.h"
#include "RVGPUUnifyDivergentExitNodes.h"
#include "GCNIterativeScheduler.h"
#include "GCNSchedStrategy.h"
#include "GCNVOPDUtils.h"
#include "RVMachineFunctionInfo.h"
#include "SIMachineScheduler.h"
#include "TargetInfo/RVGPUTargetInfo.h"
#include "Utils/RVGPUBaseInfo.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/IRTranslator.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/GlobalISel/Localizer.h"
#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"
#include "llvm/CodeGen/MIRParser/MIParser.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IntrinsicsRVGPU.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/HipStdPar/HipStdPar.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/SimplifyLibCalls.h"
#include "llvm/Transforms/Vectorize/LoadStoreVectorizer.h"
#include <optional>

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {
class SGPRRegisterRegAlloc : public RegisterRegAllocBase<SGPRRegisterRegAlloc> {
public:
  SGPRRegisterRegAlloc(const char *N, const char *D, FunctionPassCtor C)
    : RegisterRegAllocBase(N, D, C) {}
};

class VGPRRegisterRegAlloc : public RegisterRegAllocBase<VGPRRegisterRegAlloc> {
public:
  VGPRRegisterRegAlloc(const char *N, const char *D, FunctionPassCtor C)
    : RegisterRegAllocBase(N, D, C) {}
};

static bool onlyAllocateSGPRs(const TargetRegisterInfo &TRI,
                              const TargetRegisterClass &RC) {
  return static_cast<const RVRegisterInfo &>(TRI).isSGPRClass(&RC);
}

static bool onlyAllocateVGPRs(const TargetRegisterInfo &TRI,
                              const TargetRegisterClass &RC) {
  return !static_cast<const RVRegisterInfo &>(TRI).isSGPRClass(&RC);
}


/// -{sgpr|vgpr}-regalloc=... command line option.
static FunctionPass *useDefaultRegisterAllocator() { return nullptr; }

/// A dummy default pass factory indicates whether the register allocator is
/// overridden on the command line.
static llvm::once_flag InitializeDefaultSGPRRegisterAllocatorFlag;
static llvm::once_flag InitializeDefaultVGPRRegisterAllocatorFlag;

static SGPRRegisterRegAlloc
defaultSGPRRegAlloc("default",
                    "pick SGPR register allocator based on -O option",
                    useDefaultRegisterAllocator);

static cl::opt<SGPRRegisterRegAlloc::FunctionPassCtor, false,
               RegisterPassParser<SGPRRegisterRegAlloc>>
SGPRRegAlloc("sgpr-regalloc", cl::Hidden, cl::init(&useDefaultRegisterAllocator),
             cl::desc("Register allocator to use for SGPRs"));

static cl::opt<VGPRRegisterRegAlloc::FunctionPassCtor, false,
               RegisterPassParser<VGPRRegisterRegAlloc>>
VGPRRegAlloc("vgpr-regalloc", cl::Hidden, cl::init(&useDefaultRegisterAllocator),
             cl::desc("Register allocator to use for VGPRs"));


static void initializeDefaultSGPRRegisterAllocatorOnce() {
  RegisterRegAlloc::FunctionPassCtor Ctor = SGPRRegisterRegAlloc::getDefault();

  if (!Ctor) {
    Ctor = SGPRRegAlloc;
    SGPRRegisterRegAlloc::setDefault(SGPRRegAlloc);
  }
}

static void initializeDefaultVGPRRegisterAllocatorOnce() {
  RegisterRegAlloc::FunctionPassCtor Ctor = VGPRRegisterRegAlloc::getDefault();

  if (!Ctor) {
    Ctor = VGPRRegAlloc;
    VGPRRegisterRegAlloc::setDefault(VGPRRegAlloc);
  }
}

static FunctionPass *createBasicSGPRRegisterAllocator() {
  return createBasicRegisterAllocator(onlyAllocateSGPRs);
}

static FunctionPass *createGreedySGPRRegisterAllocator() {
  return createGreedyRegisterAllocator(onlyAllocateSGPRs);
}

static FunctionPass *createFastSGPRRegisterAllocator() {
  return createFastRegisterAllocator(onlyAllocateSGPRs, false);
}

static FunctionPass *createBasicVGPRRegisterAllocator() {
  return createBasicRegisterAllocator(onlyAllocateVGPRs);
}

static FunctionPass *createGreedyVGPRRegisterAllocator() {
  return createGreedyRegisterAllocator(onlyAllocateVGPRs);
}

static FunctionPass *createFastVGPRRegisterAllocator() {
  return createFastRegisterAllocator(onlyAllocateVGPRs, true);
}

static SGPRRegisterRegAlloc basicRegAllocSGPR(
  "basic", "basic register allocator", createBasicSGPRRegisterAllocator);
static SGPRRegisterRegAlloc greedyRegAllocSGPR(
  "greedy", "greedy register allocator", createGreedySGPRRegisterAllocator);

static SGPRRegisterRegAlloc fastRegAllocSGPR(
  "fast", "fast register allocator", createFastSGPRRegisterAllocator);


static VGPRRegisterRegAlloc basicRegAllocVGPR(
  "basic", "basic register allocator", createBasicVGPRRegisterAllocator);
static VGPRRegisterRegAlloc greedyRegAllocVGPR(
  "greedy", "greedy register allocator", createGreedyVGPRRegisterAllocator);

static VGPRRegisterRegAlloc fastRegAllocVGPR(
  "fast", "fast register allocator", createFastVGPRRegisterAllocator);
}

static cl::opt<bool>
EnableEarlyIfConversion("rvgpu-early-ifcvt", cl::Hidden,
                        cl::desc("Run early if-conversion"),
                        cl::init(false));

static cl::opt<bool>
OptExecMaskPreRA("rvgpu-opt-exec-mask-pre-ra", cl::Hidden,
            cl::desc("Run pre-RA exec mask optimizations"),
            cl::init(true));

static cl::opt<bool>
    LowerCtorDtor("rvgpu-lower-global-ctor-dtor",
                  cl::desc("Lower GPU ctor / dtors to globals on the device."),
                  cl::init(true), cl::Hidden);

// Option to disable vectorizer for tests.
static cl::opt<bool> EnableLoadStoreVectorizer(
  "rvgpu-load-store-vectorizer",
  cl::desc("Enable load store vectorizer"),
  cl::init(true),
  cl::Hidden);

// Option to control global loads scalarization
static cl::opt<bool> ScalarizeGlobal(
  "rvgpu-scalarize-global-loads",
  cl::desc("Enable global load scalarization"),
  cl::init(true),
  cl::Hidden);

// Option to run internalize pass.
static cl::opt<bool> InternalizeSymbols(
  "rvgpu-internalize-symbols",
  cl::desc("Enable elimination of non-kernel functions and unused globals"),
  cl::init(false),
  cl::Hidden);

// Option to inline all early.
static cl::opt<bool> EarlyInlineAll(
  "rvgpu-early-inline-all",
  cl::desc("Inline all functions early"),
  cl::init(false),
  cl::Hidden);

static cl::opt<bool> RemoveIncompatibleFunctions(
    "rvgpu-enable-remove-incompatible-functions", cl::Hidden,
    cl::desc("Enable removal of functions when they"
             "use features not supported by the target GPU"),
    cl::init(true));

static cl::opt<bool> EnableSDWAPeephole(
  "rvgpu-sdwa-peephole",
  cl::desc("Enable SDWA peepholer"),
  cl::init(true));

static cl::opt<bool> EnableDPPCombine(
  "rvgpu-dpp-combine",
  cl::desc("Enable DPP combiner"),
  cl::init(true));

// Enable address space based alias analysis
static cl::opt<bool> EnableRVGPUAliasAnalysis("enable-rvgpu-aa", cl::Hidden,
  cl::desc("Enable RVGPU Alias Analysis"),
  cl::init(true));

// Option to run late CFG structurizer
static cl::opt<bool, true> LateCFGStructurize(
  "rvgpu-late-structurize",
  cl::desc("Enable late CFG structurization"),
  cl::location(RVGPUTargetMachine::EnableLateStructurizeCFG),
  cl::Hidden);

// Enable lib calls simplifications
static cl::opt<bool> EnableLibCallSimplify(
  "rvgpu-simplify-libcall",
  cl::desc("Enable rvgpu library simplifications"),
  cl::init(true),
  cl::Hidden);

static cl::opt<bool> EnableLowerKernelArguments(
  "rvgpu-ir-lower-kernel-arguments",
  cl::desc("Lower kernel argument loads in IR pass"),
  cl::init(true),
  cl::Hidden);

static cl::opt<bool> EnableRegReassign(
  "rvgpu-reassign-regs",
  cl::desc("Enable register reassign optimizations on gfx10+"),
  cl::init(true),
  cl::Hidden);

static cl::opt<bool> OptVGPRLiveRange(
    "rvgpu-opt-vgpr-liverange",
    cl::desc("Enable VGPR liverange optimizations for if-else structure"),
    cl::init(true), cl::Hidden);

static cl::opt<ScanOptions> RVGPUAtomicOptimizerStrategy(
    "rvgpu-atomic-optimizer-strategy",
    cl::desc("Select DPP or Iterative strategy for scan"),
    cl::init(ScanOptions::Iterative),
    cl::values(
        clEnumValN(ScanOptions::DPP, "DPP", "Use DPP operations for scan"),
        clEnumValN(ScanOptions::Iterative, "Iterative",
                   "Use Iterative approach for scan"),
        clEnumValN(ScanOptions::None, "None", "Disable atomic optimizer")));

// Enable Mode register optimization
static cl::opt<bool> EnableSIModeRegisterPass(
  "rvgpu-mode-register",
  cl::desc("Enable mode register pass"),
  cl::init(true),
  cl::Hidden);

// Enable GFX11.5+ s_singleuse_vdst insertion
static cl::opt<bool>
    EnableInsertSingleUseVDST("rvgpu-enable-single-use-vdst",
                              cl::desc("Enable s_singleuse_vdst insertion"),
                              cl::init(false), cl::Hidden);

// Enable GFX11+ s_delay_alu insertion
static cl::opt<bool>
    EnableInsertDelayAlu("rvgpu-enable-delay-alu",
                         cl::desc("Enable s_delay_alu insertion"),
                         cl::init(true), cl::Hidden);

// Enable GFX11+ VOPD
static cl::opt<bool>
    EnableVOPD("rvgpu-enable-vopd",
               cl::desc("Enable VOPD, dual issue of VALU in wave32"),
               cl::init(true), cl::Hidden);

// Option is used in lit tests to prevent deadcoding of patterns inspected.
static cl::opt<bool>
EnableDCEInRA("rvgpu-dce-in-ra",
    cl::init(true), cl::Hidden,
    cl::desc("Enable machine DCE inside regalloc"));

static cl::opt<bool> EnableSetWavePriority("rvgpu-set-wave-priority",
                                           cl::desc("Adjust wave priority"),
                                           cl::init(false), cl::Hidden);

static cl::opt<bool> EnableScalarIRPasses(
  "rvgpu-scalar-ir-passes",
  cl::desc("Enable scalar IR passes"),
  cl::init(true),
  cl::Hidden);

static cl::opt<bool> EnableStructurizerWorkarounds(
    "rvgpu-enable-structurizer-workarounds",
    cl::desc("Enable workarounds for the StructurizeCFG pass"), cl::init(true),
    cl::Hidden);

static cl::opt<bool, true> EnableLowerModuleLDS(
    "rvgpu-enable-lower-module-lds", cl::desc("Enable lower module lds pass"),
    cl::location(RVGPUTargetMachine::EnableLowerModuleLDS), cl::init(true),
    cl::Hidden);

static cl::opt<bool> EnablePreRAOptimizations(
    "rvgpu-enable-pre-ra-optimizations",
    cl::desc("Enable Pre-RA optimizations pass"), cl::init(true),
    cl::Hidden);

static cl::opt<bool> EnablePromoteKernelArguments(
    "rvgpu-enable-promote-kernel-arguments",
    cl::desc("Enable promotion of flat kernel pointer arguments to global"),
    cl::Hidden, cl::init(true));

static cl::opt<bool> EnableImageIntrinsicOptimizer(
    "rvgpu-enable-image-intrinsic-optimizer",
    cl::desc("Enable image intrinsic optimizer pass"), cl::init(true),
    cl::Hidden);

static cl::opt<bool>
    EnableLoopPrefetch("rvgpu-loop-prefetch",
                       cl::desc("Enable loop data prefetch on RVGPU"),
                       cl::Hidden, cl::init(false));

static cl::opt<bool> EnableMaxIlpSchedStrategy(
    "rvgpu-enable-max-ilp-scheduling-strategy",
    cl::desc("Enable scheduling strategy to maximize ILP for a single wave."),
    cl::Hidden, cl::init(false));

static cl::opt<bool> EnableRewritePartialRegUses(
    "rvgpu-enable-rewrite-partial-reg-uses",
    cl::desc("Enable rewrite partial reg uses pass"), cl::init(true),
    cl::Hidden);

static cl::opt<bool> EnableHipStdPar(
  "rvgpu-enable-hipstdpar",
  cl::desc("Enable HIP Standard Parallelism Offload support"), cl::init(false),
  cl::Hidden);

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeRVGPUTarget() {
  // Register the target
  RegisterTargetMachine<GCNTargetMachine> Y(getTheRVGPUTarget());

  PassRegistry *PR = PassRegistry::getPassRegistry();
  initializeGlobalISel(*PR);
  initializeRVGPUDAGToDAGISelPass(*PR);
  initializeGCNDPPCombinePass(*PR);
  initializeSILowerI1CopiesPass(*PR);
  initializeRVGPUGlobalISelDivergenceLoweringPass(*PR);
  initializeSILowerWWMCopiesPass(*PR);
  initializeSILowerSGPRSpillsPass(*PR);
  initializeSIFixSGPRCopiesPass(*PR);
  initializeSIFixVGPRCopiesPass(*PR);
  initializeSIFoldOperandsPass(*PR);
  initializeSIPeepholeSDWAPass(*PR);
  initializeSIShrinkInstructionsPass(*PR);
  initializeSIOptimizeExecMaskingPreRAPass(*PR);
  initializeSIOptimizeVGPRLiveRangePass(*PR);
  initializeSILoadStoreOptimizerPass(*PR);
  initializeRVGPUCtorDtorLoweringLegacyPass(*PR);
  initializeRVGPUAlwaysInlinePass(*PR);
  initializeRVGPUAttributorLegacyPass(*PR);
  initializeRVGPUAnnotateKernelFeaturesPass(*PR);
  initializeRVGPUAnnotateUniformValuesPass(*PR);
  initializeRVGPUArgumentUsageInfoPass(*PR);
  initializeRVGPUAtomicOptimizerPass(*PR);
  initializeRVGPULowerKernelArgumentsPass(*PR);
  initializeRVGPUPromoteKernelArgumentsPass(*PR);
  initializeRVGPULowerKernelAttributesPass(*PR);
  initializeRVGPUOpenCLEnqueuedBlockLoweringPass(*PR);
  initializeRVGPUPostLegalizerCombinerPass(*PR);
  initializeRVGPUPreLegalizerCombinerPass(*PR);
  initializeRVGPURegBankCombinerPass(*PR);
  initializeRVGPURegBankSelectPass(*PR);
  initializeRVGPUPromoteAllocaPass(*PR);
  initializeRVGPUPromoteAllocaToVectorPass(*PR);
  initializeRVGPUCodeGenPreparePass(*PR);
  initializeRVGPULateCodeGenPreparePass(*PR);
  initializeRVGPURemoveIncompatibleFunctionsPass(*PR);
  initializeRVGPULowerModuleLDSLegacyPass(*PR);
  initializeRVGPURewriteOutArgumentsPass(*PR);
  initializeRVGPURewriteUndefForPHILegacyPass(*PR);
  initializeRVGPUUnifyMetadataPass(*PR);
  initializeSIAnnotateControlFlowPass(*PR);
  initializeRVGPUInsertSingleUseVDSTPass(*PR);
  initializeRVGPUInsertDelayAluPass(*PR);
  initializeSIInsertHardClausesPass(*PR);
  initializeSIInsertWaitcntsPass(*PR);
  initializeSIModeRegisterPass(*PR);
  initializeSIWholeQuadModePass(*PR);
  initializeSILowerControlFlowPass(*PR);
  initializeSIPreEmitPeepholePass(*PR);
  initializeSILateBranchLoweringPass(*PR);
  initializeSIMemoryLegalizerPass(*PR);
  initializeSIOptimizeExecMaskingPass(*PR);
  initializeSIPreAllocateWWMRegsPass(*PR);
  initializeSIFormMemoryClausesPass(*PR);
  initializeSIPostRABundlerPass(*PR);
  initializeGCNCreateVOPDPass(*PR);
  initializeRVGPUUnifyDivergentExitNodesPass(*PR);
  initializeRVGPUAAWrapperPassPass(*PR);
  initializeRVGPUExternalAAWrapperPass(*PR);
  initializeRVGPUImageIntrinsicOptimizerPass(*PR);
  initializeRVGPUPrintfRuntimeBindingPass(*PR);
  initializeRVGPUResourceUsageAnalysisPass(*PR);
  initializeGCNNSAReassignPass(*PR);
  initializeGCNPreRAOptimizationsPass(*PR);
  initializeGCNPreRALongBranchRegPass(*PR);
  initializeGCNRewritePartialRegUsesPass(*PR);
  initializeGCNRegPressurePrinterPass(*PR);
}

static std::unique_ptr<TargetLoweringObjectFile> createTLOF(const Triple &TT) {
  return std::make_unique<RVGPUTargetObjectFile>();
}

static ScheduleDAGInstrs *createSIMachineScheduler(MachineSchedContext *C) {
  return new SIScheduleDAGMI(C);
}

static ScheduleDAGInstrs *
createGCNMaxOccupancyMachineScheduler(MachineSchedContext *C) {
  const GCNSubtarget &ST = C->MF->getSubtarget<GCNSubtarget>();
  ScheduleDAGMILive *DAG =
    new GCNScheduleDAGMILive(C, std::make_unique<GCNMaxOccupancySchedStrategy>(C));
  DAG->addMutation(createLoadClusterDAGMutation(DAG->TII, DAG->TRI));
  if (ST.shouldClusterStores())
    DAG->addMutation(createStoreClusterDAGMutation(DAG->TII, DAG->TRI));
  DAG->addMutation(createIGroupLPDAGMutation(/*IsPostRA=*/false));
  DAG->addMutation(createRVGPUMacroFusionDAGMutation());
  DAG->addMutation(createRVGPUExportClusteringDAGMutation());
  return DAG;
}

static ScheduleDAGInstrs *
createGCNMaxILPMachineScheduler(MachineSchedContext *C) {
  ScheduleDAGMILive *DAG =
      new GCNScheduleDAGMILive(C, std::make_unique<GCNMaxILPSchedStrategy>(C));
  DAG->addMutation(createIGroupLPDAGMutation(/*IsPostRA=*/false));
  return DAG;
}

static ScheduleDAGInstrs *
createIterativeGCNMaxOccupancyMachineScheduler(MachineSchedContext *C) {
  const GCNSubtarget &ST = C->MF->getSubtarget<GCNSubtarget>();
  auto DAG = new GCNIterativeScheduler(C,
    GCNIterativeScheduler::SCHEDULE_LEGACYMAXOCCUPANCY);
  DAG->addMutation(createLoadClusterDAGMutation(DAG->TII, DAG->TRI));
  if (ST.shouldClusterStores())
    DAG->addMutation(createStoreClusterDAGMutation(DAG->TII, DAG->TRI));
  return DAG;
}

static ScheduleDAGInstrs *createMinRegScheduler(MachineSchedContext *C) {
  return new GCNIterativeScheduler(C,
    GCNIterativeScheduler::SCHEDULE_MINREGFORCED);
}

static ScheduleDAGInstrs *
createIterativeILPMachineScheduler(MachineSchedContext *C) {
  const GCNSubtarget &ST = C->MF->getSubtarget<GCNSubtarget>();
  auto DAG = new GCNIterativeScheduler(C,
    GCNIterativeScheduler::SCHEDULE_ILP);
  DAG->addMutation(createLoadClusterDAGMutation(DAG->TII, DAG->TRI));
  if (ST.shouldClusterStores())
    DAG->addMutation(createStoreClusterDAGMutation(DAG->TII, DAG->TRI));
  DAG->addMutation(createRVGPUMacroFusionDAGMutation());
  return DAG;
}

static MachineSchedRegistry
SISchedRegistry("si", "Run SI's custom scheduler",
                createSIMachineScheduler);

static MachineSchedRegistry
GCNMaxOccupancySchedRegistry("gcn-max-occupancy",
                             "Run GCN scheduler to maximize occupancy",
                             createGCNMaxOccupancyMachineScheduler);

static MachineSchedRegistry
    GCNMaxILPSchedRegistry("gcn-max-ilp", "Run GCN scheduler to maximize ilp",
                           createGCNMaxILPMachineScheduler);

static MachineSchedRegistry IterativeGCNMaxOccupancySchedRegistry(
    "gcn-iterative-max-occupancy-experimental",
    "Run GCN scheduler to maximize occupancy (experimental)",
    createIterativeGCNMaxOccupancyMachineScheduler);

static MachineSchedRegistry GCNMinRegSchedRegistry(
    "gcn-iterative-minreg",
    "Run GCN iterative scheduler for minimal register usage (experimental)",
    createMinRegScheduler);

static MachineSchedRegistry GCNILPSchedRegistry(
    "gcn-iterative-ilp",
    "Run GCN iterative scheduler for ILP scheduling (experimental)",
    createIterativeILPMachineScheduler);

static StringRef computeDataLayout(const Triple &TT) {
  // 32-bit private, local, and region pointers. 64-bit global, constant and
  // flat. 160-bit non-integral fat buffer pointers that include a 128-bit
  // buffer descriptor and a 32-bit offset, which are indexed by 32-bit values
  // (address space 7), and 128-bit non-integral buffer resourcees (address
  // space 8) which cannot be non-trivilally accessed by LLVM memory operations
  // like getelementptr.
  return "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
         "-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-"
         "v32:32-v48:64-v96:"
         "128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-"
         "G1-ni:7:8:9";
}

LLVM_READNONE
static StringRef getGPUOrDefault(const Triple &TT, StringRef GPU) {
  if (!GPU.empty())
    return GPU;

  // Need to default to a target with flat support for HSA.
  return TT.getOS() == Triple::RVHSA ? "generic-rvhsa" : "generic";
}

static Reloc::Model getEffectiveRelocModel(std::optional<Reloc::Model> RM) {
  // The RVGPU toolchain only supports generating shared objects, so we
  // must always use PIC.
  return Reloc::PIC_;
}

RVGPUTargetMachine::RVGPUTargetMachine(const Target &T, const Triple &TT,
                                         StringRef CPU, StringRef FS,
                                         TargetOptions Options,
                                         std::optional<Reloc::Model> RM,
                                         std::optional<CodeModel::Model> CM,
                                         CodeGenOptLevel OptLevel)
    : LLVMTargetMachine(T, computeDataLayout(TT), TT, getGPUOrDefault(TT, CPU),
                        FS, Options, getEffectiveRelocModel(RM),
                        getEffectiveCodeModel(CM, CodeModel::Small), OptLevel),
      TLOF(createTLOF(getTargetTriple())) {
  initAsmInfo();
  if (TT.getArch() == Triple::rvgpu) {
    if (getMCSubtargetInfo()->checkFeatures("+wavefrontsize64"))
      MRI.reset(llvm::createGCNMCRegisterInfo(RVGPUDwarfFlavour::Wave64));
    else if (getMCSubtargetInfo()->checkFeatures("+wavefrontsize32"))
      MRI.reset(llvm::createGCNMCRegisterInfo(RVGPUDwarfFlavour::Wave32));
  }
}

bool RVGPUTargetMachine::EnableLateStructurizeCFG = false;
bool RVGPUTargetMachine::EnableFunctionCalls = false;
bool RVGPUTargetMachine::EnableLowerModuleLDS = true;

RVGPUTargetMachine::~RVGPUTargetMachine() = default;

StringRef RVGPUTargetMachine::getGPUName(const Function &F) const {
  Attribute GPUAttr = F.getFnAttribute("target-cpu");
  return GPUAttr.isValid() ? GPUAttr.getValueAsString() : getTargetCPU();
}

StringRef RVGPUTargetMachine::getFeatureString(const Function &F) const {
  Attribute FSAttr = F.getFnAttribute("target-features");

  return FSAttr.isValid() ? FSAttr.getValueAsString()
                          : getTargetFeatureString();
}

/// Predicate for Internalize pass.
static bool mustPreserveGV(const GlobalValue &GV) {
  if (const Function *F = dyn_cast<Function>(&GV))
    return F->isDeclaration() || F->getName().starts_with("__asan_") ||
           F->getName().starts_with("__sanitizer_") ||
           RVGPU::isEntryFunctionCC(F->getCallingConv());

  GV.removeDeadConstantUsers();
  return !GV.use_empty();
}

void RVGPUTargetMachine::registerDefaultAliasAnalyses(AAManager &AAM) {
  AAM.registerFunctionAnalysis<RVGPUAA>();
}

void RVGPUTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
  PB.registerPipelineParsingCallback(
      [this](StringRef PassName, ModulePassManager &PM,
             ArrayRef<PassBuilder::PipelineElement>) {
        if (PassName == "rvgpu-attributor") {
          PM.addPass(RVGPUAttributorPass(*this));
          return true;
        }
        if (PassName == "rvgpu-unify-metadata") {
          PM.addPass(RVGPUUnifyMetadataPass());
          return true;
        }
        if (PassName == "rvgpu-printf-runtime-binding") {
          PM.addPass(RVGPUPrintfRuntimeBindingPass());
          return true;
        }
        if (PassName == "rvgpu-always-inline") {
          PM.addPass(RVGPUAlwaysInlinePass());
          return true;
        }
        if (PassName == "rvgpu-lower-module-lds") {
          PM.addPass(RVGPULowerModuleLDSPass(*this));
          return true;
        }
        if (PassName == "rvgpu-lower-ctor-dtor") {
          PM.addPass(RVGPUCtorDtorLoweringPass());
          return true;
        }
        return false;
      });
  PB.registerPipelineParsingCallback(
      [this](StringRef PassName, FunctionPassManager &PM,
             ArrayRef<PassBuilder::PipelineElement>) {
        if (PassName == "rvgpu-simplifylib") {
          PM.addPass(RVGPUSimplifyLibCallsPass());
          return true;
        }
        if (PassName == "rvgpu-image-intrinsic-opt") {
          PM.addPass(RVGPUImageIntrinsicOptimizerPass(*this));
          return true;
        }
        if (PassName == "rvgpu-usenative") {
          PM.addPass(RVGPUUseNativeCallsPass());
          return true;
        }
        if (PassName == "rvgpu-promote-alloca") {
          PM.addPass(RVGPUPromoteAllocaPass(*this));
          return true;
        }
        if (PassName == "rvgpu-promote-alloca-to-vector") {
          PM.addPass(RVGPUPromoteAllocaToVectorPass(*this));
          return true;
        }
        if (PassName == "rvgpu-lower-kernel-attributes") {
          PM.addPass(RVGPULowerKernelAttributesPass());
          return true;
        }
        if (PassName == "rvgpu-promote-kernel-arguments") {
          PM.addPass(RVGPUPromoteKernelArgumentsPass());
          return true;
        }
        if (PassName == "rvgpu-unify-divergent-exit-nodes") {
          PM.addPass(RVGPUUnifyDivergentExitNodesPass());
          return true;
        }
        if (PassName == "rvgpu-atomic-optimizer") {
          PM.addPass(
              RVGPUAtomicOptimizerPass(*this, RVGPUAtomicOptimizerStrategy));
          return true;
        }
        if (PassName == "rvgpu-codegenprepare") {
          PM.addPass(RVGPUCodeGenPreparePass(*this));
          return true;
        }
        if (PassName == "rvgpu-lower-kernel-arguments") {
          PM.addPass(RVGPULowerKernelArgumentsPass(*this));
          return true;
        }
        if (PassName == "rvgpu-rewrite-undef-for-phi") {
          PM.addPass(RVGPURewriteUndefForPHIPass());
          return true;
        }
        return false;
      });

  PB.registerAnalysisRegistrationCallback([](FunctionAnalysisManager &FAM) {
    FAM.registerPass([&] { return RVGPUAA(); });
  });

  PB.registerParseAACallback([](StringRef AAName, AAManager &AAM) {
    if (AAName == "rvgpu-aa") {
      AAM.registerFunctionAnalysis<RVGPUAA>();
      return true;
    }
    return false;
  });

  PB.registerPipelineStartEPCallback(
      [](ModulePassManager &PM, OptimizationLevel Level) {
        FunctionPassManager FPM;
        FPM.addPass(RVGPUUseNativeCallsPass());
        if (EnableLibCallSimplify && Level != OptimizationLevel::O0)
          FPM.addPass(RVGPUSimplifyLibCallsPass());
        PM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
        if (EnableHipStdPar)
          PM.addPass(HipStdParAcceleratorCodeSelectionPass());
      });

  PB.registerPipelineEarlySimplificationEPCallback(
      [](ModulePassManager &PM, OptimizationLevel Level) {
        PM.addPass(RVGPUPrintfRuntimeBindingPass());

        if (Level == OptimizationLevel::O0)
          return;

        PM.addPass(RVGPUUnifyMetadataPass());

        if (InternalizeSymbols) {
          PM.addPass(InternalizePass(mustPreserveGV));
          PM.addPass(GlobalDCEPass());
        }

        if (EarlyInlineAll && !EnableFunctionCalls)
          PM.addPass(RVGPUAlwaysInlinePass());
      });

  PB.registerCGSCCOptimizerLateEPCallback(
      [this](CGSCCPassManager &PM, OptimizationLevel Level) {
        if (Level == OptimizationLevel::O0)
          return;

        FunctionPassManager FPM;

        // Add promote kernel arguments pass to the opt pipeline right before
        // infer address spaces which is needed to do actual address space
        // rewriting.
        if (Level.getSpeedupLevel() > OptimizationLevel::O1.getSpeedupLevel() &&
            EnablePromoteKernelArguments)
          FPM.addPass(RVGPUPromoteKernelArgumentsPass());

        // Add infer address spaces pass to the opt pipeline after inlining
        // but before SROA to increase SROA opportunities.
        FPM.addPass(InferAddressSpacesPass());

        // This should run after inlining to have any chance of doing
        // anything, and before other cleanup optimizations.
        FPM.addPass(RVGPULowerKernelAttributesPass());

        if (Level != OptimizationLevel::O0) {
          // Promote alloca to vector before SROA and loop unroll. If we
          // manage to eliminate allocas before unroll we may choose to unroll
          // less.
          FPM.addPass(RVGPUPromoteAllocaToVectorPass(*this));
        }

        PM.addPass(createCGSCCToFunctionPassAdaptor(std::move(FPM)));
      });
}

int64_t RVGPUTargetMachine::getNullPointerValue(unsigned AddrSpace) {
  return (AddrSpace == RVGPUAS::LOCAL_ADDRESS ||
          AddrSpace == RVGPUAS::PRIVATE_ADDRESS ||
          AddrSpace == RVGPUAS::REGION_ADDRESS)
             ? -1
             : 0;
}

bool RVGPUTargetMachine::isNoopAddrSpaceCast(unsigned SrcAS,
                                              unsigned DestAS) const {
  return RVGPU::isFlatGlobalAddrSpace(SrcAS) &&
         RVGPU::isFlatGlobalAddrSpace(DestAS);
}

unsigned RVGPUTargetMachine::getAssumedAddrSpace(const Value *V) const {
  const auto *LD = dyn_cast<LoadInst>(V);
  if (!LD)
    return RVGPUAS::UNKNOWN_ADDRESS_SPACE;

  // It must be a generic pointer loaded.
  assert(V->getType()->isPointerTy() &&
         V->getType()->getPointerAddressSpace() == RVGPUAS::FLAT_ADDRESS);

  const auto *Ptr = LD->getPointerOperand();
  if (Ptr->getType()->getPointerAddressSpace() != RVGPUAS::CONSTANT_ADDRESS)
    return RVGPUAS::UNKNOWN_ADDRESS_SPACE;
  // For a generic pointer loaded from the constant memory, it could be assumed
  // as a global pointer since the constant memory is only populated on the
  // host side. As implied by the offload programming model, only global
  // pointers could be referenced on the host side.
  return RVGPUAS::GLOBAL_ADDRESS;
}

std::pair<const Value *, unsigned>
RVGPUTargetMachine::getPredicatedAddrSpace(const Value *V) const {
  if (auto *II = dyn_cast<IntrinsicInst>(V)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::rvgpu_is_shared:
      return std::pair(II->getArgOperand(0), RVGPUAS::LOCAL_ADDRESS);
    case Intrinsic::rvgpu_is_private:
      return std::pair(II->getArgOperand(0), RVGPUAS::PRIVATE_ADDRESS);
    default:
      break;
    }
    return std::pair(nullptr, -1);
  }
  // Check the global pointer predication based on
  // (!is_share(p) && !is_private(p)). Note that logic 'and' is commutative and
  // the order of 'is_shared' and 'is_private' is not significant.
  Value *Ptr;
  if (match(
          const_cast<Value *>(V),
          m_c_And(m_Not(m_Intrinsic<Intrinsic::rvgpu_is_shared>(m_Value(Ptr))),
                  m_Not(m_Intrinsic<Intrinsic::rvgpu_is_private>(
                      m_Deferred(Ptr))))))
    return std::pair(Ptr, RVGPUAS::GLOBAL_ADDRESS);

  return std::pair(nullptr, -1);
}

unsigned
RVGPUTargetMachine::getAddressSpaceForPseudoSourceKind(unsigned Kind) const {
  switch (Kind) {
  case PseudoSourceValue::Stack:
  case PseudoSourceValue::FixedStack:
    return RVGPUAS::PRIVATE_ADDRESS;
  case PseudoSourceValue::ConstantPool:
  case PseudoSourceValue::GOT:
  case PseudoSourceValue::JumpTable:
  case PseudoSourceValue::GlobalValueCallEntry:
  case PseudoSourceValue::ExternalSymbolCallEntry:
    return RVGPUAS::CONSTANT_ADDRESS;
  }
  return RVGPUAS::FLAT_ADDRESS;
}

//===----------------------------------------------------------------------===//
// GCN Target Machine (SI+)
//===----------------------------------------------------------------------===//

GCNTargetMachine::GCNTargetMachine(const Target &T, const Triple &TT,
                                   StringRef CPU, StringRef FS,
                                   TargetOptions Options,
                                   std::optional<Reloc::Model> RM,
                                   std::optional<CodeModel::Model> CM,
                                   CodeGenOptLevel OL, bool JIT)
    : RVGPUTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL) {}

const TargetSubtargetInfo *
GCNTargetMachine::getSubtargetImpl(const Function &F) const {
  StringRef GPU = getGPUName(F);
  StringRef FS = getFeatureString(F);

  SmallString<128> SubtargetKey(GPU);
  SubtargetKey.append(FS);

  auto &I = SubtargetMap[SubtargetKey];
  if (!I) {
    // This needs to be done before we create a new subtarget since any
    // creation will depend on the TM and the code generation flags on the
    // function that reside in TargetOptions.
    resetTargetOptions(F);
    I = std::make_unique<GCNSubtarget>(TargetTriple, GPU, FS, *this);
  }

  I->setScalarizeGlobalBehavior(ScalarizeGlobal);

  return I.get();
}

TargetTransformInfo
GCNTargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(GCNTTIImpl(this, F));
}

//===----------------------------------------------------------------------===//
// RVGPU Pass Setup
//===----------------------------------------------------------------------===//

std::unique_ptr<CSEConfigBase> llvm::RVGPUPassConfig::getCSEConfig() const {
  return getStandardCSEConfigForOpt(TM->getOptLevel());
}

namespace {

class GCNPassConfig final : public RVGPUPassConfig {
public:
  GCNPassConfig(LLVMTargetMachine &TM, PassManagerBase &PM)
    : RVGPUPassConfig(TM, PM) {
    // It is necessary to know the register usage of the entire call graph.  We
    // allow calls without EnableRVGPUFunctionCalls if they are marked
    // noinline, so this is always required.
    setRequiresCodeGenSCCOrder(true);
    substitutePass(&PostRASchedulerID, &PostMachineSchedulerID);
  }

  GCNTargetMachine &getGCNTargetMachine() const {
    return getTM<GCNTargetMachine>();
  }

  ScheduleDAGInstrs *
  createMachineScheduler(MachineSchedContext *C) const override;

  ScheduleDAGInstrs *
  createPostMachineScheduler(MachineSchedContext *C) const override {
    ScheduleDAGMI *DAG = new GCNPostScheduleDAGMILive(
        C, std::make_unique<PostGenericScheduler>(C),
        /*RemoveKillFlags=*/true);
    const GCNSubtarget &ST = C->MF->getSubtarget<GCNSubtarget>();
    DAG->addMutation(createLoadClusterDAGMutation(DAG->TII, DAG->TRI));
    if (ST.shouldClusterStores())
      DAG->addMutation(createStoreClusterDAGMutation(DAG->TII, DAG->TRI));
    DAG->addMutation(ST.createFillMFMAShadowMutation(DAG->TII));
    DAG->addMutation(createIGroupLPDAGMutation(/*IsPostRA=*/true));
    if (isPassEnabled(EnableVOPD, CodeGenOptLevel::Less))
      DAG->addMutation(createVOPDPairingMutation());
    return DAG;
  }

  bool addPreISel() override;
  void addMachineSSAOptimization() override;
  bool addILPOpts() override;
  bool addInstSelector() override;
  bool addIRTranslator() override;
  void addPreLegalizeMachineIR() override;
  bool addLegalizeMachineIR() override;
  void addPreRegBankSelect() override;
  bool addRegBankSelect() override;
  void addPreGlobalInstructionSelect() override;
  bool addGlobalInstructionSelect() override;
  void addFastRegAlloc() override;
  void addOptimizedRegAlloc() override;

  FunctionPass *createSGPRAllocPass(bool Optimized);
  FunctionPass *createVGPRAllocPass(bool Optimized);
  FunctionPass *createRegAllocPass(bool Optimized) override;

  bool addRegAssignAndRewriteFast() override;
  bool addRegAssignAndRewriteOptimized() override;

  void addPreRegAlloc() override;
  bool addPreRewrite() override;
  void addPostRegAlloc() override;
  void addPreSched2() override;
  void addPreEmitPass() override;
};

} // end anonymous namespace

RVGPUPassConfig::RVGPUPassConfig(LLVMTargetMachine &TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {
  // Exceptions and StackMaps are not supported, so these passes will never do
  // anything.
  disablePass(&StackMapLivenessID);
  disablePass(&FuncletLayoutID);
  // Garbage collection is not supported.
  disablePass(&GCLoweringID);
  disablePass(&ShadowStackGCLoweringID);
}

void RVGPUPassConfig::addEarlyCSEOrGVNPass() {
  if (getOptLevel() == CodeGenOptLevel::Aggressive)
    addPass(createGVNPass());
  else
    addPass(createEarlyCSEPass());
}

void RVGPUPassConfig::addStraightLineScalarOptimizationPasses() {
  if (isPassEnabled(EnableLoopPrefetch, CodeGenOptLevel::Aggressive))
    addPass(createLoopDataPrefetchPass());
  addPass(createSeparateConstOffsetFromGEPPass());
  // ReassociateGEPs exposes more opportunities for SLSR. See
  // the example in reassociate-geps-and-slsr.ll.
  addPass(createStraightLineStrengthReducePass());
  // SeparateConstOffsetFromGEP and SLSR creates common expressions which GVN or
  // EarlyCSE can reuse.
  addEarlyCSEOrGVNPass();
  // Run NaryReassociate after EarlyCSE/GVN to be more effective.
  addPass(createNaryReassociatePass());
  // NaryReassociate on GEPs creates redundant common expressions, so run
  // EarlyCSE after it.
  addPass(createEarlyCSEPass());
}

void RVGPUPassConfig::addIRPasses() {
  const RVGPUTargetMachine &TM = getRVGPUTargetMachine();

  Triple::ArchType Arch = TM.getTargetTriple().getArch();
  if (RemoveIncompatibleFunctions && Arch == Triple::rvgpu)
    addPass(createRVGPURemoveIncompatibleFunctionsPass(&TM));

  // There is no reason to run these.
  disablePass(&StackMapLivenessID);
  disablePass(&FuncletLayoutID);
  disablePass(&PatchableFunctionID);

  addPass(createRVGPUPrintfRuntimeBinding());
  if (LowerCtorDtor)
    addPass(createRVGPUCtorDtorLoweringLegacyPass());

  if (isPassEnabled(EnableImageIntrinsicOptimizer))
    addPass(createRVGPUImageIntrinsicOptimizerPass(&TM));

  // Function calls are not supported, so make sure we inline everything.
  addPass(createRVGPUAlwaysInlinePass());
  addPass(createAlwaysInlinerLegacyPass());

  // Replace OpenCL enqueued block function pointers with global variables.
  addPass(createRVGPUOpenCLEnqueuedBlockLoweringPass());

  // Runs before PromoteAlloca so the latter can account for function uses
  if (EnableLowerModuleLDS) {
    addPass(createRVGPULowerModuleLDSLegacyPass(&TM));
  }

  // RVGPUAttributor infers lack of llvm.rvgpu.lds.kernel.id calls, so run
  // after their introduction
  if (TM.getOptLevel() > CodeGenOptLevel::None)
    addPass(createRVGPUAttributorLegacyPass());

  if (TM.getOptLevel() > CodeGenOptLevel::None)
    addPass(createInferAddressSpacesPass());

  // Run atomic optimizer before Atomic Expand
  if ((TM.getTargetTriple().getArch() == Triple::rvgpu) &&
      (TM.getOptLevel() >= CodeGenOptLevel::Less) &&
      (RVGPUAtomicOptimizerStrategy != ScanOptions::None)) {
    addPass(createRVGPUAtomicOptimizerPass(RVGPUAtomicOptimizerStrategy));
  }

  addPass(createAtomicExpandPass());

  if (TM.getOptLevel() > CodeGenOptLevel::None) {
    addPass(createRVGPUPromoteAlloca());

    if (isPassEnabled(EnableScalarIRPasses))
      addStraightLineScalarOptimizationPasses();

    if (EnableRVGPUAliasAnalysis) {
      addPass(createRVGPUAAWrapperPass());
      addPass(createExternalAAWrapperPass([](Pass &P, Function &,
                                             AAResults &AAR) {
        if (auto *WrapperPass = P.getAnalysisIfAvailable<RVGPUAAWrapperPass>())
          AAR.addAAResult(WrapperPass->getResult());
        }));
    }

    if (TM.getTargetTriple().getArch() == Triple::rvgpu) {
      // TODO: May want to move later or split into an early and late one.
      addPass(createRVGPUCodeGenPreparePass());
    }

    // Try to hoist loop invariant parts of divisions RVGPUCodeGenPrepare may
    // have expanded.
    if (TM.getOptLevel() > CodeGenOptLevel::Less)
      addPass(createLICMPass());
  }

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
  if (isPassEnabled(EnableScalarIRPasses))
    addEarlyCSEOrGVNPass();
}

void RVGPUPassConfig::addCodeGenPrepare() {
  if (TM->getTargetTriple().getArch() == Triple::rvgpu) {
    // FIXME: This pass adds 2 hacky attributes that can be replaced with an
    // analysis, and should be removed.
    addPass(createRVGPUAnnotateKernelFeaturesPass());
  }

  if (TM->getTargetTriple().getArch() == Triple::rvgpu &&
      EnableLowerKernelArguments)
    addPass(createRVGPULowerKernelArgumentsPass());

  TargetPassConfig::addCodeGenPrepare();

  if (isPassEnabled(EnableLoadStoreVectorizer))
    addPass(createLoadStoreVectorizerPass());

  // LowerSwitch pass may introduce unreachable blocks that can
  // cause unexpected behavior for subsequent passes. Placing it
  // here seems better that these blocks would get cleaned up by
  // UnreachableBlockElim inserted next in the pass flow.
  addPass(createLowerSwitchPass());
}

bool RVGPUPassConfig::addPreISel() {
  if (TM->getOptLevel() > CodeGenOptLevel::None)
    addPass(createFlattenCFGPass());
  return false;
}

bool RVGPUPassConfig::addInstSelector() {
  addPass(createRVGPUISelDag(getRVGPUTargetMachine(), getOptLevel()));
  return false;
}

bool RVGPUPassConfig::addGCPasses() {
  // Do nothing. GC is not supported.
  return false;
}

llvm::ScheduleDAGInstrs *
RVGPUPassConfig::createMachineScheduler(MachineSchedContext *C) const {
  const GCNSubtarget &ST = C->MF->getSubtarget<GCNSubtarget>();
  ScheduleDAGMILive *DAG = createGenericSchedLive(C);
  DAG->addMutation(createLoadClusterDAGMutation(DAG->TII, DAG->TRI));
  if (ST.shouldClusterStores())
    DAG->addMutation(createStoreClusterDAGMutation(DAG->TII, DAG->TRI));
  return DAG;
}

//===----------------------------------------------------------------------===//
// GCN Pass Setup
//===----------------------------------------------------------------------===//

ScheduleDAGInstrs *GCNPassConfig::createMachineScheduler(
  MachineSchedContext *C) const {
  const GCNSubtarget &ST = C->MF->getSubtarget<GCNSubtarget>();
  if (ST.enableSIScheduler())
    return createSIMachineScheduler(C);

  if (EnableMaxIlpSchedStrategy)
    return createGCNMaxILPMachineScheduler(C);

  return createGCNMaxOccupancyMachineScheduler(C);
}

bool GCNPassConfig::addPreISel() {
  RVGPUPassConfig::addPreISel();

  if (TM->getOptLevel() > CodeGenOptLevel::None)
    addPass(createRVGPULateCodeGenPreparePass());

  if (TM->getOptLevel() > CodeGenOptLevel::None)
    addPass(createSinkingPass());

  // Merge divergent exit nodes. StructurizeCFG won't recognize the multi-exit
  // regions formed by them.
  addPass(&RVGPUUnifyDivergentExitNodesID);
  if (!LateCFGStructurize) {
    if (EnableStructurizerWorkarounds) {
      addPass(createFixIrreduciblePass());
      addPass(createUnifyLoopExitsPass());
    }
    addPass(createStructurizeCFGPass(false)); // true -> SkipUniformRegions
  }
  addPass(createRVGPUAnnotateUniformValues());
  if (!LateCFGStructurize) {
    addPass(createSIAnnotateControlFlowPass());
    // TODO: Move this right after structurizeCFG to avoid extra divergence
    // analysis. This depends on stopping SIAnnotateControlFlow from making
    // control flow modifications.
    addPass(createRVGPURewriteUndefForPHILegacyPass());
  }
  addPass(createLCSSAPass());

  if (TM->getOptLevel() > CodeGenOptLevel::Less)
    addPass(&RVGPUPerfHintAnalysisID);

  return false;
}

void GCNPassConfig::addMachineSSAOptimization() {
  TargetPassConfig::addMachineSSAOptimization();

  // We want to fold operands after PeepholeOptimizer has run (or as part of
  // it), because it will eliminate extra copies making it easier to fold the
  // real source operand. We want to eliminate dead instructions after, so that
  // we see fewer uses of the copies. We then need to clean up the dead
  // instructions leftover after the operands are folded as well.
  //
  // XXX - Can we get away without running DeadMachineInstructionElim again?
  addPass(&SIFoldOperandsID);
  if (EnableDPPCombine)
    addPass(&GCNDPPCombineID);
  addPass(&SILoadStoreOptimizerID);
  if (isPassEnabled(EnableSDWAPeephole)) {
    addPass(&SIPeepholeSDWAID);
    addPass(&EarlyMachineLICMID);
    addPass(&MachineCSEID);
    addPass(&SIFoldOperandsID);
  }
  addPass(&DeadMachineInstructionElimID);
  addPass(createSIShrinkInstructionsPass());
}

bool GCNPassConfig::addILPOpts() {
  if (EnableEarlyIfConversion)
    addPass(&EarlyIfConverterID);

  TargetPassConfig::addILPOpts();
  return false;
}

bool GCNPassConfig::addInstSelector() {
  RVGPUPassConfig::addInstSelector();
  addPass(&SIFixSGPRCopiesID);
  addPass(createSILowerI1CopiesPass());
  return false;
}

bool GCNPassConfig::addIRTranslator() {
  addPass(new IRTranslator(getOptLevel()));
  return false;
}

void GCNPassConfig::addPreLegalizeMachineIR() {
  bool IsOptNone = getOptLevel() == CodeGenOptLevel::None;
  addPass(createRVGPUPreLegalizeCombiner(IsOptNone));
  addPass(new Localizer());
}

bool GCNPassConfig::addLegalizeMachineIR() {
  addPass(new Legalizer());
  return false;
}

void GCNPassConfig::addPreRegBankSelect() {
  bool IsOptNone = getOptLevel() == CodeGenOptLevel::None;
  addPass(createRVGPUPostLegalizeCombiner(IsOptNone));
  addPass(createRVGPUGlobalISelDivergenceLoweringPass());
}

bool GCNPassConfig::addRegBankSelect() {
  addPass(new RVGPURegBankSelect());
  return false;
}

void GCNPassConfig::addPreGlobalInstructionSelect() {
  bool IsOptNone = getOptLevel() == CodeGenOptLevel::None;
  addPass(createRVGPURegBankCombiner(IsOptNone));
}

bool GCNPassConfig::addGlobalInstructionSelect() {
  addPass(new InstructionSelect(getOptLevel()));
  return false;
}

void GCNPassConfig::addPreRegAlloc() {
  if (LateCFGStructurize) {
    addPass(createRVGPUMachineCFGStructurizerPass());
  }
}

void GCNPassConfig::addFastRegAlloc() {
  // FIXME: We have to disable the verifier here because of PHIElimination +
  // TwoAddressInstructions disabling it.

  // This must be run immediately after phi elimination and before
  // TwoAddressInstructions, otherwise the processing of the tied operand of
  // SI_ELSE will introduce a copy of the tied operand source after the else.
  insertPass(&PHIEliminationID, &SILowerControlFlowID);

  insertPass(&TwoAddressInstructionPassID, &SIWholeQuadModeID);

  TargetPassConfig::addFastRegAlloc();
}

void GCNPassConfig::addOptimizedRegAlloc() {
  // Allow the scheduler to run before SIWholeQuadMode inserts exec manipulation
  // instructions that cause scheduling barriers.
  insertPass(&MachineSchedulerID, &SIWholeQuadModeID);

  if (OptExecMaskPreRA)
    insertPass(&MachineSchedulerID, &SIOptimizeExecMaskingPreRAID);

  if (EnableRewritePartialRegUses)
    insertPass(&RenameIndependentSubregsID, &GCNRewritePartialRegUsesID);

  if (isPassEnabled(EnablePreRAOptimizations))
    insertPass(&RenameIndependentSubregsID, &GCNPreRAOptimizationsID);

  // This is not an essential optimization and it has a noticeable impact on
  // compilation time, so we only enable it from O2.
  if (TM->getOptLevel() > CodeGenOptLevel::Less)
    insertPass(&MachineSchedulerID, &SIFormMemoryClausesID);

  // FIXME: when an instruction has a Killed operand, and the instruction is
  // inside a bundle, seems only the BUNDLE instruction appears as the Kills of
  // the register in LiveVariables, this would trigger a failure in verifier,
  // we should fix it and enable the verifier.
  if (OptVGPRLiveRange)
    insertPass(&LiveVariablesID, &SIOptimizeVGPRLiveRangeID);
  // This must be run immediately after phi elimination and before
  // TwoAddressInstructions, otherwise the processing of the tied operand of
  // SI_ELSE will introduce a copy of the tied operand source after the else.
  insertPass(&PHIEliminationID, &SILowerControlFlowID);

  if (EnableDCEInRA)
    insertPass(&DetectDeadLanesID, &DeadMachineInstructionElimID);

  TargetPassConfig::addOptimizedRegAlloc();
}

bool GCNPassConfig::addPreRewrite() {
  addPass(&SILowerWWMCopiesID);
  if (EnableRegReassign)
    addPass(&GCNNSAReassignID);
  return true;
}

FunctionPass *GCNPassConfig::createSGPRAllocPass(bool Optimized) {
  // Initialize the global default.
  llvm::call_once(InitializeDefaultSGPRRegisterAllocatorFlag,
                  initializeDefaultSGPRRegisterAllocatorOnce);

  RegisterRegAlloc::FunctionPassCtor Ctor = SGPRRegisterRegAlloc::getDefault();
  if (Ctor != useDefaultRegisterAllocator)
    return Ctor();

  if (Optimized)
    return createGreedyRegisterAllocator(onlyAllocateSGPRs);

  return createFastRegisterAllocator(onlyAllocateSGPRs, false);
}

FunctionPass *GCNPassConfig::createVGPRAllocPass(bool Optimized) {
  // Initialize the global default.
  llvm::call_once(InitializeDefaultVGPRRegisterAllocatorFlag,
                  initializeDefaultVGPRRegisterAllocatorOnce);

  RegisterRegAlloc::FunctionPassCtor Ctor = VGPRRegisterRegAlloc::getDefault();
  if (Ctor != useDefaultRegisterAllocator)
    return Ctor();

  if (Optimized)
    return createGreedyVGPRRegisterAllocator();

  return createFastVGPRRegisterAllocator();
}

FunctionPass *GCNPassConfig::createRegAllocPass(bool Optimized) {
  llvm_unreachable("should not be used");
}

static const char RegAllocOptNotSupportedMessage[] =
  "-regalloc not supported with rvgpu. Use -sgpr-regalloc and -vgpr-regalloc";

bool GCNPassConfig::addRegAssignAndRewriteFast() {
  if (!usingDefaultRegAlloc())
    report_fatal_error(RegAllocOptNotSupportedMessage);

  addPass(&GCNPreRALongBranchRegID);

  addPass(createSGPRAllocPass(false));

  // Equivalent of PEI for SGPRs.
  addPass(&SILowerSGPRSpillsID);
  addPass(&SIPreAllocateWWMRegsID);

  addPass(createVGPRAllocPass(false));

  addPass(&SILowerWWMCopiesID);
  return true;
}

bool GCNPassConfig::addRegAssignAndRewriteOptimized() {
  if (!usingDefaultRegAlloc())
    report_fatal_error(RegAllocOptNotSupportedMessage);

  addPass(&GCNPreRALongBranchRegID);

  addPass(createSGPRAllocPass(true));

  // Commit allocated register changes. This is mostly necessary because too
  // many things rely on the use lists of the physical registers, such as the
  // verifier. This is only necessary with allocators which use LiveIntervals,
  // since FastRegAlloc does the replacements itself.
  addPass(createVirtRegRewriter(false));

  // Equivalent of PEI for SGPRs.
  addPass(&SILowerSGPRSpillsID);
  addPass(&SIPreAllocateWWMRegsID);

  addPass(createVGPRAllocPass(true));

  addPreRewrite();
  addPass(&VirtRegRewriterID);

  return true;
}

void GCNPassConfig::addPostRegAlloc() {
  addPass(&SIFixVGPRCopiesID);
  if (getOptLevel() > CodeGenOptLevel::None)
    addPass(&SIOptimizeExecMaskingID);
  TargetPassConfig::addPostRegAlloc();
}

void GCNPassConfig::addPreSched2() {
  if (TM->getOptLevel() > CodeGenOptLevel::None)
    addPass(createSIShrinkInstructionsPass());
  addPass(&SIPostRABundlerID);
}

void GCNPassConfig::addPreEmitPass() {
  if (isPassEnabled(EnableVOPD, CodeGenOptLevel::Less))
    addPass(&GCNCreateVOPDID);
  addPass(createSIMemoryLegalizerPass());
  addPass(createSIInsertWaitcntsPass());

  addPass(createSIModeRegisterPass());

  if (getOptLevel() > CodeGenOptLevel::None)
    addPass(&SIInsertHardClausesID);

  addPass(&SILateBranchLoweringPassID);
  if (isPassEnabled(EnableSetWavePriority, CodeGenOptLevel::Less))
    addPass(createRVGPUSetWavePriorityPass());
  if (getOptLevel() > CodeGenOptLevel::None)
    addPass(&SIPreEmitPeepholeID);
  // The hazard recognizer that runs as part of the post-ra scheduler does not
  // guarantee to be able handle all hazards correctly. This is because if there
  // are multiple scheduling regions in a basic block, the regions are scheduled
  // bottom up, so when we begin to schedule a region we don't know what
  // instructions were emitted directly before it.
  //
  // Here we add a stand-alone hazard recognizer pass which can handle all
  // cases.
  addPass(&PostRAHazardRecognizerID);

  if (isPassEnabled(EnableInsertSingleUseVDST, CodeGenOptLevel::Less))
    addPass(&RVGPUInsertSingleUseVDSTID);

  if (isPassEnabled(EnableInsertDelayAlu, CodeGenOptLevel::Less))
    addPass(&RVGPUInsertDelayAluID);

  addPass(&BranchRelaxationPassID);
}

TargetPassConfig *GCNTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new GCNPassConfig(*this, PM);
}

void GCNTargetMachine::registerMachineRegisterInfoCallback(
    MachineFunction &MF) const {
  RVMachineFunctionInfo *MFI = MF.getInfo<RVMachineFunctionInfo>();
  MF.getRegInfo().addDelegate(MFI);
}

MachineFunctionInfo *GCNTargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return RVMachineFunctionInfo::create<RVMachineFunctionInfo>(
      Allocator, F, static_cast<const GCNSubtarget *>(STI));
}

yaml::MachineFunctionInfo *GCNTargetMachine::createDefaultFuncInfoYAML() const {
  return new yaml::RVMachineFunctionInfo();
}

yaml::MachineFunctionInfo *
GCNTargetMachine::convertFuncInfoToYAML(const MachineFunction &MF) const {
  const RVMachineFunctionInfo *MFI = MF.getInfo<RVMachineFunctionInfo>();
  return new yaml::RVMachineFunctionInfo(
      *MFI, *MF.getSubtarget<GCNSubtarget>().getRegisterInfo(), MF);
}

bool GCNTargetMachine::parseMachineFunctionInfo(
    const yaml::MachineFunctionInfo &MFI_, PerFunctionMIParsingState &PFS,
    SMDiagnostic &Error, SMRange &SourceRange) const {
  const yaml::RVMachineFunctionInfo &YamlMFI =
      static_cast<const yaml::RVMachineFunctionInfo &>(MFI_);
  MachineFunction &MF = PFS.MF;
  RVMachineFunctionInfo *MFI = MF.getInfo<RVMachineFunctionInfo>();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();

  if (MFI->initializeBaseYamlFields(YamlMFI, MF, PFS, Error, SourceRange))
    return true;

  if (MFI->Occupancy == 0) {
    // Fixup the subtarget dependent default value.
    MFI->Occupancy = ST.computeOccupancy(MF.getFunction(), MFI->getLDSSize());
  }

  auto parseRegister = [&](const yaml::StringValue &RegName, Register &RegVal) {
    Register TempReg;
    if (parseNamedRegisterReference(PFS, TempReg, RegName.Value, Error)) {
      SourceRange = RegName.SourceRange;
      return true;
    }
    RegVal = TempReg;

    return false;
  };

  auto parseOptionalRegister = [&](const yaml::StringValue &RegName,
                                   Register &RegVal) {
    return !RegName.Value.empty() && parseRegister(RegName, RegVal);
  };

  if (parseOptionalRegister(YamlMFI.VGPRForAGPRCopy, MFI->VGPRForAGPRCopy))
    return true;

  if (parseOptionalRegister(YamlMFI.SGPRForEXECCopy, MFI->SGPRForEXECCopy))
    return true;

  if (parseOptionalRegister(YamlMFI.LongBranchReservedReg,
                            MFI->LongBranchReservedReg))
    return true;

  auto diagnoseRegisterClass = [&](const yaml::StringValue &RegName) {
    // Create a diagnostic for a the register string literal.
    const MemoryBuffer &Buffer =
        *PFS.SM->getMemoryBuffer(PFS.SM->getMainFileID());
    Error = SMDiagnostic(*PFS.SM, SMLoc(), Buffer.getBufferIdentifier(), 1,
                         RegName.Value.size(), SourceMgr::DK_Error,
                         "incorrect register class for field", RegName.Value,
                         std::nullopt, std::nullopt);
    SourceRange = RegName.SourceRange;
    return true;
  };

  if (parseRegister(YamlMFI.ScratchRSrcReg, MFI->ScratchRSrcReg) ||
      parseRegister(YamlMFI.FrameOffsetReg, MFI->FrameOffsetReg) ||
      parseRegister(YamlMFI.StackPtrOffsetReg, MFI->StackPtrOffsetReg))
    return true;

  if (MFI->ScratchRSrcReg != RVGPU::PRIVATE_RSRC_REG &&
      !RVGPU::SGPR_128RegClass.contains(MFI->ScratchRSrcReg)) {
    return diagnoseRegisterClass(YamlMFI.ScratchRSrcReg);
  }

  if (MFI->FrameOffsetReg != RVGPU::FP_REG &&
      !RVGPU::SGPR_32RegClass.contains(MFI->FrameOffsetReg)) {
    return diagnoseRegisterClass(YamlMFI.FrameOffsetReg);
  }

  if (MFI->StackPtrOffsetReg != RVGPU::SP_REG &&
      !RVGPU::SGPR_32RegClass.contains(MFI->StackPtrOffsetReg)) {
    return diagnoseRegisterClass(YamlMFI.StackPtrOffsetReg);
  }

  for (const auto &YamlReg : YamlMFI.WWMReservedRegs) {
    Register ParsedReg;
    if (parseRegister(YamlReg, ParsedReg))
      return true;

    MFI->reserveWWMRegister(ParsedReg);
  }

  auto parseAndCheckArgument = [&](const std::optional<yaml::SIArgument> &A,
                                   const TargetRegisterClass &RC,
                                   ArgDescriptor &Arg, unsigned UserSGPRs,
                                   unsigned SystemSGPRs) {
    // Skip parsing if it's not present.
    if (!A)
      return false;

    if (A->IsRegister) {
      Register Reg;
      if (parseNamedRegisterReference(PFS, Reg, A->RegisterName.Value, Error)) {
        SourceRange = A->RegisterName.SourceRange;
        return true;
      }
      if (!RC.contains(Reg))
        return diagnoseRegisterClass(A->RegisterName);
      Arg = ArgDescriptor::createRegister(Reg);
    } else
      Arg = ArgDescriptor::createStack(A->StackOffset);
    // Check and apply the optional mask.
    if (A->Mask)
      Arg = ArgDescriptor::createArg(Arg, *A->Mask);

    MFI->NumUserSGPRs += UserSGPRs;
    MFI->NumSystemSGPRs += SystemSGPRs;
    return false;
  };

  if (YamlMFI.ArgInfo &&
      (parseAndCheckArgument(YamlMFI.ArgInfo->PrivateSegmentBuffer,
                             RVGPU::SGPR_128RegClass,
                             MFI->ArgInfo.PrivateSegmentBuffer, 4, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->DispatchPtr,
                             RVGPU::SReg_64RegClass, MFI->ArgInfo.DispatchPtr,
                             2, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->QueuePtr, RVGPU::SReg_64RegClass,
                             MFI->ArgInfo.QueuePtr, 2, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->KernargSegmentPtr,
                             RVGPU::SReg_64RegClass,
                             MFI->ArgInfo.KernargSegmentPtr, 2, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->DispatchID,
                             RVGPU::SReg_64RegClass, MFI->ArgInfo.DispatchID,
                             2, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->FlatScratchInit,
                             RVGPU::SReg_64RegClass,
                             MFI->ArgInfo.FlatScratchInit, 2, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->PrivateSegmentSize,
                             RVGPU::SGPR_32RegClass,
                             MFI->ArgInfo.PrivateSegmentSize, 0, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->LDSKernelId,
                             RVGPU::SGPR_32RegClass,
                             MFI->ArgInfo.LDSKernelId, 0, 1) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->WorkGroupIDX,
                             RVGPU::SGPR_32RegClass, MFI->ArgInfo.WorkGroupIDX,
                             0, 1) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->WorkGroupIDY,
                             RVGPU::SGPR_32RegClass, MFI->ArgInfo.WorkGroupIDY,
                             0, 1) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->WorkGroupIDZ,
                             RVGPU::SGPR_32RegClass, MFI->ArgInfo.WorkGroupIDZ,
                             0, 1) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->WorkGroupInfo,
                             RVGPU::SGPR_32RegClass,
                             MFI->ArgInfo.WorkGroupInfo, 0, 1) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->PrivateSegmentWaveByteOffset,
                             RVGPU::SGPR_32RegClass,
                             MFI->ArgInfo.PrivateSegmentWaveByteOffset, 0, 1) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->ImplicitArgPtr,
                             RVGPU::SReg_64RegClass,
                             MFI->ArgInfo.ImplicitArgPtr, 0, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->ImplicitBufferPtr,
                             RVGPU::SReg_64RegClass,
                             MFI->ArgInfo.ImplicitBufferPtr, 2, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->WorkItemIDX,
                             RVGPU::VGPR_32RegClass,
                             MFI->ArgInfo.WorkItemIDX, 0, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->WorkItemIDY,
                             RVGPU::VGPR_32RegClass,
                             MFI->ArgInfo.WorkItemIDY, 0, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->WorkItemIDZ,
                             RVGPU::VGPR_32RegClass,
                             MFI->ArgInfo.WorkItemIDZ, 0, 0)))
    return true;

  if (ST.hasIEEEMode())
    MFI->Mode.IEEE = YamlMFI.Mode.IEEE;
  if (ST.hasDX10ClampMode())
    MFI->Mode.DX10Clamp = YamlMFI.Mode.DX10Clamp;

  // FIXME: Move proper support for denormal-fp-math into base MachineFunction
  MFI->Mode.FP32Denormals.Input = YamlMFI.Mode.FP32InputDenormals
                                      ? DenormalMode::IEEE
                                      : DenormalMode::PreserveSign;
  MFI->Mode.FP32Denormals.Output = YamlMFI.Mode.FP32OutputDenormals
                                       ? DenormalMode::IEEE
                                       : DenormalMode::PreserveSign;

  MFI->Mode.FP64FP16Denormals.Input = YamlMFI.Mode.FP64FP16InputDenormals
                                          ? DenormalMode::IEEE
                                          : DenormalMode::PreserveSign;
  MFI->Mode.FP64FP16Denormals.Output = YamlMFI.Mode.FP64FP16OutputDenormals
                                           ? DenormalMode::IEEE
                                           : DenormalMode::PreserveSign;

  return false;
}
