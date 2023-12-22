add_llvm_component_group(RVGPU)

set(LLVM_TARGET_DEFINITIONS RVGPU.td)

tablegen(LLVM RVGPUGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM RVGPUGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM RVGPUGenCallingConv.inc -gen-callingconv)
tablegen(LLVM RVGPUGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM RVGPUGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM RVGPUGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM RVGPUGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM RVGPUGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM RVGPUGenRegisterBank.inc -gen-register-bank)
tablegen(LLVM RVGPUGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM RVGPUGenSearchableTables.inc -gen-searchable-tables)
tablegen(LLVM RVGPUGenSubtargetInfo.inc -gen-subtarget)

set(LLVM_TARGET_DEFINITIONS RVGPUGISel.td)
tablegen(LLVM RVGPUGenGlobalISel.inc -gen-global-isel)
tablegen(LLVM RVGPUGenPreLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="RVGPUPreLegalizerCombiner")
tablegen(LLVM RVGPUGenPostLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="RVGPUPostLegalizerCombiner")
tablegen(LLVM RVGPUGenRegBankGICombiner.inc -gen-global-isel-combiner
              -combiners="RVGPURegBankCombiner")

add_public_tablegen_target(RVGPUCommonTableGen)

set(LLVM_TARGET_DEFINITIONS RVGPUInstCombineTables.td)
tablegen(LLVM RVGPUInstCombineTables.inc -gen-searchable-tables)
add_public_tablegen_target(RVGPUInstCombineTableGen)

add_llvm_target(RVGPUCodeGen
  RVGPUAliasAnalysis.cpp
  RVGPUAlwaysInlinePass.cpp
  RVGPUAnnotateKernelFeatures.cpp
  RVGPUAnnotateUniformValues.cpp
  RVGPUArgumentUsageInfo.cpp
  RVGPUAsmPrinter.cpp
  RVGPUAtomicOptimizer.cpp
  RVGPUAttributor.cpp
  RVGPUCallLowering.cpp
  RVGPUCodeGenPrepare.cpp
  RVGPUCombinerHelper.cpp
  RVGPUCtorDtorLowering.cpp
  RVGPUExportClustering.cpp
  RVGPUFrameLowering.cpp
  RVGPUGlobalISelDivergenceLowering.cpp
  RVGPUGlobalISelUtils.cpp
  RVGPUHSAMetadataStreamer.cpp
  RVGPUInsertDelayAlu.cpp
  RVGPUInstCombineIntrinsic.cpp
  RVGPUInstrInfo.cpp
  RVGPUInstructionSelector.cpp
  RVGPUISelDAGToDAG.cpp
  RVGPUISelLowering.cpp
  RVGPULateCodeGenPrepare.cpp
  RVGPULegalizerInfo.cpp
  RVGPULibCalls.cpp
  RVGPUImageIntrinsicOptimizer.cpp
  RVGPULibFunc.cpp
  RVGPULowerKernelArguments.cpp
  RVGPULowerKernelAttributes.cpp
  RVGPULowerModuleLDSPass.cpp
  RVGPUMachineCFGStructurizer.cpp
  RVGPUMachineFunction.cpp
  RVGPUMachineModuleInfo.cpp
  RVGPUMacroFusion.cpp
  RVGPUMCInstLower.cpp
  RVGPUIGroupLP.cpp
  RVGPUInsertSingleUseVDST.cpp
  RVGPUMIRFormatter.cpp
  RVGPUOpenCLEnqueuedBlockLowering.cpp
  RVGPUPerfHintAnalysis.cpp
  RVGPUPostLegalizerCombiner.cpp
  RVGPUPreLegalizerCombiner.cpp
  RVGPUPrintfRuntimeBinding.cpp
  RVGPUPromoteAlloca.cpp
  RVGPUPromoteKernelArguments.cpp
  RVGPURegBankCombiner.cpp
  RVGPURegBankSelect.cpp
  RVGPURegisterBankInfo.cpp
  RVGPURemoveIncompatibleFunctions.cpp
  RVGPUResourceUsageAnalysis.cpp
  RVGPURewriteOutArguments.cpp
  RVGPURewriteUndefForPHI.cpp
  RVGPUSetWavePriority.cpp
  RVGPUSubtarget.cpp
  RVGPUTargetMachine.cpp
  RVGPUTargetObjectFile.cpp
  RVGPUTargetTransformInfo.cpp
  RVGPUUnifyDivergentExitNodes.cpp
  RVGPUUnifyMetadata.cpp
  GCNCreateVOPD.cpp
  GCNDPPCombine.cpp
  GCNHazardRecognizer.cpp
  GCNILPSched.cpp
  GCNIterativeScheduler.cpp
  GCNMinRegStrategy.cpp
  GCNNSAReassign.cpp
  GCNPreRAOptimizations.cpp
  GCNPreRALongBranchReg.cpp
  GCNRegPressure.cpp
  GCNRewritePartialRegUses.cpp
  GCNSchedStrategy.cpp
  GCNVOPDUtils.cpp
  SIAnnotateControlFlow.cpp
  SIFixSGPRCopies.cpp
  SIFixVGPRCopies.cpp
  SIFoldOperands.cpp
  SIFormMemoryClauses.cpp
  RVFrameLowering.cpp
  SIInsertHardClauses.cpp
  SIInsertWaitcnts.cpp
  RVInstrInfo.cpp
  RVISelLowering.cpp
  SILateBranchLowering.cpp
  SILoadStoreOptimizer.cpp
  SILowerControlFlow.cpp
  SILowerI1Copies.cpp
  SILowerWWMCopies.cpp
  SILowerSGPRSpills.cpp
  RVMachineFunctionInfo.cpp
  SIMachineScheduler.cpp
  SIMemoryLegalizer.cpp
  SIModeRegister.cpp
  SIModeRegisterDefaults.cpp
  SIOptimizeExecMasking.cpp
  SIOptimizeExecMaskingPreRA.cpp
  SIOptimizeVGPRLiveRange.cpp
  SIPeepholeSDWA.cpp
  SIPostRABundler.cpp
  SIPreAllocateWWMRegs.cpp
  SIPreEmitPeephole.cpp
  RVProgramInfo.cpp
  RVRegisterInfo.cpp
  SIShrinkInstructions.cpp
  SIWholeQuadMode.cpp

  LINK_COMPONENTS
  RVGPUDesc
  RVGPUInfo
  RVGPUUtils
  Analysis
  AsmPrinter
  BinaryFormat
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  HipStdPar
  IPO
  MC
  MIRParser
  Passes
  Scalar
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  Vectorize

  ADD_TO_COMPONENT
  RVGPU
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCA)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(Utils)
