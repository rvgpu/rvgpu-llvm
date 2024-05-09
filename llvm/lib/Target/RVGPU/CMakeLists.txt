add_llvm_component_group(RVGPU)

set(LLVM_TARGET_DEFINITIONS RVGPU.td)

tablegen(LLVM RVGPUGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM RVGPUGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM RVGPUGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM RVGPUGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM RVGPUGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM RVGPUGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM RVGPUGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM RVGPUGenDisassemblerTables.inc -gen-disassembler)

add_public_tablegen_target(RVGPUCommonTableGen)

set(RVGPUCodeGen_sources
  RVGPUAliasAnalysis.cpp
  RVGPUAllocaHoisting.cpp
  RVGPUAtomicLower.cpp
  RVGPUAsmPrinter.cpp
  RVGPUAssignValidGlobalNames.cpp
  RVGPUFrameLowering.cpp
  RVGPUGenericToNVVM.cpp
  RVGPUISelDAGToDAG.cpp
  RVGPUISelLowering.cpp
  RVGPUImageOptimizer.cpp
  RVGPUInstrInfo.cpp
  RVGPULowerAggrCopies.cpp
  RVGPULowerArgs.cpp
  RVGPULowerAlloca.cpp
  RVGPULowerUnreachable.cpp
  RVGPUPeephole.cpp
  RVGPUMCExpr.cpp
  RVGPUPrologEpilogPass.cpp
  RVGPURegisterInfo.cpp
  RVGPUReplaceImageHandles.cpp
  RVGPUSubtarget.cpp
  RVGPUTargetMachine.cpp
  RVGPUTargetTransformInfo.cpp
  RVGPUUtilities.cpp
  NVVMIntrRange.cpp
  NVVMReflect.cpp
  RVGPUProxyRegErasure.cpp
  RVGPUCtorDtorLowering.cpp
#  RVGPUHSAMetadataStreamer.cpp
  )

add_llvm_target(RVGPUCodeGen
  ${RVGPUCodeGen_sources}

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  IPO
  MC
  RVGPUDesc
  RVGPUInfo
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

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(Disassembler)
add_subdirectory(AsmParser)
