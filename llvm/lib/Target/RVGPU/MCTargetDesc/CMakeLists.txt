add_llvm_component_library(LLVMRVGPUDesc
  RVGPUAsmBackend.cpp
  RVGPUELFObjectWriter.cpp
  RVGPUELFStreamer.cpp
  RVGPUInstPrinter.cpp
  RVGPUMCAsmInfo.cpp
  RVGPUMCCodeEmitter.cpp
  RVGPUMCTargetDesc.cpp
  RVGPUTargetStreamer.cpp

  LINK_COMPONENTS
  RVGPUInfo
  RVGPUUtils
  BinaryFormat
  CodeGenTypes
  Core
  MC
  Support
  TargetParser

  ADD_TO_COMPONENT
  RVGPU
  )
