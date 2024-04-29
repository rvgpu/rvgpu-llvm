add_llvm_component_library(LLVMRVGPUDesc
  RVGPUInstPrinter.cpp
  RVGPUMCAsmInfo.cpp
  RVGPUMCTargetDesc.cpp
  RVGPUTargetStreamer.cpp
  RVGPUAsmBackend.cpp
  RVGPUELFObjectWriter.cpp
  RVGPUMCCodeEmitter.cpp
  RVGPUELFStreamer.cpp

  LINK_COMPONENTS
  MC
  RVGPUInfo
  Support
  TargetParser

  ADD_TO_COMPONENT
  RVGPU
  )
