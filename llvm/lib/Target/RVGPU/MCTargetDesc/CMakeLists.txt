add_llvm_component_library(LLVMRVGPUDesc
  RVGPUInstPrinter.cpp
  RVGPUMCAsmInfo.cpp
  RVGPUMCTargetDesc.cpp
  RVGPUTargetStreamer.cpp

  LINK_COMPONENTS
  MC
  RVGPUInfo
  Support
  TargetParser

  ADD_TO_COMPONENT
  RVGPU
  )
