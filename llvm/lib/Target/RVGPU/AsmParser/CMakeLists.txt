add_llvm_component_library(LLVMRVGPUAsmParser
  RVGPUAsmParser.cpp

  LINK_COMPONENTS
  MC
  MCParser
  RVGPUDesc
  RVGPUInfo
  Support
  TargetParser

  ADD_TO_COMPONENT
  RVGPU
  )
