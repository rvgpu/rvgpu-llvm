add_llvm_component_library(LLVMRVGPUAsmParser
  RVGPUAsmParser.cpp

  LINK_COMPONENTS
  RVGPUDesc
  RVGPUInfo
#  RVGPUUtils
  CodeGenTypes
  MC
  MCParser
  Support
  TargetParser

  ADD_TO_COMPONENT
  RVGPU
  )
