add_llvm_component_library(LLVMRVGPUTargetMCA
  RVGPUCustomBehaviour.cpp

  LINK_COMPONENTS
  RVGPUDesc
  RVGPUInfo
  RVGPUUtils
  CodeGenTypes
  MC
  MCA
  MCParser
  Support
  TargetParser

  ADD_TO_COMPONENT
  RVGPU
  )
