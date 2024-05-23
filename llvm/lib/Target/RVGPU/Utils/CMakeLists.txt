add_llvm_component_library(LLVMRVGPUUtils
  #RVGPUAsmUtils.cpp
  RVGPUBaseInfo.cpp
  #RVGPUMemoryUtils.cpp
  #RVGPUPALMetadata.cpp
  #AMDKernelCodeTUtils.cpp

  LINK_COMPONENTS
  Analysis
  BinaryFormat
  CodeGenTypes
  Core
  MC
  Support
  TargetParser

  ADD_TO_COMPONENT
  RVGPU
  )
