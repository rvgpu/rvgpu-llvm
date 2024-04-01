add_llvm_component_library(LLVMRVGPUInfo
  RVGPUTargetInfo.cpp

  LINK_COMPONENTS
  MC
  Support

  ADD_TO_COMPONENT
  RVGPU
  )
