add_llvm_component_library(LLVMRVGPUDisassembler
  RVGPUDisassembler.cpp

  LINK_COMPONENTS
  MC
  MCDisassembler
  RVGPUDesc
  RVGPUInfo
  Support

  ADD_TO_COMPONENT
  RVGPU
  )
