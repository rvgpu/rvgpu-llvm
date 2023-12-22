include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_component_library(LLVMRVGPUDisassembler
  RVGPUDisassembler.cpp

  LINK_COMPONENTS
  RVGPUDesc
  RVGPUInfo
  RVGPUUtils
  CodeGenTypes
  MC
  MCDisassembler
  Support

  ADD_TO_COMPONENT
  RVGPU
  )
