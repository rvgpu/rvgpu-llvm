#ifndef LLVM_LIB_TARGET_RISCV_RVGPU_H
#define LLVM_LIB_TARGET_RISCV_RVGPU_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"

namespace llvm {

FunctionPass *createRVGPULowerIntrinsicsPass();
void initializeRVGPULowerIntrinsicsPass(PassRegistry &);
extern char &RVGPULowerIntrinsicsID;

} // End namespace llvm
#endif // LLVM_LIB_TARGET_RISCV_RVGPU_H
