#include "RVGPU.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"

#define DEBUG_TYPE "rvgpu-lower-intrinsics"

using namespace llvm;

namespace {

class RVGPULowerIntrinsics : public FunctionPass {

public:
  static char ID;

  RVGPULowerIntrinsics() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override ;

  StringRef getPassName() const override {
    return "RVGPU Lower Intrinsics";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }

  bool lowerNVVMIntrinsics(Function &F);
};
}

char RVGPULowerIntrinsics::ID = 0;

INITIALIZE_PASS(RVGPULowerIntrinsics, DEBUG_TYPE, "Lower intrinsics", false,
                false)


FunctionPass *llvm::createRVGPULowerIntrinsicsPass() {
  return new RVGPULowerIntrinsics();
}

bool RVGPULowerIntrinsics::lowerNVVMIntrinsics(Function &F) {
  Intrinsic::ID ID = F.getIntrinsicID();
  switch(ID) {
  //case Intrinsic::nvvm_read_ptx_sreg_tid_x:
   //   printf("lower nvvm read ptx sreg tid x \n");
    //  return true;
  default:
      return false;
  }

}

bool RVGPULowerIntrinsics::runOnFunction(Function &F) {
  bool Changed = false;

  std::vector<CallInst*> to_delete;

  for (auto &BB : F) {
      for (BasicBlock::iterator I = BB.begin(); I != BB.end();) {
        auto CI = dyn_cast<CallInst>(I);
        ++I;
        if (!CI) continue;

        Function *Callee = CI->getCalledFunction();
        IRBuilder<> Builder(CI);

        if (Callee == nullptr)
          continue;

        switch (Callee->getIntrinsicID()) {
        case Intrinsic::nvvm_read_ptx_sreg_tid_x: {
          Changed = true;
          CallInst *NewCI = Builder.CreateIntrinsic(Intrinsic::riscv_rvgpu_read_tid_x,  std::nullopt,  std::nullopt);
          CI->replaceAllUsesWith(NewCI);
          //CI->eraseFromParent();
          to_delete.push_back(CI);
          break;
        }

        default :
          break;
        }
      }
  }

  if (!to_delete.empty()) {
      auto CI = to_delete.back();
      to_delete.pop_back();
      CI->eraseFromParent();
  }
  return Changed;
}
