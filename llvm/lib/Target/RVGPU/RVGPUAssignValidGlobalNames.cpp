//===-- RVGPUAssignValidGlobalNames.cpp - Assign valid names to globals ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Clean up the names of global variables in the module to not contain symbols
// that are invalid in PTX.
//
// Currently RVGPU, like other backends, relies on generic symbol name
// sanitizing done by MC. However, the ptxas assembler is more stringent and
// disallows some additional characters in symbol names. This pass makes sure
// such names do not reach MC at all.
//
//===----------------------------------------------------------------------===//

#include "RVGPU.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace llvm;

namespace {
/// RVGPUAssignValidGlobalNames
class RVGPUAssignValidGlobalNames : public ModulePass {
public:
  static char ID;
  RVGPUAssignValidGlobalNames() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

  /// Clean up the name to remove symbols invalid in PTX.
  std::string cleanUpName(StringRef Name);
};
}

char RVGPUAssignValidGlobalNames::ID = 0;

namespace llvm {
void initializeRVGPUAssignValidGlobalNamesPass(PassRegistry &);
}

INITIALIZE_PASS(RVGPUAssignValidGlobalNames, "rvgpu-assign-valid-global-names",
                "Assign valid PTX names to globals", false, false)

bool RVGPUAssignValidGlobalNames::runOnModule(Module &M) {
  for (GlobalVariable &GV : M.globals()) {
    // We are only allowed to rename local symbols.
    if (GV.hasLocalLinkage()) {
      // setName doesn't do extra work if the name does not change.
      // Note: this does not create collisions - if setName is asked to set the
      // name to something that already exists, it adds a proper postfix to
      // avoid collisions.
      GV.setName(cleanUpName(GV.getName()));
    }
  }

  // Do the same for local functions.
  for (Function &F : M.functions())
    if (F.hasLocalLinkage())
      F.setName(cleanUpName(F.getName()));

  return true;
}

std::string RVGPUAssignValidGlobalNames::cleanUpName(StringRef Name) {
  std::string ValidName;
  raw_string_ostream ValidNameStream(ValidName);
  for (char C : Name) {
    if (C == '.' || C == '@' || C == '<' || C == '>') {
      ValidNameStream << "_$_";
    } else {
      ValidNameStream << C;
    }
  }

  return ValidNameStream.str();
}

ModulePass *llvm::createRVGPUAssignValidGlobalNamesPass() {
  return new RVGPUAssignValidGlobalNames();
}
