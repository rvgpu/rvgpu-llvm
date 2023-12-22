//===- RVKernelCodeTUtils.h - helpers for rv_kernel_code_t -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file RVKernelCodeTUtils.h
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RVGPU_UTILS_RVKERNELCODETUTILS_H
#define LLVM_LIB_TARGET_RVGPU_UTILS_RVKERNELCODETUTILS_H

struct rv_kernel_code_t;

namespace llvm {

class MCAsmParser;
class raw_ostream;
class StringRef;

void printRvKernelCodeField(const rv_kernel_code_t &C, int FldIndex,
                             raw_ostream &OS);

void dumpRvKernelCode(const rv_kernel_code_t *C, raw_ostream &OS,
                       const char *tab);

bool parseRvKernelCodeField(StringRef ID, MCAsmParser &Parser,
                             rv_kernel_code_t &C, raw_ostream &Err);

} // end namespace llvm

#endif // LLVM_LIB_TARGET_RVGPU_UTILS_RVKERNELCODETUTILS_H
