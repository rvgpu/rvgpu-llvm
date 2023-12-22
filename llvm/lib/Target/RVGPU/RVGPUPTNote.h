//===-- RVGPUNoteType.h - RVGPU ELF PT_NOTE section info-------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// Enums and constants for RVGPU PT_NOTE sections.
///
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_LIB_TARGET_RVGPU_RVGPUPTNOTE_H
#define LLVM_LIB_TARGET_RVGPU_RVGPUPTNOTE_H

namespace llvm {
namespace RVGPU {

namespace ElfNote {

const char SectionName[] = ".note";

const char NoteNameV2[] = "RV";
const char NoteNameV3[] = "RVGPU";

} // End namespace ElfNote
} // End namespace RVGPU
} // End namespace llvm
#endif // LLVM_LIB_TARGET_RVGPU_RVGPUPTNOTE_H
