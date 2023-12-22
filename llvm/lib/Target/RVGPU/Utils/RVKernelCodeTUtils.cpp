//===- RVKernelCodeTUtils.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file - utility functions to parse/print rv_kernel_code_t structure
//
//===----------------------------------------------------------------------===//

#include "RVKernelCodeTUtils.h"
#include "RVKernelCodeT.h"
#include "SIDefines.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static ArrayRef<StringRef> get_rv_kernel_code_t_FldNames() {
  static StringRef const Table[] = {
    "", // not found placeholder
#define RECORD(name, altName, print, parse) #name
#include "RVKernelCodeTInfo.h"
#undef RECORD
  };
  return ArrayRef(Table);
}

static ArrayRef<StringRef> get_rv_kernel_code_t_FldAltNames() {
  static StringRef const Table[] = {
    "", // not found placeholder
#define RECORD(name, altName, print, parse) #altName
#include "RVKernelCodeTInfo.h"
#undef RECORD
  };
  return ArrayRef(Table);
}

static StringMap<int> createIndexMap(const ArrayRef<StringRef> &names,
                                     const ArrayRef<StringRef> &altNames) {
  StringMap<int> map;
  assert(names.size() == altNames.size());
  for (unsigned i = 0; i < names.size(); ++i) {
    map.insert(std::pair(names[i], i));
    map.insert(std::pair(altNames[i], i));
  }
  return map;
}

static int get_rv_kernel_code_t_FieldIndex(StringRef name) {
  static const auto map = createIndexMap(get_rv_kernel_code_t_FldNames(),
                                         get_rv_kernel_code_t_FldAltNames());
  return map.lookup(name) - 1; // returns -1 if not found
}

static StringRef get_rv_kernel_code_t_FieldName(int index) {
  return get_rv_kernel_code_t_FldNames()[index + 1];
}

// Field printing

static raw_ostream &printName(raw_ostream &OS, StringRef Name) {
  return OS << Name << " = ";
}

template <typename T, T rv_kernel_code_t::*ptr>
static void printField(StringRef Name, const rv_kernel_code_t &C,
                       raw_ostream &OS) {
  printName(OS, Name) << (int)(C.*ptr);
}

template <typename T, T rv_kernel_code_t::*ptr, int shift, int width = 1>
static void printBitField(StringRef Name, const rv_kernel_code_t &c,
                          raw_ostream &OS) {
  const auto Mask = (static_cast<T>(1) << width) - 1;
  printName(OS, Name) << (int)((c.*ptr >> shift) & Mask);
}

using PrintFx = void(*)(StringRef, const rv_kernel_code_t &, raw_ostream &);

static ArrayRef<PrintFx> getPrinterTable() {
  static const PrintFx Table[] = {
#define RECORD(name, altName, print, parse) print
#include "RVKernelCodeTInfo.h"
#undef RECORD
  };
  return ArrayRef(Table);
}

void llvm::printRvKernelCodeField(const rv_kernel_code_t &C,
                                   int FldIndex,
                                   raw_ostream &OS) {
  auto Printer = getPrinterTable()[FldIndex];
  if (Printer)
    Printer(get_rv_kernel_code_t_FieldName(FldIndex), C, OS);
}

void llvm::dumpRvKernelCode(const rv_kernel_code_t *C,
                             raw_ostream &OS,
                             const char *tab) {
  const int Size = getPrinterTable().size();
  for (int i = 0; i < Size; ++i) {
    OS << tab;
    printRvKernelCodeField(*C, i, OS);
    OS << '\n';
  }
}

// Field parsing

static bool expectAbsExpression(MCAsmParser &MCParser, int64_t &Value, raw_ostream& Err) {

  if (MCParser.getLexer().isNot(AsmToken::Equal)) {
    Err << "expected '='";
    return false;
  }
  MCParser.getLexer().Lex();

  if (MCParser.parseAbsoluteExpression(Value)) {
    Err << "integer absolute expression expected";
    return false;
  }
  return true;
}

template <typename T, T rv_kernel_code_t::*ptr>
static bool parseField(rv_kernel_code_t &C, MCAsmParser &MCParser,
                       raw_ostream &Err) {
  int64_t Value = 0;
  if (!expectAbsExpression(MCParser, Value, Err))
    return false;
  C.*ptr = (T)Value;
  return true;
}

template <typename T, T rv_kernel_code_t::*ptr, int shift, int width = 1>
static bool parseBitField(rv_kernel_code_t &C, MCAsmParser &MCParser,
                          raw_ostream &Err) {
  int64_t Value = 0;
  if (!expectAbsExpression(MCParser, Value, Err))
    return false;
  const uint64_t Mask = ((UINT64_C(1)  << width) - 1) << shift;
  C.*ptr &= (T)~Mask;
  C.*ptr |= (T)((Value << shift) & Mask);
  return true;
}

using ParseFx = bool(*)(rv_kernel_code_t &, MCAsmParser &MCParser,
                        raw_ostream &Err);

static ArrayRef<ParseFx> getParserTable() {
  static const ParseFx Table[] = {
#define RECORD(name, altName, print, parse) parse
#include "RVKernelCodeTInfo.h"
#undef RECORD
  };
  return ArrayRef(Table);
}

bool llvm::parseRvKernelCodeField(StringRef ID,
                                   MCAsmParser &MCParser,
                                   rv_kernel_code_t &C,
                                   raw_ostream &Err) {
  const int Idx = get_rv_kernel_code_t_FieldIndex(ID);
  if (Idx < 0) {
    Err << "unexpected rv_kernel_code_t field name " << ID;
    return false;
  }
  auto Parser = getParserTable()[Idx];
  return Parser ? Parser(C, MCParser, Err) : false;
}
