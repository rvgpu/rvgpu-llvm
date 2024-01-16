//===- RVGPU.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "Symbols.h"
#include "Target.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

namespace {
class RVGPU final : public TargetInfo {
private:
  uint32_t calcEFlagsV3() const;
  uint32_t calcEFlagsV4() const;

public:
  RVGPU();
  uint32_t calcEFlags() const override;
  void relocate(uint8_t *loc, const Relocation &rel,
                uint64_t val) const override;
  RelExpr getRelExpr(RelType type, const Symbol &s,
                     const uint8_t *loc) const override;
  RelType getDynRel(RelType type) const override;
  int64_t getImplicitAddend(const uint8_t *buf, RelType type) const override;
};
} // namespace

RVGPU::RVGPU() {
  relativeRel = R_RVGPU_RELATIVE64;
  gotRel = R_RVGPU_ABS64;
  symbolicRel = R_RVGPU_ABS64;
}

static uint32_t getEFlags(InputFile *file) {
  return cast<ObjFile<ELF64LE>>(file)->getObj().getHeader().e_flags;
}

uint32_t RVGPU::calcEFlagsV3() const {
  uint32_t ret = getEFlags(ctx.objectFiles[0]);

  // Verify that all input files have the same e_flags.
  for (InputFile *f : ArrayRef(ctx.objectFiles).slice(1)) {
    if (ret == getEFlags(f))
      continue;
    error("incompatible e_flags: " + toString(f));
    return 0;
  }
  return ret;
}

uint32_t RVGPU::calcEFlagsV4() const {
#if 0
  uint32_t retMach = getEFlags(ctx.objectFiles[0]) & EF_RVGPU_MACH;
  uint32_t retXnack =
      getEFlags(ctx.objectFiles[0]) & EF_RVGPU_FEATURE_XNACK_V4;
  uint32_t retSramEcc =
      getEFlags(ctx.objectFiles[0]) & EF_RVGPU_FEATURE_SRAMECC_V4;

  // Verify that all input files have compatible e_flags (same mach, all
  // features in the same category are either ANY, ANY and ON, or ANY and OFF).
  for (InputFile *f : ArrayRef(ctx.objectFiles).slice(1)) {
    if (retMach != (getEFlags(f) & EF_RVGPU_MACH)) {
      error("incompatible mach: " + toString(f));
      return 0;
    }

    if (retXnack == EF_RVGPU_FEATURE_XNACK_UNSUPPORTED_V4 ||
        (retXnack != EF_RVGPU_FEATURE_XNACK_ANY_V4 &&
            (getEFlags(f) & EF_RVGPU_FEATURE_XNACK_V4)
                != EF_RVGPU_FEATURE_XNACK_ANY_V4)) {
      if (retXnack != (getEFlags(f) & EF_RVGPU_FEATURE_XNACK_V4)) {
        error("incompatible xnack: " + toString(f));
        return 0;
      }
    } else {
      if (retXnack == EF_RVGPU_FEATURE_XNACK_ANY_V4)
        retXnack = getEFlags(f) & EF_RVGPU_FEATURE_XNACK_V4;
    }
    if (retSramEcc == EF_RVGPU_FEATURE_SRAMECC_UNSUPPORTED_V4 ||
        (retSramEcc != EF_RVGPU_FEATURE_SRAMECC_ANY_V4 &&
            (getEFlags(f) & EF_RVGPU_FEATURE_SRAMECC_V4) !=
                EF_RVGPU_FEATURE_SRAMECC_ANY_V4)) {
      if (retSramEcc != (getEFlags(f) & EF_RVGPU_FEATURE_SRAMECC_V4)) {
        error("incompatible sramecc: " + toString(f));
        return 0;
      }
    } else {
      if (retSramEcc == EF_RVGPU_FEATURE_SRAMECC_ANY_V4)
        retSramEcc = getEFlags(f) & EF_RVGPU_FEATURE_SRAMECC_V4;
    }
  }

  return retMach | retXnack | retSramEcc;
  #endif 
  return 0;
}

uint32_t RVGPU::calcEFlags() const {
  if (ctx.objectFiles.empty())
    return 0;

  uint8_t abiVersion = cast<ObjFile<ELF64LE>>(ctx.objectFiles[0])
                           ->getObj()
                           .getHeader()
                           .e_ident[EI_ABIVERSION];
  switch (abiVersion) {
  case ELFABIVERSION_RVGPU_SS_V2:
  case ELFABIVERSION_RVGPU_SS_V3:
    return calcEFlagsV3();
  case ELFABIVERSION_RVGPU_SS_V4:
  case ELFABIVERSION_RVGPU_SS_V5:
    return calcEFlagsV4();
  default:
    error("unknown abi version: " + Twine(abiVersion));
    return 0;
  }
}

void RVGPU::relocate(uint8_t *loc, const Relocation &rel, uint64_t val) const {
  switch (rel.type) {
  case R_RVGPU_ABS32:
  case R_RVGPU_GOTPCREL:
  case R_RVGPU_GOTPCREL32_LO:
  case R_RVGPU_REL32:
  case R_RVGPU_REL32_LO:
    write32le(loc, val);
    break;
  case R_RVGPU_ABS64:
  case R_RVGPU_REL64:
    write64le(loc, val);
    break;
  case R_RVGPU_GOTPCREL32_HI:
  case R_RVGPU_REL32_HI:
    write32le(loc, val >> 32);
    break;
  case R_RVGPU_REL16: {
    int64_t simm = (static_cast<int64_t>(val) - 4) / 4;
    checkInt(loc, simm, 16, rel);
    write16le(loc, simm);
    break;
  }
  default:
    llvm_unreachable("unknown relocation");
  }
}

RelExpr RVGPU::getRelExpr(RelType type, const Symbol &s,
                           const uint8_t *loc) const {
  switch (type) {
  case R_RVGPU_ABS32:
  case R_RVGPU_ABS64:
    return R_ABS;
  case R_RVGPU_REL32:
  case R_RVGPU_REL32_LO:
  case R_RVGPU_REL32_HI:
  case R_RVGPU_REL64:
  case R_RVGPU_REL16:
    return R_PC;
  case R_RVGPU_GOTPCREL:
  case R_RVGPU_GOTPCREL32_LO:
  case R_RVGPU_GOTPCREL32_HI:
    return R_GOT_PC;
  default:
    error(getErrorLocation(loc) + "unknown relocation (" + Twine(type) +
          ") against symbol " + toString(s));
    return R_NONE;
  }
}

RelType RVGPU::getDynRel(RelType type) const {
  if (type == R_RVGPU_ABS64)
    return type;
  return R_RVGPU_NONE;
}

int64_t RVGPU::getImplicitAddend(const uint8_t *buf, RelType type) const {
  switch (type) {
  case R_RVGPU_NONE:
    return 0;
  case R_RVGPU_ABS64:
  case R_RVGPU_RELATIVE64:
    return read64(buf);
  default:
    internalLinkerError(getErrorLocation(buf),
                        "cannot read addend for relocation " + toString(type));
    return 0;
  }
}

TargetInfo *elf::getRVGPUTargetInfo() {
  static RVGPU target;
  return &target;
}
