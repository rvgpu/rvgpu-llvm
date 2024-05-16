//===-- RVGPUMCExpr.h - RVGPU specific MC expression classes ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modeled after ARMMCExpr

#ifndef LLVM_LIB_TARGET_RVGPU_RVGPUMCEXPR_H
#define LLVM_LIB_TARGET_RVGPU_RVGPUMCEXPR_H

#include "llvm/ADT/APFloat.h"
#include "llvm/MC/MCExpr.h"
#include <utility>

namespace llvm {

class RVGPUFloatMCExpr : public MCTargetExpr {
public:
  enum VariantKind {
    VK_RVGPU_None,
    VK_RVGPU_BFLOAT_PREC_FLOAT, // FP constant in bfloat-precision
    VK_RVGPU_HALF_PREC_FLOAT,   // FP constant in half-precision
    VK_RVGPU_SINGLE_PREC_FLOAT, // FP constant in single-precision
    VK_RVGPU_DOUBLE_PREC_FLOAT  // FP constant in double-precision
  };

private:
  const VariantKind Kind;
  const APFloat Flt;

  explicit RVGPUFloatMCExpr(VariantKind Kind, APFloat Flt)
      : Kind(Kind), Flt(std::move(Flt)) {}

public:
  /// @name Construction
  /// @{

  static const RVGPUFloatMCExpr *create(VariantKind Kind, const APFloat &Flt,
                                        MCContext &Ctx);

  static const RVGPUFloatMCExpr *createConstantBFPHalf(const APFloat &Flt,
                                                       MCContext &Ctx) {
    return create(VK_RVGPU_BFLOAT_PREC_FLOAT, Flt, Ctx);
  }

  static const RVGPUFloatMCExpr *createConstantFPHalf(const APFloat &Flt,
                                                        MCContext &Ctx) {
    return create(VK_RVGPU_HALF_PREC_FLOAT, Flt, Ctx);
  }

  static const RVGPUFloatMCExpr *createConstantFPSingle(const APFloat &Flt,
                                                        MCContext &Ctx) {
    return create(VK_RVGPU_SINGLE_PREC_FLOAT, Flt, Ctx);
  }

  static const RVGPUFloatMCExpr *createConstantFPDouble(const APFloat &Flt,
                                                        MCContext &Ctx) {
    return create(VK_RVGPU_DOUBLE_PREC_FLOAT, Flt, Ctx);
  }

  /// @}
  /// @name Accessors
  /// @{

  /// getOpcode - Get the kind of this expression.
  VariantKind getKind() const { return Kind; }

  /// getSubExpr - Get the child of this expression.
  APFloat getAPFloat() const { return Flt; }

/// @}

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAsmLayout *Layout,
                                 const MCFixup *Fixup) const override {
    return false;
  }
  void visitUsedExpr(MCStreamer &Streamer) const override {};
  MCFragment *findAssociatedFragment() const override { return nullptr; }

  // There are no TLS RVGPUMCExprs at the moment.
  void fixELFSymbolsInTLSFixups(MCAssembler &Asm) const override {}

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }
};

/// A wrapper for MCSymbolRefExpr that tells the assembly printer that the
/// symbol should be enclosed by generic().
class RVGPUGenericMCSymbolRefExpr : public MCTargetExpr {
private:
  const MCSymbolRefExpr *SymExpr;

  explicit RVGPUGenericMCSymbolRefExpr(const MCSymbolRefExpr *_SymExpr)
      : SymExpr(_SymExpr) {}

public:
  /// @name Construction
  /// @{

  static const RVGPUGenericMCSymbolRefExpr
  *create(const MCSymbolRefExpr *SymExpr, MCContext &Ctx);

  /// @}
  /// @name Accessors
  /// @{

  /// getOpcode - Get the kind of this expression.
  const MCSymbolRefExpr *getSymbolExpr() const { return SymExpr; }

  /// @}

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAsmLayout *Layout,
                                 const MCFixup *Fixup) const override {
    return false;
  }
  void visitUsedExpr(MCStreamer &Streamer) const override {};
  MCFragment *findAssociatedFragment() const override { return nullptr; }

  // There are no TLS RVGPUMCExprs at the moment.
  void fixELFSymbolsInTLSFixups(MCAssembler &Asm) const override {}

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }
};
class RVGPUMCExpr : public MCTargetExpr {
public:
  enum VariantKind {
    VK_RVGPU_None,
    VK_RVGPU_LO,
    VK_RVGPU_HI,
    VK_RVGPU_PCREL_LO,
    VK_RVGPU_PCREL_HI,
    VK_RVGPU_GOT_HI,
    VK_RVGPU_TPREL_LO,
    VK_RVGPU_TPREL_HI,
    VK_RVGPU_TPREL_ADD,
    VK_RVGPU_TLS_GOT_HI,
    VK_RVGPU_TLS_GD_HI,
    VK_RVGPU_CALL,
    VK_RVGPU_CALL_PLT,
    VK_RVGPU_32_PCREL,
    VK_RVGPU_Invalid // Must be the last item
  };

private:
  const MCExpr *Expr;
  const VariantKind Kind;

  int64_t evaluateAsInt64(int64_t Value) const;

  explicit RVGPUMCExpr(const MCExpr *Expr, VariantKind Kind)
      : Expr(Expr), Kind(Kind) {}

public:
  static const RVGPUMCExpr *create(const MCExpr *Expr, VariantKind Kind,
                                   MCContext &Ctx);

  VariantKind getKind() const { return Kind; }

  const MCExpr *getSubExpr() const { return Expr; }

  /// Get the corresponding PC-relative HI fixup that a VK_RVGPU_PCREL_LO
  /// points to, and optionally the fragment containing it.
  ///
  /// \returns nullptr if this isn't a VK_RVGPU_PCREL_LO pointing to a
  /// known PC-relative HI fixup.
  const MCFixup *getPCRelHiFixup(const MCFragment **DFOut) const;

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res, const MCAsmLayout *Layout,
                                 const MCFixup *Fixup) const override;
  void visitUsedExpr(MCStreamer &Streamer) const override;
  MCFragment *findAssociatedFragment() const override {
    return getSubExpr()->findAssociatedFragment();
  }

  void fixELFSymbolsInTLSFixups(MCAssembler &Asm) const override;
  
  bool evaluateAsConstant(int64_t &Res) const;

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }

  static VariantKind getVariantKindForName(StringRef name);
  static StringRef getVariantKindName(VariantKind Kind);
};

} // end namespace llvm

#endif
