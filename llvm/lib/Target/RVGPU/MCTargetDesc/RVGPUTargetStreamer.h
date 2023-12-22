//===-- RVGPUTargetStreamer.h - RVGPU Target Streamer --------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RVGPU_MCTARGETDESC_RVGPUTARGETSTREAMER_H
#define LLVM_LIB_TARGET_RVGPU_MCTARGETDESC_RVGPUTARGETSTREAMER_H

#include "Utils/RVGPUBaseInfo.h"
#include "Utils/RVGPUPALMetadata.h"
#include "llvm/MC/MCStreamer.h"

struct rv_kernel_code_t;

namespace llvm {

class MCELFStreamer;
class MCSymbol;
class formatted_raw_ostream;

namespace RVGPU {
namespace HSRV {
struct Metadata;
}
} // namespace RVGPU

namespace rvhsa {
struct kernel_descriptor_t;
}

class RVGPUTargetStreamer : public MCTargetStreamer {
  RVGPUPALMetadata PALMetadata;

protected:
  // TODO: Move HSAMetadataStream to RVGPUTargetStreamer.
  std::optional<RVGPU::IsaInfo::RVGPUTargetID> TargetID;

  MCContext &getContext() const { return Streamer.getContext(); }

public:
  RVGPUTargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {}

  RVGPUPALMetadata *getPALMetadata() { return &PALMetadata; }

  virtual void EmitDirectiveRVGPUTarget(){};

  virtual void EmitDirectiveHSACodeObjectVersion(uint32_t Major,
                                                 uint32_t Minor){};

  virtual void EmitDirectiveHSACodeObjectISAV2(uint32_t Major, uint32_t Minor,
                                               uint32_t Stepping,
                                               StringRef VendorName,
                                               StringRef ArchName){};

  virtual void EmitRVKernelCodeT(const rv_kernel_code_t &Header){};

  virtual void EmitRVGPUSymbolType(StringRef SymbolName, unsigned Type){};

  virtual void emitRVGPULDS(MCSymbol *Symbol, unsigned Size, Align Alignment) {
  }

  /// \returns True on success, false on failure.
  virtual bool EmitISAVersion() { return true; }

  /// \returns True on success, false on failure.
  virtual bool EmitHSAMetadataV2(StringRef HSAMetadataString);

  /// \returns True on success, false on failure.
  virtual bool EmitHSAMetadataV3(StringRef HSAMetadataString);

  /// Emit HSA Metadata
  ///
  /// When \p Strict is true, known metadata elements must already be
  /// well-typed. When \p Strict is false, known types are inferred and
  /// the \p HSAMetadata structure is updated with the correct types.
  ///
  /// \returns True on success, false on failure.
  virtual bool EmitHSAMetadata(msgpack::Document &HSAMetadata, bool Strict) {
    return true;
  }

  /// \returns True on success, false on failure.
  virtual bool EmitHSAMetadata(const RVGPU::HSRV::Metadata &HSAMetadata) {
    return true;
  }

  /// \returns True on success, false on failure.
  virtual bool EmitCodeEnd(const MCSubtargetInfo &STI) { return true; }

  /// \returns True on success, false on failure.
  virtual bool EmitKernargPreloadHeader(const MCSubtargetInfo &STI) {
    return true;
  }

  virtual void EmitRvhsaKernelDescriptor(
      const MCSubtargetInfo &STI, StringRef KernelName,
      const rvhsa::kernel_descriptor_t &KernelDescriptor, uint64_t NextVGPR,
      uint64_t NextSGPR, bool ReserveVCC, bool ReserveFlatScr,
      unsigned CodeObjectVersion){};

  static StringRef getArchNameFromElfMach(unsigned ElfMach);
  static unsigned getElfMach(StringRef GPU);

  const std::optional<RVGPU::IsaInfo::RVGPUTargetID> &getTargetID() const {
    return TargetID;
  }
  std::optional<RVGPU::IsaInfo::RVGPUTargetID> &getTargetID() {
    return TargetID;
  }
  void initializeTargetID(const MCSubtargetInfo &STI,
                          unsigned CodeObjectVersion) {
    assert(TargetID == std::nullopt && "TargetID can only be initialized once");
    TargetID.emplace(STI);
    getTargetID()->setCodeObjectVersion(CodeObjectVersion);
  }
  void initializeTargetID(const MCSubtargetInfo &STI, StringRef FeatureString,
                          unsigned CodeObjectVersion) {
    initializeTargetID(STI, CodeObjectVersion);

    assert(getTargetID() != std::nullopt && "TargetID is None");
    getTargetID()->setTargetIDFromFeaturesString(FeatureString);
  }
};

class RVGPUTargetAsmStreamer final : public RVGPUTargetStreamer {
  formatted_raw_ostream &OS;
public:
  RVGPUTargetAsmStreamer(MCStreamer &S, formatted_raw_ostream &OS);

  void finish() override;

  void EmitDirectiveRVGPUTarget() override;

  void EmitDirectiveHSACodeObjectVersion(uint32_t Major,
                                         uint32_t Minor) override;

  void EmitDirectiveHSACodeObjectISAV2(uint32_t Major, uint32_t Minor,
                                       uint32_t Stepping, StringRef VendorName,
                                       StringRef ArchName) override;

  void EmitRVKernelCodeT(const rv_kernel_code_t &Header) override;

  void EmitRVGPUSymbolType(StringRef SymbolName, unsigned Type) override;

  void emitRVGPULDS(MCSymbol *Sym, unsigned Size, Align Alignment) override;

  /// \returns True on success, false on failure.
  bool EmitISAVersion() override;

  /// \returns True on success, false on failure.
  bool EmitHSAMetadata(msgpack::Document &HSAMetadata, bool Strict) override;

  /// \returns True on success, false on failure.
  bool EmitHSAMetadata(const RVGPU::HSRV::Metadata &HSAMetadata) override;

  /// \returns True on success, false on failure.
  bool EmitCodeEnd(const MCSubtargetInfo &STI) override;

  /// \returns True on success, false on failure.
  bool EmitKernargPreloadHeader(const MCSubtargetInfo &STI) override;

  void EmitRvhsaKernelDescriptor(
      const MCSubtargetInfo &STI, StringRef KernelName,
      const rvhsa::kernel_descriptor_t &KernelDescriptor, uint64_t NextVGPR,
      uint64_t NextSGPR, bool ReserveVCC, bool ReserveFlatScr,
      unsigned CodeObjectVersion) override;
};

class RVGPUTargetELFStreamer final : public RVGPUTargetStreamer {
  const MCSubtargetInfo &STI;
  MCStreamer &Streamer;

  void EmitNote(StringRef Name, const MCExpr *DescSize, unsigned NoteType,
                function_ref<void(MCELFStreamer &)> EmitDesc);

  unsigned getEFlags();

  unsigned getEFlagsRVGPU();

  unsigned getEFlagsUnknownOS();
  unsigned getEFlagsRVHSA();
  unsigned getEFlagsRVPAL();
  unsigned getEFlagsMesa3D();

  unsigned getEFlagsV3();
  unsigned getEFlagsV4();

public:
  RVGPUTargetELFStreamer(MCStreamer &S, const MCSubtargetInfo &STI);

  MCELFStreamer &getStreamer();

  void finish() override;

  void EmitDirectiveRVGPUTarget() override;

  void EmitDirectiveHSACodeObjectVersion(uint32_t Major,
                                         uint32_t Minor) override;

  void EmitDirectiveHSACodeObjectISAV2(uint32_t Major, uint32_t Minor,
                                       uint32_t Stepping, StringRef VendorName,
                                       StringRef ArchName) override;

  void EmitRVKernelCodeT(const rv_kernel_code_t &Header) override;

  void EmitRVGPUSymbolType(StringRef SymbolName, unsigned Type) override;

  void emitRVGPULDS(MCSymbol *Sym, unsigned Size, Align Alignment) override;

  /// \returns True on success, false on failure.
  bool EmitISAVersion() override;

  /// \returns True on success, false on failure.
  bool EmitHSAMetadata(msgpack::Document &HSAMetadata, bool Strict) override;

  /// \returns True on success, false on failure.
  bool EmitHSAMetadata(const RVGPU::HSRV::Metadata &HSAMetadata) override;

  /// \returns True on success, false on failure.
  bool EmitCodeEnd(const MCSubtargetInfo &STI) override;

  /// \returns True on success, false on failure.
  bool EmitKernargPreloadHeader(const MCSubtargetInfo &STI) override;

  void EmitRvhsaKernelDescriptor(
      const MCSubtargetInfo &STI, StringRef KernelName,
      const rvhsa::kernel_descriptor_t &KernelDescriptor, uint64_t NextVGPR,
      uint64_t NextSGPR, bool ReserveVCC, bool ReserveFlatScr,
      unsigned CodeObjectVersion) override;
};

}
#endif
