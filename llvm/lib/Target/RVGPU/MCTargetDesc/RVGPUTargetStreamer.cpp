//===-- RVGPUTargetStreamer.cpp - Mips Target Streamer Methods -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides RVGPU specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "RVGPUTargetStreamer.h"
#include "RVGPUPTNote.h"
#include "RVKernelCodeT.h"
/*#include "Utils/RVGPUBaseInfo.h"
#include "Utils/AMDKernelCodeTUtils.h"
#include "llvm/BinaryFormat/RVGPUMetadataVerifier.h"
*/
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/RVGPUMetadata.h"
#include "llvm/Support/RVHSAKernelDescriptor.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/TargetParser/TargetParser.h"

using namespace llvm;
using namespace llvm::RVGPU;

//===----------------------------------------------------------------------===//
// RVGPUTargetStreamer
//===----------------------------------------------------------------------===//

#if 0
bool RVGPUTargetStreamer::EmitHSAMetadataV3(StringRef HSAMetadataString) {
  msgpack::Document HSAMetadataDoc;
  if (!HSAMetadataDoc.fromYAML(HSAMetadataString))
    return false;
  return EmitHSAMetadata(HSAMetadataDoc, false);
}
#endif 
StringRef RVGPUTargetStreamer::getArchNameFromElfMach(unsigned ElfMach) {
  return "R1000";
}

unsigned RVGPUTargetStreamer::getElfMach(StringRef GPU) {
  return ELF::EF_RVGPU_MATCH_R1000;
}

//===----------------------------------------------------------------------===//
// RVGPUTargetAsmStreamer
//===----------------------------------------------------------------------===//

RVGPUTargetAsmStreamer::RVGPUTargetAsmStreamer(MCStreamer &S,
                                                 formatted_raw_ostream &OS)
    : RVGPUTargetStreamer(S), OS(OS) { }

// A hook for emitting stuff at the end.
// We use it for emitting the accumulated PAL metadata as directives.
// The PAL metadata is reset after it is emitted.
void RVGPUTargetAsmStreamer::finish() {
  #if 0
  std::string S;
  getPALMetadata()->toString(S);
  OS << S;

  // Reset the pal metadata so its data will not affect a compilation that
  // reuses this object.
  getPALMetadata()->reset();
  #endif 
}

void RVGPUTargetAsmStreamer::EmitDirectiveRVGPUTarget() {
//  OS << "\t.rvgpu_target \"" << getTargetID()->toString() << "\"\n";
}

void RVGPUTargetAsmStreamer::EmitDirectiveHSACodeObjectVersion(
    uint32_t Major, uint32_t Minor) {
  OS << "\t.hsa_code_object_version " <<
        Twine(Major) << "," << Twine(Minor) << '\n';
}

void
RVGPUTargetAsmStreamer::EmitDirectiveHSACodeObjectISAV2(uint32_t Major,
                                                         uint32_t Minor,
                                                         uint32_t Stepping,
                                                         StringRef VendorName,
                                                         StringRef ArchName) {
  OS << "\t.hsa_code_object_isa " << Twine(Major) << "," << Twine(Minor) << ","
     << Twine(Stepping) << ",\"" << VendorName << "\",\"" << ArchName << "\"\n";
}

void
RVGPUTargetAsmStreamer::EmitRVKernelCodeT(const rv_kernel_code_t &Header) {
  OS << "\t.rv_kernel_code_t\n";
  //dumpAmdKernelCode(&Header, OS, "\t\t");
  OS << "\t.end_rv_kernel_code_t\n";
}

void RVGPUTargetAsmStreamer::EmitRVGPUSymbolType(StringRef SymbolName,
                                                   unsigned Type) {
  switch (Type) {
    default: llvm_unreachable("Invalid RVGPU symbol type");
    case ELF::STT_RVGPU_HSA_KERNEL:
      OS << "\t.rvgpu_hsa_kernel " << SymbolName << '\n' ;
      break;
  }
}

void RVGPUTargetAsmStreamer::emitRVGPULDS(MCSymbol *Symbol, unsigned Size,
                                            Align Alignment) {
  OS << "\t.rvgpu_lds " << Symbol->getName() << ", " << Size << ", "
     << Alignment.value() << '\n';
}

bool RVGPUTargetAsmStreamer::EmitISAVersion() {
//  OS << "\t.rv_rvgpu_isa \"" << getTargetID()->toString() << "\"\n";
  return true;
}

bool RVGPUTargetAsmStreamer::EmitHSAMetadata(
    const RVGPU::HSAMD::Metadata &HSAMetadata) {
#if 0  
  std::string HSAMetadataString;
  if (HSAMD::toString(HSAMetadata, HSAMetadataString))
    return false;

  OS << '\t' << ".rv_rvgpu_hsa_metadata" << '\n';
  OS << HSAMetadataString << '\n';
  OS << '\t' << ".end_rv_rvgpu_hsa_metadata" << '\n';
#endif   
  return true;
}

bool RVGPUTargetAsmStreamer::EmitHSAMetadata(
    msgpack::Document &HSAMetadataDoc, bool Strict) {

  std::string HSAMetadataString;
  raw_string_ostream StrOS(HSAMetadataString);
  HSAMetadataDoc.toYAML(StrOS);

  OS << '\t' << ".rv_rvgpu_hsa_metadata" << '\n';
  OS << StrOS.str() << '\n';
  OS << '\t' << ".end_rv_rvgpu_hsa_metadata" << '\n';
  return true;
}

bool RVGPUTargetAsmStreamer::EmitCodeEnd(const MCSubtargetInfo &STI) {
  const uint32_t Encoded_s_code_end = 0xbf9f0000;
  const uint32_t Encoded_s_nop = 0xbf800000;
  uint32_t Encoded_pad = Encoded_s_code_end;

  // Instruction cache line size in bytes.
  const unsigned Log2CacheLineSize = 7;
  const unsigned CacheLineSize = 1u << Log2CacheLineSize;

  // Extra padding amount in bytes to support prefetch mode 3.
  unsigned FillSize = 3 * CacheLineSize;

  OS << "\t.p2alignl " << Log2CacheLineSize << ", " << Encoded_pad << '\n';
  OS << "\t.fill " << (FillSize / 4) << ", 4, " << Encoded_pad << '\n';
  return true;
}

void RVGPUTargetAsmStreamer::EmitRvhsaKernelDescriptor(
    const MCSubtargetInfo &STI, StringRef KernelName,
    const rvhsa::kernel_descriptor_t &KD, uint64_t NextVGPR, uint64_t NextSGPR,
    bool ReserveVCC, bool ReserveFlatScr, unsigned CodeObjectVersion) {

  OS << "\t.rvhsa_kernel " << KernelName << '\n';
#if 0
#define PRINT_FIELD(STREAM, DIRECTIVE, KERNEL_DESC, MEMBER_NAME, FIELD_NAME)   \
  STREAM << "\t\t" << DIRECTIVE << " "                                         \
         << AMDHSA_BITS_GET(KERNEL_DESC.MEMBER_NAME, FIELD_NAME) << '\n';

  OS << "\t\t.rvhsa_group_segment_fixed_size " << KD.group_segment_fixed_size
     << '\n';
  OS << "\t\t.rvhsa_private_segment_fixed_size "
     << KD.private_segment_fixed_size << '\n';
  OS << "\t\t.rvhsa_kernarg_size " << KD.kernarg_size << '\n';

  PRINT_FIELD(OS, ".rvhsa_user_sgpr_count", KD,
              compute_pgm_rsrc2,
              rvhsa::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT);

  if (!hasArchitectedFlatScratch(STI))
    PRINT_FIELD(
        OS, ".rvhsa_user_sgpr_private_segment_buffer", KD,
        kernel_code_properties,
        rvhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER);
  PRINT_FIELD(OS, ".rvhsa_user_sgpr_dispatch_ptr", KD,
              kernel_code_properties,
              rvhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR);
  PRINT_FIELD(OS, ".rvhsa_user_sgpr_queue_ptr", KD,
              kernel_code_properties,
              rvhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR);
  PRINT_FIELD(OS, ".rvhsa_user_sgpr_kernarg_segment_ptr", KD,
              kernel_code_properties,
              rvhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR);
  PRINT_FIELD(OS, ".rvhsa_user_sgpr_dispatch_id", KD,
              kernel_code_properties,
              rvhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID);
  if (!hasArchitectedFlatScratch(STI))
    PRINT_FIELD(OS, ".rvhsa_user_sgpr_flat_scratch_init", KD,
                kernel_code_properties,
                rvhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT);
  if (hasKernargPreload(STI)) {
    PRINT_FIELD(OS, ".rvhsa_user_sgpr_kernarg_preload_length ", KD,
                kernarg_preload, rvhsa::KERNARG_PRELOAD_SPEC_LENGTH);
    PRINT_FIELD(OS, ".rvhsa_user_sgpr_kernarg_preload_offset ", KD,
                kernarg_preload, rvhsa::KERNARG_PRELOAD_SPEC_OFFSET);
  }
  PRINT_FIELD(OS, ".rvhsa_user_sgpr_private_segment_size", KD,
              kernel_code_properties,
              rvhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE);
  if (IVersion.Major >= 10)
    PRINT_FIELD(OS, ".rvhsa_wavefront_size32", KD,
                kernel_code_properties,
                rvhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);
  if (CodeObjectVersion >= RVGPU::AMDHSA_COV5)
    PRINT_FIELD(OS, ".rvhsa_uses_dynamic_stack", KD, kernel_code_properties,
                rvhsa::KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK);
  PRINT_FIELD(OS,
              (hasArchitectedFlatScratch(STI)
                   ? ".rvhsa_enable_private_segment"
                   : ".rvhsa_system_sgpr_private_segment_wavefront_offset"),
              KD, compute_pgm_rsrc2,
              rvhsa::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT);
  PRINT_FIELD(OS, ".rvhsa_system_sgpr_workgroup_id_x", KD,
              compute_pgm_rsrc2,
              rvhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X);
  PRINT_FIELD(OS, ".rvhsa_system_sgpr_workgroup_id_y", KD,
              compute_pgm_rsrc2,
              rvhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y);
  PRINT_FIELD(OS, ".rvhsa_system_sgpr_workgroup_id_z", KD,
              compute_pgm_rsrc2,
              rvhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z);
  PRINT_FIELD(OS, ".rvhsa_system_sgpr_workgroup_info", KD,
              compute_pgm_rsrc2,
              rvhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO);
  PRINT_FIELD(OS, ".rvhsa_system_vgpr_workitem_id", KD,
              compute_pgm_rsrc2,
              rvhsa::COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID);

  // These directives are required.
  OS << "\t\t.rvhsa_next_free_vgpr " << NextVGPR << '\n';
  OS << "\t\t.rvhsa_next_free_sgpr " << NextSGPR << '\n';

  if (RVGPU::isGFX90A(STI))
    OS << "\t\t.rvhsa_accum_offset " <<
      (AMDHSA_BITS_GET(KD.compute_pgm_rsrc3,
                       rvhsa::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET) + 1) * 4
      << '\n';

  if (!ReserveVCC)
    OS << "\t\t.rvhsa_reserve_vcc " << ReserveVCC << '\n';
  if (IVersion.Major >= 7 && !ReserveFlatScr && !hasArchitectedFlatScratch(STI))
    OS << "\t\t.rvhsa_reserve_flat_scratch " << ReserveFlatScr << '\n';


  PRINT_FIELD(OS, ".rvhsa_float_round_mode_32", KD,
              compute_pgm_rsrc1,
              rvhsa::COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_32);
  PRINT_FIELD(OS, ".rvhsa_float_round_mode_16_64", KD,
              compute_pgm_rsrc1,
              rvhsa::COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_16_64);
  PRINT_FIELD(OS, ".rvhsa_float_denorm_mode_32", KD,
              compute_pgm_rsrc1,
              rvhsa::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_32);
  PRINT_FIELD(OS, ".rvhsa_float_denorm_mode_16_64", KD,
              compute_pgm_rsrc1,
              rvhsa::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64);
  if (IVersion.Major < 12) {
    PRINT_FIELD(OS, ".rvhsa_dx10_clamp", KD, compute_pgm_rsrc1,
                rvhsa::COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_DX10_CLAMP);
    PRINT_FIELD(OS, ".rvhsa_ieee_mode", KD, compute_pgm_rsrc1,
                rvhsa::COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_IEEE_MODE);
  }
  if (IVersion.Major >= 9)
    PRINT_FIELD(OS, ".rvhsa_fp16_overflow", KD,
                compute_pgm_rsrc1,
                rvhsa::COMPUTE_PGM_RSRC1_GFX9_PLUS_FP16_OVFL);
  if (RVGPU::isGFX90A(STI))
    PRINT_FIELD(OS, ".rvhsa_tg_split", KD,
                compute_pgm_rsrc3,
                rvhsa::COMPUTE_PGM_RSRC3_GFX90A_TG_SPLIT);
  if (IVersion.Major >= 10) {
    PRINT_FIELD(OS, ".rvhsa_workgroup_processor_mode", KD,
                compute_pgm_rsrc1,
                rvhsa::COMPUTE_PGM_RSRC1_GFX10_PLUS_WGP_MODE);
    PRINT_FIELD(OS, ".rvhsa_memory_ordered", KD,
                compute_pgm_rsrc1,
                rvhsa::COMPUTE_PGM_RSRC1_GFX10_PLUS_MEM_ORDERED);
    PRINT_FIELD(OS, ".rvhsa_forward_progress", KD,
                compute_pgm_rsrc1,
                rvhsa::COMPUTE_PGM_RSRC1_GFX10_PLUS_FWD_PROGRESS);
    PRINT_FIELD(OS, ".rvhsa_shared_vgpr_count", KD, compute_pgm_rsrc3,
                rvhsa::COMPUTE_PGM_RSRC3_GFX10_PLUS_SHARED_VGPR_COUNT);
  }
  if (IVersion.Major >= 12)
    PRINT_FIELD(OS, ".rvhsa_round_robin_scheduling", KD, compute_pgm_rsrc1,
                rvhsa::COMPUTE_PGM_RSRC1_GFX12_PLUS_ENABLE_WG_RR_EN);
  PRINT_FIELD(
      OS, ".rvhsa_exception_fp_ieee_invalid_op", KD,
      compute_pgm_rsrc2,
      rvhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION);
  PRINT_FIELD(OS, ".rvhsa_exception_fp_denorm_src", KD,
              compute_pgm_rsrc2,
              rvhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE);
  PRINT_FIELD(
      OS, ".rvhsa_exception_fp_ieee_div_zero", KD,
      compute_pgm_rsrc2,
      rvhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO);
  PRINT_FIELD(OS, ".rvhsa_exception_fp_ieee_overflow", KD,
              compute_pgm_rsrc2,
              rvhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW);
  PRINT_FIELD(OS, ".rvhsa_exception_fp_ieee_underflow", KD,
              compute_pgm_rsrc2,
              rvhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW);
  PRINT_FIELD(OS, ".rvhsa_exception_fp_ieee_inexact", KD,
              compute_pgm_rsrc2,
              rvhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT);
  PRINT_FIELD(OS, ".rvhsa_exception_int_div_zero", KD,
              compute_pgm_rsrc2,
              rvhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_INT_DIVIDE_BY_ZERO);
#undef PRINT_FIELD
#endif 
  OS << "\t.end_rvhsa_kernel\n";
}

//===----------------------------------------------------------------------===//
// RVGPUTargetELFStreamer
//===----------------------------------------------------------------------===//

RVGPUTargetELFStreamer::RVGPUTargetELFStreamer(MCStreamer &S,
                                                 const MCSubtargetInfo &STI)
    : RVGPUTargetStreamer(S), STI(STI), Streamer(S) {}

MCELFStreamer &RVGPUTargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(Streamer);
}

// A hook for emitting stuff at the end.
// We use it for emitting the accumulated PAL metadata as a .note record.
// The PAL metadata is reset after it is emitted.
void RVGPUTargetELFStreamer::finish() {
  MCAssembler &MCA = getStreamer().getAssembler();
  MCA.setELFHeaderEFlags(0);
#if 0
  std::string Blob;
  const char *Vendor = getPALMetadata()->getVendor();
  unsigned Type = getPALMetadata()->getType();
  getPALMetadata()->toBlob(Type, Blob);
  if (Blob.empty())
    return;
  EmitNote(Vendor, MCConstantExpr::create(Blob.size(), getContext()), Type,
           [&](MCELFStreamer &OS) { OS.emitBytes(Blob); });

  // Reset the pal metadata so its data will not affect a compilation that
  // reuses this object.
  getPALMetadata()->reset();
  #endif 
}

void RVGPUTargetELFStreamer::EmitNote(
    StringRef Name, const MCExpr *DescSZ, unsigned NoteType,
    function_ref<void(MCELFStreamer &)> EmitDesc) {
  auto &S = getStreamer();
  auto &Context = S.getContext();

  auto NameSZ = Name.size() + 1;

  unsigned NoteFlags = 0;
  // TODO Apparently, this is currently needed for OpenCL as mentioned in
  // https://reviews.llvm.org/D74995
  NoteFlags = ELF::SHF_ALLOC;

  S.pushSection();
  S.switchSection(
      Context.getELFSection(ElfNote::SectionName, ELF::SHT_NOTE, NoteFlags));
  S.emitInt32(NameSZ);                                        // namesz
  S.emitValue(DescSZ, 4);                                     // descz
  S.emitInt32(NoteType);                                      // type
  S.emitBytes(Name);                                          // name
  S.emitValueToAlignment(Align(4), 0, 1, 0);                  // padding 0
  EmitDesc(S);                                                // desc
  S.emitValueToAlignment(Align(4), 0, 1, 0);                  // padding 0
  S.popSection();
}

void RVGPUTargetELFStreamer::EmitDirectiveRVGPUTarget() {}

void RVGPUTargetELFStreamer::EmitDirectiveHSACodeObjectVersion(
    uint32_t Major, uint32_t Minor) {

  EmitNote(ElfNote::NoteNameV2, MCConstantExpr::create(8, getContext()),
           ELF::NT_AMD_HSA_CODE_OBJECT_VERSION, [&](MCELFStreamer &OS) {
             OS.emitInt32(Major);
             OS.emitInt32(Minor);
           });
}

void
RVGPUTargetELFStreamer::EmitDirectiveHSACodeObjectISAV2(uint32_t Major,
                                                         uint32_t Minor,
                                                         uint32_t Stepping,
                                                         StringRef VendorName,
                                                         StringRef ArchName) {
  uint16_t VendorNameSize = VendorName.size() + 1;
  uint16_t ArchNameSize = ArchName.size() + 1;

  unsigned DescSZ = sizeof(VendorNameSize) + sizeof(ArchNameSize) +
    sizeof(Major) + sizeof(Minor) + sizeof(Stepping) +
    VendorNameSize + ArchNameSize;

  EmitNote(ElfNote::NoteNameV2, MCConstantExpr::create(DescSZ, getContext()),
           ELF::NT_AMD_HSA_ISA_VERSION, [&](MCELFStreamer &OS) {
             OS.emitInt16(VendorNameSize);
             OS.emitInt16(ArchNameSize);
             OS.emitInt32(Major);
             OS.emitInt32(Minor);
             OS.emitInt32(Stepping);
             OS.emitBytes(VendorName);
             OS.emitInt8(0); // NULL terminate VendorName
             OS.emitBytes(ArchName);
             OS.emitInt8(0); // NULL terminate ArchName
           });
}

void
RVGPUTargetELFStreamer::EmitRVKernelCodeT(const rv_kernel_code_t &Header) {

  MCStreamer &OS = getStreamer();
  OS.pushSection();
  OS.emitBytes(StringRef((const char*)&Header, sizeof(Header)));
  OS.popSection();
}

void RVGPUTargetELFStreamer::EmitRVGPUSymbolType(StringRef SymbolName,
                                                   unsigned Type) {
  MCSymbolELF *Symbol = cast<MCSymbolELF>(
      getStreamer().getContext().getOrCreateSymbol(SymbolName));
  Symbol->setType(Type);
}

void RVGPUTargetELFStreamer::emitRVGPULDS(MCSymbol *Symbol, unsigned Size,
                                            Align Alignment) {
  MCSymbolELF *SymbolELF = cast<MCSymbolELF>(Symbol);
  SymbolELF->setType(ELF::STT_OBJECT);

  if (!SymbolELF->isBindingSet()) {
    SymbolELF->setBinding(ELF::STB_GLOBAL);
    SymbolELF->setExternal(true);
  }

  if (SymbolELF->declareCommon(Size, Alignment, true)) {
    report_fatal_error("Symbol: " + Symbol->getName() +
                       " redeclared as different type");
  }

  SymbolELF->setIndex(ELF::SHN_RVGPU_LDS);
  SymbolELF->setSize(MCConstantExpr::create(Size, getContext()));
}

bool RVGPUTargetELFStreamer::EmitISAVersion() {
  // Create two labels to mark the beginning and end of the desc field
  // and a MCExpr to calculate the size of the desc field.
  auto &Context = getContext();
  auto *DescBegin = Context.createTempSymbol();
  auto *DescEnd = Context.createTempSymbol();
  auto *DescSZ = MCBinaryExpr::createSub(
    MCSymbolRefExpr::create(DescEnd, Context),
    MCSymbolRefExpr::create(DescBegin, Context), Context);

  EmitNote(ElfNote::NoteNameV2, DescSZ, ELF::NT_AMD_HSA_ISA_NAME,
           [&](MCELFStreamer &OS) {
             OS.emitLabel(DescBegin);
             OS.emitBytes("");
             //OS.emitBytes(getTargetID()->toString());
             OS.emitLabel(DescEnd);
           });
  return true;
}

bool RVGPUTargetELFStreamer::EmitHSAMetadata(msgpack::Document &HSAMetadataDoc,
                                              bool Strict) {
#if 0
  HSAMD::V3::MetadataVerifier Verifier(Strict);
  if (!Verifier.verify(HSAMetadataDoc.getRoot()))
    return false;

  std::string HSAMetadataString;
  HSAMetadataDoc.writeToBlob(HSAMetadataString);

  // Create two labels to mark the beginning and end of the desc field
  // and a MCExpr to calculate the size of the desc field.
  auto &Context = getContext();
  auto *DescBegin = Context.createTempSymbol();
  auto *DescEnd = Context.createTempSymbol();
  auto *DescSZ = MCBinaryExpr::createSub(
      MCSymbolRefExpr::create(DescEnd, Context),
      MCSymbolRefExpr::create(DescBegin, Context), Context);

  EmitNote(ElfNote::NoteNameV3, DescSZ, ELF::NT_RVGPU_METADATA,
           [&](MCELFStreamer &OS) {
             OS.emitLabel(DescBegin);
             OS.emitBytes(HSAMetadataString);
             OS.emitLabel(DescEnd);
           });
#endif                                                 
  return true;
}

bool RVGPUTargetELFStreamer::EmitHSAMetadata(
    const RVGPU::HSAMD::Metadata &HSAMetadata) {
#if 0
  std::string HSAMetadataString;
  if (HSAMD::toString(HSAMetadata, HSAMetadataString))
    return false;

  // Create two labels to mark the beginning and end of the desc field
  // and a MCExpr to calculate the size of the desc field.
  auto &Context = getContext();
  auto *DescBegin = Context.createTempSymbol();
  auto *DescEnd = Context.createTempSymbol();
  auto *DescSZ = MCBinaryExpr::createSub(
    MCSymbolRefExpr::create(DescEnd, Context),
    MCSymbolRefExpr::create(DescBegin, Context), Context);

  EmitNote(ElfNote::NoteNameV2, DescSZ, ELF::NT_AMD_HSA_METADATA,
           [&](MCELFStreamer &OS) {
             OS.emitLabel(DescBegin);
             OS.emitBytes(HSAMetadataString);
             OS.emitLabel(DescEnd);
           });
#endif 
  return true;
}

bool RVGPUTargetAsmStreamer::EmitKernargPreloadHeader(
    const MCSubtargetInfo &STI) {
      #if 0
  for (int i = 0; i < 64; ++i) {
    OS << "\ts_nop 0\n";
  }
  return true;
      #endif 
  return false;
}

bool RVGPUTargetELFStreamer::EmitKernargPreloadHeader(
    const MCSubtargetInfo &STI) {
#if 0      
  const uint32_t Encoded_s_nop = 0xbf800000;
  MCStreamer &OS = getStreamer();
  for (int i = 0; i < 64; ++i) {
    OS.emitInt32(Encoded_s_nop);
  }
  return true;
#endif 
  return false;
}

bool RVGPUTargetELFStreamer::EmitCodeEnd(const MCSubtargetInfo &STI) {
  #if 0
  const uint32_t Encoded_s_code_end = 0xbf9f0000;
  const uint32_t Encoded_s_nop = 0xbf800000;
  uint32_t Encoded_pad = Encoded_s_code_end;

  // Instruction cache line size in bytes.
  const unsigned Log2CacheLineSize = 7;
  const unsigned CacheLineSize = 1u << Log2CacheLineSize;

  // Extra padding amount in bytes to support prefetch mode 3.
  unsigned FillSize = 3 * CacheLineSize;


  MCStreamer &OS = getStreamer();
  OS.pushSection();
  OS.emitValueToAlignment(Align(CacheLineSize), Encoded_pad, 4);
  for (unsigned I = 0; I < FillSize; I += 4)
    OS.emitInt32(Encoded_pad);
  OS.popSection();
  return true;
#endif   
}

void RVGPUTargetELFStreamer::EmitRvhsaKernelDescriptor(
    const MCSubtargetInfo &STI, StringRef KernelName,
    const rvhsa::kernel_descriptor_t &KernelDescriptor, uint64_t NextVGPR,
    uint64_t NextSGPR, bool ReserveVCC, bool ReserveFlatScr,
    unsigned CodeObjectVersion) {
  auto &Streamer = getStreamer();
  auto &Context = Streamer.getContext();

  MCSymbolELF *KernelCodeSymbol = cast<MCSymbolELF>(
      Context.getOrCreateSymbol(Twine(KernelName)));
  MCSymbolELF *KernelDescriptorSymbol = cast<MCSymbolELF>(
      Context.getOrCreateSymbol(Twine(KernelName) + Twine(".kd")));

  // Copy kernel descriptor symbol's binding, other and visibility from the
  // kernel code symbol.
  KernelDescriptorSymbol->setBinding(KernelCodeSymbol->getBinding());
  KernelDescriptorSymbol->setOther(KernelCodeSymbol->getOther());
  KernelDescriptorSymbol->setVisibility(KernelCodeSymbol->getVisibility());
  // Kernel descriptor symbol's type and size are fixed.
  KernelDescriptorSymbol->setType(ELF::STT_OBJECT);
  KernelDescriptorSymbol->setSize(
      MCConstantExpr::create(sizeof(KernelDescriptor), Context));

  // The visibility of the kernel code symbol must be protected or less to allow
  // static relocations from the kernel descriptor to be used.
  if (KernelCodeSymbol->getVisibility() == ELF::STV_DEFAULT)
    KernelCodeSymbol->setVisibility(ELF::STV_PROTECTED);

  Streamer.emitLabel(KernelDescriptorSymbol);
  Streamer.emitInt32(KernelDescriptor.group_segment_fixed_size);
  Streamer.emitInt32(KernelDescriptor.private_segment_fixed_size);
  Streamer.emitInt32(KernelDescriptor.kernarg_size);
#if 0
  for (uint8_t Res : KernelDescriptor.reserved0)
    Streamer.emitInt8(Res);

  // FIXME: Remove the use of VK_RVGPU_REL64 in the expression below. The
  // expression being created is:
  //   (start of kernel code) - (start of kernel descriptor)
  // It implies R_RVGPU_REL64, but ends up being R_RVGPU_ABS64.
  Streamer.emitValue(MCBinaryExpr::createSub(
      MCSymbolRefExpr::create(
          KernelCodeSymbol, MCSymbolRefExpr::VK_RVGPU_REL64, Context),
      MCSymbolRefExpr::create(
          KernelDescriptorSymbol, MCSymbolRefExpr::VK_None, Context),
      Context),
      sizeof(KernelDescriptor.kernel_code_entry_byte_offset));
  for (uint8_t Res : KernelDescriptor.reserved1)
    Streamer.emitInt8(Res);
  Streamer.emitInt32(KernelDescriptor.compute_pgm_rsrc3);
  Streamer.emitInt32(KernelDescriptor.compute_pgm_rsrc1);
  Streamer.emitInt32(KernelDescriptor.compute_pgm_rsrc2);
  Streamer.emitInt16(KernelDescriptor.kernel_code_properties);
  Streamer.emitInt16(KernelDescriptor.kernarg_preload);
  for (uint8_t Res : KernelDescriptor.reserved3)
    Streamer.emitInt8(Res);
#endif     
}
