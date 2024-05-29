//===-- RVGPUAsmPrinter.cpp - RVGPU assembly printer --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// The RVGPUAsmPrinter is used to print both assembly string and also binary
/// code.  When passed an MCAsmStreamer it prints assembly and when passed
/// an MCObjectStreamer it outputs binary code.
//
//===----------------------------------------------------------------------===//
//

#include "RVGPUAsmPrinter.h"
#include "RVGPU.h"
#include "RVGPUHSAMetadataStreamer.h"
#include "RVGPUResourceUsageAnalysis.h"
#include "RVKernelCodeT.h"
#include "RVGPUSubtarget.h"
#include "MCTargetDesc/RVGPUInstPrinter.h"
#include "MCTargetDesc/RVGPUTargetStreamer.h"
#include "RVGPUMachineFunctionInfo.h"
#include "TargetInfo/RVGPUTargetInfo.h"
#include "Utils/RVGPUBaseInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/RVHSAKernelDescriptor.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/TargetParser.h"

using namespace llvm;
using namespace llvm::RVGPU;

// This should get the default rounding mode from the kernel. We just set the
// default here, but this could change if the OpenCL rounding mode pragmas are
// used.
//
// The denormal mode here should match what is reported by the OpenCL runtime
// for the CL_FP_DENORM bit from CL_DEVICE_{HALF|SINGLE|DOUBLE}_FP_CONFIG, but
// can also be override to flush with the -cl-denorms-are-zero compiler flag.
//
// OpenCL only sets flush none and reports CL_FP_DENORM for double
// precision, and leaves single precision to flush all and does not report
// CL_FP_DENORM for CL_DEVICE_SINGLE_FP_CONFIG. Mesa's OpenCL currently reports
// CL_FP_DENORM for both.
//
// FIXME: It seems some instructions do not support single precision denormals
// regardless of the mode (exp_*_f32, rcp_*_f32, rsq_*_f32, rsq_*f32, sqrt_f32,
// and sin_f32, cos_f32 on most parts).

// We want to use these instructions, and using fp32 denormals also causes
// instructions to run at the double precision rate for the device so it's
// probably best to just report no single precision denormals.
static uint32_t getFPMode(RVModeRegisterDefaults Mode) {
  return FP_ROUND_MODE_SP(FP_ROUND_ROUND_TO_NEAREST) |
         FP_ROUND_MODE_DP(FP_ROUND_ROUND_TO_NEAREST) |
         FP_DENORM_MODE_SP(Mode.fpDenormModeSPValue()) |
         FP_DENORM_MODE_DP(Mode.fpDenormModeDPValue());
}

static AsmPrinter *
createRVGPUAsmPrinterPass(TargetMachine &tm,
                           std::unique_ptr<MCStreamer> &&Streamer) {
  return new RVGPUAsmPrinter(tm, std::move(Streamer));
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeRVGPUAsmPrinter() {
  TargetRegistry::RegisterAsmPrinter(getTheRVGPUTarget64(),
                                     createRVGPUAsmPrinterPass);
}

RVGPUAsmPrinter::RVGPUAsmPrinter(TargetMachine &TM,
                                   std::unique_ptr<MCStreamer> Streamer)
    : AsmPrinter(TM, std::move(Streamer)) {
  assert(OutStreamer && "AsmPrinter constructed without streamer");
}

StringRef RVGPUAsmPrinter::getPassName() const {
  return "RVGPU Assembly Printer";
}

const MCSubtargetInfo *RVGPUAsmPrinter::getGlobalSTI() const {
  return TM.getMCSubtargetInfo();
}

RVGPUTargetStreamer* RVGPUAsmPrinter::getTargetStreamer() const {
  if (!OutStreamer)
    return nullptr;
  return static_cast<RVGPUTargetStreamer*>(OutStreamer->getTargetStreamer());
}

void RVGPUAsmPrinter::emitStartOfAsmFile(Module &M) {
  IsTargetStreamerInitialized = false;
}
/*
void RVGPUAsmPrinter::printOperand(const MachineInstr *MI, unsigned OpNum,
                                   raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(OpNum);
  switch (MO.getType()) {
  case MachineOperand::MO_Register:
   //     O << NVPTXInstPrinter::getRegisterName(MO.getReg());
    RVGPUInstPrinter::printRegOperand(MO.getReg(), O,
                                       *MF->getSubtarget().getRegisterInfo());
    break;

  case MachineOperand::MO_Immediate:
    O << MO.getImm();
    break;

  case MachineOperand::MO_MachineBasicBlock:
    MO.getMBB()->getSymbol()->print(O, MAI);
    break;
  default:
    llvm_unreachable("Operand type not supported.");
  }
}

void RVGPUAsmPrinter::printMemOperand(const MachineInstr *MI, unsigned OpNum,
                                      raw_ostream &O, const char *Modifier) {
  printOperand(MI, OpNum, O);

  if (Modifier && strcmp(Modifier, "add") == 0) {
    O << ", ";
    printOperand(MI, OpNum + 1, O);
  } else {
    if (MI->getOperand(OpNum + 1).isImm() &&
        MI->getOperand(OpNum + 1).getImm() == 0)
      return; // don't print ',0' or '+0'
    O << "+";
    printOperand(MI, OpNum + 1, O);
  }
}
*/
void RVGPUAsmPrinter::initTargetStreamer(Module &M) {
  IsTargetStreamerInitialized = true;

  // TODO: Which one is called first, emitStartOfAsmFile or
  // emitFunctionBodyStart?
  if (getTargetStreamer() && !getTargetStreamer()->getTargetID())
    initializeTargetID(M);

  getTargetStreamer()->EmitDirectiveRVGPUTarget();

  HSAMetadataStream->begin(M, *getTargetStreamer()->getTargetID());
}

void RVGPUAsmPrinter::emitEndOfAsmFile(Module &M) {
  // Init target streamer if it has not yet happened
  if (!IsTargetStreamerInitialized)
    initTargetStreamer(M);

  // Emit HSA Metadata (NT_AMD_RVGPU_HSA_METADATA).
  // Emit HSA Metadata (NT_AMD_HSA_METADATA).
  HSAMetadataStream->end();
  bool Success = HSAMetadataStream->emitTo(*getTargetStreamer());
  (void)Success;
  assert(Success && "Malformed HSA Metadata");
}

void RVGPUAsmPrinter::emitFunctionBodyStart() {
  const RVGPUMachineFunctionInfo &MFI = *MF->getInfo<RVGPUMachineFunctionInfo>();
  const RVGPUSubtarget &STM = MF->getSubtarget<RVGPUSubtarget>();
  const Function &F = MF->getFunction();

  // TODO: Which one is called first, emitStartOfAsmFile or
  // emitFunctionBodyStart?
  if (!getTargetStreamer()->getTargetID())
    initializeTargetID(*F.getParent());

  if (!MFI.isEntryFunction())
    return;

  HSAMetadataStream->emitKernel(*MF, CurrentProgramInfo);

}

void RVGPUAsmPrinter::emitFunctionBodyEnd() {
  const RVGPUMachineFunctionInfo &MFI = *MF->getInfo<RVGPUMachineFunctionInfo>();
  if (!MFI.isEntryFunction())
    return;

  auto &Streamer = getTargetStreamer()->getStreamer();
  auto &Context = Streamer.getContext();
  auto &ObjectFileInfo = *Context.getObjectFileInfo();
  auto &ReadOnlySection = *ObjectFileInfo.getReadOnlySection();

  Streamer.pushSection();
  Streamer.switchSection(&ReadOnlySection);

  // CP microcode requires the kernel descriptor to be allocated on 64 byte
  // alignment.
  Streamer.emitValueToAlignment(Align(64), 0, 1, 0);
  ReadOnlySection.ensureMinAlignment(Align(64));

  const RVGPUSubtarget &STM = MF->getSubtarget<RVGPUSubtarget>();

  SmallString<128> KernelName;
  getNameWithPrefix(KernelName, &MF->getFunction());
  getTargetStreamer()->EmitRvhsaKernelDescriptor(
      STM, KernelName, getRvhsaKernelDescriptor(*MF, CurrentProgramInfo),
      CurrentProgramInfo.NumVGPRsForWavesPerEU,
      CurrentProgramInfo.VCCUsed, CurrentProgramInfo.FlatUsed);

  Streamer.popSection();
}

void RVGPUAsmPrinter::emitImplicitDef(const MachineInstr *MI) const {
  Register RegNo = MI->getOperand(0).getReg();

  SmallString<128> Str;
  raw_svector_ostream OS(Str);
  OS << "implicit-def: "
     << printReg(RegNo, MF->getSubtarget().getRegisterInfo());

  OutStreamer->AddComment(OS.str());
  OutStreamer->addBlankLine();
}

void RVGPUAsmPrinter::emitFunctionEntryLabel() {
    AsmPrinter::emitFunctionEntryLabel();
}

void RVGPUAsmPrinter::emitBasicBlockStart(const MachineBasicBlock &MBB) {
  if (DumpCodeInstEmitter && !isBlockOnlyReachableByFallthrough(&MBB)) {
    // Write a line for the basic block label if it is not only fallthrough.
    DisasmLines.push_back(
        (Twine("BB") + Twine(getFunctionNumber())
         + "_" + Twine(MBB.getNumber()) + ":").str());
    DisasmLineMaxLen = std::max(DisasmLineMaxLen, DisasmLines.back().size());
    HexLines.push_back("");
  }
  AsmPrinter::emitBasicBlockStart(MBB);
}

void RVGPUAsmPrinter::emitGlobalVariable(const GlobalVariable *GV) {
  if (GV->getAddressSpace() == RVGPUAS::LOCAL_ADDRESS) {
    if (GV->hasInitializer() && !isa<UndefValue>(GV->getInitializer())) {
      OutContext.reportError({},
                             Twine(GV->getName()) +
                                 ": unsupported initializer for address space");
      return;
    }
    return;
  }

  AsmPrinter::emitGlobalVariable(GV);
}

bool RVGPUAsmPrinter::doInitialization(Module &M) {
  HSAMetadataStream.reset(new HSAMD::MetadataStreamerMsgPackV5());
  return AsmPrinter::doInitialization(M);
}

bool RVGPUAsmPrinter::doFinalization(Module &M) {
  const MCSubtargetInfo &STI = *getGlobalSTI(); 
  OutStreamer->switchSection(getObjFileLowering().getTextSection());                                                                                                                                        
  getTargetStreamer()->EmitCodeEnd(STI);
    
  return AsmPrinter::doFinalization(M);
}

// Print comments that apply to both callable functions and entry points.
void RVGPUAsmPrinter::emitCommonFunctionComments(
    uint32_t NumVGPR, uint32_t TotalNumVGPR,
     uint64_t ScratchSize, uint64_t CodeSize,
    const RVGPUMachineFunctionInfo *MFI) {
  OutStreamer->emitRawComment(" codeLenInByte = " + Twine(CodeSize), false);
  OutStreamer->emitRawComment(" NumVgprs: " + Twine(NumVGPR), false);
  OutStreamer->emitRawComment(" ScratchSize: " + Twine(ScratchSize), false);
//  OutStreamer->emitRawComment(" MemoryBound: " + Twine(MFI->isMemoryBound()), false);
}

uint16_t RVGPUAsmPrinter::getRvhsaKernelCodeProperties(
    const MachineFunction &MF) const {
  const RVGPUMachineFunctionInfo &MFI = *MF.getInfo<RVGPUMachineFunctionInfo>();
  uint16_t KernelCodeProperties = 0;
  KernelCodeProperties |=
        rvhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;

  return KernelCodeProperties;
}

rvhsa::kernel_descriptor_t RVGPUAsmPrinter::getRvhsaKernelDescriptor(
    const MachineFunction &MF,
    const RVProgramInfo &PI) const {
  const RVGPUSubtarget &STM = MF.getSubtarget<RVGPUSubtarget>();
  const Function &F = MF.getFunction();
  const RVGPUMachineFunctionInfo *Info = MF.getInfo<RVGPUMachineFunctionInfo>();

  rvhsa::kernel_descriptor_t KernelDescriptor;
  memset(&KernelDescriptor, 0x0, sizeof(KernelDescriptor));

  assert(isUInt<32>(PI.ScratchSize));
  //assert(isUInt<32>(PI.getComputePGMRSrc1(STM)));
  //assert(isUInt<32>(PI.getComputePGMRSrc2()));

  KernelDescriptor.group_segment_fixed_size = PI.LDSSize;
  KernelDescriptor.private_segment_fixed_size = PI.ScratchSize;

  Align MaxKernArgAlign;
  KernelDescriptor.kernarg_size = STM.getKernArgSegmentSize(F, MaxKernArgAlign);

  KernelDescriptor.kernel_code_properties = getRvhsaKernelCodeProperties(MF);
#if 0
  if (RVGPU::hasKernargPreload(STM))
    KernelDescriptor.kernarg_preload =
        static_cast<uint16_t>(Info->getNumKernargPreloadedSGPRs());
#endif 
  return KernelDescriptor;
}

bool RVGPUAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  // Init target streamer lazily on the first function so that previous passes
  // can set metadata.
  if (!IsTargetStreamerInitialized)
    initTargetStreamer(*MF.getFunction().getParent());

  ResourceUsage = &getAnalysis<RVGPUResourceUsageAnalysis>();
  CurrentProgramInfo = RVProgramInfo();

  const RVGPUMachineFunctionInfo *MFI = MF.getInfo<RVGPUMachineFunctionInfo>();

  // The starting address of all shader programs must be 256 bytes aligned.
  // Regular functions just need the basic required instruction alignment.
  MF.setAlignment(MFI->isEntryFunction() ? Align(256) : Align(4));

  SetupMachineFunction(MF);

  const RVGPUSubtarget &STM = MF.getSubtarget<RVGPUSubtarget>();
  MCContext &Context = getObjFileLowering().getContext();

  if (MFI->isModuleEntryFunction()) {
    getRVProgramInfo(CurrentProgramInfo, MF);
  }


  DumpCodeInstEmitter = nullptr;
#if 0    
  if (STM.dumpCode()) {
    // For -dumpcode, get the assembler out of the streamer, even if it does
    // not really want to let us have it. This only works with -filetype=obj.
    bool SaveFlag = OutStreamer->getUseAssemblerInfoForParsing();
    OutStreamer->setUseAssemblerInfoForParsing(true);
    MCAssembler *Assembler = OutStreamer->getAssemblerPtr();
    OutStreamer->setUseAssemblerInfoForParsing(SaveFlag);
    if (Assembler)
      DumpCodeInstEmitter = Assembler->getEmitterPtr();
  }
#endif 
  DisasmLines.clear();
  HexLines.clear();
  DisasmLineMaxLen = 0;

  emitFunctionBody();

  emitResourceUsageRemarks(MF, CurrentProgramInfo, MFI->isModuleEntryFunction(), false);

  if (isVerbose()) {
    MCSectionELF *CommentSection =
        Context.getELFSection(".RVGPU.csdata", ELF::SHT_PROGBITS, 0);
    OutStreamer->switchSection(CommentSection);

    if (!MFI->isEntryFunction()) {
      OutStreamer->emitRawComment(" Function info:", false);
      const RVGPUResourceUsageAnalysis::RVFunctionResourceInfo &Info =
          ResourceUsage->getResourceInfo(&MF.getFunction());
      emitCommonFunctionComments(
          Info.NumVGPR,
          Info.getTotalNumVGPRs(STM),
          Info.PrivateSegmentSize,
          getFunctionCodeSize(MF),
          MFI);
      return false;
    }

    OutStreamer->emitRawComment(" Kernel info:", false);
    emitCommonFunctionComments(
        CurrentProgramInfo.NumArchVGPR,
        CurrentProgramInfo.NumVGPR, 
        CurrentProgramInfo.ScratchSize, 
        getFunctionCodeSize(MF),
        MFI);

    OutStreamer->emitRawComment(
      " FloatMode: " + Twine(CurrentProgramInfo.FloatMode), false);
    OutStreamer->emitRawComment(
      " IeeeMode: " + Twine(CurrentProgramInfo.IEEEMode), false);
    OutStreamer->emitRawComment(
      " LDSByteSize: " + Twine(CurrentProgramInfo.LDSSize) +
      " bytes/workgroup (compile time only)", false);

    OutStreamer->emitRawComment(
      " VGPRBlocks: " + Twine(CurrentProgramInfo.VGPRBlocks), false);

    OutStreamer->emitRawComment(
      " NumVGPRsForWavesPerEU: " +
      Twine(CurrentProgramInfo.NumVGPRsForWavesPerEU), false);

    OutStreamer->emitRawComment(
      " Occupancy: " +
      Twine(CurrentProgramInfo.Occupancy), false);

#if 0
    OutStreamer->emitRawComment(
      " WaveLimiterHint : " + Twine(MFI->needsWaveLimiter()), false);
    OutStreamer->emitRawComment(" COMPUTE_PGM_RSRC2:SCRATCH_EN: " +
                                    Twine(CurrentProgramInfo.ScratchEnable),
                                false);
    OutStreamer->emitRawComment(" COMPUTE_PGM_RSRC2:TRAP_HANDLER: " +
                                    Twine(CurrentProgramInfo.TrapHandlerEnable),
                                false);
    OutStreamer->emitRawComment(" COMPUTE_PGM_RSRC2:TGID_X_EN: " +
                                    Twine(CurrentProgramInfo.TGIdXEnable),
                                false);
    OutStreamer->emitRawComment(" COMPUTE_PGM_RSRC2:TGID_Y_EN: " +
                                    Twine(CurrentProgramInfo.TGIdYEnable),
                                false);
    OutStreamer->emitRawComment(" COMPUTE_PGM_RSRC2:TGID_Z_EN: " +
                                    Twine(CurrentProgramInfo.TGIdZEnable),
                                false);
    OutStreamer->emitRawComment(" COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: " +
                                    Twine(CurrentProgramInfo.TIdIGCompCount),
                                false);
#endif 
  }

  if (DumpCodeInstEmitter) {

    OutStreamer->switchSection(
        Context.getELFSection(".RVGPU.disasm", ELF::SHT_PROGBITS, 0));

    for (size_t i = 0; i < DisasmLines.size(); ++i) {
      std::string Comment = "\n";
      if (!HexLines[i].empty()) {
        Comment = std::string(DisasmLineMaxLen - DisasmLines[i].size(), ' ');
        Comment += " ; " + HexLines[i] + "\n";
      }

      OutStreamer->emitBytes(StringRef(DisasmLines[i]));
      OutStreamer->emitBytes(StringRef(Comment));
    }
  }

  return false;
}

// TODO: Fold this into emitFunctionBodyStart.
void RVGPUAsmPrinter::initializeTargetID(const Module &M) {
  // In the beginning all features are either 'Any' or 'NotSupported',
  // depending on global target features. This will cover empty modules.
  getTargetStreamer()->initializeTargetID(
      *getGlobalSTI(), getGlobalSTI()->getFeatureString(), CodeObjectVersion);
}

uint64_t RVGPUAsmPrinter::getFunctionCodeSize(const MachineFunction &MF) const {
  const RVGPUSubtarget &STM = MF.getSubtarget<RVGPUSubtarget>();
  const RVGPUInstrInfo *TII = STM.getInstrInfo();

  uint64_t CodeSize = 0;

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      // TODO: CodeSize should account for multiple functions.

      // TODO: Should we count size of debug info?
      if (MI.isDebugInstr())
        continue;

      CodeSize += TII->getInstSizeInBytes(MI);
    }
  }

  return CodeSize;
}

void RVGPUAsmPrinter::getRVProgramInfo(RVProgramInfo &ProgInfo,
                                        const MachineFunction &MF) {
  const RVGPUResourceUsageAnalysis::RVFunctionResourceInfo &Info =
      ResourceUsage->getResourceInfo(&MF.getFunction());
  const RVGPUSubtarget &STM = MF.getSubtarget<RVGPUSubtarget>();

  ProgInfo.NumArchVGPR = Info.NumVGPR;
  ProgInfo.NumAccVGPR = Info.NumAGPR;
  ProgInfo.NumVGPR = Info.getTotalNumVGPRs(STM);
  ProgInfo.AccumOffset = alignTo(std::max(1, Info.NumVGPR), 4) / 4 - 1;
  ProgInfo.ScratchSize = Info.PrivateSegmentSize;
  ProgInfo.VCCUsed = Info.UsesVCC;
  ProgInfo.FlatUsed = Info.UsesFlatScratch;
  ProgInfo.DynamicCallStack = Info.HasDynamicallySizedStack || Info.HasRecursion;

  const uint64_t MaxScratchPerWorkitem =
      STM.getMaxWaveScratchSize() / STM.getWavefrontSize();
  if (ProgInfo.ScratchSize > MaxScratchPerWorkitem) {
    DiagnosticInfoStackSize DiagStackSize(MF.getFunction(),
                                          ProgInfo.ScratchSize,
                                          MaxScratchPerWorkitem, DS_Error);
    MF.getFunction().getContext().diagnose(DiagStackSize);
  }

  const RVGPUMachineFunctionInfo *MFI = MF.getInfo<RVGPUMachineFunctionInfo>();

  const Function &F = MF.getFunction();

  unsigned WaveDispatchNumSGPR = 0, WaveDispatchNumVGPR = 0;
  // Adjust number of registers used to meet default/requested minimum/maximum
  // number of waves per execution unit request.
  ProgInfo.NumVGPRsForWavesPerEU = std::max(
    std::max(ProgInfo.NumVGPR, 1u), STM.getMinNumVGPRs(MFI->getMaxWavesPerEU()));
#if 0
  if (MFI->getLDSSize() >
      static_cast<unsigned>(STM.getAddressableLocalMemorySize())) {
    LLVMContext &Ctx = MF.getFunction().getContext();
    DiagnosticInfoResourceLimit Diag(
        MF.getFunction(), "local memory", MFI->getLDSSize(),
        STM.getAddressableLocalMemorySize(), DS_Error);
    Ctx.diagnose(Diag);
  }
#endif 
  ProgInfo.VGPRBlocks = IsaInfo::getNumVGPRBlocks(
      &STM, ProgInfo.NumVGPRsForWavesPerEU);

  const RVModeRegisterDefaults Mode = MFI->getMode();

  // Set the value to initialize FP_ROUND and FP_DENORM parts of the mode
  // register.
  ProgInfo.FloatMode = getFPMode(Mode);

  ProgInfo.IEEEMode = Mode.IEEE;

  // Make clamp modifier on NaN input returns 0.
  ProgInfo.DX10Clamp = Mode.DX10Clamp;

  // LDS is allocated in 128 dword blocks.
  unsigned LDSAlignShift = 9;

//  ProgInfo.VGPRSpill = MFI->getNumSpilledVGPRs();

//  ProgInfo.LDSSize = MFI->getLDSSize();
  ProgInfo.LDSBlocks =
      alignTo(ProgInfo.LDSSize, 1ULL << LDSAlignShift) >> LDSAlignShift;

  // Scratch is allocated in 64-dword or 256-dword blocks.
  unsigned ScratchAlignShift = 8;
  // We need to program the hardware with the amount of scratch memory that
  // is used by the entire wave.  ProgInfo.ScratchSize is the amount of
  // scratch memory used per thread.
  ProgInfo.ScratchBlocks = divideCeil(
      ProgInfo.ScratchSize * STM.getWavefrontSize(), 1ULL << ScratchAlignShift);

  ProgInfo.MemOrdered = 1;

  // 0 = X, 1 = XY, 2 = XYZ
  unsigned TIDIGCompCnt = 0;
  if (MFI->hasWorkItemIDZ())
    TIDIGCompCnt = 2;
  else if (MFI->hasWorkItemIDY())
    TIDIGCompCnt = 1;

  // The private segment wave byte offset is the last of the system SGPRs. We
  // initially assumed it was allocated, and may have used it. It shouldn't harm
  // anything to disable it if we know the stack isn't used here. We may still
  // have emitted code reading it to initialize scratch, but if that's unused
  // reading garbage should be OK.
  ProgInfo.ScratchEnable =
      ProgInfo.ScratchBlocks > 0 || ProgInfo.DynamicCallStack;
  
  ProgInfo.TrapHandlerEnable = 0;
  ProgInfo.TGIdXEnable = MFI->hasWorkGroupIDX();
  ProgInfo.TGIdYEnable = MFI->hasWorkGroupIDY();
  ProgInfo.TGIdZEnable = MFI->hasWorkGroupIDZ();
  ProgInfo.TGSizeEnable = MFI->hasWorkGroupInfo();
  ProgInfo.TIdIGCompCount = TIDIGCompCnt;
  ProgInfo.EXCPEnMSB = 0;
  ProgInfo.LdsSize = 0;
  ProgInfo.EXCPEnable = 0;

  const auto [MinWEU, MaxWEU] =
      RVGPU::getIntegerPairAttribute(F, "rvgpu-waves-per-eu", {0, 0}, true);
  if (ProgInfo.Occupancy < MinWEU) {
    DiagnosticInfoOptimizationFailure Diag(
        F, F.getSubprogram(),
        "failed to meet occupancy target given by 'rvgpu-waves-per-eu' in "
        "'" +
            F.getName() + "': desired occupancy was " + Twine(MinWEU) +
            ", final occupancy is " + Twine(ProgInfo.Occupancy));
    F.getContext().diagnose(Diag);
  }
}

void RVGPUAsmPrinter::EmitProgramInfoSI(const MachineFunction &MF,
                                         const RVProgramInfo &CurrentProgramInfo) {
#if 0
  const RVGPUMachineFunctionInfo *MFI = MF.getInfo<RVGPUMachineFunctionInfo>();
  const RVGPUSubtarget &STM = MF.getSubtarget<RVGPUSubtarget>();
  OutStreamer->emitInt32(R_00B848_COMPUTE_PGM_RSRC1);
  OutStreamer->emitInt32(CurrentProgramInfo.getComputePGMRSrc1(STM));

  OutStreamer->emitInt32(R_00B84C_COMPUTE_PGM_RSRC2);
  OutStreamer->emitInt32(CurrentProgramInfo.getComputePGMRSrc2());

  OutStreamer->emitInt32(R_00B860_COMPUTE_TMPRING_SIZE);
  OutStreamer->emitInt32(S_00B860_WAVESIZE_GFX11Plus(CurrentProgramInfo.ScratchBlocks));

  OutStreamer->emitInt32(R_SPILLED_VGPRS);
  OutStreamer->emitInt32(MFI->getNumSpilledVGPRs());
#endif                                              
}

void RVGPUAsmPrinter::getRvKernelCode(rv_kernel_code_t &Out,
                                        const RVProgramInfo &CurrentProgramInfo,
                                        const MachineFunction &MF) const {
#if 0
  const Function &F = MF.getFunction();
  assert(F.getCallingConv() == CallingConv::RVGPU_KERNEL ||
         F.getCallingConv() == CallingConv::SPIR_KERNEL);

  const RVGPUMachineFunctionInfo *MFI = MF.getInfo<RVGPUMachineFunctionInfo>();
  const RVGPUSubtarget &STM = MF.getSubtarget<RVGPUSubtarget>();

  RVGPU::initDefaultAMDKernelCodeT(Out, &STM);
  Out.compute_pgm_resource_registers =
      CurrentProgramInfo.getComputePGMRSrc1(STM) |
      (CurrentProgramInfo.getComputePGMRSrc2() << 32);
  Out.code_properties |= AMD_CODE_PROPERTY_IS_PTR64;

  if (CurrentProgramInfo.DynamicCallStack)
    Out.code_properties |= AMD_CODE_PROPERTY_IS_DYNAMIC_CALLSTACK;

  AMD_HSA_BITS_SET(Out.code_properties,
                   AMD_CODE_PROPERTY_PRIVATE_ELEMENT_SIZE,
                   getElementByteSizeValue(STM.getMaxPrivateElementSize(true)));

  const GCNUserSGPRUsageInfo &UserSGPRInfo = MFI->getUserSGPRInfo();
  if (UserSGPRInfo.hasPrivateSegmentBuffer()) {
    Out.code_properties |=
      AMD_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER;
  }

  if (UserSGPRInfo.hasDispatchPtr())
    Out.code_properties |= AMD_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR;

  if (UserSGPRInfo.hasKernargSegmentPtr())
    Out.code_properties |= AMD_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR;

  if (UserSGPRInfo.hasDispatchID())
    Out.code_properties |= AMD_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID;

  if (UserSGPRInfo.hasFlatScratchInit())
    Out.code_properties |= AMD_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT;

  if (UserSGPRInfo.hasDispatchPtr())
    Out.code_properties |= AMD_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR;


  Align MaxKernArgAlign;
  Out.kernarg_segment_byte_size = STM.getKernArgSegmentSize(F, MaxKernArgAlign);
  Out.wavefront_sgpr_count = CurrentProgramInfo.NumSGPR;
  Out.workitem_vgpr_count = CurrentProgramInfo.NumVGPR;
  Out.workitem_private_segment_byte_size = CurrentProgramInfo.ScratchSize;
  Out.workgroup_group_segment_byte_size = CurrentProgramInfo.LDSSize;

  // kernarg_segment_alignment is specified as log of the alignment.
  // The minimum alignment is 16.
  // FIXME: The metadata treats the minimum as 4?
  Out.kernarg_segment_alignment = Log2(std::max(Align(16), MaxKernArgAlign));
#endif 
}

bool RVGPUAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                       const char *ExtraCode, raw_ostream &O) {
  // First try the generic code, which knows about modifiers like 'c' and 'n'.
  if (!AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, O))
    return false;

  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0)
      return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    case 'r':
      break;
    default:
      return true;
    }
  }

  // TODO: Should be able to support other operand types like globals.
  const MachineOperand &MO = MI->getOperand(OpNo);
  if (MO.isReg()) {
    RVGPUInstPrinter::printRegOperand(MO.getReg(), O,
                                       *MF->getSubtarget().getRegisterInfo());
    return false;
  } else if (MO.isImm()) {
    int64_t Val = MO.getImm();
    if (RVGPU::isInlinableIntLiteral(Val)) {
      O << Val;
    } else if (isUInt<16>(Val)) {
      O << format("0x%" PRIx16, static_cast<uint16_t>(Val));
    } else if (isUInt<32>(Val)) {
      O << format("0x%" PRIx32, static_cast<uint32_t>(Val));
    } else {
      O << format("0x%" PRIx64, static_cast<uint64_t>(Val));
    }
    return false;
  }
  return true;
}

void RVGPUAsmPrinter::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<RVGPUResourceUsageAnalysis>();
  AU.addPreserved<RVGPUResourceUsageAnalysis>();
  AsmPrinter::getAnalysisUsage(AU);
}

void RVGPUAsmPrinter::emitResourceUsageRemarks(
    const MachineFunction &MF, const RVProgramInfo &CurrentProgramInfo,
    bool isModuleEntryFunction, bool hasMAIInsts) {
  if (!ORE)
    return;

  const char *Name = "kernel-resource-usage";
  const char *Indent = "    ";

  // If the remark is not specifically enabled, do not output to yaml
  LLVMContext &Ctx = MF.getFunction().getContext();
  if (!Ctx.getDiagHandlerPtr()->isAnalysisRemarkEnabled(Name))
    return;

  auto EmitResourceUsageRemark = [&](StringRef RemarkName,
                                     StringRef RemarkLabel, auto Argument) {
    // Add an indent for every line besides the line with the kernel name. This
    // makes it easier to tell which resource usage go with which kernel since
    // the kernel name will always be displayed first.
    std::string LabelStr = RemarkLabel.str() + ": ";
    if (!RemarkName.equals("FunctionName"))
      LabelStr = Indent + LabelStr;

    ORE->emit([&]() {
      return MachineOptimizationRemarkAnalysis(Name, RemarkName,
                                               MF.getFunction().getSubprogram(),
                                               &MF.front())
             << LabelStr << ore::NV(RemarkName, Argument);
    });
  };

  // FIXME: Formatting here is pretty nasty because clang does not accept
  // newlines from diagnostics. This forces us to emit multiple diagnostic
  // remarks to simulate newlines. If and when clang does accept newlines, this
  // formatting should be aggregated into one remark with newlines to avoid
  // printing multiple diagnostic location and diag opts.
  EmitResourceUsageRemark("FunctionName", "Function Name",
                          MF.getFunction().getName());
  EmitResourceUsageRemark("NumVGPR", "VGPRs", CurrentProgramInfo.NumArchVGPR);
  EmitResourceUsageRemark("ScratchSize", "ScratchSize [bytes/lane]",
                          CurrentProgramInfo.ScratchSize);
  StringRef DynamicStackStr =
      CurrentProgramInfo.DynamicCallStack ? "True" : "False";
  EmitResourceUsageRemark("DynamicStack", "Dynamic Stack", DynamicStackStr);
  EmitResourceUsageRemark("Occupancy", "Occupancy [waves/SIMD]",
                          CurrentProgramInfo.Occupancy);
  EmitResourceUsageRemark("VGPRSpill", "VGPRs Spill",
                          CurrentProgramInfo.VGPRSpill);
  if (isModuleEntryFunction)
    EmitResourceUsageRemark("BytesLDS", "LDS Size [bytes/block]",
                            CurrentProgramInfo.LDSSize);
}
