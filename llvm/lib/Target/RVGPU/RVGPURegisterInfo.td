//===-- RVGPURegisterInfo.td - RVGPU Register defs ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the PTX register file
//===----------------------------------------------------------------------===//

let Namespace = "RVGPU" in {

def lo16 : SubRegIndex<16, 0>;
def hi16 : SubRegIndex<16, 16>;

foreach Index = 0...31 in {
  def sub#Index : SubRegIndex<32, !shl(Index, 5)>;
}

foreach Index = 1...31 in {
  def sub#Index#_lo16 : ComposedSubRegIndex<!cast<SubRegIndex>(sub#Index), lo16>;
  def sub#Index#_hi16 : ComposedSubRegIndex<!cast<SubRegIndex>(sub#Index), hi16>;
}

foreach Size = {2...6,8,16} in {
  foreach Index = !range(!sub(33, Size)) in {
    def !interleave(!foreach(cur, !range(Size), "sub"#!add(cur, Index)), "_") :
      SubRegIndex<!mul(Size, 32), !shl(Index, 5)> {
      let CoveringSubRegIndices =
        !foreach(cur, !range(Size), !cast<SubRegIndex>(sub#!add(cur, Index)));
    }
  }
}

}

//===----------------------------------------------------------------------===//
//  Helpers
//===----------------------------------------------------------------------===//

class getSubRegs<int size> {
  list<SubRegIndex> ret2 = [sub0, sub1];
  list<SubRegIndex> ret3 = [sub0, sub1, sub2];
  list<SubRegIndex> ret4 = [sub0, sub1, sub2, sub3];
  list<SubRegIndex> ret5 = [sub0, sub1, sub2, sub3, sub4];
  list<SubRegIndex> ret6 = [sub0, sub1, sub2, sub3, sub4, sub5];
  list<SubRegIndex> ret7 = [sub0, sub1, sub2, sub3, sub4, sub5, sub6];
  list<SubRegIndex> ret8 = [sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7];
  list<SubRegIndex> ret9 = [sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8];
  list<SubRegIndex> ret10 = [sub0, sub1, sub2, sub3,
                             sub4, sub5, sub6, sub7,
                             sub8, sub9];
  list<SubRegIndex> ret11 = [sub0, sub1, sub2, sub3,
                             sub4, sub5, sub6, sub7,
                             sub8, sub9, sub10];
  list<SubRegIndex> ret12 = [sub0, sub1, sub2, sub3,
                             sub4, sub5, sub6, sub7,
                             sub8, sub9, sub10, sub11];
  list<SubRegIndex> ret16 = [sub0, sub1, sub2, sub3,
                             sub4, sub5, sub6, sub7,
                             sub8, sub9, sub10, sub11,
                             sub12, sub13, sub14, sub15];
  list<SubRegIndex> ret32 = [sub0, sub1, sub2, sub3,
                             sub4, sub5, sub6, sub7,
                             sub8, sub9, sub10, sub11,
                             sub12, sub13, sub14, sub15,
                             sub16, sub17, sub18, sub19,
                             sub20, sub21, sub22, sub23,
                             sub24, sub25, sub26, sub27,
                             sub28, sub29, sub30, sub31];

  list<SubRegIndex> ret = !if(!eq(size, 2), ret2,
                              !if(!eq(size, 3), ret3,
                                  !if(!eq(size, 4), ret4,
                                      !if(!eq(size, 5), ret5,
                                          !if(!eq(size, 6), ret6,
                                              !if(!eq(size, 7), ret7,
                                                  !if(!eq(size, 8), ret8,
                                                      !if(!eq(size, 9), ret9,
                                                          !if(!eq(size, 10), ret10,
                                                              !if(!eq(size, 11), ret11,
                                                                  !if(!eq(size, 12), ret12,
                                                                      !if(!eq(size, 16), ret16,
                                                                          ret32))))))))))));
}

// Generates list of sequential register tuple names.
// E.g. RegSeq<3,2,2,"s">.ret -> [ "s[0:1]", "s[2:3]" ]
class RegSeqNames<int last_reg, int stride, int size, string prefix,
                  int start = 0> {
  int next = !add(start, stride);
  int end_reg = !add(start, size, -1);
  list<string> ret =
    !if(!le(end_reg, last_reg),
        !listconcat([prefix # "[" # start # ":" # end_reg # "]"],
                    RegSeqNames<last_reg, stride, size, prefix, next>.ret),
                    []);
}

// Generates list of dags for register tuples.
class RegSeqDags<RegisterClass RC, int last_reg, int stride, int size,
                int start = 0> {
  dag trunc_rc = (trunc RC,
                  !if(!and(!eq(stride, 1), !eq(start, 0)),
                      !sub(!add(last_reg, 2), size),
                      !add(last_reg, 1)));
  list<dag> ret =
    !if(!lt(start, size),
        !listconcat([(add (decimate (shl trunc_rc, start), stride))],
                    RegSeqDags<RC, last_reg, stride, size, !add(start, 1)>.ret),
        []);
}

class RVRegisterTuples<list<SubRegIndex> Indices, RegisterClass RC,
                       int last_reg, int stride, int size, string prefix> :
  RegisterTuples<Indices,
                 RegSeqDags<RC, last_reg, stride, size>.ret,
                 RegSeqNames<last_reg, stride, size, prefix>.ret>;

/*
class RVGPUReg<string n> : Register<n> {
  let Namespace = "RVGPU";
}
*/
//===----------------------------------------------------------------------===//
//  Declarations that describe the SI registers
//===----------------------------------------------------------------------===//
class RVReg <string n, bits<8> regIdx = 0, bit isHi = 0> : Register<n> {
  let Namespace = "RVGPU";

  // These are generic helper values we use to form actual register
  // codes. They should not be assumed to match any particular register
  // encodings on any particular subtargets.
  let HWEncoding{7-0} = regIdx;
  let HWEncoding{8} = 0b1;
  let HWEncoding{9} = isHi;

  int Index = !cast<int>(regIdx);
}
// For register classes that use TSFlags.
class RVGPURegClass <list<ValueType> rTypes, int Align, dag rList>
  : RegisterClass <"RVGPU", rTypes, Align, rList> {
  // Alignment of the first register in tuple (in 32-bit units).
  field int RegTupleAlignUnits = 1;

  // These need to be kept in sync with the enum SIRCFlags.
  let TSFlags{1-0} = RegTupleAlignUnits;
}

//class RVGPURegClass<list<ValueType> regTypes, int alignment, dag regList>
//     : RegisterClass <"RVGPU", regTypes, alignment, regList>;

//===----------------------------------------------------------------------===//
//  Registers
//===----------------------------------------------------------------------===//

multiclass RVRegLoHi16 <string n, bits<8> regIdx, bit ArtificialHigh = 1> {
  def _LO16 : RVReg<n#".l", regIdx >;
  def _HI16 : RVReg<!if(ArtificialHigh, "", n#".h"), regIdx,
                    /* isHi */ 1> {
    let isArtificial = ArtificialHigh;
  }
  def "" : RegisterWithSubRegs<n, [!cast<Register>(NAME#"_LO16"),
                                   !cast<Register>(NAME#"_HI16")]> {
    let Namespace = "RVGPU";
    let SubRegIndices = [lo16, hi16];
    let CoveredBySubRegs = !not(ArtificialHigh);
    let HWEncoding{7-0} = regIdx;

    int Index = !cast<int>(regIdx);
  }
}
// Special Registers
defm VCC_LO : RVRegLoHi16<"vcc_lo", 106>;
defm VCC_HI : RVRegLoHi16<"vcc_hi", 107>;

// VCC for 64-bit instructions
def VCC : RegisterWithSubRegs<"vcc", [VCC_LO, VCC_HI]> {
  let Namespace = "RVGPU";
  let SubRegIndices = [sub0, sub1];
  let HWEncoding = VCC_LO.HWEncoding;
}

defm EXEC_LO : RVRegLoHi16<"exec_lo", 126>, DwarfRegNum<[1, 1]>;
defm EXEC_HI : RVRegLoHi16<"exec_hi", 127>;

def EXEC : RegisterWithSubRegs<"exec", [EXEC_LO, EXEC_HI]>, DwarfRegNum<[17, 1]> {
  let Namespace = "RVGPU";
  let SubRegIndices = [sub0, sub1];
  let HWEncoding = EXEC_LO.HWEncoding;
}
// GPR registers
foreach Index = 0...127 in {
  defm GPR_#Index :
    RVRegLoHi16 <"r"#Index, Index, 0>,
    DwarfRegNum<[!add(Index, 2560), !add(Index, 1536)]>;
}

// 32-bit real registers, for MC only.
// May be used with both 32-bit and 64-bit operands.
defm SRC_VCCZ : RVRegLoHi16<"src_vccz", 251>;
defm SRC_EXECZ : RVRegLoHi16<"src_execz", 252>;
defm SRC_SCC : RVRegLoHi16<"src_scc", 253>;

def SReg_1 : RVGPURegClass<[i1], 32,
  (add EXEC, EXEC_LO, EXEC_HI)> {
  let CopyCost = 1;
  let isAllocatable = 0;
}

// Pseudo-registers: Used as placeholders during isel and immediately
// replaced, never seeing the verifier.
def PRIVATE_RSRC_REG : RVReg<"private_rsrc", 0>;
def FP_REG : RVReg<"fp", 0>;
def SP_REG : RVReg<"sp", 0>;

class RegisterTypes<list<ValueType> reg_types> {
  list<ValueType> types = reg_types;
}

def Reg16Types : RegisterTypes<[i16, f16, bf16]>;
def Reg32Types : RegisterTypes<[i32, f32, v2i16, v2f16, v2bf16, 
                                v4i8, p2, p3, p5, p6]>;

def GPR_LO16 : RVGPURegClass<Reg16Types.types, 16,
                              (add (sequence "GPR_%u_LO16", 0, 127))> {
  let AllocationPriority = 0;
  let Size = 16;
  let GeneratePressureSet = 0;
}

def GPR_HI16 : RVGPURegClass<Reg16Types.types, 16,
                              (add (sequence "GPR_%u_HI16", 0, 127))> {
  let AllocationPriority = 0;
  let Size = 16;
  let GeneratePressureSet = 0;
}

def GPR16 : RVGPURegClass<Reg16Types.types, 16,
                            (add (interleave (sequence "GPR_%u_LO16", 0, 127),
                                             (sequence "GPR_%u_HI16", 0, 127)))> {
  let AllocationPriority = 2;
  let Size = 16;
  let GeneratePressureSet = 0;

  // This is the base class for GPR{128..255}_{LO16,HI16}.
  let BaseClassOrder = 17;
}
// VOP1/2/C can access the First 128 lo and 128 hi registers.
// The order of registers in the class determines order of allocation, so it is
// important to interleave lo and hi registers.
def GPR16_Lo128 : RVGPURegClass<Reg16Types.types, 16,
                            (add (interleave (sequence "GPR_%u_LO16", 0, 127),
                                             (sequence "GPR_%u_HI16", 0, 127)))> {
  let Size = 16;
  let GeneratePressureSet = 0;
  let isAllocatable = 0;

  // This is the base class for GPR{0..127}_{LO16,HI16}.
  let BaseClassOrder = 16;
}

// GPR 32-bit registers
// i16/f16 only on VI+
def GPR32 : RVGPURegClass<!listconcat(Reg32Types.types, Reg16Types.types), 32,
                            (add (sequence "GPR_%u", 0, 127))> {
  let AllocationPriority = 0;
  let Size = 32;
  let Weight = 1;
  let BaseClassOrder = 32;
}

// Identical to GPR32 except it only contains the low 128 (Lo128) registers.
def GPR32_Lo128 : RVGPURegClass<!listconcat(Reg32Types.types, Reg16Types.types), 32,
                            (add (sequence "GPR_%u", 0, 127))> {
  let AllocationPriority = 0;
  let GeneratePressureSet = 0;
  let Size = 32;
  let Weight = 1;
}

// GPR 64-bit registers
def VReg64 : RVRegisterTuples<getSubRegs<2>.ret, GPR32, 127, 1, 2, "r">;

// GPR 96-bit registers
def VReg96 : RVRegisterTuples<getSubRegs<3>.ret, GPR32, 127, 1, 3, "r">;

// GPR 128-bit registers
def VReg128 : RVRegisterTuples<getSubRegs<4>.ret, GPR32, 127, 1, 4, "r">;

// GPR 160-bit registers
def VReg160 : RVRegisterTuples<getSubRegs<5>.ret, GPR32, 127, 1, 5, "r">;

// GPR 192-bit registers
def VReg192 : RVRegisterTuples<getSubRegs<6>.ret, GPR32, 127, 1, 6, "r">;

// GPR 224-bit registers
def VReg224 : RVRegisterTuples<getSubRegs<7>.ret, GPR32, 127, 1, 7, "r">;

// GPR 256-bit registers
def VReg256 : RVRegisterTuples<getSubRegs<8>.ret, GPR32, 127, 1, 8, "r">;

// GPR 288-bit registers
def VReg288 : RVRegisterTuples<getSubRegs<9>.ret, GPR32, 127, 1, 9, "r">;

// GPR 320-bit registers
def VReg320 : RVRegisterTuples<getSubRegs<10>.ret, GPR32, 127, 1, 10, "r">;

// GPR 352-bit registers
def VReg352 : RVRegisterTuples<getSubRegs<11>.ret, GPR32, 127, 1, 11, "r">;

// GPR 384-bit registers
def VReg384 : RVRegisterTuples<getSubRegs<12>.ret, GPR32, 127, 1, 12, "r">;

// GPR 512-bit registers
def VReg512 : RVRegisterTuples<getSubRegs<16>.ret, GPR32, 127, 1, 16, "r">;

// GPR 1024-bit registers
def VReg1024 : RVRegisterTuples<getSubRegs<32>.ret, GPR32, 127, 1, 32, "r">;

// Register class for all vector registers (VGPRs + Interpolation Registers)
class RegClassBase<int numRegs, list<ValueType> regTypes, dag regList> :
    RVGPURegClass<regTypes, 32, regList> {
  let Size = !mul(numRegs, 32);

  // Requires n mov_b32 to copy
  let CopyCost = numRegs;
  let AllocationPriority = !sub(numRegs, 1);
  let Weight = numRegs;
}

def VRegOrLds_32 : RVGPURegClass<[i32, f32, i16, f16, v2i16, v2f16], 32,
                                 (add GPR_32)> {
                                 //(add GPR32, LDS_DIRECT_CLASS)> {
  let isAllocatable = 0;
}

// Define a register tuple class, along with one requiring an even
// aligned base register.
multiclass VRegClass<int numRegs, list<ValueType> regTypes, dag regList> {
    // Define the regular class.
    def "" : RegClassBase<numRegs, regTypes, regList> {
      let BaseClassOrder = !mul(numRegs, 32);
    }

    // Define 2-aligned variant
    def _Align2 : RegClassBase<numRegs, regTypes, (decimate regList, 2)> {
      // Give aligned class higher priority in base class resolution
      let BaseClassOrder = !sub(!mul(numRegs, 32), 1);
      let RegTupleAlignUnits = 2;
    }
}

defm GPR64 : VRegClass<2, [i64, f64, v2i32, v2f32, v4f16, v4i16, p0, p1, p4],
                                (add VReg64)>;
defm GPR96 : VRegClass<3, [v3i32, v3f32], (add VReg96)>;
defm GPR128 : VRegClass<4, [v4i32, v4f32, v2i64, v2f64, v8i16, v8f16], (add VReg128)>;
defm GPR160 : VRegClass<5, [v5i32, v5f32], (add VReg160)>;

defm GPR192 : VRegClass<6, [v6i32, v6f32, v3i64, v3f64], (add VReg192)>;
defm GPR224 : VRegClass<7, [v7i32, v7f32], (add VReg224)>;
defm GPR256 : VRegClass<8, [v8i32, v8f32, v4i64, v4f64, v16i16, v16f16], (add VReg256)>;
defm GPR288 : VRegClass<9, [v9i32, v9f32], (add VReg288)>;
defm GPR320 : VRegClass<10, [v10i32, v10f32], (add VReg320)>;
defm GPR352 : VRegClass<11, [v11i32, v11f32], (add VReg352)>;
defm GPR384 : VRegClass<12, [v12i32, v12f32], (add VReg384)>;

let GlobalPriority = true in {
defm GPR512 : VRegClass<16, [v16i32, v16f32, v8i64, v8f64, v32i16, v32f16], (add VReg512)>;
defm GPR1024 : VRegClass<32, [v32i32, v32f32, v16i64, v16f64], (add VReg1024)>;
}


//===----------------------------------------------------------------------===//
// VGPRSrc_*
//===----------------------------------------------------------------------===//

// An 8-bit RegisterOperand wrapper for a VGPR
def GPRSrc_32 : RegisterOperand<GPR32> {
  let DecoderMethod = "DecodeGPR32RegisterClass";
}
def GPRSrc_32_Lo128 : RegisterOperand<GPR32_Lo128> {
  let DecoderMethod = "DecodeGPR32RegisterClass";
}

def GPRSrc_16_Lo128 : RegisterOperand<GPR16_Lo128> {
  let DecoderMethod = "DecodeGPR16_Lo128RegisterClass";
  let EncoderMethod = "getMachineOpValueT16Lo128";
}

let GeneratePressureSet = 0 in {
// No register should ever be allocated using VReg_1. This is a hack for
// SelectionDAG that should always be lowered by SILowerI1Copies.  TableGen
// sorts register classes based on the number of registers in them so this is
// sorted to the end and not preferred over VGPR_32.
def VReg_1 : RVGPURegClass<[i1], 32, (add)> {
  let Size = 1;
}

def SReg_1_XEXEC : RVGPURegClass<[i1], 32,
  (add GPR32)> {
  let CopyCost = 1;
  let isAllocatable = 0;
}

def VS_16 : RVGPURegClass<Reg16Types.types, 16,
                          (add GPR16)> {
  let isAllocatable = 0;
}

def VS_16_Lo128 : RVGPURegClass<Reg16Types.types, 16,
                          (add GPR16_Lo128)> {
  let isAllocatable = 0;
}

def VS_32 : RVGPURegClass<[i32, f32, i16, f16, v2i16, v2f16], 32,
                          (add GPR32)> {
  let isAllocatable = 0;
}

def VS_32_Lo128 : RVGPURegClass<[i32, f32, i16, f16, v2i16, v2f16], 32,
                          (add GPR32_Lo128)> {
  let isAllocatable = 0;
}

def VS_64 : RVGPURegClass<[i64, f64, v2f32], 32, (add GPR64)> {
  let isAllocatable = 0;
}

def AV_32 : RVGPURegClass<GPR32.RegTypes, 32, (add GPR32)> {
}
} // End GeneratePressureSet = 0

//===----------------------------------------------------------------------===//
//  Register operands
//===----------------------------------------------------------------------===//

class RegImmMatcher<string name> : AsmOperandClass {
  let Name = name;
  let RenderMethod = "addRegOrImmOperands";
}

class RegOrImmOperand <string RegisterClassName, string OperandTypeName,
                       string ParserMatchClassName, string decoderImmSize>
  : RegisterOperand<!cast<RegisterClass>(RegisterClassName)> {
    let OperandNamespace = "RVGPU";
    let OperandType = OperandTypeName;
    let ParserMatchClass = RegImmMatcher<ParserMatchClassName>;
    let DecoderMethod = "decodeOperand_" # RegisterClassName # decoderImmSize;
 }

class RegOrB16 <string RegisterClass, string OperandTypePrefix>
  : RegOrImmOperand <RegisterClass, OperandTypePrefix # "_INT16",
                     !subst("_b16", "B16", NAME), "_Imm16">;

class RegOrF16 <string RegisterClass, string OperandTypePrefix>
  : RegOrImmOperand <RegisterClass, OperandTypePrefix # "_FP16",
                     !subst("_f16", "F16", NAME), "_Imm16">;

class RegOrB16T <string RegisterClass, string OperandTypePrefix>
  : RegOrImmOperand <RegisterClass, OperandTypePrefix # "_INT16",
                     !subst("_b16", "B16", NAME), "_Imm16"> {
  let EncoderMethod = "getMachineOpValueT16";
}

class RegOrF16T <string RegisterClass, string OperandTypePrefix>
  : RegOrImmOperand <RegisterClass, OperandTypePrefix # "_FP16",
                     !subst("_f16", "F16", NAME), "_Imm16"> {
  let EncoderMethod = "getMachineOpValueT16";
}

class RegOrB16_Lo128T <string RegisterClass, string OperandTypePrefix>
  : RegOrImmOperand <RegisterClass, OperandTypePrefix # "_INT16",
                     !subst("_b16_Lo128", "B16_Lo128", NAME), "_Imm16"> {
  let EncoderMethod = "getMachineOpValueT16Lo128";
}

class RegOrF16_Lo128T <string RegisterClass, string OperandTypePrefix>
  : RegOrImmOperand <RegisterClass, OperandTypePrefix # "_FP16",
                     !subst("_f16_Lo128", "F16_Lo128", NAME), "_Imm16"> {
  let EncoderMethod = "getMachineOpValueT16Lo128";
}

class RegOrB32 <string RegisterClass, string OperandTypePrefix>
  : RegOrImmOperand <RegisterClass, OperandTypePrefix # "_INT32",
                     !subst("_b32", "B32", NAME), "_Imm32">;

class RegOrF32 <string RegisterClass, string OperandTypePrefix>
  : RegOrImmOperand <RegisterClass, OperandTypePrefix # "_FP32",
                     !subst("_f32", "F32", NAME), "_Imm32">;

class RegOrV2B16 <string RegisterClass, string OperandTypePrefix>
  : RegOrImmOperand <RegisterClass, OperandTypePrefix # "_V2INT16",
                     !subst("_v2b16", "V2B16", NAME), "_Imm16">;

class RegOrV2F16 <string RegisterClass, string OperandTypePrefix>
  : RegOrImmOperand <RegisterClass, OperandTypePrefix # "_V2FP16",
                     !subst("_v2f16", "V2F16", NAME), "_Imm16">;

class RegOrF64 <string RegisterClass, string OperandTypePrefix>
  : RegOrImmOperand <RegisterClass, OperandTypePrefix # "_FP64",
                     !subst("_f64", "F64", NAME), "_Imm64">;

class RegOrB64 <string RegisterClass, string OperandTypePrefix>
  : RegOrImmOperand <RegisterClass, OperandTypePrefix # "_INT64",
                     !subst("_b64", "B64", NAME), "_Imm64">;

class RegOrV2F32 <string RegisterClass, string OperandTypePrefix>
  : RegOrImmOperand <RegisterClass, OperandTypePrefix # "_V2FP32",
                     !subst("_v2f32", "V2FP32", NAME), "_Imm32">;

class RegOrV2B32 <string RegisterClass, string OperandTypePrefix>
  : RegOrImmOperand <RegisterClass, OperandTypePrefix # "_V2INT32",
                     !subst("_v2b32", "V2INT32", NAME), "_Imm32">;

// For VOP1,2,C True16 instructions. _Lo128 use first 128 32-bit VGPRs only.
class RegOrB16_Lo128 <string RegisterClass, string OperandTypePrefix>
  : RegOrImmOperand <RegisterClass, OperandTypePrefix # "_INT16",
                     !subst("_b16_Lo128", "B16_Lo128", NAME), "_Imm16">;

class RegOrF16_Lo128 <string RegisterClass, string OperandTypePrefix>
  : RegOrImmOperand <RegisterClass, OperandTypePrefix # "_FP16",
                     !subst("_f16_Lo128", "F16_Lo128", NAME), "_Imm16">;

// Deferred operands
class RegOrF16_Deferred <string RegisterClass, string OperandTypePrefix>
  : RegOrImmOperand <RegisterClass, OperandTypePrefix # "_FP16_DEFERRED",
                     !subst("_f16_Deferred", "F16", NAME), "_Deferred_Imm16">;

class RegOrF32_Deferred <string RegisterClass, string OperandTypePrefix>
  : RegOrImmOperand <RegisterClass, OperandTypePrefix # "_FP32_DEFERRED",
                     !subst("_f32_Deferred", "F32", NAME), "_Deferred_Imm32">;

class RegOrF16_Lo128_Deferred <string RegisterClass,
                               string OperandTypePrefix>
  : RegOrImmOperand <RegisterClass, OperandTypePrefix # "_FP16_DEFERRED",
                     !subst("_f16_Lo128_Deferred", "F16_Lo128", NAME),
                     "_Deferred_Imm16">;

//===----------------------------------------------------------------------===//
//  VSrc_* Operands with an VGPR or a 32-bit immediate
//===----------------------------------------------------------------------===//

// The current and temporary future default used case for VOP3.
def VSrc_b16 : RegOrB16 <"VS_32", "OPERAND_REG_IMM">;
def VSrc_f16 : RegOrF16 <"VS_32", "OPERAND_REG_IMM">;

// True16 VOP3 operands.
def VSrcT_b16 : RegOrB16T <"VS_16", "OPERAND_REG_IMM"> {
  let DecoderMethod = "decodeOperand_VSrcT16";
}
def VSrcT_f16 : RegOrF16T <"VS_16", "OPERAND_REG_IMM"> {
  let DecoderMethod = "decodeOperand_VSrcT16";
}

// True16 VOP1/2/C operands.
def VSrcT_b16_Lo128 : RegOrB16_Lo128T <"VS_16_Lo128", "OPERAND_REG_IMM"> {
  let DecoderMethod = "decodeOperand_VSrcT16_Lo128";
}
def VSrcT_f16_Lo128 : RegOrF16_Lo128T <"VS_16_Lo128", "OPERAND_REG_IMM"> {
  let DecoderMethod = "decodeOperand_VSrcT16_Lo128";
}

// The current and temporary future default used case for fake VOP1/2/C.
def VSrcFake16_b16_Lo128 : RegOrB16_Lo128 <"VS_32_Lo128", "OPERAND_REG_IMM">;
def VSrcFake16_f16_Lo128 : RegOrF16_Lo128 <"VS_32_Lo128", "OPERAND_REG_IMM">;

def VSrc_b32 : RegOrB32 <"VS_32", "OPERAND_REG_IMM">;
def VSrc_f32 : RegOrF32 <"VS_32", "OPERAND_REG_IMM">;
def VSrc_v2b16 : RegOrV2B16 <"VS_32", "OPERAND_REG_IMM">;
def VSrc_v2f16 : RegOrV2F16 <"VS_32", "OPERAND_REG_IMM">;
def VSrc_b64 : RegOrB64 <"VS_64", "OPERAND_REG_IMM">;
def VSrc_f64 : RegOrF64 <"VS_64", "OPERAND_REG_IMM"> {
  let DecoderMethod = "decodeOperand_VSrc_f64";
}
def VSrc_v2b32 : RegOrV2B32 <"VS_64", "OPERAND_REG_IMM">;
def VSrc_v2f32 : RegOrV2F32 <"VS_64", "OPERAND_REG_IMM">;

//===----------------------------------------------------------------------===//
//  VSrc_*_Deferred Operands with an SGPR, VGPR or a 32-bit immediate for use
//  with FMAMK/FMAAK
//===----------------------------------------------------------------------===//

def VSrc_f16_Deferred : RegOrF16_Deferred<"VS_32", "OPERAND_REG_IMM">;
def VSrc_f32_Deferred : RegOrF32_Deferred<"VS_32", "OPERAND_REG_IMM">;

def VSrcFake16_f16_Lo128_Deferred : RegOrF16_Lo128_Deferred<"VS_32_Lo128",
                                                            "OPERAND_REG_IMM">;

//===----------------------------------------------------------------------===//
//  VRegSrc_* Operands with a VGPR
//===----------------------------------------------------------------------===//

// This is for operands with the enum(9), VSrc encoding restriction,
// but only allows VGPRs.
def VRegSrc_32 : RegisterOperand<GPR32> {
  let DecoderMethod = "decodeOperand_GPR32";
}

def VRegSrc_64 : RegisterOperand<GPR64> {
  let DecoderMethod = "decodeOperand_VReg_64";
}

def VRegSrc_128 : RegisterOperand<GPR128> {
  let DecoderMethod = "decodeOperand_VReg_128";
}

def VRegSrc_256 : RegisterOperand<GPR256> {
  let DecoderMethod = "decodeOperand_VReg_256";
}

def VRegOrLdsSrc_32 : RegisterOperand<VRegOrLds_32> {
  let DecoderMethod = "decodeOperand_VRegOrLds_32";
}

//===----------------------------------------------------------------------===//
// VGPRSrc_*
//===----------------------------------------------------------------------===//

// An 8-bit RegisterOperand wrapper for a VGPR
def VGPRSrc_32 : RegisterOperand<GPR32> {
  let DecoderMethod = "DecodeGPR32RegisterClass";
}
def VGPRSrc_32_Lo128 : RegisterOperand<GPR32_Lo128> {
  let DecoderMethod = "DecodeGPR32RegisterClass";
}

def VGPRSrc_16_Lo128 : RegisterOperand<GPR16_Lo128> {
  let DecoderMethod = "DecodeGPR_16_Lo128RegisterClass";
  let EncoderMethod = "getMachineOpValueT16Lo128";
}




// Special Registers used as stack pointer
//def VRFrame32         : RVReg<"%SP">;
//def VRFrame64         : RVReg<"%SP">;
//def VRFrameLocal32    : RVReg<"%SPL">;
//def VRFrameLocal64    : RVReg<"%SPL">;

// Special Registers used as the stack
//def VRDepot  : RVReg<"%Depot">;

// 定义1个Predicate Register
def PREG : RVReg<"%p">;

// 定义 GR0 ~ GR127，共128个32位的通用寄存器
/*
foreach i = 0...127 in {
  def R#i : RVReg<"%r" # i>;
}
*/
foreach i = 0...31 in {
  def ENVREG#i : RVReg<"%sreg"#i>;
}

//===----------------------------------------------------------------------===//
//  Register classes: 为128个通用寄存器定义
//===----------------------------------------------------------------------===//
def Int1Regs : RVGPURegClass<[i1], 32, (add PREG)>;
/*

def GPR16: RVGPURegClass<[i16, f16, bf16], 32, (add (sequence "R%u", 0, 127))>;
def GPR32: RVGPURegClass<[i32, f32, v2f16, v2bf16, v2i16, v4i8], 32, (add (sequence "R%u", 0, 127))>;
def GPR64: RVGPURegClass<[i64, f64], 64, (add (sequence "R%u", 0, 127, 2))>;
*/
// Read RVGPURegisterInfo.cpp to see how VRFrame and VRDepot are used.
def SpecialRegs : RVGPURegClass<[i32], 32, (add (sequence "ENVREG%u", 0, 31))>;
