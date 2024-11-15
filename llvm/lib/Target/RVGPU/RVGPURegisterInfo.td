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

def sub0 : SubRegIndex<32, 0>;
def sub1 : SubRegIndex<32, 32>;
def sub2 : SubRegIndex<32, 64>;
def sub3 : SubRegIndex<32, 128>;

def sub0_lo16 : ComposedSubRegIndex<!cast<SubRegIndex>(sub0), lo16>;
def sub0_hi16 : ComposedSubRegIndex<!cast<SubRegIndex>(sub0), hi16>;
def sub1_lo16 : ComposedSubRegIndex<!cast<SubRegIndex>(sub1), lo16>;
def sub1_hi16 : ComposedSubRegIndex<!cast<SubRegIndex>(sub1), hi16>;
def sub2_lo16 : ComposedSubRegIndex<!cast<SubRegIndex>(sub2), lo16>;
def sub2_hi16 : ComposedSubRegIndex<!cast<SubRegIndex>(sub2), hi16>;
def sub3_lo16 : ComposedSubRegIndex<!cast<SubRegIndex>(sub3), lo16>;
def sub3_hi16 : ComposedSubRegIndex<!cast<SubRegIndex>(sub3), hi16>;

foreach Size = {2, 3, 4} in {
  foreach Index = !range(!sub(4, Size)) in {
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

  list<SubRegIndex> ret = !if(!eq(size, 2), ret2,
                              !if(!eq(size, 3), ret3, ret4));
}

// Generates list of sequential register tuple names.
// E.g. RegSeq<3,2,2,"s">.ret -> [ "s[0:1]", "s[2:3]" ]
class RegSeqNames<int last_reg, int stride, int size, string prefix,
                  int start = 0> {
  int next = !add(start, stride);
  int end_reg = !add(start, size, -1);
  list<string> ret =
    !if(!le(end_reg, last_reg),
        !listconcat([prefix # start],
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

class RVGPURegisterTuples<list<SubRegIndex> Indices, RegisterClass RC,
                       int last_reg, int stride, int size, string prefix> :
  RegisterTuples<Indices,
                 RegSeqDags<RC, last_reg, stride, size>.ret,
                 RegSeqNames<last_reg, stride, size, prefix>.ret>;

//===----------------------------------------------------------------------===//
//  Declarations that describe the SI registers
//===----------------------------------------------------------------------===//
class RVGPUReg <string n, bits<8> regIdx = 0, bit isHi = 0> : Register<n> {
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

//===----------------------------------------------------------------------===//
//  Registers
//===----------------------------------------------------------------------===//

multiclass RVGPURegLoHi16 <string n, bits<8> regIdx, bit ArtificialHigh = 1> {
  def _LO16 : RVGPUReg<n#".l", regIdx >;
  def _HI16 : RVGPUReg<!if(ArtificialHigh, "", n#".h"), regIdx,
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

// GPR registers
foreach Index = 0...127 in {
  defm GPR_#Index : RVGPURegLoHi16 <"r"#Index, Index, 0>;
}

// Pseudo-registers: Used as placeholders during isel and immediately
// replaced, never seeing the verifier.
def FP_REG : RVGPUReg<"fp", 0>;
def SP_REG : RVGPUReg<"sp", 0>;

class RegisterTypes<list<ValueType> reg_types> {
  list<ValueType> types = reg_types;
}

def Reg16Types : RegisterTypes<[i16, f16, bf16]>;
def Reg32Types : RegisterTypes<[i32, f32, v2i16, v2f16, v2bf16, v4i8, p2, p3, p5, p6]>;

def GPR_LO16 : RVGPURegClass<Reg16Types.types, 16, (add (sequence "GPR_%u_LO16", 0, 127))> {
  let AllocationPriority = 0;
  let Size = 16;
  let GeneratePressureSet = 0;
}

def GPR_HI16 : RVGPURegClass<Reg16Types.types, 16, (add (sequence "GPR_%u_HI16", 0, 127))> {
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
def VReg64 : RVGPURegisterTuples<getSubRegs<2>.ret, GPR32, 127, 1, 2, "dr">;

// GPR 96-bit registers
def VReg96 : RVGPURegisterTuples<getSubRegs<3>.ret, GPR32, 127, 1, 3, "tr">;

// GPR 128-bit registers
def VReg128 : RVGPURegisterTuples<getSubRegs<4>.ret, GPR32, 127, 1, 4, "qr">;


// Register class for all vector registers (VGPRs + Interpolation Registers)
class RegClassBase<int numRegs, list<ValueType> regTypes, dag regList> :
    RVGPURegClass<regTypes, 32, regList> {
  let Size = !mul(numRegs, 32);

  // Requires n mov_b32 to copy
  let CopyCost = numRegs;
  let AllocationPriority = !sub(numRegs, 1);
  let Weight = numRegs;
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

defm GPR64 : VRegClass<2, [i64, f64, v2i32, v2f32, v4f16, v4i16, p0, p1, p4], (add VReg64)>;
defm GPR96 : VRegClass<3, [v3i32, v3f32], (add VReg96)>;
defm GPR128 : VRegClass<4, [v4i32, v4f32, v2i64, v2f64, v8i16, v8f16], (add VReg128)>;

// 定义1个Predicate Register
def PREG : RVGPUReg<"%p">;

// 定义 GR0 ~ GR127，共128个32位的通用寄存器
foreach i = 0...31 in {
  def ENVREG#i : RVGPUReg<"%sreg"#i>;
}

//===----------------------------------------------------------------------===//
//  Register classes: 为128个通用寄存器定义
//===----------------------------------------------------------------------===//
def Int1Regs : RVGPURegClass<[i1], 32, (add PREG)>;

// Read RVGPURegisterInfo.cpp to see how VRFrame and VRDepot are used.
def SpecialRegs : RVGPURegClass<[i32], 32, (add (sequence "ENVREG%u", 0, 31))>;
