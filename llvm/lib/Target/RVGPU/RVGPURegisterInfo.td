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
  // 分别描述16位，32位，64位，128位寄存器

  def lo16 : SubRegIndex<16, 0>;
  def hi16 : SubRegIndex<16, 16>;

  def sub0 : SubRegIndex<32, 0>;
  def sub1 : SubRegIndex<32, 32>;
  def sub2 : SubRegIndex<32, 64>;
  def sub3 : SubRegIndex<32, 96>;

  def sub0_lo16 : ComposedSubRegIndex<!cast<SubRegIndex>(sub0), lo16>;
  def sub0_hi16 : ComposedSubRegIndex<!cast<SubRegIndex>(sub0), hi16>;
  def sub1_lo16 : ComposedSubRegIndex<!cast<SubRegIndex>(sub1), lo16>;
  def sub1_hi16 : ComposedSubRegIndex<!cast<SubRegIndex>(sub1), hi16>;
  def sub2_lo16 : ComposedSubRegIndex<!cast<SubRegIndex>(sub2), lo16>;
  def sub2_hi16 : ComposedSubRegIndex<!cast<SubRegIndex>(sub2), hi16>;
  def sub3_lo16 : ComposedSubRegIndex<!cast<SubRegIndex>(sub3), lo16>;
  def sub3_hi16 : ComposedSubRegIndex<!cast<SubRegIndex>(sub3), hi16>;

  def sub0_sub1 : SubRegIndex<64, 0>  { let CoveringSubRegIndices = [sub0, sub1]; }
  def sub1_sub2 : SubRegIndex<64, 32> { let CoveringSubRegIndices = [sub1, sub2]; }
  def sub2_sub3 : SubRegIndex<64, 96> { let CoveringSubRegIndices = [sub2, sub3]; }

  def sub0_sub1_sub2 : SubRegIndex<96, 0>  { let CoveringSubRegIndices = [sub0, sub1, sub2]; }
  def sub1_sub2_sub3 : SubRegIndex<96, 32> { let CoveringSubRegIndices = [sub1, sub2, sub3]; }
}

//===----------------------------------------------------------------------===//
//  Helpers
//===----------------------------------------------------------------------===//
// size为2时，表示64位寄存器：[dr0, dr1, dr2, ... , dr126]
// size为3时，表示96位寄存器: [tr0, tr1, tr2, ... , tr125]
// size为4时，表示128位寄存器：[qr0, qr1, qr2, ... , qr124]
class RegSeqNames<int last_reg, int stride, int size, string prefix, int start = 0> {
  int next = !add(start, stride);
  int end_reg = !add(start, size, -1);
  list<string> ret = 
    !if(!le(end_reg, last_reg),
        !listconcat([prefix # start], RegSeqNames<last_reg, stride, size, prefix, next>.ret),
        []);
}

// Generates list of dags for register tuples.
class RegSeqDags<RegisterClass RC, int last_reg, int stride, int size, int start = 0> {
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
//  定义寄存器描述
//===----------------------------------------------------------------------===//
// GPR
class RVGPUGPR <string n, bits<7> regIdx, bit isHi = 0> : Register<n> {
  let Namespace = "RVGPU";

  let HWEncoding{6-0} = regIdx;

  int Index = !cast<int>(regIdx);
}

class RVGPUSREG <string n, bits<7> regIdx = 0, bit isHi = 0> : Register<n> {
  let Namespace = "RVGPU";

  let HWEncoding{6-0} = regIdx;

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

multiclass RVGPURegLoHi16 <string n, bits<7> regIdx, bit ArtificialHigh = 1> {
  def _LO16 : RVGPUGPR<n#".l", regIdx >;
  def _HI16 : RVGPUGPR<!if(ArtificialHigh, "", n#".h"), regIdx,
                    /* isHi */ 1> {
    let isArtificial = ArtificialHigh;
  }
  def "" : RegisterWithSubRegs<n, [!cast<Register>(NAME#"_LO16"),
                                   !cast<Register>(NAME#"_HI16")]> {
    let Namespace = "RVGPU";
    let SubRegIndices = [lo16, hi16];
    let CoveredBySubRegs = !not(ArtificialHigh);
    let HWEncoding{6-0} = regIdx;

    int Index = !cast<int>(regIdx);
  }
}

// GPR registers
foreach Index = 0...127 in {
  defm GPR_#Index : RVGPURegLoHi16 <"r"#Index, Index, 0>;
}

// Pseudo-registers: Used as placeholders during isel and immediately
// replaced, never seeing the verifier.
def FP_REG : RVGPUGPR<"fp", 0>;
def SP_REG : RVGPUGPR<"sp", 0>;

class RegisterTypes<list<ValueType> reg_types> {
  list<ValueType> types = reg_types;
}

def Reg16Types : RegisterTypes<[i16, f16, bf16]>;
def Reg32Types : RegisterTypes<[i8, i16, i32, f32, v2i16, v2f16, v2bf16, v4i8, p2, p3, p5, p6]>;

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

// GPR 64-bit registers
def VReg64 : RVGPURegisterTuples<[sub0, sub1], GPR32, 127, 1, 2, "dr">;

// GPR 96-bit registers
def VReg96 : RVGPURegisterTuples<[sub0, sub1, sub2], GPR32, 127, 1, 3, "tr">;

// GPR 128-bit registers
def VReg128 : RVGPURegisterTuples<[sub0, sub1, sub2, sub3], GPR32, 127, 1, 4, "qr">;

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
def PREG : RVGPUSREG<"%p", 127>;

// 定义Special Registers
def SREG0 : RVGPUSREG<"tid.x", 0>;
def SREG1 : RVGPUSREG<"tid.y", 1>;
def SREG2 : RVGPUSREG<"tid.z", 2>;
def SREG3 : RVGPUSREG<"tid.w", 3>;
def SREG4 : RVGPUSREG<"ntid.x", 4>;
def SREG5 : RVGPUSREG<"ntid.y", 5>;
def SREG6 : RVGPUSREG<"ntid.z", 6>;
def SREG7 : RVGPUSREG<"ntid.w", 7>;
def SREG8 : RVGPUSREG<"ctaid.x", 8>;
def SREG9 : RVGPUSREG<"ctaid.y", 9>;
def SREG10 : RVGPUSREG<"ctaid.z", 10>;
def SREG11 : RVGPUSREG<"ctaid.w", 11>;
def SREG12 : RVGPUSREG<"nctaid.x", 12>;
def SREG13 : RVGPUSREG<"nctaid.y", 13>;
def SREG14 : RVGPUSREG<"nctaid.z", 14>;
def SREG15 : RVGPUSREG<"nctaid.w", 15>;
def SREG16 : RVGPUSREG<"clusterid.x", 16>;
def SREG17 : RVGPUSREG<"clusterid.y", 17>;
def SREG18 : RVGPUSREG<"clusterid.z", 18>;
def SREG19 : RVGPUSREG<"clusterid.w", 19>;
def SREG20 : RVGPUSREG<"nclusterid.x", 20>;
def SREG21 : RVGPUSREG<"nclusterid.y", 21>;
def SREG22 : RVGPUSREG<"nclusterid.z", 22>;
def SREG23 : RVGPUSREG<"nclusterid.w", 23>;
def SREG24 : RVGPUSREG<"cluster_ctaid.x", 24>;
def SREG25 : RVGPUSREG<"cluster_ctaid.y", 25>;
def SREG26 : RVGPUSREG<"cluster_ctaid.z", 26>;
def SREG27 : RVGPUSREG<"cluster_ctaid.w", 27>;
def SREG28 : RVGPUSREG<"cluster_nctaid.x", 28>;
def SREG29 : RVGPUSREG<"cluster_nctaid.y", 29>;
def SREG30 : RVGPUSREG<"cluster_nctaid.z", 30>;
def SREG31 : RVGPUSREG<"cluster_nctaid.w", 31>;

def SREG32 : RVGPUSREG<"cluster_ctarank", 32>;
def SREG33 : RVGPUSREG<"cluster_nctarank", 33>;

def SREG34 : RVGPUSREG<"laneid", 34>;
def SREG35 : RVGPUSREG<"warpid", 35>;
def SREG36 : RVGPUSREG<"nwarpid", 36>;
def SREG37 : RVGPUSREG<"smid", 37>;
def SREG38 : RVGPUSREG<"nsmid", 38>;
def SREG39 : RVGPUSREG<"gridid", 39>;
def SREG40 : RVGPUSREG<"lanemask_eq", 40>;
def SREG41 : RVGPUSREG<"lanemask_le", 41>;
def SREG42 : RVGPUSREG<"lanemask_lt", 42>;
def SREG43 : RVGPUSREG<"lanemask_ge", 43>;
def SREG44 : RVGPUSREG<"lanemask_gt", 44>;
def SREG45 : RVGPUSREG<"clock", 45>;
def SREG46 : RVGPUSREG<"pm0", 46>;
def SREG47 : RVGPUSREG<"pm1", 47>;
def SREG48 : RVGPUSREG<"pm2", 48>;
def SREG49 : RVGPUSREG<"pm3", 49>;

//===----------------------------------------------------------------------===//
//  Register classes: 为128个通用寄存器定义
//===----------------------------------------------------------------------===//
def Int1Regs : RVGPURegClass<[i1], 32, (add PREG)>;

// Read RVGPURegisterInfo.cpp to see how VRFrame and VRDepot are used.
def SpecialRegs : RVGPURegClass<[i32], 32, (add (sequence "SREG%u", 0, 49))>;
