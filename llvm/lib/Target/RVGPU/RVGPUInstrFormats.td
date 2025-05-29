//===- RVGPUInstrFormats.td - RVGPU Instruction Formats-------*- tblgen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Describe RVGPU instructions format
//
//===----------------------------------------------------------------------===//

// Vector instruction type enum
class VecInstTypeEnum<bits<4> val> {
  bits<4> Value=val;
}
def VecNOP : VecInstTypeEnum<0>;

class VOPDstOperand <RegisterClass rc> : RegisterOperand <rc, "printVOPDst">;

def VOPDstOperand_t16 : VOPDstOperand <GPR16> {
  let EncoderMethod = "getMachineOpValueT16";
  let DecoderMethod = "DecodeGPR_16RegisterClass";
}

// Generic RVGPU Format
class RVGPUInst<dag outs, dag ins, string asmstr, list<dag> pattern>
  : Instruction {
  field bits<32> Inst = 0;

  let Namespace = "RVGPU";
  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;

  bits<4> VecInstType = VecNOP.Value;
  bit IsSimpleMove = false;
  bit IsLoad = false;
  bit IsStore = false;

  bit IsTex = false;
  bit IsSust = false;
  bit IsSurfTexQuery = false;
  bit IsTexModeUnified = false;

  // The following field is encoded as log2 of the vector size minus one,
  // with 0 meaning the operation is not a surface instruction.  For example,
  // if IsSuld == 2, then the instruction is a suld instruction with vector size
  // 2**(2-1) = 2.
  bits<2> IsSuld = 0;
}

class RVGPUInstR<bits<8> opcode, bits<3> optype, dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<7> rd = 0;
  bits<7> rs1 = 0;
  bits<7> rs2 = 0;

  let Inst{31-24} = opcode;
}

class CMP<bits<8> opcode, dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<8> rd = 0;
  bits<8> rs1 = 0;
  bits<8> rs2 = 0;

  let Inst{31-24} = opcode;
}

class RVGPULDrr<bits<8> op1, bits<3> type, bits<8> op2,
      dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<8> dst = 0;
  bits<8> addr = 0;

  let Inst{31-24} = op1;
}

class RVGPUInst_CVT <bits<4> op, bits<5> dtype, bits<5> stype, dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<7> src;
  bits<7> dst;
  bits<4> mode;

  let Inst{31-28} = op;      // 4-bit opcode
  let Inst{27-23} = dtype;   // 5-bit destination type
  let Inst{22-18} = stype;   // 5-bit source type
  let Inst{17-14} = mode;    // 4-bit conversion mode
  let Inst{13-7} = src;      // 7-bit source register
  let Inst{6-0} = dst;       // 7-bit destination register
}

class RVGPUInst_RI <bits<6> op, dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<7> src;
  bits<7> dst;
  bits<12> imm12;

  let Inst{31-26} = op;      // 6-bit opcode
  let Inst{25-14} = imm12;   // 12-bit immediate or function code
  let Inst{13-7} = src;      // 7-bit source register
  let Inst{6-0} = dst;       // 7-bit destination register
}

class RVGPUInst_RR <bits<6> op, bits<5> func, dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<7> src1;
  bits<7> src2;
  bits<7> dst;

  let Inst{31-26} = op;      // 6-bit opcode
  let Inst{25-21} = func;    // 5-bit function code
  let Inst{20-14} = src2;    // 7-bit second source register 
  let Inst{13-7} = src1;     // 7-bit first source register
  let Inst{6-0} = dst;       // 7-bit destination register
}

class RVGPUInst_R <bits<6> op, bits<12> func, dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<7> src;
  bits<7> dst;

  let Inst{31-26} = op;      // 6-bit opcode
  let Inst{25-14} = func;    // 12-bit immediate
  let Inst{13-7} = src;      // 7-bit source register
  let Inst{6-0} = dst;       // 7-bit destination register
}


class RVGPULDri<bits<8> op1, bits<3> type, bits<8> op2,
      dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<8> dst = 0;
  bits<8> addr = 0;
  bits<16> offset = 0;

  let Inst{31-24} = op1;
}

class RVGPUSTrr<bits<8> op, dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<8> addr = 0;
  bits<8> src1 = 0;

  let Inst{31-24} = op;
}

class RVGPUSTri<bits<8> op, dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<8> src1 = 0;
  bits<8> addr = 0;
  bits<16> offset = 0;

  let Inst{31-24} = 0b11000110;
}

class RVGPUInstI<bits<8> opcode, bits<3> optype, dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<8> rd = 0;
  bits<8> rs1 = 0;
  bits<16> imm16 = 0;

  let Inst{31-24} = opcode;
}
