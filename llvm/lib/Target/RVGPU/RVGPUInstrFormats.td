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


// Generic RVGPU Format

class RVGPUInstBase<dag outs, dag ins, string asmstr, list<dag> pattern>
  : Instruction {
  let Namespace = "RVGPU";
  dag OutOperandList = outs;
  dag InOperandList = ins;
  
  let AsmString = asmstr;
  let Pattern = pattern;

  // TSFlagFields
//  bits<4> VecInstType = VecNOP.Value;
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

 // let TSFlags{3...0}   = VecInstType;
  let TSFlags{4...4}   = IsSimpleMove;
  let TSFlags{5...5}   = IsLoad;
  let TSFlags{6...6}   = IsStore;
  let TSFlags{7}       = IsTex;
  let TSFlags{9...8}   = IsSuld;
  let TSFlags{10}      = IsSust;
  let TSFlags{11}      = IsSurfTexQuery;
  let TSFlags{12}      = IsTexModeUnified;
}

class RVGPUInst<dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInstBase<outs, ins, asmstr, pattern> {
  field bits<64> Inst;
  field bits<64> SoftFail = 0; 
  let Size = 8;
}

class RVGPUInstR<bits<8> op1, bits<3> optype,
                bits<8> op2,
                dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<8> rd;
  bits<8> rs1;
  bits<8> rs2;

  let Inst{63-56} = op1;
  let Inst{55-53} = optype;
  let Inst{52-45} = op2;
  let Inst{23-16} = rs2;
  let Inst{15-8}  = rs1;
  let Inst{7-0}   = rd;
}

class CMP<bits<8> opcode, dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<8> rd;
  bits<8> rs1;
  bits<8> rs2;

  let Inst{63-56} = 0b11000000;
  let Inst{55-53} = 0b000;
  let Inst{52-45} = opcode;
  let Inst{23-16} = rs2;
  let Inst{15-8}  = rs1;
  let Inst{7-0}   = rd;
}

class RVGPULDrr<bits<8> op1, bits<3> type, bits<8> op2,
      dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<8> dst;
  bits<8> addr;

  let Inst{63-56} = op1;
  let Inst{55-53} = type;
  let Inst{52-45} = op2;
  let Inst{31-24} = dst;
  let Inst{23-16} = addr;
  let Inst{15-0} = 0;
}

class RVGPULDri<bits<8> op1, bits<3> type, bits<8> op2,
      dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<8> dst;
  bits<8> addr;
  bits<16> offset;

  let Inst{63-56} = op1;
  let Inst{55-53} = type;
  let Inst{52-45} = op2;
  let Inst{31-24} = dst;
  let Inst{23-16} = addr;
  let Inst{15-0}  = offset;
}

class RVGPUSTrr<bits<8> op, dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<8> addr;
  bits<8> src1;

  let Inst{63-56} = 0b11000110;
  let Inst{55-53} = 0b000;
  let Inst{52-45} = op;
  let Inst{31-24} = src1;
  let Inst{23-16} = addr;
  let Inst{15-0} = 0;
}

class RVGPUSTri<bits<8> op, dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<8> src1;
  bits<8> addr;
  bits<16> offset;

  let Inst{63-56} = 0b11000110;
  let Inst{55-53} = 0b000;
  let Inst{52-45} = op;
  let Inst{31-24} = src1;
  let Inst{23-16} = addr;
  let Inst{15-0}  = offset;
}

class RVGPUInstI<bits<8> op1, bits<3> optype,
                 bits<8> op2, dag outs, dag ins,
                 string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<8> rd;
  bits<8> rs1;
  bits<16> imm16;

  let Inst{63-56} = op1;
  let Inst{55-53} = optype;
  let Inst{52-45} = op2;
  let Inst{31-16} = imm16;
  let Inst{15-8}  = rs1;
  let Inst{7-0}   = rd;
  let Inst{52} = 0b1;
}

class UnaryR<bits<8> enc, bits<3> type, bits<8> op,
      dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<8> rd;
  bits<8> rs1;

  let Inst{63-56} = enc;
  let Inst{55-53} = type;
  let Inst{52-45} = op;
  let Inst{31-24} = rd;
  let Inst{23-16}  = rs1;
}

class UnaryI<bits<8> enc, bits<3> type, bits<8> op,
      dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<8> rd;
  bits<24> imm;

  let Inst{63-56} = enc;
  let Inst{55-53} = type;
  let Inst{52-45} = op;
  let Inst{31-24} = rd;
  let Inst{23-0}  = imm;
}
