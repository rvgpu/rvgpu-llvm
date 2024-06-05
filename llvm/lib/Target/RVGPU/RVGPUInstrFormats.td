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

class AddressSpacesImpl {
//  int Flat = 0;
  int Global = 1;
  int Region = 2;
  int Local = 3;
  int Constant = 4;
  int Private = 5;
  int Constant32Bit = 6;
}

def AddrSpaces : AddressSpacesImpl;

class AddressSpaceList<list<int> AS> {
  list<int> AddrSpaces = AS;
}

def LoadAddress_constant : AddressSpaceList<[ AddrSpaces.Constant,
                                              AddrSpaces.Constant32Bit ]>;
def LoadAddress_global : AddressSpaceList<[ AddrSpaces.Global,
                                            AddrSpaces.Constant,
                                            AddrSpaces.Constant32Bit ]>;
def StoreAddress_global : AddressSpaceList<[ AddrSpaces.Global ]>;
/*
def LoadAddress_flat : AddressSpaceList<[ AddrSpaces.Flat,
                                          AddrSpaces.Global,
                                          AddrSpaces.Constant,
                                          AddrSpaces.Constant32Bit ]>;
def StoreAddress_flat : AddressSpaceList<[ AddrSpaces.Flat, AddrSpaces.Global ]>;
*/
def LoadAddress_private : AddressSpaceList<[ AddrSpaces.Private ]>;
def StoreAddress_private : AddressSpaceList<[ AddrSpaces.Private ]>;

def LoadAddress_local : AddressSpaceList<[ AddrSpaces.Local ]>;
def StoreAddress_local : AddressSpaceList<[ AddrSpaces.Local ]>;

def LoadAddress_region : AddressSpaceList<[ AddrSpaces.Region ]>;
def StoreAddress_region : AddressSpaceList<[ AddrSpaces.Region ]>;

foreach as = [ "global", "constant", "local", "private", "region" ] in {
let AddressSpaces = !cast<AddressSpaceList>("LoadAddress_"#as).AddrSpaces in {

def load_#as : PatFrag<(ops node:$ptr), (unindexedload node:$ptr)> {
  let IsLoad = 1;
  let IsNonExtLoad = 1;
}

def extloadi8_#as  : PatFrag<(ops node:$ptr), (extloadi8 node:$ptr)> {
  let IsLoad = 1;
}

def extloadi16_#as : PatFrag<(ops node:$ptr), (extloadi16 node:$ptr)> {
  let IsLoad = 1;
}

def sextloadi8_#as  : PatFrag<(ops node:$ptr), (sextloadi8 node:$ptr)> {
  let IsLoad = 1;
}

def sextloadi16_#as : PatFrag<(ops node:$ptr), (sextloadi16 node:$ptr)> {
  let IsLoad = 1;
}

def zextloadi8_#as  : PatFrag<(ops node:$ptr), (zextloadi8 node:$ptr)> {
  let IsLoad = 1;
}

def zextloadi16_#as : PatFrag<(ops node:$ptr), (zextloadi16 node:$ptr)> {
  let IsLoad = 1;
}

def atomic_load_8_#as : PatFrag<(ops node:$ptr), (atomic_load_8 node:$ptr)> {
  let IsAtomic = 1;
  let MemoryVT = i8;
}

def atomic_load_16_#as : PatFrag<(ops node:$ptr), (atomic_load_16 node:$ptr)> {
  let IsAtomic = 1;
  let MemoryVT = i16;
}

def atomic_load_32_#as : PatFrag<(ops node:$ptr), (atomic_load_32 node:$ptr)> {
  let IsAtomic = 1;
  let MemoryVT = i32;
}

def atomic_load_64_#as : PatFrag<(ops node:$ptr), (atomic_load_64 node:$ptr)> {
  let IsAtomic = 1;
  let MemoryVT = i64;
}
} // End let AddressSpaces
} // End foreach as


foreach as = [ "global", "local", "private", "region" ] in {
let IsStore = 1, AddressSpaces = !cast<AddressSpaceList>("StoreAddress_"#as).AddrSpaces in {
def store_#as : PatFrag<(ops node:$val, node:$ptr),
                    (unindexedstore node:$val, node:$ptr)> {
  let IsTruncStore = 0;
}

// truncstore fragments.
def truncstore_#as : PatFrag<(ops node:$val, node:$ptr),
                             (unindexedstore node:$val, node:$ptr)> {
  let IsTruncStore = 1;
}

// TODO: We don't really need the truncstore here. We can use
// unindexedstore with MemoryVT directly, which will save an
// unnecessary check that the memory size is less than the value type
// in the generated matcher table.
def truncstorei8_#as : PatFrag<(ops node:$val, node:$ptr),
                               (truncstorei8 node:$val, node:$ptr)>;
def truncstorei16_#as : PatFrag<(ops node:$val, node:$ptr),
                                (truncstorei16 node:$val, node:$ptr)>;

def store_hi16_#as : StoreHi16 <truncstorei16, i16>;
def truncstorei8_hi16_#as : StoreHi16<truncstorei8, i8>;
def truncstorei16_hi16_#as : StoreHi16<truncstorei16, i16>;
} // End let IsStore = 1, AddressSpaces = ...

let IsAtomic = 1, AddressSpaces = !cast<AddressSpaceList>("StoreAddress_"#as).AddrSpaces in {
def atomic_store_8_#as : PatFrag<(ops node:$val, node:$ptr),
                                 (atomic_store_8 node:$val, node:$ptr)>;
def atomic_store_16_#as : PatFrag<(ops node:$val, node:$ptr),
                                  (atomic_store_16 node:$val, node:$ptr)>;
def atomic_store_32_#as : PatFrag<(ops node:$val, node:$ptr),
                                  (atomic_store_32 node:$val, node:$ptr)>;
def atomic_store_64_#as : PatFrag<(ops node:$val, node:$ptr),
                                  (atomic_store_64 node:$val, node:$ptr)>;
} // End let IsAtomic = 1, AddressSpaces = ...
} // End foreach as


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
  let Inst{31-24} = rd;
  let Inst{23-16} = rs1;
  let Inst{15-0}  = imm16;
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
  let Inst{52} = 0b1;
}

class READ_ID<bits<6> op,bits<2> dim,
      dag outs, dag ins, string asmstr, list<dag> pattern>
    : RVGPUInst<outs, ins, asmstr, pattern> {
  bits<8> rd;
  
  let Inst{63-56} = 0b11000001;
  let Inst{55-53} = 0b001;
  let Inst{52-45} = 0b00000000;
  let Inst{31-24} = rd;
  let Inst{23-18} = op;
  let Inst{17-16} = dim;
}
