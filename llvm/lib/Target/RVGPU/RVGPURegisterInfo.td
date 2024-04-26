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

class RVGPUReg<string n> : Register<n> {
  let Namespace = "RVGPU";
}

class RVGPURegClass<list<ValueType> regTypes, int alignment, dag regList>
     : RegisterClass <"RVGPU", regTypes, alignment, regList>;

//===----------------------------------------------------------------------===//
//  Registers
//===----------------------------------------------------------------------===//

// Special Registers used as stack pointer
def VRFrame32         : RVGPUReg<"%SP">;
def VRFrame64         : RVGPUReg<"%SP">;
def VRFrameLocal32    : RVGPUReg<"%SPL">;
def VRFrameLocal64    : RVGPUReg<"%SPL">;

// Special Registers used as the stack
def VRDepot  : RVGPUReg<"%Depot">;

// 定义1个Predicate Register
def PREG : RVGPUReg<"%p">;

// 定义 GR0 ~ GR127，共128个32位的通用寄存器
foreach i = 0...127 in {
  def R#i : RVGPUReg<"%r" # i>;
}

foreach i = 0...31 in {
  def ENVREG#i : RVGPUReg<"%sreg"#i>;
}

//===----------------------------------------------------------------------===//
//  Register classes: 为128个通用寄存器定义
//===----------------------------------------------------------------------===//
def Int1Regs : RVGPURegClass<[i1], 32, (add PREG)>;

def Int16Regs : RVGPURegClass<[i16, f16, bf16], 32, (add (sequence "R%u", 0, 127))>;
def Int32Regs : RVGPURegClass<[i32, v2f16, v2bf16, v2i16, v4i8], 32, (add (sequence "R%u", 0, 127))>;
def Int64Regs : RVGPURegClass<[i64], 64, (add (sequence "R%u", 0, 127, 2))>;
def Float32Regs : RVGPURegClass<[f32], 32, (add (sequence "R%u", 0, 127))>;
def Float64Regs : RVGPURegClass<[f64], 64, (add (sequence "R%u", 0, 127, 2))>;
def Int32ArgRegs : RVGPURegClass<[i32], 32, (add (sequence "R%u", 0, 127))>;
def Int64ArgRegs : RVGPURegClass<[i64], 64, (add (sequence "R%u", 0, 127, 2))>;
def Float32ArgRegs : RVGPURegClass<[f32], 32, (add (sequence "R%u", 0, 127))>;
def Float64ArgRegs : RVGPURegClass<[f64], 64, (add (sequence "R%u", 0, 127, 2))>;

// Read RVGPURegisterInfo.cpp to see how VRFrame and VRDepot are used.
def SpecialRegs : RVGPURegClass<[i32], 32, (add (sequence "ENVREG%u", 0, 31))>;
