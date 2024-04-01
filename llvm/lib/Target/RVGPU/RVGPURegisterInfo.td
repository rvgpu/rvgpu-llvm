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

// We use virtual registers, but define a few physical registers here to keep
// SDAG and the MachineInstr layers happy.
foreach i = 0...4 in {
  def P#i  : RVGPUReg<"%p"#i>;  // Predicate
  def RS#i : RVGPUReg<"%rs"#i>; // 16-bit
  def R#i  : RVGPUReg<"%r"#i>;  // 32-bit
  def RL#i : RVGPUReg<"%rd"#i>; // 64-bit
  def H#i  : RVGPUReg<"%h"#i>;  // 16-bit float
  def HH#i : RVGPUReg<"%hh"#i>; // 2x16-bit float
  def F#i  : RVGPUReg<"%f"#i>;  // 32-bit float
  def FL#i : RVGPUReg<"%fd"#i>; // 64-bit float

  // Arguments
  def ia#i : RVGPUReg<"%ia"#i>;
  def la#i : RVGPUReg<"%la"#i>;
  def fa#i : RVGPUReg<"%fa"#i>;
  def da#i : RVGPUReg<"%da"#i>;
}

foreach i = 0...31 in {
  def ENVREG#i : RVGPUReg<"%envreg"#i>;
}

//===----------------------------------------------------------------------===//
//  Register classes
//===----------------------------------------------------------------------===//
def Int1Regs : RVGPURegClass<[i1], 8, (add (sequence "P%u", 0, 4))>;
def Int16Regs : RVGPURegClass<[i16, f16, bf16], 16, (add (sequence "RS%u", 0, 4))>;
def Int32Regs : RVGPURegClass<[i32, v2f16, v2bf16, v2i16, v4i8], 32,
                              (add (sequence "R%u", 0, 4),
                              VRFrame32, VRFrameLocal32)>;
def Int64Regs : RVGPURegClass<[i64], 64, (add (sequence "RL%u", 0, 4), VRFrame64, VRFrameLocal64)>;
def Float32Regs : RVGPURegClass<[f32], 32, (add (sequence "F%u", 0, 4))>;
def Float64Regs : RVGPURegClass<[f64], 64, (add (sequence "FL%u", 0, 4))>;
def Int32ArgRegs : RVGPURegClass<[i32], 32, (add (sequence "ia%u", 0, 4))>;
def Int64ArgRegs : RVGPURegClass<[i64], 64, (add (sequence "la%u", 0, 4))>;
def Float32ArgRegs : RVGPURegClass<[f32], 32, (add (sequence "fa%u", 0, 4))>;
def Float64ArgRegs : RVGPURegClass<[f64], 64, (add (sequence "da%u", 0, 4))>;

// Read RVGPURegisterInfo.cpp to see how VRFrame and VRDepot are used.
def SpecialRegs : RVGPURegClass<[i32], 32, (add VRFrame32, VRFrameLocal32, VRDepot,
                                            (sequence "ENVREG%u", 0, 31))>;
