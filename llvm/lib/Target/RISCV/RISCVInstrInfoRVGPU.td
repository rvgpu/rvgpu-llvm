//===-- RISCVInstrInfoRVGPU.td - RVGPU instructions ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the vendor extensions defined by Core-V extensions.
//
//===----------------------------------------------------------------------===//

// custom-0 instruction
// rvg.ldtreg rd, threg
// threg is:
// 0: tid.x
// 1: tid.y
// 2: tid.z
// 3: ntid.x
// 4: ntid.y
// 5: ntid.z
// 6: laneid
// 7: warpid
// 8: nwarpid
// 9: ctaid.x
// 10: ctaid.y
// 11: ctaid.z
// 12: simd
// 13: nsimd
//
// decoder to :  imm12_threg_000_rd_0001011

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVGInstLoad_treg<bits<3> funct3, string opcodestr>
    : RVInstI<funct3, OPC_CUSTOM_0, (outs GPR:$rd), (ins GPRMem:$rs1, simm12:$imm12),
              opcodestr, "$rd, ${rs1}">,
      Sched<[WriteIALU]>;

def RVG_LDTREG : RVGInstLoad_treg<0b000, "rvg.ldtreg">;
