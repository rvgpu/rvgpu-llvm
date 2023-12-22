//===-- RVGPUInstrInfo.td - RVGPU DAG nodes --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains DAG node definitions for the RVGPU target.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RVGPU DAG Profiles
//===----------------------------------------------------------------------===//

def RVGPUDTIntTernaryOp : SDTypeProfile<1, 3, [
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisInt<0>, SDTCisInt<3>
]>;

def RVGPUFPClassOp : SDTypeProfile<1, 2,
  [SDTCisInt<0>, SDTCisFP<1>, SDTCisInt<2>]
>;

def RVGPUFPPackOp : SDTypeProfile<1, 2,
  [SDTCisFP<1>, SDTCisSameAs<1, 2>]
>;

def RVGPUIntPackOp : SDTypeProfile<1, 2,
  [SDTCisInt<1>, SDTCisSameAs<1, 2>]
>;

def RVGPUDivScaleOp : SDTypeProfile<2, 3,
  [SDTCisFP<0>, SDTCisInt<1>, SDTCisSameAs<0, 2>, SDTCisSameAs<0, 3>, SDTCisSameAs<0, 4>]
>;

// float, float, float, vcc
def RVGPUFmasOp : SDTypeProfile<1, 4,
  [SDTCisFP<0>, SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisSameAs<0, 3>, SDTCisInt<4>]
>;

def ImmOp : SDTypeProfile<0, 1, [SDTCisInt<0>]>;
def RVGPUKillSDT : SDTypeProfile<0, 1, [SDTCisInt<0>]>;

def RVGPUIfOp : SDTypeProfile<1, 2,
  [SDTCisVT<0, i1>, SDTCisVT<1, i1>, SDTCisVT<2, OtherVT>]
>;

def RVGPUElseOp : SDTypeProfile<1, 2,
  [SDTCisVT<0, i1>, SDTCisVT<1, i1>, SDTCisVT<2, OtherVT>]
>;

def RVGPULoopOp : SDTypeProfile<0, 2,
  [SDTCisVT<0, i1>, SDTCisVT<1, OtherVT>]
>;

def RVGPUIfBreakOp : SDTypeProfile<1, 2,
  [SDTCisVT<0, i1>, SDTCisVT<1, i1>, SDTCisVT<2, i1>]
>;

//===----------------------------------------------------------------------===//
// RVGPU DAG Nodes
//

def RVGPUif : SDNode<"RVGPUISD::IF", RVGPUIfOp, [SDNPHasChain]>;
def RVGPUelse : SDNode<"RVGPUISD::ELSE", RVGPUElseOp, [SDNPHasChain]>;
def RVGPUloop : SDNode<"RVGPUISD::LOOP", RVGPULoopOp, [SDNPHasChain]>;

def callseq_start : SDNode<"ISD::CALLSEQ_START",
  SDCallSeqStart<[ SDTCisVT<0, i32>, SDTCisVT<1, i32> ]>,
  [SDNPHasChain, SDNPOutGlue]
>;

def callseq_end : SDNode<"ISD::CALLSEQ_END",
 SDCallSeqEnd<[ SDTCisVT<0, i32>, SDTCisVT<1, i32> ]>,
  [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]
>;

def RVGPUcall : SDNode<"RVGPUISD::CALL",
  SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>,
  [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
  SDNPVariadic]
>;

def RVGPUTCReturnTP : SDTypeProfile<0, 3, [
  SDTCisPtrTy<0>
]>;

def RVGPUtc_return: SDNode<"RVGPUISD::TC_RETURN", RVGPUTCReturnTP,
[SDNPHasChain, SDNPOptInGlue, SDNPVariadic]
>;

def RVGPUtc_return_gfx: SDNode<"RVGPUISD::TC_RETURN_GFX", RVGPUTCReturnTP,
[SDNPHasChain, SDNPOptInGlue, SDNPVariadic]
>;

def RVGPUtc_return_chain: SDNode<"RVGPUISD::TC_RETURN_CHAIN",
  SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>,
  [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]
>;

def RVGPUtrap : SDNode<"RVGPUISD::TRAP",
  SDTypeProfile<0, -1, [SDTCisVT<0, i16>]>,
    [SDNPHasChain, SDNPVariadic, SDNPSideEffect, SDNPInGlue]
>;

def RVGPUconstdata_ptr : SDNode<
  "RVGPUISD::CONST_DATA_PTR", SDTypeProfile <1, 1, [SDTCisVT<0, iPTR>,
                                                     SDTCisVT<0, iPTR>]>
>;

// This argument to this node is a dword address.
def RVGPUdwordaddr : SDNode<"RVGPUISD::DWORDADDR", SDTIntUnaryOp>;

def RVGPUcos_impl : SDNode<"RVGPUISD::COS_HW", SDTFPUnaryOp>;
def RVGPUsin_impl : SDNode<"RVGPUISD::SIN_HW", SDTFPUnaryOp>;
// out = a - floor(a)
def RVGPUfract_impl : SDNode<"RVGPUISD::FRACT", SDTFPUnaryOp>;

// out = 1.0 / a
def RVGPUrcp_impl : SDNode<"RVGPUISD::RCP", SDTFPUnaryOp>;

// v_log_f32, which is log2
def RVGPUlog_impl : SDNode<"RVGPUISD::LOG", SDTFPUnaryOp>;

// v_exp_f32, which is exp2
def RVGPUexp_impl : SDNode<"RVGPUISD::EXP", SDTFPUnaryOp>;

// out = 1.0 / sqrt(a)
def RVGPUrsq_impl : SDNode<"RVGPUISD::RSQ", SDTFPUnaryOp>;

def RVGPUrcp_legacy_impl : SDNode<"RVGPUISD::RCP_LEGACY", SDTFPUnaryOp>;

def RVGPUrcp_iflag : SDNode<"RVGPUISD::RCP_IFLAG", SDTFPUnaryOp>;

// out = 1.0 / sqrt(a) result clamped to +/- max_float.
def RVGPUrsq_clamp_impl : SDNode<"RVGPUISD::RSQ_CLAMP", SDTFPUnaryOp>;

def RVGPUpkrtz_f16_f32_impl : SDNode<"RVGPUISD::CVT_PKRTZ_F16_F32", RVGPUFPPackOp>;
def RVGPUpknorm_i16_f32_impl : SDNode<"RVGPUISD::CVT_PKNORM_I16_F32", RVGPUFPPackOp>;
def RVGPUpknorm_u16_f32_impl : SDNode<"RVGPUISD::CVT_PKNORM_U16_F32", RVGPUFPPackOp>;
def RVGPUpk_i16_i32_impl : SDNode<"RVGPUISD::CVT_PK_I16_I32", RVGPUIntPackOp>;
def RVGPUpk_u16_u32_impl : SDNode<"RVGPUISD::CVT_PK_U16_U32", RVGPUIntPackOp>;
def RVGPUfp_to_f16 : SDNode<"RVGPUISD::FP_TO_FP16" , SDTFPToIntOp>;


def RVGPUfp_class_impl : SDNode<"RVGPUISD::FP_CLASS", RVGPUFPClassOp>;

// out = max(a, b) a and b are floats, where a nan comparison fails.
// This is not commutative because this gives the second operand:
//   x < nan ? x : nan -> nan
//   nan < x ? nan : x -> x
def RVGPUfmax_legacy : SDNode<"RVGPUISD::FMAX_LEGACY", SDTFPBinOp,
  []
>;

def RVGPUfmul_legacy_impl : SDNode<"RVGPUISD::FMUL_LEGACY", SDTFPBinOp,
  [SDNPCommutative, SDNPAssociative]
>;

// out = min(a, b) a and b are floats, where a nan comparison fails.
def RVGPUfmin_legacy : SDNode<"RVGPUISD::FMIN_LEGACY", SDTFPBinOp,
  []
>;

// FIXME: TableGen doesn't like commutative instructions with more
// than 2 operands.
// out = max(a, b, c) a, b and c are floats
def RVGPUfmax3 : SDNode<"RVGPUISD::FMAX3", SDTFPTernaryOp,
  [/*SDNPCommutative, SDNPAssociative*/]
>;

// out = max(a, b, c) a, b and c are floats. Operation is IEEE2019 compliant.
def RVGPUfmaximum3 : SDNode<"RVGPUISD::FMAXIMUM3", SDTFPTernaryOp,
  [/*SDNPCommutative, SDNPAssociative*/]
>;

// out = max(a, b, c) a, b, and c are signed ints
def RVGPUsmax3 : SDNode<"RVGPUISD::SMAX3", RVGPUDTIntTernaryOp,
  [/*SDNPCommutative, SDNPAssociative*/]
>;

// out = max(a, b, c) a, b and c are unsigned ints
def RVGPUumax3 : SDNode<"RVGPUISD::UMAX3", RVGPUDTIntTernaryOp,
  [/*SDNPCommutative, SDNPAssociative*/]
>;

// out = min(a, b, c) a, b and c are floats
def RVGPUfmin3 : SDNode<"RVGPUISD::FMIN3", SDTFPTernaryOp,
  [/*SDNPCommutative, SDNPAssociative*/]
>;

// out = min(a, b, c) a, b and c are floats. Operation is IEEE2019 compliant.
def RVGPUfminimum3 : SDNode<"RVGPUISD::FMINIMUM3", SDTFPTernaryOp,
  [/*SDNPCommutative, SDNPAssociative*/]
>;

// out = min(a, b, c) a, b and c are signed ints
def RVGPUsmin3 : SDNode<"RVGPUISD::SMIN3", RVGPUDTIntTernaryOp,
  [/*SDNPCommutative, SDNPAssociative*/]
>;

// out = min(a, b) a and b are unsigned ints
def RVGPUumin3 : SDNode<"RVGPUISD::UMIN3", RVGPUDTIntTernaryOp,
  [/*SDNPCommutative, SDNPAssociative*/]
>;

// out = (src0 + src1 > 0xFFFFFFFF) ? 1 : 0
def RVGPUcarry : SDNode<"RVGPUISD::CARRY", SDTIntBinOp, []>;

// out = (src1 > src0) ? 1 : 0
def RVGPUborrow : SDNode<"RVGPUISD::BORROW", SDTIntBinOp, []>;

def RVGPUSetCCOp : SDTypeProfile<1, 3, [        // setcc
  SDTCisInt<0>, SDTCisSameAs<1, 2>, SDTCisVT<3, OtherVT>
]>;

def RVGPUsetcc : SDNode<"RVGPUISD::SETCC", RVGPUSetCCOp>;

def RVGPUfma : SDNode<"RVGPUISD::FMA_W_CHAIN", SDTFPTernaryOp, [
   SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

def RVGPUmul : SDNode<"RVGPUISD::FMUL_W_CHAIN", SDTFPBinOp, [
  SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

def RVGPUcvt_f32_ubyte0 : SDNode<"RVGPUISD::CVT_F32_UBYTE0",
  SDTIntToFPOp, []>;
def RVGPUcvt_f32_ubyte1 : SDNode<"RVGPUISD::CVT_F32_UBYTE1",
  SDTIntToFPOp, []>;
def RVGPUcvt_f32_ubyte2 : SDNode<"RVGPUISD::CVT_F32_UBYTE2",
  SDTIntToFPOp, []>;
def RVGPUcvt_f32_ubyte3 : SDNode<"RVGPUISD::CVT_F32_UBYTE3",
  SDTIntToFPOp, []>;

def RVGPUcvt_pk_i16_i32 : SDNode<"RVGPUISD::CVT_PK_I16_I32",
  RVGPUIntPackOp, []>;

// urecip - This operation is a helper for integer division, it returns the
// result of 1 / a as a fractional unsigned integer.
// out = (2^32 / a) + e
// e is rounding error
def RVGPUurecip : SDNode<"RVGPUISD::URECIP", SDTIntUnaryOp>;

// Special case divide preop and flags.
def RVGPUdiv_scale : SDNode<"RVGPUISD::DIV_SCALE", RVGPUDivScaleOp>;

//  Special case divide FMA with scale and flags (src0 = Quotient,
//  src1 = Denominator, src2 = Numerator).
def RVGPUdiv_fmas_impl : SDNode<"RVGPUISD::DIV_FMAS", RVGPUFmasOp,
                            [SDNPOptInGlue]>;

// Single or double precision division fixup.
// Special case divide fixup and flags(src0 = Quotient, src1 =
// Denominator, src2 = Numerator).
def RVGPUdiv_fixup_impl : SDNode<"RVGPUISD::DIV_FIXUP", SDTFPTernaryOp>;

def RVGPUfmad_ftz_impl : SDNode<"RVGPUISD::FMAD_FTZ", SDTFPTernaryOp>;

def RVGPUregister_load : SDNode<"RVGPUISD::REGISTER_LOAD",
                          SDTypeProfile<1, 2, [SDTCisPtrTy<1>, SDTCisInt<2>]>,
                          [SDNPHasChain, SDNPMayLoad]>;

def RVGPUregister_store : SDNode<"RVGPUISD::REGISTER_STORE",
                           SDTypeProfile<0, 3, [SDTCisPtrTy<1>, SDTCisInt<2>]>,
                           [SDNPHasChain, SDNPMayStore]>;

// MSKOR instructions are atomic memory instructions used mainly for storing
// 8-bit and 16-bit values.  The definition is:
//
// MSKOR(dst, mask, src) MEM[dst] = ((MEM[dst] & ~mask) | src)
//
// src0: vec4(src, 0, 0, mask)
// src1: dst - rat offset (aka pointer) in dwords
def RVGPUstore_mskor : SDNode<"RVGPUISD::STORE_MSKOR",
                        SDTypeProfile<0, 2, []>,
                        [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

def RVGPUatomic_cmp_swap : SDNode<"RVGPUISD::ATOMIC_CMP_SWAP",
                            SDTypeProfile<1, 2, [SDTCisPtrTy<1>, SDTCisVec<2>]>,
                            [SDNPHasChain, SDNPMayStore, SDNPMayLoad,
                             SDNPMemOperand]>;

def RVGPUbfe_u32_impl : SDNode<"RVGPUISD::BFE_U32", RVGPUDTIntTernaryOp>;
def RVGPUbfe_i32_impl : SDNode<"RVGPUISD::BFE_I32", RVGPUDTIntTernaryOp>;
def RVGPUbfi : SDNode<"RVGPUISD::BFI", RVGPUDTIntTernaryOp>;
def RVGPUbfm : SDNode<"RVGPUISD::BFM", SDTIntBinOp>;

def RVGPUffbh_u32_impl : SDNode<"RVGPUISD::FFBH_U32", SDTIntBitCountUnaryOp>;
def RVGPUffbh_i32_impl : SDNode<"RVGPUISD::FFBH_I32", SDTIntBitCountUnaryOp>;

def RVGPUffbl_b32_impl : SDNode<"RVGPUISD::FFBL_B32", SDTIntBitCountUnaryOp>;

// Signed and unsigned 24-bit multiply. The highest 8-bits are ignore
// when performing the multiply. The result is a 32 or 64 bit value.
def RVGPUMul24Op : SDTypeProfile<1, 2, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisSameAs<1, 2>
]>;

def RVGPUmul_u24_impl : SDNode<"RVGPUISD::MUL_U24", RVGPUMul24Op,
  [SDNPCommutative, SDNPAssociative]
>;
def RVGPUmul_i24_impl : SDNode<"RVGPUISD::MUL_I24", RVGPUMul24Op,
  [SDNPCommutative, SDNPAssociative]
>;

// mulhi24 yields the high-order 16 bits of the 48-bit result. Here's an example
// that shows mulhi24 is not associative:
//
// Given a = 0x10002, b = c = 0xffffff:
// mulhi24(mulhi24(a, b), c) = mulhi24(0x100, 0xffffff) = 0
// Which is not equal to:
// mulhi24(a, mulhi24(b, c)) = mulhi24(0x10002, 0xffff) = 1
def RVGPUmulhi_u24_impl : SDNode<"RVGPUISD::MULHI_U24", SDTIntBinOp,
  [SDNPCommutative]
>;
def RVGPUmulhi_i24_impl : SDNode<"RVGPUISD::MULHI_I24", SDTIntBinOp,
  [SDNPCommutative]
>;

def RVGPUmad_u24 : SDNode<"RVGPUISD::MAD_U24", RVGPUDTIntTernaryOp,
  []
>;
def RVGPUmad_i24 : SDNode<"RVGPUISD::MAD_I24", RVGPUDTIntTernaryOp,
  []
>;

def RVGPUsmed3 : SDNode<"RVGPUISD::SMED3", RVGPUDTIntTernaryOp,
  []
>;

def RVGPUumed3 : SDNode<"RVGPUISD::UMED3", RVGPUDTIntTernaryOp,
  []
>;

def RVGPUfmed3_impl : SDNode<"RVGPUISD::FMED3", SDTFPTernaryOp, []>;

def RVGPUfdot2_impl : SDNode<"RVGPUISD::FDOT2",
                  SDTypeProfile<1, 4, [SDTCisSameAs<0, 3>, SDTCisSameAs<1, 2>,
                                       SDTCisFP<0>, SDTCisVec<1>,
                                       SDTCisInt<4>]>,
                  []>;

def RVGPUperm_impl : SDNode<"RVGPUISD::PERM", RVGPUDTIntTernaryOp, []>;

// SI+ export
def RVGPUExportOp : SDTypeProfile<0, 8, [
  SDTCisInt<0>,       // i8 tgt
  SDTCisInt<1>,       // i8 en
                      // i32 or f32 src0
  SDTCisSameAs<3, 2>, // f32 src1
  SDTCisSameAs<4, 2>, // f32 src2
  SDTCisSameAs<5, 2>, // f32 src3
  SDTCisInt<6>,       // i1 compr
  // skip done
  SDTCisInt<1>        // i1 vm

]>;


//===----------------------------------------------------------------------===//
// Flow Control Profile Types
//===----------------------------------------------------------------------===//
// Branch instruction where second and third are basic blocks
def SDTIL_BRCond : SDTypeProfile<0, 2, [
    SDTCisVT<0, OtherVT>
    ]>;

//===----------------------------------------------------------------------===//
// Flow Control DAG Nodes
//===----------------------------------------------------------------------===//
def IL_brcond      : SDNode<"RVGPUISD::BRANCH_COND", SDTIL_BRCond, [SDNPHasChain]>;

//===----------------------------------------------------------------------===//
// Call/Return DAG Nodes
//===----------------------------------------------------------------------===//
def RVGPUendpgm : SDNode<"RVGPUISD::ENDPGM", SDTNone,
    [SDNPHasChain, SDNPOptInGlue]>;
def RVGPUendpgm_trap : SDNode<"RVGPUISD::ENDPGM_TRAP", SDTNone,
    [SDNPHasChain]>;

def RVGPUreturn_to_epilog : SDNode<"RVGPUISD::RETURN_TO_EPILOG", SDTNone,
    [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;

def RVGPUret_glue : SDNode<"RVGPUISD::RET_GLUE", SDTNone,
  [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]
>;


//===----------------------------------------------------------------------===//
// Intrinsic/Custom node compatibility PatFrags
//===----------------------------------------------------------------------===//

def RVGPUrcp : PatFrags<(ops node:$src), [(int_rvgpu_rcp node:$src),
                                           (RVGPUrcp_impl node:$src)]>;
def RVGPUrcp_legacy : PatFrags<(ops node:$src), [(int_rvgpu_rcp_legacy node:$src),
                                                  (RVGPUrcp_legacy_impl node:$src)]>;

def RVGPUrsq : PatFrags<(ops node:$src), [(int_rvgpu_rsq node:$src),
                                           (RVGPUrsq_impl node:$src)]>;

def RVGPUrsq_clamp : PatFrags<(ops node:$src), [(int_rvgpu_rsq_clamp node:$src),
                                                 (RVGPUrsq_clamp_impl node:$src)]>;

def RVGPUsin : PatFrags<(ops node:$src), [(int_rvgpu_sin node:$src),
                                           (RVGPUsin_impl node:$src)]>;
def RVGPUcos : PatFrags<(ops node:$src), [(int_rvgpu_cos node:$src),
                                           (RVGPUcos_impl node:$src)]>;
def RVGPUfract : PatFrags<(ops node:$src), [(int_rvgpu_fract node:$src),
                                             (RVGPUfract_impl node:$src)]>;
def RVGPUlog : PatFrags<(ops node:$src), [(int_rvgpu_log node:$src),
                                           (RVGPUlog_impl node:$src)]>;
def RVGPUlogf16 : PatFrags<(ops node:$src), [(int_rvgpu_log node:$src),
                                              (flog2 node:$src)]>;

def RVGPUexp : PatFrags<(ops node:$src), [(int_rvgpu_exp2 node:$src),
                                           (RVGPUexp_impl node:$src)]>;
def RVGPUexpf16 : PatFrags<(ops node:$src), [(int_rvgpu_exp2 node:$src),
                                              (fexp2 node:$src)]>;

def RVGPUfp_class : PatFrags<(ops node:$src0, node:$src1),
  [(int_rvgpu_class node:$src0, node:$src1),
   (RVGPUfp_class_impl node:$src0, node:$src1)]>;

def RVGPUfmed3 : PatFrags<(ops node:$src0, node:$src1, node:$src2),
  [(int_rvgpu_fmed3 node:$src0, node:$src1, node:$src2),
   (RVGPUfmed3_impl node:$src0, node:$src1, node:$src2)]>;

def RVGPUdiv_fixup : PatFrags<(ops node:$src0, node:$src1, node:$src2),
  [(int_rvgpu_div_fixup node:$src0, node:$src1, node:$src2),
   (RVGPUdiv_fixup_impl node:$src0, node:$src1, node:$src2)]>;

def RVGPUffbh_i32 : PatFrags<(ops node:$src),
  [(int_rvgpu_sffbh node:$src),
   (RVGPUffbh_i32_impl node:$src)]>;

def RVGPUffbh_u32 : PatFrags<(ops node:$src),
  [(ctlz_zero_undef node:$src),
   (RVGPUffbh_u32_impl node:$src)]>;

def RVGPUffbl_b32 : PatFrags<(ops node:$src),
  [(cttz_zero_undef node:$src),
   (RVGPUffbl_b32_impl node:$src)]>;

def RVGPUpkrtz_f16_f32 : PatFrags<(ops node:$src0, node:$src1),
  [(int_rvgpu_cvt_pkrtz node:$src0, node:$src1),
  (RVGPUpkrtz_f16_f32_impl node:$src0, node:$src1)]>;

def RVGPUpknorm_i16_f32 : PatFrags<(ops node:$src0, node:$src1),
  [(int_rvgpu_cvt_pknorm_i16 node:$src0, node:$src1),
  (RVGPUpknorm_i16_f32_impl node:$src0, node:$src1)]>;

def RVGPUpknorm_u16_f32 : PatFrags<(ops node:$src0, node:$src1),
  [(int_rvgpu_cvt_pknorm_u16 node:$src0, node:$src1),
  (RVGPUpknorm_u16_f32_impl node:$src0, node:$src1)]>;

def RVGPUpk_i16_i32 : PatFrags<(ops node:$src0, node:$src1),
  [(int_rvgpu_cvt_pk_i16 node:$src0, node:$src1),
  (RVGPUpk_i16_i32_impl node:$src0, node:$src1)]>;

def RVGPUpk_u16_u32 : PatFrags<(ops node:$src0, node:$src1),
  [(int_rvgpu_cvt_pk_u16 node:$src0, node:$src1),
  (RVGPUpk_u16_u32_impl node:$src0, node:$src1)]>;

def RVGPUfmad_ftz : PatFrags<(ops node:$src0, node:$src1, node:$src2),
  [(int_rvgpu_fmad_ftz node:$src0, node:$src1, node:$src2),
   (RVGPUfmad_ftz_impl node:$src0, node:$src1, node:$src2)]>;

def RVGPUmul_u24 : PatFrags<(ops node:$src0, node:$src1),
  [(int_rvgpu_mul_u24 node:$src0, node:$src1),
   (RVGPUmul_u24_impl node:$src0, node:$src1)]>;

def RVGPUmul_i24 : PatFrags<(ops node:$src0, node:$src1),
  [(int_rvgpu_mul_i24 node:$src0, node:$src1),
   (RVGPUmul_i24_impl node:$src0, node:$src1)]>;

def RVGPUmulhi_u24 : PatFrags<(ops node:$src0, node:$src1),
  [(int_rvgpu_mulhi_u24 node:$src0, node:$src1),
   (RVGPUmulhi_u24_impl node:$src0, node:$src1)]>;

def RVGPUmulhi_i24 : PatFrags<(ops node:$src0, node:$src1),
  [(int_rvgpu_mulhi_i24 node:$src0, node:$src1),
   (RVGPUmulhi_i24_impl node:$src0, node:$src1)]>;

def RVGPUbfe_i32 : PatFrags<(ops node:$src0, node:$src1, node:$src2),
  [(int_rvgpu_sbfe node:$src0, node:$src1, node:$src2),
   (RVGPUbfe_i32_impl node:$src0, node:$src1, node:$src2)]>;

def RVGPUbfe_u32 : PatFrags<(ops node:$src0, node:$src1, node:$src2),
  [(int_rvgpu_ubfe node:$src0, node:$src1, node:$src2),
   (RVGPUbfe_u32_impl node:$src0, node:$src1, node:$src2)]>;

def RVGPUfmul_legacy : PatFrags<(ops node:$src0, node:$src1),
  [(int_rvgpu_fmul_legacy node:$src0, node:$src1),
   (RVGPUfmul_legacy_impl node:$src0, node:$src1)]>;

def RVGPUfdot2 : PatFrags<(ops node:$src0, node:$src1, node:$src2, node:$clamp),
  [(int_rvgpu_fdot2 node:$src0, node:$src1, node:$src2, node:$clamp),
   (RVGPUfdot2_impl node:$src0, node:$src1, node:$src2, node:$clamp)]>;

def RVGPUdiv_fmas : PatFrags<(ops node:$src0, node:$src1, node:$src2, node:$vcc),
  [(int_rvgpu_div_fmas node:$src0, node:$src1, node:$src2, node:$vcc),
   (RVGPUdiv_fmas_impl node:$src0, node:$src1, node:$src2, node:$vcc)]>;

def RVGPUperm : PatFrags<(ops node:$src0, node:$src1, node:$src2),
  [(int_rvgpu_perm node:$src0, node:$src1, node:$src2),
   (RVGPUperm_impl node:$src0, node:$src1, node:$src2)]>;
