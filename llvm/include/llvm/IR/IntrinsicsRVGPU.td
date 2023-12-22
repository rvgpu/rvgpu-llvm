//===- IntrinsicsRVGPU.td - Defines RVGPU intrinsics -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the R600-specific intrinsics.
//
//===----------------------------------------------------------------------===//

class RVGPUReadPreloadRegisterIntrinsic
  : DefaultAttrsIntrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrSpeculatable]>;

class RVGPUReadPreloadRegisterIntrinsicNamed<string name>
  : DefaultAttrsIntrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrSpeculatable]>, ClangBuiltin<name>;

// Used to tag image and resource intrinsics with information used to generate
// mem operands.
class RVGPURsrcIntrinsic<int rsrcarg, bit isimage = false> {
  int RsrcArg = rsrcarg;
  bit IsImage = isimage;
}

let TargetPrefix = "rvgpu" in {

//===----------------------------------------------------------------------===//
// ABI Special Intrinsics
//===----------------------------------------------------------------------===//

multiclass RVGPUReadPreloadRegisterIntrinsic_xyz {
  def _x : RVGPUReadPreloadRegisterIntrinsic;
  def _y : RVGPUReadPreloadRegisterIntrinsic;
  def _z : RVGPUReadPreloadRegisterIntrinsic;
}

multiclass RVGPUReadPreloadRegisterIntrinsic_xyz_named<string prefix> {
  def _x : RVGPUReadPreloadRegisterIntrinsicNamed<!strconcat(prefix, "_x")>;
  def _y : RVGPUReadPreloadRegisterIntrinsicNamed<!strconcat(prefix, "_y")>;
  def _z : RVGPUReadPreloadRegisterIntrinsicNamed<!strconcat(prefix, "_z")>;
}
defm int_rvgpu_workitem_id : RVGPUReadPreloadRegisterIntrinsic_xyz;
defm int_rvgpu_workgroup_id : RVGPUReadPreloadRegisterIntrinsic_xyz_named
                               <"__builtin_rvgpu_workgroup_id">;

def int_rvgpu_dispatch_ptr :
  DefaultAttrsIntrinsic<[LLVMQualPointerType<4>], [],
  [Align<RetIndex, 4>, IntrNoMem, IntrSpeculatable]>;

def int_rvgpu_queue_ptr :
  ClangBuiltin<"__builtin_rvgpu_queue_ptr">,
  DefaultAttrsIntrinsic<[LLVMQualPointerType<4>], [],
  [Align<RetIndex, 4>, IntrNoMem, IntrSpeculatable]>;

def int_rvgpu_kernarg_segment_ptr :
  ClangBuiltin<"__builtin_rvgpu_kernarg_segment_ptr">,
  DefaultAttrsIntrinsic<[LLVMQualPointerType<4>], [],
  [Align<RetIndex, 4>, IntrNoMem, IntrSpeculatable]>;

def int_rvgpu_implicitarg_ptr :
  ClangBuiltin<"__builtin_rvgpu_implicitarg_ptr">,
  DefaultAttrsIntrinsic<[LLVMQualPointerType<4>], [],
  [Align<RetIndex, 4>, IntrNoMem, IntrSpeculatable]>;

def int_rvgpu_groupstaticsize :
  ClangBuiltin<"__builtin_rvgpu_groupstaticsize">,
  DefaultAttrsIntrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrSpeculatable]>;

def int_rvgpu_dispatch_id :
  ClangBuiltin<"__builtin_rvgpu_dispatch_id">,
  DefaultAttrsIntrinsic<[llvm_i64_ty], [], [IntrNoMem, IntrSpeculatable]>;

// For internal use. Coordinates LDS lowering between IR transform and backend.
def int_rvgpu_lds_kernel_id :
  DefaultAttrsIntrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrSpeculatable]>;

def int_rvgpu_implicit_buffer_ptr :
  ClangBuiltin<"__builtin_rvgpu_implicit_buffer_ptr">,
  DefaultAttrsIntrinsic<[LLVMQualPointerType<4>], [],
  [Align<RetIndex, 4>, IntrNoMem, IntrSpeculatable]>;

// Set EXEC to the 64-bit value given.
// This is always moved to the beginning of the basic block.
// FIXME: Should be mangled for wave size.
def int_rvgpu_init_exec : Intrinsic<[],
  [llvm_i64_ty],      // 64-bit literal constant
  [IntrConvergent, IntrNoMem, IntrHasSideEffects, IntrNoCallback,
   IntrNoFree, IntrWillReturn, ImmArg<ArgIndex<0>>]>;

// Set EXEC according to a thread count packed in an SGPR input:
//    thread_count = (input >> bitoffset) & 0x7f;
// This is always moved to the beginning of the basic block.
// Note: only inreg arguments to the parent function are valid as
// inputs to this intrinsic, computed values cannot be used.
def int_rvgpu_init_exec_from_input : Intrinsic<[],
  [llvm_i32_ty,       // 32-bit SGPR input
   llvm_i32_ty],      // bit offset of the thread count
  [IntrConvergent, IntrHasSideEffects, IntrNoMem, IntrNoCallback,
   IntrNoFree, IntrWillReturn, ImmArg<ArgIndex<1>>]>;

def int_rvgpu_wavefrontsize :
  ClangBuiltin<"__builtin_rvgpu_wavefrontsize">,
  DefaultAttrsIntrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrSpeculatable]>;

// Represent a relocation constant.
def int_rvgpu_reloc_constant : DefaultAttrsIntrinsic<
  [llvm_i32_ty], [llvm_metadata_ty],
  [IntrNoMem, IntrSpeculatable]
>;

//===----------------------------------------------------------------------===//
// Instruction Intrinsics
//===----------------------------------------------------------------------===//

// The first parameter is s_sendmsg immediate (i16),
// the second one is copied to m0
def int_rvgpu_s_sendmsg : ClangBuiltin<"__builtin_rvgpu_s_sendmsg">,
  Intrinsic <[], [llvm_i32_ty, llvm_i32_ty],
  [ImmArg<ArgIndex<0>>, IntrNoMem, IntrHasSideEffects]>;
def int_rvgpu_s_sendmsghalt : ClangBuiltin<"__builtin_rvgpu_s_sendmsghalt">,
  Intrinsic <[], [llvm_i32_ty, llvm_i32_ty],
  [ImmArg<ArgIndex<0>>, IntrNoMem, IntrHasSideEffects]>;


// gfx11 intrinsic
// The first parameter is s_sendmsg immediate (i16). Return type is i32 or i64.
def int_rvgpu_s_sendmsg_rtn : Intrinsic <[llvm_anyint_ty], [llvm_i32_ty],
  [ImmArg<ArgIndex<0>>, IntrNoMem, IntrHasSideEffects]>;

def int_rvgpu_s_barrier : ClangBuiltin<"__builtin_rvgpu_s_barrier">,
  Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

def int_rvgpu_s_barrier_signal : ClangBuiltin<"__builtin_rvgpu_s_barrier_signal">,
  Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>, IntrNoMem, IntrHasSideEffects, IntrConvergent, IntrWillReturn,
                                IntrNoCallback, IntrNoFree]>;

def int_rvgpu_s_barrier_signal_var : ClangBuiltin<"__builtin_rvgpu_s_barrier_signal_var">,
  Intrinsic<[], [llvm_i32_ty], [IntrNoMem, IntrHasSideEffects, IntrConvergent, IntrWillReturn,
                                IntrNoCallback, IntrNoFree]>;

def int_rvgpu_s_barrier_signal_isfirst : ClangBuiltin<"__builtin_rvgpu_s_barrier_signal_isfirst">,
  Intrinsic<[llvm_i1_ty], [llvm_i32_ty], [ImmArg<ArgIndex<0>>, IntrNoMem, IntrHasSideEffects, IntrConvergent,
                                IntrWillReturn, IntrNoCallback, IntrNoFree]>;

def int_rvgpu_s_barrier_signal_isfirst_var : ClangBuiltin<"__builtin_rvgpu_s_barrier_signal_isfirst_var">,
  Intrinsic<[llvm_i1_ty], [llvm_i32_ty], [IntrNoMem, IntrHasSideEffects, IntrConvergent, IntrWillReturn,
                                IntrNoCallback, IntrNoFree]>;

def int_rvgpu_s_barrier_init : ClangBuiltin<"__builtin_rvgpu_s_barrier_init">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem, IntrHasSideEffects, IntrConvergent,
                                IntrWillReturn, IntrNoCallback, IntrNoFree]>;

def int_rvgpu_s_barrier_join : ClangBuiltin<"__builtin_rvgpu_s_barrier_join">,
  Intrinsic<[], [llvm_i32_ty], [IntrNoMem, IntrHasSideEffects, IntrConvergent, IntrWillReturn,
                                IntrNoCallback, IntrNoFree]>;

def int_rvgpu_s_wakeup_barrier : ClangBuiltin<"__builtin_rvgpu_s_wakeup_barrier">,
  Intrinsic<[], [llvm_i32_ty], [IntrNoMem, IntrHasSideEffects, IntrConvergent, IntrWillReturn,
                                IntrNoCallback, IntrNoFree]>;

def int_rvgpu_s_barrier_wait : ClangBuiltin<"__builtin_rvgpu_s_barrier_wait">,
  Intrinsic<[], [llvm_i16_ty], [ImmArg<ArgIndex<0>>, IntrNoMem, IntrHasSideEffects, IntrConvergent,
                                IntrWillReturn, IntrNoCallback, IntrNoFree]>;

def int_rvgpu_s_barrier_leave : ClangBuiltin<"__builtin_rvgpu_s_barrier_leave">,
  Intrinsic<[llvm_i1_ty], [], [IntrNoMem, IntrHasSideEffects, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

def int_rvgpu_s_get_barrier_state : ClangBuiltin<"__builtin_rvgpu_s_get_barrier_state">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem, IntrHasSideEffects, IntrConvergent, IntrWillReturn,
                                IntrNoCallback, IntrNoFree]>;

def int_rvgpu_wave_barrier : ClangBuiltin<"__builtin_rvgpu_wave_barrier">,
  Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

// The 1st parameter is a mask for the types of instructions that may be allowed
// to cross the SCHED_BARRIER during scheduling.
//     MASK = 0x0000 0000: No instructions may be scheduled across SCHED_BARRIER.
//     MASK = 0x0000 0001: ALL, non-memory, non-side-effect producing instructions may be
//                         scheduled across SCHED_BARRIER, i.e. allow ALU instructions to pass.
//     MASK = 0x0000 0002: VALU instructions may be scheduled across SCHED_BARRIER.
//     MASK = 0x0000 0004: SALU instructions may be scheduled across SCHED_BARRIER.
//     MASK = 0x0000 0008: MFMA/WMMA instructions may be scheduled across SCHED_BARRIER.
//     MASK = 0x0000 0010: ALL VMEM instructions may be scheduled across SCHED_BARRIER.
//     MASK = 0x0000 0020: VMEM read instructions may be scheduled across SCHED_BARRIER.
//     MASK = 0x0000 0040: VMEM write instructions may be scheduled across SCHED_BARRIER.
//     MASK = 0x0000 0080: ALL DS instructions may be scheduled across SCHED_BARRIER.
//     MASK = 0x0000 0100: ALL DS read instructions may be scheduled accoss SCHED_BARRIER.
//     MASK = 0x0000 0200: ALL DS write instructions may be scheduled across SCHED_BARRIER.
def int_rvgpu_sched_barrier : ClangBuiltin<"__builtin_rvgpu_sched_barrier">,
  Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>, IntrNoMem, IntrHasSideEffects, IntrConvergent,
                                IntrWillReturn, IntrNoCallback, IntrNoFree]>;

// The first parameter is a mask that determines the types of instructions that
// you would like to synchronize around and add to a scheduling group. The
// values of the mask are defined above for sched_barrier. These instructions
// will be selected from the bottom up starting from the sched_group_barrier's
// location during instruction scheduling. The second parameter is the number of
// matching instructions that will be associated with this sched_group_barrier.
// The third parameter is an identifier which is used to describe what other
// sched_group_barriers should be synchronized with.
def int_rvgpu_sched_group_barrier : ClangBuiltin<"__builtin_rvgpu_sched_group_barrier">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
  [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, IntrNoMem, IntrHasSideEffects,
   IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

// Scheduler optimization hint.
//     MASK = 0: Small gemm opt
def int_rvgpu_iglp_opt : ClangBuiltin<"__builtin_rvgpu_iglp_opt">,
  Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>, IntrNoMem, IntrHasSideEffects, IntrConvergent,
                                IntrWillReturn, IntrNoCallback, IntrNoFree]>;

def int_rvgpu_s_waitcnt : ClangBuiltin<"__builtin_rvgpu_s_waitcnt">,
  Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>, IntrNoMem, IntrHasSideEffects, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

def int_rvgpu_div_scale : DefaultAttrsIntrinsic<
  // 1st parameter: Numerator
  // 2nd parameter: Denominator
  // 3rd parameter: Select quotient. Must equal Numerator or Denominator.
  //                (0 = Denominator, 1 = Numerator).
  [llvm_anyfloat_ty, llvm_i1_ty],
  [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i1_ty],
  [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<2>>]
>;

def int_rvgpu_div_fmas : DefaultAttrsIntrinsic<[llvm_anyfloat_ty],
  [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>, llvm_i1_ty],
  [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_div_fixup : DefaultAttrsIntrinsic<[llvm_anyfloat_ty],
  [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
  [IntrNoMem, IntrSpeculatable]
>;

// Look Up 2.0 / pi src0 with segment select src1[4:0]
def int_rvgpu_trig_preop : DefaultAttrsIntrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>, llvm_i32_ty],
  [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_sin : DefaultAttrsIntrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>],
  [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_cos : DefaultAttrsIntrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable]
>;

// v_log_{f16|f32}, performs log2. f32 version does not handle
// denormals. There is no reason to use this for f16 as it does
// support denormals, and the generic log2 intrinsic should be
// preferred.
def int_rvgpu_log : DefaultAttrsIntrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable]
>;

// v_exp_{f16|f32} (int_rvgpu_exp was taken by export
// already). Performs exp2. f32 version does not handle
// denormals. There is no reason to use this for f16 as it does
// support denormals, and the generic exp2 intrinsic should be
// preferred.
def int_rvgpu_exp2 : DefaultAttrsIntrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_log_clamp : DefaultAttrsIntrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_fmul_legacy : ClangBuiltin<"__builtin_rvgpu_fmul_legacy">,
  DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
  [IntrNoMem, IntrSpeculatable, Commutative]
>;

// Fused single-precision multiply-add with legacy behaviour for the multiply,
// which is that +/- 0.0 * anything (even NaN or infinity) is +0.0. This is
// intended for use on subtargets that have the v_fma_legacy_f32 and/or
// v_fmac_legacy_f32 instructions. (Note that v_fma_legacy_f16 is unrelated and
// has a completely different kind of legacy behaviour.)
def int_rvgpu_fma_legacy :
  DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty],
  [IntrNoMem, IntrSpeculatable, Commutative]
>;

def int_rvgpu_rcp : DefaultAttrsIntrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_rcp_legacy : ClangBuiltin<"__builtin_rvgpu_rcp_legacy">,
  DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_float_ty],
  [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_sqrt :  DefaultAttrsIntrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_rsq :  DefaultAttrsIntrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_rsq_legacy :  ClangBuiltin<"__builtin_rvgpu_rsq_legacy">,
  DefaultAttrsIntrinsic<
  [llvm_float_ty], [llvm_float_ty], [IntrNoMem, IntrSpeculatable]
>;

// out = 1.0 / sqrt(a) result clamped to +/- max_float.
def int_rvgpu_rsq_clamp : DefaultAttrsIntrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable]>;

def int_rvgpu_frexp_mant : DefaultAttrsIntrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_frexp_exp : DefaultAttrsIntrinsic<
  [llvm_anyint_ty], [llvm_anyfloat_ty], [IntrNoMem, IntrSpeculatable]
>;

// v_fract is buggy on SI/CI. It mishandles infinities, may return 1.0
// and always uses rtz, so is not suitable for implementing the OpenCL
// fract function. It should be ok on VI.
def int_rvgpu_fract : DefaultAttrsIntrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_cvt_pkrtz : ClangBuiltin<"__builtin_rvgpu_cvt_pkrtz">,
  DefaultAttrsIntrinsic<[llvm_v2f16_ty], [llvm_float_ty, llvm_float_ty],
            [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_cvt_pknorm_i16 :
  ClangBuiltin<"__builtin_rvgpu_cvt_pknorm_i16">,
  DefaultAttrsIntrinsic<[llvm_v2i16_ty], [llvm_float_ty, llvm_float_ty],
            [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_cvt_pknorm_u16 :
  ClangBuiltin<"__builtin_rvgpu_cvt_pknorm_u16">,
  DefaultAttrsIntrinsic<[llvm_v2i16_ty], [llvm_float_ty, llvm_float_ty],
            [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_cvt_pk_i16 :
    ClangBuiltin<"__builtin_rvgpu_cvt_pk_i16">,
    DefaultAttrsIntrinsic<
  [llvm_v2i16_ty], [llvm_i32_ty, llvm_i32_ty],
  [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_cvt_pk_u16 : ClangBuiltin<"__builtin_rvgpu_cvt_pk_u16">,
  DefaultAttrsIntrinsic<[llvm_v2i16_ty], [llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_class : DefaultAttrsIntrinsic<
  [llvm_i1_ty], [llvm_anyfloat_ty, llvm_i32_ty],
  [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_fmed3 :
  DefaultAttrsIntrinsic<[llvm_anyfloat_ty],
    [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
    [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_cubeid : ClangBuiltin<"__builtin_rvgpu_cubeid">,
  DefaultAttrsIntrinsic<[llvm_float_ty],
    [llvm_float_ty, llvm_float_ty, llvm_float_ty],
    [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_cubema : ClangBuiltin<"__builtin_rvgpu_cubema">,
  DefaultAttrsIntrinsic<[llvm_float_ty],
  [llvm_float_ty, llvm_float_ty, llvm_float_ty],
  [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_cubesc : ClangBuiltin<"__builtin_rvgpu_cubesc">,
  DefaultAttrsIntrinsic<[llvm_float_ty],
    [llvm_float_ty, llvm_float_ty, llvm_float_ty],
    [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_cubetc : ClangBuiltin<"__builtin_rvgpu_cubetc">,
  DefaultAttrsIntrinsic<[llvm_float_ty],
    [llvm_float_ty, llvm_float_ty, llvm_float_ty],
    [IntrNoMem, IntrSpeculatable]
>;

// v_ffbh_i32, as opposed to v_ffbh_u32. For v_ffbh_u32, llvm.ctlz
// should be used.
def int_rvgpu_sffbh :
  DefaultAttrsIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>],
  [IntrNoMem, IntrSpeculatable]
>;

// v_mad_f32|f16/v_mac_f32|f16, selected regardless of denorm support.
def int_rvgpu_fmad_ftz :
  DefaultAttrsIntrinsic<[llvm_anyfloat_ty],
            [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
            [IntrNoMem, IntrSpeculatable]
>;

class RVGPULDSIntrin :
  Intrinsic<[llvm_any_ty],
    [LLVMQualPointerType<3>,
    LLVMMatchType<0>,
    llvm_i32_ty, // ordering
    llvm_i32_ty, // scope
    llvm_i1_ty], // isVolatile
    [IntrArgMemOnly, IntrWillReturn, NoCapture<ArgIndex<0>>,
     ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, IntrNoCallback, IntrNoFree]
>;

// FIXME: The m0 argument should be moved after the normal arguments
class RVGPUDSOrderedIntrinsic : Intrinsic<
  [llvm_i32_ty],
  // M0 = {hi16:address, lo16:waveID}. Allow passing M0 as a pointer, so that
  // the bit packing can be optimized at the IR level.
  [LLVMQualPointerType<2>, // IntToPtr(M0)
   llvm_i32_ty, // value to add or swap
   llvm_i32_ty, // ordering
   llvm_i32_ty, // scope
   llvm_i1_ty,  // isVolatile
   llvm_i32_ty, // ordered count index (OA index), also added to the address
                // gfx10: bits 24-27 indicate the number of active threads/dwords
   llvm_i1_ty,  // wave release, usually set to 1
   llvm_i1_ty], // wave done, set to 1 for the last ordered instruction
  [IntrWillReturn, NoCapture<ArgIndex<0>>,
   ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>,
   ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>, IntrNoCallback, IntrNoFree
  ]
>;

class RVGPUDSAppendConsumedIntrinsic : Intrinsic<
  [llvm_i32_ty],
  [llvm_anyptr_ty, // LDS or GDS ptr
   llvm_i1_ty], // isVolatile
   [IntrConvergent, IntrWillReturn, IntrArgMemOnly,
    NoCapture<ArgIndex<0>>, ImmArg<ArgIndex<1>>, IntrNoCallback, IntrNoFree],
   "",
   [SDNPMemOperand]
>;

def int_rvgpu_ds_ordered_add : RVGPUDSOrderedIntrinsic;
def int_rvgpu_ds_ordered_swap : RVGPUDSOrderedIntrinsic;

// The pointer argument is assumed to be dynamically uniform if a VGPR.
def int_rvgpu_ds_append : RVGPUDSAppendConsumedIntrinsic;
def int_rvgpu_ds_consume : RVGPUDSAppendConsumedIntrinsic;

def int_rvgpu_ds_fadd : RVGPULDSIntrin;
def int_rvgpu_ds_fmin : RVGPULDSIntrin;
def int_rvgpu_ds_fmax : RVGPULDSIntrin;

} // TargetPrefix = "rvgpu"

// New-style image intrinsics

//////////////////////////////////////////////////////////////////////////
// Dimension-aware image intrinsics framework
//////////////////////////////////////////////////////////////////////////

// Helper class to represent (type, name) combinations of arguments. The
// argument names are explanatory and used as DAG operand names for codegen
// pattern matching.
class RVGPUArg<LLVMType ty, string name> {
  LLVMType Type = ty;
  string Name = name;
}

// Return [RVGPUArg<basety, names[0]>, RVGPUArg<LLVMMatchType<0>, names[1]>, ...]
class RVmakeArgList<list<string> names, LLVMType basety> {
  list<RVGPUArg> ret =
    !listconcat([RVGPUArg<basety, names[0]>],
                !foreach(name, !tail(names), RVGPUArg<LLVMMatchType<0>, name>));
}

// Return arglist, with LLVMMatchType's references shifted by 'shift'.
class RVarglistmatchshift<list<RVGPUArg> arglist, int shift> {
  list<RVGPUArg> ret =
    !foreach(arg, arglist,
             !if(!isa<LLVMMatchType>(arg.Type),
                 RVGPUArg<LLVMMatchType<!add(!cast<LLVMMatchType>(arg.Type).Number, shift)>,
                           arg.Name>,
                 arg));
}

// Return the concatenation of the given arglists. LLVMMatchType's are adjusted
// accordingly, and shifted by an additional 'shift'.
class RVarglistconcat<list<list<RVGPUArg>> arglists, int shift = 0> {
  list<RVGPUArg> ret =
    !foldl([]<RVGPUArg>, arglists, lhs, rhs,
           !listconcat(
             lhs,
             RVarglistmatchshift<rhs,
                               !add(shift, !foldl(0, lhs, a, b,
                                                  !add(a, b.Type.isAny)))>.ret));
}

// Represent texture/image types / dimensionality.
class RVGPUDimProps<bits<3> enc, string name, string asmsuffix,
                     list<string> coord_names, list<string> slice_names,
                     bit msaa = 0> {
  RVGPUDimProps Dim = !cast<RVGPUDimProps>(NAME);
  string Name = name; // e.g. "2darraymsaa"
  string AsmSuffix = asmsuffix; // e.g. 2D_MSAA_ARRAY (used in assembly strings)
  bits<3> Encoding = enc;
  bit DA = 0; // DA bit in MIMG encoding
  bit MSAA = msaa;

  list<RVGPUArg> CoordSliceArgs =
    RVmakeArgList<!listconcat(coord_names, slice_names), llvm_anyfloat_ty>.ret;
  list<RVGPUArg> CoordSliceIntArgs =
    RVmakeArgList<!listconcat(coord_names, slice_names), llvm_anyint_ty>.ret;
  list<RVGPUArg> GradientArgs =
    RVmakeArgList<!listconcat(!foreach(name, coord_names, "d" # name # "dh"),
                            !foreach(name, coord_names, "d" # name # "dv")),
                llvm_anyfloat_ty>.ret;

  bits<8> NumCoords = !size(CoordSliceArgs);
  bits<8> NumGradients = !size(GradientArgs);
}

def RVGPUDim1D : RVGPUDimProps<0x0, "1d", "1D", ["s"], []>;
def RVGPUDim2D : RVGPUDimProps<0x1, "2d", "2D", ["s", "t"], []>;
def RVGPUDim3D : RVGPUDimProps<0x2, "3d", "3D", ["s", "t", "r"], []>;
let DA = 1 in {
  def RVGPUDimCube : RVGPUDimProps<0x3, "cube", "CUBE", ["s", "t"], ["face"]>;
  def RVGPUDim1DArray : RVGPUDimProps<0x4, "1darray", "1D_ARRAY", ["s"], ["slice"]>;
  def RVGPUDim2DArray : RVGPUDimProps<0x5, "2darray", "2D_ARRAY", ["s", "t"], ["slice"]>;
}
def RVGPUDim2DMsaa : RVGPUDimProps<0x6, "2dmsaa", "2D_MSAA", ["s", "t"], ["fragid"], 1>;
let DA = 1 in {
  def RVGPUDim2DArrayMsaa : RVGPUDimProps<0x7, "2darraymsaa", "2D_MSAA_ARRAY", ["s", "t"], ["slice", "fragid"], 1>;
}

def RVGPUDims {
  list<RVGPUDimProps> NoMsaa = [RVGPUDim1D, RVGPUDim2D, RVGPUDim3D,
                                 RVGPUDimCube, RVGPUDim1DArray,
                                 RVGPUDim2DArray];
  list<RVGPUDimProps> Msaa = [RVGPUDim2DMsaa, RVGPUDim2DArrayMsaa];
  list<RVGPUDimProps> All = !listconcat(NoMsaa, Msaa);
}

// Represent sample variants, i.e. _C, _O, _B, ... and combinations thereof.
class RVGPUSampleVariant<string ucmod, string lcmod, list<RVGPUArg> extra_addr> {
  string UpperCaseMod = ucmod;
  string LowerCaseMod = lcmod;

  // {offset} {bias} {z-compare}
  list<RVGPUArg> ExtraAddrArgs = extra_addr;
  bit Offset = false;
  bit Bias = false;
  bit ZCompare = false;
  bit Gradients = false;

  // Name of the {lod} or {clamp} argument that is appended to the coordinates,
  // if any.
  string LodOrClamp = "";
}

// RVGPUSampleVariants: all variants supported by IMAGE_SAMPLE
// RVGPUSampleVariantsNoGradients: variants supported by IMAGE_GATHER4
defset list<RVGPUSampleVariant> RVGPUSampleVariants = {
  multiclass RVGPUSampleHelper_Offset<string ucmod, string lcmod,
                                       list<RVGPUArg> extra_addr> {
    def NAME#lcmod : RVGPUSampleVariant<ucmod, lcmod, extra_addr>;
    let Offset = true in
    def NAME#lcmod#_o : RVGPUSampleVariant<
        ucmod#"_O", lcmod#"_o", !listconcat([RVGPUArg<llvm_i32_ty, "offset">], extra_addr)>;
  }

  multiclass RVGPUSampleHelper_Compare<string ucmod, string lcmod,
                                        list<RVGPUArg> extra_addr> {
    defm NAME : RVGPUSampleHelper_Offset<ucmod, lcmod, extra_addr>;
    let ZCompare = true in
    defm NAME : RVGPUSampleHelper_Offset<
        "_C"#ucmod, "_c"#lcmod, !listconcat(extra_addr, [RVGPUArg<llvm_float_ty, "zcompare">])>;
  }

  multiclass RVGPUSampleHelper_Clamp<string ucmod, string lcmod,
                                      list<RVGPUArg> extra_addr> {
    defm NAME : RVGPUSampleHelper_Compare<ucmod, lcmod, extra_addr>;
    let LodOrClamp = "clamp" in
    defm NAME : RVGPUSampleHelper_Compare<ucmod#"_CL", lcmod#"_cl", extra_addr>;
  }

  defset list<RVGPUSampleVariant> RVGPUSampleVariantsNoGradients = {
    defm RVGPUSample : RVGPUSampleHelper_Clamp<"", "", []>;
    let Bias = true in
    defm RVGPUSample : RVGPUSampleHelper_Clamp<
        "_B", "_b", [RVGPUArg<llvm_anyfloat_ty, "bias">]>;
    let LodOrClamp = "lod" in
    defm RVGPUSample : RVGPUSampleHelper_Compare<"_L", "_l", []>;
    defm RVGPUSample : RVGPUSampleHelper_Compare<"_LZ", "_lz", []>;
  }

  let Gradients = true in {
    defm RVGPUSample : RVGPUSampleHelper_Clamp<"_D", "_d", []>;
    defm RVGPUSample : RVGPUSampleHelper_Clamp<"_CD", "_cd", []>;
  }
}

// Helper class to capture the profile of a dimension-aware image intrinsic.
// This information is used to generate the intrinsic's type and to inform
// codegen pattern matching.
class RVGPUDimProfile<string opmod,
                       RVGPUDimProps dim> {
  RVGPUDimProps Dim = dim;
  string OpMod = opmod; // the corresponding instruction is named IMAGE_OpMod

  // These are intended to be overwritten by subclasses
  bit IsSample = false;
  bit IsAtomic = false;
  list<LLVMType> RetTypes = [];
  list<RVGPUArg> DataArgs = [];
  list<RVGPUArg> ExtraAddrArgs = [];
  bit Offset = false;
  bit Bias = false;
  bit ZCompare = false;
  bit Gradients = false;
  string LodClampMip = "";

  int NumRetAndDataAnyTypes =
    !foldl(0, !listconcat(RetTypes, !foreach(arg, DataArgs, arg.Type)), a, b,
           !add(a, b.isAny));

  list<RVGPUArg> AddrArgs =
    RVarglistconcat<[ExtraAddrArgs,
                   !if(Gradients, dim.GradientArgs, []),
                   !listconcat(!if(IsSample, dim.CoordSliceArgs, dim.CoordSliceIntArgs),
                               !if(!empty(LodClampMip),
                                   []<RVGPUArg>,
                                   [RVGPUArg<LLVMMatchType<0>, LodClampMip>]))],
                  NumRetAndDataAnyTypes>.ret;
  list<LLVMType> AddrTypes = !foreach(arg, AddrArgs, arg.Type);
  list<RVGPUArg> AddrDefaultArgs =
    !foreach(arg, AddrArgs,
             RVGPUArg<!if(!or(arg.Type.isAny, !isa<LLVMMatchType>(arg.Type)),
                           !if(IsSample, llvm_float_ty, llvm_i32_ty), arg.Type),
                       arg.Name>);
  list<RVGPUArg> AddrA16Args =
    !foreach(arg, AddrArgs,
             RVGPUArg<!if(!or(arg.Type.isAny, !isa<LLVMMatchType>(arg.Type)),
                           !if(IsSample, llvm_half_ty, llvm_i16_ty), arg.Type),
                       arg.Name>);
}

class RVGPUDimProfileCopy<RVGPUDimProfile base> : RVGPUDimProfile<base.OpMod, base.Dim> {
  let IsSample = base.IsSample;
  let IsAtomic = base.IsAtomic;
  let RetTypes = base.RetTypes;
  let DataArgs = base.DataArgs;
  let ExtraAddrArgs = base.ExtraAddrArgs;
  let Offset = base.Offset;
  let Bias = base.Bias;
  let ZCompare = base.ZCompare;
  let Gradients = base.Gradients;
  let LodClampMip = base.LodClampMip;
}

class RVGPUDimSampleProfile<string opmod,
                             RVGPUDimProps dim,
                             RVGPUSampleVariant sample> : RVGPUDimProfile<opmod, dim> {
  let IsSample = true;
  let RetTypes = [llvm_any_ty];
  let ExtraAddrArgs = sample.ExtraAddrArgs;
  let Offset = sample.Offset;
  let Bias = sample.Bias;
  let ZCompare = sample.ZCompare;
  let Gradients = sample.Gradients;
  let LodClampMip = sample.LodOrClamp;
}

class RVGPUDimNoSampleProfile<string opmod,
                               RVGPUDimProps dim,
                               list<LLVMType> retty,
                               list<RVGPUArg> dataargs,
                               bit Mip = false> : RVGPUDimProfile<opmod, dim> {
  let RetTypes = retty;
  let DataArgs = dataargs;
  let LodClampMip = !if(Mip, "mip", "");
}

class RVGPUDimAtomicProfile<string opmod,
                             RVGPUDimProps dim,
                             list<RVGPUArg> dataargs> : RVGPUDimProfile<opmod, dim> {
  let RetTypes = [llvm_anyint_ty];
  let DataArgs = dataargs;
  let IsAtomic = true;
}

class RVGPUDimAtomicFloatProfile<string opmod, RVGPUDimProps dim,
                                  list<RVGPUArg> dataargs>
    : RVGPUDimAtomicProfile<opmod, dim, dataargs> {
  let RetTypes = [llvm_anyfloat_ty];
}

class RVGPUDimGetResInfoProfile<RVGPUDimProps dim>
    : RVGPUDimProfile<"GET_RESINFO", dim> {
  let RetTypes = [llvm_anyfloat_ty];
  let DataArgs = [];
  let AddrArgs = [RVGPUArg<llvm_anyint_ty, "mip">];
  let LodClampMip = "mip";
}

// Helper class for figuring out image intrinsic argument indexes.
class RVGPUImageDimIntrinsicEval<RVGPUDimProfile P_> {
  int NumDataArgs = !size(P_.DataArgs);
  int NumDmaskArgs = !not(P_.IsAtomic);
  int NumOffsetArgs = !if(P_.Offset, 1, 0);
  int NumBiasArgs = !if(P_.Bias, 1, 0);
  int NumZCompareArgs = !if(P_.ZCompare, 1, 0);
  int NumExtraAddrArgs = !add(NumOffsetArgs, NumBiasArgs, NumZCompareArgs);
  int NumVAddrArgs = !size(P_.AddrArgs);
  int NumGradientArgs = !if(P_.Gradients, !size(P_.Dim.GradientArgs), 0);
  int NumCoordArgs = !if(P_.IsSample, !size(P_.Dim.CoordSliceArgs), !size(P_.Dim.CoordSliceIntArgs));
  int NumRSrcArgs = 1;
  int NumSampArgs = !if(P_.IsSample, 2, 0);
  int DmaskArgIndex = NumDataArgs;
  int VAddrArgIndex = !add(DmaskArgIndex, NumDmaskArgs);
  int OffsetArgIndex = VAddrArgIndex;
  int BiasArgIndex = !add(VAddrArgIndex, NumOffsetArgs);
  int ZCompareArgIndex = !add(BiasArgIndex, NumBiasArgs);
  int GradientArgIndex = !add(VAddrArgIndex, NumExtraAddrArgs);
  int CoordArgIndex = !add(GradientArgIndex, NumGradientArgs);
  int LodArgIndex = !add(VAddrArgIndex, NumVAddrArgs, -1);
  int MipArgIndex = LodArgIndex;
  int RsrcArgIndex = !add(VAddrArgIndex, NumVAddrArgs);
  int SampArgIndex = !add(RsrcArgIndex, NumRSrcArgs);
  int UnormArgIndex = !add(SampArgIndex, 1);
  int TexFailCtrlArgIndex = !add(SampArgIndex, NumSampArgs);
  int CachePolicyArgIndex = !add(TexFailCtrlArgIndex, 1);
}

// All dimension-aware intrinsics are derived from this class.
class RVGPUImageDimIntrinsic<RVGPUDimProfile P_,
                              list<IntrinsicProperty> props,
                              list<SDNodeProperty> sdnodeprops> : DefaultAttrsIntrinsic<
    P_.RetTypes,        // vdata(VGPR) -- for load/atomic-with-return
    !listconcat(
      !foreach(arg, P_.DataArgs, arg.Type),      // vdata(VGPR) -- for store/atomic
      !if(P_.IsAtomic, [], [llvm_i32_ty]),       // dmask(imm)
      P_.AddrTypes,                              // vaddr(VGPR)
      [llvm_v8i32_ty],                           // rsrc(SGPR)
      !if(P_.IsSample, [llvm_v4i32_ty,           // samp(SGPR)
                        llvm_i1_ty], []),        // unorm(imm)
      [llvm_i32_ty,                              // texfailctrl(imm; bit 0 = tfe, bit 1 = lwe)
       llvm_i32_ty]),                            // cachepolicy(imm; bit 0 = glc, bit 1 = slc, bit 2 = dlc;
                                                 //   gfx12+ imm: bits [0-2] = th, bits [3-4] = scope)
                                                 // TODO-GFX12: Update all other cachepolicy descriptions.

     !listconcat(props,
          !if(P_.IsAtomic, [], [ImmArg<ArgIndex<RVGPUImageDimIntrinsicEval<P_>.DmaskArgIndex>>]),
          !if(P_.IsSample, [ImmArg<ArgIndex<RVGPUImageDimIntrinsicEval<P_>.UnormArgIndex>>], []),
          [ImmArg<ArgIndex<RVGPUImageDimIntrinsicEval<P_>.TexFailCtrlArgIndex>>,
           ImmArg<ArgIndex<RVGPUImageDimIntrinsicEval<P_>.CachePolicyArgIndex>>]),


      "", sdnodeprops>,
  RVGPURsrcIntrinsic<!add(!size(P_.DataArgs), !size(P_.AddrTypes),
                           !if(P_.IsAtomic, 0, 1)), 1> {
  RVGPUDimProfile P = P_;

  RVGPUImageDimIntrinsic Intr = !cast<RVGPUImageDimIntrinsic>(NAME);

  let TargetPrefix = "rvgpu";
}

// Marker class for intrinsics with a DMask that determines the returned
// channels.
class RVGPUImageDMaskIntrinsic;

defset list<RVGPUImageDimIntrinsic> RVGPUImageDimIntrinsics = {

  //////////////////////////////////////////////////////////////////////////
  // Load and store intrinsics
  //////////////////////////////////////////////////////////////////////////
  multiclass RVGPUImageDimIntrinsicsNoMsaa<string opmod,
                                            list<LLVMType> retty,
                                            list<RVGPUArg> dataargs,
                                            list<IntrinsicProperty> props,
                                            list<SDNodeProperty> sdnodeprops,
                                            bit Mip = false> {
    foreach dim = RVGPUDims.NoMsaa in {
      def !strconcat(NAME, "_", dim.Name)
        : RVGPUImageDimIntrinsic<
            RVGPUDimNoSampleProfile<opmod, dim, retty, dataargs, Mip>,
            props, sdnodeprops>;
    }
  }

  multiclass RVGPUImageDimIntrinsicsAll<string opmod,
                                         list<LLVMType> retty,
                                         list<RVGPUArg> dataargs,
                                         list<IntrinsicProperty> props,
                                         list<SDNodeProperty> sdnodeprops,
                                         bit Mip = false> {
    foreach dim = RVGPUDims.All in {
      def !strconcat(NAME, "_", dim.Name)
        : RVGPUImageDimIntrinsic<
            RVGPUDimNoSampleProfile<opmod, dim, retty, dataargs, Mip>,
            props, sdnodeprops>;
    }
  }

  defm int_rvgpu_image_load
    : RVGPUImageDimIntrinsicsAll<"LOAD", [llvm_any_ty], [], [IntrReadMem],
                                  [SDNPMemOperand]>,
      RVGPUImageDMaskIntrinsic;
  defm int_rvgpu_image_load_mip
    : RVGPUImageDimIntrinsicsNoMsaa<"LOAD_MIP", [llvm_any_ty], [],
                                     [IntrReadMem, IntrWillReturn], [SDNPMemOperand], 1>,
      RVGPUImageDMaskIntrinsic;

  defm int_rvgpu_image_store : RVGPUImageDimIntrinsicsAll<
              "STORE", [], [RVGPUArg<llvm_anyfloat_ty, "vdata">],
              [IntrWriteMem, IntrWillReturn], [SDNPMemOperand]>,
              RVGPUImageDMaskIntrinsic;
  defm int_rvgpu_image_store_mip : RVGPUImageDimIntrinsicsNoMsaa<
              "STORE_MIP", [], [RVGPUArg<llvm_anyfloat_ty, "vdata">],
              [IntrWriteMem, IntrWillReturn], [SDNPMemOperand], 1>,
              RVGPUImageDMaskIntrinsic;

  //////////////////////////////////////////////////////////////////////////
  // MSAA intrinsics
  //////////////////////////////////////////////////////////////////////////
  foreach dim = RVGPUDims.Msaa in {
    def int_rvgpu_image_msaa_load_x # _ # dim.Name:
        RVGPUImageDimIntrinsic<
            RVGPUDimNoSampleProfile<"MSAA_LOAD_X", dim, [llvm_any_ty], []>,
            [IntrReadMem], [SDNPMemOperand]>;
  }

  foreach dim = RVGPUDims.Msaa in {
    def int_rvgpu_image_msaa_load # _ # dim.Name:
        RVGPUImageDimIntrinsic<
            RVGPUDimNoSampleProfile<"MSAA_LOAD", dim, [llvm_any_ty], []>,
            [IntrReadMem], [SDNPMemOperand]>;
  }

  //////////////////////////////////////////////////////////////////////////
  // sample and getlod intrinsics
  //////////////////////////////////////////////////////////////////////////
  multiclass RVGPUImageDimSampleDims<string opmod,
                                      RVGPUSampleVariant sample,
                                      bit NoMem = false> {
    foreach dim = RVGPUDims.NoMsaa in {
      def !strconcat(NAME, "_", dim.Name) : RVGPUImageDimIntrinsic<
          RVGPUDimSampleProfile<opmod, dim, sample>,
          !if(NoMem, [IntrNoMem], [IntrReadMem]),
          !if(NoMem, [], [SDNPMemOperand])>;
    }
  }

  foreach sample = RVGPUSampleVariants in {
    defm int_rvgpu_image_sample # sample.LowerCaseMod
      : RVGPUImageDimSampleDims<"SAMPLE" # sample.UpperCaseMod, sample>,
        RVGPUImageDMaskIntrinsic;
  }

  defm int_rvgpu_image_getlod
    : RVGPUImageDimSampleDims<"GET_LOD", RVGPUSample, 1>,
      RVGPUImageDMaskIntrinsic;

  //////////////////////////////////////////////////////////////////////////
  // getresinfo intrinsics
  //////////////////////////////////////////////////////////////////////////
  foreach dim = RVGPUDims.All in {
    def !strconcat("int_rvgpu_image_getresinfo_", dim.Name)
      : RVGPUImageDimIntrinsic<RVGPUDimGetResInfoProfile<dim>, [IntrNoMem], []>,
        RVGPUImageDMaskIntrinsic;
  }

  //////////////////////////////////////////////////////////////////////////
  // gather4 intrinsics
  //////////////////////////////////////////////////////////////////////////
  foreach sample = RVGPUSampleVariantsNoGradients in {
    foreach dim = [RVGPUDim2D, RVGPUDimCube, RVGPUDim2DArray] in {
      def int_rvgpu_image_gather4 # sample.LowerCaseMod # _ # dim.Name:
          RVGPUImageDimIntrinsic<
              RVGPUDimSampleProfile<"GATHER4" # sample.UpperCaseMod, dim, sample>,
              [IntrReadMem], [SDNPMemOperand]>;
    }
  }
}

//////////////////////////////////////////////////////////////////////////
// atomic intrinsics
//////////////////////////////////////////////////////////////////////////
defset list<RVGPUImageDimIntrinsic> RVGPUImageDimAtomicIntrinsics = {
  multiclass RVGPUImageDimAtomicX<string opmod, list<RVGPUArg> dataargs,
                                   int isFloat = 0> {
        foreach dim = RVGPUDims.All in {
          def !strconcat(NAME, "_", dim.Name): RVGPUImageDimIntrinsic<
              !if (isFloat, RVGPUDimAtomicFloatProfile<opmod, dim, dataargs>,
                   RVGPUDimAtomicProfile<opmod, dim, dataargs>),
              [], [SDNPMemOperand]>;
        }
  }

  multiclass RVGPUImageDimAtomic<string opmod, int isFloat = 0> {
    defm ""
        : RVGPUImageDimAtomicX<opmod, [RVGPUArg<LLVMMatchType<0>, "vdata">],
                                isFloat>;
  }

  multiclass RVGPUImageDimFloatAtomic<string opmod> {
    defm "" : RVGPUImageDimAtomic<opmod, 1 /*isFloat*/>;
  }

  defm int_rvgpu_image_atomic_swap : RVGPUImageDimAtomic<"ATOMIC_SWAP">;
  defm int_rvgpu_image_atomic_add : RVGPUImageDimAtomic<"ATOMIC_ADD">;
  defm int_rvgpu_image_atomic_sub : RVGPUImageDimAtomic<"ATOMIC_SUB">;
  defm int_rvgpu_image_atomic_smin : RVGPUImageDimAtomic<"ATOMIC_SMIN">;
  defm int_rvgpu_image_atomic_umin : RVGPUImageDimAtomic<"ATOMIC_UMIN">;
  defm int_rvgpu_image_atomic_fmin : RVGPUImageDimFloatAtomic<"ATOMIC_FMIN">;
  defm int_rvgpu_image_atomic_smax : RVGPUImageDimAtomic<"ATOMIC_SMAX">;
  defm int_rvgpu_image_atomic_umax : RVGPUImageDimAtomic<"ATOMIC_UMAX">;
  defm int_rvgpu_image_atomic_fmax : RVGPUImageDimFloatAtomic<"ATOMIC_FMAX">;
  defm int_rvgpu_image_atomic_and : RVGPUImageDimAtomic<"ATOMIC_AND">;
  defm int_rvgpu_image_atomic_or : RVGPUImageDimAtomic<"ATOMIC_OR">;
  defm int_rvgpu_image_atomic_xor : RVGPUImageDimAtomic<"ATOMIC_XOR">;
  defm int_rvgpu_image_atomic_inc : RVGPUImageDimAtomic<"ATOMIC_INC">;
  defm int_rvgpu_image_atomic_dec : RVGPUImageDimAtomic<"ATOMIC_DEC">;

  defm int_rvgpu_image_atomic_cmpswap :
      RVGPUImageDimAtomicX<"ATOMIC_CMPSWAP", [RVGPUArg<LLVMMatchType<0>, "src">,
                                               RVGPUArg<LLVMMatchType<0>, "cmp">]>;
}

//////////////////////////////////////////////////////////////////////////
// Buffer intrinsics
//////////////////////////////////////////////////////////////////////////

// Data type for buffer resources (V#). Maybe, in the future, we can create a
// similar one for textures (T#).
def RVGPUBufferRsrcTy : LLVMQualPointerType<8>;

let TargetPrefix = "rvgpu" in {

def int_rvgpu_make_buffer_rsrc : DefaultAttrsIntrinsic <
  [RVGPUBufferRsrcTy],
  [llvm_anyptr_ty, // base
   llvm_i16_ty,    // stride (and swizzle control)
   llvm_i32_ty,    // NumRecords / extent
   llvm_i32_ty],   // flags
  // Attributes lifted from ptrmask + some extra argument attributes.
  [IntrNoMem, ReadNone<ArgIndex<0>>,
   IntrSpeculatable, IntrWillReturn]>;

defset list<RVGPURsrcIntrinsic> RVGPUBufferIntrinsics = {

class RVGPUBufferLoad<LLVMType data_ty = llvm_any_ty> : DefaultAttrsIntrinsic <
  [data_ty],
  [llvm_v4i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(SGPR/VGPR/imm)
   llvm_i1_ty,        // glc(imm)
   llvm_i1_ty],       // slc(imm)
  [IntrReadMem, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<0>;
def int_rvgpu_buffer_load_format : RVGPUBufferLoad<llvm_anyfloat_ty>;
def int_rvgpu_buffer_load : RVGPUBufferLoad;

// Generate a buffer_load instruction that may be optimized to s_buffer_load if
// the offset argument is uniform.
def int_rvgpu_s_buffer_load : DefaultAttrsIntrinsic <
  [llvm_any_ty],
  [llvm_v4i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // byte offset
   llvm_i32_ty],      // cachepolicy(imm; bit 0 = glc, bit 2 = dlc)
  [IntrNoMem, ImmArg<ArgIndex<2>>]>,
  RVGPURsrcIntrinsic<0>;

class RVGPUBufferStore<LLVMType data_ty = llvm_any_ty> : DefaultAttrsIntrinsic <
  [],
  [data_ty,          // vdata(VGPR)
   llvm_v4i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(SGPR/VGPR/imm)
   llvm_i1_ty,        // glc(imm)
   llvm_i1_ty],       // slc(imm)
  [IntrWriteMem, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<1>;
def int_rvgpu_buffer_store_format : RVGPUBufferStore<llvm_anyfloat_ty>;
def int_rvgpu_buffer_store : RVGPUBufferStore;

// New buffer intrinsics with separate raw and struct variants.  The raw
// variant never has an index. The struct variant always has an index, even if
// it is const 0. A struct intrinsic with constant 0 index is different to the
// corresponding raw intrinsic on gfx9+ because the behavior of bound checking
// and swizzling changes depending on whether idxen is set in the instruction.
// These new instrinsics also keep the offset and soffset arguments separate as
// they behave differently in bounds checking and swizzling.

// The versions of these intrinsics that take <4 x i32> arguments are deprecated
// in favor of their .ptr.buffer variants that take ptr addrspace(8) arguments,
// which allow for improved reasoning about memory accesses.
class RVGPURawBufferLoad<LLVMType data_ty = llvm_any_ty> : DefaultAttrsIntrinsic <
  [data_ty],
  [llvm_v4i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,       // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],      // auxiliary data (imm, cachepolicy     (bit 0 = glc,
                      //                                       bit 1 = slc,
                      //                                       bit 2 = dlc on gfx10+),
                      //                      swizzled buffer (bit 3 = swz))
  [IntrReadMem, ImmArg<ArgIndex<3>>], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<0>;
def int_rvgpu_raw_buffer_load_format : RVGPURawBufferLoad<llvm_anyfloat_ty>;
def int_rvgpu_raw_buffer_load : RVGPURawBufferLoad;

class RVGPURawPtrBufferLoad<LLVMType data_ty = llvm_any_ty> : DefaultAttrsIntrinsic <
  [data_ty],
  [RVGPUBufferRsrcTy,         // rsrc(SGPR)
   llvm_i32_ty,                 // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,                 // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],                // auxiliary data (imm, cachepolicy (bit 0 = glc,
                                //                                   bit 1 = slc,
                                //                                   bit 2 = dlc on gfx10+),
                                //                      swizzled buffer (bit 3 = swz))
  [IntrArgMemOnly, IntrReadMem, ReadOnly<ArgIndex<0>>, NoCapture<ArgIndex<0>>,
  ImmArg<ArgIndex<3>>], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<0>;
def int_rvgpu_raw_ptr_buffer_load_format : RVGPURawPtrBufferLoad<llvm_anyfloat_ty>;
def int_rvgpu_raw_ptr_buffer_load : RVGPURawPtrBufferLoad;

class RVGPUStructBufferLoad<LLVMType data_ty = llvm_any_ty> : DefaultAttrsIntrinsic <
  [data_ty],
  [llvm_v4i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,       // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],      // auxiliary data (imm, cachepolicy     (bit 0 = glc,
                      //                                       bit 1 = slc,
                      //                                       bit 2 = dlc on gfx10+),
                      //                      swizzled buffer (bit 3 = swz))
  [IntrReadMem, ImmArg<ArgIndex<4>>], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<0>;
def int_rvgpu_struct_buffer_load_format : RVGPUStructBufferLoad;
def int_rvgpu_struct_buffer_load : RVGPUStructBufferLoad;

class RVGPUStructPtrBufferLoad<LLVMType data_ty = llvm_any_ty> : DefaultAttrsIntrinsic <
  [data_ty],
  [RVGPUBufferRsrcTy,          // rsrc(SGPR)
   llvm_i32_ty,                 // vindex(VGPR)
   llvm_i32_ty,                 // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,                 // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],                // auxiliary data (imm, cachepolicy (bit 0 = glc,
                                //                                   bit 1 = slc,
                                //                                   bit 2 = dlc on gfx10+),
                                //                      swizzled buffer (bit 3 = swz))
  [IntrArgMemOnly, IntrReadMem, ReadOnly<ArgIndex<0>>, NoCapture<ArgIndex<0>>,
   ImmArg<ArgIndex<4>>], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<0>;
def int_rvgpu_struct_ptr_buffer_load_format : RVGPUStructPtrBufferLoad;
def int_rvgpu_struct_ptr_buffer_load : RVGPUStructPtrBufferLoad;

class RVGPURawBufferStore<LLVMType data_ty = llvm_any_ty> : DefaultAttrsIntrinsic <
  [],
  [data_ty,           // vdata(VGPR)
   llvm_v4i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,       // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],      // auxiliary data (imm, cachepolicy     (bit 0 = glc,
                      //                                       bit 1 = slc,
                      //                                       bit 2 = dlc on gfx10+),
                      //                      swizzled buffer (bit 3 = swz))
  [IntrWriteMem, ImmArg<ArgIndex<4>>], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<1>;
def int_rvgpu_raw_buffer_store_format : RVGPURawBufferStore<llvm_anyfloat_ty>;
def int_rvgpu_raw_buffer_store : RVGPURawBufferStore;

class RVGPURawPtrBufferStore<LLVMType data_ty = llvm_any_ty> : DefaultAttrsIntrinsic <
  [],
  [data_ty,                     // vdata(VGPR)
   RVGPUBufferRsrcTy,          // rsrc(SGPR)
   llvm_i32_ty,                 // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,                 // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],                // auxiliary data (imm, cachepolicy (bit 0 = glc,
                                //                                   bit 1 = slc,
                                //                                   bit 2 = dlc on gfx10+),
                                //                      swizzled buffer (bit 3 = swz))
  [IntrArgMemOnly, IntrWriteMem, WriteOnly<ArgIndex<1>>, NoCapture<ArgIndex<1>>,
  ImmArg<ArgIndex<4>>], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<1>;
def int_rvgpu_raw_ptr_buffer_store_format : RVGPURawPtrBufferStore<llvm_anyfloat_ty>;
def int_rvgpu_raw_ptr_buffer_store : RVGPURawPtrBufferStore;

class RVGPUStructBufferStore<LLVMType data_ty = llvm_any_ty> : DefaultAttrsIntrinsic <
  [],
  [data_ty,           // vdata(VGPR)
   llvm_v4i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,       // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],      // auxiliary data (imm, cachepolicy     (bit 0 = glc,
                      //                                       bit 1 = slc,
                      //                                       bit 2 = dlc on gfx10+),
                      //                      swizzled buffer (bit 3 = swz))
  [IntrWriteMem, ImmArg<ArgIndex<5>>], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<1>;
def int_rvgpu_struct_buffer_store_format : RVGPUStructBufferStore;
def int_rvgpu_struct_buffer_store : RVGPUStructBufferStore;

class RVGPUStructPtrBufferStore<LLVMType data_ty = llvm_any_ty> : DefaultAttrsIntrinsic <
  [],
  [data_ty,                     // vdata(VGPR)
   RVGPUBufferRsrcTy,          // rsrc(SGPR)
   llvm_i32_ty,                 // vindex(VGPR)
   llvm_i32_ty,                 // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,                 // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],                // auxiliary data (imm, cachepolicy (bit 0 = glc,
                                //                                   bit 1 = slc,
                                //                                   bit 2 = dlc on gfx10+),
                                //                      swizzled buffer (bit 3 = swz))
  [IntrArgMemOnly, IntrWriteMem, WriteOnly<ArgIndex<1>>, NoCapture<ArgIndex<1>>,
   ImmArg<ArgIndex<5>>], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<1>;
def int_rvgpu_struct_ptr_buffer_store_format : RVGPUStructPtrBufferStore;
def int_rvgpu_struct_ptr_buffer_store : RVGPUStructPtrBufferStore;

class RVGPURawBufferAtomic<LLVMType data_ty = llvm_any_ty> : Intrinsic <
  [data_ty],
  [LLVMMatchType<0>,  // vdata(VGPR)
   llvm_v4i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,       // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],      // cachepolicy(imm; bit 1 = slc)
  [ImmArg<ArgIndex<4>>, IntrWillReturn, IntrNoCallback, IntrNoFree], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<1, 0>;
def int_rvgpu_raw_buffer_atomic_swap : RVGPURawBufferAtomic;
def int_rvgpu_raw_buffer_atomic_add : RVGPURawBufferAtomic;
def int_rvgpu_raw_buffer_atomic_sub : RVGPURawBufferAtomic;
def int_rvgpu_raw_buffer_atomic_smin : RVGPURawBufferAtomic;
def int_rvgpu_raw_buffer_atomic_umin : RVGPURawBufferAtomic;
def int_rvgpu_raw_buffer_atomic_fmin : RVGPURawBufferAtomic<llvm_anyfloat_ty>;
def int_rvgpu_raw_buffer_atomic_smax : RVGPURawBufferAtomic;
def int_rvgpu_raw_buffer_atomic_umax : RVGPURawBufferAtomic;
def int_rvgpu_raw_buffer_atomic_fmax : RVGPURawBufferAtomic<llvm_anyfloat_ty>;
def int_rvgpu_raw_buffer_atomic_and : RVGPURawBufferAtomic;
def int_rvgpu_raw_buffer_atomic_or : RVGPURawBufferAtomic;
def int_rvgpu_raw_buffer_atomic_xor : RVGPURawBufferAtomic;
def int_rvgpu_raw_buffer_atomic_inc : RVGPURawBufferAtomic;
def int_rvgpu_raw_buffer_atomic_dec : RVGPURawBufferAtomic;
def int_rvgpu_raw_buffer_atomic_cmpswap : Intrinsic<
  [llvm_anyint_ty],
  [LLVMMatchType<0>,  // src(VGPR)
   LLVMMatchType<0>,  // cmp(VGPR)
   llvm_v4i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,       // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],      // cachepolicy(imm; bit 1 = slc)
  [ImmArg<ArgIndex<5>>, IntrWillReturn, IntrNoCallback, IntrNoFree], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<2, 0>;

class RVGPURawPtrBufferAtomic<LLVMType data_ty = llvm_any_ty> : Intrinsic <
  [data_ty],
  [LLVMMatchType<0>,            // vdata(VGPR)
   RVGPUBufferRsrcTy,          // rsrc(SGPR)
   llvm_i32_ty,                 // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,                 // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],                // cachepolicy(imm; bit 1 = slc)
  [IntrArgMemOnly, NoCapture<ArgIndex<1>>,
   ImmArg<ArgIndex<4>>, IntrWillReturn, IntrNoCallback, IntrNoFree], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<1, 0>;

def int_rvgpu_raw_ptr_buffer_atomic_swap : RVGPURawPtrBufferAtomic;
def int_rvgpu_raw_ptr_buffer_atomic_add : RVGPURawPtrBufferAtomic;
def int_rvgpu_raw_ptr_buffer_atomic_sub : RVGPURawPtrBufferAtomic;
def int_rvgpu_raw_ptr_buffer_atomic_smin : RVGPURawPtrBufferAtomic;
def int_rvgpu_raw_ptr_buffer_atomic_umin : RVGPURawPtrBufferAtomic;
def int_rvgpu_raw_ptr_buffer_atomic_fmin : RVGPURawPtrBufferAtomic<llvm_anyfloat_ty>;
def int_rvgpu_raw_ptr_buffer_atomic_smax : RVGPURawPtrBufferAtomic;
def int_rvgpu_raw_ptr_buffer_atomic_umax : RVGPURawPtrBufferAtomic;
def int_rvgpu_raw_ptr_buffer_atomic_fmax : RVGPURawPtrBufferAtomic<llvm_anyfloat_ty>;
def int_rvgpu_raw_ptr_buffer_atomic_and : RVGPURawPtrBufferAtomic;
def int_rvgpu_raw_ptr_buffer_atomic_or : RVGPURawPtrBufferAtomic;
def int_rvgpu_raw_ptr_buffer_atomic_xor : RVGPURawPtrBufferAtomic;
def int_rvgpu_raw_ptr_buffer_atomic_inc : RVGPURawPtrBufferAtomic;
def int_rvgpu_raw_ptr_buffer_atomic_dec : RVGPURawPtrBufferAtomic;
def int_rvgpu_raw_ptr_buffer_atomic_cmpswap : Intrinsic<
  [llvm_anyint_ty],
  [LLVMMatchType<0>,  // src(VGPR)
   LLVMMatchType<0>,  // cmp(VGPR)
   RVGPUBufferRsrcTy, // rsrc(SGPR)
   llvm_i32_ty,       // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,       // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],      // cachepolicy(imm; bit 1 = slc)
  [IntrArgMemOnly, NoCapture<ArgIndex<2>>,
   ImmArg<ArgIndex<5>>, IntrWillReturn, IntrNoCallback, IntrNoFree], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<2, 0>;

// gfx908 intrinsic
def int_rvgpu_raw_buffer_atomic_fadd : RVGPURawBufferAtomic<llvm_anyfloat_ty>;
def int_rvgpu_raw_ptr_buffer_atomic_fadd : RVGPURawPtrBufferAtomic<llvm_anyfloat_ty>;

class RVGPUStructBufferAtomic<LLVMType data_ty = llvm_any_ty> : Intrinsic <
  [data_ty],
  [LLVMMatchType<0>,  // vdata(VGPR)
   llvm_v4i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,       // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],      // cachepolicy(imm; bit 1 = slc)
  [ImmArg<ArgIndex<5>>, IntrWillReturn, IntrNoCallback, IntrNoFree], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<1, 0>;
def int_rvgpu_struct_buffer_atomic_swap : RVGPUStructBufferAtomic;
def int_rvgpu_struct_buffer_atomic_add : RVGPUStructBufferAtomic;
def int_rvgpu_struct_buffer_atomic_sub : RVGPUStructBufferAtomic;
def int_rvgpu_struct_buffer_atomic_smin : RVGPUStructBufferAtomic;
def int_rvgpu_struct_buffer_atomic_umin : RVGPUStructBufferAtomic;
def int_rvgpu_struct_buffer_atomic_smax : RVGPUStructBufferAtomic;
def int_rvgpu_struct_buffer_atomic_umax : RVGPUStructBufferAtomic;
def int_rvgpu_struct_buffer_atomic_and : RVGPUStructBufferAtomic;
def int_rvgpu_struct_buffer_atomic_or : RVGPUStructBufferAtomic;
def int_rvgpu_struct_buffer_atomic_xor : RVGPUStructBufferAtomic;
def int_rvgpu_struct_buffer_atomic_inc : RVGPUStructBufferAtomic;
def int_rvgpu_struct_buffer_atomic_dec : RVGPUStructBufferAtomic;
def int_rvgpu_struct_buffer_atomic_cmpswap : Intrinsic<
  [llvm_anyint_ty],
  [LLVMMatchType<0>,  // src(VGPR)
   LLVMMatchType<0>,  // cmp(VGPR)
   llvm_v4i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,       // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],      // cachepolicy(imm; bit 1 = slc)
  [ImmArg<ArgIndex<6>>, IntrWillReturn, IntrNoCallback, IntrNoFree], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<2, 0>;

class RVGPUStructPtrBufferAtomic<LLVMType data_ty = llvm_any_ty> : Intrinsic <
  [data_ty],
  [LLVMMatchType<0>,            // vdata(VGPR)
   RVGPUBufferRsrcTy,          // rsrc(SGPR)
   llvm_i32_ty,                 // vindex(VGPR)
   llvm_i32_ty,                 // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,                 // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],                // cachepolicy(imm; bit 1 = slc)
  [IntrArgMemOnly, NoCapture<ArgIndex<1>>,
   ImmArg<ArgIndex<5>>, IntrWillReturn, IntrNoCallback, IntrNoFree], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<1, 0>;
def int_rvgpu_struct_ptr_buffer_atomic_swap : RVGPUStructPtrBufferAtomic;
def int_rvgpu_struct_ptr_buffer_atomic_add : RVGPUStructPtrBufferAtomic;
def int_rvgpu_struct_ptr_buffer_atomic_sub : RVGPUStructPtrBufferAtomic;
def int_rvgpu_struct_ptr_buffer_atomic_smin : RVGPUStructPtrBufferAtomic;
def int_rvgpu_struct_ptr_buffer_atomic_umin : RVGPUStructPtrBufferAtomic;
def int_rvgpu_struct_ptr_buffer_atomic_smax : RVGPUStructPtrBufferAtomic;
def int_rvgpu_struct_ptr_buffer_atomic_umax : RVGPUStructPtrBufferAtomic;
def int_rvgpu_struct_ptr_buffer_atomic_and : RVGPUStructPtrBufferAtomic;
def int_rvgpu_struct_ptr_buffer_atomic_or : RVGPUStructPtrBufferAtomic;
def int_rvgpu_struct_ptr_buffer_atomic_xor : RVGPUStructPtrBufferAtomic;
def int_rvgpu_struct_ptr_buffer_atomic_inc : RVGPUStructPtrBufferAtomic;
def int_rvgpu_struct_ptr_buffer_atomic_dec : RVGPUStructPtrBufferAtomic;
def int_rvgpu_struct_ptr_buffer_atomic_cmpswap : Intrinsic<
  [llvm_anyint_ty],
  [LLVMMatchType<0>,  // src(VGPR)
   LLVMMatchType<0>,  // cmp(VGPR)
   RVGPUBufferRsrcTy, // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,       // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],      // cachepolicy(imm; bit 1 = slc)
  [IntrArgMemOnly, NoCapture<ArgIndex<2>>,
   ImmArg<ArgIndex<6>>, IntrWillReturn, IntrNoCallback, IntrNoFree], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<2, 0>;

// gfx908 intrinsic
def int_rvgpu_struct_buffer_atomic_fadd : RVGPUStructBufferAtomic<llvm_anyfloat_ty>;
def int_rvgpu_struct_ptr_buffer_atomic_fadd : RVGPUStructPtrBufferAtomic<llvm_anyfloat_ty>;

// gfx90a intrinsics
def int_rvgpu_struct_buffer_atomic_fmin : RVGPUStructBufferAtomic<llvm_anyfloat_ty>;
def int_rvgpu_struct_buffer_atomic_fmax : RVGPUStructBufferAtomic<llvm_anyfloat_ty>;

def int_rvgpu_struct_ptr_buffer_atomic_fmin : RVGPUStructPtrBufferAtomic<llvm_anyfloat_ty>;
def int_rvgpu_struct_ptr_buffer_atomic_fmax : RVGPUStructPtrBufferAtomic<llvm_anyfloat_ty>;

// Obsolescent tbuffer intrinsics.
def int_rvgpu_tbuffer_load : DefaultAttrsIntrinsic <
    [llvm_any_ty],    // overloaded for types f32/i32, v2f32/v2i32, v4f32/v4i32
    [llvm_v4i32_ty,   // rsrc(SGPR)
     llvm_i32_ty,     // vindex(VGPR)
     llvm_i32_ty,     // voffset(VGPR)
     llvm_i32_ty,     // soffset(SGPR)
     llvm_i32_ty,     // offset(imm)
     llvm_i32_ty,     // dfmt(imm)
     llvm_i32_ty,     // nfmt(imm)
     llvm_i1_ty,     // glc(imm)
     llvm_i1_ty],    // slc(imm)
    [IntrReadMem,
     ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>,
     ImmArg<ArgIndex<7>>, ImmArg<ArgIndex<8>>], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<0>;

def int_rvgpu_tbuffer_store : DefaultAttrsIntrinsic <
    [],
    [llvm_any_ty,    // vdata(VGPR), overloaded for types f32/i32, v2f32/v2i32, v4f32/v4i32
     llvm_v4i32_ty,  // rsrc(SGPR)
     llvm_i32_ty,    // vindex(VGPR)
     llvm_i32_ty,    // voffset(VGPR)
     llvm_i32_ty,    // soffset(SGPR)
     llvm_i32_ty,    // offset(imm)
     llvm_i32_ty,    // dfmt(imm)
     llvm_i32_ty,    // nfmt(imm)
     llvm_i1_ty,     // glc(imm)
     llvm_i1_ty],    // slc(imm)
    [IntrWriteMem, ImmArg<ArgIndex<5>>,
     ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>,
     ImmArg<ArgIndex<8>>, ImmArg<ArgIndex<9>>], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<1>;

// New tbuffer intrinsics, with:
// - raw and struct variants
// - joint format field
// - joint cachepolicy field
def int_rvgpu_raw_tbuffer_load : DefaultAttrsIntrinsic <
    [llvm_any_ty],    // overloaded for types f32/i32, v2f32/v2i32, v4f32/v4i32
    [llvm_v4i32_ty,   // rsrc(SGPR)
     llvm_i32_ty,     // offset(VGPR/imm, included in bounds checking and swizzling)
     llvm_i32_ty,     // soffset(SGPR/imm, excluded from bounds checking and swizzling)
     llvm_i32_ty,     // format(imm; bits 3..0 = dfmt, bits 6..4 = nfmt)
     llvm_i32_ty],    // auxiliary data (imm, cachepolicy     (bit 0 = glc,
                      //                                       bit 1 = slc,
                      //                                       bit 2 = dlc on gfx10+),
                      //                      swizzled buffer (bit 3 = swz))
    [IntrReadMem,
     ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<0>;

def int_rvgpu_raw_ptr_tbuffer_load : DefaultAttrsIntrinsic <
    [llvm_any_ty],      // overloaded for types f32/i32, v2f32/v2i32, v4f32/v4i32
    [RVGPUBufferRsrcTy, // rsrc(SGPR)
     llvm_i32_ty,     // offset(VGPR/imm, included in bounds` checking and swizzling)
     llvm_i32_ty,     // soffset(SGPR/imm, excluded from bounds checking and swizzling)
     llvm_i32_ty,     // format(imm; bits 3..0 = dfmt, bits 6..4 = nfmt)
     llvm_i32_ty],    // auxiliary data (imm, cachepolicy     (bit 0 = glc,
                      //                                       bit 1 = slc,
                      //                                       bit 2 = dlc on gfx10+),
                      //                      swizzled buffer (bit 3 = swz))
    [IntrArgMemOnly, IntrReadMem, ReadOnly<ArgIndex<0>>, NoCapture<ArgIndex<0>>,
     ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<0>;

def int_rvgpu_raw_tbuffer_store : DefaultAttrsIntrinsic <
    [],
    [llvm_any_ty,    // vdata(VGPR), overloaded for types f32/i32, v2f32/v2i32, v4f32/v4i32
     llvm_v4i32_ty,  // rsrc(SGPR)
     llvm_i32_ty,    // offset(VGPR/imm, included in bounds checking and swizzling)
     llvm_i32_ty,    // soffset(SGPR/imm, excluded from bounds checking and swizzling)
     llvm_i32_ty,    // format(imm; bits 3..0 = dfmt, bits 6..4 = nfmt)
     llvm_i32_ty],   // auxiliary data (imm, cachepolicy     (bit 0 = glc,
                     //                                       bit 1 = slc,
                     //                                       bit 2 = dlc on gfx10+),
                     //                      swizzled buffer (bit 3 = swz))
    [IntrWriteMem,
     ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<1>;

def int_rvgpu_raw_ptr_tbuffer_store : DefaultAttrsIntrinsic <
    [],
    [llvm_any_ty,    // vdata(VGPR), overloaded for types f32/i32, v2f32/v2i32, v4f32/v4i32
     RVGPUBufferRsrcTy, // rsrc(SGPR)
     llvm_i32_ty,    // offset(VGPR/imm, included in bounds checking and swizzling)
     llvm_i32_ty,    // soffset(SGPR/imm, excluded from bounds checking and swizzling)
     llvm_i32_ty,    // format(imm; bits 3..0 = dfmt, bits 6..4 = nfmt)
     llvm_i32_ty],   // auxiliary data (imm, cachepolicy     (bit 0 = glc,
                     //                                       bit 1 = slc,
                     //                                       bit 2 = dlc on gfx10+),
                     //                      swizzled buffer (bit 3 = swz))
    [IntrArgMemOnly, IntrWriteMem, WriteOnly<ArgIndex<1>>, NoCapture<ArgIndex<1>>,
     ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<1>;

def int_rvgpu_struct_tbuffer_load : DefaultAttrsIntrinsic <
    [llvm_any_ty],    // overloaded for types f32/i32, v2f32/v2i32, v4f32/v4i32
    [llvm_v4i32_ty,   // rsrc(SGPR)
     llvm_i32_ty,     // vindex(VGPR)
     llvm_i32_ty,     // offset(VGPR/imm, included in bounds checking and swizzling)
     llvm_i32_ty,     // soffset(SGPR/imm, excluded from bounds checking and swizzling)
     llvm_i32_ty,     // format(imm; bits 3..0 = dfmt, bits 6..4 = nfmt)
     llvm_i32_ty],    // auxiliary data (imm, cachepolicy     (bit 0 = glc,
                      //                                       bit 1 = slc,
                      //                                       bit 2 = dlc on gfx10+),
                      //                      swizzled buffer (bit 3 = swz))
    [IntrReadMem,
     ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<0>;

def int_rvgpu_struct_ptr_tbuffer_load : DefaultAttrsIntrinsic <
    [llvm_any_ty],    // overloaded for types f32/i32, v2f32/v2i32, v4f32/v4i32
    [RVGPUBufferRsrcTy, // rsrc(SGPR)
     llvm_i32_ty,     // vindex(VGPR)
     llvm_i32_ty,     // offset(VGPR/imm, included in bounds checking and swizzling)
     llvm_i32_ty,     // soffset(SGPR/imm, excluded from bounds checking and swizzling)
     llvm_i32_ty,     // format(imm; bits 3..0 = dfmt, bits 6..4 = nfmt)
     llvm_i32_ty],    // auxiliary data (imm, cachepolicy     (bit 0 = glc,
                      //                                       bit 1 = slc,
                      //                                       bit 2 = dlc on gfx10+),
                      //                      swizzled buffer (bit 3 = swz))
    [IntrArgMemOnly, IntrReadMem, ReadOnly<ArgIndex<0>>, NoCapture<ArgIndex<0>>,
     ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<0>;

def int_rvgpu_struct_ptr_tbuffer_store : DefaultAttrsIntrinsic <
    [],
    [llvm_any_ty,    // vdata(VGPR), overloaded for types f32/i32, v2f32/v2i32, v4f32/v4i32
     RVGPUBufferRsrcTy, // rsrc(SGPR)
     llvm_i32_ty,    // vindex(VGPR)
     llvm_i32_ty,    // offset(VGPR/imm, included in bounds checking and swizzling)
     llvm_i32_ty,    // soffset(SGPR/imm, excluded from bounds checking and swizzling)
     llvm_i32_ty,    // format(imm; bits 3..0 = dfmt, bits 6..4 = nfmt)
     llvm_i32_ty],   // auxiliary data (imm, cachepolicy     (bit 0 = glc,
                     //                                       bit 1 = slc,
                     //                                       bit 2 = dlc on gfx10+),
                     //                      swizzled buffer (bit 3 = swz))
    [IntrArgMemOnly, IntrWriteMem, WriteOnly<ArgIndex<1>>, NoCapture<ArgIndex<1>>,
     ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<1>;

def int_rvgpu_struct_tbuffer_store : DefaultAttrsIntrinsic <
    [],
    [llvm_any_ty,    // vdata(VGPR), overloaded for types f32/i32, v2f32/v2i32, v4f32/v4i32
     llvm_v4i32_ty,  // rsrc(SGPR)
     llvm_i32_ty,    // vindex(VGPR)
     llvm_i32_ty,    // offset(VGPR/imm, included in bounds checking and swizzling)
     llvm_i32_ty,    // soffset(SGPR/imm, excluded from bounds checking and swizzling)
     llvm_i32_ty,    // format(imm; bits 3..0 = dfmt, bits 6..4 = nfmt)
     llvm_i32_ty],   // auxiliary data (imm, cachepolicy     (bit 0 = glc,
                     //                                       bit 1 = slc,
                     //                                       bit 2 = dlc on gfx10+),
                     //                      swizzled buffer (bit 3 = swz))
    [IntrWriteMem,
     ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<1>;

class RVGPUBufferAtomic : Intrinsic <
  [llvm_anyint_ty],
  [LLVMMatchType<0>,       // vdata(VGPR)
   llvm_v4i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(SGPR/VGPR/imm)
   llvm_i1_ty],       // slc(imm)
  [ImmArg<ArgIndex<4>>, IntrWillReturn, IntrNoCallback, IntrNoFree], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<1, 0>;
def int_rvgpu_buffer_atomic_swap : RVGPUBufferAtomic;
def int_rvgpu_buffer_atomic_add : RVGPUBufferAtomic;
def int_rvgpu_buffer_atomic_sub : RVGPUBufferAtomic;
def int_rvgpu_buffer_atomic_smin : RVGPUBufferAtomic;
def int_rvgpu_buffer_atomic_umin : RVGPUBufferAtomic;
def int_rvgpu_buffer_atomic_smax : RVGPUBufferAtomic;
def int_rvgpu_buffer_atomic_umax : RVGPUBufferAtomic;
def int_rvgpu_buffer_atomic_and : RVGPUBufferAtomic;
def int_rvgpu_buffer_atomic_or : RVGPUBufferAtomic;
def int_rvgpu_buffer_atomic_xor : RVGPUBufferAtomic;
def int_rvgpu_buffer_atomic_cmpswap : Intrinsic<
  [llvm_i32_ty],
  [llvm_i32_ty,       // src(VGPR)
   llvm_i32_ty,       // cmp(VGPR)
   llvm_v4i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(SGPR/VGPR/imm)
   llvm_i1_ty],       // slc(imm)
  [ImmArg<ArgIndex<5>>, IntrWillReturn, IntrNoCallback, IntrNoFree], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<2, 0>;

def int_rvgpu_buffer_atomic_csub : RVGPUBufferAtomic;

class RVGPUBufferAtomicFP : Intrinsic <
  [llvm_anyfloat_ty],
  [LLVMMatchType<0>, // vdata(VGPR)
   llvm_v4i32_ty,    // rsrc(SGPR)
   llvm_i32_ty,      // vindex(VGPR)
   llvm_i32_ty,      // offset(SGPR/VGPR/imm)
   llvm_i1_ty],      // slc(imm)
  [ImmArg<ArgIndex<4>>, IntrWillReturn, IntrNoCallback, IntrNoFree], "", [SDNPMemOperand]>,
  RVGPURsrcIntrinsic<1, 0>;

// Legacy form of the intrinsic. raw and struct forms should be preferred.
def int_rvgpu_buffer_atomic_fadd : RVGPUBufferAtomicFP;

class RVGPURawBufferLoadLDS : Intrinsic <
  [],
  [llvm_v4i32_ty,                      // rsrc(SGPR)
   LLVMQualPointerType<3>,             // LDS base offset
   llvm_i32_ty,                        // Data byte size: 1/2/4
   llvm_i32_ty,                        // voffset(VGPR, included in bounds checking and swizzling)
   llvm_i32_ty,                        // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty,                        // imm offset(imm, included in bounds checking and swizzling)
   llvm_i32_ty],                       // auxiliary data (imm, cachepolicy     (bit 0 = glc,
                                       //                                       bit 1 = slc,
                                       //                                       bit 2 = dlc on gfx10+))
                                       //                      swizzled buffer (bit 3 = swz))
  [IntrWillReturn, NoCapture<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<5>>,
   ImmArg<ArgIndex<6>>, IntrNoCallback, IntrNoFree], "", [SDNPMemOperand]>, RVGPURsrcIntrinsic<0>;
def int_rvgpu_raw_buffer_load_lds : RVGPURawBufferLoadLDS;

class RVGPURawPtrBufferLoadLDS : Intrinsic <
  [],
  [RVGPUBufferRsrcTy,                 // rsrc(SGPR)
   LLVMQualPointerType<3>,             // LDS base offset
   llvm_i32_ty,                        // Data byte size: 1/2/4
   llvm_i32_ty,                        // voffset(VGPR, included in bounds checking and swizzling)
   llvm_i32_ty,                        // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty,                        // imm offset(imm, included in bounds checking and swizzling)
   llvm_i32_ty],                       // auxiliary data (imm, cachepolicy     (bit 0 = glc,
                                       //                                       bit 1 = slc,
                                       //                                       bit 2 = dlc on gfx10+))
                                       //                      swizzled buffer (bit 3 = swz))
  [IntrWillReturn, IntrArgMemOnly,
   ReadOnly<ArgIndex<0>>, NoCapture<ArgIndex<0>>,
   WriteOnly<ArgIndex<1>>, NoCapture<ArgIndex<1>>,
   ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<5>>,
   ImmArg<ArgIndex<6>>, IntrNoCallback, IntrNoFree], "", [SDNPMemOperand]>, RVGPURsrcIntrinsic<0>;
def int_rvgpu_raw_ptr_buffer_load_lds : RVGPURawPtrBufferLoadLDS;

class RVGPUStructBufferLoadLDS : Intrinsic <
  [],
  [llvm_v4i32_ty,                      // rsrc(SGPR)
   LLVMQualPointerType<3>,             // LDS base offset
   llvm_i32_ty,                        // Data byte size: 1/2/4
   llvm_i32_ty,                        // vindex(VGPR)
   llvm_i32_ty,                        // voffset(VGPR, included in bounds checking and swizzling)
   llvm_i32_ty,                        // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty,                        // imm offset(imm, included in bounds checking and swizzling)
   llvm_i32_ty],                       // auxiliary data (imm, cachepolicy     (bit 0 = glc,
                                       //                                       bit 1 = slc,
                                       //                                       bit 2 = dlc on gfx10+))
                                       //                      swizzled buffer (bit 3 = swz))
  [IntrWillReturn, NoCapture<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<6>>,
   ImmArg<ArgIndex<7>>, IntrNoCallback, IntrNoFree], "", [SDNPMemOperand]>, RVGPURsrcIntrinsic<0>;
def int_rvgpu_struct_buffer_load_lds : RVGPUStructBufferLoadLDS;

class RVGPUStructPtrBufferLoadLDS : Intrinsic <
  [],
  [RVGPUBufferRsrcTy,                 // rsrc(SGPR)
   LLVMQualPointerType<3> ,            // LDS base offset
   llvm_i32_ty,                        // Data byte size: 1/2/4
   llvm_i32_ty,                        // vindex(VGPR)
   llvm_i32_ty,                        // voffset(VGPR, included in bounds checking and swizzling)
   llvm_i32_ty,                        // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty,                        // imm offset(imm, included in bounds checking and swizzling)
   llvm_i32_ty],                       // auxiliary data (imm, cachepolicy     (bit 0 = glc,
                                       //                                       bit 1 = slc,
                                       //                                       bit 2 = dlc on gfx10+))
                                       //                      swizzled buffer (bit 3 = swz))
  [IntrWillReturn, IntrArgMemOnly,
   ReadOnly<ArgIndex<0>>, NoCapture<ArgIndex<0>>,
   WriteOnly<ArgIndex<1>>, NoCapture<ArgIndex<1>>,
   ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<6>>,
   ImmArg<ArgIndex<7>>, IntrNoCallback, IntrNoFree], "", [SDNPMemOperand]>, RVGPURsrcIntrinsic<0>;
def int_rvgpu_struct_ptr_buffer_load_lds : RVGPUStructPtrBufferLoadLDS;

} // defset RVGPUBufferIntrinsics

// Uses that do not set the done bit should set IntrWriteMem on the
// call site.
def int_rvgpu_exp : DefaultAttrsIntrinsic <[], [
  llvm_i32_ty,       // tgt,
  llvm_i32_ty,       // en
  llvm_any_ty,       // src0 (f32 or i32)
  LLVMMatchType<0>,  // src1
  LLVMMatchType<0>,  // src2
  LLVMMatchType<0>,  // src3
  llvm_i1_ty,        // done
  llvm_i1_ty         // vm (ignored on GFX11+)
  ],
  [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<6>>,
   ImmArg<ArgIndex<7>>, IntrWriteMem, IntrInaccessibleMemOnly]
>;

// exp with row_en bit set. Only supported on GFX11+.
def int_rvgpu_exp_row : DefaultAttrsIntrinsic <[], [
  llvm_i32_ty,       // tgt,
  llvm_i32_ty,       // en
  llvm_any_ty,       // src0 (f32 or i32)
  LLVMMatchType<0>,  // src1
  LLVMMatchType<0>,  // src2
  LLVMMatchType<0>,  // src3
  llvm_i1_ty,        // done
  llvm_i32_ty],      // row number
  [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<6>>,
   IntrWriteMem, IntrInaccessibleMemOnly]
>;

// exp with compr bit set. Not supported on GFX11+.
def int_rvgpu_exp_compr : DefaultAttrsIntrinsic <[], [
  llvm_i32_ty,       // tgt,
  llvm_i32_ty,       // en
  llvm_anyvector_ty, // src0 (v2f16 or v2i16)
  LLVMMatchType<0>,  // src1
  llvm_i1_ty,        // done
  llvm_i1_ty],       // vm
  [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<4>>,
   ImmArg<ArgIndex<5>>, IntrWriteMem, IntrInaccessibleMemOnly]
>;

def int_rvgpu_buffer_wbinvl1_sc :
  ClangBuiltin<"__builtin_rvgpu_buffer_wbinvl1_sc">,
  DefaultAttrsIntrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;

def int_rvgpu_buffer_wbinvl1 :
  ClangBuiltin<"__builtin_rvgpu_buffer_wbinvl1">,
  DefaultAttrsIntrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;

def int_rvgpu_s_dcache_inv :
  ClangBuiltin<"__builtin_rvgpu_s_dcache_inv">,
  DefaultAttrsIntrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;

def int_rvgpu_s_memtime :
  ClangBuiltin<"__builtin_rvgpu_s_memtime">,
  DefaultAttrsIntrinsic<[llvm_i64_ty], [], [IntrNoMem, IntrHasSideEffects]>;

def int_rvgpu_s_sleep :
  ClangBuiltin<"__builtin_rvgpu_s_sleep">,
  DefaultAttrsIntrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>, IntrNoMem,
                                IntrHasSideEffects]> {
}

def int_rvgpu_s_sleep_var
    : ClangBuiltin<"__builtin_rvgpu_s_sleep_var">,
      Intrinsic<[], [llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects, IntrWillReturn]> {
}

def int_rvgpu_s_nop :
  DefaultAttrsIntrinsic<[], [llvm_i16_ty], [ImmArg<ArgIndex<0>>, IntrNoMem,
                                IntrHasSideEffects]> {
}

def int_rvgpu_s_incperflevel :
  ClangBuiltin<"__builtin_rvgpu_s_incperflevel">,
  DefaultAttrsIntrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>, IntrNoMem,
                                IntrHasSideEffects]> {
}

def int_rvgpu_s_decperflevel :
  ClangBuiltin<"__builtin_rvgpu_s_decperflevel">,
  DefaultAttrsIntrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>, IntrNoMem,
                                IntrHasSideEffects]> {
}

def int_rvgpu_s_sethalt :
  DefaultAttrsIntrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>, IntrNoMem,
                                IntrHasSideEffects]>;

def int_rvgpu_s_setprio :
  ClangBuiltin<"__builtin_rvgpu_s_setprio">,
  DefaultAttrsIntrinsic<[], [llvm_i16_ty], [ImmArg<ArgIndex<0>>, IntrNoMem,
                                IntrHasSideEffects]>;

def int_rvgpu_s_ttracedata :
  DefaultAttrsIntrinsic<[], [llvm_i32_ty],
                        [IntrNoMem, IntrHasSideEffects]>;
def int_rvgpu_s_ttracedata_imm :
  DefaultAttrsIntrinsic<[], [llvm_i16_ty],
                        [IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

// This is IntrHasSideEffects so it can be used to read cycle counters.
def int_rvgpu_s_getreg :
  ClangBuiltin<"__builtin_rvgpu_s_getreg">,
  DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
  [IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>]
>;

// Note this can be used to set FP environment properties that are
// unsafe to change in non-strictfp functions. The register properties
// available (and value required to access them) may differ per
// subtarget. llvm.rvgpu.s.setreg(hwmode, value)
def int_rvgpu_s_setreg :
  ClangBuiltin<"__builtin_rvgpu_s_setreg">,
  DefaultAttrsIntrinsic<[], [llvm_i32_ty, llvm_i32_ty],
  [IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>]
>;

// int_rvgpu_s_getpc is provided to allow a specific style of position
// independent code to determine the high part of its address when it is
// known (through convention) that the code and any data of interest does
// not cross a 4Gb address boundary. Use for any other purpose may not
// produce the desired results as optimizations may cause code movement,
// especially as we explicitly use IntrNoMem to allow optimizations.
def int_rvgpu_s_getpc :
  ClangBuiltin<"__builtin_rvgpu_s_getpc">,
  DefaultAttrsIntrinsic<[llvm_i64_ty], [], [IntrNoMem, IntrSpeculatable,
                                IntrWillReturn]>;

// __builtin_rvgpu_interp_mov <param>, <attr_chan>, <attr>, <m0>
// param values: 0 = P10, 1 = P20, 2 = P0
def int_rvgpu_interp_mov :
  ClangBuiltin<"__builtin_rvgpu_interp_mov">,
  DefaultAttrsIntrinsic<[llvm_float_ty],
            [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem, IntrSpeculatable,
              ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

// __builtin_rvgpu_interp_p1 <i>, <attr_chan>, <attr>, <m0>
// This intrinsic reads from lds, but the memory values are constant,
// so it behaves like IntrNoMem.
def int_rvgpu_interp_p1 :
  ClangBuiltin<"__builtin_rvgpu_interp_p1">,
  DefaultAttrsIntrinsic<[llvm_float_ty],
            [llvm_float_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem, IntrSpeculatable,
             ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

// __builtin_rvgpu_interp_p2 <p1>, <j>, <attr_chan>, <attr>, <m0>
def int_rvgpu_interp_p2 :
  ClangBuiltin<"__builtin_rvgpu_interp_p2">,
  DefaultAttrsIntrinsic<[llvm_float_ty],
            [llvm_float_ty, llvm_float_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem, IntrSpeculatable,
             ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
          // See int_rvgpu_v_interp_p1 for why this is IntrNoMem.

// __builtin_rvgpu_interp_p1_f16 <i>, <attr_chan>, <attr>, <high>, <m0>
// high selects whether high or low 16-bits are loaded from LDS
def int_rvgpu_interp_p1_f16 :
  ClangBuiltin<"__builtin_rvgpu_interp_p1_f16">,
  DefaultAttrsIntrinsic<[llvm_float_ty],
            [llvm_float_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty, llvm_i32_ty],
            [IntrNoMem, IntrSpeculatable,
             ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

// __builtin_rvgpu_interp_p2_f16 <p1>, <j>, <attr_chan>, <attr>, <high>, <m0>
// high selects whether high or low 16-bits are loaded from LDS
def int_rvgpu_interp_p2_f16 :
  ClangBuiltin<"__builtin_rvgpu_interp_p2_f16">,
  DefaultAttrsIntrinsic<[llvm_half_ty],
            [llvm_float_ty, llvm_float_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty, llvm_i32_ty],
            [IntrNoMem, IntrSpeculatable,
             ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

// llvm.rvgpu.lds.direct.load <m0>
// The input argument is m0, which contains a packed combination of address
// offset and flags describing the data type.
def int_rvgpu_lds_direct_load :
  DefaultAttrsIntrinsic<[llvm_any_ty], // overloaded for types u8, u16, i32/f32, i8, i16
            [llvm_i32_ty],
            [IntrReadMem, IntrSpeculatable]>;

// llvm.rvgpu.lds.param.load <attr_chan>, <attr>, <m0>
// Like interp intrinsics, this reads from lds, but the memory values are constant,
// so it behaves like IntrNoMem.
def int_rvgpu_lds_param_load :
  DefaultAttrsIntrinsic<[llvm_float_ty],
            [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem, IntrSpeculatable,
             ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

// llvm.rvgpu.interp.inreg.p10 <p>, <i>, <p0>
def int_rvgpu_interp_inreg_p10 :
  DefaultAttrsIntrinsic<[llvm_float_ty],
            [llvm_float_ty, llvm_float_ty, llvm_float_ty],
            [IntrNoMem, IntrSpeculatable]>;

// llvm.rvgpu.interp.inreg.p2 <p>, <j>, <tmp>
def int_rvgpu_interp_inreg_p2 :
  DefaultAttrsIntrinsic<[llvm_float_ty],
            [llvm_float_ty, llvm_float_ty, llvm_float_ty],
            [IntrNoMem, IntrSpeculatable]>;

// llvm.rvgpu.interp.inreg.p10.f16 <p>, <i>, <p0>, <high>
// high selects whether high or low 16-bits are used for p and p0 operands
def int_rvgpu_interp_inreg_p10_f16:
  DefaultAttrsIntrinsic<[llvm_float_ty],
            [llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_i1_ty],
            [IntrNoMem, IntrSpeculatable,
             ImmArg<ArgIndex<3>>]>;

// llvm.rvgpu.interp.inreg.p2.f16 <p>, <j>, <tmp>, <high>
// high selects whether high or low 16-bits are used for p operand
def int_rvgpu_interp_inreg_p2_f16 :
  DefaultAttrsIntrinsic<[llvm_half_ty],
            [llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_i1_ty],
            [IntrNoMem, IntrSpeculatable,
             ImmArg<ArgIndex<3>>]>;

// Deprecated: use llvm.rvgpu.live.mask instead.
def int_rvgpu_ps_live : DefaultAttrsIntrinsic <
  [llvm_i1_ty],
  [],
  [IntrNoMem]>;

// Query currently live lanes.
// Returns true if lane is live (and not a helper lane).
def int_rvgpu_live_mask : DefaultAttrsIntrinsic <[llvm_i1_ty],
  [], [IntrReadMem, IntrInaccessibleMemOnly]
>;

def int_rvgpu_mbcnt_lo :
  ClangBuiltin<"__builtin_rvgpu_mbcnt_lo">,
  DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
   [IntrNoMem]>;

def int_rvgpu_mbcnt_hi :
  ClangBuiltin<"__builtin_rvgpu_mbcnt_hi">,
  DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem]>;

// llvm.rvgpu.ds.swizzle src offset
def int_rvgpu_ds_swizzle :
  ClangBuiltin<"__builtin_rvgpu_ds_swizzle">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree,
             ImmArg<ArgIndex<1>>]>;

def int_rvgpu_ubfe : DefaultAttrsIntrinsic<[llvm_anyint_ty],
    [LLVMMatchType<0>, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_sbfe : DefaultAttrsIntrinsic<[llvm_anyint_ty],
    [LLVMMatchType<0>, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_lerp :
  ClangBuiltin<"__builtin_rvgpu_lerp">,
  DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
  [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_sad_u8 :
  ClangBuiltin<"__builtin_rvgpu_sad_u8">,
  DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
  [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_msad_u8 :
  ClangBuiltin<"__builtin_rvgpu_msad_u8">,
  DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
  [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_sad_hi_u8 :
  ClangBuiltin<"__builtin_rvgpu_sad_hi_u8">,
  DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
  [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_sad_u16 :
  ClangBuiltin<"__builtin_rvgpu_sad_u16">,
  DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
  [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_qsad_pk_u16_u8 :
  ClangBuiltin<"__builtin_rvgpu_qsad_pk_u16_u8">,
  DefaultAttrsIntrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i32_ty, llvm_i64_ty],
  [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_mqsad_pk_u16_u8 :
  ClangBuiltin<"__builtin_rvgpu_mqsad_pk_u16_u8">,
  DefaultAttrsIntrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i32_ty, llvm_i64_ty],
  [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_mqsad_u32_u8 :
  ClangBuiltin<"__builtin_rvgpu_mqsad_u32_u8">,
  DefaultAttrsIntrinsic<[llvm_v4i32_ty], [llvm_i64_ty, llvm_i32_ty, llvm_v4i32_ty],
  [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_cvt_pk_u8_f32 :
  ClangBuiltin<"__builtin_rvgpu_cvt_pk_u8_f32">,
  DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_float_ty, llvm_i32_ty, llvm_i32_ty],
  [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_icmp :
  Intrinsic<[llvm_anyint_ty], [llvm_anyint_ty, LLVMMatchType<1>, llvm_i32_ty],
            [IntrNoMem, IntrConvergent,
             ImmArg<ArgIndex<2>>, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

def int_rvgpu_fcmp :
  Intrinsic<[llvm_anyint_ty], [llvm_anyfloat_ty, LLVMMatchType<1>, llvm_i32_ty],
            [IntrNoMem, IntrConvergent,
             ImmArg<ArgIndex<2>>, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

def int_rvgpu_ballot :
  Intrinsic<[llvm_anyint_ty], [llvm_i1_ty],
            [IntrNoMem, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

def int_rvgpu_inverse_ballot :
  Intrinsic<[llvm_i1_ty], [llvm_anyint_ty],
            [IntrNoMem, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

// Lowers to S_BITREPLICATE_B64_B32.
// The argument must be uniform; otherwise, the result is undefined.
def int_rvgpu_s_bitreplicate :
  DefaultAttrsIntrinsic<[llvm_i64_ty], [llvm_i32_ty], [IntrNoMem, IntrConvergent]>;

// Lowers to S_QUADMASK_B{32,64}
// The argument must be uniform; otherwise, the result is undefined.
def int_rvgpu_s_quadmask :
  DefaultAttrsIntrinsic<[llvm_anyint_ty], [llvm_anyint_ty], [IntrNoMem, IntrConvergent]>;

// Lowers to S_WQM_B{32,64}
// The argument must be uniform; otherwise, the result is undefined.
// Does not set WQM; merely calculates the bitmask.
def int_rvgpu_s_wqm :
  DefaultAttrsIntrinsic<[llvm_anyint_ty], [llvm_anyint_ty], [IntrNoMem, IntrConvergent]>;

class RVGPUWaveReduce<LLVMType data_ty = llvm_anyint_ty> : Intrinsic<
    [data_ty],
    [
      LLVMMatchType<0>,   // llvm value to reduce (SGPR/VGPR)
      llvm_i32_ty         // Reduction Strategy Switch for lowering ( 0: Default,
                          //                                          1: Iterative strategy, and
                          //                                          2. DPP)
    ],
    [IntrNoMem, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree, ImmArg<ArgIndex<1>>]>;

def int_rvgpu_wave_reduce_umin : RVGPUWaveReduce;
def int_rvgpu_wave_reduce_umax : RVGPUWaveReduce;

def int_rvgpu_readfirstlane :
  ClangBuiltin<"__builtin_rvgpu_readfirstlane">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty],
            [IntrNoMem, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

// The lane argument must be uniform across the currently active threads of the
// current wave. Otherwise, the result is undefined.
def int_rvgpu_readlane :
  ClangBuiltin<"__builtin_rvgpu_readlane">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

// The value to write and lane select arguments must be uniform across the
// currently active threads of the current wave. Otherwise, the result is
// undefined.
def int_rvgpu_writelane :
  ClangBuiltin<"__builtin_rvgpu_writelane">,
  Intrinsic<[llvm_i32_ty], [
    llvm_i32_ty,    // uniform value to write: returned by the selected lane
    llvm_i32_ty,    // uniform lane select
    llvm_i32_ty     // returned by all lanes other than the selected one
  ],
  [IntrNoMem, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

def int_rvgpu_alignbyte : ClangBuiltin<"__builtin_rvgpu_alignbyte">,
  DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
  [IntrNoMem, IntrSpeculatable]
>;

// mul24 intrinsics can return i32 or i64.
// When returning i64, they're lowered to a mul24/mulhi24 pair.
def int_rvgpu_mul_i24 : DefaultAttrsIntrinsic<[llvm_anyint_ty],
  [llvm_i32_ty, llvm_i32_ty],
  [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_mul_u24 : DefaultAttrsIntrinsic<[llvm_anyint_ty],
  [llvm_i32_ty, llvm_i32_ty],
  [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_mulhi_i24 : DefaultAttrsIntrinsic<[llvm_i32_ty],
  [llvm_i32_ty, llvm_i32_ty],
  [IntrNoMem, IntrSpeculatable]
>;

def int_rvgpu_mulhi_u24 : DefaultAttrsIntrinsic<[llvm_i32_ty],
  [llvm_i32_ty, llvm_i32_ty],
  [IntrNoMem, IntrSpeculatable]
>;

// llvm.rvgpu.ds.gws.init(i32 bar_val, i32 resource_id)
//
// bar_val is the total number of waves that will wait on this
// barrier, minus 1.
def int_rvgpu_ds_gws_init :
  ClangBuiltin<"__builtin_rvgpu_ds_gws_init">,
  Intrinsic<[],
  [llvm_i32_ty, llvm_i32_ty],
  [IntrConvergent, IntrWriteMem,
   IntrInaccessibleMemOnly, IntrWillReturn, IntrNoCallback, IntrNoFree], "",
  [SDNPMemOperand]
>;

// llvm.rvgpu.ds.gws.barrier(i32 vsrc0, i32 resource_id)
// bar_val is the total number of waves that will wait on this
// barrier, minus 1.
def int_rvgpu_ds_gws_barrier :
  ClangBuiltin<"__builtin_rvgpu_ds_gws_barrier">,
  Intrinsic<[],
  [llvm_i32_ty, llvm_i32_ty],
  [IntrConvergent, IntrInaccessibleMemOnly, IntrWillReturn, IntrNoCallback, IntrNoFree], "",
  [SDNPMemOperand]
>;

// llvm.rvgpu.ds.gws.sema.v(i32 resource_id)
def int_rvgpu_ds_gws_sema_v :
  ClangBuiltin<"__builtin_rvgpu_ds_gws_sema_v">,
  Intrinsic<[],
  [llvm_i32_ty],
  [IntrConvergent, IntrInaccessibleMemOnly, IntrWillReturn, IntrNoCallback, IntrNoFree], "",
  [SDNPMemOperand]
>;

// llvm.rvgpu.ds.gws.sema.br(i32 vsrc, i32 resource_id)
def int_rvgpu_ds_gws_sema_br :
  ClangBuiltin<"__builtin_rvgpu_ds_gws_sema_br">,
  Intrinsic<[],
  [llvm_i32_ty, llvm_i32_ty],
  [IntrConvergent, IntrInaccessibleMemOnly, IntrWillReturn, IntrNoCallback, IntrNoFree], "",
  [SDNPMemOperand]
>;

// llvm.rvgpu.ds.gws.sema.p(i32 resource_id)
def int_rvgpu_ds_gws_sema_p :
  ClangBuiltin<"__builtin_rvgpu_ds_gws_sema_p">,
  Intrinsic<[],
  [llvm_i32_ty],
  [IntrConvergent, IntrInaccessibleMemOnly, IntrWillReturn, IntrNoCallback, IntrNoFree], "",
  [SDNPMemOperand]
>;

// llvm.rvgpu.ds.gws.sema.release.all(i32 resource_id)
def int_rvgpu_ds_gws_sema_release_all :
  ClangBuiltin<"__builtin_rvgpu_ds_gws_sema_release_all">,
  Intrinsic<[],
  [llvm_i32_ty],
  [IntrConvergent, IntrInaccessibleMemOnly, IntrWillReturn, IntrNoCallback, IntrNoFree], "",
  [SDNPMemOperand]
>;


// Copies the source value to the destination value, with the guarantee that
// the source value is computed as if the entire program were executed in WQM.
def int_rvgpu_wqm : Intrinsic<[llvm_any_ty],
  [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

// Copies the source value to the destination value, such that the source
// is computed as if the entire program were executed in WQM if any other
// program code executes in WQM.
def int_rvgpu_softwqm : Intrinsic<[llvm_any_ty],
  [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

// Return true if at least one thread within the pixel quad passes true into
// the function.
def int_rvgpu_wqm_vote : Intrinsic<[llvm_i1_ty],
  [llvm_i1_ty], [IntrNoMem, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

// If false, set EXEC=0 for the current thread until the end of program.
// FIXME: Should this be IntrNoMem, IntrHasSideEffects, or IntrWillReturn?
def int_rvgpu_kill : Intrinsic<[], [llvm_i1_ty], [IntrNoCallback, IntrNoFree]>;

def int_rvgpu_endpgm : ClangBuiltin<"__builtin_rvgpu_endpgm">,
  Intrinsic<[], [], [IntrNoReturn, IntrCold, IntrNoMem, IntrHasSideEffects, IntrConvergent,
                     IntrNoCallback, IntrNoFree]
>;

// If false, mark all active lanes as helper lanes until the end of program.
def int_rvgpu_wqm_demote : Intrinsic<[],
  [llvm_i1_ty], [IntrWriteMem, IntrInaccessibleMemOnly, IntrNoCallback, IntrNoFree]
>;

// Copies the active channels of the source value to the destination value,
// with the guarantee that the source value is computed as if the entire
// program were executed in Whole Wavefront Mode, i.e. with all channels
// enabled, with a few exceptions: - Phi nodes which require WWM return an
// undefined value.
def int_rvgpu_strict_wwm : Intrinsic<[llvm_any_ty],
  [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable,
                       IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;
// Deprecated. Use int_rvgpu_strict_wwm instead.
def int_rvgpu_wwm : Intrinsic<[llvm_any_ty],
  [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable,
                       IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;
def int_rvgpu_strict_wqm : Intrinsic<[llvm_any_ty],
  [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable,
                       IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

// Given a value, copies it while setting all the inactive lanes to a given
// value. Note that OpenGL helper lanes are considered active, so if the
// program ever uses WQM, then the instruction and the first source will be
// computed in WQM.
def int_rvgpu_set_inactive :
  Intrinsic<[llvm_anyint_ty],
            [LLVMMatchType<0>, // value to be copied
             LLVMMatchType<0>], // value for the inactive lanes to take
            [IntrNoMem, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

// Similar to int_rvgpu_set_inactive, but the value for the inactive lanes must
// be a VGPR function argument.
// Can only be used in functions with the `rvgpu_cs_chain` or
// `rvgpu_cs_chain_preserve` calling conventions, and only in uniform control
// flow.
def int_rvgpu_set_inactive_chain_arg :
  Intrinsic<[llvm_anyint_ty],
            [LLVMMatchType<0>, // value to be copied
             LLVMMatchType<0>], // value for the inactive lanes to take
            [IntrNoMem, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

// Return if the given flat pointer points to a local memory address.
def int_rvgpu_is_shared : ClangBuiltin<"__builtin_rvgpu_is_shared">,
  DefaultAttrsIntrinsic<[llvm_i1_ty], [llvm_ptr_ty],
  [IntrNoMem, IntrSpeculatable, NoCapture<ArgIndex<0>>]
>;

// Return if the given flat pointer points to a prvate memory address.
def int_rvgpu_is_private : ClangBuiltin<"__builtin_rvgpu_is_private">,
  DefaultAttrsIntrinsic<[llvm_i1_ty], [llvm_ptr_ty],
  [IntrNoMem, IntrSpeculatable, NoCapture<ArgIndex<0>>]
>;

// A uniform tail call to a function with the `rvgpu_cs_chain` or
// `rvgpu_cs_chain_preserve` calling convention. It will populate the SGPRs
// starting at s0 and the VGPRs starting at v8, set EXEC and perform a jump to
// the given function.
// Can only be used in functions with the `rvgpu_cs`, `rvgpu_cs_chain` or
// `rvgpu_cs_chain_preserve` calling conventions, and only in uniform control
// flow.
def int_rvgpu_cs_chain:
  Intrinsic<[],
            [llvm_anyptr_ty, // The function to jump to.
             llvm_anyint_ty, // Value to put in EXEC (should be i32 or i64).
             llvm_any_ty, // Arguments that will be copied into SGPRs (s0+).
                          // Must be uniform.
             llvm_any_ty, // Arguments that will be copied into VGPRs (v8+).
                          // Need not be uniform.
             llvm_i32_ty, // Flags.
             llvm_vararg_ty // Additional arguments. Only present if Flags is
                            // non-zero.
            ],
            [IntrConvergent, IntrNoReturn, ImmArg<ArgIndex<4>>]>;


//===----------------------------------------------------------------------===//
// CI+ Intrinsics
//===----------------------------------------------------------------------===//

def int_rvgpu_s_dcache_inv_vol :
  ClangBuiltin<"__builtin_rvgpu_s_dcache_inv_vol">,
  DefaultAttrsIntrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;

def int_rvgpu_buffer_wbinvl1_vol :
  ClangBuiltin<"__builtin_rvgpu_buffer_wbinvl1_vol">,
  DefaultAttrsIntrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;

//===----------------------------------------------------------------------===//
// VI Intrinsics
//===----------------------------------------------------------------------===//

// llvm.rvgpu.mov.dpp.i32 <src> <dpp_ctrl> <row_mask> <bank_mask> <bound_ctrl>
def int_rvgpu_mov_dpp :
  Intrinsic<[llvm_anyint_ty],
            [LLVMMatchType<0>, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
             llvm_i1_ty],
             [IntrNoMem, IntrConvergent, IntrWillReturn,
             ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>,
             ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, IntrNoCallback, IntrNoFree]>;

// llvm.rvgpu.update.dpp.i32 <old> <src> <dpp_ctrl> <row_mask> <bank_mask> <bound_ctrl>
// Should be equivalent to:
// v_mov_b32 <dest> <old>
// v_mov_b32 <dest> <src> <dpp_ctrl> <row_mask> <bank_mask> <bound_ctrl>
def int_rvgpu_update_dpp :
  Intrinsic<[llvm_any_ty],
            [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty,
            llvm_i32_ty, llvm_i32_ty, llvm_i1_ty],
             [IntrNoMem, IntrConvergent, IntrWillReturn,
              ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>,
              ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, IntrNoCallback, IntrNoFree]>;

def int_rvgpu_s_dcache_wb :
  ClangBuiltin<"__builtin_rvgpu_s_dcache_wb">,
  Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

def int_rvgpu_s_dcache_wb_vol :
  ClangBuiltin<"__builtin_rvgpu_s_dcache_wb_vol">,
  Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

def int_rvgpu_s_memrealtime :
  ClangBuiltin<"__builtin_rvgpu_s_memrealtime">,
  Intrinsic<[llvm_i64_ty], [], [IntrNoMem, IntrHasSideEffects, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

// llvm.rvgpu.ds.permute <index> <src>
def int_rvgpu_ds_permute :
  ClangBuiltin<"__builtin_rvgpu_ds_permute">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

// llvm.rvgpu.ds.bpermute <index> <src>
def int_rvgpu_ds_bpermute :
  ClangBuiltin<"__builtin_rvgpu_ds_bpermute">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
     [IntrNoMem, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

// llvm.rvgpu.perm <src0> <src1> <selector>
def int_rvgpu_perm :
  ClangBuiltin<"__builtin_rvgpu_perm">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
     [IntrNoMem, IntrSpeculatable, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

//===----------------------------------------------------------------------===//
// GFX9 Intrinsics
//===----------------------------------------------------------------------===//

class RVGPUGlobalLoadLDS : Intrinsic <
  [],
  [LLVMQualPointerType<1>,             // Base global pointer to load from
   LLVMQualPointerType<3>,             // LDS base pointer to store to
   llvm_i32_ty,                        // Data byte size: 1/2/4
   llvm_i32_ty,                        // imm offset (applied to both global and LDS address)
   llvm_i32_ty],                       // auxiliary data (imm, cachepolicy (bit 0 = glc/sc0,
                                       //                                   bit 1 = slc/sc1,
                                       //                                   bit 2 = dlc on gfx10+))
                                       //                                   bit 4 = scc/nt on gfx90a+))
  [IntrWillReturn, NoCapture<ArgIndex<0>>, NoCapture<ArgIndex<1>>,
   ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, IntrNoCallback, IntrNoFree],
  "", [SDNPMemOperand]>;
def int_rvgpu_global_load_lds : RVGPUGlobalLoadLDS;

//===----------------------------------------------------------------------===//
// GFX10 Intrinsics
//===----------------------------------------------------------------------===//

// llvm.rvgpu.permlane16 <old> <src0> <src1> <src2> <fi> <bound_control>
def int_rvgpu_permlane16 : ClangBuiltin<"__builtin_rvgpu_permlane16">,
  Intrinsic<[llvm_i32_ty],
            [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty, llvm_i1_ty],
            [IntrNoMem, IntrConvergent, IntrWillReturn,
             ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, IntrNoCallback, IntrNoFree]>;

// llvm.rvgpu.permlanex16 <old> <src0> <src1> <src2> <fi> <bound_control>
def int_rvgpu_permlanex16 : ClangBuiltin<"__builtin_rvgpu_permlanex16">,
  Intrinsic<[llvm_i32_ty],
            [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty, llvm_i1_ty],
            [IntrNoMem, IntrConvergent, IntrWillReturn,
             ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, IntrNoCallback, IntrNoFree]>;

// llvm.rvgpu.mov.dpp8.i32 <src> <sel>
// <sel> is a 32-bit constant whose high 8 bits must be zero which selects
// the lanes to read from.
def int_rvgpu_mov_dpp8 :
  Intrinsic<[llvm_anyint_ty],
            [LLVMMatchType<0>, llvm_i32_ty],
            [IntrNoMem, IntrConvergent, IntrWillReturn,
             ImmArg<ArgIndex<1>>, IntrNoCallback, IntrNoFree]>;

def int_rvgpu_s_get_waveid_in_workgroup :
  ClangBuiltin<"__builtin_rvgpu_s_get_waveid_in_workgroup">,
  Intrinsic<[llvm_i32_ty], [],
    [IntrNoMem, IntrHasSideEffects, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

class RVGPUGlobalAtomicRtn<LLVMType vt> : Intrinsic <
  [vt],
  [llvm_anyptr_ty,    // vaddr
   vt],               // vdata(VGPR)
  [IntrArgMemOnly, IntrWillReturn, NoCapture<ArgIndex<0>>, IntrNoCallback, IntrNoFree], "",
  [SDNPMemOperand]>;

def int_rvgpu_global_atomic_csub : RVGPUGlobalAtomicRtn<llvm_i32_ty>;

// uint4 llvm.rvgpu.image.bvh.intersect.ray <node_ptr>, <ray_extent>, <ray_origin>,
//                                           <ray_dir>, <ray_inv_dir>, <texture_descr>
// <node_ptr> is i32 or i64.
// <ray_dir> and <ray_inv_dir> are both v3f16 or both v3f32.
def int_rvgpu_image_bvh_intersect_ray :
  DefaultAttrsIntrinsic<[llvm_v4i32_ty],
            [llvm_anyint_ty, llvm_float_ty, llvm_v3f32_ty, llvm_anyvector_ty,
             LLVMMatchType<1>, llvm_v4i32_ty],
            [IntrReadMem, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

//===----------------------------------------------------------------------===//
// GFX11 Intrinsics
//===----------------------------------------------------------------------===//

// llvm.rvgpu.permlane64 <src0>
def int_rvgpu_permlane64 :
  ClangBuiltin<"__builtin_rvgpu_permlane64">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty],
            [IntrNoMem, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

def int_rvgpu_ds_add_gs_reg_rtn :
  ClangBuiltin<"__builtin_rvgpu_ds_add_gs_reg_rtn">,
  Intrinsic<[llvm_anyint_ty], [llvm_i32_ty, llvm_i32_ty],
            [ImmArg<ArgIndex<1>>, IntrHasSideEffects, IntrWillReturn, IntrNoCallback, IntrNoFree],
            "", [SDNPMemOperand]>;

def int_rvgpu_ds_sub_gs_reg_rtn :
  ClangBuiltin<"__builtin_rvgpu_ds_sub_gs_reg_rtn">,
  Intrinsic<[llvm_anyint_ty], [llvm_i32_ty, llvm_i32_ty],
            [ImmArg<ArgIndex<1>>, IntrHasSideEffects, IntrWillReturn, IntrNoCallback, IntrNoFree],
            "", [SDNPMemOperand]>;

def int_rvgpu_ds_bvh_stack_rtn :
  Intrinsic<
    [llvm_i32_ty, llvm_i32_ty], // %vdst, %addr
    [
      llvm_i32_ty,   // %addr
      llvm_i32_ty,   // %data0
      llvm_v4i32_ty, // %data1
      llvm_i32_ty,   // %offset
    ],
    [ImmArg<ArgIndex<3>>, IntrWillReturn, IntrNoCallback, IntrNoFree]
  >;

// WMMA (Wave Matrix Multiply-Accumulate) intrinsics
//
// These operations perform a matrix multiplication and accumulation of
// the form: D = A * B + C .

class RVGPUWmmaIntrinsic<LLVMType AB, LLVMType CD> :
  Intrinsic<
    [CD],               // %D
    [
      AB,               // %A
      AB,               // %B
      LLVMMatchType<0>, // %C
    ],
    [IntrNoMem, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

class RVGPUWmmaIntrinsicOPSEL<LLVMType AB, LLVMType CD> :
  Intrinsic<
    [CD],               // %D
    [
      AB,               // %A
      AB,               // %B
      LLVMMatchType<0>, // %C
      llvm_i1_ty,       // %high
    ],
    [IntrNoMem, IntrConvergent, ImmArg<ArgIndex<3>>, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

class RVGPUWmmaIntrinsicIU<LLVMType AB, LLVMType CD> :
  Intrinsic<
    [CD],               // %D
    [
      llvm_i1_ty,       // %A_sign
      AB,               // %A
      llvm_i1_ty,       // %B_sign
      AB,               // %B
      LLVMMatchType<0>, // %C
      llvm_i1_ty,       // %clamp
    ],
    [IntrNoMem, IntrConvergent, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<5>>, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

def int_rvgpu_wmma_f32_16x16x16_f16   : RVGPUWmmaIntrinsic<llvm_v16f16_ty, llvm_anyfloat_ty>;
def int_rvgpu_wmma_f32_16x16x16_bf16  : RVGPUWmmaIntrinsic<llvm_v16i16_ty, llvm_anyfloat_ty>;
// The regular, untied f16/bf16 wmma intrinsics only write to one half
// of the registers (set via the op_sel bit).
// The content of the other 16-bit of the registers is undefined.
def int_rvgpu_wmma_f16_16x16x16_f16   : RVGPUWmmaIntrinsicOPSEL<llvm_v16f16_ty, llvm_anyfloat_ty>;
def int_rvgpu_wmma_bf16_16x16x16_bf16 : RVGPUWmmaIntrinsicOPSEL<llvm_v16i16_ty, llvm_anyint_ty>;
// The tied versions of the f16/bf16 wmma intrinsics tie the destination matrix
// registers to the input accumulator registers.
// Essentially, the content of the other 16-bit is preserved from the input.
def int_rvgpu_wmma_f16_16x16x16_f16_tied   : RVGPUWmmaIntrinsicOPSEL<llvm_v16f16_ty, llvm_anyfloat_ty>;
def int_rvgpu_wmma_bf16_16x16x16_bf16_tied : RVGPUWmmaIntrinsicOPSEL<llvm_v16i16_ty, llvm_anyint_ty>;
def int_rvgpu_wmma_i32_16x16x16_iu8   : RVGPUWmmaIntrinsicIU<llvm_v4i32_ty, llvm_anyint_ty>;
def int_rvgpu_wmma_i32_16x16x16_iu4   : RVGPUWmmaIntrinsicIU<llvm_v2i32_ty, llvm_anyint_ty>;

def int_rvgpu_s_wait_event_export_ready :
  ClangBuiltin<"__builtin_rvgpu_s_wait_event_export_ready">,
  Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects, IntrWillReturn]
>;

//===----------------------------------------------------------------------===//
// GFX12 Intrinsics
//===----------------------------------------------------------------------===//

// llvm.rvgpu.permlane16.var <old> <src0> <src1> <fi> <bound_control>
def int_rvgpu_permlane16_var : ClangBuiltin<"__builtin_rvgpu_permlane16_var">,
  Intrinsic<[llvm_i32_ty],
            [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty, llvm_i1_ty],
            [IntrNoMem, IntrConvergent, IntrWillReturn,
             ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, IntrNoCallback, IntrNoFree]>;

// llvm.rvgpu.permlanex16.var <old> <src0> <src1> <fi> <bound_control>
def int_rvgpu_permlanex16_var : ClangBuiltin<"__builtin_rvgpu_permlanex16_var">,
  Intrinsic<[llvm_i32_ty],
            [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty, llvm_i1_ty],
            [IntrNoMem, IntrConvergent, IntrWillReturn,
             ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, IntrNoCallback, IntrNoFree]>;

def int_rvgpu_flat_atomic_fmin_num   : RVGPUGlobalAtomicRtn<llvm_anyfloat_ty>;
def int_rvgpu_flat_atomic_fmax_num   : RVGPUGlobalAtomicRtn<llvm_anyfloat_ty>;
def int_rvgpu_global_atomic_fmin_num : RVGPUGlobalAtomicRtn<llvm_anyfloat_ty>;
def int_rvgpu_global_atomic_fmax_num : RVGPUGlobalAtomicRtn<llvm_anyfloat_ty>;

//===----------------------------------------------------------------------===//
// Deep learning intrinsics.
//===----------------------------------------------------------------------===//

// f32 %r = llvm.rvgpu.fdot2(v2f16 %a, v2f16 %b, f32 %c, i1 %clamp)
//   %r = %a[0] * %b[0] + %a[1] * %b[1] + %c
def int_rvgpu_fdot2 :
  ClangBuiltin<"__builtin_rvgpu_fdot2">,
  DefaultAttrsIntrinsic<
    [llvm_float_ty], // %r
    [
      llvm_v2f16_ty, // %a
      llvm_v2f16_ty, // %b
      llvm_float_ty, // %c
      llvm_i1_ty     // %clamp
    ],
    [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<3>>]
  >;

// f16 %r = llvm.rvgpu.fdot2.f16.f16(v2f16 %a, v2f16 %b, f16 %c)
//   %r = %a[0] * %b[0] + %a[1] * %b[1] + %c
def int_rvgpu_fdot2_f16_f16 :
  ClangBuiltin<"__builtin_rvgpu_fdot2_f16_f16">,
  DefaultAttrsIntrinsic<
    [llvm_half_ty],  // %r
    [
      llvm_v2f16_ty, // %a
      llvm_v2f16_ty, // %b
      llvm_half_ty   // %c
    ],
    [IntrNoMem, IntrSpeculatable]
  >;

// bf16 %r = llvm.rvgpu.fdot2.bf16.bf16(v2bf16 %a, v2bf16 %b, bf16 %c)
//   %r = %a[0] * %b[0] + %a[1] * %b[1] + %c
def int_rvgpu_fdot2_bf16_bf16 :
  ClangBuiltin<"__builtin_rvgpu_fdot2_bf16_bf16">,
  DefaultAttrsIntrinsic<
    [llvm_i16_ty],   // %r
    [
      llvm_v2i16_ty, // %a
      llvm_v2i16_ty, // %b
      llvm_i16_ty    // %c
    ],
    [IntrNoMem, IntrSpeculatable]
  >;

// f32 %r = llvm.rvgpu.fdot2.f32.bf16(v2bf16 %a, v2bf16 %b, f32 %c, i1 %clamp)
//   %r = %a[0] * %b[0] + %a[1] * %b[1] + %c
def int_rvgpu_fdot2_f32_bf16 :
  ClangBuiltin<"__builtin_rvgpu_fdot2_f32_bf16">,
  DefaultAttrsIntrinsic<
    [llvm_float_ty], // %r
    [
      llvm_v2i16_ty, // %a
      llvm_v2i16_ty, // %b
      llvm_float_ty, // %c
      llvm_i1_ty     // %clamp
    ],
    [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<3>>]
  >;

// i32 %r = llvm.rvgpu.sdot2(v2i16 %a, v2i16 %b, i32 %c, i1 %clamp)
//   %r = %a[0] * %b[0] + %a[1] * %b[1] + %c
def int_rvgpu_sdot2 :
  ClangBuiltin<"__builtin_rvgpu_sdot2">,
  DefaultAttrsIntrinsic<
    [llvm_i32_ty], // %r
    [
      llvm_v2i16_ty, // %a
      llvm_v2i16_ty, // %b
      llvm_i32_ty,   // %c
      llvm_i1_ty     // %clamp
    ],
    [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<3>>]
  >;

// u32 %r = llvm.rvgpu.udot2(v2u16 %a, v2u16 %b, u32 %c, i1 %clamp)
//   %r = %a[0] * %b[0] + %a[1] * %b[1] + %c
def int_rvgpu_udot2 :
  ClangBuiltin<"__builtin_rvgpu_udot2">,
  DefaultAttrsIntrinsic<
    [llvm_i32_ty], // %r
    [
      llvm_v2i16_ty, // %a
      llvm_v2i16_ty, // %b
      llvm_i32_ty,   // %c
      llvm_i1_ty     // %clamp
    ],
    [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<3>>]
  >;

// i32 %r = llvm.rvgpu.sdot4(v4i8 (as i32) %a, v4i8 (as i32) %b, i32 %c, i1 %clamp)
//   %r = %a[0] * %b[0] + %a[1] * %b[1] + %a[2] * %b[2] + %a[3] * %b[3] + %c
def int_rvgpu_sdot4 :
  ClangBuiltin<"__builtin_rvgpu_sdot4">,
  DefaultAttrsIntrinsic<
    [llvm_i32_ty], // %r
    [
      llvm_i32_ty, // %a
      llvm_i32_ty, // %b
      llvm_i32_ty, // %c
      llvm_i1_ty   // %clamp
    ],
    [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<3>>]
  >;

// u32 %r = llvm.rvgpu.udot4(v4u8 (as u32) %a, v4u8 (as u32) %b, u32 %c, i1 %clamp)
//   %r = %a[0] * %b[0] + %a[1] * %b[1] + %a[2] * %b[2] + %a[3] * %b[3] + %c
def int_rvgpu_udot4 :
  ClangBuiltin<"__builtin_rvgpu_udot4">,
  DefaultAttrsIntrinsic<
    [llvm_i32_ty], // %r
    [
      llvm_i32_ty, // %a
      llvm_i32_ty, // %b
      llvm_i32_ty, // %c
      llvm_i1_ty   // %clamp
    ],
    [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<3>>]
  >;

// i32 %r = llvm.rvgpu.sudot4(i1 %a_sign, v4i8 (as i32) %a, i1 %b_sign, v4i8 (as i32) %b, i32 %c, i1 %clamp)
// Treat input as signed (_sign = 1) or unsigned (_sign = 0).
// a[i in 0. . . 3] = (%a_sign ? a.i8[i] : promoteToSigned(a.u8[i]));
// b[i in 0. . . 3] = (%b_sign ? b.i8[i] : promoteToSigned(b.u8[i]));
//   %r = %a[0] * %b[0] + %a[1] * %b[1] + %a[2] * %b[2] + %a[3] * %b[3] + %c
def int_rvgpu_sudot4 :
  ClangBuiltin<"__builtin_rvgpu_sudot4">,
  DefaultAttrsIntrinsic<
    [llvm_i32_ty], // %r
    [
      llvm_i1_ty,  // %a_sign
      llvm_i32_ty, // %a
      llvm_i1_ty,  // %b_sign
      llvm_i32_ty, // %b
      llvm_i32_ty, // %c
      llvm_i1_ty   // %clamp
    ],
    [IntrNoMem, IntrSpeculatable,
     ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<5>>]
  >;

// i32 %r = llvm.rvgpu.sdot8(v8i4 (as i32) %a, v8i4 (as i32) %b, i32 %c, i1 %clamp)
//   %r = %a[0] * %b[0] + %a[1] * %b[1] + %a[2] * %b[2] + %a[3] * %b[3] +
//        %a[4] * %b[4] + %a[5] * %b[5] + %a[6] * %b[6] + %a[7] * %b[7] + %c
def int_rvgpu_sdot8 :
  ClangBuiltin<"__builtin_rvgpu_sdot8">,
  DefaultAttrsIntrinsic<
    [llvm_i32_ty], // %r
    [
      llvm_i32_ty, // %a
      llvm_i32_ty, // %b
      llvm_i32_ty, // %c
      llvm_i1_ty   // %clamp
    ],
    [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<3>>]
  >;

// u32 %r = llvm.rvgpu.udot8(v8u4 (as u32) %a, v8u4 (as u32) %b, u32 %c, i1 %clamp)
//   %r = %a[0] * %b[0] + %a[1] * %b[1] + %a[2] * %b[2] + %a[3] * %b[3] +
//        %a[4] * %b[4] + %a[5] * %b[5] + %a[6] * %b[6] + %a[7] * %b[7] + %c
def int_rvgpu_udot8 :
  ClangBuiltin<"__builtin_rvgpu_udot8">,
  DefaultAttrsIntrinsic<
    [llvm_i32_ty], // %r
    [
      llvm_i32_ty, // %a
      llvm_i32_ty, // %b
      llvm_i32_ty, // %c
      llvm_i1_ty   // %clamp
    ],
    [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<3>>]
  >;

// i32 %r = llvm.rvgpu.sudot8(i1 %a_sign, v8i4 (as i32) %a, i1 %b_sign, v8i4 (as i32) %b, i32 %c, i1 %clamp)
// Treat input as signed (_sign = 1) or unsigned (_sign = 0).
// a[i in 0. . . 7] = (%a_sign ? a.i4[i] : promoteToSigned(a.u4[i]));
// b[i in 0. . . 7] = (%b_sign ? b.i4[i] : promoteToSigned(b.u4[i]));
//   %r = %a[0] * %b[0] + %a[1] * %b[1] + %a[2] * %b[2] + %a[3] * %b[3] +
//        %a[4] * %b[4] + %a[5] * %b[5] + %a[6] * %b[6] + %a[7] * %b[7] + %c
  def int_rvgpu_sudot8 :
  ClangBuiltin<"__builtin_rvgpu_sudot8">,
  DefaultAttrsIntrinsic<
    [llvm_i32_ty], // %r
    [
      llvm_i1_ty,  // %a_sign
      llvm_i32_ty, // %a
      llvm_i1_ty,  // %b_sign
      llvm_i32_ty, // %b
      llvm_i32_ty, // %c
      llvm_i1_ty   // %clamp
    ],
    [IntrNoMem, IntrSpeculatable,
     ImmArg<ArgIndex<0>>,  ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<5>>]
  >;

//===----------------------------------------------------------------------===//
// gfx908 intrinsics
// ===----------------------------------------------------------------------===//

def int_rvgpu_global_atomic_fadd : RVGPUGlobalAtomicRtn<llvm_anyfloat_ty>;

// llvm.rvgpu.mfma.*.* vdst, srcA, srcB, srcC, cbsz, abid, blgp
class RVGPUMfmaIntrinsic<LLVMType DestTy, LLVMType SrcABTy> :
  ClangBuiltin<!subst("int", "__builtin", NAME)>,
  DefaultAttrsIntrinsic<[DestTy],
            [SrcABTy, SrcABTy, DestTy,
             llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrConvergent, IntrNoMem,
             ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

def int_rvgpu_mfma_f32_32x32x1f32  : RVGPUMfmaIntrinsic<llvm_v32f32_ty, llvm_float_ty>;
def int_rvgpu_mfma_f32_16x16x1f32  : RVGPUMfmaIntrinsic<llvm_v16f32_ty, llvm_float_ty>;
def int_rvgpu_mfma_f32_4x4x1f32    : RVGPUMfmaIntrinsic<llvm_v4f32_ty,  llvm_float_ty>;
def int_rvgpu_mfma_f32_32x32x2f32  : RVGPUMfmaIntrinsic<llvm_v16f32_ty, llvm_float_ty>;
def int_rvgpu_mfma_f32_16x16x4f32  : RVGPUMfmaIntrinsic<llvm_v4f32_ty,  llvm_float_ty>;
def int_rvgpu_mfma_f32_32x32x4f16  : RVGPUMfmaIntrinsic<llvm_v32f32_ty, llvm_v4f16_ty>;
def int_rvgpu_mfma_f32_16x16x4f16  : RVGPUMfmaIntrinsic<llvm_v16f32_ty, llvm_v4f16_ty>;
def int_rvgpu_mfma_f32_4x4x4f16    : RVGPUMfmaIntrinsic<llvm_v4f32_ty,  llvm_v4f16_ty>;
def int_rvgpu_mfma_f32_32x32x8f16  : RVGPUMfmaIntrinsic<llvm_v16f32_ty, llvm_v4f16_ty>;
def int_rvgpu_mfma_f32_16x16x16f16 : RVGPUMfmaIntrinsic<llvm_v4f32_ty,  llvm_v4f16_ty>;
def int_rvgpu_mfma_i32_32x32x4i8   : RVGPUMfmaIntrinsic<llvm_v32i32_ty, llvm_i32_ty>;
def int_rvgpu_mfma_i32_16x16x4i8   : RVGPUMfmaIntrinsic<llvm_v16i32_ty, llvm_i32_ty>;
def int_rvgpu_mfma_i32_4x4x4i8     : RVGPUMfmaIntrinsic<llvm_v4i32_ty,  llvm_i32_ty>;
def int_rvgpu_mfma_i32_32x32x8i8   : RVGPUMfmaIntrinsic<llvm_v16i32_ty, llvm_i32_ty>;
def int_rvgpu_mfma_i32_16x16x16i8  : RVGPUMfmaIntrinsic<llvm_v4i32_ty,  llvm_i32_ty>;
def int_rvgpu_mfma_f32_32x32x2bf16 : RVGPUMfmaIntrinsic<llvm_v32f32_ty, llvm_v2i16_ty>;
def int_rvgpu_mfma_f32_16x16x2bf16 : RVGPUMfmaIntrinsic<llvm_v16f32_ty, llvm_v2i16_ty>;
def int_rvgpu_mfma_f32_4x4x2bf16   : RVGPUMfmaIntrinsic<llvm_v4f32_ty,  llvm_v2i16_ty>;
def int_rvgpu_mfma_f32_32x32x4bf16 : RVGPUMfmaIntrinsic<llvm_v16f32_ty, llvm_v2i16_ty>;
def int_rvgpu_mfma_f32_16x16x8bf16 : RVGPUMfmaIntrinsic<llvm_v4f32_ty,  llvm_v2i16_ty>;

//===----------------------------------------------------------------------===//
// gfx90a intrinsics
// ===----------------------------------------------------------------------===//

def int_rvgpu_global_atomic_fmin : RVGPUGlobalAtomicRtn<llvm_anyfloat_ty>;
def int_rvgpu_global_atomic_fmax : RVGPUGlobalAtomicRtn<llvm_anyfloat_ty>;
def int_rvgpu_flat_atomic_fadd   : RVGPUGlobalAtomicRtn<llvm_anyfloat_ty>;
def int_rvgpu_flat_atomic_fmin   : RVGPUGlobalAtomicRtn<llvm_anyfloat_ty>;
def int_rvgpu_flat_atomic_fmax   : RVGPUGlobalAtomicRtn<llvm_anyfloat_ty>;

def int_rvgpu_mfma_f32_32x32x4bf16_1k  : RVGPUMfmaIntrinsic<llvm_v32f32_ty, llvm_v4i16_ty>;
def int_rvgpu_mfma_f32_16x16x4bf16_1k  : RVGPUMfmaIntrinsic<llvm_v16f32_ty, llvm_v4i16_ty>;
def int_rvgpu_mfma_f32_4x4x4bf16_1k    : RVGPUMfmaIntrinsic<llvm_v4f32_ty,  llvm_v4i16_ty>;
def int_rvgpu_mfma_f32_32x32x8bf16_1k  : RVGPUMfmaIntrinsic<llvm_v16f32_ty, llvm_v4i16_ty>;
def int_rvgpu_mfma_f32_16x16x16bf16_1k : RVGPUMfmaIntrinsic<llvm_v4f32_ty,  llvm_v4i16_ty>;

// Note: in gfx940 BLGP argument is replaced by NEG bitfield in the DGEMM MFMA.
//       Three bits corresponding to the neg modifier applied to the respective
//       source operand.
def int_rvgpu_mfma_f64_16x16x4f64      : RVGPUMfmaIntrinsic<llvm_v4f64_ty,  llvm_double_ty>;
def int_rvgpu_mfma_f64_4x4x4f64        : RVGPUMfmaIntrinsic<llvm_double_ty, llvm_double_ty>;

//===----------------------------------------------------------------------===//
// gfx940 intrinsics
// ===----------------------------------------------------------------------===//

// bf16 atomics use v2i16 argument since there is no bf16 data type in the llvm.
def int_rvgpu_global_atomic_fadd_v2bf16 : RVGPUGlobalAtomicRtn<llvm_v2i16_ty>;
def int_rvgpu_flat_atomic_fadd_v2bf16   : RVGPUGlobalAtomicRtn<llvm_v2i16_ty>;
def int_rvgpu_ds_fadd_v2bf16 : DefaultAttrsIntrinsic<
    [llvm_v2i16_ty],
    [LLVMQualPointerType<3>, llvm_v2i16_ty],
    [IntrArgMemOnly, NoCapture<ArgIndex<0>>]>,
    ClangBuiltin<"__builtin_rvgpu_ds_atomic_fadd_v2bf16">;

def int_rvgpu_mfma_i32_16x16x32_i8     : RVGPUMfmaIntrinsic<llvm_v4i32_ty,  llvm_i64_ty>;
def int_rvgpu_mfma_i32_32x32x16_i8     : RVGPUMfmaIntrinsic<llvm_v16i32_ty, llvm_i64_ty>;
def int_rvgpu_mfma_f32_16x16x8_xf32    : RVGPUMfmaIntrinsic<llvm_v4f32_ty,  llvm_v2f32_ty>;
def int_rvgpu_mfma_f32_32x32x4_xf32    : RVGPUMfmaIntrinsic<llvm_v16f32_ty, llvm_v2f32_ty>;

class RVGPUMFp8MfmaIntrinsic<LLVMType DestTy> :
  RVGPUMfmaIntrinsic<DestTy, llvm_i64_ty>;

multiclass RVGPUMFp8MfmaIntrinsic<LLVMType DestTy> {
  foreach kind = ["bf8_bf8", "bf8_fp8", "fp8_bf8", "fp8_fp8"] in
    def NAME#"_"#kind : RVGPUMFp8MfmaIntrinsic<DestTy>;
}

defm int_rvgpu_mfma_f32_16x16x32 : RVGPUMFp8MfmaIntrinsic<llvm_v4f32_ty>;
defm int_rvgpu_mfma_f32_32x32x16 : RVGPUMFp8MfmaIntrinsic<llvm_v16f32_ty>;

// llvm.rvgpu.smfmac.?32.* vdst, srcA, srcB, srcC, index, cbsz, abid
class RVGPUMSmfmacIntrinsic<LLVMType DestTy, LLVMType SrcA, LLVMType SrcB> :
  ClangBuiltin<!subst("int", "__builtin", NAME)>,
  DefaultAttrsIntrinsic<[DestTy],
            [SrcA, SrcB, DestTy, llvm_i32_ty,
             llvm_i32_ty, llvm_i32_ty],
            [IntrConvergent, IntrNoMem,
             ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

def int_rvgpu_smfmac_f32_16x16x32_f16  : RVGPUMSmfmacIntrinsic<llvm_v4f32_ty,  llvm_v4f16_ty, llvm_v8f16_ty>;
def int_rvgpu_smfmac_f32_32x32x16_f16  : RVGPUMSmfmacIntrinsic<llvm_v16f32_ty, llvm_v4f16_ty, llvm_v8f16_ty>;
def int_rvgpu_smfmac_f32_16x16x32_bf16 : RVGPUMSmfmacIntrinsic<llvm_v4f32_ty,  llvm_v4i16_ty, llvm_v8i16_ty>;
def int_rvgpu_smfmac_f32_32x32x16_bf16 : RVGPUMSmfmacIntrinsic<llvm_v16f32_ty, llvm_v4i16_ty, llvm_v8i16_ty>;
def int_rvgpu_smfmac_i32_16x16x64_i8   : RVGPUMSmfmacIntrinsic<llvm_v4i32_ty,  llvm_v2i32_ty, llvm_v4i32_ty>;
def int_rvgpu_smfmac_i32_32x32x32_i8   : RVGPUMSmfmacIntrinsic<llvm_v16i32_ty, llvm_v2i32_ty, llvm_v4i32_ty>;

class RVGPUMFp8SmfmacIntrinsic<LLVMType DestTy> :
  RVGPUMSmfmacIntrinsic<DestTy, llvm_v2i32_ty, llvm_v4i32_ty>;

multiclass RVGPUMFp8SmfmacIntrinsic<LLVMType DestTy> {
  foreach kind = ["bf8_bf8", "bf8_fp8", "fp8_bf8", "fp8_fp8"] in
    def NAME#"_"#kind : RVGPUMFp8SmfmacIntrinsic<DestTy>;
}

defm int_rvgpu_smfmac_f32_16x16x64 : RVGPUMFp8SmfmacIntrinsic<llvm_v4f32_ty>;
defm int_rvgpu_smfmac_f32_32x32x32 : RVGPUMFp8SmfmacIntrinsic<llvm_v16f32_ty>;

// llvm.rvgpu.cvt.f32.bf8 float vdst, int srcA, imm byte_sel [0..3]
// byte_sel selects byte from srcA.
def int_rvgpu_cvt_f32_bf8 : ClangBuiltin<"__builtin_rvgpu_cvt_f32_bf8">,
  DefaultAttrsIntrinsic<[llvm_float_ty],
            [llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem, ImmArg<ArgIndex<1>>]>;

// llvm.rvgpu.cvt.f32.fp8 float vdst, int srcA, imm byte_sel [0..3]
def int_rvgpu_cvt_f32_fp8 : ClangBuiltin<"__builtin_rvgpu_cvt_f32_fp8">,
  DefaultAttrsIntrinsic<[llvm_float_ty],
            [llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem, ImmArg<ArgIndex<1>>]>;

// llvm.rvgpu.cvt.pk.f32.bf8 float2 vdst, int srcA, imm word_sel
// word_sel = 1 selects 2 high bytes, 0 selects 2 low bytes.
def int_rvgpu_cvt_pk_f32_bf8 : ClangBuiltin<"__builtin_rvgpu_cvt_pk_f32_bf8">,
  DefaultAttrsIntrinsic<[llvm_v2f32_ty],
            [llvm_i32_ty, llvm_i1_ty],
            [IntrNoMem, ImmArg<ArgIndex<1>>]>;

// llvm.rvgpu.cvt.pk.f32.fp8 float2 vdst, int srcA, imm word_sel.
def int_rvgpu_cvt_pk_f32_fp8 : ClangBuiltin<"__builtin_rvgpu_cvt_pk_f32_fp8">,
  DefaultAttrsIntrinsic<[llvm_v2f32_ty],
            [llvm_i32_ty, llvm_i1_ty],
            [IntrNoMem, ImmArg<ArgIndex<1>>]>;

// llvm.rvgpu.cvt.pk.bf8.f32 int vdst, float srcA, float srcB, int old, imm word_sel
// word_sel = 1 selects 2 high bytes in the vdst, 0 selects 2 low bytes.
def int_rvgpu_cvt_pk_bf8_f32 : ClangBuiltin<"__builtin_rvgpu_cvt_pk_bf8_f32">,
  DefaultAttrsIntrinsic<[llvm_i32_ty],
            [llvm_float_ty, llvm_float_ty, llvm_i32_ty, llvm_i1_ty],
            [IntrNoMem, ImmArg<ArgIndex<3>>]>;

// llvm.rvgpu.cvt.pk.fp8.f32 int vdst, float srcA, float srcB, int old, imm word_sel
def int_rvgpu_cvt_pk_fp8_f32 : ClangBuiltin<"__builtin_rvgpu_cvt_pk_fp8_f32">,
  DefaultAttrsIntrinsic<[llvm_i32_ty],
            [llvm_float_ty, llvm_float_ty, llvm_i32_ty, llvm_i1_ty],
            [IntrNoMem, ImmArg<ArgIndex<3>>]>;

// llvm.rvgpu.cvt.sr.bf8.f32 int vdst, float srcA, int srcB, int old, imm byte_sel [0..3]
// byte_sel selects byte to write into vdst.
def int_rvgpu_cvt_sr_bf8_f32 : ClangBuiltin<"__builtin_rvgpu_cvt_sr_bf8_f32">,
  DefaultAttrsIntrinsic<[llvm_i32_ty],
            [llvm_float_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem, ImmArg<ArgIndex<3>>]>;

// llvm.rvgpu.cvt.sr.fp8.f32 int vdst, float srcA, int srcB, int old, imm byte_sel [0..3]
def int_rvgpu_cvt_sr_fp8_f32 : ClangBuiltin<"__builtin_rvgpu_cvt_sr_fp8_f32">,
  DefaultAttrsIntrinsic<[llvm_i32_ty],
            [llvm_float_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem, ImmArg<ArgIndex<3>>]>;

//===----------------------------------------------------------------------===//
// Special Intrinsics for backend internal use only. No frontend
// should emit calls to these.
// ===----------------------------------------------------------------------===//
def int_rvgpu_if : Intrinsic<[llvm_i1_ty, llvm_anyint_ty],
  [llvm_i1_ty], [IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

def int_rvgpu_else : Intrinsic<[llvm_i1_ty, llvm_anyint_ty],
  [llvm_anyint_ty], [IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

def int_rvgpu_if_break : Intrinsic<[llvm_anyint_ty],
  [llvm_i1_ty, LLVMMatchType<0>],
  [IntrNoMem, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

def int_rvgpu_loop : Intrinsic<[llvm_i1_ty],
  [llvm_anyint_ty], [IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

def int_rvgpu_end_cf : Intrinsic<[], [llvm_anyint_ty],
  [IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]>;

// Represent unreachable in a divergent region.
def int_rvgpu_unreachable : Intrinsic<[], [], [IntrConvergent, IntrNoCallback, IntrNoFree]>;

// Emit 2.5 ulp, no denormal division. Should only be inserted by
// pass based on !fpmath metadata.
def int_rvgpu_fdiv_fast : DefaultAttrsIntrinsic<
  [llvm_float_ty], [llvm_float_ty, llvm_float_ty],
  [IntrNoMem, IntrSpeculatable]
>;
}
