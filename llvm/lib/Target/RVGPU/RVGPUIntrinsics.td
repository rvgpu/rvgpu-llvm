//===- RVGPUIntrinsics.td - PTX Intrinsics Instructions -------*- tblgen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def immFloat0 : PatLeaf<(fpimm), [{
    float f = (float)N->getValueAPF().convertToFloat();
    return (f==0.0f);
}]>;

def immFloat1 : PatLeaf<(fpimm), [{
    float f = (float)N->getValueAPF().convertToFloat();
    return (f==1.0f);
}]>;

def immDouble0 : PatLeaf<(fpimm), [{
    double d = (double)N->getValueAPF().convertToDouble();
    return (d==0.0);
}]>;

def immDouble1 : PatLeaf<(fpimm), [{
    double d = (double)N->getValueAPF().convertToDouble();
    return (d==1.0);
}]>;

def AS_match {
  code generic = [{
   return ChkMemSDNodeAddressSpace(N, llvm::ADDRESS_SPACE_GENERIC);
  }];
  code shared = [{
   return ChkMemSDNodeAddressSpace(N, llvm::ADDRESS_SPACE_SHARED);
  }];
  code global = [{
   return ChkMemSDNodeAddressSpace(N, llvm::ADDRESS_SPACE_GLOBAL);
  }];
}

// A node that will be replaced with the current PTX version.
class PTX {
  SDNodeXForm PTXVerXform = SDNodeXForm<imm, [{
    return getI32Imm(Subtarget->getPTXVersion(), SDLoc(N));
  }]>;
  // (i32 0) will be XForm'ed to the currently used PTX version.
  dag version = (PTXVerXform (i32 0));
}
def ptx : PTX;

// Generates list of n sequential register names.
// E.g. RegNames<3,"r">.ret -> ["r0", "r1", "r2" ]
class RegSeq<int n, string prefix> {
  list<string> ret = !if(n, !listconcat(RegSeq<!sub(n, 1), prefix>.ret,
                                        [prefix # !sub(n, 1)]),
                            []);
}

class THREADMASK_INFO<bit sync> {
  list<bit> ret = !if(sync, [0, 1], [0]);
}

//-----------------------------------
// Synchronization and shuffle functions
//-----------------------------------
let isConvergent = true in {
def INT_BARRIER0 : RVGPUInst<(outs), (ins),
                  "bar.sync \t0",
      [(int_rvgpu_barrier0)]>;
def INT_BARRIERN : RVGPUInst<(outs), (ins GPR32:$src1),
                  "bar.sync \t$src1",
      [(int_rvgpu_barrier_n GPR32:$src1)]>;
def INT_BARRIER : RVGPUInst<(outs), (ins GPR32:$src1, GPR32:$src2),
                  "bar.sync \t$src1, $src2",
      [(int_rvgpu_barrier GPR32:$src1, GPR32:$src2)]>;

let isCodeGenOnly = 1 in {
def INT_BARRIER0_POPC : RVGPUInst<(outs GPR32:$dst), (ins GPR32:$pred),
  !strconcat("{{ \n\t",
             ".reg .pred \t%p1; \n\t",
             "setp.ne.u32 \t%p1, $pred, 0; \n\t",
             "bar.red.popc.u32 \t$dst, 0, %p1; \n\t",
             "}}"),
      [(set GPR32:$dst, (int_rvgpu_barrier0_popc GPR32:$pred))]>;
def INT_BARRIER0_AND : RVGPUInst<(outs GPR32:$dst), (ins GPR32:$pred),
  !strconcat("{{ \n\t",
             ".reg .pred \t%p1; \n\t",
             ".reg .pred \t%p2; \n\t",
             "setp.ne.u32 \t%p1, $pred, 0; \n\t",
             "bar.red.and.pred \t%p2, 0, %p1; \n\t",
             "selp.u32 \t$dst, 1, 0, %p2; \n\t",
             "}}"),
      [(set GPR32:$dst, (int_rvgpu_barrier0_and GPR32:$pred))]>;
def INT_BARRIER0_OR : RVGPUInst<(outs GPR32:$dst), (ins GPR32:$pred),
  !strconcat("{{ \n\t",
             ".reg .pred \t%p1; \n\t",
             ".reg .pred \t%p2; \n\t",
             "setp.ne.u32 \t%p1, $pred, 0; \n\t",
             "bar.red.or.pred \t%p2, 0, %p1; \n\t",
             "selp.u32 \t$dst, 1, 0, %p2; \n\t",
             "}}"),
      [(set GPR32:$dst, (int_rvgpu_barrier0_or GPR32:$pred))]>;
}

def INT_BAR_SYNC : RVGPUInst<(outs), (ins i32imm:$i), "bar.sync \t$i",
                             [(int_rvgpu_bar_sync imm:$i)]>;

def INT_BAR_WARP_SYNC_I : RVGPUInst<(outs), (ins i32imm:$i), "bar.warp.sync \t$i",
                             [(int_rvgpu_bar_warp_sync imm:$i)]>,
        Requires<[hasPTX<60>, hasSM<30>]>;
def INT_BAR_WARP_SYNC_R : RVGPUInst<(outs), (ins GPR32:$i), "bar.warp.sync \t$i",
                             [(int_rvgpu_bar_warp_sync GPR32:$i)]>,
        Requires<[hasPTX<60>, hasSM<30>]>;

def INT_BARRIER_SYNC_I : RVGPUInst<(outs), (ins i32imm:$i), "barrier.sync \t$i",
                                   [(int_rvgpu_barrier_sync imm:$i)]>,
        Requires<[hasPTX<60>, hasSM<30>]>;
def INT_BARRIER_SYNC_R : RVGPUInst<(outs), (ins GPR32:$i), "barrier.sync \t$i",
                                   [(int_rvgpu_barrier_sync GPR32:$i)]>,
        Requires<[hasPTX<60>, hasSM<30>]>;

def INT_BARRIER_SYNC_CNT_RR : RVGPUInst<(outs), (ins GPR32:$id, GPR32:$cnt),
                 "barrier.sync \t$id, $cnt",
                 [(int_rvgpu_barrier_sync_cnt GPR32:$id, GPR32:$cnt)]>,
        Requires<[hasPTX<60>, hasSM<30>]>;
def INT_BARRIER_SYNC_CNT_RI : RVGPUInst<(outs), (ins GPR32:$id, i32imm:$cnt),
                 "barrier.sync \t$id, $cnt",
                 [(int_rvgpu_barrier_sync_cnt GPR32:$id, imm:$cnt)]>,
        Requires<[hasPTX<60>, hasSM<30>]>;
def INT_BARRIER_SYNC_CNT_IR : RVGPUInst<(outs), (ins i32imm:$id, GPR32:$cnt),
                 "barrier.sync \t$id, $cnt",
                 [(int_rvgpu_barrier_sync_cnt imm:$id, GPR32:$cnt)]>,
        Requires<[hasPTX<60>, hasSM<30>]>;
def INT_BARRIER_SYNC_CNT_II : RVGPUInst<(outs), (ins i32imm:$id, i32imm:$cnt),
                 "barrier.sync \t$id, $cnt",
                 [(int_rvgpu_barrier_sync_cnt imm:$id, imm:$cnt)]>,
        Requires<[hasPTX<60>, hasSM<30>]>;
class INT_BARRIER_CLUSTER<string variant, Intrinsic Intr,
                          list<Predicate> Preds = [hasPTX<78>, hasSM<90>]>:
        RVGPUInst<(outs), (ins), "barrier.cluster."# variant #"", [(Intr)]>,
        Requires<Preds>;

def barrier_cluster_arrive:
        INT_BARRIER_CLUSTER<"arrive", int_rvgpu_barrier_cluster_arrive>;
def barrier_cluster_arrive_relaxed:
        INT_BARRIER_CLUSTER<"arrive.relaxed",
        int_rvgpu_barrier_cluster_arrive_relaxed, [hasPTX<80>, hasSM<90>]>;
def barrier_cluster_wait:
        INT_BARRIER_CLUSTER<"wait", int_rvgpu_barrier_cluster_wait>;

let isCodeGenOnly = 1 in {
class SHFL_INSTR<bit sync, string mode, string reg, bit return_pred,
                 bit offset_imm, bit mask_imm, bit threadmask_imm>
      : RVGPUInst<(outs), (ins), "?", []> {
  RVGPURegClass rc = !cond(
    !eq(reg, "i32"): GPR32,
    !eq(reg, "f32"): GPR32);
  string IntrName = "int_rvgpu_shfl_"
                    # !if(sync, "sync_", "")
                    # mode
                    # "_" # reg
                    # !if(return_pred, "p", "");
  Intrinsic Intr = !cast<Intrinsic>(IntrName);
  let InOperandList = !con(
    !if(sync,
        !dag(ins, !if(threadmask_imm, [i32imm], [GPR32]), ["threadmask"]),
        (ins)),
    (ins rc:$src),
    !dag(ins, !if(offset_imm, [i32imm], [GPR32]), ["offset"]),
    !dag(ins, !if(mask_imm, [i32imm], [GPR32]), ["mask"])
    );
  let OutOperandList = !if(return_pred, (outs rc:$dst, Int1Regs:$pred), (outs rc:$dst));
  let AsmString = "shfl."
     # !if(sync, "sync.", "")
     # mode # ".b32\t"
     # "$dst"
     # !if(return_pred, "|$pred", "") # ", "
     # "$src, $offset, $mask"
     # !if(sync, ", $threadmask", "")
     # ""
     ;
  let Pattern = [!con(
      !foreach(tmp, OutOperandList,
             !subst(outs, set,
             !subst(i32imm, imm, tmp))),
      (set !foreach(tmp, InOperandList,
             !subst(ins, Intr,
             !subst(i32imm, imm, tmp))))
  )];
}
}

foreach sync = [false, true] in {
  foreach mode = ["up", "down", "bfly", "idx"] in {
    foreach regclass = ["i32", "f32"] in {
      foreach return_pred = [false, true] in {
        foreach offset_imm = [false, true] in {
          foreach mask_imm = [false, true] in {
            foreach threadmask_imm = THREADMASK_INFO<sync>.ret in {
              def : SHFL_INSTR<sync, mode, regclass, return_pred,
                               offset_imm, mask_imm, threadmask_imm>,
                    Requires<!if(sync, [hasSM<30>, hasPTX<60>], [hasSM<30>, hasSHFL])>;
            }
          }
        }
      }
    }
  }
}

// vote.{all,any,uni,ballot}
multiclass VOTE<RVGPURegClass regclass, string mode, Intrinsic IntOp> {
  def : RVGPUInst<(outs regclass:$dest), (ins Int1Regs:$pred),
              "vote." # mode # " \t$dest, $pred",
              [(set regclass:$dest, (IntOp Int1Regs:$pred))]>,
        Requires<[hasPTX<60>, hasSM<30>]>;
}

defm VOTE_ALL : VOTE<Int1Regs, "all.pred", int_rvgpu_vote_all>;
defm VOTE_ANY : VOTE<Int1Regs, "any.pred", int_rvgpu_vote_any>;
defm VOTE_UNI : VOTE<Int1Regs, "uni.pred", int_rvgpu_vote_uni>;
defm VOTE_BALLOT : VOTE<GPR32, "ballot.b32", int_rvgpu_vote_ballot>;

// vote.sync.{all,any,uni,ballot}
multiclass VOTE_SYNC<RVGPURegClass regclass, string mode, Intrinsic IntOp> {
  def i : RVGPUInst<(outs regclass:$dest), (ins i32imm:$mask, Int1Regs:$pred),
              "vote.sync." # mode # " \t$dest, $pred, $mask",
              [(set regclass:$dest, (IntOp imm:$mask, Int1Regs:$pred))]>,
          Requires<[hasPTX<60>, hasSM<30>]>;
  def r : RVGPUInst<(outs regclass:$dest), (ins GPR32:$mask, Int1Regs:$pred),
              "vote.sync." # mode #" \t$dest, $pred, $mask",
              [(set regclass:$dest, (IntOp GPR32:$mask, Int1Regs:$pred))]>,
          Requires<[hasPTX<60>, hasSM<30>]>;
}

defm VOTE_SYNC_ALL : VOTE_SYNC<Int1Regs, "all.pred", int_rvgpu_vote_all_sync>;
defm VOTE_SYNC_ANY : VOTE_SYNC<Int1Regs, "any.pred", int_rvgpu_vote_any_sync>;
defm VOTE_SYNC_UNI : VOTE_SYNC<Int1Regs, "uni.pred", int_rvgpu_vote_uni_sync>;
defm VOTE_SYNC_BALLOT : VOTE_SYNC<GPR32, "ballot.b32", int_rvgpu_vote_ballot_sync>;

multiclass MATCH_ANY_SYNC<RVGPURegClass regclass, string ptxtype, Intrinsic IntOp,
                          Operand ImmOp> {
  def ii : RVGPUInst<(outs GPR32:$dest), (ins i32imm:$mask, ImmOp:$value),
              "match.any.sync." # ptxtype # " \t$dest, $value, $mask",
              [(set GPR32:$dest, (IntOp imm:$mask, imm:$value))]>,
           Requires<[hasPTX<60>, hasSM<70>]>;
  def ir : RVGPUInst<(outs GPR32:$dest), (ins GPR32:$mask, ImmOp:$value),
              "match.any.sync." # ptxtype # " \t$dest, $value, $mask",
              [(set GPR32:$dest, (IntOp GPR32:$mask, imm:$value))]>,
           Requires<[hasPTX<60>, hasSM<70>]>;
  def ri : RVGPUInst<(outs GPR32:$dest), (ins i32imm:$mask, regclass:$value),
              "match.any.sync." # ptxtype # " \t$dest, $value, $mask",
              [(set GPR32:$dest, (IntOp imm:$mask, regclass:$value))]>,
           Requires<[hasPTX<60>, hasSM<70>]>;
  def rr : RVGPUInst<(outs GPR32:$dest), (ins GPR32:$mask, regclass:$value),
              "match.any.sync." # ptxtype # " \t$dest, $value, $mask",
              [(set GPR32:$dest, (IntOp GPR32:$mask, regclass:$value))]>,
           Requires<[hasPTX<60>, hasSM<70>]>;
}

defm MATCH_ANY_SYNC_32 : MATCH_ANY_SYNC<GPR32, "b32", int_rvgpu_match_any_sync_i32,
                                        i32imm>;
defm MATCH_ANY_SYNC_64 : MATCH_ANY_SYNC<GPR64, "b64", int_rvgpu_match_any_sync_i64,
                                        i64imm>;
let isCodeGenOnly = 1 in {
multiclass MATCH_ALLP_SYNC<RVGPURegClass regclass, string ptxtype, Intrinsic IntOp,
                          Operand ImmOp> {
  def ii : RVGPUInst<(outs GPR32:$dest, Int1Regs:$pred),
                     (ins i32imm:$mask, ImmOp:$value),
              "match.all.sync." # ptxtype # " \t$dest|$pred, $value, $mask",
              [(set GPR32:$dest, Int1Regs:$pred, (IntOp imm:$mask, imm:$value))]>,
           Requires<[hasPTX<60>, hasSM<70>]>;
  def ir : RVGPUInst<(outs GPR32:$dest, Int1Regs:$pred),
                     (ins GPR32:$mask, ImmOp:$value),
              "match.all.sync." # ptxtype # " \t$dest|$pred, $value, $mask",
              [(set GPR32:$dest, Int1Regs:$pred, (IntOp GPR32:$mask, imm:$value))]>,
           Requires<[hasPTX<60>, hasSM<70>]>;
  def ri : RVGPUInst<(outs GPR32:$dest, Int1Regs:$pred),
                     (ins i32imm:$mask, regclass:$value),
              "match.all.sync." # ptxtype # " \t$dest|$pred, $value, $mask",
              [(set GPR32:$dest, Int1Regs:$pred, (IntOp imm:$mask, regclass:$value))]>,
           Requires<[hasPTX<60>, hasSM<70>]>;
  def rr : RVGPUInst<(outs GPR32:$dest, Int1Regs:$pred),
                     (ins GPR32:$mask, regclass:$value),
              "match.all.sync." # ptxtype # " \t$dest|$pred, $value, $mask",
              [(set GPR32:$dest, Int1Regs:$pred, (IntOp GPR32:$mask, regclass:$value))]>,
           Requires<[hasPTX<60>, hasSM<70>]>;
}
}
defm MATCH_ALLP_SYNC_32 : MATCH_ALLP_SYNC<GPR32, "b32", int_rvgpu_match_all_sync_i32p,
                                         i32imm>;
defm MATCH_ALLP_SYNC_64 : MATCH_ALLP_SYNC<GPR64, "b64", int_rvgpu_match_all_sync_i64p,
                                         i64imm>;

multiclass REDUX_SYNC<string BinOp, string PTXType, Intrinsic Intrin> {
  def : RVGPUInst<(outs GPR32:$dst), (ins GPR32:$src, GPR32:$mask),
          "redux.sync." # BinOp # "." # PTXType # " $dst, $src, $mask",
          [(set GPR32:$dst, (Intrin GPR32:$src, GPR32:$mask))]>,
        Requires<[hasPTX<70>, hasSM<80>]>;
}

defm REDUX_SYNC_UMIN : REDUX_SYNC<"min", "u32", int_rvgpu_redux_sync_umin>;
defm REDUX_SYNC_UMAX : REDUX_SYNC<"max", "u32", int_rvgpu_redux_sync_umax>;
defm REDUX_SYNC_ADD : REDUX_SYNC<"add", "s32", int_rvgpu_redux_sync_add>;
defm REDUX_SYNC_MIN : REDUX_SYNC<"min", "s32", int_rvgpu_redux_sync_min>;
defm REDUX_SYNC_MAX : REDUX_SYNC<"max", "s32", int_rvgpu_redux_sync_max>;
defm REDUX_SYNC_AND : REDUX_SYNC<"and", "b32", int_rvgpu_redux_sync_and>;
defm REDUX_SYNC_XOR : REDUX_SYNC<"xor", "b32", int_rvgpu_redux_sync_xor>;
defm REDUX_SYNC_OR : REDUX_SYNC<"or", "b32", int_rvgpu_redux_sync_or>;

} // isConvergent = true

//-----------------------------------
// Explicit Memory Fence Functions
//-----------------------------------
class MEMBAR<string StrOp, Intrinsic IntOP> :
              RVGPUInst<(outs), (ins),
            StrOp, [(IntOP)]>;

def INT_MEMBAR_CTA : MEMBAR<"membar.cta", int_rvgpu_membar_cta>;
def INT_MEMBAR_GL  : MEMBAR<"membar.gl",  int_rvgpu_membar_gl>;
def INT_MEMBAR_SYS : MEMBAR<"membar.sys", int_rvgpu_membar_sys>;

def INT_FENCE_SC_CLUSTER:
       MEMBAR<"fence.sc.cluster", int_rvgpu_fence_sc_cluster>,
       Requires<[hasPTX<78>, hasSM<90>]>;

//-----------------------------------
// Async Copy Functions
//-----------------------------------

multiclass CP_ASYNC_MBARRIER_ARRIVE<string NoInc, string AddrSpace, Intrinsic Intrin> {
  def _32 : RVGPUInst<(outs), (ins GPR32:$addr),
            !strconcat("cp.async.mbarrier.arrive", NoInc, AddrSpace, ".b64 [$addr]"),
            [(Intrin GPR32:$addr)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64 : RVGPUInst<(outs), (ins GPR64:$addr),
            !strconcat("cp.async.mbarrier.arrive", NoInc, AddrSpace, ".b64 [$addr]"),
            [(Intrin GPR64:$addr)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
}

defm CP_ASYNC_MBARRIER_ARRIVE :
  CP_ASYNC_MBARRIER_ARRIVE<"", "", int_rvgpu_cp_async_mbarrier_arrive>;
defm CP_ASYNC_MBARRIER_ARRIVE_SHARED :
  CP_ASYNC_MBARRIER_ARRIVE<"", ".shared", int_rvgpu_cp_async_mbarrier_arrive_shared>;
defm CP_ASYNC_MBARRIER_ARRIVE_NOINC :
  CP_ASYNC_MBARRIER_ARRIVE<".noinc", "", int_rvgpu_cp_async_mbarrier_arrive_noinc>;
defm CP_ASYNC_MBARRIER_ARRIVE_NOINC_SHARED :
  CP_ASYNC_MBARRIER_ARRIVE<".noinc", ".shared", int_rvgpu_cp_async_mbarrier_arrive_noinc_shared>;

multiclass CP_ASYNC_SHARED_GLOBAL_I<string cc, string cpsize, Intrinsic Intrin, Intrinsic IntrinS> {
  def _32 : RVGPUInst<(outs), (ins GPR32:$dst, GPR32:$src),
            !strconcat("cp.async.", cc, ".shared.global [$dst], [$src], ", cpsize, ""),
            [(Intrin GPR32:$dst, GPR32:$src)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64 : RVGPUInst<(outs), (ins GPR64:$dst, GPR64:$src),
            !strconcat("cp.async.", cc, ".shared.global [$dst], [$src], ", cpsize, ""),
            [(Intrin GPR64:$dst, GPR64:$src)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  // Variant with src_size parameter
  def _32s : RVGPUInst<(outs), (ins GPR32:$dst, GPR32:$src, GPR32:$src_size),
             !strconcat("cp.async.", cc, ".shared.global [$dst], [$src], ", cpsize, ", $src_size"),
             [(IntrinS GPR32:$dst, GPR32:$src, GPR32:$src_size)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _32si: RVGPUInst<(outs), (ins GPR32:$dst, GPR32:$src, i32imm:$src_size),
             !strconcat("cp.async.", cc, ".shared.global [$dst], [$src], ", cpsize, ", $src_size"),
             [(IntrinS GPR32:$dst, GPR32:$src, imm:$src_size)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64s : RVGPUInst<(outs), (ins GPR64:$dst, GPR64:$src, GPR32:$src_size),
             !strconcat("cp.async.", cc, ".shared.global [$dst], [$src], ", cpsize, ", $src_size"),
             [(IntrinS GPR64:$dst, GPR64:$src, GPR32:$src_size)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64si: RVGPUInst<(outs), (ins GPR64:$dst, GPR64:$src, i32imm:$src_size),
             !strconcat("cp.async.", cc, ".shared.global [$dst], [$src], ", cpsize, ", $src_size"),
             [(IntrinS GPR64:$dst, GPR64:$src, imm:$src_size)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
}

defm CP_ASYNC_CA_SHARED_GLOBAL_4 :
  CP_ASYNC_SHARED_GLOBAL_I<"ca", "4", int_rvgpu_cp_async_ca_shared_global_4,
                                      int_rvgpu_cp_async_ca_shared_global_4_s>;

defm CP_ASYNC_CA_SHARED_GLOBAL_8 :
  CP_ASYNC_SHARED_GLOBAL_I<"ca", "8", int_rvgpu_cp_async_ca_shared_global_8,
                                      int_rvgpu_cp_async_ca_shared_global_8_s>;

defm CP_ASYNC_CA_SHARED_GLOBAL_16 :
  CP_ASYNC_SHARED_GLOBAL_I<"ca", "16", int_rvgpu_cp_async_ca_shared_global_16,
                                       int_rvgpu_cp_async_ca_shared_global_16_s>;

defm CP_ASYNC_CG_SHARED_GLOBAL_16 :
  CP_ASYNC_SHARED_GLOBAL_I<"cg", "16", int_rvgpu_cp_async_cg_shared_global_16,
                                       int_rvgpu_cp_async_cg_shared_global_16_s>;

def CP_ASYNC_COMMIT_GROUP :
  RVGPUInst<(outs), (ins), "cp.async.commit_group", [(int_rvgpu_cp_async_commit_group)]>,
  Requires<[hasPTX<70>, hasSM<80>]>;

def CP_ASYNC_WAIT_GROUP :
  RVGPUInst<(outs), (ins i32imm:$n), "cp.async.wait_group $n",
  [(int_rvgpu_cp_async_wait_group (i32 timm:$n))]>,
  Requires<[hasPTX<70>, hasSM<80>]>;

def CP_ASYNC_WAIT_ALL :
  RVGPUInst<(outs), (ins), "cp.async.wait_all",
  [(int_rvgpu_cp_async_wait_all)]>,
  Requires<[hasPTX<70>, hasSM<80>]>;

//-----------------------------------
// MBarrier Functions
//-----------------------------------

multiclass MBARRIER_INIT<string AddrSpace, Intrinsic Intrin> {
  def _32 : RVGPUInst<(outs), (ins GPR32:$addr, GPR32:$count),
           !strconcat("mbarrier.init", AddrSpace, ".b64 [$addr], $count"),
    [(Intrin GPR32:$addr, GPR32:$count)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64 : RVGPUInst<(outs), (ins GPR64:$addr, GPR32:$count),
           !strconcat("mbarrier.init", AddrSpace, ".b64 [$addr], $count"),
    [(Intrin GPR64:$addr, GPR32:$count)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
}

defm MBARRIER_INIT : MBARRIER_INIT<"", int_rvgpu_mbarrier_init>;
defm MBARRIER_INIT_SHARED : MBARRIER_INIT<".shared",
                                          int_rvgpu_mbarrier_init_shared>;

multiclass MBARRIER_INVAL<string AddrSpace, Intrinsic Intrin> {
  def _32 : RVGPUInst<(outs), (ins GPR32:$addr),
           !strconcat("mbarrier.inval", AddrSpace, ".b64 [$addr]"),
    [(Intrin GPR32:$addr)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64 : RVGPUInst<(outs), (ins GPR64:$addr),
           !strconcat("mbarrier.inval", AddrSpace, ".b64 [$addr]"),
    [(Intrin GPR64:$addr)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
}

defm MBARRIER_INVAL : MBARRIER_INVAL<"", int_rvgpu_mbarrier_inval>;
defm MBARRIER_INVAL_SHARED : MBARRIER_INVAL<".shared",
                                            int_rvgpu_mbarrier_inval_shared>;

multiclass MBARRIER_ARRIVE<string AddrSpace, Intrinsic Intrin> {
  def _32 : RVGPUInst<(outs GPR64:$state), (ins GPR32:$addr),
           !strconcat("mbarrier.arrive", AddrSpace, ".b64 $state, [$addr]"),
    [(set GPR64:$state, (Intrin GPR32:$addr))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64 : RVGPUInst<(outs GPR64:$state), (ins GPR64:$addr),
           !strconcat("mbarrier.arrive", AddrSpace, ".b64 $state, [$addr]"),
    [(set GPR64:$state, (Intrin GPR64:$addr))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
}

defm MBARRIER_ARRIVE : MBARRIER_ARRIVE<"", int_rvgpu_mbarrier_arrive>;
defm MBARRIER_ARRIVE_SHARED :
  MBARRIER_ARRIVE<".shared", int_rvgpu_mbarrier_arrive_shared>;

multiclass MBARRIER_ARRIVE_NOCOMPLETE<string AddrSpace, Intrinsic Intrin> {
  def _32 : RVGPUInst<(outs GPR64:$state),
           (ins GPR32:$addr, GPR32:$count),
           !strconcat("mbarrier.arrive.noComplete", AddrSpace,
                      ".b64 $state, [$addr], $count"),
    [(set GPR64:$state, (Intrin GPR32:$addr, GPR32:$count))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64 : RVGPUInst<(outs GPR64:$state),
           (ins GPR64:$addr, GPR32:$count),
           !strconcat("mbarrier.arrive.noComplete", AddrSpace,
                      ".b64 $state, [$addr], $count"),
    [(set GPR64:$state, (Intrin GPR64:$addr, GPR32:$count))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
}

defm MBARRIER_ARRIVE_NOCOMPLETE :
  MBARRIER_ARRIVE_NOCOMPLETE<"", int_rvgpu_mbarrier_arrive_noComplete>;
defm MBARRIER_ARRIVE_NOCOMPLETE_SHARED :
  MBARRIER_ARRIVE_NOCOMPLETE<".shared", int_rvgpu_mbarrier_arrive_noComplete_shared>;

multiclass MBARRIER_ARRIVE_DROP<string AddrSpace, Intrinsic Intrin> {
  def _32 : RVGPUInst<(outs GPR64:$state), (ins GPR32:$addr),
           !strconcat("mbarrier.arrive_drop", AddrSpace,
                      ".b64 $state, [$addr]"),
           [(set GPR64:$state, (Intrin GPR32:$addr))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64 : RVGPUInst<(outs GPR64:$state), (ins GPR64:$addr),
           !strconcat("mbarrier.arrive_drop", AddrSpace,
                      ".b64 $state, [$addr]"),
           [(set GPR64:$state, (Intrin GPR64:$addr))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
}

defm MBARRIER_ARRIVE_DROP :
  MBARRIER_ARRIVE_DROP<"", int_rvgpu_mbarrier_arrive_drop>;
defm MBARRIER_ARRIVE_DROP_SHARED :
  MBARRIER_ARRIVE_DROP<".shared", int_rvgpu_mbarrier_arrive_drop_shared>;

multiclass MBARRIER_ARRIVE_DROP_NOCOMPLETE<string AddrSpace, Intrinsic Intrin> {
  def _32 : RVGPUInst<(outs GPR64:$state),
           (ins GPR32:$addr, GPR32:$count),
           !strconcat("mbarrier.arrive_drop.noComplete", AddrSpace,
                      ".b64 $state, [$addr], $count"),
           [(set GPR64:$state, (Intrin GPR32:$addr, GPR32:$count))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64 : RVGPUInst<(outs GPR64:$state),
           (ins GPR64:$addr, GPR32:$count),
           !strconcat("mbarrier.arrive_drop.noComplete", AddrSpace,
                      ".b64 $state, [$addr], $count"),
           [(set GPR64:$state, (Intrin GPR64:$addr, GPR32:$count))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
}

defm MBARRIER_ARRIVE_DROP_NOCOMPLETE :
  MBARRIER_ARRIVE_DROP_NOCOMPLETE<"", int_rvgpu_mbarrier_arrive_drop_noComplete>;
defm MBARRIER_ARRIVE_DROP_NOCOMPLETE_SHARED :
  MBARRIER_ARRIVE_DROP_NOCOMPLETE<".shared",
                       int_rvgpu_mbarrier_arrive_drop_noComplete_shared>;

multiclass MBARRIER_TEST_WAIT<string AddrSpace, Intrinsic Intrin> {
  def _32 : RVGPUInst<(outs Int1Regs:$res), (ins GPR32:$addr, GPR64:$state),
           !strconcat("mbarrier.test_wait", AddrSpace, ".b64 $res, [$addr], $state"),
           [(set Int1Regs:$res, (Intrin GPR32:$addr, GPR64:$state))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64 : RVGPUInst<(outs Int1Regs:$res), (ins GPR64:$addr, GPR64:$state),
           !strconcat("mbarrier.test_wait", AddrSpace, ".b64 $res, [$addr], $state"),
           [(set Int1Regs:$res, (Intrin GPR64:$addr, GPR64:$state))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
}

defm MBARRIER_TEST_WAIT :
  MBARRIER_TEST_WAIT<"", int_rvgpu_mbarrier_test_wait>;
defm MBARRIER_TEST_WAIT_SHARED :
  MBARRIER_TEST_WAIT<".shared", int_rvgpu_mbarrier_test_wait_shared>;

class MBARRIER_PENDING_COUNT<Intrinsic Intrin> :
           RVGPUInst<(outs GPR32:$res), (ins GPR64:$state),
           "mbarrier.pending_count.b64 $res, $state",
           [(set GPR32:$res, (Intrin GPR64:$state))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;

def MBARRIER_PENDING_COUNT :
  MBARRIER_PENDING_COUNT<int_rvgpu_mbarrier_pending_count>;

//-----------------------------------
// Math Functions
//-----------------------------------

// Map min(1.0, max(0.0, x)) to sat(x)
// Note that max(0.0, min(x, 1.0)) cannot be mapped to sat(x) because when x is
// NaN
// max(0.0, min(x, 1.0)) is 1.0 while sat(x) is 0.
// Same story for fmax, fmin.

def : Pat<(int_rvgpu_fmin_f immFloat1,
            (int_rvgpu_fmax_f immFloat0, GPR32:$a)),
          (CVT_f32_f32 GPR32:$a, CvtSAT)>;
def : Pat<(int_rvgpu_fmin_f immFloat1,
            (int_rvgpu_fmax_f GPR32:$a, immFloat0)),
          (CVT_f32_f32 GPR32:$a, CvtSAT)>;
def : Pat<(int_rvgpu_fmin_f
            (int_rvgpu_fmax_f immFloat0, GPR32:$a), immFloat1),
          (CVT_f32_f32 GPR32:$a, CvtSAT)>;
def : Pat<(int_rvgpu_fmin_f
            (int_rvgpu_fmax_f GPR32:$a, immFloat0), immFloat1),
          (CVT_f32_f32 GPR32:$a, CvtSAT)>;

def : Pat<(int_rvgpu_fmin_d immDouble1,
            (int_rvgpu_fmax_d immDouble0, GPR64:$a)),
          (CVT_f64_f64 GPR64:$a, CvtSAT)>;
def : Pat<(int_rvgpu_fmin_d immDouble1,
            (int_rvgpu_fmax_d GPR64:$a, immDouble0)),
          (CVT_f64_f64 GPR64:$a, CvtSAT)>;
def : Pat<(int_rvgpu_fmin_d
            (int_rvgpu_fmax_d immDouble0, GPR64:$a), immDouble1),
          (CVT_f64_f64 GPR64:$a, CvtSAT)>;
def : Pat<(int_rvgpu_fmin_d
            (int_rvgpu_fmax_d GPR64:$a, immDouble0), immDouble1),
          (CVT_f64_f64 GPR64:$a, CvtSAT)>;


// We need a full string for OpcStr here because we need to deal with case like
// INT_PTX_RECIP.
class F_MATH_1<string OpcStr, RVGPURegClass target_regclass,
  RVGPURegClass src_regclass, Intrinsic IntOP, list<Predicate> Preds = []>
            : RVGPUInst<(outs target_regclass:$dst), (ins src_regclass:$src0),
            OpcStr,
        [(set target_regclass:$dst, (IntOP src_regclass:$src0))]>,
        Requires<Preds>;

// We need a full string for OpcStr here because we need to deal with the case
// like INT_PTX_NATIVE_POWR_F.
class F_MATH_2<string OpcStr, RVGPURegClass t_regclass,
  RVGPURegClass s0_regclass, RVGPURegClass s1_regclass, Intrinsic IntOP,
  list<Predicate> Preds = []>
            : RVGPUInst<(outs t_regclass:$dst),
              (ins s0_regclass:$src0, s1_regclass:$src1),
            OpcStr,
        [(set t_regclass:$dst, (IntOP s0_regclass:$src0, s1_regclass:$src1))]>,
        Requires<Preds>;

class F_MATH_3<string OpcStr, RVGPURegClass t_regclass,
  RVGPURegClass s0_regclass, RVGPURegClass s1_regclass,
  RVGPURegClass s2_regclass, Intrinsic IntOP, list<Predicate> Preds = []>
            : RVGPUInst<(outs t_regclass:$dst),
              (ins s0_regclass:$src0, s1_regclass:$src1, s2_regclass:$src2),
            OpcStr,
        [(set t_regclass:$dst,
          (IntOP s0_regclass:$src0, s1_regclass:$src1, s2_regclass:$src2))]>,
          Requires<Preds>;

//
// MISC
//

def INT_NVVM_PRMT : F_MATH_3<"prmt.b32 \t$dst, $src0, $src1, $src2", GPR32,
  GPR32, GPR32, GPR32, int_rvgpu_prmt>;

//
// Min Max
//

def INT_NVVM_FMIN_F : F_MATH_2<"min.f32 \t$dst, $src0, $src1", GPR32,
  GPR32, GPR32, int_rvgpu_fmin_f>;
def INT_NVVM_FMIN_FTZ_F : F_MATH_2<"min.ftz.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_fmin_ftz_f>;
def INT_NVVM_FMIN_NAN_F : F_MATH_2<"min.NaN.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_fmin_nan_f,
  [hasPTX<70>, hasSM<80>]>;
def INT_NVVM_FMIN_FTZ_NAN_F : F_MATH_2<"min.ftz.NaN.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_fmin_ftz_nan_f,
  [hasPTX<70>, hasSM<80>]>;
def INT_NVVM_FMIN_XORSIGN_ABS_F :
  F_MATH_2<"min.xorsign.abs.f32 \t$dst, $src0, $src1",
    GPR32, GPR32, GPR32, int_rvgpu_fmin_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;
def INT_NVVM_FMIN_FTZ_XORSIGN_ABS_F :
  F_MATH_2<"min.ftz.xorsign.abs.f32 \t$dst, $src0, $src1",
    GPR32, GPR32, GPR32, int_rvgpu_fmin_ftz_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;
def INT_NVVM_FMIN_NAN_XORSIGN_ABS_F :
  F_MATH_2<"min.NaN.xorsign.abs.f32 \t$dst, $src0, $src1",
    GPR32, GPR32, GPR32, int_rvgpu_fmin_nan_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;
def INT_NVVM_FMIN_FTZ_NAN_XORSIGN_ABS_F :
  F_MATH_2<"min.ftz.NaN.xorsign.abs.f32 \t$dst, $src0, $src1",
    GPR32, GPR32, GPR32, int_rvgpu_fmin_ftz_nan_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;

def INT_NVVM_FMAX_F : F_MATH_2<"max.f32 \t$dst, $src0, $src1", GPR32,
  GPR32, GPR32, int_rvgpu_fmax_f>;
def INT_NVVM_FMAX_FTZ_F : F_MATH_2<"max.ftz.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_fmax_ftz_f>;
def INT_NVVM_FMAX_NAN_F : F_MATH_2<"max.NaN.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_fmax_nan_f,
  [hasPTX<70>, hasSM<80>]>;
def INT_NVVM_FMAX_FTZ_NAN_F : F_MATH_2<"max.ftz.NaN.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_fmax_ftz_nan_f,
  [hasPTX<70>, hasSM<80>]>;
def INT_NVVM_FMAX_XORSIGN_ABS_F :
  F_MATH_2<"max.xorsign.abs.f32 \t$dst, $src0, $src1",
    GPR32, GPR32, GPR32, int_rvgpu_fmax_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;
def INT_NVVM_FMAX_FTZ_XORSIGN_ABS_F :
  F_MATH_2<"max.ftz.xorsign.abs.f32 \t$dst, $src0, $src1",
    GPR32, GPR32, GPR32, int_rvgpu_fmax_ftz_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;
def INT_NVVM_FMAX_NAN_XORSIGN_ABS_F :
  F_MATH_2<"max.NaN.xorsign.abs.f32 \t$dst, $src0, $src1",
    GPR32, GPR32, GPR32, int_rvgpu_fmax_nan_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;
def INT_NVVM_FMAX_FTZ_NAN_XORSIGN_ABS_F :
  F_MATH_2<"max.ftz.NaN.xorsign.abs.f32 \t$dst, $src0, $src1",
    GPR32, GPR32, GPR32, int_rvgpu_fmax_ftz_nan_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;

def INT_NVVM_FMIN_D : F_MATH_2<"min.f64 \t$dst, $src0, $src1", GPR64,
  GPR64, GPR64, int_rvgpu_fmin_d>;
def INT_NVVM_FMAX_D : F_MATH_2<"max.f64 \t$dst, $src0, $src1", GPR64,
  GPR64, GPR64, int_rvgpu_fmax_d>;

//
// Min Max f16, f16x2, bf16, bf16x2
//

class MIN_MAX_TUPLE<string V, Intrinsic I, RVGPURegClass RC,
                    list<Predicate> Preds = [hasPTX<70>, hasSM<80>]> {
  string Variant = V;
  Intrinsic Intr = I;
  RVGPURegClass RegClass = RC;
  list<Predicate> Predicates = Preds;
}

multiclass MIN_MAX<string IntName> {
  foreach P = [
    MIN_MAX_TUPLE<"_f16", !if(!eq(IntName, "min"), int_rvgpu_fmin_f16,
      int_rvgpu_fmax_f16), GPR16>,
    MIN_MAX_TUPLE<"_ftz_f16", !if(!eq(IntName, "min"), int_rvgpu_fmin_ftz_f16,
      int_rvgpu_fmax_ftz_f16), GPR16>,
    MIN_MAX_TUPLE<"_NaN_f16", !if(!eq(IntName, "min"), int_rvgpu_fmin_nan_f16,
      int_rvgpu_fmax_nan_f16), GPR16>,
    MIN_MAX_TUPLE<"_ftz_NaN_f16", !if(!eq(IntName, "min"),
      int_rvgpu_fmin_ftz_nan_f16, int_rvgpu_fmax_ftz_nan_f16), GPR16>,
    MIN_MAX_TUPLE<"_xorsign_abs_f16", !if(!eq(IntName, "min"),
      int_rvgpu_fmin_xorsign_abs_f16, int_rvgpu_fmax_xorsign_abs_f16),
      GPR16, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_ftz_xorsign_abs_f16", !if(!eq(IntName, "min"),
      int_rvgpu_fmin_ftz_xorsign_abs_f16, int_rvgpu_fmax_ftz_xorsign_abs_f16),
      GPR16, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_NaN_xorsign_abs_f16", !if(!eq(IntName, "min"),
      int_rvgpu_fmin_nan_xorsign_abs_f16, int_rvgpu_fmax_nan_xorsign_abs_f16),
      GPR16, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_ftz_NaN_xorsign_abs_f16", !if(!eq(IntName, "min"),
      int_rvgpu_fmin_ftz_nan_xorsign_abs_f16,
      int_rvgpu_fmax_ftz_nan_xorsign_abs_f16), GPR16, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_f16x2", !if(!eq(IntName, "min"), int_rvgpu_fmin_f16x2,
      int_rvgpu_fmax_f16x2), GPR32>,
    MIN_MAX_TUPLE<"_ftz_f16x2", !if(!eq(IntName, "min"),
      int_rvgpu_fmin_ftz_f16x2, int_rvgpu_fmax_ftz_f16x2), GPR32>,
    MIN_MAX_TUPLE<"_NaN_f16x2", !if(!eq(IntName, "min"),
      int_rvgpu_fmin_nan_f16x2, int_rvgpu_fmax_nan_f16x2), GPR32>,
    MIN_MAX_TUPLE<"_ftz_NaN_f16x2", !if(!eq(IntName, "min"),
      int_rvgpu_fmin_ftz_nan_f16x2, int_rvgpu_fmax_ftz_nan_f16x2), GPR32>,
    MIN_MAX_TUPLE<"_xorsign_abs_f16x2", !if(!eq(IntName, "min"),
      int_rvgpu_fmin_xorsign_abs_f16x2, int_rvgpu_fmax_xorsign_abs_f16x2),
      GPR32, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_ftz_xorsign_abs_f16x2", !if(!eq(IntName, "min"),
      int_rvgpu_fmin_ftz_xorsign_abs_f16x2, int_rvgpu_fmax_ftz_xorsign_abs_f16x2),
      GPR32, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_NaN_xorsign_abs_f16x2", !if(!eq(IntName, "min"),
      int_rvgpu_fmin_nan_xorsign_abs_f16x2, int_rvgpu_fmax_nan_xorsign_abs_f16x2),
      GPR32, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_ftz_NaN_xorsign_abs_f16x2", !if(!eq(IntName, "min"),
      int_rvgpu_fmin_ftz_nan_xorsign_abs_f16x2,
      int_rvgpu_fmax_ftz_nan_xorsign_abs_f16x2),
      GPR32, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_bf16", !if(!eq(IntName, "min"),
      int_rvgpu_fmin_bf16, int_rvgpu_fmax_bf16), GPR16>,
    MIN_MAX_TUPLE<"_NaN_bf16", !if(!eq(IntName, "min"), int_rvgpu_fmin_nan_bf16,
      int_rvgpu_fmax_nan_bf16), GPR16>,
    MIN_MAX_TUPLE<"_xorsign_abs_bf16", !if(!eq(IntName, "min"),
      int_rvgpu_fmin_xorsign_abs_bf16, int_rvgpu_fmax_xorsign_abs_bf16),
      GPR16, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_NaN_xorsign_abs_bf16", !if(!eq(IntName, "min"),
      int_rvgpu_fmin_nan_xorsign_abs_bf16, int_rvgpu_fmax_nan_xorsign_abs_bf16),
      GPR16, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_bf16x2", !if(!eq(IntName, "min"), int_rvgpu_fmin_bf16x2,
      int_rvgpu_fmax_bf16x2), GPR32>,
    MIN_MAX_TUPLE<"_NaN_bf16x2", !if(!eq(IntName, "min"),
      int_rvgpu_fmin_nan_bf16x2, int_rvgpu_fmax_nan_bf16x2), GPR32>,
    MIN_MAX_TUPLE<"_xorsign_abs_bf16x2", !if(!eq(IntName, "min"),
      int_rvgpu_fmin_xorsign_abs_bf16x2, int_rvgpu_fmax_xorsign_abs_bf16x2),
      GPR32, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_NaN_xorsign_abs_bf16x2", !if(!eq(IntName, "min"),
      int_rvgpu_fmin_nan_xorsign_abs_bf16x2,
      int_rvgpu_fmax_nan_xorsign_abs_bf16x2),
      GPR32, [hasPTX<72>, hasSM<86>]>] in {
        def P.Variant : F_MATH_2<!strconcat(
          IntName, !subst("_", ".", P.Variant), " \t$dst, $src0, $src1"),
          P.RegClass, P.RegClass, P.RegClass, P.Intr, P.Predicates>;
  }
}

defm INT_NVVM_FMIN : MIN_MAX<"min">;
defm INT_NVVM_FMAN : MIN_MAX<"max">;

//
// Multiplication
//

def INT_NVVM_MULHI_I : F_MATH_2<"mul.hi.s32 \t$dst, $src0, $src1", GPR32,
  GPR32, GPR32, int_rvgpu_mulhi_i>;
def INT_NVVM_MULHI_UI : F_MATH_2<"mul.hi.u32 \t$dst, $src0, $src1", GPR32,
  GPR32, GPR32, int_rvgpu_mulhi_ui>;

def INT_NVVM_MULHI_LL : F_MATH_2<"mul.hi.s64 \t$dst, $src0, $src1", GPR64,
  GPR64, GPR64, int_rvgpu_mulhi_ll>;
def INT_NVVM_MULHI_ULL : F_MATH_2<"mul.hi.u64 \t$dst, $src0, $src1", GPR64,
  GPR64, GPR64, int_rvgpu_mulhi_ull>;

def INT_NVVM_MUL_RN_FTZ_F : F_MATH_2<"mul.rn.ftz.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_mul_rn_ftz_f>;
def INT_NVVM_MUL_RN_F : F_MATH_2<"mul.rn.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_mul_rn_f>;
def INT_NVVM_MUL_RZ_FTZ_F : F_MATH_2<"mul.rz.ftz.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_mul_rz_ftz_f>;
def INT_NVVM_MUL_RZ_F : F_MATH_2<"mul.rz.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_mul_rz_f>;
def INT_NVVM_MUL_RM_FTZ_F : F_MATH_2<"mul.rm.ftz.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_mul_rm_ftz_f>;
def INT_NVVM_MUL_RM_F : F_MATH_2<"mul.rm.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_mul_rm_f>;
def INT_NVVM_MUL_RP_FTZ_F : F_MATH_2<"mul.rp.ftz.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_mul_rp_ftz_f>;
def INT_NVVM_MUL_RP_F : F_MATH_2<"mul.rp.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_mul_rp_f>;

def INT_NVVM_MUL_RN_D : F_MATH_2<"mul.rn.f64 \t$dst, $src0, $src1",
  GPR64, GPR64, GPR64, int_rvgpu_mul_rn_d>;
def INT_NVVM_MUL_RZ_D : F_MATH_2<"mul.rz.f64 \t$dst, $src0, $src1",
  GPR64, GPR64, GPR64, int_rvgpu_mul_rz_d>;
def INT_NVVM_MUL_RM_D : F_MATH_2<"mul.rm.f64 \t$dst, $src0, $src1",
  GPR64, GPR64, GPR64, int_rvgpu_mul_rm_d>;
def INT_NVVM_MUL_RP_D : F_MATH_2<"mul.rp.f64 \t$dst, $src0, $src1",
  GPR64, GPR64, GPR64, int_rvgpu_mul_rp_d>;

def INT_NVVM_MUL24_I : F_MATH_2<"mul24.lo.s32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_mul24_i>;
def INT_NVVM_MUL24_UI : F_MATH_2<"mul24.lo.u32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_mul24_ui>;

//
// Div
//

def INT_NVVM_DIV_APPROX_FTZ_F
  : F_MATH_2<"div.approx.ftz.f32 \t$dst, $src0, $src1", GPR32,
    GPR32, GPR32, int_rvgpu_div_approx_ftz_f>;
def INT_NVVM_DIV_APPROX_F : F_MATH_2<"div.approx.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_div_approx_f>;

def INT_NVVM_DIV_RN_FTZ_F : F_MATH_2<"div.rn.ftz.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_div_rn_ftz_f>;
def INT_NVVM_DIV_RN_F     : F_MATH_2<"div.rn.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_div_rn_f>;
def INT_NVVM_DIV_RZ_FTZ_F : F_MATH_2<"div.rz.ftz.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_div_rz_ftz_f>;
def INT_NVVM_DIV_RZ_F     : F_MATH_2<"div.rz.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_div_rz_f>;
def INT_NVVM_DIV_RM_FTZ_F : F_MATH_2<"div.rm.ftz.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_div_rm_ftz_f>;
def INT_NVVM_DIV_RM_F     : F_MATH_2<"div.rm.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_div_rm_f>;
def INT_NVVM_DIV_RP_FTZ_F : F_MATH_2<"div.rp.ftz.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_div_rp_ftz_f>;
def INT_NVVM_DIV_RP_F     : F_MATH_2<"div.rp.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_div_rp_f>;

def INT_NVVM_DIV_RN_D : F_MATH_2<"div.rn.f64 \t$dst, $src0, $src1",
  GPR64, GPR64, GPR64, int_rvgpu_div_rn_d>;
def INT_NVVM_DIV_RZ_D : F_MATH_2<"div.rz.f64 \t$dst, $src0, $src1",
  GPR64, GPR64, GPR64, int_rvgpu_div_rz_d>;
def INT_NVVM_DIV_RM_D : F_MATH_2<"div.rm.f64 \t$dst, $src0, $src1",
  GPR64, GPR64, GPR64, int_rvgpu_div_rm_d>;
def INT_NVVM_DIV_RP_D : F_MATH_2<"div.rp.f64 \t$dst, $src0, $src1",
  GPR64, GPR64, GPR64, int_rvgpu_div_rp_d>;

//
// Sad
//

def INT_NVVM_SAD_I : F_MATH_3<"sad.s32 \t$dst, $src0, $src1, $src2",
  GPR32, GPR32, GPR32, GPR32, int_rvgpu_sad_i>;
def INT_NVVM_SAD_UI : F_MATH_3<"sad.u32 \t$dst, $src0, $src1, $src2",
  GPR32, GPR32, GPR32, GPR32, int_rvgpu_sad_ui>;

//
// Floor  Ceil
//

def : Pat<(int_rvgpu_floor_ftz_f GPR32:$a),
          (CVT_f32_f32 GPR32:$a, CvtRMI_FTZ)>;
def : Pat<(int_rvgpu_floor_f GPR32:$a),
          (CVT_f32_f32 GPR32:$a, CvtRMI)>;
def : Pat<(int_rvgpu_floor_d GPR64:$a),
          (CVT_f64_f64 GPR64:$a, CvtRMI)>;

def : Pat<(int_rvgpu_ceil_ftz_f GPR32:$a),
          (CVT_f32_f32 GPR32:$a, CvtRPI_FTZ)>;
def : Pat<(int_rvgpu_ceil_f GPR32:$a),
          (CVT_f32_f32 GPR32:$a, CvtRPI)>;
def : Pat<(int_rvgpu_ceil_d GPR64:$a),
          (CVT_f64_f64 GPR64:$a, CvtRPI)>;

//
// Abs
//

def INT_NVVM_FABS_FTZ_F : F_MATH_1<"abs.ftz.f32 \t$dst, $src0", GPR32,
  GPR32, int_rvgpu_fabs_ftz_f>;
def INT_NVVM_FABS_F : F_MATH_1<"abs.f32 \t$dst, $src0", GPR32,
  GPR32, int_rvgpu_fabs_f>;

def INT_NVVM_FABS_D : F_MATH_1<"abs.f64 \t$dst, $src0", GPR64,
  GPR64, int_rvgpu_fabs_d>;

//
// Abs, Neg bf16, bf16x2
//

def INT_NVVM_ABS_BF16 : F_MATH_1<"abs.bf16 \t$dst, $src0", GPR16,
  GPR16, int_rvgpu_abs_bf16, [hasPTX<70>, hasSM<80>]>;
def INT_NVVM_ABS_BF16X2 : F_MATH_1<"abs.bf16x2 \t$dst, $src0", GPR32,
  GPR32, int_rvgpu_abs_bf16x2, [hasPTX<70>, hasSM<80>]>;
def INT_NVVM_NEG_BF16 : F_MATH_1<"neg.bf16 \t$dst, $src0", GPR16,
  GPR16, int_rvgpu_neg_bf16, [hasPTX<70>, hasSM<80>]>;
def INT_NVVM_NEG_BF16X2 : F_MATH_1<"neg.bf16x2 \t$dst, $src0", GPR32,
  GPR32, int_rvgpu_neg_bf16x2, [hasPTX<70>, hasSM<80>]>;

//
// Round
//

def : Pat<(int_rvgpu_round_ftz_f GPR32:$a),
          (CVT_f32_f32 GPR32:$a, CvtRNI_FTZ)>;
def : Pat<(int_rvgpu_round_f GPR32:$a),
          (CVT_f32_f32 GPR32:$a, CvtRNI)>;
def : Pat<(int_rvgpu_round_d GPR64:$a),
          (CVT_f64_f64 GPR64:$a, CvtRNI)>;

//
// Trunc
//

def : Pat<(int_rvgpu_trunc_ftz_f GPR32:$a),
          (CVT_f32_f32 GPR32:$a, CvtRZI_FTZ)>;
def : Pat<(int_rvgpu_trunc_f GPR32:$a),
          (CVT_f32_f32 GPR32:$a, CvtRZI)>;
def : Pat<(int_rvgpu_trunc_d GPR64:$a),
          (CVT_f64_f64 GPR64:$a, CvtRZI)>;

//
// Saturate
//

def : Pat<(int_rvgpu_saturate_ftz_f GPR32:$a),
          (CVT_f32_f32 GPR32:$a, CvtSAT_FTZ)>;
def : Pat<(int_rvgpu_saturate_f GPR32:$a),
          (CVT_f32_f32 GPR32:$a, CvtSAT)>;
def : Pat<(int_rvgpu_saturate_d GPR64:$a),
          (CVT_f64_f64 GPR64:$a, CvtSAT)>;

//
// Exp2  Log2
//

def INT_NVVM_EX2_APPROX_FTZ_F : F_MATH_1<"ex2.approx.ftz.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_ex2_approx_ftz_f>;
def INT_NVVM_EX2_APPROX_F : F_MATH_1<"ex2.approx.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_ex2_approx_f>;
def INT_NVVM_EX2_APPROX_D : F_MATH_1<"ex2.approx.f64 \t$dst, $src0",
  GPR64, GPR64, int_rvgpu_ex2_approx_d>;
def INT_NVVM_EX2_APPROX_F16 : F_MATH_1<"ex2.approx.f16 \t$dst, $src0",
  GPR16, GPR16, int_rvgpu_ex2_approx_f16, [hasPTX<70>, hasSM<75>]>;
def INT_NVVM_EX2_APPROX_F16X2 : F_MATH_1<"ex2.approx.f16x2 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_ex2_approx_f16x2, [hasPTX<70>, hasSM<75>]>;

def INT_NVVM_LG2_APPROX_FTZ_F : F_MATH_1<"lg2.approx.ftz.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_lg2_approx_ftz_f>;
def INT_NVVM_LG2_APPROX_F : F_MATH_1<"lg2.approx.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_lg2_approx_f>;
def INT_NVVM_LG2_APPROX_D : F_MATH_1<"lg2.approx.f64 \t$dst, $src0",
  GPR64, GPR64, int_rvgpu_lg2_approx_d>;

//
// Sin  Cos
//

def INT_NVVM_SIN_APPROX_FTZ_F : F_MATH_1<"sin.approx.ftz.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_sin_approx_ftz_f>;
def INT_NVVM_SIN_APPROX_F : F_MATH_1<"sin.approx.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_sin_approx_f>;

def INT_NVVM_COS_APPROX_FTZ_F : F_MATH_1<"cos.approx.ftz.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_cos_approx_ftz_f>;
def INT_NVVM_COS_APPROX_F : F_MATH_1<"cos.approx.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_cos_approx_f>;

//
// Fma
//

class FMA_TUPLE<string V, Intrinsic I, RVGPURegClass RC,
                list<Predicate> Preds = []> {
  string Variant = V;
  Intrinsic Intr = I;
  RVGPURegClass RegClass = RC;
  list<Predicate> Predicates = Preds;
}

multiclass FMA_INST {
  foreach P = [
    FMA_TUPLE<"_rn_f64", int_rvgpu_fma_rn_d, GPR64>,
    FMA_TUPLE<"_rz_f64", int_rvgpu_fma_rz_d, GPR64>,
    FMA_TUPLE<"_rm_f64", int_rvgpu_fma_rm_d, GPR64>,
    FMA_TUPLE<"_rp_f64", int_rvgpu_fma_rp_d, GPR64>,

    FMA_TUPLE<"_rn_ftz_f32", int_rvgpu_fma_rn_ftz_f, GPR32>,
    FMA_TUPLE<"_rn_f32", int_rvgpu_fma_rn_f, GPR32>,
    FMA_TUPLE<"_rz_ftz_f32", int_rvgpu_fma_rz_ftz_f, GPR32>,
    FMA_TUPLE<"_rz_f32", int_rvgpu_fma_rz_f, GPR32>,
    FMA_TUPLE<"_rm_f32", int_rvgpu_fma_rm_f, GPR32>,
    FMA_TUPLE<"_rm_ftz_f32", int_rvgpu_fma_rm_ftz_f, GPR32>,
    FMA_TUPLE<"_rp_f32", int_rvgpu_fma_rp_f, GPR32>,
    FMA_TUPLE<"_rp_ftz_f32", int_rvgpu_fma_rp_ftz_f, GPR32>,

    FMA_TUPLE<"_rn_f16", int_rvgpu_fma_rn_f16, GPR16, [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_ftz_f16", int_rvgpu_fma_rn_ftz_f16, GPR16,
      [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_sat_f16", int_rvgpu_fma_rn_sat_f16, GPR16,
      [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_ftz_sat_f16", int_rvgpu_fma_rn_ftz_sat_f16, GPR16,
      [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_relu_f16", int_rvgpu_fma_rn_relu_f16, GPR16,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_ftz_relu_f16", int_rvgpu_fma_rn_ftz_relu_f16, GPR16,
      [hasPTX<70>, hasSM<80>]>,

    FMA_TUPLE<"_rn_bf16", int_rvgpu_fma_rn_bf16, GPR16, [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_ftz_bf16", int_rvgpu_fma_rn_ftz_bf16, GPR16,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_sat_bf16", int_rvgpu_fma_rn_sat_bf16, GPR16,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_ftz_sat_bf16", int_rvgpu_fma_rn_ftz_sat_bf16, GPR16,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_relu_bf16", int_rvgpu_fma_rn_relu_bf16, GPR16,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_ftz_relu_bf16", int_rvgpu_fma_rn_ftz_relu_bf16, GPR16,
      [hasPTX<70>, hasSM<80>]>,

    FMA_TUPLE<"_rn_f16x2", int_rvgpu_fma_rn_f16x2, GPR32,
      [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_ftz_f16x2", int_rvgpu_fma_rn_ftz_f16x2, GPR32,
      [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_sat_f16x2", int_rvgpu_fma_rn_sat_f16x2, GPR32,
      [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_ftz_sat_f16x2", int_rvgpu_fma_rn_ftz_sat_f16x2,
      GPR32, [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_relu_f16x2", int_rvgpu_fma_rn_relu_f16x2, GPR32,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_ftz_relu_f16x2", int_rvgpu_fma_rn_ftz_relu_f16x2,
      GPR32, [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_bf16x2", int_rvgpu_fma_rn_bf16x2, GPR32,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_relu_bf16x2", int_rvgpu_fma_rn_relu_bf16x2, GPR32,
      [hasPTX<70>, hasSM<80>]>
  ] in {
    def P.Variant :
      F_MATH_3<!strconcat("fma",
        !subst("_", ".", P.Variant), " \t$dst, $src0, $src1, $src2"),
        P.RegClass, P.RegClass, P.RegClass, P.RegClass, P.Intr, P.Predicates>;
  }
}

defm INT_NVVM_FMA : FMA_INST;

//
// Rcp
//

def INT_NVVM_RCP_RN_FTZ_F : F_MATH_1<"rcp.rn.ftz.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_rcp_rn_ftz_f>;
def INT_NVVM_RCP_RN_F : F_MATH_1<"rcp.rn.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_rcp_rn_f>;
def INT_NVVM_RCP_RZ_FTZ_F : F_MATH_1<"rcp.rz.ftz.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_rcp_rz_ftz_f>;
def INT_NVVM_RCP_RZ_F : F_MATH_1<"rcp.rz.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_rcp_rz_f>;
def INT_NVVM_RCP_RM_FTZ_F : F_MATH_1<"rcp.rm.ftz.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_rcp_rm_ftz_f>;
def INT_NVVM_RCP_RM_F : F_MATH_1<"rcp.rm.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_rcp_rm_f>;
def INT_NVVM_RCP_RP_FTZ_F : F_MATH_1<"rcp.rp.ftz.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_rcp_rp_ftz_f>;
def INT_NVVM_RCP_RP_F : F_MATH_1<"rcp.rp.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_rcp_rp_f>;

def INT_NVVM_RCP_RN_D : F_MATH_1<"rcp.rn.f64 \t$dst, $src0", GPR64,
  GPR64, int_rvgpu_rcp_rn_d>;
def INT_NVVM_RCP_RZ_D : F_MATH_1<"rcp.rz.f64 \t$dst, $src0", GPR64,
  GPR64, int_rvgpu_rcp_rz_d>;
def INT_NVVM_RCP_RM_D : F_MATH_1<"rcp.rm.f64 \t$dst, $src0", GPR64,
  GPR64, int_rvgpu_rcp_rm_d>;
def INT_NVVM_RCP_RP_D : F_MATH_1<"rcp.rp.f64 \t$dst, $src0", GPR64,
  GPR64, int_rvgpu_rcp_rp_d>;

def INT_NVVM_RCP_APPROX_FTZ_F : F_MATH_1<"rcp.approx.ftz.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_rcp_approx_ftz_f>;
def INT_NVVM_RCP_APPROX_FTZ_D : F_MATH_1<"rcp.approx.ftz.f64 \t$dst, $src0",
  GPR64, GPR64, int_rvgpu_rcp_approx_ftz_d>;

//
// Sqrt
//

def INT_NVVM_SQRT_RN_FTZ_F : F_MATH_1<"sqrt.rn.ftz.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_sqrt_rn_ftz_f>;
def INT_NVVM_SQRT_RN_F : F_MATH_1<"sqrt.rn.f32 \t$dst, $src0", GPR32,
  GPR32, int_rvgpu_sqrt_rn_f>;
def INT_NVVM_SQRT_RZ_FTZ_F : F_MATH_1<"sqrt.rz.ftz.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_sqrt_rz_ftz_f>;
def INT_NVVM_SQRT_RZ_F : F_MATH_1<"sqrt.rz.f32 \t$dst, $src0", GPR32,
  GPR32, int_rvgpu_sqrt_rz_f>;
def INT_NVVM_SQRT_RM_FTZ_F : F_MATH_1<"sqrt.rm.ftz.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_sqrt_rm_ftz_f>;
def INT_NVVM_SQRT_RM_F : F_MATH_1<"sqrt.rm.f32 \t$dst, $src0", GPR32,
  GPR32, int_rvgpu_sqrt_rm_f>;
def INT_NVVM_SQRT_RP_FTZ_F : F_MATH_1<"sqrt.rp.ftz.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_sqrt_rp_ftz_f>;
def INT_NVVM_SQRT_RP_F : F_MATH_1<"sqrt.rp.f32 \t$dst, $src0", GPR32,
  GPR32, int_rvgpu_sqrt_rp_f>;
def INT_NVVM_SQRT_APPROX_FTZ_F : F_MATH_1<"sqrt.approx.ftz.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_sqrt_approx_ftz_f>;
def INT_NVVM_SQRT_APPROX_F : F_MATH_1<"sqrt.approx.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_sqrt_approx_f>;

def INT_NVVM_SQRT_RN_D : F_MATH_1<"sqrt.rn.f64 \t$dst, $src0", GPR64,
  GPR64, int_rvgpu_sqrt_rn_d>;
def INT_NVVM_SQRT_RZ_D : F_MATH_1<"sqrt.rz.f64 \t$dst, $src0", GPR64,
  GPR64, int_rvgpu_sqrt_rz_d>;
def INT_NVVM_SQRT_RM_D : F_MATH_1<"sqrt.rm.f64 \t$dst, $src0", GPR64,
  GPR64, int_rvgpu_sqrt_rm_d>;
def INT_NVVM_SQRT_RP_D : F_MATH_1<"sqrt.rp.f64 \t$dst, $src0", GPR64,
  GPR64, int_rvgpu_sqrt_rp_d>;

// nvvm_sqrt intrinsic
def : Pat<(int_rvgpu_sqrt_f GPR32:$a),
          (INT_NVVM_SQRT_RN_FTZ_F GPR32:$a)>, Requires<[doF32FTZ, do_SQRTF32_RN]>;
def : Pat<(int_rvgpu_sqrt_f GPR32:$a),
          (INT_NVVM_SQRT_RN_F GPR32:$a)>, Requires<[do_SQRTF32_RN]>;
def : Pat<(int_rvgpu_sqrt_f GPR32:$a),
          (INT_NVVM_SQRT_APPROX_FTZ_F GPR32:$a)>, Requires<[doF32FTZ]>;
def : Pat<(int_rvgpu_sqrt_f GPR32:$a),
          (INT_NVVM_SQRT_APPROX_F GPR32:$a)>;

//
// Rsqrt
//

def INT_NVVM_RSQRT_APPROX_FTZ_F
  : F_MATH_1<"rsqrt.approx.ftz.f32 \t$dst, $src0", GPR32, GPR32,
    int_rvgpu_rsqrt_approx_ftz_f>;
def INT_NVVM_RSQRT_APPROX_F : F_MATH_1<"rsqrt.approx.f32 \t$dst, $src0",
  GPR32, GPR32, int_rvgpu_rsqrt_approx_f>;
def INT_NVVM_RSQRT_APPROX_D : F_MATH_1<"rsqrt.approx.f64 \t$dst, $src0",
  GPR64, GPR64, int_rvgpu_rsqrt_approx_d>;

//
// Add
//

def INT_NVVM_ADD_RN_FTZ_F : F_MATH_2<"add.rn.ftz.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_add_rn_ftz_f>;
def INT_NVVM_ADD_RN_F : F_MATH_2<"add.rn.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_add_rn_f>;
def INT_NVVM_ADD_RZ_FTZ_F : F_MATH_2<"add.rz.ftz.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_add_rz_ftz_f>;
def INT_NVVM_ADD_RZ_F : F_MATH_2<"add.rz.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_add_rz_f>;
def INT_NVVM_ADD_RM_FTZ_F : F_MATH_2<"add.rm.ftz.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_add_rm_ftz_f>;
def INT_NVVM_ADD_RM_F : F_MATH_2<"add.rm.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_add_rm_f>;
def INT_NVVM_ADD_RP_FTZ_F : F_MATH_2<"add.rp.ftz.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_add_rp_ftz_f>;
def INT_NVVM_ADD_RP_F : F_MATH_2<"add.rp.f32 \t$dst, $src0, $src1",
  GPR32, GPR32, GPR32, int_rvgpu_add_rp_f>;

def INT_NVVM_ADD_RN_D : F_MATH_2<"add.rn.f64 \t$dst, $src0, $src1",
  GPR64, GPR64, GPR64, int_rvgpu_add_rn_d>;
def INT_NVVM_ADD_RZ_D : F_MATH_2<"add.rz.f64 \t$dst, $src0, $src1",
  GPR64, GPR64, GPR64, int_rvgpu_add_rz_d>;
def INT_NVVM_ADD_RM_D : F_MATH_2<"add.rm.f64 \t$dst, $src0, $src1",
  GPR64, GPR64, GPR64, int_rvgpu_add_rm_d>;
def INT_NVVM_ADD_RP_D : F_MATH_2<"add.rp.f64 \t$dst, $src0, $src1",
  GPR64, GPR64, GPR64, int_rvgpu_add_rp_d>;

//
// Convert
//

def : Pat<(int_rvgpu_d2f_rn_ftz GPR64:$a),
          (CVT_f32_f64 GPR64:$a, CvtRN_FTZ)>;
def : Pat<(int_rvgpu_d2f_rn GPR64:$a),
          (CVT_f32_f64 GPR64:$a, CvtRN)>;
def : Pat<(int_rvgpu_d2f_rz_ftz GPR64:$a),
          (CVT_f32_f64 GPR64:$a, CvtRZ_FTZ)>;
def : Pat<(int_rvgpu_d2f_rz GPR64:$a),
          (CVT_f32_f64 GPR64:$a, CvtRZ)>;
def : Pat<(int_rvgpu_d2f_rm_ftz GPR64:$a),
          (CVT_f32_f64 GPR64:$a, CvtRM_FTZ)>;
def : Pat<(int_rvgpu_d2f_rm GPR64:$a),
          (CVT_f32_f64 GPR64:$a, CvtRM)>;
def : Pat<(int_rvgpu_d2f_rp_ftz GPR64:$a),
          (CVT_f32_f64 GPR64:$a, CvtRP_FTZ)>;
def : Pat<(int_rvgpu_d2f_rp GPR64:$a),
          (CVT_f32_f64 GPR64:$a, CvtRP)>;

def : Pat<(int_rvgpu_d2i_rn GPR64:$a),
          (CVT_s32_f64 GPR64:$a, CvtRNI)>;
def : Pat<(int_rvgpu_d2i_rz GPR64:$a),
          (CVT_s32_f64 GPR64:$a, CvtRZI)>;
def : Pat<(int_rvgpu_d2i_rm GPR64:$a),
          (CVT_s32_f64 GPR64:$a, CvtRMI)>;
def : Pat<(int_rvgpu_d2i_rp GPR64:$a),
          (CVT_s32_f64 GPR64:$a, CvtRPI)>;

def : Pat<(int_rvgpu_d2ui_rn GPR64:$a),
          (CVT_u32_f64 GPR64:$a, CvtRNI)>;
def : Pat<(int_rvgpu_d2ui_rz GPR64:$a),
          (CVT_u32_f64 GPR64:$a, CvtRZI)>;
def : Pat<(int_rvgpu_d2ui_rm GPR64:$a),
          (CVT_u32_f64 GPR64:$a, CvtRMI)>;
def : Pat<(int_rvgpu_d2ui_rp GPR64:$a),
          (CVT_u32_f64 GPR64:$a, CvtRPI)>;

def : Pat<(int_rvgpu_i2d_rn GPR32:$a),
          (CVT_f64_s32 GPR32:$a, CvtRN)>;
def : Pat<(int_rvgpu_i2d_rz GPR32:$a),
          (CVT_f64_s32 GPR32:$a, CvtRZ)>;
def : Pat<(int_rvgpu_i2d_rm GPR32:$a),
          (CVT_f64_s32 GPR32:$a, CvtRM)>;
def : Pat<(int_rvgpu_i2d_rp GPR32:$a),
          (CVT_f64_s32 GPR32:$a, CvtRP)>;

def : Pat<(int_rvgpu_ui2d_rn GPR32:$a),
          (CVT_f64_u32 GPR32:$a, CvtRN)>;
def : Pat<(int_rvgpu_ui2d_rz GPR32:$a),
          (CVT_f64_u32 GPR32:$a, CvtRZ)>;
def : Pat<(int_rvgpu_ui2d_rm GPR32:$a),
          (CVT_f64_u32 GPR32:$a, CvtRM)>;
def : Pat<(int_rvgpu_ui2d_rp GPR32:$a),
          (CVT_f64_u32 GPR32:$a, CvtRP)>;

def : Pat<(int_rvgpu_f2i_rn_ftz GPR32:$a),
          (CVT_s32_f32 GPR32:$a, CvtRNI_FTZ)>;
def : Pat<(int_rvgpu_f2i_rn GPR32:$a),
          (CVT_s32_f32 GPR32:$a, CvtRNI)>;
def : Pat<(int_rvgpu_f2i_rz_ftz GPR32:$a),
          (CVT_s32_f32 GPR32:$a, CvtRZI_FTZ)>;
def : Pat<(int_rvgpu_f2i_rz GPR32:$a),
          (CVT_s32_f32 GPR32:$a, CvtRZI)>;
def : Pat<(int_rvgpu_f2i_rm_ftz GPR32:$a),
          (CVT_s32_f32 GPR32:$a, CvtRMI_FTZ)>;
def : Pat<(int_rvgpu_f2i_rm GPR32:$a),
          (CVT_s32_f32 GPR32:$a, CvtRMI)>;
def : Pat<(int_rvgpu_f2i_rp_ftz GPR32:$a),
          (CVT_s32_f32 GPR32:$a, CvtRPI_FTZ)>;
def : Pat<(int_rvgpu_f2i_rp GPR32:$a),
          (CVT_s32_f32 GPR32:$a, CvtRPI)>;

def : Pat<(int_rvgpu_f2ui_rn_ftz GPR32:$a),
          (CVT_u32_f32 GPR32:$a, CvtRNI_FTZ)>;
def : Pat<(int_rvgpu_f2ui_rn GPR32:$a),
          (CVT_u32_f32 GPR32:$a, CvtRNI)>;
def : Pat<(int_rvgpu_f2ui_rz_ftz GPR32:$a),
          (CVT_u32_f32 GPR32:$a, CvtRZI_FTZ)>;
def : Pat<(int_rvgpu_f2ui_rz GPR32:$a),
          (CVT_u32_f32 GPR32:$a, CvtRZI)>;
def : Pat<(int_rvgpu_f2ui_rm_ftz GPR32:$a),
          (CVT_u32_f32 GPR32:$a, CvtRMI_FTZ)>;
def : Pat<(int_rvgpu_f2ui_rm GPR32:$a),
          (CVT_u32_f32 GPR32:$a, CvtRMI)>;
def : Pat<(int_rvgpu_f2ui_rp_ftz GPR32:$a),
          (CVT_u32_f32 GPR32:$a, CvtRPI_FTZ)>;
def : Pat<(int_rvgpu_f2ui_rp GPR32:$a),
          (CVT_u32_f32 GPR32:$a, CvtRPI)>;

def : Pat<(int_rvgpu_i2f_rn GPR32:$a),
          (CVT_f32_s32 GPR32:$a, CvtRN)>;
def : Pat<(int_rvgpu_i2f_rz GPR32:$a),
          (CVT_f32_s32 GPR32:$a, CvtRZ)>;
def : Pat<(int_rvgpu_i2f_rm GPR32:$a),
          (CVT_f32_s32 GPR32:$a, CvtRM)>;
def : Pat<(int_rvgpu_i2f_rp GPR32:$a),
          (CVT_f32_s32 GPR32:$a, CvtRP)>;

def : Pat<(int_rvgpu_ui2f_rn GPR32:$a),
          (CVT_f32_u32 GPR32:$a, CvtRN)>;
def : Pat<(int_rvgpu_ui2f_rz GPR32:$a),
          (CVT_f32_u32 GPR32:$a, CvtRZ)>;
def : Pat<(int_rvgpu_ui2f_rm GPR32:$a),
          (CVT_f32_u32 GPR32:$a, CvtRM)>;
def : Pat<(int_rvgpu_ui2f_rp GPR32:$a),
          (CVT_f32_u32 GPR32:$a, CvtRP)>;

def : Pat<(int_rvgpu_ff2bf16x2_rn GPR32:$a, GPR32:$b),
          (CVT_bf16x2_f32 GPR32:$a, GPR32:$b, CvtRN)>;
def : Pat<(int_rvgpu_ff2bf16x2_rn_relu GPR32:$a, GPR32:$b),
          (CVT_bf16x2_f32 GPR32:$a, GPR32:$b, CvtRN_RELU)>;
def : Pat<(int_rvgpu_ff2bf16x2_rz GPR32:$a, GPR32:$b),
          (CVT_bf16x2_f32 GPR32:$a, GPR32:$b, CvtRZ)>;
def : Pat<(int_rvgpu_ff2bf16x2_rz_relu GPR32:$a, GPR32:$b),
          (CVT_bf16x2_f32 GPR32:$a, GPR32:$b, CvtRZ_RELU)>;

def : Pat<(int_rvgpu_ff2f16x2_rn GPR32:$a, GPR32:$b),
          (CVT_f16x2_f32 GPR32:$a, GPR32:$b, CvtRN)>;
def : Pat<(int_rvgpu_ff2f16x2_rn_relu GPR32:$a, GPR32:$b),
          (CVT_f16x2_f32 GPR32:$a, GPR32:$b, CvtRN_RELU)>;
def : Pat<(int_rvgpu_ff2f16x2_rz GPR32:$a, GPR32:$b),
          (CVT_f16x2_f32 GPR32:$a, GPR32:$b, CvtRZ)>;
def : Pat<(int_rvgpu_ff2f16x2_rz_relu GPR32:$a, GPR32:$b),
          (CVT_f16x2_f32 GPR32:$a, GPR32:$b, CvtRZ_RELU)>;

def : Pat<(int_rvgpu_f2bf16_rn GPR32:$a),
          (CVT_bf16_f32 GPR32:$a, CvtRN)>;
def : Pat<(int_rvgpu_f2bf16_rn_relu GPR32:$a),
          (CVT_bf16_f32 GPR32:$a, CvtRN_RELU)>;
def : Pat<(int_rvgpu_f2bf16_rz GPR32:$a),
          (CVT_bf16_f32 GPR32:$a, CvtRZ)>;
def : Pat<(int_rvgpu_f2bf16_rz_relu GPR32:$a),
          (CVT_bf16_f32 GPR32:$a, CvtRZ_RELU)>;

def CVT_tf32_f32 :
   RVGPUInst<(outs GPR32:$dest), (ins GPR32:$a),
                   "cvt.rna.tf32.f32 \t$dest, $a",
       [(set GPR32:$dest, (int_rvgpu_f2tf32_rna GPR32:$a))]>;


let isCodeGenOnly = 1 in {
def INT_NVVM_LOHI_I2D : F_MATH_2<"mov.b64 \t$dst, {{$src0, $src1}}",
  GPR64, GPR32, GPR32, int_rvgpu_lohi_i2d>;

def INT_NVVM_D2I_LO : F_MATH_1<
  !strconcat("{{\n\t",
             ".reg .b32 %temp; \n\t",
             "mov.b64 \t{$dst, %temp}, $src0;\n\t",
             "}}"),
  GPR32, GPR64, int_rvgpu_d2i_lo>;
def INT_NVVM_D2I_HI : F_MATH_1<
  !strconcat("{{\n\t",
             ".reg .b32 %temp; \n\t",
             "mov.b64 \t{%temp, $dst}, $src0;\n\t",
             "}}"),
  GPR32, GPR64, int_rvgpu_d2i_hi>;
}

def : Pat<(int_rvgpu_f2ll_rn_ftz GPR32:$a),
          (CVT_s64_f32 GPR32:$a, CvtRNI_FTZ)>;
def : Pat<(int_rvgpu_f2ll_rn GPR32:$a),
          (CVT_s64_f32 GPR32:$a, CvtRNI)>;
def : Pat<(int_rvgpu_f2ll_rz_ftz GPR32:$a),
          (CVT_s64_f32 GPR32:$a, CvtRZI_FTZ)>;
def : Pat<(int_rvgpu_f2ll_rz GPR32:$a),
          (CVT_s64_f32 GPR32:$a, CvtRZI)>;
def : Pat<(int_rvgpu_f2ll_rm_ftz GPR32:$a),
          (CVT_s64_f32 GPR32:$a, CvtRMI_FTZ)>;
def : Pat<(int_rvgpu_f2ll_rm GPR32:$a),
          (CVT_s64_f32 GPR32:$a, CvtRMI)>;
def : Pat<(int_rvgpu_f2ll_rp_ftz GPR32:$a),
          (CVT_s64_f32 GPR32:$a, CvtRPI_FTZ)>;
def : Pat<(int_rvgpu_f2ll_rp GPR32:$a),
          (CVT_s64_f32 GPR32:$a, CvtRPI)>;

def : Pat<(int_rvgpu_f2ull_rn_ftz GPR32:$a),
          (CVT_u64_f32 GPR32:$a, CvtRNI_FTZ)>;
def : Pat<(int_rvgpu_f2ull_rn GPR32:$a),
          (CVT_u64_f32 GPR32:$a, CvtRNI)>;
def : Pat<(int_rvgpu_f2ull_rz_ftz GPR32:$a),
          (CVT_u64_f32 GPR32:$a, CvtRZI_FTZ)>;
def : Pat<(int_rvgpu_f2ull_rz GPR32:$a),
          (CVT_u64_f32 GPR32:$a, CvtRZI)>;
def : Pat<(int_rvgpu_f2ull_rm_ftz GPR32:$a),
          (CVT_u64_f32 GPR32:$a, CvtRMI_FTZ)>;
def : Pat<(int_rvgpu_f2ull_rm GPR32:$a),
          (CVT_u64_f32 GPR32:$a, CvtRMI)>;
def : Pat<(int_rvgpu_f2ull_rp_ftz GPR32:$a),
          (CVT_u64_f32 GPR32:$a, CvtRPI_FTZ)>;
def : Pat<(int_rvgpu_f2ull_rp GPR32:$a),
          (CVT_u64_f32 GPR32:$a, CvtRPI)>;

def : Pat<(int_rvgpu_d2ll_rn GPR64:$a),
          (CVT_s64_f64 GPR64:$a, CvtRNI)>;
def : Pat<(int_rvgpu_d2ll_rz GPR64:$a),
          (CVT_s64_f64 GPR64:$a, CvtRZI)>;
def : Pat<(int_rvgpu_d2ll_rm GPR64:$a),
          (CVT_s64_f64 GPR64:$a, CvtRMI)>;
def : Pat<(int_rvgpu_d2ll_rp GPR64:$a),
          (CVT_s64_f64 GPR64:$a, CvtRPI)>;

def : Pat<(int_rvgpu_d2ull_rn GPR64:$a),
          (CVT_u64_f64 GPR64:$a, CvtRNI)>;
def : Pat<(int_rvgpu_d2ull_rz GPR64:$a),
          (CVT_u64_f64 GPR64:$a, CvtRZI)>;
def : Pat<(int_rvgpu_d2ull_rm GPR64:$a),
          (CVT_u64_f64 GPR64:$a, CvtRMI)>;
def : Pat<(int_rvgpu_d2ull_rp GPR64:$a),
          (CVT_u64_f64 GPR64:$a, CvtRPI)>;

def : Pat<(int_rvgpu_ll2f_rn GPR64:$a),
          (CVT_f32_s64 GPR64:$a, CvtRN)>;
def : Pat<(int_rvgpu_ll2f_rz GPR64:$a),
          (CVT_f32_s64 GPR64:$a, CvtRZ)>;
def : Pat<(int_rvgpu_ll2f_rm GPR64:$a),
          (CVT_f32_s64 GPR64:$a, CvtRM)>;
def : Pat<(int_rvgpu_ll2f_rp GPR64:$a),
          (CVT_f32_s64 GPR64:$a, CvtRP)>;

def : Pat<(int_rvgpu_ull2f_rn GPR64:$a),
          (CVT_f32_u64 GPR64:$a, CvtRN)>;
def : Pat<(int_rvgpu_ull2f_rz GPR64:$a),
          (CVT_f32_u64 GPR64:$a, CvtRZ)>;
def : Pat<(int_rvgpu_ull2f_rm GPR64:$a),
          (CVT_f32_u64 GPR64:$a, CvtRM)>;
def : Pat<(int_rvgpu_ull2f_rp GPR64:$a),
          (CVT_f32_u64 GPR64:$a, CvtRP)>;

def : Pat<(int_rvgpu_ll2d_rn GPR64:$a),
          (CVT_f64_s64 GPR64:$a, CvtRN)>;
def : Pat<(int_rvgpu_ll2d_rz GPR64:$a),
          (CVT_f64_s64 GPR64:$a, CvtRZ)>;
def : Pat<(int_rvgpu_ll2d_rm GPR64:$a),
          (CVT_f64_s64 GPR64:$a, CvtRM)>;
def : Pat<(int_rvgpu_ll2d_rp GPR64:$a),
          (CVT_f64_s64 GPR64:$a, CvtRP)>;

def : Pat<(int_rvgpu_ull2d_rn GPR64:$a),
          (CVT_f64_u64 GPR64:$a, CvtRN)>;
def : Pat<(int_rvgpu_ull2d_rz GPR64:$a),
          (CVT_f64_u64 GPR64:$a, CvtRZ)>;
def : Pat<(int_rvgpu_ull2d_rm GPR64:$a),
          (CVT_f64_u64 GPR64:$a, CvtRM)>;
def : Pat<(int_rvgpu_ull2d_rp GPR64:$a),
          (CVT_f64_u64 GPR64:$a, CvtRP)>;


def : Pat<(int_rvgpu_f2h_rn_ftz GPR32:$a),
          (CVT_f16_f32 GPR32:$a, CvtRN_FTZ)>;
def : Pat<(int_rvgpu_f2h_rn GPR32:$a),
          (CVT_f16_f32 GPR32:$a, CvtRN)>;

//
// Bitcast
//

def INT_NVVM_BITCAST_F2I : F_MATH_1<"mov.b32 \t$dst, $src0", GPR32,
  GPR32, int_rvgpu_bitcast_f2i>;
def INT_NVVM_BITCAST_I2F : F_MATH_1<"mov.b32 \t$dst, $src0", GPR32,
  GPR32, int_rvgpu_bitcast_i2f>;

def INT_NVVM_BITCAST_LL2D : F_MATH_1<"mov.b64 \t$dst, $src0", GPR64,
  GPR64, int_rvgpu_bitcast_ll2d>;
def INT_NVVM_BITCAST_D2LL : F_MATH_1<"mov.b64 \t$dst, $src0", GPR64,
  GPR64, int_rvgpu_bitcast_d2ll>;

//
// FNS
//

class INT_FNS_MBO<dag ins, dag Operands>
  : RVGPUInst<(outs GPR32:$dst), ins,
               "fns.b32 \t$dst, $mask, $base, $offset",
               [(set GPR32:$dst, Operands )]>,
    Requires<[hasPTX<60>, hasSM<30>]>;

def INT_FNS_rrr : INT_FNS_MBO<(ins GPR32:$mask, GPR32:$base, GPR32:$offset),
                     (int_rvgpu_fns GPR32:$mask, GPR32:$base, GPR32:$offset)>;
def INT_FNS_rri : INT_FNS_MBO<(ins GPR32:$mask, GPR32:$base,    i32imm:$offset),
                     (int_rvgpu_fns GPR32:$mask, GPR32:$base,       imm:$offset)>;
def INT_FNS_rir : INT_FNS_MBO<(ins GPR32:$mask,    i32imm:$base, GPR32:$offset),
                     (int_rvgpu_fns GPR32:$mask,       imm:$base, GPR32:$offset)>;
def INT_FNS_rii : INT_FNS_MBO<(ins GPR32:$mask,    i32imm:$base,    i32imm:$offset),
                     (int_rvgpu_fns GPR32:$mask,       imm:$base,       imm:$offset)>;
def INT_FNS_irr : INT_FNS_MBO<(ins    i32imm:$mask, GPR32:$base, GPR32:$offset),
                     (int_rvgpu_fns       imm:$mask, GPR32:$base, GPR32:$offset)>;
def INT_FNS_iri : INT_FNS_MBO<(ins    i32imm:$mask, GPR32:$base,    i32imm:$offset),
                     (int_rvgpu_fns       imm:$mask, GPR32:$base,       imm:$offset)>;
def INT_FNS_iir : INT_FNS_MBO<(ins    i32imm:$mask,    i32imm:$base, GPR32:$offset),
                     (int_rvgpu_fns       imm:$mask,       imm:$base, GPR32:$offset)>;
def INT_FNS_iii : INT_FNS_MBO<(ins    i32imm:$mask,    i32imm:$base,    i32imm:$offset),
                     (int_rvgpu_fns       imm:$mask,       imm:$base,       imm:$offset)>;

//-----------------------------------
// Atomic Functions
//-----------------------------------

class ATOMIC_GLOBAL_CHK <dag ops, dag frag>
 : PatFrag<ops, frag, AS_match.global>;
class ATOMIC_SHARED_CHK <dag ops, dag frag>
 : PatFrag<ops, frag, AS_match.shared>;
class ATOMIC_GENERIC_CHK <dag ops, dag frag>
 : PatFrag<ops, frag, AS_match.generic>;

multiclass F_ATOMIC_2_imp<ValueType ptrT, RVGPURegClass ptrclass,
  ValueType regT, RVGPURegClass regclass,
  string SpaceStr, string TypeStr, string OpcStr, PatFrag IntOp,
  Operand IMMType, SDNode IMM, list<Predicate> Pred> {
  def reg : RVGPUInst<(outs regclass:$dst), (ins ptrclass:$addr, regclass:$b),
    !strconcat("atom", SpaceStr, OpcStr, TypeStr, " \t$dst, [$addr], $b"),
    [(set (regT regclass:$dst), (IntOp (ptrT ptrclass:$addr), (regT regclass:$b)))]>,
  Requires<Pred>;
  def imm : RVGPUInst<(outs regclass:$dst), (ins ptrclass:$addr, IMMType:$b),
    !strconcat("atom", SpaceStr, OpcStr, TypeStr, " \t$dst, [$addr], $b", ""),
    [(set (regT regclass:$dst), (IntOp (ptrT ptrclass:$addr), IMM:$b))]>,
  Requires<Pred>;
}
multiclass F_ATOMIC_2<ValueType regT, RVGPURegClass regclass, string SpaceStr, string TypeStr,
  string OpcStr, PatFrag IntOp, Operand IMMType, SDNode IMM,
  list<Predicate> Pred = []> {
  defm p32 : F_ATOMIC_2_imp<i32, GPR32, regT, regclass, SpaceStr, TypeStr, OpcStr,
    IntOp, IMMType, IMM, Pred>;
  defm p64 : F_ATOMIC_2_imp<i64, GPR64, regT, regclass, SpaceStr, TypeStr, OpcStr,
    IntOp, IMMType, IMM, Pred>;
}

// has 2 operands, neg the second one
multiclass F_ATOMIC_2_NEG_imp<ValueType ptrT, RVGPURegClass ptrclass,
  ValueType regT, RVGPURegClass regclass,
  string SpaceStr, string TypeStr, string OpcStr, PatFrag IntOp,
  list<Predicate> Pred> {
  def reg : RVGPUInst<(outs regclass:$dst), (ins ptrclass:$addr, regclass:$b),
    !strconcat(
      "{{ \n\t",
      ".reg \t.s", TypeStr, " temp; \n\t",
      "neg.s", TypeStr, " \ttemp, $b; \n\t",
      "atom", SpaceStr, OpcStr, ".u", TypeStr, " \t$dst, [$addr], temp; \n\t",
      "}}"),
    [(set (regT regclass:$dst), (IntOp (ptrT ptrclass:$addr), (regT regclass:$b)))]>,
  Requires<Pred>;
}
multiclass F_ATOMIC_2_NEG<ValueType regT, RVGPURegClass regclass, string SpaceStr,
  string TypeStr, string OpcStr, PatFrag IntOp, list<Predicate> Pred = []> {
 defm p32: F_ATOMIC_2_NEG_imp<i32, GPR32, regT, regclass, SpaceStr, TypeStr, OpcStr,
   IntOp, Pred> ;
 defm p64: F_ATOMIC_2_NEG_imp<i64, GPR64, regT, regclass, SpaceStr, TypeStr, OpcStr,
   IntOp, Pred> ;
}

// has 3 operands
multiclass F_ATOMIC_3_imp<ValueType ptrT, RVGPURegClass ptrclass,
  ValueType regT, RVGPURegClass regclass,
  string SpaceStr, string TypeStr, string OpcStr, PatFrag IntOp,
  Operand IMMType, list<Predicate> Pred> {
  def reg : RVGPUInst<(outs regclass:$dst),
    (ins ptrclass:$addr, regclass:$b, regclass:$c),
    !strconcat("atom", SpaceStr, OpcStr, TypeStr, " \t$dst, [$addr], $b, $c"),
    [(set (regT regclass:$dst), (IntOp (ptrT ptrclass:$addr), (regT regclass:$b), (regT regclass:$c)))]>,
  Requires<Pred>;

  def imm1 : RVGPUInst<(outs regclass:$dst),
    (ins ptrclass:$addr, IMMType:$b, regclass:$c),
    !strconcat("atom", SpaceStr, OpcStr, TypeStr, " \t$dst, [$addr], $b, $c"),
    [(set (regT regclass:$dst), (IntOp (ptrT ptrclass:$addr), imm:$b, (regT regclass:$c)))]>,
  Requires<Pred>;

  def imm2 : RVGPUInst<(outs regclass:$dst),
    (ins ptrclass:$addr, regclass:$b, IMMType:$c),
    !strconcat("atom", SpaceStr, OpcStr, TypeStr, " \t$dst, [$addr], $b, $c", ""),
    [(set (regT regclass:$dst), (IntOp (ptrT ptrclass:$addr), (regT regclass:$b), imm:$c))]>,
  Requires<Pred>;

  def imm3 : RVGPUInst<(outs regclass:$dst),
    (ins ptrclass:$addr, IMMType:$b, IMMType:$c),
    !strconcat("atom", SpaceStr, OpcStr, TypeStr, " \t$dst, [$addr], $b, $c"),
    [(set (regT regclass:$dst), (IntOp (ptrT ptrclass:$addr), imm:$b, imm:$c))]>,
  Requires<Pred>;
}
multiclass F_ATOMIC_3<ValueType regT, RVGPURegClass regclass, string SpaceStr, string TypeStr,
  string OpcStr, PatFrag IntOp, Operand IMMType, list<Predicate> Pred = []> {
  defm p32 : F_ATOMIC_3_imp<i32, GPR32, regT, regclass, SpaceStr, TypeStr, OpcStr,
    IntOp, IMMType, Pred>;
  defm p64 : F_ATOMIC_3_imp<i64, GPR64, regT, regclass, SpaceStr, TypeStr, OpcStr,
    IntOp, IMMType, Pred>;
}

// atom_add

def atomic_load_add_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_add_32 node:$a, node:$b)>;
def atomic_load_add_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_add_32 node:$a, node:$b)>;
def atomic_load_add_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_add_32 node:$a, node:$b)>;
def atomic_load_add_64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_add_64 node:$a, node:$b)>;
def atomic_load_add_64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_add_64 node:$a, node:$b)>;
def atomic_load_add_64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_add_64 node:$a, node:$b)>;
def atomic_load_add_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_fadd node:$a, node:$b)>;
def atomic_load_add_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_fadd node:$a, node:$b)>;
def atomic_load_add_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_fadd node:$a, node:$b)>;

defm INT_PTX_ATOM_ADD_G_32 : F_ATOMIC_2<i32, GPR32, ".global", ".u32", ".add",
  atomic_load_add_32_g, i32imm, imm>;
defm INT_PTX_ATOM_ADD_S_32 : F_ATOMIC_2<i32, GPR32, ".shared", ".u32", ".add",
  atomic_load_add_32_s, i32imm, imm>;
defm INT_PTX_ATOM_ADD_GEN_32 : F_ATOMIC_2<i32, GPR32, "", ".u32", ".add",
  atomic_load_add_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_ADD_GEN_32_USE_G : F_ATOMIC_2<i32, GPR32, ".global", ".u32",
  ".add", atomic_load_add_32_gen, i32imm, imm>;

defm INT_PTX_ATOM_ADD_G_64 : F_ATOMIC_2<i64, GPR64, ".global", ".u64", ".add",
  atomic_load_add_64_g, i64imm, imm>;
defm INT_PTX_ATOM_ADD_S_64 : F_ATOMIC_2<i64, GPR64, ".shared", ".u64", ".add",
  atomic_load_add_64_s, i64imm, imm>;
defm INT_PTX_ATOM_ADD_GEN_64 : F_ATOMIC_2<i64, GPR64, "", ".u64", ".add",
  atomic_load_add_64_gen, i64imm, imm>;
defm INT_PTX_ATOM_ADD_GEN_64_USE_G : F_ATOMIC_2<i64, GPR64, ".global", ".u64",
  ".add", atomic_load_add_64_gen, i64imm, imm>;

defm INT_PTX_ATOM_ADD_G_F32 : F_ATOMIC_2<f32, GPR32, ".global", ".f32", ".add",
  atomic_load_add_g, f32imm, fpimm>;
defm INT_PTX_ATOM_ADD_S_F32 : F_ATOMIC_2<f32, GPR32, ".shared", ".f32", ".add",
  atomic_load_add_s, f32imm, fpimm>;
defm INT_PTX_ATOM_ADD_GEN_F32 : F_ATOMIC_2<f32, GPR32, "", ".f32", ".add",
  atomic_load_add_gen, f32imm, fpimm>;

defm INT_PTX_ATOM_ADD_G_F64 : F_ATOMIC_2<f64, GPR64, ".global", ".f64", ".add",
  atomic_load_add_g, f64imm, fpimm, [hasAtomAddF64]>;
defm INT_PTX_ATOM_ADD_S_F64 : F_ATOMIC_2<f64, GPR64, ".shared", ".f64", ".add",
  atomic_load_add_s, f64imm, fpimm, [hasAtomAddF64]>;
defm INT_PTX_ATOM_ADD_GEN_F64 : F_ATOMIC_2<f64, GPR64, "", ".f64", ".add",
  atomic_load_add_gen, f64imm, fpimm, [hasAtomAddF64]>;

// atom_sub

def atomic_load_sub_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_sub_32 node:$a, node:$b)>;
def atomic_load_sub_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_sub_32 node:$a, node:$b)>;
def atomic_load_sub_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_sub_32 node:$a, node:$b)>;
def atomic_load_sub_64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_sub_64 node:$a, node:$b)>;
def atomic_load_sub_64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_sub_64 node:$a, node:$b)>;
def atomic_load_sub_64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_sub_64 node:$a, node:$b)>;

let isCodeGenOnly = 1 in {
defm INT_PTX_ATOM_SUB_G_32 : F_ATOMIC_2_NEG<i32, GPR32, ".global", "32", ".add",
  atomic_load_sub_32_g>;
defm INT_PTX_ATOM_SUB_G_64 : F_ATOMIC_2_NEG<i64, GPR64, ".global", "64", ".add",
  atomic_load_sub_64_g>;
defm INT_PTX_ATOM_SUB_GEN_32 : F_ATOMIC_2_NEG<i32, GPR32, "", "32", ".add",
  atomic_load_sub_32_gen>;
defm INT_PTX_ATOM_SUB_GEN_32_USE_G : F_ATOMIC_2_NEG<i32, GPR32, ".global", "32",
  ".add", atomic_load_sub_32_gen>;
defm INT_PTX_ATOM_SUB_S_32 : F_ATOMIC_2_NEG<i32, GPR32, ".shared", "32", ".add",
  atomic_load_sub_32_s>;
defm INT_PTX_ATOM_SUB_S_64 : F_ATOMIC_2_NEG<i64, GPR64, ".shared", "64", ".add",
  atomic_load_sub_64_s>;
defm INT_PTX_ATOM_SUB_GEN_64 : F_ATOMIC_2_NEG<i64, GPR64, "", "64", ".add",
  atomic_load_sub_64_gen>;
defm INT_PTX_ATOM_SUB_GEN_64_USE_G : F_ATOMIC_2_NEG<i64, GPR64, ".global", "64",
  ".add", atomic_load_sub_64_gen>;
}
// atom_swap

def atomic_swap_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_swap_32 node:$a, node:$b)>;
def atomic_swap_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_swap_32 node:$a, node:$b)>;
def atomic_swap_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_swap_32 node:$a, node:$b)>;
def atomic_swap_64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_swap_64 node:$a, node:$b)>;
def atomic_swap_64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_swap_64 node:$a, node:$b)>;
def atomic_swap_64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_swap_64 node:$a, node:$b)>;

defm INT_PTX_ATOM_SWAP_G_32 : F_ATOMIC_2<i32, GPR32, ".global", ".b32", ".exch",
  atomic_swap_32_g, i32imm, imm>;
defm INT_PTX_ATOM_SWAP_S_32 : F_ATOMIC_2<i32, GPR32, ".shared", ".b32", ".exch",
  atomic_swap_32_s, i32imm, imm>;
defm INT_PTX_ATOM_SWAP_GEN_32 : F_ATOMIC_2<i32, GPR32, "", ".b32", ".exch",
  atomic_swap_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_SWAP_GEN_32_USE_G : F_ATOMIC_2<i32, GPR32, ".global", ".b32",
  ".exch", atomic_swap_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_SWAP_G_64 : F_ATOMIC_2<i64, GPR64, ".global", ".b64", ".exch",
  atomic_swap_64_g, i64imm, imm>;
defm INT_PTX_ATOM_SWAP_S_64 : F_ATOMIC_2<i64, GPR64, ".shared", ".b64", ".exch",
  atomic_swap_64_s, i64imm, imm>;
defm INT_PTX_ATOM_SWAP_GEN_64 : F_ATOMIC_2<i64, GPR64, "", ".b64", ".exch",
  atomic_swap_64_gen, i64imm, imm>;
defm INT_PTX_ATOM_SWAP_GEN_64_USE_G : F_ATOMIC_2<i64, GPR64, ".global", ".b64",
  ".exch", atomic_swap_64_gen, i64imm, imm>;

// atom_max

def atomic_load_max_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b)
  , (atomic_load_max_32 node:$a, node:$b)>;
def atomic_load_max_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_max_32 node:$a, node:$b)>;
def atomic_load_max_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_max_32 node:$a, node:$b)>;
def atomic_load_max_64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b)
  , (atomic_load_max_64 node:$a, node:$b)>;
def atomic_load_max_64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_max_64 node:$a, node:$b)>;
def atomic_load_max_64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_max_64 node:$a, node:$b)>;
def atomic_load_umax_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_umax_32 node:$a, node:$b)>;
def atomic_load_umax_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_umax_32 node:$a, node:$b)>;
def atomic_load_umax_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_umax_32 node:$a, node:$b)>;
def atomic_load_umax_64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_umax_64 node:$a, node:$b)>;
def atomic_load_umax_64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_umax_64 node:$a, node:$b)>;
def atomic_load_umax_64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_umax_64 node:$a, node:$b)>;

defm INT_PTX_ATOM_LOAD_MAX_G_32 : F_ATOMIC_2<i32, GPR32, ".global", ".s32",
  ".max", atomic_load_max_32_g, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_MAX_S_32 : F_ATOMIC_2<i32, GPR32, ".shared", ".s32",
  ".max", atomic_load_max_32_s, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_MAX_GEN_32 : F_ATOMIC_2<i32, GPR32, "", ".s32", ".max",
  atomic_load_max_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_MAX_GEN_32_USE_G : F_ATOMIC_2<i32, GPR32, ".global",
  ".s32", ".max", atomic_load_max_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_MAX_G_64 : F_ATOMIC_2<i64, GPR64, ".global", ".s64",
  ".max", atomic_load_max_64_g, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_MAX_S_64 : F_ATOMIC_2<i64, GPR64, ".shared", ".s64",
  ".max", atomic_load_max_64_s, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_MAX_GEN_64 : F_ATOMIC_2<i64, GPR64, "", ".s64", ".max",
  atomic_load_max_64_gen, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_MAX_GEN_64_USE_G : F_ATOMIC_2<i64, GPR64, ".global",
  ".s64", ".max", atomic_load_max_64_gen, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_UMAX_G_32 : F_ATOMIC_2<i32, GPR32, ".global", ".u32",
  ".max", atomic_load_umax_32_g, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_UMAX_S_32 : F_ATOMIC_2<i32, GPR32, ".shared", ".u32",
  ".max", atomic_load_umax_32_s, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_UMAX_GEN_32 : F_ATOMIC_2<i32, GPR32, "", ".u32", ".max",
  atomic_load_umax_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_UMAX_GEN_32_USE_G : F_ATOMIC_2<i32, GPR32, ".global",
  ".u32", ".max", atomic_load_umax_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_UMAX_G_64 : F_ATOMIC_2<i64, GPR64, ".global", ".u64",
  ".max", atomic_load_umax_64_g, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_UMAX_S_64 : F_ATOMIC_2<i64, GPR64, ".shared", ".u64",
  ".max", atomic_load_umax_64_s, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_UMAX_GEN_64 : F_ATOMIC_2<i64, GPR64, "", ".u64", ".max",
  atomic_load_umax_64_gen, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_UMAX_GEN_64_USE_G : F_ATOMIC_2<i64, GPR64, ".global",
  ".u64", ".max", atomic_load_umax_64_gen, i64imm, imm, [hasSM<32>]>;

// atom_min

def atomic_load_min_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_min_32 node:$a, node:$b)>;
def atomic_load_min_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_min_32 node:$a, node:$b)>;
def atomic_load_min_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_min_32 node:$a, node:$b)>;
def atomic_load_min_64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_min_64 node:$a, node:$b)>;
def atomic_load_min_64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_min_64 node:$a, node:$b)>;
def atomic_load_min_64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_min_64 node:$a, node:$b)>;
def atomic_load_umin_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_umin_32 node:$a, node:$b)>;
def atomic_load_umin_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_umin_32 node:$a, node:$b)>;
def atomic_load_umin_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_umin_32 node:$a, node:$b)>;
def atomic_load_umin_64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_umin_64 node:$a, node:$b)>;
def atomic_load_umin_64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_umin_64 node:$a, node:$b)>;
def atomic_load_umin_64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_umin_64 node:$a, node:$b)>;

defm INT_PTX_ATOM_LOAD_MIN_G_32 : F_ATOMIC_2<i32, GPR32, ".global", ".s32",
  ".min", atomic_load_min_32_g, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_MIN_S_32 : F_ATOMIC_2<i32, GPR32, ".shared", ".s32",
  ".min", atomic_load_min_32_s, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_MIN_GEN_32 : F_ATOMIC_2<i32, GPR32, "", ".s32", ".min",
  atomic_load_min_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_MIN_GEN_32_USE_G : F_ATOMIC_2<i32, GPR32, ".global",
  ".s32", ".min", atomic_load_min_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_MIN_G_64 : F_ATOMIC_2<i64, GPR64, ".global", ".s64",
  ".min", atomic_load_min_64_g, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_MIN_S_64 : F_ATOMIC_2<i64, GPR64, ".shared", ".s64",
  ".min", atomic_load_min_64_s, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_MIN_GEN_64 : F_ATOMIC_2<i64, GPR64, "", ".s64", ".min",
  atomic_load_min_64_gen, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_MIN_GEN_64_USE_G : F_ATOMIC_2<i64, GPR64, ".global",
  ".s64", ".min", atomic_load_min_64_gen, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_UMIN_G_32 : F_ATOMIC_2<i32, GPR32, ".global", ".u32",
  ".min", atomic_load_umin_32_g, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_UMIN_S_32 : F_ATOMIC_2<i32, GPR32, ".shared", ".u32",
  ".min", atomic_load_umin_32_s, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_UMIN_GEN_32 : F_ATOMIC_2<i32, GPR32, "", ".u32", ".min",
  atomic_load_umin_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_UMIN_GEN_32_USE_G : F_ATOMIC_2<i32, GPR32, ".global",
  ".u32", ".min", atomic_load_umin_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_UMIN_G_64 : F_ATOMIC_2<i64, GPR64, ".global", ".u64",
  ".min", atomic_load_umin_64_g, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_UMIN_S_64 : F_ATOMIC_2<i64, GPR64, ".shared", ".u64",
  ".min", atomic_load_umin_64_s, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_UMIN_GEN_64 : F_ATOMIC_2<i64, GPR64, "", ".u64", ".min",
  atomic_load_umin_64_gen, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_UMIN_GEN_64_USE_G : F_ATOMIC_2<i64, GPR64, ".global",
  ".u64", ".min", atomic_load_umin_64_gen, i64imm, imm, [hasSM<32>]>;

// atom_inc  atom_dec

def atomic_load_inc_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (int_rvgpu_atomic_load_inc_32 node:$a, node:$b)>;
def atomic_load_inc_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (int_rvgpu_atomic_load_inc_32 node:$a, node:$b)>;
def atomic_load_inc_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (int_rvgpu_atomic_load_inc_32 node:$a, node:$b)>;
def atomic_load_dec_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (int_rvgpu_atomic_load_dec_32 node:$a, node:$b)>;
def atomic_load_dec_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (int_rvgpu_atomic_load_dec_32 node:$a, node:$b)>;
def atomic_load_dec_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (int_rvgpu_atomic_load_dec_32 node:$a, node:$b)>;

defm INT_PTX_ATOM_INC_G_32 : F_ATOMIC_2<i32, GPR32, ".global", ".u32", ".inc",
  atomic_load_inc_32_g, i32imm, imm>;
defm INT_PTX_ATOM_INC_S_32 : F_ATOMIC_2<i32, GPR32, ".shared", ".u32", ".inc",
  atomic_load_inc_32_s, i32imm, imm>;
defm INT_PTX_ATOM_INC_GEN_32 : F_ATOMIC_2<i32, GPR32, "", ".u32", ".inc",
  atomic_load_inc_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_INC_GEN_32_USE_G : F_ATOMIC_2<i32, GPR32, ".global", ".u32",
  ".inc", atomic_load_inc_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_DEC_G_32 : F_ATOMIC_2<i32, GPR32, ".global", ".u32", ".dec",
  atomic_load_dec_32_g, i32imm, imm>;
defm INT_PTX_ATOM_DEC_S_32 : F_ATOMIC_2<i32, GPR32, ".shared", ".u32", ".dec",
  atomic_load_dec_32_s, i32imm, imm>;
defm INT_PTX_ATOM_DEC_GEN_32 : F_ATOMIC_2<i32, GPR32, "", ".u32", ".dec",
  atomic_load_dec_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_DEC_GEN_32_USE_G : F_ATOMIC_2<i32, GPR32, ".global", ".u32",
  ".dec", atomic_load_dec_32_gen, i32imm, imm>;

// atom_and

def atomic_load_and_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_and_32 node:$a, node:$b)>;
def atomic_load_and_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_and_32 node:$a, node:$b)>;
def atomic_load_and_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_and_32 node:$a, node:$b)>;
def atomic_load_and_64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_and_64 node:$a, node:$b)>;
def atomic_load_and_64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_and_64 node:$a, node:$b)>;
def atomic_load_and_64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_and_64 node:$a, node:$b)>;

defm INT_PTX_ATOM_AND_G_32 : F_ATOMIC_2<i32, GPR32, ".global", ".b32", ".and",
  atomic_load_and_32_g, i32imm, imm>;
defm INT_PTX_ATOM_AND_S_32 : F_ATOMIC_2<i32, GPR32, ".shared", ".b32", ".and",
  atomic_load_and_32_s, i32imm, imm>;
defm INT_PTX_ATOM_AND_GEN_32 : F_ATOMIC_2<i32, GPR32, "", ".b32", ".and",
  atomic_load_and_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_AND_GEN_32_USE_G : F_ATOMIC_2<i32, GPR32, ".global", ".b32",
  ".and", atomic_load_and_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_AND_G_64 : F_ATOMIC_2<i64, GPR64, ".global", ".b64", ".and",
  atomic_load_and_64_g, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_AND_S_64 : F_ATOMIC_2<i64, GPR64, ".shared", ".b64", ".and",
  atomic_load_and_64_s, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_AND_GEN_64 : F_ATOMIC_2<i64, GPR64, "", ".b64", ".and",
  atomic_load_and_64_gen, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_AND_GEN_64_USE_G : F_ATOMIC_2<i64, GPR64, ".global", ".b64",
  ".and", atomic_load_and_64_gen, i64imm, imm, [hasSM<32>]>;

// atom_or

def atomic_load_or_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_or_32 node:$a, node:$b)>;
def atomic_load_or_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_or_32 node:$a, node:$b)>;
def atomic_load_or_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_or_32 node:$a, node:$b)>;
def atomic_load_or_64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_or_64 node:$a, node:$b)>;
def atomic_load_or_64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_or_64 node:$a, node:$b)>;
def atomic_load_or_64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_or_64 node:$a, node:$b)>;

defm INT_PTX_ATOM_OR_G_32 : F_ATOMIC_2<i32, GPR32, ".global", ".b32", ".or",
  atomic_load_or_32_g, i32imm, imm>;
defm INT_PTX_ATOM_OR_GEN_32 : F_ATOMIC_2<i32, GPR32, "", ".b32", ".or",
  atomic_load_or_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_OR_GEN_32_USE_G : F_ATOMIC_2<i32, GPR32, ".global", ".b32",
  ".or", atomic_load_or_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_OR_S_32 : F_ATOMIC_2<i32, GPR32, ".shared", ".b32", ".or",
  atomic_load_or_32_s, i32imm, imm>;
defm INT_PTX_ATOM_OR_G_64 : F_ATOMIC_2<i64, GPR64, ".global", ".b64", ".or",
  atomic_load_or_64_g, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_OR_GEN_64 : F_ATOMIC_2<i64, GPR64, "", ".b64", ".or",
  atomic_load_or_64_gen, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_OR_GEN_64_USE_G : F_ATOMIC_2<i64, GPR64, ".global", ".b64",
  ".or", atomic_load_or_64_gen, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_OR_S_64 : F_ATOMIC_2<i64, GPR64, ".shared", ".b64", ".or",
  atomic_load_or_64_s, i64imm, imm, [hasSM<32>]>;

// atom_xor

def atomic_load_xor_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_xor_32 node:$a, node:$b)>;
def atomic_load_xor_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_xor_32 node:$a, node:$b)>;
def atomic_load_xor_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_xor_32 node:$a, node:$b)>;
def atomic_load_xor_64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_xor_64 node:$a, node:$b)>;
def atomic_load_xor_64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_xor_64 node:$a, node:$b)>;
def atomic_load_xor_64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_xor_64 node:$a, node:$b)>;

defm INT_PTX_ATOM_XOR_G_32 : F_ATOMIC_2<i32, GPR32, ".global", ".b32", ".xor",
  atomic_load_xor_32_g, i32imm, imm>;
defm INT_PTX_ATOM_XOR_S_32 : F_ATOMIC_2<i32, GPR32, ".shared", ".b32", ".xor",
  atomic_load_xor_32_s, i32imm, imm>;
defm INT_PTX_ATOM_XOR_GEN_32 : F_ATOMIC_2<i32, GPR32, "", ".b32", ".xor",
  atomic_load_xor_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_XOR_GEN_32_USE_G : F_ATOMIC_2<i32, GPR32, ".global", ".b32",
  ".xor", atomic_load_xor_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_XOR_G_64 : F_ATOMIC_2<i64, GPR64, ".global", ".b64", ".xor",
  atomic_load_xor_64_g, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_XOR_S_64 : F_ATOMIC_2<i64, GPR64, ".shared", ".b64", ".xor",
  atomic_load_xor_64_s, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_XOR_GEN_64 : F_ATOMIC_2<i64, GPR64, "", ".b64", ".xor",
  atomic_load_xor_64_gen, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_XOR_GEN_64_USE_G : F_ATOMIC_2<i64, GPR64, ".global", ".b64",
  ".xor", atomic_load_xor_64_gen, i64imm, imm, [hasSM<32>]>;

// atom_cas

def atomic_cmp_swap_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_32 node:$a, node:$b, node:$c)>;
def atomic_cmp_swap_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_32 node:$a, node:$b, node:$c)>;
def atomic_cmp_swap_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_32 node:$a, node:$b, node:$c)>;
def atomic_cmp_swap_64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_64 node:$a, node:$b, node:$c)>;
def atomic_cmp_swap_64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_64 node:$a, node:$b, node:$c)>;
def atomic_cmp_swap_64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_64 node:$a, node:$b, node:$c)>;

defm INT_PTX_ATOM_CAS_G_32 : F_ATOMIC_3<i32, GPR32, ".global", ".b32", ".cas",
  atomic_cmp_swap_32_g, i32imm>;
defm INT_PTX_ATOM_CAS_S_32 : F_ATOMIC_3<i32, GPR32, ".shared", ".b32", ".cas",
  atomic_cmp_swap_32_s, i32imm>;
defm INT_PTX_ATOM_CAS_GEN_32 : F_ATOMIC_3<i32, GPR32, "", ".b32", ".cas",
  atomic_cmp_swap_32_gen, i32imm>;
defm INT_PTX_ATOM_CAS_GEN_32_USE_G : F_ATOMIC_3<i32, GPR32, ".global", ".b32",
  ".cas", atomic_cmp_swap_32_gen, i32imm>;
defm INT_PTX_ATOM_CAS_G_64 : F_ATOMIC_3<i64, GPR64, ".global", ".b64", ".cas",
  atomic_cmp_swap_64_g, i64imm>;
defm INT_PTX_ATOM_CAS_S_64 : F_ATOMIC_3<i64, GPR64, ".shared", ".b64", ".cas",
  atomic_cmp_swap_64_s, i64imm>;
defm INT_PTX_ATOM_CAS_GEN_64 : F_ATOMIC_3<i64, GPR64, "", ".b64", ".cas",
  atomic_cmp_swap_64_gen, i64imm>;
defm INT_PTX_ATOM_CAS_GEN_64_USE_G : F_ATOMIC_3<i64, GPR64, ".global", ".b64",
  ".cas", atomic_cmp_swap_64_gen, i64imm>;

// Support for scoped atomic operations.  Matches
// int_rvgpu_atomic_{op}_{space}_{type}_{scope}
// and converts it into the appropriate instruction.
// NOTE: not all possible combinations are implemented
//  'space' is limited to generic as it's the only one needed to support CUDA.
//  'scope' = 'gpu' is default and is handled by regular atomic instructions.
class ATOM23_impl<string AsmStr, ValueType regT, RVGPURegClass regclass, list<Predicate> Preds,
                  dag ins, dag Operands>
      : RVGPUInst<(outs regclass:$result), ins,
                  AsmStr,
                  [(set (regT regclass:$result), Operands)]>,
        Requires<Preds>;

// Define instruction variants for all addressing modes.
multiclass ATOM2P_impl<string AsmStr,  Intrinsic Intr,
                       ValueType regT, RVGPURegClass regclass, Operand ImmType,
                       SDNode Imm, ValueType ImmTy,
                       list<Predicate> Preds> {
  let AddedComplexity = 1 in {
    def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                      (ins GPR32:$src, regclass:$b),
                      (Intr (i32 GPR32:$src), (regT regclass:$b))>;
    def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                      (ins GPR64:$src, regclass:$b),
                      (Intr (i64 GPR64:$src), (regT regclass:$b))>;
  }
  // tablegen can't infer argument types from Intrinsic (though it can
  // from Instruction) so we have to enforce specific type on
  // immediates via explicit cast to ImmTy.
  def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                    (ins GPR32:$src, ImmType:$b),
                    (Intr (i32 GPR32:$src), (ImmTy Imm:$b))>;
  def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                    (ins GPR64:$src, ImmType:$b),
                    (Intr (i64 GPR64:$src), (ImmTy Imm:$b))>;
}

multiclass ATOM3P_impl<string AsmStr,  Intrinsic Intr,
                       ValueType regT, RVGPURegClass regclass,
                       Operand ImmType, SDNode Imm, ValueType ImmTy,
                       list<Predicate> Preds> {
  // Variants for register/immediate permutations of $b and $c
  let AddedComplexity = 2 in {
    def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                      (ins GPR32:$src, regclass:$b, regclass:$c),
                      (Intr (i32 GPR32:$src), (regT regclass:$b), (regT regclass:$c))>;
    def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                      (ins GPR64:$src, regclass:$b, regclass:$c),
                      (Intr (i64 GPR64:$src), (regT regclass:$b), (regT regclass:$c))>;
  }
  let AddedComplexity = 1 in {
    def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                      (ins GPR32:$src, ImmType:$b, regclass:$c),
                      (Intr (i32 GPR32:$src), (ImmTy Imm:$b), (regT regclass:$c))>;
    def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                      (ins GPR64:$src, ImmType:$b, regclass:$c),
                      (Intr (i64 GPR64:$src), (ImmTy Imm:$b), (regT regclass:$c))>;
    def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                      (ins GPR32:$src, regclass:$b, ImmType:$c),
                      (Intr (i32 GPR32:$src), (regT regclass:$b), (ImmTy Imm:$c))>;
    def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                      (ins GPR64:$src, regclass:$b, ImmType:$c),
                      (Intr (i64 GPR64:$src), (regT regclass:$b), (ImmTy Imm:$c))>;
  }
  def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                    (ins GPR32:$src, ImmType:$b, ImmType:$c),
                    (Intr (i32 GPR32:$src), (ImmTy Imm:$b), (ImmTy Imm:$c))>;
  def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                    (ins GPR64:$src, ImmType:$b, ImmType:$c),
                    (Intr (i64 GPR64:$src), (ImmTy Imm:$b), (ImmTy Imm:$c))>;
}

// Constructs intrinsic name and instruction asm strings.
multiclass ATOM2N_impl<string OpStr, string IntTypeStr, string TypeStr,
                       string ScopeStr, string SpaceStr,
                       ValueType regT, RVGPURegClass regclass, Operand ImmType, SDNode Imm,
                       ValueType ImmTy, list<Predicate> Preds> {
  defm : ATOM2P_impl<"atom" # !if(!eq(SpaceStr, "gen"), "", "." # SpaceStr)
                            # !if(!eq(ScopeStr, "gpu"), "", "." # ScopeStr)
                            # "." # OpStr # "." # TypeStr
                            # " \t$result, [$src], $b",
                     !cast<Intrinsic>(
                            "int_rvgpu_atomic_" # OpStr
                            # "_" # SpaceStr # "_" # IntTypeStr
                            # !if(!empty(ScopeStr), "", "_" # ScopeStr)),
                     regT, regclass, ImmType, Imm, ImmTy, Preds>;
}
multiclass ATOM3N_impl<string OpStr, string IntTypeStr, string TypeStr,
                       string ScopeStr, string SpaceStr,
                       ValueType regT, RVGPURegClass regclass, Operand ImmType, SDNode Imm,
                       ValueType ImmTy, list<Predicate> Preds> {
  defm : ATOM3P_impl<"atom" # !if(!eq(SpaceStr, "gen"), "", "." # SpaceStr)
                            # !if(!eq(ScopeStr, "gpu"), "", "." # ScopeStr)
                            # "." # OpStr # "." # TypeStr
                            # " \t$result, [$src], $b, $c",
                     !cast<Intrinsic>(
                            "int_rvgpu_atomic_" # OpStr
                            # "_" # SpaceStr # "_" # IntTypeStr
                            # !if(!empty(ScopeStr), "", "_" # ScopeStr)),
                     regT, regclass, ImmType, Imm, ImmTy, Preds>;
}

// Constructs variants for different address spaces.
// For now we only need variants for generic space pointers.
multiclass ATOM2A_impl<string OpStr, string IntTypeStr, string TypeStr,
                       string ScopeStr, ValueType regT, RVGPURegClass regclass, Operand ImmType,
                       SDNode Imm, ValueType ImmTy, list<Predicate> Preds> {
   defm _gen_ : ATOM2N_impl<OpStr, IntTypeStr, TypeStr, ScopeStr, "gen",
                            regT, regclass, ImmType, Imm, ImmTy, Preds>;
}
multiclass ATOM3A_impl<string OpStr, string IntTypeStr, string TypeStr,
                       string ScopeStr, ValueType regT, RVGPURegClass regclass, Operand ImmType,
                       SDNode Imm, ValueType ImmTy, list<Predicate> Preds> {
   defm _gen_ : ATOM3N_impl<OpStr, IntTypeStr, TypeStr, ScopeStr, "gen",
                            regT, regclass, ImmType, Imm, ImmTy, Preds>;
}

// Constructs variants for different scopes of atomic op.
multiclass ATOM2S_impl<string OpStr, string IntTypeStr, string TypeStr,
                       ValueType regT, RVGPURegClass regclass, Operand ImmType, SDNode Imm,
                       ValueType ImmTy, list<Predicate> Preds> {
   // .gpu scope is default and is currently covered by existing
   // atomics w/o explicitly specified scope.
   defm _cta : ATOM2A_impl<OpStr, IntTypeStr, TypeStr, "cta",
                           regT, regclass, ImmType, Imm, ImmTy,
                           !listconcat(Preds,[hasAtomScope])>;
   defm _sys : ATOM2A_impl<OpStr, IntTypeStr, TypeStr, "sys",
                           regT, regclass, ImmType, Imm, ImmTy,
                           !listconcat(Preds,[hasAtomScope])>;
}
multiclass ATOM3S_impl<string OpStr, string IntTypeStr, string TypeStr,
           ValueType regT, RVGPURegClass regclass, Operand ImmType, SDNode Imm, ValueType ImmTy,
           list<Predicate> Preds> {
   // No need to define ".gpu"-scoped atomics.  They do the same thing
   // as the regular, non-scoped atomics defined elsewhere.
   defm _cta : ATOM3A_impl<OpStr, IntTypeStr, TypeStr, "cta",
                           regT, regclass, ImmType, Imm, ImmTy,
                           !listconcat(Preds,[hasAtomScope])>;
   defm _sys : ATOM3A_impl<OpStr, IntTypeStr, TypeStr, "sys",
                           regT, regclass, ImmType, Imm, ImmTy,
                           !listconcat(Preds,[hasAtomScope])>;
}

// atom.add
multiclass ATOM2_add_impl<string OpStr> {
   defm _s32  : ATOM2S_impl<OpStr, "i", "s32", i32, GPR32, i32imm, imm, i32, []>;
   defm _u32  : ATOM2S_impl<OpStr, "i", "u32", i32, GPR32, i32imm, imm, i32, []>;
   defm _u64  : ATOM2S_impl<OpStr, "i", "u64", i64, GPR64, i64imm, imm, i64, []>;
   defm _f32  : ATOM2S_impl<OpStr, "f", "f32", f32, GPR32, f32imm, fpimm, f32,
                            []>;
   defm _f64  : ATOM2S_impl<OpStr, "f", "f64", f64, GPR64, f64imm, fpimm, f64,
                            [hasAtomAddF64]>;
}

// atom.{and,or,xor}
multiclass ATOM2_bitwise_impl<string OpStr> {
   defm _b32  : ATOM2S_impl<OpStr, "i", "b32", i32, GPR32, i32imm, imm, i32, []>;
   defm _b64  : ATOM2S_impl<OpStr, "i", "b64", i64, GPR64, i64imm, imm, i64,
                            [hasAtomBitwise64]>;
}

// atom.exch
multiclass ATOM2_exch_impl<string OpStr> {
   defm _b32 : ATOM2S_impl<OpStr, "i", "b32", i32, GPR32, i32imm, imm, i32, []>;
   defm _b64 : ATOM2S_impl<OpStr, "i", "b64", i64, GPR64, i64imm, imm, i64, []>;
}

// atom.{min,max}
multiclass ATOM2_minmax_impl<string OpStr> {
   defm _s32  : ATOM2S_impl<OpStr, "i", "s32", i32, GPR32, i32imm, imm, i32, []>;
   defm _u32  : ATOM2S_impl<OpStr, "i", "u32", i32, GPR32, i32imm, imm, i32, []>;
   defm _s64  : ATOM2S_impl<OpStr, "i", "s64", i64, GPR64, i64imm, imm, i64,
                            [hasAtomMinMax64]>;
   defm _u64  : ATOM2S_impl<OpStr, "i", "u64", i64, GPR64, i64imm, imm, i64,
                            [hasAtomMinMax64]>;
}

// atom.{inc,dec}
multiclass ATOM2_incdec_impl<string OpStr> {
   defm _u32  : ATOM2S_impl<OpStr, "i", "u32", i32, GPR32, i32imm, imm, i32, []>;
}

// atom.cas
multiclass ATOM3_cas_impl<string OpStr> {
   defm _b32  : ATOM3S_impl<OpStr, "i", "b32", i32, GPR32, i32imm, imm, i32, []>;
   defm _b64  : ATOM3S_impl<OpStr, "i", "b64", i64, GPR64, i64imm, imm, i64, []>;
}

defm INT_PTX_SATOM_ADD : ATOM2_add_impl<"add">;
defm INT_PTX_SATOM_AND : ATOM2_bitwise_impl<"and">;
defm INT_PTX_SATOM_CAS : ATOM3_cas_impl<"cas">;
defm INT_PTX_SATOM_DEC : ATOM2_incdec_impl<"dec">;
defm INT_PTX_SATOM_EXCH: ATOM2_exch_impl<"exch">;
defm INT_PTX_SATOM_INC : ATOM2_incdec_impl<"inc">;
defm INT_PTX_SATOM_MAX : ATOM2_minmax_impl<"max">;
defm INT_PTX_SATOM_MIN : ATOM2_minmax_impl<"min">;
defm INT_PTX_SATOM_OR  : ATOM2_bitwise_impl<"or">;
defm INT_PTX_SATOM_XOR : ATOM2_bitwise_impl<"xor">;

//-----------------------------------
// Support for ldu on sm_20 or later
//-----------------------------------

// Don't annotate ldu instructions as mayLoad, as they load from memory that is
// read-only in a kernel.

// Scalar

let isCodeGenOnly = 1 in {
multiclass LDU_G<string TyStr, RVGPURegClass regclass> {
  def areg: RVGPUInst<(outs regclass:$result), (ins GPR32:$src),
               !strconcat("ldu.global.", TyStr),
                      []>, Requires<[hasLDU]>;
  def areg64: RVGPUInst<(outs regclass:$result), (ins GPR64:$src),
               !strconcat("ldu.global.", TyStr),
                        []>, Requires<[hasLDU]>;
 def avar:  RVGPUInst<(outs regclass:$result), (ins imemAny:$src),
               !strconcat("ldu.global.", TyStr),
                      []>, Requires<[hasLDU]>;
 def ari :  RVGPUInst<(outs regclass:$result), (ins MEMri:$src),
               !strconcat("ldu.global.", TyStr),
                      []>, Requires<[hasLDU]>;
 def ari64 :  RVGPUInst<(outs regclass:$result), (ins MEMri64:$src),
               !strconcat("ldu.global.", TyStr),
                        []>, Requires<[hasLDU]>;
}
}

defm INT_PTX_LDU_GLOBAL_i8  : LDU_G<"u8 \t$result, [$src]", GPR16>;
defm INT_PTX_LDU_GLOBAL_i16 : LDU_G<"u16 \t$result, [$src]", GPR16>;
defm INT_PTX_LDU_GLOBAL_i32 : LDU_G<"u32 \t$result, [$src]", GPR32>;
defm INT_PTX_LDU_GLOBAL_i64 : LDU_G<"u64 \t$result, [$src]", GPR64>;
defm INT_PTX_LDU_GLOBAL_f32 : LDU_G<"f32 \t$result, [$src]", GPR32>;
defm INT_PTX_LDU_GLOBAL_f64 : LDU_G<"f64 \t$result, [$src]", GPR64>;

// vector

// Elementized vector ldu
let isCodeGenOnly = 1 in {
multiclass VLDU_G_ELE_V2<string TyStr, RVGPURegClass regclass> {
 def _areg32: RVGPUInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins GPR32:$src),
                     !strconcat("ldu.global.", TyStr), []>;
 def _areg64: RVGPUInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins GPR64:$src),
                     !strconcat("ldu.global.", TyStr), []>;
 def _ari32: RVGPUInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins MEMri:$src),
                     !strconcat("ldu.global.", TyStr), []>;
 def _ari64: RVGPUInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins MEMri64:$src),
                     !strconcat("ldu.global.", TyStr), []>;
 def _avar: RVGPUInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins imemAny:$src),
                     !strconcat("ldu.global.", TyStr), []>;
}

multiclass VLDU_G_ELE_V4<string TyStr, RVGPURegClass regclass> {
 def _areg32: RVGPUInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                            regclass:$dst4), (ins GPR32:$src),
               !strconcat("ldu.global.", TyStr), []>;
 def _areg64: RVGPUInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                            regclass:$dst4), (ins GPR64:$src),
               !strconcat("ldu.global.", TyStr), []>;
 def _ari32: RVGPUInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                            regclass:$dst4), (ins MEMri:$src),
               !strconcat("ldu.global.", TyStr), []>;
 def _ari64: RVGPUInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                            regclass:$dst4), (ins MEMri64:$src),
               !strconcat("ldu.global.", TyStr), []>;
 def _avar: RVGPUInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                            regclass:$dst4), (ins imemAny:$src),
               !strconcat("ldu.global.", TyStr), []>;
}
}

defm INT_PTX_LDU_G_v2i8_ELE
  : VLDU_G_ELE_V2<"v2.u8 \t{{$dst1, $dst2}}, [$src]",  GPR16>;
defm INT_PTX_LDU_G_v2i16_ELE
  : VLDU_G_ELE_V2<"v2.u16 \t{{$dst1, $dst2}}, [$src]", GPR16>;
defm INT_PTX_LDU_G_v2i32_ELE
  : VLDU_G_ELE_V2<"v2.u32 \t{{$dst1, $dst2}}, [$src]", GPR32>;
defm INT_PTX_LDU_G_v2f32_ELE
  : VLDU_G_ELE_V2<"v2.f32 \t{{$dst1, $dst2}}, [$src]", GPR32>;
defm INT_PTX_LDU_G_v2i64_ELE
  : VLDU_G_ELE_V2<"v2.u64 \t{{$dst1, $dst2}}, [$src]", GPR64>;
defm INT_PTX_LDU_G_v2f64_ELE
  : VLDU_G_ELE_V2<"v2.f64 \t{{$dst1, $dst2}}, [$src]", GPR64>;
defm INT_PTX_LDU_G_v4i8_ELE
  : VLDU_G_ELE_V4<"v4.u8 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src]", GPR16>;
defm INT_PTX_LDU_G_v4i16_ELE
  : VLDU_G_ELE_V4<"v4.u16 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src]",
    GPR16>;
defm INT_PTX_LDU_G_v4i32_ELE
  : VLDU_G_ELE_V4<"v4.u32 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src]",
    GPR32>;
defm INT_PTX_LDU_G_v4f16_ELE
  : VLDU_G_ELE_V4<"v4.b16 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src]",
    GPR16>;
defm INT_PTX_LDU_G_v4f16x2_ELE
  : VLDU_G_ELE_V4<"v4.b32 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src]",
    GPR32>;
defm INT_PTX_LDU_G_v4f32_ELE
  : VLDU_G_ELE_V4<"v4.f32 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src]",
    GPR32>;


//-----------------------------------
// Support for ldg on sm_35 or later
//-----------------------------------

// Don't annotate ld.global.nc as mayLoad, because these loads go through the
// non-coherent texture cache, and therefore the values read must be read-only
// during the lifetime of the kernel.

let isCodeGenOnly = 1 in {
multiclass LDG_G<string TyStr, RVGPURegClass regclass> {
  def areg: RVGPUInst<(outs regclass:$result), (ins GPR32:$src),
               !strconcat("ld.global.nc.", TyStr),
                      []>, Requires<[hasLDG]>;
  def areg64: RVGPUInst<(outs regclass:$result), (ins GPR64:$src),
               !strconcat("ld.global.nc.", TyStr),
                        []>, Requires<[hasLDG]>;
 def avar:  RVGPUInst<(outs regclass:$result), (ins imemAny:$src),
               !strconcat("ld.global.nc.", TyStr),
                      []>, Requires<[hasLDG]>;
 def ari :  RVGPUInst<(outs regclass:$result), (ins MEMri:$src),
               !strconcat("ld.global.nc.", TyStr),
                      []>, Requires<[hasLDG]>;
 def ari64 :  RVGPUInst<(outs regclass:$result), (ins MEMri64:$src),
               !strconcat("ld.global.nc.", TyStr),
                        []>, Requires<[hasLDG]>;
}
}
defm INT_PTX_LDG_GLOBAL_i8
  : LDG_G<"u8 \t$result, [$src]", GPR16>;
defm INT_PTX_LDG_GLOBAL_i16
  : LDG_G<"u16 \t$result, [$src]", GPR16>;
defm INT_PTX_LDG_GLOBAL_i32
  : LDG_G<"u32 \t$result, [$src]", GPR32>;
defm INT_PTX_LDG_GLOBAL_i64
  : LDG_G<"u64 \t$result, [$src]", GPR64>;
defm INT_PTX_LDG_GLOBAL_f32
  : LDG_G<"f32 \t$result, [$src]", GPR32>;
defm INT_PTX_LDG_GLOBAL_f64
  : LDG_G<"f64 \t$result, [$src]", GPR64>;

// vector

// Elementized vector ldg
let isCodeGenOnly = 1 in {

multiclass VLDG_G_ELE_V2<string TyStr, RVGPURegClass regclass> {
 def _areg32: RVGPUInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins GPR32:$src),
                     !strconcat("ld.global.nc.", TyStr), []>;
 def _areg64: RVGPUInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins GPR64:$src),
                     !strconcat("ld.global.nc.", TyStr), []>;
 def _ari32: RVGPUInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins MEMri:$src),
                     !strconcat("ld.global.nc.", TyStr), []>;
 def _ari64: RVGPUInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins MEMri64:$src),
                     !strconcat("ld.global.nc.", TyStr), []>;
 def _avar: RVGPUInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins imemAny:$src),
                     !strconcat("ld.global.nc.", TyStr), []>;
}

multiclass VLDG_G_ELE_V4<string TyStr, RVGPURegClass regclass> {
  def _areg32: RVGPUInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                              regclass:$dst4), (ins GPR32:$src),
               !strconcat("ld.global.nc.", TyStr), []>;
  def _areg64: RVGPUInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                               regclass:$dst4), (ins GPR64:$src),
               !strconcat("ld.global.nc.", TyStr), []>;
  def _ari32: RVGPUInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                              regclass:$dst4), (ins MEMri:$src),
               !strconcat("ld.global.nc.", TyStr), []>;
  def _ari64: RVGPUInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                              regclass:$dst4), (ins MEMri64:$src),
               !strconcat("ld.global.nc.", TyStr), []>;
  def _avar: RVGPUInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                             regclass:$dst4), (ins imemAny:$src),
               !strconcat("ld.global.nc.", TyStr), []>;
}
}

// FIXME: 8-bit LDG should be fixed once LDG/LDU nodes are made into proper loads.
defm INT_PTX_LDG_G_v2i8_ELE
  : VLDG_G_ELE_V2<"v2.u8 \t{{$dst1, $dst2}}, [$src]",  GPR16>;
defm INT_PTX_LDG_G_v2i16_ELE
  : VLDG_G_ELE_V2<"v2.u16 \t{{$dst1, $dst2}}, [$src]", GPR16>;
defm INT_PTX_LDG_G_v2i32_ELE
  : VLDG_G_ELE_V2<"v2.u32 \t{{$dst1, $dst2}}, [$src]", GPR32>;
defm INT_PTX_LDG_G_v2f32_ELE
  : VLDG_G_ELE_V2<"v2.f32 \t{{$dst1, $dst2}}, [$src]", GPR32>;
defm INT_PTX_LDG_G_v2i64_ELE
  : VLDG_G_ELE_V2<"v2.u64 \t{{$dst1, $dst2}}, [$src]", GPR64>;
defm INT_PTX_LDG_G_v2f64_ELE
  : VLDG_G_ELE_V2<"v2.f64 \t{{$dst1, $dst2}}, [$src]", GPR64>;
defm INT_PTX_LDG_G_v4i8_ELE
  : VLDG_G_ELE_V4<"v4.u8 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src]", GPR16>;
defm INT_PTX_LDG_G_v4i16_ELE
  : VLDG_G_ELE_V4<"v4.u16 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src]", GPR16>;
defm INT_PTX_LDG_G_v4i32_ELE
  : VLDG_G_ELE_V4<"v4.u32 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src]", GPR32>;
defm INT_PTX_LDG_G_v4f32_ELE
  : VLDG_G_ELE_V4<"v4.f32 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src]", GPR32>;


multiclass NG_TO_G<string Str, Intrinsic Intrin> {
   def _yes : RVGPUInst<(outs GPR32:$result), (ins GPR32:$src),
          !strconcat("cvta.", Str, ".u32 \t$result, $src"),
      [(set GPR32:$result, (Intrin GPR32:$src))]>;
   def _yes_64 : RVGPUInst<(outs GPR64:$result), (ins GPR64:$src),
          !strconcat("cvta.", Str, ".u64 \t$result, $src"),
      [(set GPR64:$result, (Intrin GPR64:$src))]>;
   let isCodeGenOnly = 1 in {
     def _yes_6432 : RVGPUInst<(outs GPR64:$result), (ins GPR32:$src),
          "{{ .reg .b64 %tmp;\n\t"
          #"  cvt.u64.u32 \t%tmp, $src;\n\t"
          #"  cvta." # Str # ".u64 \t$result, %tmp; }}",
      [(set GPR64:$result, (Intrin GPR32:$src))]>,
      Requires<[useShortPtr]>;
    }
}

multiclass G_TO_NG<string Str, Intrinsic Intrin> {
   def _yes : RVGPUInst<(outs GPR32:$result), (ins GPR32:$src),
          !strconcat("cvta.to.", Str, ".u32 \t$result, $src"),
      [(set GPR32:$result, (Intrin GPR32:$src))]>;
   def _yes_64 : RVGPUInst<(outs GPR64:$result), (ins GPR64:$src),
          !strconcat("cvta.to.", Str, ".u64 \t$result, $src"),
      [(set GPR64:$result, (Intrin GPR64:$src))]>;
   let isCodeGenOnly = 1 in {
     def _yes_3264 : RVGPUInst<(outs GPR32:$result), (ins GPR64:$src),
          "{{ .reg .b64 %tmp;\n\t"
          #"  cvta.to." # Str # ".u64 \t%tmp, $src;\n\t"
          #"  cvt.u32.u64 \t$result, %tmp; }}",
      [(set GPR32:$result, (Intrin GPR64:$src))]>,
      Requires<[useShortPtr]>;
  }
}

defm cvta_local  : NG_TO_G<"local", int_rvgpu_ptr_local_to_gen>;
defm cvta_shared : NG_TO_G<"shared", int_rvgpu_ptr_shared_to_gen>;
defm cvta_global : NG_TO_G<"global", int_rvgpu_ptr_global_to_gen>;
defm cvta_const  : NG_TO_G<"const", int_rvgpu_ptr_constant_to_gen>;

defm cvta_to_local   : G_TO_NG<"local", int_rvgpu_ptr_gen_to_local>;
defm cvta_to_shared : G_TO_NG<"shared", int_rvgpu_ptr_gen_to_shared>;
defm cvta_to_global : G_TO_NG<"global", int_rvgpu_ptr_gen_to_global>;
defm cvta_to_const  : G_TO_NG<"const", int_rvgpu_ptr_gen_to_constant>;


// nvvm.ptr.gen.to.param
def nvvm_ptr_gen_to_param : RVGPUInst<(outs GPR32:$result),
  (ins GPR32:$src),
                        "mov.u32 \t$result, $src",
                              [(set GPR32:$result,
                                (int_rvgpu_ptr_gen_to_param GPR32:$src))]>;
def nvvm_ptr_gen_to_param_64 : RVGPUInst<(outs GPR64:$result),
  (ins GPR64:$src),
                        "mov.u64 \t$result, $src",
                              [(set GPR64:$result,
                                (int_rvgpu_ptr_gen_to_param GPR64:$src))]>;


// nvvm.move intrinsicc
def nvvm_move_i16 : RVGPUInst<(outs GPR16:$r), (ins GPR16:$s),
                             "mov.b16 \t$r, $s",
                             [(set GPR16:$r,
                               (int_rvgpu_move_i16 GPR16:$s))]>;
def nvvm_move_i32 : RVGPUInst<(outs GPR32:$r), (ins GPR32:$s),
                             "mov.b32 \t$r, $s",
                             [(set GPR32:$r,
                               (int_rvgpu_move_i32 GPR32:$s))]>;
def nvvm_move_i64 : RVGPUInst<(outs GPR64:$r), (ins GPR64:$s),
                             "mov.b64 \t$r, $s",
                             [(set GPR64:$r,
                               (int_rvgpu_move_i64 GPR64:$s))]>;
def nvvm_move_float : RVGPUInst<(outs GPR32:$r), (ins GPR32:$s),
                             "mov.f32 \t$r, $s",
                             [(set GPR32:$r,
                               (int_rvgpu_move_float GPR32:$s))]>;
def nvvm_move_double : RVGPUInst<(outs GPR64:$r), (ins GPR64:$s),
                             "mov.f64 \t$r, $s",
                             [(set GPR64:$r,
                               (int_rvgpu_move_double GPR64:$s))]>;
def nvvm_move_ptr32 : RVGPUInst<(outs GPR32:$r), (ins GPR32:$s),
                             "mov.u32 \t$r, $s",
                             [(set GPR32:$r,
                               (int_rvgpu_move_ptr GPR32:$s))]>;
def nvvm_move_ptr64 : RVGPUInst<(outs GPR64:$r), (ins GPR64:$s),
                             "mov.u64 \t$r, $s",
                             [(set GPR64:$r,
                               (int_rvgpu_move_ptr GPR64:$s))]>;

// @TODO: Are these actually needed, or will we always just see symbols
// copied to registers first?
/*def nvvm_move_sym32 : RVGPUInst<(outs GPR32:$r), (ins imem:$s),
                             "mov.u32 \t$r, $s",
                             [(set GPR32:$r,
                             (int_rvgpu_move_ptr texternalsym:$s))]>;
def nvvm_move_sym64 : RVGPUInst<(outs GPR64:$r), (ins imem:$s),
                             "mov.u64 \t$r, $s",
                             [(set GPR64:$r,
                             (int_rvgpu_move_ptr texternalsym:$s))]>;*/


// MoveParam        %r1, param
// ptr_local_to_gen %r2, %r1
// ptr_gen_to_local %r3, %r2
// ->
// mov %r1, param

// @TODO: Revisit this.  There is a type
// contradiction between iPTRAny and iPTR for the addr defs, so the move_sym
// instructions are not currently defined. However, we can use the ptr
// variants and the asm printer will do the right thing.
def : Pat<(i64 (int_rvgpu_ptr_gen_to_local (int_rvgpu_ptr_local_to_gen
                (MoveParam texternalsym:$src)))),
               (nvvm_move_ptr64  texternalsym:$src)>;
def : Pat<(i32 (int_rvgpu_ptr_gen_to_local (int_rvgpu_ptr_local_to_gen
                (MoveParam texternalsym:$src)))),
               (nvvm_move_ptr32  texternalsym:$src)>;

def texsurf_handles
  : RVGPUInst<(outs GPR64:$result), (ins imem:$src),
              "mov.u64 \t$result, $src", []>;

//-----------------------------------
// Compiler Error Warn
// - Just ignore them in codegen
//-----------------------------------

def INT_NVVM_COMPILER_WARN_32 : RVGPUInst<(outs), (ins GPR32:$a),
                "// llvm.nvvm.compiler.warn()",
                [(int_rvgpu_compiler_warn GPR32:$a)]>;
def INT_NVVM_COMPILER_WARN_64 : RVGPUInst<(outs), (ins GPR64:$a),
                "// llvm.nvvm.compiler.warn()",
                [(int_rvgpu_compiler_warn GPR64:$a)]>;
def INT_NVVM_COMPILER_ERROR_32 : RVGPUInst<(outs), (ins GPR32:$a),
                "// llvm.nvvm.compiler.error()",
                [(int_rvgpu_compiler_error GPR32:$a)]>;
def INT_NVVM_COMPILER_ERROR_64 : RVGPUInst<(outs), (ins GPR64:$a),
                "// llvm.nvvm.compiler.error()",
                [(int_rvgpu_compiler_error GPR64:$a)]>;


// isspacep

multiclass ISSPACEP<string suffix, Intrinsic Intr, list<Predicate> Preds = []> {
  def _32: RVGPUInst<(outs Int1Regs:$d), (ins GPR32:$a),
              "isspacep." # suffix # "\t$d, $a",
              [(set Int1Regs:$d, (Intr GPR32:$a))]>,
    Requires<Preds>;
  def _64: RVGPUInst<(outs Int1Regs:$d), (ins GPR64:$a),
              "isspacep." # suffix # "\t$d, $a",
              [(set Int1Regs:$d, (Intr GPR64:$a))]>,
    Requires<Preds>;
}

defm isspace_const  : ISSPACEP<"const", int_rvgpu_isspacep_const, [hasPTX<31>]>;
defm isspace_global : ISSPACEP<"global", int_rvgpu_isspacep_global>;
defm isspace_local  : ISSPACEP<"local", int_rvgpu_isspacep_local>;
defm isspace_shared : ISSPACEP<"shared", int_rvgpu_isspacep_shared>;
defm isspace_shared_cluster : ISSPACEP<"shared::cluster",
                                       int_rvgpu_isspacep_shared_cluster,
                                       [hasPTX<78>, hasSM<90>]>;

// Special register reads
def MOV_SPECIAL : RVGPUInst<(outs GPR32:$d),
                            (ins SpecialRegs:$r),
                            "mov.b32 \t$d, $r", []>;
/*
def : Pat<(int_rvgpu_read_ptx_sreg_envreg0), (MOV_SPECIAL ENVREG0)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg1), (MOV_SPECIAL ENVREG1)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg2), (MOV_SPECIAL ENVREG2)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg3), (MOV_SPECIAL ENVREG3)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg4), (MOV_SPECIAL ENVREG4)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg5), (MOV_SPECIAL ENVREG5)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg6), (MOV_SPECIAL ENVREG6)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg7), (MOV_SPECIAL ENVREG7)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg8), (MOV_SPECIAL ENVREG8)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg9), (MOV_SPECIAL ENVREG9)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg10), (MOV_SPECIAL ENVREG10)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg11), (MOV_SPECIAL ENVREG11)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg12), (MOV_SPECIAL ENVREG12)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg13), (MOV_SPECIAL ENVREG13)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg14), (MOV_SPECIAL ENVREG14)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg15), (MOV_SPECIAL ENVREG15)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg16), (MOV_SPECIAL ENVREG16)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg17), (MOV_SPECIAL ENVREG17)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg18), (MOV_SPECIAL ENVREG18)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg19), (MOV_SPECIAL ENVREG19)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg20), (MOV_SPECIAL ENVREG20)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg21), (MOV_SPECIAL ENVREG21)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg22), (MOV_SPECIAL ENVREG22)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg23), (MOV_SPECIAL ENVREG23)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg24), (MOV_SPECIAL ENVREG24)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg25), (MOV_SPECIAL ENVREG25)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg26), (MOV_SPECIAL ENVREG26)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg27), (MOV_SPECIAL ENVREG27)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg28), (MOV_SPECIAL ENVREG28)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg29), (MOV_SPECIAL ENVREG29)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg30), (MOV_SPECIAL ENVREG30)>;
def : Pat<(int_rvgpu_read_ptx_sreg_envreg31), (MOV_SPECIAL ENVREG31)>;
*/

// rotate builtin support

def ROTATE_B32_HW_IMM
  : RVGPUInst<(outs GPR32:$dst),
              (ins  GPR32:$src, i32imm:$amt),
              "shf.l.wrap.b32 \t$dst, $src, $src, $amt",
              [(set GPR32:$dst,
                 (int_rvgpu_rotate_b32 GPR32:$src, (i32 imm:$amt)))]>,
              Requires<[hasHWROT32]> ;

def ROTATE_B32_HW_REG
  : RVGPUInst<(outs GPR32:$dst),
              (ins  GPR32:$src, GPR32:$amt),
              "shf.l.wrap.b32 \t$dst, $src, $src, $amt",
              [(set GPR32:$dst,
                 (int_rvgpu_rotate_b32 GPR32:$src, GPR32:$amt))]>,
              Requires<[hasHWROT32]> ;

def : Pat<(int_rvgpu_rotate_b32 GPR32:$src, (i32 imm:$amt)),
          (ROT32imm_sw GPR32:$src, imm:$amt, (SUB_FRM_32 node:$amt))>,
      Requires<[noHWROT32]> ;

def : Pat<(int_rvgpu_rotate_b32 GPR32:$src, GPR32:$amt),
          (ROTL32reg_sw GPR32:$src, GPR32:$amt)>,
      Requires<[noHWROT32]> ;

let hasSideEffects = false, isCodeGenOnly = 1 in {
  def GET_LO_INT64 : RVGPUInst<(outs GPR32:$dst), (ins GPR64:$src),
    !strconcat("{{\n\t",
               ".reg .b32 %dummy;\n\t",
               "mov.b64 \t{$dst,%dummy}, $src;\n\t",
               "}}"),
          []> ;

  def GET_HI_INT64 : RVGPUInst<(outs GPR32:$dst), (ins GPR64:$src),
    !strconcat("{{\n\t",
               ".reg .b32 %dummy;\n\t",
               "mov.b64 \t{%dummy,$dst}, $src;\n\t",
               "}}"),
          []> ;
}

let hasSideEffects = false in {
  def PACK_TWO_INT32
    : RVGPUInst<(outs GPR64:$dst), (ins GPR32:$lo, GPR32:$hi),
                "mov.b64 \t$dst, $lo, $hi", []> ;
}

def : Pat<(int_rvgpu_swap_lo_hi_b64 GPR64:$src),
          (PACK_TWO_INT32 (GET_HI_INT64 GPR64:$src),
                          (GET_LO_INT64 GPR64:$src))> ;

// Funnel shift, requires >= sm_32.  Does not trap if amt is out of range, so
// no side effects.
let hasSideEffects = false in {
  def SHF_L_WRAP_B32_IMM
    : RVGPUInst<(outs GPR32:$dst),
                (ins  GPR32:$lo, GPR32:$hi, i32imm:$amt),
                "shf.l.wrap.b32 \t$dst, $lo, $hi, $amt",[]>,
      Requires<[hasHWROT32]>;

  def SHF_L_WRAP_B32_REG
    : RVGPUInst<(outs GPR32:$dst),
                (ins  GPR32:$lo, GPR32:$hi, GPR32:$amt),
                "shf.l.wrap.b32 \t$dst, $lo, $hi, $amt",[]>,
      Requires<[hasHWROT32]>;

  def SHF_R_WRAP_B32_IMM
    : RVGPUInst<(outs GPR32:$dst),
                (ins  GPR32:$lo, GPR32:$hi, i32imm:$amt),
                "shf.r.wrap.b32 \t$dst, $lo, $hi, $amt",[]>,
      Requires<[hasHWROT32]>;

  def SHF_R_WRAP_B32_REG
    : RVGPUInst<(outs GPR32:$dst),
                (ins  GPR32:$lo, GPR32:$hi, GPR32:$amt),
                "shf.r.wrap.b32 \t$dst, $lo, $hi, $amt",[]>,
      Requires<[hasHWROT32]>;
}

// HW version of rotate 64
def : Pat<(int_rvgpu_rotate_b64 GPR64:$src, (i32 imm:$amt)),
          (PACK_TWO_INT32
            (SHF_L_WRAP_B32_IMM (GET_HI_INT64 GPR64:$src),
                                (GET_LO_INT64 GPR64:$src), imm:$amt),
            (SHF_L_WRAP_B32_IMM (GET_LO_INT64 GPR64:$src),
                                (GET_HI_INT64 GPR64:$src), imm:$amt))>,
      Requires<[hasHWROT32]>;

def : Pat<(int_rvgpu_rotate_b64 GPR64:$src, GPR32:$amt),
          (PACK_TWO_INT32
            (SHF_L_WRAP_B32_REG (GET_HI_INT64 GPR64:$src),
                                (GET_LO_INT64 GPR64:$src), GPR32:$amt),
            (SHF_L_WRAP_B32_REG (GET_LO_INT64 GPR64:$src),
                               (GET_HI_INT64 GPR64:$src), GPR32:$amt))>,
      Requires<[hasHWROT32]>;


def : Pat<(int_rvgpu_rotate_right_b64 GPR64:$src, (i32 imm:$amt)),
          (PACK_TWO_INT32
            (SHF_R_WRAP_B32_IMM (GET_LO_INT64 GPR64:$src),
                                (GET_HI_INT64 GPR64:$src), imm:$amt),
            (SHF_R_WRAP_B32_IMM (GET_HI_INT64 GPR64:$src),
                                (GET_LO_INT64 GPR64:$src), imm:$amt))>,
      Requires<[hasHWROT32]>;

def : Pat<(int_rvgpu_rotate_right_b64 GPR64:$src, GPR32:$amt),
          (PACK_TWO_INT32
            (SHF_R_WRAP_B32_REG (GET_LO_INT64 GPR64:$src),
                                (GET_HI_INT64 GPR64:$src), GPR32:$amt),
            (SHF_R_WRAP_B32_REG (GET_HI_INT64 GPR64:$src),
                               (GET_LO_INT64 GPR64:$src), GPR32:$amt))>,
      Requires<[hasHWROT32]>;

// SW version of rotate 64
def : Pat<(int_rvgpu_rotate_b64 GPR64:$src, (i32 imm:$amt)),
          (ROT64imm_sw GPR64:$src, imm:$amt, (SUB_FRM_64 node:$amt))>,
      Requires<[noHWROT32]>;
def : Pat<(int_rvgpu_rotate_b64 GPR64:$src, GPR32:$amt),
          (ROTL64reg_sw GPR64:$src, GPR32:$amt)>,
      Requires<[noHWROT32]>;
def : Pat<(int_rvgpu_rotate_right_b64 GPR64:$src, (i32 imm:$amt)),
          (ROT64imm_sw GPR64:$src, (SUB_FRM_64 node:$amt), imm:$amt)>,
      Requires<[noHWROT32]>;
def : Pat<(int_rvgpu_rotate_right_b64 GPR64:$src, GPR32:$amt),
          (ROTR64reg_sw GPR64:$src, GPR32:$amt)>,
      Requires<[noHWROT32]>;


//-----------------------------------
// Texture Intrinsics
//-----------------------------------

// NOTE: For Fermi support, any new texture/surface/sampler intrinsics must be
// also defined in RVGPUReplaceImageHandles.cpp

// texmode_independent
let IsTex = true, IsTexModeUnified = false in {
// Texture fetch instructions using handles

class TEX_1D_base<string inst, RVGPURegClass outtype,
                  RVGPURegClass intype, dag texsamp>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$x\\}]",
                 []>;

multiclass TEX_1D<string inst, RVGPURegClass outtype, RVGPURegClass intype> {
  def _RR : TEX_1D_base<inst, outtype, intype,
                        (ins GPR64:$t, GPR64:$s)>;
  def _RI : TEX_1D_base<inst, outtype, intype,
                        (ins GPR64:$t, i64imm:$s)>;
  def _IR : TEX_1D_base<inst, outtype, intype,
                        (ins i64imm:$t, GPR64:$s)>;
  def _II : TEX_1D_base<inst, outtype, intype,
                        (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_1D_F32_S32 : TEX_1D<"tex.1d.v4.f32.s32", GPR32, GPR32>;
defm TEX_1D_F32_F32 : TEX_1D<"tex.1d.v4.f32.f32", GPR32, GPR32>;
defm TEX_1D_S32_S32 : TEX_1D<"tex.1d.v4.s32.s32", GPR32, GPR32>;
defm TEX_1D_S32_F32 : TEX_1D<"tex.1d.v4.s32.f32", GPR32, GPR32>;
defm TEX_1D_U32_S32 : TEX_1D<"tex.1d.v4.u32.s32", GPR32, GPR32>;
defm TEX_1D_U32_F32 : TEX_1D<"tex.1d.v4.u32.f32", GPR32, GPR32>;

class TEX_1D_LEVEL_base<string inst, RVGPURegClass outtype,
                        RVGPURegClass intype, dag texsamp>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x, intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$x\\}], $lod",
                 []>;

multiclass TEX_1D_LEVEL<string inst, RVGPURegClass outtype,
                        RVGPURegClass intype> {
  def _RR : TEX_1D_LEVEL_base<inst, outtype, intype,
                              (ins GPR64:$t, GPR64:$s)>;
  def _RI : TEX_1D_LEVEL_base<inst, outtype, intype,
                              (ins GPR64:$t, i64imm:$s)>;
  def _IR : TEX_1D_LEVEL_base<inst, outtype, intype,
                              (ins i64imm:$t, GPR64:$s)>;
  def _II : TEX_1D_LEVEL_base<inst, outtype, intype,
                              (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_1D_F32_F32_LEVEL :
  TEX_1D_LEVEL<"tex.level.1d.v4.f32.f32", GPR32, GPR32>;
defm TEX_1D_S32_F32_LEVEL :
  TEX_1D_LEVEL<"tex.level.1d.v4.s32.f32", GPR32, GPR32>;
defm TEX_1D_U32_F32_LEVEL :
  TEX_1D_LEVEL<"tex.level.1d.v4.u32.f32", GPR32, GPR32>;

class TEX_1D_GRAD_base<string inst, RVGPURegClass outtype,
                       RVGPURegClass intype, dag texsamp>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x, intype:$gradx, intype:$grady)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$x\\}],"
                        " \\{$gradx\\}, \\{$grady\\}",
                 []>;

multiclass TEX_1D_GRAD<string inst, RVGPURegClass outtype,
                       RVGPURegClass intype> {
  def _RR : TEX_1D_GRAD_base<inst, outtype, intype,
                             (ins GPR64:$t, GPR64:$s)>;
  def _RI : TEX_1D_GRAD_base<inst, outtype, intype,
                             (ins GPR64:$t, i64imm:$s)>;
  def _IR : TEX_1D_GRAD_base<inst, outtype, intype,
                             (ins i64imm:$t, GPR64:$s)>;
  def _II : TEX_1D_GRAD_base<inst, outtype, intype,
                             (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_1D_F32_F32_GRAD
  : TEX_1D_GRAD<"tex.grad.1d.v4.f32.f32", GPR32, GPR32>;
defm TEX_1D_S32_F32_GRAD
  : TEX_1D_GRAD<"tex.grad.1d.v4.s32.f32", GPR32, GPR32>;
defm TEX_1D_U32_F32_GRAD
  : TEX_1D_GRAD<"tex.grad.1d.v4.u32.f32", GPR32, GPR32>;

class TEX_1D_ARRAY_base<string inst, RVGPURegClass outtype,
                        RVGPURegClass intype, dag texsamp>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins GPR32:$l, intype:$x)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$l, $x\\}]",
                 []>;

multiclass TEX_1D_ARRAY<string inst, RVGPURegClass outtype,
                        RVGPURegClass intype> {
  def _RR : TEX_1D_ARRAY_base<inst, outtype, intype,
                              (ins GPR64:$t, GPR64:$s)>;
  def _RI : TEX_1D_ARRAY_base<inst, outtype, intype,
                              (ins GPR64:$t, i64imm:$s)>;
  def _IR : TEX_1D_ARRAY_base<inst, outtype, intype,
                              (ins i64imm:$t, GPR64:$s)>;
  def _II : TEX_1D_ARRAY_base<inst, outtype, intype,
                              (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_1D_ARRAY_F32_F32
  : TEX_1D_ARRAY<"tex.a1d.v4.f32.f32", GPR32, GPR32>;
defm TEX_1D_ARRAY_F32_S32
  : TEX_1D_ARRAY<"tex.a1d.v4.f32.s32", GPR32, GPR32>;
defm TEX_1D_ARRAY_S32_S32
  : TEX_1D_ARRAY<"tex.a1d.v4.s32.s32", GPR32, GPR32>;
defm TEX_1D_ARRAY_S32_F32
  : TEX_1D_ARRAY<"tex.a1d.v4.s32.f32", GPR32, GPR32>;
defm TEX_1D_ARRAY_U32_S32
  : TEX_1D_ARRAY<"tex.a1d.v4.u32.s32", GPR32, GPR32>;
defm TEX_1D_ARRAY_U32_F32
  : TEX_1D_ARRAY<"tex.a1d.v4.u32.f32", GPR32, GPR32>;

class TEX_1D_ARRAY_LEVEL_base<string inst, RVGPURegClass outtype,
                              RVGPURegClass intype, dag texsamp>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins GPR32:$l, intype:$x, intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$l, $x\\}], $lod",
                 []>;

multiclass TEX_1D_ARRAY_LEVEL<string inst, RVGPURegClass outtype,
                              RVGPURegClass intype> {
  def _RR : TEX_1D_ARRAY_LEVEL_base<inst, outtype, intype,
                                    (ins GPR64:$t, GPR64:$s)>;
  def _RI : TEX_1D_ARRAY_LEVEL_base<inst, outtype, intype,
                                    (ins GPR64:$t, i64imm:$s)>;
  def _IR : TEX_1D_ARRAY_LEVEL_base<inst, outtype, intype,
                                    (ins i64imm:$t, GPR64:$s)>;
  def _II : TEX_1D_ARRAY_LEVEL_base<inst, outtype, intype,
                                    (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_1D_ARRAY_F32_F32_LEVEL
  : TEX_1D_ARRAY_LEVEL<"tex.level.a1d.v4.f32.f32", GPR32, GPR32>;
defm TEX_1D_ARRAY_S32_F32_LEVEL
  : TEX_1D_ARRAY_LEVEL<"tex.level.a1d.v4.s32.f32", GPR32, GPR32>;
defm TEX_1D_ARRAY_U32_F32_LEVEL
  : TEX_1D_ARRAY_LEVEL<"tex.level.a1d.v4.u32.f32", GPR32, GPR32>;

class TEX_1D_ARRAY_GRAD_base<string inst, RVGPURegClass outtype,
                             RVGPURegClass intype, dag texsamp>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins GPR32:$l, intype:$x,
                                    intype:$gradx, intype:$grady)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$l, $x\\}],"
                        " \\{$gradx\\}, \\{$grady\\}",
                 []>;

multiclass TEX_1D_ARRAY_GRAD<string inst, RVGPURegClass outtype,
                             RVGPURegClass intype> {
  def _RR : TEX_1D_ARRAY_GRAD_base<inst, outtype, intype,
                                   (ins GPR64:$t, GPR64:$s)>;
  def _RI : TEX_1D_ARRAY_GRAD_base<inst, outtype, intype,
                                   (ins GPR64:$t, i64imm:$s)>;
  def _IR : TEX_1D_ARRAY_GRAD_base<inst, outtype, intype,
                                   (ins i64imm:$t, GPR64:$s)>;
  def _II : TEX_1D_ARRAY_GRAD_base<inst, outtype, intype,
                                   (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_1D_ARRAY_F32_F32_GRAD
  : TEX_1D_ARRAY_GRAD<"tex.grad.a1d.v4.f32.f32", GPR32, GPR32>;
defm TEX_1D_ARRAY_S32_F32_GRAD
  : TEX_1D_ARRAY_GRAD<"tex.grad.a1d.v4.s32.f32", GPR32, GPR32>;
defm TEX_1D_ARRAY_U32_F32_GRAD
  : TEX_1D_ARRAY_GRAD<"tex.grad.a1d.v4.u32.f32", GPR32, GPR32>;

class TEX_2D_base<string inst, RVGPURegClass outtype,
                  RVGPURegClass intype, dag texsamp>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x, intype:$y)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$x, $y\\}]",
                 []>;

multiclass TEX_2D<string inst, RVGPURegClass outtype, RVGPURegClass intype> {
  def _RR : TEX_2D_base<inst, outtype, intype,
                        (ins GPR64:$t, GPR64:$s)>;
  def _RI : TEX_2D_base<inst, outtype, intype, (ins GPR64:$t, i64imm:$s)>;
  def _IR : TEX_2D_base<inst, outtype, intype, (ins i64imm:$t, GPR64:$s)>;
  def _II : TEX_2D_base<inst, outtype, intype, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_2D_F32_F32 : TEX_2D<"tex.2d.v4.f32.f32", GPR32, GPR32>;
defm TEX_2D_F32_S32 : TEX_2D<"tex.2d.v4.f32.s32", GPR32, GPR32>;
defm TEX_2D_S32_S32 : TEX_2D<"tex.2d.v4.s32.s32", GPR32, GPR32>;
defm TEX_2D_S32_F32 : TEX_2D<"tex.2d.v4.s32.f32", GPR32, GPR32>;
defm TEX_2D_U32_S32 : TEX_2D<"tex.2d.v4.u32.s32", GPR32, GPR32>;
defm TEX_2D_U32_F32 : TEX_2D<"tex.2d.v4.u32.f32", GPR32, GPR32>;

class TEX_2D_LEVEL_base<string inst, RVGPURegClass outtype,
                        RVGPURegClass intype, dag texsamp>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x, intype:$y, intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$x, $y\\}], $lod",
                 []>;

multiclass TEX_2D_LEVEL<string inst, RVGPURegClass outtype,
                        RVGPURegClass intype> {
  def _RR : TEX_2D_LEVEL_base<inst, outtype, intype,
                              (ins GPR64:$t, GPR64:$s)>;
  def _RI : TEX_2D_LEVEL_base<inst, outtype, intype,
                              (ins GPR64:$t, i64imm:$s)>;
  def _IR : TEX_2D_LEVEL_base<inst, outtype, intype,
                              (ins i64imm:$t, GPR64:$s)>;
  def _II : TEX_2D_LEVEL_base<inst, outtype, intype,
                              (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_2D_F32_F32_LEVEL :
  TEX_2D_LEVEL<"tex.level.2d.v4.f32.f32", GPR32, GPR32>;
defm TEX_2D_S32_F32_LEVEL :
  TEX_2D_LEVEL<"tex.level.2d.v4.s32.f32", GPR32, GPR32>;
defm TEX_2D_U32_F32_LEVEL :
  TEX_2D_LEVEL<"tex.level.2d.v4.u32.f32", GPR32, GPR32>;

class TEX_2D_GRAD_base<string inst, RVGPURegClass outtype,
                       RVGPURegClass intype, dag texsamp>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x, intype:$y,
                                    intype:$gradx0, intype:$gradx1,
                                    intype:$grady0, intype:$grady1)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$x, $y\\}],"
                        " \\{$gradx0, $gradx1\\}, \\{$grady0, $grady1\\}",
                 []>;

multiclass TEX_2D_GRAD<string inst, RVGPURegClass outtype,
                       RVGPURegClass intype> {
  def _RR : TEX_2D_GRAD_base<inst, outtype, intype,
                              (ins GPR64:$t, GPR64:$s)>;
  def _RI : TEX_2D_GRAD_base<inst, outtype, intype,
                              (ins GPR64:$t, i64imm:$s)>;
  def _IR : TEX_2D_GRAD_base<inst, outtype, intype,
                              (ins i64imm:$t, GPR64:$s)>;
  def _II : TEX_2D_GRAD_base<inst, outtype, intype,
                              (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_2D_F32_F32_GRAD :
  TEX_2D_GRAD<"tex.grad.2d.v4.f32.f32", GPR32, GPR32>;
defm TEX_2D_S32_F32_GRAD :
  TEX_2D_GRAD<"tex.grad.2d.v4.s32.f32", GPR32, GPR32>;
defm TEX_2D_U32_F32_GRAD :
  TEX_2D_GRAD<"tex.grad.2d.v4.u32.f32", GPR32, GPR32>;

class TEX_2D_ARRAY_base<string inst, RVGPURegClass outtype,
                        RVGPURegClass intype, dag texsamp>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins GPR32:$l, intype:$x, intype:$y)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$l, $x, $y, $y\\}]",
                 []>;

multiclass TEX_2D_ARRAY<string inst, RVGPURegClass outtype,
                        RVGPURegClass intype> {
  def _RR : TEX_2D_ARRAY_base<inst, outtype, intype,
                              (ins GPR64:$t, GPR64:$s)>;
  def _RI : TEX_2D_ARRAY_base<inst, outtype, intype,
                              (ins GPR64:$t, i64imm:$s)>;
  def _IR : TEX_2D_ARRAY_base<inst, outtype, intype,
                              (ins i64imm:$t, GPR64:$s)>;
  def _II : TEX_2D_ARRAY_base<inst, outtype, intype,
                              (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_2D_ARRAY_F32_F32
  : TEX_2D_ARRAY<"tex.a2d.v4.f32.f32", GPR32, GPR32>;
defm TEX_2D_ARRAY_F32_S32
  : TEX_2D_ARRAY<"tex.a2d.v4.f32.s32", GPR32, GPR32>;
defm TEX_2D_ARRAY_S32_S32
  : TEX_2D_ARRAY<"tex.a2d.v4.s32.s32", GPR32, GPR32>;
defm TEX_2D_ARRAY_S32_F32
  : TEX_2D_ARRAY<"tex.a2d.v4.s32.f32", GPR32, GPR32>;
defm TEX_2D_ARRAY_U32_S32
  : TEX_2D_ARRAY<"tex.a2d.v4.u32.s32", GPR32, GPR32>;
defm TEX_2D_ARRAY_U32_F32
  : TEX_2D_ARRAY<"tex.a2d.v4.u32.f32", GPR32, GPR32>;

class TEX_2D_ARRAY_LEVEL_base<string inst, RVGPURegClass outtype,
                              RVGPURegClass intype, dag texsamp>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins GPR32:$l, intype:$x, intype:$y,
                                    intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$l, $x, $y, $y\\}], $lod",
                 []>;

multiclass TEX_2D_ARRAY_LEVEL<string inst, RVGPURegClass outtype,
                              RVGPURegClass intype> {
  def _RR : TEX_2D_ARRAY_LEVEL_base<inst, outtype, intype,
                              (ins GPR64:$t, GPR64:$s)>;
  def _RI : TEX_2D_ARRAY_LEVEL_base<inst, outtype, intype,
                              (ins GPR64:$t, i64imm:$s)>;
  def _IR : TEX_2D_ARRAY_LEVEL_base<inst, outtype, intype,
                              (ins i64imm:$t, GPR64:$s)>;
  def _II : TEX_2D_ARRAY_LEVEL_base<inst, outtype, intype,
                              (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_2D_ARRAY_F32_F32_LEVEL
  : TEX_2D_ARRAY_LEVEL<"tex.level.a2d.v4.f32.f32", GPR32, GPR32>;
defm TEX_2D_ARRAY_S32_F32_LEVEL
  : TEX_2D_ARRAY_LEVEL<"tex.level.a2d.v4.s32.f32", GPR32, GPR32>;
defm TEX_2D_ARRAY_U32_F32_LEVEL
  : TEX_2D_ARRAY_LEVEL<"tex.level.a2d.v4.u32.f32", GPR32, GPR32>;

class TEX_2D_ARRAY_GRAD_base<string inst, RVGPURegClass outtype,
                             RVGPURegClass intype, dag texsamp>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins GPR32:$l, intype:$x, intype:$y,
                                    intype:$gradx0, intype:$gradx1,
                                    intype:$grady0, intype:$grady1)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$l, $x, $y, $y\\}],"
                        " \\{$gradx0, $gradx1\\}, \\{$grady0, $grady1\\}",
                 []>;

multiclass TEX_2D_ARRAY_GRAD<string inst, RVGPURegClass outtype,
                             RVGPURegClass intype> {
  def _RR : TEX_2D_ARRAY_GRAD_base<inst, outtype, intype,
                              (ins GPR64:$t, GPR64:$s)>;
  def _RI : TEX_2D_ARRAY_GRAD_base<inst, outtype, intype,
                              (ins GPR64:$t, i64imm:$s)>;
  def _IR : TEX_2D_ARRAY_GRAD_base<inst, outtype, intype,
                              (ins i64imm:$t, GPR64:$s)>;
  def _II : TEX_2D_ARRAY_GRAD_base<inst, outtype, intype,
                              (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_2D_ARRAY_F32_F32_GRAD
  : TEX_2D_ARRAY_GRAD<"tex.grad.a2d.v4.f32.f32", GPR32, GPR32>;
defm TEX_2D_ARRAY_S32_F32_GRAD
  : TEX_2D_ARRAY_GRAD<"tex.grad.a2d.v4.s32.f32", GPR32, GPR32>;
defm TEX_2D_ARRAY_U32_F32_GRAD
  : TEX_2D_ARRAY_GRAD<"tex.grad.a2d.v4.u32.f32", GPR32, GPR32>;

class TEX_3D_base<string inst, RVGPURegClass outtype,
                  RVGPURegClass intype, dag texsamp>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x, intype:$y, intype:$z)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$x, $y, $z, $z\\}]",
                 []>;

multiclass TEX_3D<string inst, RVGPURegClass outtype, RVGPURegClass intype> {
  def _RR : TEX_3D_base<inst, outtype, intype,
                              (ins GPR64:$t, GPR64:$s)>;
  def _RI : TEX_3D_base<inst, outtype, intype,
                              (ins GPR64:$t, i64imm:$s)>;
  def _IR : TEX_3D_base<inst, outtype, intype,
                              (ins i64imm:$t, GPR64:$s)>;
  def _II : TEX_3D_base<inst, outtype, intype,
                              (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_3D_F32_F32 : TEX_3D<"tex.3d.v4.f32.f32", GPR32, GPR32>;
defm TEX_3D_F32_S32 : TEX_3D<"tex.3d.v4.f32.s32", GPR32, GPR32>;
defm TEX_3D_S32_S32 : TEX_3D<"tex.3d.v4.s32.s32", GPR32, GPR32>;
defm TEX_3D_S32_F32 : TEX_3D<"tex.3d.v4.s32.f32", GPR32, GPR32>;
defm TEX_3D_U32_S32 : TEX_3D<"tex.3d.v4.u32.s32", GPR32, GPR32>;
defm TEX_3D_U32_F32 : TEX_3D<"tex.3d.v4.u32.f32", GPR32, GPR32>;

class TEX_3D_LEVEL_base<string inst, RVGPURegClass outtype,
                        RVGPURegClass intype, dag texsamp>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x, intype:$y, intype:$z,
                                    intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$x, $y, $z, $z\\}], $lod",
                 []>;

multiclass TEX_3D_LEVEL<string inst, RVGPURegClass outtype,
                        RVGPURegClass intype> {
  def _RR : TEX_3D_LEVEL_base<inst, outtype, intype,
                              (ins GPR64:$t, GPR64:$s)>;
  def _RI : TEX_3D_LEVEL_base<inst, outtype, intype,
                              (ins GPR64:$t, i64imm:$s)>;
  def _IR : TEX_3D_LEVEL_base<inst, outtype, intype,
                              (ins i64imm:$t, GPR64:$s)>;
  def _II : TEX_3D_LEVEL_base<inst, outtype, intype,
                              (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_3D_F32_F32_LEVEL
  : TEX_3D_LEVEL<"tex.level.3d.v4.f32.f32", GPR32, GPR32>;
defm TEX_3D_S32_F32_LEVEL
  : TEX_3D_LEVEL<"tex.level.3d.v4.s32.f32", GPR32, GPR32>;
defm TEX_3D_U32_F32_LEVEL
  : TEX_3D_LEVEL<"tex.level.3d.v4.u32.f32", GPR32, GPR32>;

class TEX_3D_GRAD_base<string inst, RVGPURegClass outtype,
                       RVGPURegClass intype, dag texsamp>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x, intype:$y, intype:$z,
                                    intype :$gradx0, intype:$gradx1,
                                    intype:$gradx2, intype:$grady0,
                                    intype:$grady1, intype:$grady2)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$x, $y, $z, $z\\}],"
                        " \\{$gradx0, $gradx1, $gradx2, $gradx2\\},"
                        " \\{$grady0, $grady1, $grady2, $grady2\\}",
                 []>;

multiclass TEX_3D_GRAD<string inst, RVGPURegClass outtype,
                       RVGPURegClass intype> {
  def _RR : TEX_3D_GRAD_base<inst, outtype, intype,
                             (ins GPR64:$t, GPR64:$s)>;
  def _RI : TEX_3D_GRAD_base<inst, outtype, intype,
                             (ins GPR64:$t, i64imm:$s)>;
  def _IR : TEX_3D_GRAD_base<inst, outtype, intype,
                             (ins i64imm:$t, GPR64:$s)>;
  def _II : TEX_3D_GRAD_base<inst, outtype, intype,
                             (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_3D_F32_F32_GRAD
  : TEX_3D_GRAD<"tex.grad.3d.v4.f32.f32", GPR32, GPR32>;
defm TEX_3D_S32_F32_GRAD
  : TEX_3D_GRAD<"tex.grad.3d.v4.s32.f32", GPR32, GPR32>;
defm TEX_3D_U32_F32_GRAD
  : TEX_3D_GRAD<"tex.grad.3d.v4.u32.f32", GPR32, GPR32>;

class TEX_CUBE_base<string inst, RVGPURegClass outtype,
                    RVGPURegClass intype, dag texsamp>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x, intype:$y, intype:$z)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$x, $y, $z, $z\\}]",
                 []>;

multiclass TEX_CUBE<string inst, RVGPURegClass outtype, RVGPURegClass intype> {
  def _RR : TEX_CUBE_base<inst, outtype, intype,
                          (ins GPR64:$t, GPR64:$s)>;
  def _RI : TEX_CUBE_base<inst, outtype, intype,
                          (ins GPR64:$t, i64imm:$s)>;
  def _IR : TEX_CUBE_base<inst, outtype, intype,
                          (ins i64imm:$t, GPR64:$s)>;
  def _II : TEX_CUBE_base<inst, outtype, intype,
                          (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_CUBE_F32_F32
  : TEX_CUBE<"tex.cube.v4.f32.f32", GPR32, GPR32>;
defm TEX_CUBE_S32_F32
  : TEX_CUBE<"tex.cube.v4.s32.f32", GPR32, GPR32>;
defm TEX_CUBE_U32_F32
  : TEX_CUBE<"tex.cube.v4.u32.f32", GPR32, GPR32>;

class TEX_CUBE_LEVEL_base<string inst, RVGPURegClass outtype,
                          RVGPURegClass intype, dag texsamp>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x, intype:$y, intype:$z,
                                    intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$x, $y, $z, $z\\}], $lod",
                 []>;

multiclass TEX_CUBE_LEVEL<string inst, RVGPURegClass outtype,
                          RVGPURegClass intype> {
  def _RR : TEX_CUBE_LEVEL_base<inst, outtype, intype,
                                (ins GPR64:$t, GPR64:$s)>;
  def _RI : TEX_CUBE_LEVEL_base<inst, outtype, intype,
                                (ins GPR64:$t, i64imm:$s)>;
  def _IR : TEX_CUBE_LEVEL_base<inst, outtype, intype,
                                (ins i64imm:$t, GPR64:$s)>;
  def _II : TEX_CUBE_LEVEL_base<inst, outtype, intype,
                                (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_CUBE_F32_F32_LEVEL
  : TEX_CUBE_LEVEL<"tex.level.cube.v4.f32.f32", GPR32, GPR32>;
defm TEX_CUBE_S32_F32_LEVEL
  : TEX_CUBE_LEVEL<"tex.level.cube.v4.s32.f32", GPR32, GPR32>;
defm TEX_CUBE_U32_F32_LEVEL
  : TEX_CUBE_LEVEL<"tex.level.cube.v4.u32.f32", GPR32, GPR32>;

class TEX_CUBE_ARRAY_base<string inst, RVGPURegClass outtype,
                          RVGPURegClass intype, dag texsamp>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins GPR32:$l, intype:$x, intype:$y,
                                    intype:$z)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$l, $x, $y, $z\\}]",
                 []>;

multiclass TEX_CUBE_ARRAY<string inst, RVGPURegClass outtype,
                          RVGPURegClass intype> {
  def _RR : TEX_CUBE_ARRAY_base<inst, outtype, intype,
                                (ins GPR64:$t, GPR64:$s)>;
  def _RI : TEX_CUBE_ARRAY_base<inst, outtype, intype,
                                (ins GPR64:$t, i64imm:$s)>;
  def _IR : TEX_CUBE_ARRAY_base<inst, outtype, intype,
                                (ins i64imm:$t, GPR64:$s)>;
  def _II : TEX_CUBE_ARRAY_base<inst, outtype, intype,
                                (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_CUBE_ARRAY_F32_F32
  : TEX_CUBE_ARRAY<"tex.acube.v4.f32.f32", GPR32, GPR32>;
defm TEX_CUBE_ARRAY_S32_F32
  : TEX_CUBE_ARRAY<"tex.acube.v4.s32.f32", GPR32, GPR32>;
defm TEX_CUBE_ARRAY_U32_F32
  : TEX_CUBE_ARRAY<"tex.acube.v4.u32.f32", GPR32, GPR32>;

class TEX_CUBE_ARRAY_LEVEL_base<string inst, RVGPURegClass outtype,
                                RVGPURegClass intype, dag texsamp>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins GPR32:$l, intype:$x, intype:$y,
                                    intype:$z, intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$l, $x, $y, $z\\}], $lod",
                 []>;

multiclass TEX_CUBE_ARRAY_LEVEL<string inst, RVGPURegClass outtype,
                                RVGPURegClass intype> {
  def _RR : TEX_CUBE_ARRAY_LEVEL_base<inst, outtype, intype,
                                      (ins GPR64:$t, GPR64:$s)>;
  def _RI : TEX_CUBE_ARRAY_LEVEL_base<inst, outtype, intype,
                                      (ins GPR64:$t, i64imm:$s)>;
  def _IR : TEX_CUBE_ARRAY_LEVEL_base<inst, outtype, intype,
                                      (ins i64imm:$t, GPR64:$s)>;
  def _II : TEX_CUBE_ARRAY_LEVEL_base<inst, outtype, intype,
                                      (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_CUBE_ARRAY_F32_F32_LEVEL
  : TEX_CUBE_ARRAY_LEVEL<"tex.level.acube.v4.f32.f32",
                         GPR32, GPR32>;
defm TEX_CUBE_ARRAY_S32_F32_LEVEL
  : TEX_CUBE_ARRAY_LEVEL<"tex.level.acube.v4.s32.f32",
                         GPR32, GPR32>;
defm TEX_CUBE_ARRAY_U32_F32_LEVEL
  : TEX_CUBE_ARRAY_LEVEL<"tex.level.acube.v4.u32.f32",
                         GPR32, GPR32>;

class TLD4_2D_base<string inst, RVGPURegClass outtype,
                   RVGPURegClass intype, dag texsamp>
    : RVGPUInst<(outs outtype:$v0, outtype:$v1,
                      outtype:$v2, outtype:$v3),
                 !con(texsamp, (ins intype:$x, intype:$y)),
                 inst # " \t\\{$v0, $v1, $v2, $v3\\}, [$t, $s, \\{$x, $y\\}]",
                 []>;

multiclass TLD4_2D<string inst, RVGPURegClass outtype, RVGPURegClass intype> {
  def _RR : TLD4_2D_base<inst, outtype, intype,
                         (ins GPR64:$t, GPR64:$s)>;
  def _RI : TLD4_2D_base<inst, outtype, intype,
                         (ins GPR64:$t, i64imm:$s)>;
  def _IR : TLD4_2D_base<inst, outtype, intype,
                         (ins i64imm:$t, GPR64:$s)>;
  def _II : TLD4_2D_base<inst, outtype, intype,
                         (ins i64imm:$t, i64imm:$s)>;
}

defm TLD4_R_2D_F32_F32
  : TLD4_2D<"tld4.r.2d.v4.f32.f32", GPR32, GPR32>;
defm TLD4_G_2D_F32_F32
  : TLD4_2D<"tld4.g.2d.v4.f32.f32", GPR32, GPR32>;
defm TLD4_B_2D_F32_F32
  : TLD4_2D<"tld4.b.2d.v4.f32.f32", GPR32, GPR32>;
defm TLD4_A_2D_F32_F32
  : TLD4_2D<"tld4.a.2d.v4.f32.f32", GPR32, GPR32>;

defm TLD4_R_2D_S32_F32
  : TLD4_2D<"tld4.r.2d.v4.s32.f32", GPR32, GPR32>;
defm TLD4_G_2D_S32_F32
  : TLD4_2D<"tld4.g.2d.v4.s32.f32", GPR32, GPR32>;
defm TLD4_B_2D_S32_F32
  : TLD4_2D<"tld4.b.2d.v4.s32.f32", GPR32, GPR32>;
defm TLD4_A_2D_S32_F32
  : TLD4_2D<"tld4.a.2d.v4.s32.f32", GPR32, GPR32>;

defm TLD4_R_2D_U32_F32
  : TLD4_2D<"tld4.r.2d.v4.u32.f32", GPR32, GPR32>;
defm TLD4_G_2D_U32_F32
  : TLD4_2D<"tld4.g.2d.v4.u32.f32", GPR32, GPR32>;
defm TLD4_B_2D_U32_F32
  : TLD4_2D<"tld4.b.2d.v4.u32.f32", GPR32, GPR32>;
defm TLD4_A_2D_U32_F32
  : TLD4_2D<"tld4.a.2d.v4.u32.f32", GPR32, GPR32>;

}


// texmode_unified
let IsTex = true, IsTexModeUnified = true in {
// Texture fetch instructions using handles

class TEX_UNIFIED_1D_base<string inst, RVGPURegClass outtype,
                          RVGPURegClass intype, dag tex>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x\\}]",
                 []>;

multiclass TEX_UNIFIED_1D<string inst, RVGPURegClass outtype,
                          RVGPURegClass intype> {
  def _R : TEX_UNIFIED_1D_base<inst, outtype, intype, (ins GPR64:$t)>;
  def _I : TEX_UNIFIED_1D_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_1D_F32_S32
  : TEX_UNIFIED_1D<"tex.1d.v4.f32.s32", GPR32, GPR32>;
defm TEX_UNIFIED_1D_F32_F32
  : TEX_UNIFIED_1D<"tex.1d.v4.f32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_1D_S32_S32
  : TEX_UNIFIED_1D<"tex.1d.v4.s32.s32", GPR32, GPR32>;
defm TEX_UNIFIED_1D_S32_F32
  : TEX_UNIFIED_1D<"tex.1d.v4.s32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_1D_U32_S32
  : TEX_UNIFIED_1D<"tex.1d.v4.u32.s32", GPR32, GPR32>;
defm TEX_UNIFIED_1D_U32_F32
  : TEX_UNIFIED_1D<"tex.1d.v4.u32.f32", GPR32, GPR32>;

class TEX_UNIFIED_1D_LEVEL_base<string inst, RVGPURegClass outtype,
                                RVGPURegClass intype, dag tex>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x\\}], $lod",
                 []>;

multiclass TEX_UNIFIED_1D_LEVEL<string inst, RVGPURegClass outtype,
                                RVGPURegClass intype> {
  def _R : TEX_UNIFIED_1D_LEVEL_base<inst, outtype, intype, (ins GPR64:$t)>;
  def _I : TEX_UNIFIED_1D_LEVEL_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_1D_F32_F32_LEVEL
  : TEX_UNIFIED_1D_LEVEL<"tex.level.1d.v4.f32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_1D_S32_F32_LEVEL
  : TEX_UNIFIED_1D_LEVEL<"tex.level.1d.v4.s32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_1D_U32_F32_LEVEL
  : TEX_UNIFIED_1D_LEVEL<"tex.level.1d.v4.u32.f32", GPR32, GPR32>;

class TEX_UNIFIED_1D_GRAD_base<string inst, RVGPURegClass outtype,
                               RVGPURegClass intype, dag tex>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$gradx, intype:$grady)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, \\{$x\\}], \\{$gradx\\}, \\{$grady\\}",
                 []>;

multiclass TEX_UNIFIED_1D_GRAD<string inst, RVGPURegClass outtype,
                               RVGPURegClass intype> {
  def _R : TEX_UNIFIED_1D_GRAD_base<inst, outtype, intype, (ins GPR64:$t)>;
  def _I : TEX_UNIFIED_1D_GRAD_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_1D_F32_F32_GRAD
  : TEX_UNIFIED_1D_GRAD<"tex.grad.1d.v4.f32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_1D_S32_F32_GRAD
  : TEX_UNIFIED_1D_GRAD<"tex.grad.1d.v4.s32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_1D_U32_F32_GRAD
  : TEX_UNIFIED_1D_GRAD<"tex.grad.1d.v4.u32.f32", GPR32, GPR32>;

class TEX_UNIFIED_1D_ARRAY_base<string inst, RVGPURegClass outtype,
                                RVGPURegClass intype, dag tex>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins GPR32:$l, intype:$x)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$l, $x\\}]",
                 []>;

multiclass TEX_UNIFIED_1D_ARRAY<string inst, RVGPURegClass outtype,
                                RVGPURegClass intype> {
  def _R : TEX_UNIFIED_1D_ARRAY_base<inst, outtype, intype, (ins GPR64:$t)>;
  def _I : TEX_UNIFIED_1D_ARRAY_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_1D_ARRAY_F32_S32
  : TEX_UNIFIED_1D_ARRAY<"tex.a1d.v4.f32.s32", GPR32, GPR32>;
defm TEX_UNIFIED_1D_ARRAY_F32_F32
  : TEX_UNIFIED_1D_ARRAY<"tex.a1d.v4.f32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_1D_ARRAY_S32_S32
  : TEX_UNIFIED_1D_ARRAY<"tex.a1d.v4.s32.s32", GPR32, GPR32>;
defm TEX_UNIFIED_1D_ARRAY_S32_F32
  : TEX_UNIFIED_1D_ARRAY<"tex.a1d.v4.s32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_1D_ARRAY_U32_S32
  : TEX_UNIFIED_1D_ARRAY<"tex.a1d.v4.u32.s32", GPR32, GPR32>;
defm TEX_UNIFIED_1D_ARRAY_U32_F32
  : TEX_UNIFIED_1D_ARRAY<"tex.a1d.v4.u32.f32", GPR32, GPR32>;

class TEX_UNIFIED_1D_ARRAY_LEVEL_base<string inst, RVGPURegClass outtype,
                                      RVGPURegClass intype, dag tex>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins GPR32:$l, intype:$x, intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$l, $x\\}], $lod",
                 []>;

multiclass TEX_UNIFIED_1D_ARRAY_LEVEL<string inst, RVGPURegClass outtype,
                                      RVGPURegClass intype> {
  def _R : TEX_UNIFIED_1D_ARRAY_LEVEL_base<inst, outtype, intype,
                                           (ins GPR64:$t)>;
  def _I : TEX_UNIFIED_1D_ARRAY_LEVEL_base<inst, outtype, intype,
                                           (ins i64imm:$t)>;
}

defm TEX_UNIFIED_1D_ARRAY_F32_F32_LEVEL
  : TEX_UNIFIED_1D_ARRAY_LEVEL<"tex.level.a1d.v4.f32.f32",
                               GPR32, GPR32>;
defm TEX_UNIFIED_1D_ARRAY_S32_F32_LEVEL
  : TEX_UNIFIED_1D_ARRAY_LEVEL<"tex.level.a1d.v4.s32.f32",
                               GPR32, GPR32>;
defm TEX_UNIFIED_1D_ARRAY_U32_F32_LEVEL
  : TEX_UNIFIED_1D_ARRAY_LEVEL<"tex.level.a1d.v4.u32.f32",
                               GPR32, GPR32>;

class TEX_UNIFIED_1D_ARRAY_GRAD_base<string inst, RVGPURegClass outtype,
                                     RVGPURegClass intype, dag tex>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins GPR32:$l, intype:$x,
                                intype:$gradx, intype:$grady)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        "  [$t, \\{$l, $x\\}], \\{$gradx\\}, \\{$grady\\}",
                 []>;

multiclass TEX_UNIFIED_1D_ARRAY_GRAD<string inst, RVGPURegClass outtype,
                                     RVGPURegClass intype> {
  def _R : TEX_UNIFIED_1D_ARRAY_GRAD_base<inst, outtype, intype,
                                          (ins GPR64:$t)>;
  def _I : TEX_UNIFIED_1D_ARRAY_GRAD_base<inst, outtype, intype,
                                          (ins i64imm:$t)>;
}

defm TEX_UNIFIED_1D_ARRAY_F32_F32_GRAD
  : TEX_UNIFIED_1D_ARRAY_GRAD<"tex.grad.a1d.v4.f32.f32",
                              GPR32, GPR32>;
defm TEX_UNIFIED_1D_ARRAY_S32_F32_GRAD
  : TEX_UNIFIED_1D_ARRAY_GRAD<"tex.grad.a1d.v4.s32.f32",
                              GPR32, GPR32>;
defm TEX_UNIFIED_1D_ARRAY_U32_F32_GRAD
  : TEX_UNIFIED_1D_ARRAY_GRAD<"tex.grad.a1d.v4.u32.f32",
                              GPR32, GPR32>;

class TEX_UNIFIED_2D_base<string inst, RVGPURegClass outtype,
                          RVGPURegClass intype, dag tex>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$y)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x, $y\\}]",
                 []>;

multiclass TEX_UNIFIED_2D<string inst, RVGPURegClass outtype,
                          RVGPURegClass intype> {
  def _R : TEX_UNIFIED_2D_base<inst, outtype, intype, (ins GPR64:$t)>;
  def _I : TEX_UNIFIED_2D_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_2D_F32_S32
  : TEX_UNIFIED_2D<"tex.2d.v4.f32.s32", GPR32, GPR32>;
defm TEX_UNIFIED_2D_F32_F32
  : TEX_UNIFIED_2D<"tex.2d.v4.f32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_2D_S32_S32
  : TEX_UNIFIED_2D<"tex.2d.v4.s32.s32", GPR32, GPR32>;
defm TEX_UNIFIED_2D_S32_F32
  : TEX_UNIFIED_2D<"tex.2d.v4.s32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_2D_U32_S32
  : TEX_UNIFIED_2D<"tex.2d.v4.u32.s32", GPR32, GPR32>;
defm TEX_UNIFIED_2D_U32_F32
  : TEX_UNIFIED_2D<"tex.2d.v4.u32.f32", GPR32, GPR32>;

class TEX_UNIFIED_2D_LEVEL_base<string inst, RVGPURegClass outtype,
                                RVGPURegClass intype, dag tex>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$y, intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x, $y\\}], $lod",
                 []>;

multiclass TEX_UNIFIED_2D_LEVEL<string inst, RVGPURegClass outtype,
                                RVGPURegClass intype> {
  def _R : TEX_UNIFIED_2D_LEVEL_base<inst, outtype, intype, (ins GPR64:$t)>;
  def _I : TEX_UNIFIED_2D_LEVEL_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_2D_F32_F32_LEVEL
  : TEX_UNIFIED_2D_LEVEL<"tex.level.2d.v4.f32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_2D_S32_F32_LEVEL
  : TEX_UNIFIED_2D_LEVEL<"tex.level.2d.v4.s32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_2D_U32_F32_LEVEL
  : TEX_UNIFIED_2D_LEVEL<"tex.level.2d.v4.u32.f32", GPR32, GPR32>;

class TEX_UNIFIED_2D_GRAD_base<string inst, RVGPURegClass outtype,
                               RVGPURegClass intype, dag tex>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$y,
                                intype:$gradx0, intype:$gradx1,
                                intype:$grady0, intype:$grady1)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x, $y\\}],"
                        " \\{$gradx0, $gradx1\\}, \\{$grady0, $grady1\\}",
                 []>;
multiclass TEX_UNIFIED_2D_GRAD<string inst, RVGPURegClass outtype,
                               RVGPURegClass intype> {
  def _R : TEX_UNIFIED_2D_GRAD_base<inst, outtype, intype, (ins GPR64:$t)>;
  def _I : TEX_UNIFIED_2D_GRAD_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_2D_F32_F32_GRAD
  : TEX_UNIFIED_2D_GRAD<"tex.grad.2d.v4.f32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_2D_S32_F32_GRAD
  : TEX_UNIFIED_2D_GRAD<"tex.grad.2d.v4.s32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_2D_U32_F32_GRAD
  : TEX_UNIFIED_2D_GRAD<"tex.grad.2d.v4.u32.f32", GPR32, GPR32>;

class TEX_UNIFIED_2D_ARRAY_base<string inst, RVGPURegClass outtype,
                                RVGPURegClass intype, dag tex>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins GPR32:$l, intype:$x, intype:$y)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$l, $x, $y, $y\\}]",
                 []>;
multiclass TEX_UNIFIED_2D_ARRAY<string inst, RVGPURegClass outtype,
                                RVGPURegClass intype> {
  def _R : TEX_UNIFIED_2D_ARRAY_base<inst, outtype, intype, (ins GPR64:$t)>;
  def _I : TEX_UNIFIED_2D_ARRAY_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_2D_ARRAY_F32_S32
  : TEX_UNIFIED_2D_ARRAY<"tex.a2d.v4.f32.s32", GPR32, GPR32>;
defm TEX_UNIFIED_2D_ARRAY_F32_F32
  : TEX_UNIFIED_2D_ARRAY<"tex.a2d.v4.f32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_2D_ARRAY_S32_S32
  : TEX_UNIFIED_2D_ARRAY<"tex.a2d.v4.s32.s32", GPR32, GPR32>;
defm TEX_UNIFIED_2D_ARRAY_S32_F32
  : TEX_UNIFIED_2D_ARRAY<"tex.a2d.v4.s32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_2D_ARRAY_U32_S32
  : TEX_UNIFIED_2D_ARRAY<"tex.a2d.v4.u32.s32", GPR32, GPR32>;
defm TEX_UNIFIED_2D_ARRAY_U32_F32
  : TEX_UNIFIED_2D_ARRAY<"tex.a2d.v4.u32.f32", GPR32, GPR32>;

class TEX_UNIFIED_2D_ARRAY_LEVEL_base<string inst, RVGPURegClass outtype,
                                      RVGPURegClass intype, dag tex>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins GPR32:$l, intype:$x, intype:$y,
                                intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        "  [$t, \\{$l, $x, $y, $y\\}], $lod",
                 []>;
multiclass TEX_UNIFIED_2D_ARRAY_LEVEL<string inst, RVGPURegClass outtype,
                                      RVGPURegClass intype> {
  def _R : TEX_UNIFIED_2D_ARRAY_LEVEL_base<inst, outtype, intype,
                                           (ins GPR64:$t)>;
  def _I : TEX_UNIFIED_2D_ARRAY_LEVEL_base<inst, outtype, intype,
                                           (ins i64imm:$t)>;
}

defm TEX_UNIFIED_2D_ARRAY_F32_F32_LEVEL
  : TEX_UNIFIED_2D_ARRAY_LEVEL<"tex.level.a2d.v4.f32.f32",
                               GPR32, GPR32>;
defm TEX_UNIFIED_2D_ARRAY_S32_F32_LEVEL
  : TEX_UNIFIED_2D_ARRAY_LEVEL<"tex.level.a2d.v4.s32.f32",
                               GPR32, GPR32>;
defm TEX_UNIFIED_2D_ARRAY_U32_F32_LEVEL
  : TEX_UNIFIED_2D_ARRAY_LEVEL<"tex.level.a2d.v4.u32.f32",
                               GPR32, GPR32>;

class TEX_UNIFIED_2D_ARRAY_GRAD_base<string inst, RVGPURegClass outtype,
                                     RVGPURegClass intype, dag tex>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins GPR32:$l, intype:$x, intype:$y,
                                intype:$gradx0, intype:$gradx1,
                                intype:$grady0, intype:$grady1)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$l, $x, $y, $y\\}],"
                        " \\{$gradx0, $gradx1\\}, \\{$grady0, $grady1\\}",
                 []>;
multiclass TEX_UNIFIED_2D_ARRAY_GRAD<string inst, RVGPURegClass outtype,
                                     RVGPURegClass intype> {
  def _R : TEX_UNIFIED_2D_ARRAY_GRAD_base<inst, outtype, intype,
                                          (ins GPR64:$t)>;
  def _I : TEX_UNIFIED_2D_ARRAY_GRAD_base<inst, outtype, intype,
                                          (ins i64imm:$t)>;
}

defm TEX_UNIFIED_2D_ARRAY_F32_F32_GRAD
  : TEX_UNIFIED_2D_ARRAY_GRAD<"tex.grad.a2d.v4.f32.f32",
                              GPR32, GPR32>;
defm TEX_UNIFIED_2D_ARRAY_S32_F32_GRAD
  : TEX_UNIFIED_2D_ARRAY_GRAD<"tex.grad.a2d.v4.s32.f32",
                              GPR32, GPR32>;
defm TEX_UNIFIED_2D_ARRAY_U32_F32_GRAD
  : TEX_UNIFIED_2D_ARRAY_GRAD<"tex.grad.a2d.v4.u32.f32",
                              GPR32, GPR32>;

class TEX_UNIFIED_3D_base<string inst, RVGPURegClass outtype,
                          RVGPURegClass intype, dag tex>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$y, intype:$z)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x, $y, $z, $z\\}]",
                 []>;
multiclass TEX_UNIFIED_3D<string inst, RVGPURegClass outtype,
                          RVGPURegClass intype> {
  def _R : TEX_UNIFIED_3D_base<inst, outtype, intype, (ins GPR64:$t)>;
  def _I : TEX_UNIFIED_3D_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_3D_F32_S32
  : TEX_UNIFIED_3D<"tex.3d.v4.f32.s32", GPR32, GPR32>;
defm TEX_UNIFIED_3D_F32_F32
  : TEX_UNIFIED_3D<"tex.3d.v4.f32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_3D_S32_S32
  : TEX_UNIFIED_3D<"tex.3d.v4.s32.s32", GPR32, GPR32>;
defm TEX_UNIFIED_3D_S32_F32
  : TEX_UNIFIED_3D<"tex.3d.v4.s32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_3D_U32_S32
  : TEX_UNIFIED_3D<"tex.3d.v4.u32.s32", GPR32, GPR32>;
defm TEX_UNIFIED_3D_U32_F32
  : TEX_UNIFIED_3D<"tex.3d.v4.u32.f32", GPR32, GPR32>;

class TEX_UNIFIED_3D_LEVEL_base<string inst, RVGPURegClass outtype,
                                RVGPURegClass intype, dag tex>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$y, intype:$z, intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, \\{$x, $y, $z, $z\\}], $lod",
                 []>;
multiclass TEX_UNIFIED_3D_LEVEL<string inst, RVGPURegClass outtype,
                                RVGPURegClass intype> {
  def _R : TEX_UNIFIED_3D_LEVEL_base<inst, outtype, intype, (ins GPR64:$t)>;
  def _I : TEX_UNIFIED_3D_LEVEL_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_3D_F32_F32_LEVEL
  : TEX_UNIFIED_3D_LEVEL<"tex.level.3d.v4.f32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_3D_S32_F32_LEVEL
  : TEX_UNIFIED_3D_LEVEL<"tex.level.3d.v4.s32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_3D_U32_F32_LEVEL
  : TEX_UNIFIED_3D_LEVEL<"tex.level.3d.v4.u32.f32", GPR32, GPR32>;

class TEX_UNIFIED_3D_GRAD_base<string inst, RVGPURegClass outtype,
                               RVGPURegClass intype, dag tex>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$y, intype:$z,
                                intype:$gradx0, intype:$gradx1,
                                intype:$gradx2, intype:$grady0,
                                intype:$grady1, intype:$grady2)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x, $y, $z, $z\\}],"
                        " \\{$gradx0, $gradx1, $gradx2, $gradx2\\},"
                        " \\{$grady0, $grady1, $grady2, $grady2\\}",
                 []>;
multiclass TEX_UNIFIED_3D_GRAD<string inst, RVGPURegClass outtype,
                               RVGPURegClass intype> {
  def _R : TEX_UNIFIED_3D_GRAD_base<inst, outtype, intype, (ins GPR64:$t)>;
  def _I : TEX_UNIFIED_3D_GRAD_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_3D_F32_F32_GRAD
  : TEX_UNIFIED_3D_GRAD<"tex.grad.3d.v4.f32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_3D_S32_F32_GRAD
  : TEX_UNIFIED_3D_GRAD<"tex.grad.3d.v4.s32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_3D_U32_F32_GRAD
  : TEX_UNIFIED_3D_GRAD<"tex.grad.3d.v4.u32.f32", GPR32, GPR32>;

class TEX_UNIFIED_CUBE_base<string inst, RVGPURegClass outtype,
                            RVGPURegClass intype, dag tex>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$y, intype:$z)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x, $y, $z, $z\\}]",
                 []>;
multiclass TEX_UNIFIED_CUBE<string inst, RVGPURegClass outtype,
                            RVGPURegClass intype> {
  def _R : TEX_UNIFIED_CUBE_base<inst, outtype, intype, (ins GPR64:$t)>;
  def _I : TEX_UNIFIED_CUBE_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_CUBE_F32_F32
  : TEX_UNIFIED_CUBE<"tex.cube.v4.f32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_CUBE_S32_F32
  : TEX_UNIFIED_CUBE<"tex.cube.v4.s32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_CUBE_U32_F32
  : TEX_UNIFIED_CUBE<"tex.cube.v4.u32.f32", GPR32, GPR32>;

class TEX_UNIFIED_CUBE_LEVEL_base<string inst, RVGPURegClass outtype,
                                  RVGPURegClass intype, dag tex>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$y, intype:$z, intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, \\{$x, $y, $z, $z\\}], $lod",
                 []>;
multiclass TEX_UNIFIED_CUBE_LEVEL<string inst, RVGPURegClass outtype,
                                  RVGPURegClass intype> {
  def _R : TEX_UNIFIED_CUBE_LEVEL_base<inst, outtype, intype,
                                       (ins GPR64:$t)>;
  def _I : TEX_UNIFIED_CUBE_LEVEL_base<inst, outtype, intype,
                                       (ins i64imm:$t)>;
}

defm TEX_UNIFIED_CUBE_F32_F32_LEVEL
  : TEX_UNIFIED_CUBE_LEVEL<"tex.level.cube.v4.f32.f32",
                           GPR32, GPR32>;
defm TEX_UNIFIED_CUBE_S32_F32_LEVEL
  : TEX_UNIFIED_CUBE_LEVEL<"tex.level.cube.v4.s32.f32",
                           GPR32, GPR32>;
defm TEX_UNIFIED_CUBE_U32_F32_LEVEL
  : TEX_UNIFIED_CUBE_LEVEL<"tex.level.cube.v4.u32.f32",
                           GPR32, GPR32>;

class TEX_UNIFIED_CUBE_ARRAY_base<string inst, RVGPURegClass outtype,
                                  RVGPURegClass intype, dag tex>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins GPR32:$l, intype:$x, intype:$y, intype:$z)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$l, $x, $y, $z\\}]",
                 []>;
multiclass TEX_UNIFIED_CUBE_ARRAY<string inst, RVGPURegClass outtype,
                                  RVGPURegClass intype> {
  def _R : TEX_UNIFIED_CUBE_ARRAY_base<inst, outtype, intype,
                                       (ins GPR64:$t)>;
  def _I : TEX_UNIFIED_CUBE_ARRAY_base<inst, outtype, intype,
                                       (ins i64imm:$t)>;
}

defm TEX_UNIFIED_CUBE_ARRAY_F32_F32
  : TEX_UNIFIED_CUBE_ARRAY<"tex.acube.v4.f32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_CUBE_ARRAY_S32_F32
  : TEX_UNIFIED_CUBE_ARRAY<"tex.acube.v4.s32.f32", GPR32, GPR32>;
defm TEX_UNIFIED_CUBE_ARRAY_U32_F32
  : TEX_UNIFIED_CUBE_ARRAY<"tex.acube.v4.u32.f32", GPR32, GPR32>;

class TEX_UNIFIED_CUBE_ARRAY_LEVEL_base<string inst, RVGPURegClass outtype,
                                        RVGPURegClass intype, dag tex>
    : RVGPUInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins GPR32:$l, intype:$x, intype:$y, intype:$z,
                                intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, \\{$l, $x, $y, $z\\}], $lod",
                 []>;
multiclass TEX_UNIFIED_CUBE_ARRAY_LEVEL<string inst, RVGPURegClass outtype,
                                        RVGPURegClass intype> {
  def _R : TEX_UNIFIED_CUBE_ARRAY_LEVEL_base<inst, outtype, intype,
                                             (ins GPR64:$t)>;
  def _I : TEX_UNIFIED_CUBE_ARRAY_LEVEL_base<inst, outtype, intype,
                                             (ins i64imm:$t)>;
}

defm TEX_UNIFIED_CUBE_ARRAY_F32_F32_LEVEL
  : TEX_UNIFIED_CUBE_ARRAY_LEVEL<"tex.level.acube.v4.f32.f32",
                                 GPR32, GPR32>;
defm TEX_UNIFIED_CUBE_ARRAY_S32_F32_LEVEL
  : TEX_UNIFIED_CUBE_ARRAY_LEVEL<"tex.level.acube.v4.s32.f32",
                                 GPR32, GPR32>;
defm TEX_UNIFIED_CUBE_ARRAY_U32_F32_LEVEL
  : TEX_UNIFIED_CUBE_ARRAY_LEVEL<"tex.level.acube.v4.u32.f32",
                                 GPR32, GPR32>;

class TLD4_UNIFIED_2D_base<string inst, RVGPURegClass outtype,
                           RVGPURegClass intype, dag tex>
    : RVGPUInst<(outs outtype:$v0, outtype:$v1,
                      outtype:$v2, outtype:$v3),
                 !con(tex, (ins intype:$x, intype:$y)),
                 inst # " \t\\{$v0, $v1, $v2, $v3\\}, [$t, \\{$x, $y\\}]",
                 []>;
multiclass TLD4_UNIFIED_2D<string inst, RVGPURegClass outtype,
                           RVGPURegClass intype> {
  def _R : TLD4_UNIFIED_2D_base<inst, outtype, intype, (ins GPR64:$t)>;
  def _I : TLD4_UNIFIED_2D_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TLD4_UNIFIED_R_2D_F32_F32
  : TLD4_UNIFIED_2D<"tld4.r.2d.v4.f32.f32", GPR32, GPR32>;
defm TLD4_UNIFIED_G_2D_F32_F32
  : TLD4_UNIFIED_2D<"tld4.g.2d.v4.f32.f32", GPR32, GPR32>;
defm TLD4_UNIFIED_B_2D_F32_F32
  : TLD4_UNIFIED_2D<"tld4.b.2d.v4.f32.f32", GPR32, GPR32>;
defm TLD4_UNIFIED_A_2D_F32_F32
  : TLD4_UNIFIED_2D<"tld4.a.2d.v4.f32.f32", GPR32, GPR32>;

defm TLD4_UNIFIED_R_2D_S32_F32
  : TLD4_UNIFIED_2D<"tld4.r.2d.v4.s32.f32", GPR32, GPR32>;
defm TLD4_UNIFIED_G_2D_S32_F32
  : TLD4_UNIFIED_2D<"tld4.g.2d.v4.s32.f32", GPR32, GPR32>;
defm TLD4_UNIFIED_B_2D_S32_F32
  : TLD4_UNIFIED_2D<"tld4.b.2d.v4.s32.f32", GPR32, GPR32>;
defm TLD4_UNIFIED_A_2D_S32_F32
  : TLD4_UNIFIED_2D<"tld4.a.2d.v4.s32.f32", GPR32, GPR32>;

defm TLD4_UNIFIED_R_2D_U32_F32
  : TLD4_UNIFIED_2D<"tld4.r.2d.v4.u32.f32", GPR32, GPR32>;
defm TLD4_UNIFIED_G_2D_U32_F32
  : TLD4_UNIFIED_2D<"tld4.g.2d.v4.u32.f32", GPR32, GPR32>;
defm TLD4_UNIFIED_B_2D_U32_F32
  : TLD4_UNIFIED_2D<"tld4.b.2d.v4.u32.f32", GPR32, GPR32>;
defm TLD4_UNIFIED_A_2D_U32_F32
  : TLD4_UNIFIED_2D<"tld4.a.2d.v4.u32.f32", GPR32, GPR32>;

}



//=== Surface load instructions

let IsSuld = true in {

class SULD_1D_base<string inst, RVGPURegClass outtype, dag surf>
    : RVGPUInst<(outs outtype:$r),
                !con(surf, (ins GPR32:$x)),
                inst # " \\{$r\\}, [$s, \\{$x\\}]",
                []>;
multiclass SULD_1D<string inst, RVGPURegClass outtype> {
  def _R : SULD_1D_base<inst, outtype, (ins GPR64:$s)>;
  def _I : SULD_1D_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_1D_I8_CLAMP : SULD_1D<"suld.b.1d.b8.clamp", GPR16>;
defm SULD_1D_I16_CLAMP : SULD_1D<"suld.b.1d.b16.clamp", GPR16>;
defm SULD_1D_I32_CLAMP : SULD_1D<"suld.b.1d.b32.clamp", GPR32>;
defm SULD_1D_I64_CLAMP : SULD_1D<"suld.b.1d.b64.clamp", GPR64>;

defm SULD_1D_I8_TRAP : SULD_1D<"suld.b.1d.b8.trap", GPR16>;
defm SULD_1D_I16_TRAP : SULD_1D<"suld.b.1d.b16.trap", GPR16>;
defm SULD_1D_I32_TRAP : SULD_1D<"suld.b.1d.b32.trap", GPR32>;
defm SULD_1D_I64_TRAP : SULD_1D<"suld.b.1d.b64.trap", GPR64>;

defm SULD_1D_I8_ZERO : SULD_1D<"suld.b.1d.b8.zero", GPR16>;
defm SULD_1D_I16_ZERO : SULD_1D<"suld.b.1d.b16.zero", GPR16>;
defm SULD_1D_I32_ZERO : SULD_1D<"suld.b.1d.b32.zero", GPR32>;
defm SULD_1D_I64_ZERO : SULD_1D<"suld.b.1d.b64.zero", GPR64>;

class SULD_1D_ARRAY_base<string inst, RVGPURegClass outtype, dag surf>
    : RVGPUInst<(outs outtype:$r),
                !con(surf, (ins GPR32:$l, GPR32:$x)),
                inst # " \\{$r\\}, [$s, \\{$l, $x\\}]",
                []>;
multiclass SULD_1D_ARRAY<string inst, RVGPURegClass outtype> {
  def _R : SULD_1D_ARRAY_base<inst, outtype, (ins GPR64:$s)>;
  def _I : SULD_1D_ARRAY_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_1D_ARRAY_I8_CLAMP
  : SULD_1D_ARRAY<"suld.b.a1d.b8.clamp", GPR16>;
defm SULD_1D_ARRAY_I16_CLAMP
  : SULD_1D_ARRAY<"suld.b.a1d.b16.clamp", GPR16>;
defm SULD_1D_ARRAY_I32_CLAMP
  : SULD_1D_ARRAY<"suld.b.a1d.b32.clamp", GPR32>;
defm SULD_1D_ARRAY_I64_CLAMP
  : SULD_1D_ARRAY<"suld.b.a1d.b64.clamp", GPR64>;

defm SULD_1D_ARRAY_I8_TRAP
  : SULD_1D_ARRAY<"suld.b.a1d.b8.trap", GPR16>;
defm SULD_1D_ARRAY_I16_TRAP
  : SULD_1D_ARRAY<"suld.b.a1d.b16.trap", GPR16>;
defm SULD_1D_ARRAY_I32_TRAP
  : SULD_1D_ARRAY<"suld.b.a1d.b32.trap", GPR32>;
defm SULD_1D_ARRAY_I64_TRAP
  : SULD_1D_ARRAY<"suld.b.a1d.b64.trap", GPR64>;

defm SULD_1D_ARRAY_I8_ZERO
  : SULD_1D_ARRAY<"suld.b.a1d.b8.zero", GPR16>;
defm SULD_1D_ARRAY_I16_ZERO
  : SULD_1D_ARRAY<"suld.b.a1d.b16.zero", GPR16>;
defm SULD_1D_ARRAY_I32_ZERO
  : SULD_1D_ARRAY<"suld.b.a1d.b32.zero", GPR32>;
defm SULD_1D_ARRAY_I64_ZERO
  : SULD_1D_ARRAY<"suld.b.a1d.b64.zero", GPR64>;

class SULD_2D_base<string inst, RVGPURegClass outtype, dag surf>
    : RVGPUInst<(outs outtype:$r),
                !con(surf, (ins GPR32:$x, GPR32:$y)),
                inst # " \\{$r\\}, [$s, \\{$x, $y\\}]",
                []>;
multiclass SULD_2D<string inst, RVGPURegClass outtype> {
  def _R : SULD_2D_base<inst, outtype, (ins GPR64:$s)>;
  def _I : SULD_2D_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_2D_I8_CLAMP : SULD_2D<"suld.b.2d.b8.clamp", GPR16>;
defm SULD_2D_I16_CLAMP : SULD_2D<"suld.b.2d.b16.clamp", GPR16>;
defm SULD_2D_I32_CLAMP : SULD_2D<"suld.b.2d.b32.clamp", GPR32>;
defm SULD_2D_I64_CLAMP : SULD_2D<"suld.b.2d.b64.clamp", GPR64>;

defm SULD_2D_I8_TRAP : SULD_2D<"suld.b.2d.b8.trap", GPR16>;
defm SULD_2D_I16_TRAP : SULD_2D<"suld.b.2d.b16.trap", GPR16>;
defm SULD_2D_I32_TRAP : SULD_2D<"suld.b.2d.b32.trap", GPR32>;
defm SULD_2D_I64_TRAP : SULD_2D<"suld.b.2d.b64.trap", GPR64>;

defm SULD_2D_I8_ZERO : SULD_2D<"suld.b.2d.b8.zero", GPR16>;
defm SULD_2D_I16_ZERO : SULD_2D<"suld.b.2d.b16.zero", GPR16>;
defm SULD_2D_I32_ZERO : SULD_2D<"suld.b.2d.b32.zero", GPR32>;
defm SULD_2D_I64_ZERO : SULD_2D<"suld.b.2d.b64.zero", GPR64>;

class SULD_2D_ARRAY_base<string inst, RVGPURegClass outtype, dag surf>
    : RVGPUInst<(outs outtype:$r),
                !con(surf, (ins GPR32:$l, GPR32:$x, GPR32:$y)),
                inst # " \\{$r\\}, [$s, \\{$l, $x, $y, $y\\}]",
                []>;
multiclass SULD_2D_ARRAY<string inst, RVGPURegClass outtype> {
  def _R : SULD_2D_ARRAY_base<inst, outtype, (ins GPR64:$s)>;
  def _I : SULD_2D_ARRAY_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_2D_ARRAY_I8_CLAMP : SULD_2D_ARRAY<"suld.b.a2d.b8.clamp", GPR16>;
defm SULD_2D_ARRAY_I16_CLAMP : SULD_2D_ARRAY<"suld.b.a2d.b16.clamp", GPR16>;
defm SULD_2D_ARRAY_I32_CLAMP : SULD_2D_ARRAY<"suld.b.a2d.b32.clamp", GPR32>;
defm SULD_2D_ARRAY_I64_CLAMP : SULD_2D_ARRAY<"suld.b.a2d.b64.clamp", GPR64>;

defm SULD_2D_ARRAY_I8_TRAP : SULD_2D_ARRAY<"suld.b.a2d.b8.trap", GPR16>;
defm SULD_2D_ARRAY_I16_TRAP : SULD_2D_ARRAY<"suld.b.a2d.b16.trap", GPR16>;
defm SULD_2D_ARRAY_I32_TRAP : SULD_2D_ARRAY<"suld.b.a2d.b32.trap", GPR32>;
defm SULD_2D_ARRAY_I64_TRAP : SULD_2D_ARRAY<"suld.b.a2d.b64.trap", GPR64>;

defm SULD_2D_ARRAY_I8_ZERO : SULD_2D_ARRAY<"suld.b.a2d.b8.zero", GPR16>;
defm SULD_2D_ARRAY_I16_ZERO : SULD_2D_ARRAY<"suld.b.a2d.b16.zero", GPR16>;
defm SULD_2D_ARRAY_I32_ZERO : SULD_2D_ARRAY<"suld.b.a2d.b32.zero", GPR32>;
defm SULD_2D_ARRAY_I64_ZERO : SULD_2D_ARRAY<"suld.b.a2d.b64.zero", GPR64>;

class SULD_3D_base<string inst, RVGPURegClass outtype, dag surf>
    : RVGPUInst<(outs outtype:$r),
                !con(surf, (ins GPR32:$x, GPR32:$y, GPR32:$z)),
                inst # " \\{$r\\}, [$s, \\{$x, $y, $z, $z\\}]",
                []>;
multiclass SULD_3D<string inst, RVGPURegClass outtype> {
  def _R : SULD_3D_base<inst, outtype, (ins GPR64:$s)>;
  def _I : SULD_3D_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_3D_I8_CLAMP : SULD_3D<"suld.b.3d.b8.clamp", GPR16>;
defm SULD_3D_I16_CLAMP : SULD_3D<"suld.b.3d.b16.clamp", GPR16>;
defm SULD_3D_I32_CLAMP : SULD_3D<"suld.b.3d.b32.clamp", GPR32>;
defm SULD_3D_I64_CLAMP : SULD_3D<"suld.b.3d.b64.clamp", GPR64>;

defm SULD_3D_I8_TRAP : SULD_3D<"suld.b.3d.b8.trap", GPR16>;
defm SULD_3D_I16_TRAP : SULD_3D<"suld.b.3d.b16.trap", GPR16>;
defm SULD_3D_I32_TRAP : SULD_3D<"suld.b.3d.b32.trap", GPR32>;
defm SULD_3D_I64_TRAP : SULD_3D<"suld.b.3d.b64.trap", GPR64>;

defm SULD_3D_I8_ZERO : SULD_3D<"suld.b.3d.b8.zero", GPR16>;
defm SULD_3D_I16_ZERO : SULD_3D<"suld.b.3d.b16.zero", GPR16>;
defm SULD_3D_I32_ZERO : SULD_3D<"suld.b.3d.b32.zero", GPR32>;
defm SULD_3D_I64_ZERO : SULD_3D<"suld.b.3d.b64.zero", GPR64>;
}

let IsSuld = 2 in {

class SULD_1D_V2_base<string inst, RVGPURegClass outtype, dag surf>
    : RVGPUInst<(outs outtype:$r, outtype:$g),
                !con(surf, (ins GPR32:$x)),
                inst # " \\{$r, $g\\}, [$s, \\{$x\\}]",
                []>;
multiclass SULD_1D_V2<string inst, RVGPURegClass outtype> {
  def _R : SULD_1D_V2_base<inst, outtype, (ins GPR64:$s)>;
  def _I : SULD_1D_V2_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_1D_V2I8_CLAMP : SULD_1D_V2<"suld.b.1d.v2.b8.clamp", GPR16>;
defm SULD_1D_V2I16_CLAMP : SULD_1D_V2<"suld.b.1d.v2.b16.clamp", GPR16>;
defm SULD_1D_V2I32_CLAMP : SULD_1D_V2<"suld.b.1d.v2.b32.clamp", GPR32>;
defm SULD_1D_V2I64_CLAMP : SULD_1D_V2<"suld.b.1d.v2.b64.clamp", GPR64>;

defm SULD_1D_V2I8_TRAP : SULD_1D_V2<"suld.b.1d.v2.b8.trap", GPR16>;
defm SULD_1D_V2I16_TRAP : SULD_1D_V2<"suld.b.1d.v2.b16.trap", GPR16>;
defm SULD_1D_V2I32_TRAP : SULD_1D_V2<"suld.b.1d.v2.b32.trap", GPR32>;
defm SULD_1D_V2I64_TRAP : SULD_1D_V2<"suld.b.1d.v2.b64.trap", GPR64>;

defm SULD_1D_V2I8_ZERO : SULD_1D_V2<"suld.b.1d.v2.b8.zero", GPR16>;
defm SULD_1D_V2I16_ZERO : SULD_1D_V2<"suld.b.1d.v2.b16.zero", GPR16>;
defm SULD_1D_V2I32_ZERO : SULD_1D_V2<"suld.b.1d.v2.b32.zero", GPR32>;
defm SULD_1D_V2I64_ZERO : SULD_1D_V2<"suld.b.1d.v2.b64.zero", GPR64>;

class SULD_1D_ARRAY_V2_base<string inst, RVGPURegClass outtype, dag surf>
    : RVGPUInst<(outs outtype:$r, outtype:$g),
                !con(surf, (ins GPR32:$l, GPR32:$x)),
                inst # " \\{$r, $g\\}, [$s, \\{$l, $x\\}]",
                []>;
multiclass SULD_1D_ARRAY_V2<string inst, RVGPURegClass outtype> {
  def _R : SULD_1D_ARRAY_V2_base<inst, outtype, (ins GPR64:$s)>;
  def _I : SULD_1D_ARRAY_V2_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_1D_ARRAY_V2I8_CLAMP
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b8.clamp", GPR16>;
defm SULD_1D_ARRAY_V2I16_CLAMP
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b16.clamp", GPR16>;
defm SULD_1D_ARRAY_V2I32_CLAMP
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b32.clamp", GPR32>;
defm SULD_1D_ARRAY_V2I64_CLAMP
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b64.clamp", GPR64>;

defm SULD_1D_ARRAY_V2I8_TRAP
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b8.trap", GPR16>;
defm SULD_1D_ARRAY_V2I16_TRAP
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b16.trap", GPR16>;
defm SULD_1D_ARRAY_V2I32_TRAP
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b32.trap", GPR32>;
defm SULD_1D_ARRAY_V2I64_TRAP
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b64.trap", GPR64>;

defm SULD_1D_ARRAY_V2I8_ZERO
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b8.zero", GPR16>;
defm SULD_1D_ARRAY_V2I16_ZERO
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b16.zero", GPR16>;
defm SULD_1D_ARRAY_V2I32_ZERO
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b32.zero", GPR32>;
defm SULD_1D_ARRAY_V2I64_ZERO
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b64.zero", GPR64>;

class SULD_2D_V2_base<string inst, RVGPURegClass outtype, dag surf>
    : RVGPUInst<(outs outtype:$r, outtype:$g),
                !con(surf, (ins GPR32:$x, GPR32:$y)),
                inst # " \\{$r, $g\\}, [$s, \\{$x, $y\\}]",
                []>;
multiclass SULD_2D_V2<string inst, RVGPURegClass outtype> {
  def _R : SULD_2D_V2_base<inst, outtype, (ins GPR64:$s)>;
  def _I : SULD_2D_V2_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_2D_V2I8_CLAMP
  : SULD_2D_V2<"suld.b.2d.v2.b8.clamp", GPR16>;
defm SULD_2D_V2I16_CLAMP
  : SULD_2D_V2<"suld.b.2d.v2.b16.clamp", GPR16>;
defm SULD_2D_V2I32_CLAMP
  : SULD_2D_V2<"suld.b.2d.v2.b32.clamp", GPR32>;
defm SULD_2D_V2I64_CLAMP
  : SULD_2D_V2<"suld.b.2d.v2.b64.clamp", GPR64>;

defm SULD_2D_V2I8_TRAP
  : SULD_2D_V2<"suld.b.2d.v2.b8.trap", GPR16>;
defm SULD_2D_V2I16_TRAP
  : SULD_2D_V2<"suld.b.2d.v2.b16.trap", GPR16>;
defm SULD_2D_V2I32_TRAP
  : SULD_2D_V2<"suld.b.2d.v2.b32.trap", GPR32>;
defm SULD_2D_V2I64_TRAP
  : SULD_2D_V2<"suld.b.2d.v2.b64.trap", GPR64>;

defm SULD_2D_V2I8_ZERO
  : SULD_2D_V2<"suld.b.2d.v2.b8.zero", GPR16>;
defm SULD_2D_V2I16_ZERO
  : SULD_2D_V2<"suld.b.2d.v2.b16.zero", GPR16>;
defm SULD_2D_V2I32_ZERO
  : SULD_2D_V2<"suld.b.2d.v2.b32.zero", GPR32>;
defm SULD_2D_V2I64_ZERO
  : SULD_2D_V2<"suld.b.2d.v2.b64.zero", GPR64>;

class SULD_2D_ARRAY_V2_base<string inst, RVGPURegClass outtype, dag surf>
    : RVGPUInst<(outs outtype:$r, outtype:$g),
                !con(surf, (ins GPR32:$l, GPR32:$x, GPR32:$y)),
                inst # " \\{$r, $g\\}, [$s, \\{$l, $x, $y, $y\\}]",
                []>;
multiclass SULD_2D_ARRAY_V2<string inst, RVGPURegClass outtype> {
  def _R : SULD_2D_ARRAY_V2_base<inst, outtype, (ins GPR64:$s)>;
  def _I : SULD_2D_ARRAY_V2_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_2D_ARRAY_V2I8_CLAMP
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b8.clamp", GPR16>;
defm SULD_2D_ARRAY_V2I16_CLAMP
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b16.clamp", GPR16>;
defm SULD_2D_ARRAY_V2I32_CLAMP
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b32.clamp", GPR32>;
defm SULD_2D_ARRAY_V2I64_CLAMP
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b64.clamp", GPR64>;

defm SULD_2D_ARRAY_V2I8_TRAP
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b8.trap", GPR16>;
defm SULD_2D_ARRAY_V2I16_TRAP
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b16.trap", GPR16>;
defm SULD_2D_ARRAY_V2I32_TRAP
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b32.trap", GPR32>;
defm SULD_2D_ARRAY_V2I64_TRAP
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b64.trap", GPR64>;

defm SULD_2D_ARRAY_V2I8_ZERO
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b8.zero", GPR16>;
defm SULD_2D_ARRAY_V2I16_ZERO
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b16.zero", GPR16>;
defm SULD_2D_ARRAY_V2I32_ZERO
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b32.zero", GPR32>;
defm SULD_2D_ARRAY_V2I64_ZERO
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b64.zero", GPR64>;

class SULD_3D_V2_base<string inst, RVGPURegClass outtype, dag surf>
    : RVGPUInst<(outs outtype:$r, outtype:$g),
                !con(surf, (ins GPR32:$x, GPR32:$y, GPR32:$z)),
                inst # " \\{$r, $g\\}, [$s, \\{$x, $y, $z, $z\\}]",
                []>;
multiclass SULD_3D_V2<string inst, RVGPURegClass outtype> {
  def _R : SULD_3D_V2_base<inst, outtype, (ins GPR64:$s)>;
  def _I : SULD_3D_V2_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_3D_V2I8_CLAMP : SULD_3D_V2<"suld.b.3d.v2.b8.clamp", GPR16>;
defm SULD_3D_V2I16_CLAMP : SULD_3D_V2<"suld.b.3d.v2.b16.clamp", GPR16>;
defm SULD_3D_V2I32_CLAMP : SULD_3D_V2<"suld.b.3d.v2.b32.clamp", GPR32>;
defm SULD_3D_V2I64_CLAMP : SULD_3D_V2<"suld.b.3d.v2.b64.clamp", GPR64>;

defm SULD_3D_V2I8_TRAP : SULD_3D_V2<"suld.b.3d.v2.b8.trap", GPR16>;
defm SULD_3D_V2I16_TRAP : SULD_3D_V2<"suld.b.3d.v2.b16.trap", GPR16>;
defm SULD_3D_V2I32_TRAP : SULD_3D_V2<"suld.b.3d.v2.b32.trap", GPR32>;
defm SULD_3D_V2I64_TRAP : SULD_3D_V2<"suld.b.3d.v2.b64.trap", GPR64>;

defm SULD_3D_V2I8_ZERO : SULD_3D_V2<"suld.b.3d.v2.b8.zero", GPR16>;
defm SULD_3D_V2I16_ZERO : SULD_3D_V2<"suld.b.3d.v2.b16.zero", GPR16>;
defm SULD_3D_V2I32_ZERO : SULD_3D_V2<"suld.b.3d.v2.b32.zero", GPR32>;
defm SULD_3D_V2I64_ZERO : SULD_3D_V2<"suld.b.3d.v2.b64.zero", GPR64>;

}

let IsSuld = 3 in {

class SULD_1D_V4_base<string inst, RVGPURegClass outtype, dag surf>
    : RVGPUInst<(outs outtype:$r, outtype:$g, outtype:$b, outtype:$a),
                !con(surf, (ins GPR32:$x)),
                inst # " \\{$r, $g, $b, $a\\}, [$s, \\{$x\\}]",
                []>;
multiclass SULD_1D_V4<string inst, RVGPURegClass outtype> {
  def _R : SULD_1D_V4_base<inst, outtype, (ins GPR64:$s)>;
  def _I : SULD_1D_V4_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_1D_V4I8_CLAMP : SULD_1D_V4<"suld.b.1d.v4.b8.clamp", GPR16>;
defm SULD_1D_V4I16_CLAMP : SULD_1D_V4<"suld.b.1d.v4.b16.clamp", GPR16>;
defm SULD_1D_V4I32_CLAMP : SULD_1D_V4<"suld.b.1d.v4.b32.clamp", GPR32>;

defm SULD_1D_V4I8_TRAP : SULD_1D_V4<"suld.b.1d.v4.b8.trap", GPR16>;
defm SULD_1D_V4I16_TRAP : SULD_1D_V4<"suld.b.1d.v4.b16.trap", GPR16>;
defm SULD_1D_V4I32_TRAP : SULD_1D_V4<"suld.b.1d.v4.b32.trap", GPR32>;

defm SULD_1D_V4I8_ZERO : SULD_1D_V4<"suld.b.1d.v4.b8.zero", GPR16>;
defm SULD_1D_V4I16_ZERO : SULD_1D_V4<"suld.b.1d.v4.b16.zero", GPR16>;
defm SULD_1D_V4I32_ZERO : SULD_1D_V4<"suld.b.1d.v4.b32.zero", GPR32>;

class SULD_1D_ARRAY_V4_base<string inst, RVGPURegClass outtype, dag surf>
    : RVGPUInst<(outs outtype:$r, outtype:$g, outtype:$b, outtype:$a),
                !con(surf, (ins GPR32:$l, GPR32:$x)),
                inst # " \\{$r, $g, $b, $a\\}, [$s, \\{$l, $x\\}]",
                []>;
multiclass SULD_1D_ARRAY_V4<string inst, RVGPURegClass outtype> {
  def _R : SULD_1D_ARRAY_V4_base<inst, outtype, (ins GPR64:$s)>;
  def _I : SULD_1D_ARRAY_V4_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_1D_ARRAY_V4I8_CLAMP
  : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b8.clamp", GPR16>;
defm SULD_1D_ARRAY_V4I16_CLAMP
  : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b16.clamp", GPR16>;
defm SULD_1D_ARRAY_V4I32_CLAMP
  : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b32.clamp", GPR32>;

defm SULD_1D_ARRAY_V4I8_TRAP
  : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b8.trap", GPR16>;
defm SULD_1D_ARRAY_V4I16_TRAP
  : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b16.trap", GPR16>;
defm SULD_1D_ARRAY_V4I32_TRAP
  : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b32.trap", GPR32>;

defm SULD_1D_ARRAY_V4I8_ZERO
  : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b8.zero", GPR16>;
defm SULD_1D_ARRAY_V4I16_ZERO
  : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b16.zero", GPR16>;
defm SULD_1D_ARRAY_V4I32_ZERO
  : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b32.zero", GPR32>;

class SULD_2D_V4_base<string inst, RVGPURegClass outtype, dag surf>
    : RVGPUInst<(outs outtype:$r, outtype:$g, outtype:$b, outtype:$a),
                !con(surf, (ins GPR32:$x, GPR32:$y)),
                inst # " \\{$r, $g, $b, $a\\}, [$s, \\{$x, $y\\}]",
                []>;
multiclass SULD_2D_V4<string inst, RVGPURegClass outtype> {
  def _R : SULD_2D_V4_base<inst, outtype, (ins GPR64:$s)>;
  def _I : SULD_2D_V4_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_2D_V4I8_CLAMP : SULD_2D_V4<"suld.b.2d.v4.b8.clamp", GPR16>;
defm SULD_2D_V4I16_CLAMP : SULD_2D_V4<"suld.b.2d.v4.b16.clamp", GPR16>;
defm SULD_2D_V4I32_CLAMP : SULD_2D_V4<"suld.b.2d.v4.b32.clamp", GPR32>;

defm SULD_2D_V4I8_TRAP : SULD_2D_V4<"suld.b.2d.v4.b8.trap", GPR16>;
defm SULD_2D_V4I16_TRAP : SULD_2D_V4<"suld.b.2d.v4.b16.trap", GPR16>;
defm SULD_2D_V4I32_TRAP : SULD_2D_V4<"suld.b.2d.v4.b32.trap", GPR32>;

defm SULD_2D_V4I8_ZERO : SULD_2D_V4<"suld.b.2d.v4.b8.zero", GPR16>;
defm SULD_2D_V4I16_ZERO : SULD_2D_V4<"suld.b.2d.v4.b16.zero", GPR16>;
defm SULD_2D_V4I32_ZERO : SULD_2D_V4<"suld.b.2d.v4.b32.zero", GPR32>;

class SULD_2D_ARRAY_V4_base<string inst, RVGPURegClass outtype, dag surf>
    : RVGPUInst<(outs outtype:$r, outtype:$g, outtype:$b, outtype:$a),
                !con(surf, (ins GPR32:$l, GPR32:$x, GPR32:$y)),
                inst # " \\{$r, $g, $b, $a\\}, [$s, \\{$l, $x, $y, $y\\}]",
                []>;
multiclass SULD_2D_ARRAY_V4<string inst, RVGPURegClass outtype> {
  def _R : SULD_2D_ARRAY_V4_base<inst, outtype, (ins GPR64:$s)>;
  def _I : SULD_2D_ARRAY_V4_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_2D_ARRAY_V4I8_CLAMP
  : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b8.clamp", GPR16>;
defm SULD_2D_ARRAY_V4I16_CLAMP
  : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b16.clamp", GPR16>;
defm SULD_2D_ARRAY_V4I32_CLAMP
  : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b32.clamp", GPR32>;

defm SULD_2D_ARRAY_V4I8_TRAP
  : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b8.trap", GPR16>;
defm SULD_2D_ARRAY_V4I16_TRAP
  : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b16.trap", GPR16>;
defm SULD_2D_ARRAY_V4I32_TRAP
  : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b32.trap", GPR32>;

defm SULD_2D_ARRAY_V4I8_ZERO
  : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b8.zero", GPR16>;
defm SULD_2D_ARRAY_V4I16_ZERO
  : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b16.zero", GPR16>;
defm SULD_2D_ARRAY_V4I32_ZERO
  : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b32.zero", GPR32>;

class SULD_3D_V4_base<string inst, RVGPURegClass outtype, dag surf>
    : RVGPUInst<(outs outtype:$r, outtype:$g, outtype:$b, outtype:$a),
                !con(surf, (ins GPR32:$x, GPR32:$y, GPR32:$z)),
                inst # " \\{$r, $g, $b, $a\\}, [$s, \\{$x, $y, $z, $z\\}]",
                []>;
multiclass SULD_3D_V4<string inst, RVGPURegClass outtype> {
  def _R : SULD_3D_V4_base<inst, outtype, (ins GPR64:$s)>;
  def _I : SULD_3D_V4_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_3D_V4I8_CLAMP : SULD_3D_V4<"suld.b.3d.v4.b8.clamp", GPR16>;
defm SULD_3D_V4I16_CLAMP : SULD_3D_V4<"suld.b.3d.v4.b16.clamp", GPR16>;
defm SULD_3D_V4I32_CLAMP : SULD_3D_V4<"suld.b.3d.v4.b32.clamp", GPR32>;

defm SULD_3D_V4I8_TRAP : SULD_3D_V4<"suld.b.3d.v4.b8.trap", GPR16>;
defm SULD_3D_V4I16_TRAP : SULD_3D_V4<"suld.b.3d.v4.b16.trap", GPR16>;
defm SULD_3D_V4I32_TRAP : SULD_3D_V4<"suld.b.3d.v4.b32.trap", GPR32>;

defm SULD_3D_V4I8_ZERO : SULD_3D_V4<"suld.b.3d.v4.b8.zero", GPR16>;
defm SULD_3D_V4I16_ZERO : SULD_3D_V4<"suld.b.3d.v4.b16.zero", GPR16>;
defm SULD_3D_V4I32_ZERO : SULD_3D_V4<"suld.b.3d.v4.b32.zero", GPR32>;

}

//-----------------------------------
// Texture Query Intrinsics
//-----------------------------------

let IsSurfTexQuery = true in {
def TXQ_CHANNEL_ORDER_R
  : RVGPUInst<(outs GPR32:$d), (ins GPR64:$a),
              "txq.channel_order.b32 \t$d, [$a]",
              []>;
def TXQ_CHANNEL_ORDER_I
  : RVGPUInst<(outs GPR32:$d), (ins i64imm:$a),
              "txq.channel_order.b32 \t$d, [$a]",
              []>;
def TXQ_CHANNEL_DATA_TYPE_R
  : RVGPUInst<(outs GPR32:$d), (ins GPR64:$a),
              "txq.channel_data_type.b32 \t$d, [$a]",
              []>;
def TXQ_CHANNEL_DATA_TYPE_I
  : RVGPUInst<(outs GPR32:$d), (ins i64imm:$a),
              "txq.channel_data_type.b32 \t$d, [$a]",
              []>;
def TXQ_WIDTH_R
  : RVGPUInst<(outs GPR32:$d), (ins GPR64:$a),
              "txq.width.b32 \t$d, [$a]",
              []>;
def TXQ_WIDTH_I
  : RVGPUInst<(outs GPR32:$d), (ins i64imm:$a),
              "txq.width.b32 \t$d, [$a]",
              []>;
def TXQ_HEIGHT_R
  : RVGPUInst<(outs GPR32:$d), (ins GPR64:$a),
              "txq.height.b32 \t$d, [$a]",
              []>;
def TXQ_HEIGHT_I
  : RVGPUInst<(outs GPR32:$d), (ins i64imm:$a),
              "txq.height.b32 \t$d, [$a]",
              []>;
def TXQ_DEPTH_R
  : RVGPUInst<(outs GPR32:$d), (ins GPR64:$a),
              "txq.depth.b32 \t$d, [$a]",
              []>;
def TXQ_DEPTH_I
  : RVGPUInst<(outs GPR32:$d), (ins i64imm:$a),
              "txq.depth.b32 \t$d, [$a]",
              []>;
def TXQ_ARRAY_SIZE_R
  : RVGPUInst<(outs GPR32:$d), (ins GPR64:$a),
              "txq.array_size.b32 \t$d, [$a]",
              []>;
def TXQ_ARRAY_SIZE_I
  : RVGPUInst<(outs GPR32:$d), (ins i64imm:$a),
              "txq.array_size.b32 \t$d, [$a]",
              []>;
def TXQ_NUM_SAMPLES_R
  : RVGPUInst<(outs GPR32:$d), (ins GPR64:$a),
              "txq.num_samples.b32 \t$d, [$a]",
              []>;
def TXQ_NUM_SAMPLES_I
  : RVGPUInst<(outs GPR32:$d), (ins i64imm:$a),
              "txq.num_samples.b32 \t$d, [$a]",
              []>;
def TXQ_NUM_MIPMAP_LEVELS_R
  : RVGPUInst<(outs GPR32:$d), (ins GPR64:$a),
              "txq.num_mipmap_levels.b32 \t$d, [$a]",
              []>;
def TXQ_NUM_MIPMAP_LEVELS_I
  : RVGPUInst<(outs GPR32:$d), (ins i64imm:$a),
              "txq.num_mipmap_levels.b32 \t$d, [$a]",
              []>;
}

def : Pat<(int_rvgpu_txq_channel_order GPR64:$a),
          (TXQ_CHANNEL_ORDER_R GPR64:$a)>;
def : Pat<(int_rvgpu_txq_channel_data_type GPR64:$a),
          (TXQ_CHANNEL_DATA_TYPE_R GPR64:$a)>;
def : Pat<(int_rvgpu_txq_width GPR64:$a),
          (TXQ_WIDTH_R GPR64:$a)>;
def : Pat<(int_rvgpu_txq_height GPR64:$a),
          (TXQ_HEIGHT_R GPR64:$a)>;
def : Pat<(int_rvgpu_txq_depth GPR64:$a),
          (TXQ_DEPTH_R GPR64:$a)>;
def : Pat<(int_rvgpu_txq_array_size GPR64:$a),
          (TXQ_ARRAY_SIZE_R GPR64:$a)>;
def : Pat<(int_rvgpu_txq_num_samples GPR64:$a),
          (TXQ_NUM_SAMPLES_R GPR64:$a)>;
def : Pat<(int_rvgpu_txq_num_mipmap_levels GPR64:$a),
          (TXQ_NUM_MIPMAP_LEVELS_R GPR64:$a)>;


//-----------------------------------
// Surface Query Intrinsics
//-----------------------------------

let IsSurfTexQuery = true in {
def SUQ_CHANNEL_ORDER_R
  : RVGPUInst<(outs GPR32:$d), (ins GPR64:$a),
              "suq.channel_order.b32 \t$d, [$a]",
              []>;
def SUQ_CHANNEL_ORDER_I
  : RVGPUInst<(outs GPR32:$d), (ins i64imm:$a),
              "suq.channel_order.b32 \t$d, [$a]",
              []>;
def SUQ_CHANNEL_DATA_TYPE_R
  : RVGPUInst<(outs GPR32:$d), (ins GPR64:$a),
              "suq.channel_data_type.b32 \t$d, [$a]",
              []>;
def SUQ_CHANNEL_DATA_TYPE_I
  : RVGPUInst<(outs GPR32:$d), (ins i64imm:$a),
              "suq.channel_data_type.b32 \t$d, [$a]",
              []>;
def SUQ_WIDTH_R
  : RVGPUInst<(outs GPR32:$d), (ins GPR64:$a),
              "suq.width.b32 \t$d, [$a]",
              []>;
def SUQ_WIDTH_I
  : RVGPUInst<(outs GPR32:$d), (ins i64imm:$a),
              "suq.width.b32 \t$d, [$a]",
              []>;
def SUQ_HEIGHT_R
  : RVGPUInst<(outs GPR32:$d), (ins GPR64:$a),
              "suq.height.b32 \t$d, [$a]",
              []>;
def SUQ_HEIGHT_I
  : RVGPUInst<(outs GPR32:$d), (ins i64imm:$a),
              "suq.height.b32 \t$d, [$a]",
              []>;
def SUQ_DEPTH_R
  : RVGPUInst<(outs GPR32:$d), (ins GPR64:$a),
              "suq.depth.b32 \t$d, [$a]",
              []>;
def SUQ_DEPTH_I
  : RVGPUInst<(outs GPR32:$d), (ins i64imm:$a),
              "suq.depth.b32 \t$d, [$a]",
              []>;
def SUQ_ARRAY_SIZE_R
  : RVGPUInst<(outs GPR32:$d), (ins GPR64:$a),
              "suq.array_size.b32 \t$d, [$a]",
              []>;
def SUQ_ARRAY_SIZE_I
  : RVGPUInst<(outs GPR32:$d), (ins i64imm:$a),
              "suq.array_size.b32 \t$d, [$a]",
              []>;
}

def : Pat<(int_rvgpu_suq_channel_order GPR64:$a),
          (SUQ_CHANNEL_ORDER_R GPR64:$a)>;
def : Pat<(int_rvgpu_suq_channel_data_type GPR64:$a),
          (SUQ_CHANNEL_DATA_TYPE_R GPR64:$a)>;
def : Pat<(int_rvgpu_suq_width GPR64:$a),
          (SUQ_WIDTH_R GPR64:$a)>;
def : Pat<(int_rvgpu_suq_height GPR64:$a),
          (SUQ_HEIGHT_R GPR64:$a)>;
def : Pat<(int_rvgpu_suq_depth GPR64:$a),
          (SUQ_DEPTH_R GPR64:$a)>;
def : Pat<(int_rvgpu_suq_array_size GPR64:$a),
          (SUQ_ARRAY_SIZE_R GPR64:$a)>;


//===- Handle Query -------------------------------------------------------===//

// TODO: These intrinsics are not yet finalized, pending PTX ISA design work
def ISTYPEP_SAMPLER
  : RVGPUInst<(outs Int1Regs:$d), (ins GPR64:$a),
              "istypep.samplerref \t$d, $a",
              [(set Int1Regs:$d, (int_rvgpu_istypep_sampler GPR64:$a))]>;
def ISTYPEP_SURFACE
  : RVGPUInst<(outs Int1Regs:$d), (ins GPR64:$a),
              "istypep.surfref \t$d, $a",
              [(set Int1Regs:$d, (int_rvgpu_istypep_surface GPR64:$a))]>;
def ISTYPEP_TEXTURE
  : RVGPUInst<(outs Int1Regs:$d), (ins GPR64:$a),
              "istypep.texref \t$d, $a",
              [(set Int1Regs:$d, (int_rvgpu_istypep_texture GPR64:$a))]>;

//===- Surface Stores -----------------------------------------------------===//

let IsSust = true in {

class SUST_1D_base<string inst, RVGPURegClass intype, dag surf>
    : RVGPUInst<(outs),
                !con(surf, (ins GPR32:$x, intype:$r)),
                inst # " \t[$s, \\{$x\\}], \\{$r\\}",
                []>;
multiclass SUST_1D<string inst, RVGPURegClass intype> {
  def _R : SUST_1D_base<inst, intype, (ins GPR64:$s)>;
  def _I : SUST_1D_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_1D_B8_CLAMP : SUST_1D<"sust.b.1d.b8.clamp", GPR16>;
defm SUST_B_1D_B16_CLAMP : SUST_1D<"sust.b.1d.b16.clamp", GPR16>;
defm SUST_B_1D_B32_CLAMP : SUST_1D<"sust.b.1d.b32.clamp", GPR32>;
defm SUST_B_1D_B64_CLAMP : SUST_1D<"sust.b.1d.b64.clamp", GPR64>;

defm SUST_B_1D_B8_TRAP : SUST_1D<"sust.b.1d.b8.trap", GPR16>;
defm SUST_B_1D_B16_TRAP : SUST_1D<"sust.b.1d.b16.trap", GPR16>;
defm SUST_B_1D_B32_TRAP : SUST_1D<"sust.b.1d.b32.trap", GPR32>;
defm SUST_B_1D_B64_TRAP : SUST_1D<"sust.b.1d.b64.trap", GPR64>;

defm SUST_B_1D_B8_ZERO : SUST_1D<"sust.b.1d.b8.zero", GPR16>;
defm SUST_B_1D_B16_ZERO : SUST_1D<"sust.b.1d.b16.zero", GPR16>;
defm SUST_B_1D_B32_ZERO : SUST_1D<"sust.b.1d.b32.zero", GPR32>;
defm SUST_B_1D_B64_ZERO : SUST_1D<"sust.b.1d.b64.zero", GPR64>;

defm SUST_P_1D_B8_TRAP : SUST_1D<"sust.p.1d.b8.trap", GPR16>;
defm SUST_P_1D_B16_TRAP : SUST_1D<"sust.p.1d.b16.trap", GPR16>;
defm SUST_P_1D_B32_TRAP : SUST_1D<"sust.p.1d.b32.trap", GPR32>;

class SUST_1D_V2_base<string inst, RVGPURegClass intype, dag surf>
    : RVGPUInst<(outs),
                !con(surf, (ins GPR32:$x, intype:$r, intype:$g)),
                inst # " \t[$s, \\{$x\\}], \\{$r, $g\\}",
                []>;
multiclass SUST_1D_V2<string inst, RVGPURegClass intype> {
  def _R : SUST_1D_V2_base<inst, intype, (ins GPR64:$s)>;
  def _I : SUST_1D_V2_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_1D_V2B8_CLAMP : SUST_1D_V2<"sust.b.1d.v2.b8.clamp", GPR16>;
defm SUST_B_1D_V2B16_CLAMP : SUST_1D_V2<"sust.b.1d.v2.b16.clamp", GPR16>;
defm SUST_B_1D_V2B32_CLAMP : SUST_1D_V2<"sust.b.1d.v2.b32.clamp", GPR32>;
defm SUST_B_1D_V2B64_CLAMP : SUST_1D_V2<"sust.b.1d.v2.b64.clamp", GPR64>;

defm SUST_B_1D_V2B8_TRAP : SUST_1D_V2<"sust.b.1d.v2.b8.trap", GPR16>;
defm SUST_B_1D_V2B16_TRAP : SUST_1D_V2<"sust.b.1d.v2.b16.trap", GPR16>;
defm SUST_B_1D_V2B32_TRAP : SUST_1D_V2<"sust.b.1d.v2.b32.trap", GPR32>;
defm SUST_B_1D_V2B64_TRAP : SUST_1D_V2<"sust.b.1d.v2.b64.trap", GPR64>;

defm SUST_B_1D_V2B8_ZERO : SUST_1D_V2<"sust.b.1d.v2.b8.zero", GPR16>;
defm SUST_B_1D_V2B16_ZERO : SUST_1D_V2<"sust.b.1d.v2.b16.zero", GPR16>;
defm SUST_B_1D_V2B32_ZERO : SUST_1D_V2<"sust.b.1d.v2.b32.zero", GPR32>;
defm SUST_B_1D_V2B64_ZERO : SUST_1D_V2<"sust.b.1d.v2.b64.zero", GPR64>;

defm SUST_P_1D_V2B8_TRAP : SUST_1D_V2<"sust.p.1d.v2.b8.trap", GPR16>;
defm SUST_P_1D_V2B16_TRAP : SUST_1D_V2<"sust.p.1d.v2.b16.trap", GPR16>;
defm SUST_P_1D_V2B32_TRAP : SUST_1D_V2<"sust.p.1d.v2.b32.trap", GPR32>;

class SUST_1D_V4_base<string inst, RVGPURegClass intype, dag surf>
    : RVGPUInst<(outs),
                !con(surf, (ins GPR32:$x, intype:$r, intype:$g,
                                intype:$b, intype:$a)),
                inst # " \t[$s, \\{$x\\}], \\{$r, $g, $b, $a\\}",
                []>;
multiclass SUST_1D_V4<string inst, RVGPURegClass intype> {
  def _R : SUST_1D_V4_base<inst, intype, (ins GPR64:$s)>;
  def _I : SUST_1D_V4_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_1D_V4B8_CLAMP : SUST_1D_V4<"sust.b.1d.v4.b8.clamp", GPR16>;
defm SUST_B_1D_V4B16_CLAMP : SUST_1D_V4<"sust.b.1d.v4.b16.clamp", GPR16>;
defm SUST_B_1D_V4B32_CLAMP : SUST_1D_V4<"sust.b.1d.v4.b32.clamp", GPR32>;

defm SUST_B_1D_V4B8_TRAP : SUST_1D_V4<"sust.b.1d.v4.b8.trap", GPR16>;
defm SUST_B_1D_V4B16_TRAP : SUST_1D_V4<"sust.b.1d.v4.b16.trap", GPR16>;
defm SUST_B_1D_V4B32_TRAP : SUST_1D_V4<"sust.b.1d.v4.b32.trap", GPR32>;

defm SUST_B_1D_V4B8_ZERO : SUST_1D_V4<"sust.b.1d.v4.b8.zero", GPR16>;
defm SUST_B_1D_V4B16_ZERO : SUST_1D_V4<"sust.b.1d.v4.b16.zero", GPR16>;
defm SUST_B_1D_V4B32_ZERO : SUST_1D_V4<"sust.b.1d.v4.b32.zero", GPR32>;

defm SUST_P_1D_V4B8_TRAP : SUST_1D_V4<"sust.p.1d.v4.b8.trap", GPR16>;
defm SUST_P_1D_V4B16_TRAP : SUST_1D_V4<"sust.p.1d.v4.b16.trap", GPR16>;
defm SUST_P_1D_V4B32_TRAP : SUST_1D_V4<"sust.p.1d.v4.b32.trap", GPR32>;

class SUST_1D_ARRAY_base<string inst, RVGPURegClass intype, dag surf>
    : RVGPUInst<(outs),
                !con(surf, (ins GPR32:$idx, GPR32:$x, intype:$r)),
                inst # " \t[$s, \\{$idx, $x\\}], \\{$r\\}",
                []>;
multiclass SUST_1D_ARRAY<string inst, RVGPURegClass intype> {
  def _R : SUST_1D_ARRAY_base<inst, intype, (ins GPR64:$s)>;
  def _I : SUST_1D_ARRAY_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_1D_ARRAY_B8_CLAMP
  : SUST_1D_ARRAY<"sust.b.a1d.b8.clamp", GPR16>;
defm SUST_B_1D_ARRAY_B16_CLAMP
  : SUST_1D_ARRAY<"sust.b.a1d.b16.clamp", GPR16>;
defm SUST_B_1D_ARRAY_B32_CLAMP
  : SUST_1D_ARRAY<"sust.b.a1d.b32.clamp", GPR32>;
defm SUST_B_1D_ARRAY_B64_CLAMP
  : SUST_1D_ARRAY<"sust.b.a1d.b64.clamp", GPR64>;

defm SUST_B_1D_ARRAY_B8_TRAP
  : SUST_1D_ARRAY<"sust.b.a1d.b8.trap", GPR16>;
defm SUST_B_1D_ARRAY_B16_TRAP
  : SUST_1D_ARRAY<"sust.b.a1d.b16.trap", GPR16>;
defm SUST_B_1D_ARRAY_B32_TRAP
  : SUST_1D_ARRAY<"sust.b.a1d.b32.trap", GPR32>;
defm SUST_B_1D_ARRAY_B64_TRAP
  : SUST_1D_ARRAY<"sust.b.a1d.b64.trap", GPR64>;

defm SUST_B_1D_ARRAY_B8_ZERO
  : SUST_1D_ARRAY<"sust.b.a1d.b8.zero", GPR16>;
defm SUST_B_1D_ARRAY_B16_ZERO
  : SUST_1D_ARRAY<"sust.b.a1d.b16.zero", GPR16>;
defm SUST_B_1D_ARRAY_B32_ZERO
  : SUST_1D_ARRAY<"sust.b.a1d.b32.zero", GPR32>;
defm SUST_B_1D_ARRAY_B64_ZERO
  : SUST_1D_ARRAY<"sust.b.a1d.b64.zero", GPR64>;

defm SUST_P_1D_ARRAY_B8_TRAP
  : SUST_1D_ARRAY<"sust.p.a1d.b8.trap", GPR16>;
defm SUST_P_1D_ARRAY_B16_TRAP
  : SUST_1D_ARRAY<"sust.p.a1d.b16.trap", GPR16>;
defm SUST_P_1D_ARRAY_B32_TRAP
  : SUST_1D_ARRAY<"sust.p.a1d.b32.trap", GPR32>;

class SUST_1D_ARRAY_V2_base<string inst, RVGPURegClass intype, dag surf>
    : RVGPUInst<(outs),
                !con(surf, (ins GPR32:$idx, GPR32:$x,
                                intype:$r, intype:$g)),
                inst # " \t[$s, \\{$idx, $x\\}], \\{$r, $g\\}",
                []>;
multiclass SUST_1D_ARRAY_V2<string inst, RVGPURegClass intype> {
  def _R : SUST_1D_ARRAY_V2_base<inst, intype, (ins GPR64:$s)>;
  def _I : SUST_1D_ARRAY_V2_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_1D_ARRAY_V2B8_CLAMP
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b8.clamp", GPR16>;
defm SUST_B_1D_ARRAY_V2B16_CLAMP
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b16.clamp", GPR16>;
defm SUST_B_1D_ARRAY_V2B32_CLAMP
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b32.clamp", GPR32>;
defm SUST_B_1D_ARRAY_V2B64_CLAMP
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b64.clamp", GPR64>;

defm SUST_B_1D_ARRAY_V2B8_TRAP
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b8.trap", GPR16>;
defm SUST_B_1D_ARRAY_V2B16_TRAP
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b16.trap", GPR16>;
defm SUST_B_1D_ARRAY_V2B32_TRAP
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b32.trap", GPR32>;
defm SUST_B_1D_ARRAY_V2B64_TRAP
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b64.trap", GPR64>;

defm SUST_B_1D_ARRAY_V2B8_ZERO
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b8.zero", GPR16>;
defm SUST_B_1D_ARRAY_V2B16_ZERO
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b16.zero", GPR16>;
defm SUST_B_1D_ARRAY_V2B32_ZERO
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b32.zero", GPR32>;
defm SUST_B_1D_ARRAY_V2B64_ZERO
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b64.zero", GPR64>;

defm SUST_P_1D_ARRAY_V2B8_TRAP
  : SUST_1D_ARRAY_V2<"sust.p.a1d.v2.b8.trap", GPR16>;
defm SUST_P_1D_ARRAY_V2B16_TRAP
  : SUST_1D_ARRAY_V2<"sust.p.a1d.v2.b16.trap", GPR16>;
defm SUST_P_1D_ARRAY_V2B32_TRAP
  : SUST_1D_ARRAY_V2<"sust.p.a1d.v2.b32.trap", GPR32>;

class SUST_1D_ARRAY_V4_base<string inst, RVGPURegClass intype, dag surf>
    : RVGPUInst<(outs),
                !con(surf, (ins GPR32:$idx, GPR32:$x,
                                intype:$r, intype:$g, intype:$b, intype:$a)),
                inst # " \t[$s, \\{$idx, $x\\}], \\{$r, $g, $b, $a\\}",
                []>;
multiclass SUST_1D_ARRAY_V4<string inst, RVGPURegClass intype> {
  def _R : SUST_1D_ARRAY_V4_base<inst, intype, (ins GPR64:$s)>;
  def _I : SUST_1D_ARRAY_V4_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_1D_ARRAY_V4B8_CLAMP
  : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b8.clamp", GPR16>;
defm SUST_B_1D_ARRAY_V4B16_CLAMP
  : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b16.clamp", GPR16>;
defm SUST_B_1D_ARRAY_V4B32_CLAMP
  : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b32.clamp", GPR32>;

defm SUST_B_1D_ARRAY_V4B8_TRAP
  : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b8.trap", GPR16>;
defm SUST_B_1D_ARRAY_V4B16_TRAP
  : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b16.trap", GPR16>;
defm SUST_B_1D_ARRAY_V4B32_TRAP
  : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b32.trap", GPR32>;

defm SUST_B_1D_ARRAY_V4B8_ZERO
  : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b8.zero", GPR16>;
defm SUST_B_1D_ARRAY_V4B16_ZERO
  : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b16.zero", GPR16>;
defm SUST_B_1D_ARRAY_V4B32_ZERO
  : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b32.zero", GPR32>;

defm SUST_P_1D_ARRAY_V4B8_TRAP
  : SUST_1D_ARRAY_V4<"sust.p.a1d.v4.b8.trap", GPR16>;
defm SUST_P_1D_ARRAY_V4B16_TRAP
  : SUST_1D_ARRAY_V4<"sust.p.a1d.v4.b16.trap", GPR16>;
defm SUST_P_1D_ARRAY_V4B32_TRAP
  : SUST_1D_ARRAY_V4<"sust.p.a1d.v4.b32.trap", GPR32>;

class SUST_2D_base<string inst, RVGPURegClass intype, dag surf>
    : RVGPUInst<(outs),
                !con(surf, (ins GPR32:$x, GPR32:$y, intype:$r)),
                inst # " \t[$s, \\{$x, $y\\}], \\{$r\\}",
                []>;
multiclass SUST_2D<string inst, RVGPURegClass intype> {
  def _R : SUST_2D_base<inst, intype, (ins GPR64:$s)>;
  def _I : SUST_2D_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_2D_B8_CLAMP : SUST_2D<"sust.b.2d.b8.clamp", GPR16>;
defm SUST_B_2D_B16_CLAMP : SUST_2D<"sust.b.2d.b16.clamp", GPR16>;
defm SUST_B_2D_B32_CLAMP : SUST_2D<"sust.b.2d.b32.clamp", GPR32>;
defm SUST_B_2D_B64_CLAMP : SUST_2D<"sust.b.2d.b64.clamp", GPR64>;

defm SUST_B_2D_B8_TRAP : SUST_2D<"sust.b.2d.b8.trap", GPR16>;
defm SUST_B_2D_B16_TRAP : SUST_2D<"sust.b.2d.b16.trap", GPR16>;
defm SUST_B_2D_B32_TRAP : SUST_2D<"sust.b.2d.b32.trap", GPR32>;
defm SUST_B_2D_B64_TRAP : SUST_2D<"sust.b.2d.b64.trap", GPR64>;

defm SUST_B_2D_B8_ZERO : SUST_2D<"sust.b.2d.b8.zero", GPR16>;
defm SUST_B_2D_B16_ZERO : SUST_2D<"sust.b.2d.b16.zero", GPR16>;
defm SUST_B_2D_B32_ZERO : SUST_2D<"sust.b.2d.b32.zero", GPR32>;
defm SUST_B_2D_B64_ZERO : SUST_2D<"sust.b.2d.b64.zero", GPR64>;

defm SUST_P_2D_B8_TRAP : SUST_2D<"sust.p.2d.b8.trap", GPR16>;
defm SUST_P_2D_B16_TRAP : SUST_2D<"sust.p.2d.b16.trap", GPR16>;
defm SUST_P_2D_B32_TRAP : SUST_2D<"sust.p.2d.b32.trap", GPR32>;

class SUST_2D_V2_base<string inst, RVGPURegClass intype, dag surf>
    : RVGPUInst<(outs),
                !con(surf, (ins GPR32:$x, GPR32:$y,
                                intype:$r, intype:$g)),
                inst # " \t[$s, \\{$x, $y\\}], \\{$r, $g\\}",
                []>;
multiclass SUST_2D_V2<string inst, RVGPURegClass intype> {
  def _R : SUST_2D_V2_base<inst, intype, (ins GPR64:$s)>;
  def _I : SUST_2D_V2_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_2D_V2B8_CLAMP : SUST_2D_V2<"sust.b.2d.v2.b8.clamp", GPR16>;
defm SUST_B_2D_V2B16_CLAMP : SUST_2D_V2<"sust.b.2d.v2.b16.clamp", GPR16>;
defm SUST_B_2D_V2B32_CLAMP : SUST_2D_V2<"sust.b.2d.v2.b32.clamp", GPR32>;
defm SUST_B_2D_V2B64_CLAMP : SUST_2D_V2<"sust.b.2d.v2.b64.clamp", GPR64>;

defm SUST_B_2D_V2B8_TRAP : SUST_2D_V2<"sust.b.2d.v2.b8.trap", GPR16>;
defm SUST_B_2D_V2B16_TRAP : SUST_2D_V2<"sust.b.2d.v2.b16.trap", GPR16>;
defm SUST_B_2D_V2B32_TRAP : SUST_2D_V2<"sust.b.2d.v2.b32.trap", GPR32>;
defm SUST_B_2D_V2B64_TRAP : SUST_2D_V2<"sust.b.2d.v2.b64.trap", GPR64>;

defm SUST_B_2D_V2B8_ZERO : SUST_2D_V2<"sust.b.2d.v2.b8.zero", GPR16>;
defm SUST_B_2D_V2B16_ZERO : SUST_2D_V2<"sust.b.2d.v2.b16.zero", GPR16>;
defm SUST_B_2D_V2B32_ZERO : SUST_2D_V2<"sust.b.2d.v2.b32.zero", GPR32>;
defm SUST_B_2D_V2B64_ZERO : SUST_2D_V2<"sust.b.2d.v2.b64.zero", GPR64>;

defm SUST_P_2D_V2B8_TRAP : SUST_2D_V2<"sust.p.2d.v2.b8.trap", GPR16>;
defm SUST_P_2D_V2B16_TRAP : SUST_2D_V2<"sust.p.2d.v2.b16.trap", GPR16>;
defm SUST_P_2D_V2B32_TRAP : SUST_2D_V2<"sust.p.2d.v2.b32.trap", GPR32>;

class SUST_2D_V4_base<string inst, RVGPURegClass intype, dag surf>
    : RVGPUInst<(outs),
                !con(surf, (ins GPR32:$x, GPR32:$y,
                                intype:$r, intype:$g, intype:$b, intype:$a)),
                inst # " \t[$s, \\{$x, $y\\}], \\{$r, $g, $b, $a\\}",
                []>;
multiclass SUST_2D_V4<string inst, RVGPURegClass intype> {
  def _R : SUST_2D_V4_base<inst, intype, (ins GPR64:$s)>;
  def _I : SUST_2D_V4_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_2D_V4B8_CLAMP : SUST_2D_V4<"sust.b.2d.v4.b8.clamp", GPR16>;
defm SUST_B_2D_V4B16_CLAMP : SUST_2D_V4<"sust.b.2d.v4.b16.clamp", GPR16>;
defm SUST_B_2D_V4B32_CLAMP : SUST_2D_V4<"sust.b.2d.v4.b32.clamp", GPR32>;

defm SUST_B_2D_V4B8_TRAP : SUST_2D_V4<"sust.b.2d.v4.b8.trap", GPR16>;
defm SUST_B_2D_V4B16_TRAP : SUST_2D_V4<"sust.b.2d.v4.b16.trap", GPR16>;
defm SUST_B_2D_V4B32_TRAP : SUST_2D_V4<"sust.b.2d.v4.b32.trap", GPR32>;

defm SUST_B_2D_V4B8_ZERO : SUST_2D_V4<"sust.b.2d.v4.b8.zero", GPR16>;
defm SUST_B_2D_V4B16_ZERO : SUST_2D_V4<"sust.b.2d.v4.b16.zero", GPR16>;
defm SUST_B_2D_V4B32_ZERO : SUST_2D_V4<"sust.b.2d.v4.b32.zero", GPR32>;

defm SUST_P_2D_V4B8_TRAP : SUST_2D_V4<"sust.p.2d.v4.b8.trap", GPR16>;
defm SUST_P_2D_V4B16_TRAP : SUST_2D_V4<"sust.p.2d.v4.b16.trap", GPR16>;
defm SUST_P_2D_V4B32_TRAP : SUST_2D_V4<"sust.p.2d.v4.b32.trap", GPR32>;

class SUST_2D_ARRAY_base<string inst, RVGPURegClass intype, dag surf>
    : RVGPUInst<(outs),
                !con(surf, (ins GPR32:$idx, GPR32:$x, GPR32:$y,
                                intype:$r)),
                inst # " \t[$s, \\{$idx, $x, $y, $y\\}], \\{$r\\}",
                []>;
multiclass SUST_2D_ARRAY<string inst, RVGPURegClass intype> {
  def _R : SUST_2D_ARRAY_base<inst, intype, (ins GPR64:$s)>;
  def _I : SUST_2D_ARRAY_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_2D_ARRAY_B8_CLAMP
  : SUST_2D_ARRAY<"sust.b.a2d.b8.clamp", GPR16>;
defm SUST_B_2D_ARRAY_B16_CLAMP
  : SUST_2D_ARRAY<"sust.b.a2d.b16.clamp", GPR16>;
defm SUST_B_2D_ARRAY_B32_CLAMP
  : SUST_2D_ARRAY<"sust.b.a2d.b32.clamp", GPR32>;
defm SUST_B_2D_ARRAY_B64_CLAMP
  : SUST_2D_ARRAY<"sust.b.a2d.b64.clamp", GPR64>;

defm SUST_B_2D_ARRAY_B8_TRAP
  : SUST_2D_ARRAY<"sust.b.a2d.b8.trap", GPR16>;
defm SUST_B_2D_ARRAY_B16_TRAP
  : SUST_2D_ARRAY<"sust.b.a2d.b16.trap", GPR16>;
defm SUST_B_2D_ARRAY_B32_TRAP
  : SUST_2D_ARRAY<"sust.b.a2d.b32.trap", GPR32>;
defm SUST_B_2D_ARRAY_B64_TRAP
  : SUST_2D_ARRAY<"sust.b.a2d.b64.trap", GPR64>;

defm SUST_B_2D_ARRAY_B8_ZERO
  : SUST_2D_ARRAY<"sust.b.a2d.b8.zero", GPR16>;
defm SUST_B_2D_ARRAY_B16_ZERO
  : SUST_2D_ARRAY<"sust.b.a2d.b16.zero", GPR16>;
defm SUST_B_2D_ARRAY_B32_ZERO
  : SUST_2D_ARRAY<"sust.b.a2d.b32.zero", GPR32>;
defm SUST_B_2D_ARRAY_B64_ZERO
  : SUST_2D_ARRAY<"sust.b.a2d.b64.zero", GPR64>;

defm SUST_P_2D_ARRAY_B8_TRAP
  : SUST_2D_ARRAY<"sust.p.a2d.b8.trap", GPR16>;
defm SUST_P_2D_ARRAY_B16_TRAP
  : SUST_2D_ARRAY<"sust.p.a2d.b16.trap", GPR16>;
defm SUST_P_2D_ARRAY_B32_TRAP
  : SUST_2D_ARRAY<"sust.p.a2d.b32.trap", GPR32>;

class SUST_2D_ARRAY_V2_base<string inst, RVGPURegClass intype, dag surf>
    : RVGPUInst<(outs),
                !con(surf, (ins GPR32:$idx, GPR32:$x, GPR32:$y,
                                intype:$r, intype:$g)),
                inst # " \t[$s, \\{$idx, $x, $y, $y\\}], \\{$r, $g\\}",
                []>;
multiclass SUST_2D_ARRAY_V2<string inst, RVGPURegClass intype> {
  def _R : SUST_2D_ARRAY_V2_base<inst, intype, (ins GPR64:$s)>;
  def _I : SUST_2D_ARRAY_V2_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_2D_ARRAY_V2B8_CLAMP
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b8.clamp", GPR16>;
defm SUST_B_2D_ARRAY_V2B16_CLAMP
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b16.clamp", GPR16>;
defm SUST_B_2D_ARRAY_V2B32_CLAMP
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b32.clamp", GPR32>;
defm SUST_B_2D_ARRAY_V2B64_CLAMP
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b64.clamp", GPR64>;

defm SUST_B_2D_ARRAY_V2B8_TRAP
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b8.trap", GPR16>;
defm SUST_B_2D_ARRAY_V2B16_TRAP
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b16.trap", GPR16>;
defm SUST_B_2D_ARRAY_V2B32_TRAP
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b32.trap", GPR32>;
defm SUST_B_2D_ARRAY_V2B64_TRAP
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b64.trap", GPR64>;

defm SUST_B_2D_ARRAY_V2B8_ZERO
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b8.zero", GPR16>;
defm SUST_B_2D_ARRAY_V2B16_ZERO
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b16.zero", GPR16>;
defm SUST_B_2D_ARRAY_V2B32_ZERO
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b32.zero", GPR32>;
defm SUST_B_2D_ARRAY_V2B64_ZERO
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b64.zero", GPR64>;

defm SUST_P_2D_ARRAY_V2B8_TRAP
  : SUST_2D_ARRAY_V2<"sust.p.a2d.v2.b8.trap", GPR16>;
defm SUST_P_2D_ARRAY_V2B16_TRAP
  : SUST_2D_ARRAY_V2<"sust.p.a2d.v2.b16.trap", GPR16>;
defm SUST_P_2D_ARRAY_V2B32_TRAP
  : SUST_2D_ARRAY_V2<"sust.p.a2d.v2.b32.trap", GPR32>;

class SUST_2D_ARRAY_V4_base<string inst, RVGPURegClass intype, dag surf>
    : RVGPUInst<(outs),
                !con(surf, (ins GPR32:$idx, GPR32:$x, GPR32:$y,
                                intype:$r, intype:$g, intype:$b, intype:$a)),
                inst # " \t[$s, \\{$idx, $x, $y, $y\\}], \\{$r, $g, $b, $a\\}",
                []>;
multiclass SUST_2D_ARRAY_V4<string inst, RVGPURegClass intype> {
  def _R : SUST_2D_ARRAY_V4_base<inst, intype, (ins GPR64:$s)>;
  def _I : SUST_2D_ARRAY_V4_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_2D_ARRAY_V4B8_CLAMP
  : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b8.clamp", GPR16>;
defm SUST_B_2D_ARRAY_V4B16_CLAMP
  : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b16.clamp", GPR16>;
defm SUST_B_2D_ARRAY_V4B32_CLAMP
  : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b32.clamp", GPR32>;

defm SUST_B_2D_ARRAY_V4B8_TRAP
  : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b8.trap", GPR16>;
defm SUST_B_2D_ARRAY_V4B16_TRAP
  : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b16.trap", GPR16>;
defm SUST_B_2D_ARRAY_V4B32_TRAP
  : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b32.trap", GPR32>;

defm SUST_B_2D_ARRAY_V4B8_ZERO
  : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b8.zero", GPR16>;
defm SUST_B_2D_ARRAY_V4B16_ZERO
  : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b16.zero", GPR16>;
defm SUST_B_2D_ARRAY_V4B32_ZERO
  : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b32.zero", GPR32>;

defm SUST_P_2D_ARRAY_V4B8_TRAP
  : SUST_2D_ARRAY_V4<"sust.p.a2d.v4.b8.trap", GPR16>;
defm SUST_P_2D_ARRAY_V4B16_TRAP
  : SUST_2D_ARRAY_V4<"sust.p.a2d.v4.b16.trap", GPR16>;
defm SUST_P_2D_ARRAY_V4B32_TRAP
  : SUST_2D_ARRAY_V4<"sust.p.a2d.v4.b32.trap", GPR32>;

class SUST_3D_base<string inst, RVGPURegClass intype, dag surf>
    : RVGPUInst<(outs),
                !con(surf, (ins GPR32:$x, GPR32:$y, GPR32:$z,
                                intype:$r)),
                inst # " \t[$s, \\{$x, $y, $z, $z\\}], \\{$r\\}",
                []>;
multiclass SUST_3D<string inst, RVGPURegClass intype> {
  def _R : SUST_3D_base<inst, intype, (ins GPR64:$s)>;
  def _I : SUST_3D_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_3D_B8_CLAMP : SUST_3D<"sust.b.3d.b8.clamp", GPR16>;
defm SUST_B_3D_B16_CLAMP : SUST_3D<"sust.b.3d.b16.clamp", GPR16>;
defm SUST_B_3D_B32_CLAMP : SUST_3D<"sust.b.3d.b32.clamp", GPR32>;
defm SUST_B_3D_B64_CLAMP : SUST_3D<"sust.b.3d.b64.clamp", GPR64>;

defm SUST_B_3D_B8_TRAP : SUST_3D<"sust.b.3d.b8.trap", GPR16>;
defm SUST_B_3D_B16_TRAP : SUST_3D<"sust.b.3d.b16.trap", GPR16>;
defm SUST_B_3D_B32_TRAP : SUST_3D<"sust.b.3d.b32.trap", GPR32>;
defm SUST_B_3D_B64_TRAP : SUST_3D<"sust.b.3d.b64.trap", GPR64>;

defm SUST_B_3D_B8_ZERO : SUST_3D<"sust.b.3d.b8.zero", GPR16>;
defm SUST_B_3D_B16_ZERO : SUST_3D<"sust.b.3d.b16.zero", GPR16>;
defm SUST_B_3D_B32_ZERO : SUST_3D<"sust.b.3d.b32.zero", GPR32>;
defm SUST_B_3D_B64_ZERO : SUST_3D<"sust.b.3d.b64.zero", GPR64>;

defm SUST_P_3D_B8_TRAP : SUST_3D<"sust.p.3d.b8.trap", GPR16>;
defm SUST_P_3D_B16_TRAP : SUST_3D<"sust.p.3d.b16.trap", GPR16>;
defm SUST_P_3D_B32_TRAP : SUST_3D<"sust.p.3d.b32.trap", GPR32>;

class SUST_3D_V2_base<string inst, RVGPURegClass intype, dag surf>
    : RVGPUInst<(outs),
                !con(surf, (ins GPR32:$x, GPR32:$y, GPR32:$z,
                                intype:$r, intype:$g)),
                inst # " \t[$s, \\{$x, $y, $z, $z\\}], \\{$r, $g\\}",
                []>;
multiclass SUST_3D_V2<string inst, RVGPURegClass intype> {
  def _R : SUST_3D_V2_base<inst, intype, (ins GPR64:$s)>;
  def _I : SUST_3D_V2_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_3D_V2B8_CLAMP : SUST_3D_V2<"sust.b.3d.v2.b8.clamp", GPR16>;
defm SUST_B_3D_V2B16_CLAMP : SUST_3D_V2<"sust.b.3d.v2.b16.clamp", GPR16>;
defm SUST_B_3D_V2B32_CLAMP : SUST_3D_V2<"sust.b.3d.v2.b32.clamp", GPR32>;
defm SUST_B_3D_V2B64_CLAMP : SUST_3D_V2<"sust.b.3d.v2.b64.clamp", GPR64>;

defm SUST_B_3D_V2B8_TRAP : SUST_3D_V2<"sust.b.3d.v2.b8.trap", GPR16>;
defm SUST_B_3D_V2B16_TRAP : SUST_3D_V2<"sust.b.3d.v2.b16.trap", GPR16>;
defm SUST_B_3D_V2B32_TRAP : SUST_3D_V2<"sust.b.3d.v2.b32.trap", GPR32>;
defm SUST_B_3D_V2B64_TRAP : SUST_3D_V2<"sust.b.3d.v2.b64.trap", GPR64>;

defm SUST_B_3D_V2B8_ZERO : SUST_3D_V2<"sust.b.3d.v2.b8.zero", GPR16>;
defm SUST_B_3D_V2B16_ZERO : SUST_3D_V2<"sust.b.3d.v2.b16.zero", GPR16>;
defm SUST_B_3D_V2B32_ZERO : SUST_3D_V2<"sust.b.3d.v2.b32.zero", GPR32>;
defm SUST_B_3D_V2B64_ZERO : SUST_3D_V2<"sust.b.3d.v2.b64.zero", GPR64>;

defm SUST_P_3D_V2B8_TRAP : SUST_3D_V2<"sust.p.3d.v2.b8.trap", GPR16>;
defm SUST_P_3D_V2B16_TRAP : SUST_3D_V2<"sust.p.3d.v2.b16.trap", GPR16>;
defm SUST_P_3D_V2B32_TRAP : SUST_3D_V2<"sust.p.3d.v2.b32.trap", GPR32>;

class SUST_3D_V4_base<string inst, RVGPURegClass intype, dag surf>
    : RVGPUInst<(outs),
                !con(surf, (ins GPR32:$x, GPR32:$y, GPR32:$z,
                                intype:$r, intype:$g, intype:$b, intype:$a)),
                inst # " \t[$s, \\{$x, $y, $z, $z\\}], \\{$r, $g, $b, $a\\}",
                []>;
multiclass SUST_3D_V4<string inst, RVGPURegClass intype> {
  def _R : SUST_3D_V4_base<inst, intype, (ins GPR64:$s)>;
  def _I : SUST_3D_V4_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_3D_V4B8_CLAMP : SUST_3D_V4<"sust.b.3d.v4.b8.clamp", GPR16>;
defm SUST_B_3D_V4B16_CLAMP : SUST_3D_V4<"sust.b.3d.v4.b16.clamp", GPR16>;
defm SUST_B_3D_V4B32_CLAMP : SUST_3D_V4<"sust.b.3d.v4.b32.clamp", GPR32>;

defm SUST_B_3D_V4B8_TRAP : SUST_3D_V4<"sust.b.3d.v4.b8.trap", GPR16>;
defm SUST_B_3D_V4B16_TRAP : SUST_3D_V4<"sust.b.3d.v4.b16.trap", GPR16>;
defm SUST_B_3D_V4B32_TRAP : SUST_3D_V4<"sust.b.3d.v4.b32.trap", GPR32>;

defm SUST_B_3D_V4B8_ZERO : SUST_3D_V4<"sust.b.3d.v4.b8.zero", GPR16>;
defm SUST_B_3D_V4B16_ZERO : SUST_3D_V4<"sust.b.3d.v4.b16.zero", GPR16>;
defm SUST_B_3D_V4B32_ZERO : SUST_3D_V4<"sust.b.3d.v4.b32.zero", GPR32>;

defm SUST_P_3D_V4B8_TRAP : SUST_3D_V4<"sust.p.3d.v4.b8.trap", GPR16>;
defm SUST_P_3D_V4B16_TRAP : SUST_3D_V4<"sust.p.3d.v4.b16.trap", GPR16>;
defm SUST_P_3D_V4B32_TRAP : SUST_3D_V4<"sust.p.3d.v4.b32.trap", GPR32>;

}

// Surface store instruction patterns
// I'm not sure why we can't just include these in the instruction definitions,
// but TableGen complains of type errors :(

// .clamp variant
def : Pat<(int_rvgpu_sust_b_1d_i8_clamp
           GPR64:$s, GPR32:$x, GPR16:$r),
          (SUST_B_1D_B8_CLAMP_R GPR64:$s, GPR32:$x, GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_i16_clamp
           GPR64:$s, GPR32:$x, GPR16:$r),
          (SUST_B_1D_B16_CLAMP_R GPR64:$s, GPR32:$x, GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_i32_clamp
           GPR64:$s, GPR32:$x, GPR32:$r),
          (SUST_B_1D_B32_CLAMP_R GPR64:$s, GPR32:$x, GPR32:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_i64_clamp
           GPR64:$s, GPR32:$x, GPR64:$r),
          (SUST_B_1D_B64_CLAMP_R GPR64:$s, GPR32:$x, GPR64:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_v2i8_clamp
           GPR64:$s, GPR32:$x, GPR16:$r, GPR16:$g),
          (SUST_B_1D_V2B8_CLAMP_R GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_v2i16_clamp
           GPR64:$s, GPR32:$x, GPR16:$r, GPR16:$g),
          (SUST_B_1D_V2B16_CLAMP_R GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_v2i32_clamp
           GPR64:$s, GPR32:$x, GPR32:$r, GPR32:$g),
          (SUST_B_1D_V2B32_CLAMP_R GPR64:$s, GPR32:$x,
           GPR32:$r, GPR32:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_v2i64_clamp
           GPR64:$s, GPR32:$x, GPR64:$r, GPR64:$g),
          (SUST_B_1D_V2B64_CLAMP_R GPR64:$s, GPR32:$x,
           GPR64:$r, GPR64:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_v4i8_clamp
           GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_1D_V4B8_CLAMP_R GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_1d_v4i16_clamp
           GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_1D_V4B16_CLAMP_R GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_1d_v4i32_clamp
           GPR64:$s, GPR32:$x,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a),
          (SUST_B_1D_V4B32_CLAMP_R GPR64:$s, GPR32:$x,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a)>;



def : Pat<(int_rvgpu_sust_b_1d_array_i8_clamp
           GPR64:$s, GPR32:$l, GPR32:$x, GPR16:$r),
          (SUST_B_1D_ARRAY_B8_CLAMP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_array_i16_clamp
           GPR64:$s, GPR32:$l, GPR32:$x, GPR16:$r),
          (SUST_B_1D_ARRAY_B16_CLAMP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_array_i32_clamp
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$r),
          (SUST_B_1D_ARRAY_B32_CLAMP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR32:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_array_i64_clamp
           GPR64:$s, GPR32:$l, GPR32:$x, GPR64:$r),
          (SUST_B_1D_ARRAY_B64_CLAMP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR64:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v2i8_clamp
          GPR64:$s, GPR32:$l, GPR32:$x, GPR16:$r, GPR16:$g),
          (SUST_B_1D_ARRAY_V2B8_CLAMP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v2i16_clamp
          GPR64:$s, GPR32:$l, GPR32:$x, GPR16:$r, GPR16:$g),
          (SUST_B_1D_ARRAY_V2B16_CLAMP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v2i32_clamp
          GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$r, GPR32:$g),
          (SUST_B_1D_ARRAY_V2B32_CLAMP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR32:$r, GPR32:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v2i64_clamp
          GPR64:$s, GPR32:$l, GPR32:$x, GPR64:$r, GPR64:$g),
          (SUST_B_1D_ARRAY_V2B64_CLAMP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR64:$r, GPR64:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v4i8_clamp
           GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_1D_ARRAY_V4B8_CLAMP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v4i16_clamp
           GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_1D_ARRAY_V4B16_CLAMP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v4i32_clamp
           GPR64:$s, GPR32:$l, GPR32:$x,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a),
          (SUST_B_1D_ARRAY_V4B32_CLAMP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a)>;



def : Pat<(int_rvgpu_sust_b_2d_i8_clamp
           GPR64:$s, GPR32:$x, GPR32:$y, GPR16:$r),
          (SUST_B_2D_B8_CLAMP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_i16_clamp
           GPR64:$s, GPR32:$x, GPR32:$y, GPR16:$r),
          (SUST_B_2D_B16_CLAMP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_i32_clamp
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$r),
          (SUST_B_2D_B32_CLAMP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR32:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_i64_clamp
           GPR64:$s, GPR32:$x, GPR32:$y, GPR64:$r),
          (SUST_B_2D_B64_CLAMP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR64:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_v2i8_clamp
          GPR64:$s, GPR32:$x, GPR32:$y, GPR16:$r, GPR16:$g),
          (SUST_B_2D_V2B8_CLAMP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_v2i16_clamp
          GPR64:$s, GPR32:$x, GPR32:$y, GPR16:$r, GPR16:$g),
          (SUST_B_2D_V2B16_CLAMP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_v2i32_clamp
          GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$r, GPR32:$g),
          (SUST_B_2D_V2B32_CLAMP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR32:$r, GPR32:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_v2i64_clamp
          GPR64:$s, GPR32:$x, GPR32:$y, GPR64:$r, GPR64:$g),
          (SUST_B_2D_V2B64_CLAMP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR64:$r, GPR64:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_v4i8_clamp
           GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_2D_V4B8_CLAMP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_2d_v4i16_clamp
           GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_2D_V4B16_CLAMP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_2d_v4i32_clamp
           GPR64:$s, GPR32:$x, GPR32:$y,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a),
          (SUST_B_2D_V4B32_CLAMP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a)>;



def : Pat<(int_rvgpu_sust_b_2d_array_i8_clamp
          GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR16:$r),
          (SUST_B_2D_ARRAY_B8_CLAMP_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_array_i16_clamp
          GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR16:$r),
          (SUST_B_2D_ARRAY_B16_CLAMP_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_array_i32_clamp
          GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR32:$r),
          (SUST_B_2D_ARRAY_B32_CLAMP_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR32:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_array_i64_clamp
          GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR64:$r),
          (SUST_B_2D_ARRAY_B64_CLAMP_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR64:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v2i8_clamp
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g),
          (SUST_B_2D_ARRAY_V2B8_CLAMP_R GPR64:$s, GPR32:$l,
           GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v2i16_clamp
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g),
          (SUST_B_2D_ARRAY_V2B16_CLAMP_R GPR64:$s, GPR32:$l,
           GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v2i32_clamp
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR32:$r,
           GPR32:$g),
          (SUST_B_2D_ARRAY_V2B32_CLAMP_R GPR64:$s, GPR32:$l,
           GPR32:$x, GPR32:$y, GPR32:$r, GPR32:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v2i64_clamp
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR64:$r,
           GPR64:$g),
          (SUST_B_2D_ARRAY_V2B64_CLAMP_R GPR64:$s, GPR32:$l,
           GPR32:$x, GPR32:$y, GPR64:$r, GPR64:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v4i8_clamp
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_2D_ARRAY_V4B8_CLAMP_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v4i16_clamp
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_2D_ARRAY_V4B16_CLAMP_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v4i32_clamp
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a),
          (SUST_B_2D_ARRAY_V4B32_CLAMP_R GPR64:$s, GPR32:$l,
           GPR32:$x, GPR32:$y,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a)>;



def : Pat<(int_rvgpu_sust_b_3d_i8_clamp
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r),
          (SUST_B_3D_B8_CLAMP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_3d_i16_clamp
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r),
          (SUST_B_3D_B16_CLAMP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_3d_i32_clamp
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r),
          (SUST_B_3D_B32_CLAMP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r)>;

def : Pat<(int_rvgpu_sust_b_3d_i64_clamp
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR64:$r),
          (SUST_B_3D_B64_CLAMP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR64:$r)>;

def : Pat<(int_rvgpu_sust_b_3d_v2i8_clamp
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g),
          (SUST_B_3D_V2B8_CLAMP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_3d_v2i16_clamp
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g),
          (SUST_B_3D_V2B16_CLAMP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_3d_v2i32_clamp
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r, GPR32:$g),
          (SUST_B_3D_V2B32_CLAMP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r, GPR32:$g)>;

def : Pat<(int_rvgpu_sust_b_3d_v2i64_clamp
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR64:$r, GPR64:$g),
          (SUST_B_3D_V2B64_CLAMP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR64:$r, GPR64:$g)>;

def : Pat<(int_rvgpu_sust_b_3d_v4i8_clamp
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_3D_V4B8_CLAMP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_3d_v4i16_clamp
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_3D_V4B16_CLAMP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_3d_v4i32_clamp
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a),
          (SUST_B_3D_V4B32_CLAMP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a)>;


// .trap variant
def : Pat<(int_rvgpu_sust_b_1d_i8_trap
           GPR64:$s, GPR32:$x, GPR16:$r),
          (SUST_B_1D_B8_TRAP_R GPR64:$s, GPR32:$x, GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_i16_trap
           GPR64:$s, GPR32:$x, GPR16:$r),
          (SUST_B_1D_B16_TRAP_R GPR64:$s, GPR32:$x, GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_i32_trap
           GPR64:$s, GPR32:$x, GPR32:$r),
          (SUST_B_1D_B32_TRAP_R GPR64:$s, GPR32:$x, GPR32:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_i64_trap
           GPR64:$s, GPR32:$x, GPR64:$r),
          (SUST_B_1D_B64_TRAP_R GPR64:$s, GPR32:$x, GPR64:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_v2i8_trap
           GPR64:$s, GPR32:$x, GPR16:$r, GPR16:$g),
          (SUST_B_1D_V2B8_TRAP_R GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_v2i16_trap
           GPR64:$s, GPR32:$x, GPR16:$r, GPR16:$g),
          (SUST_B_1D_V2B16_TRAP_R GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_v2i32_trap
           GPR64:$s, GPR32:$x, GPR32:$r, GPR32:$g),
          (SUST_B_1D_V2B32_TRAP_R GPR64:$s, GPR32:$x,
           GPR32:$r, GPR32:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_v2i64_trap
           GPR64:$s, GPR32:$x, GPR64:$r, GPR64:$g),
          (SUST_B_1D_V2B64_TRAP_R GPR64:$s, GPR32:$x,
           GPR64:$r, GPR64:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_v4i8_trap
           GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_1D_V4B8_TRAP_R GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_1d_v4i16_trap
           GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_1D_V4B16_TRAP_R GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_1d_v4i32_trap
           GPR64:$s, GPR32:$x,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a),
          (SUST_B_1D_V4B32_TRAP_R GPR64:$s, GPR32:$x,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a)>;



def : Pat<(int_rvgpu_sust_b_1d_array_i8_trap
           GPR64:$s, GPR32:$l, GPR32:$x, GPR16:$r),
          (SUST_B_1D_ARRAY_B8_TRAP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_array_i16_trap
           GPR64:$s, GPR32:$l, GPR32:$x, GPR16:$r),
          (SUST_B_1D_ARRAY_B16_TRAP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_array_i32_trap
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$r),
          (SUST_B_1D_ARRAY_B32_TRAP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR32:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_array_i64_trap
           GPR64:$s, GPR32:$l, GPR32:$x, GPR64:$r),
          (SUST_B_1D_ARRAY_B64_TRAP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR64:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v2i8_trap
          GPR64:$s, GPR32:$l, GPR32:$x, GPR16:$r, GPR16:$g),
          (SUST_B_1D_ARRAY_V2B8_TRAP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v2i16_trap
          GPR64:$s, GPR32:$l, GPR32:$x, GPR16:$r, GPR16:$g),
          (SUST_B_1D_ARRAY_V2B16_TRAP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v2i32_trap
          GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$r, GPR32:$g),
          (SUST_B_1D_ARRAY_V2B32_TRAP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR32:$r, GPR32:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v2i64_trap
          GPR64:$s, GPR32:$l, GPR32:$x, GPR64:$r, GPR64:$g),
          (SUST_B_1D_ARRAY_V2B64_TRAP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR64:$r, GPR64:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v4i8_trap
           GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_1D_ARRAY_V4B8_TRAP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v4i16_trap
           GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_1D_ARRAY_V4B16_TRAP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v4i32_trap
           GPR64:$s, GPR32:$l, GPR32:$x,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a),
          (SUST_B_1D_ARRAY_V4B32_TRAP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a)>;



def : Pat<(int_rvgpu_sust_b_2d_i8_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR16:$r),
          (SUST_B_2D_B8_TRAP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_i16_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR16:$r),
          (SUST_B_2D_B16_TRAP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_i32_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$r),
          (SUST_B_2D_B32_TRAP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR32:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_i64_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR64:$r),
          (SUST_B_2D_B64_TRAP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR64:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_v2i8_trap
          GPR64:$s, GPR32:$x, GPR32:$y, GPR16:$r, GPR16:$g),
          (SUST_B_2D_V2B8_TRAP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_v2i16_trap
          GPR64:$s, GPR32:$x, GPR32:$y, GPR16:$r, GPR16:$g),
          (SUST_B_2D_V2B16_TRAP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_v2i32_trap
          GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$r, GPR32:$g),
          (SUST_B_2D_V2B32_TRAP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR32:$r, GPR32:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_v2i64_trap
          GPR64:$s, GPR32:$x, GPR32:$y, GPR64:$r, GPR64:$g),
          (SUST_B_2D_V2B64_TRAP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR64:$r, GPR64:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_v4i8_trap
           GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_2D_V4B8_TRAP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_2d_v4i16_trap
           GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_2D_V4B16_TRAP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_2d_v4i32_trap
           GPR64:$s, GPR32:$x, GPR32:$y,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a),
          (SUST_B_2D_V4B32_TRAP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a)>;



def : Pat<(int_rvgpu_sust_b_2d_array_i8_trap
          GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR16:$r),
          (SUST_B_2D_ARRAY_B8_TRAP_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_array_i16_trap
          GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR16:$r),
          (SUST_B_2D_ARRAY_B16_TRAP_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_array_i32_trap
          GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR32:$r),
          (SUST_B_2D_ARRAY_B32_TRAP_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR32:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_array_i64_trap
          GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR64:$r),
          (SUST_B_2D_ARRAY_B64_TRAP_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR64:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v2i8_trap
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g),
          (SUST_B_2D_ARRAY_V2B8_TRAP_R GPR64:$s, GPR32:$l,
           GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v2i16_trap
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g),
          (SUST_B_2D_ARRAY_V2B16_TRAP_R GPR64:$s, GPR32:$l,
           GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v2i32_trap
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR32:$r,
           GPR32:$g),
          (SUST_B_2D_ARRAY_V2B32_TRAP_R GPR64:$s, GPR32:$l,
           GPR32:$x, GPR32:$y, GPR32:$r, GPR32:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v2i64_trap
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR64:$r,
           GPR64:$g),
          (SUST_B_2D_ARRAY_V2B64_TRAP_R GPR64:$s, GPR32:$l,
           GPR32:$x, GPR32:$y, GPR64:$r, GPR64:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v4i8_trap
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_2D_ARRAY_V4B8_TRAP_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v4i16_trap
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_2D_ARRAY_V4B16_TRAP_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v4i32_trap
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a),
          (SUST_B_2D_ARRAY_V4B32_TRAP_R GPR64:$s, GPR32:$l,
           GPR32:$x, GPR32:$y,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a)>;



def : Pat<(int_rvgpu_sust_b_3d_i8_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r),
          (SUST_B_3D_B8_TRAP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_3d_i16_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r),
          (SUST_B_3D_B16_TRAP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_3d_i32_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r),
          (SUST_B_3D_B32_TRAP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r)>;

def : Pat<(int_rvgpu_sust_b_3d_i64_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR64:$r),
          (SUST_B_3D_B64_TRAP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR64:$r)>;

def : Pat<(int_rvgpu_sust_b_3d_v2i8_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g),
          (SUST_B_3D_V2B8_TRAP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_3d_v2i16_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g),
          (SUST_B_3D_V2B16_TRAP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_3d_v2i32_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r, GPR32:$g),
          (SUST_B_3D_V2B32_TRAP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r, GPR32:$g)>;

def : Pat<(int_rvgpu_sust_b_3d_v2i64_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR64:$r, GPR64:$g),
          (SUST_B_3D_V2B64_TRAP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR64:$r, GPR64:$g)>;

def : Pat<(int_rvgpu_sust_b_3d_v4i8_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_3D_V4B8_TRAP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_3d_v4i16_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_3D_V4B16_TRAP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_3d_v4i32_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a),
          (SUST_B_3D_V4B32_TRAP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a)>;


// .zero variant
def : Pat<(int_rvgpu_sust_b_1d_i8_zero
           GPR64:$s, GPR32:$x, GPR16:$r),
          (SUST_B_1D_B8_ZERO_R GPR64:$s, GPR32:$x, GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_i16_zero
           GPR64:$s, GPR32:$x, GPR16:$r),
          (SUST_B_1D_B16_ZERO_R GPR64:$s, GPR32:$x, GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_i32_zero
           GPR64:$s, GPR32:$x, GPR32:$r),
          (SUST_B_1D_B32_ZERO_R GPR64:$s, GPR32:$x, GPR32:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_i64_zero
           GPR64:$s, GPR32:$x, GPR64:$r),
          (SUST_B_1D_B64_ZERO_R GPR64:$s, GPR32:$x, GPR64:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_v2i8_zero
           GPR64:$s, GPR32:$x, GPR16:$r, GPR16:$g),
          (SUST_B_1D_V2B8_ZERO_R GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_v2i16_zero
           GPR64:$s, GPR32:$x, GPR16:$r, GPR16:$g),
          (SUST_B_1D_V2B16_ZERO_R GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_v2i32_zero
           GPR64:$s, GPR32:$x, GPR32:$r, GPR32:$g),
          (SUST_B_1D_V2B32_ZERO_R GPR64:$s, GPR32:$x,
           GPR32:$r, GPR32:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_v2i64_zero
           GPR64:$s, GPR32:$x, GPR64:$r, GPR64:$g),
          (SUST_B_1D_V2B64_ZERO_R GPR64:$s, GPR32:$x,
           GPR64:$r, GPR64:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_v4i8_zero
           GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_1D_V4B8_ZERO_R GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_1d_v4i16_zero
           GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_1D_V4B16_ZERO_R GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_1d_v4i32_zero
           GPR64:$s, GPR32:$x,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a),
          (SUST_B_1D_V4B32_ZERO_R GPR64:$s, GPR32:$x,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a)>;



def : Pat<(int_rvgpu_sust_b_1d_array_i8_zero
           GPR64:$s, GPR32:$l, GPR32:$x, GPR16:$r),
          (SUST_B_1D_ARRAY_B8_ZERO_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_array_i16_zero
           GPR64:$s, GPR32:$l, GPR32:$x, GPR16:$r),
          (SUST_B_1D_ARRAY_B16_ZERO_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_array_i32_zero
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$r),
          (SUST_B_1D_ARRAY_B32_ZERO_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR32:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_array_i64_zero
           GPR64:$s, GPR32:$l, GPR32:$x, GPR64:$r),
          (SUST_B_1D_ARRAY_B64_ZERO_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR64:$r)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v2i8_zero
          GPR64:$s, GPR32:$l, GPR32:$x, GPR16:$r, GPR16:$g),
          (SUST_B_1D_ARRAY_V2B8_ZERO_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v2i16_zero
          GPR64:$s, GPR32:$l, GPR32:$x, GPR16:$r, GPR16:$g),
          (SUST_B_1D_ARRAY_V2B16_ZERO_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v2i32_zero
          GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$r, GPR32:$g),
          (SUST_B_1D_ARRAY_V2B32_ZERO_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR32:$r, GPR32:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v2i64_zero
          GPR64:$s, GPR32:$l, GPR32:$x, GPR64:$r, GPR64:$g),
          (SUST_B_1D_ARRAY_V2B64_ZERO_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR64:$r, GPR64:$g)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v4i8_zero
           GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_1D_ARRAY_V4B8_ZERO_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v4i16_zero
           GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_1D_ARRAY_V4B16_ZERO_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_1d_array_v4i32_zero
           GPR64:$s, GPR32:$l, GPR32:$x,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a),
          (SUST_B_1D_ARRAY_V4B32_ZERO_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a)>;



def : Pat<(int_rvgpu_sust_b_2d_i8_zero
           GPR64:$s, GPR32:$x, GPR32:$y, GPR16:$r),
          (SUST_B_2D_B8_ZERO_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_i16_zero
           GPR64:$s, GPR32:$x, GPR32:$y, GPR16:$r),
          (SUST_B_2D_B16_ZERO_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_i32_zero
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$r),
          (SUST_B_2D_B32_ZERO_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR32:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_i64_zero
           GPR64:$s, GPR32:$x, GPR32:$y, GPR64:$r),
          (SUST_B_2D_B64_ZERO_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR64:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_v2i8_zero
          GPR64:$s, GPR32:$x, GPR32:$y, GPR16:$r, GPR16:$g),
          (SUST_B_2D_V2B8_ZERO_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_v2i16_zero
          GPR64:$s, GPR32:$x, GPR32:$y, GPR16:$r, GPR16:$g),
          (SUST_B_2D_V2B16_ZERO_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_v2i32_zero
          GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$r, GPR32:$g),
          (SUST_B_2D_V2B32_ZERO_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR32:$r, GPR32:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_v2i64_zero
          GPR64:$s, GPR32:$x, GPR32:$y, GPR64:$r, GPR64:$g),
          (SUST_B_2D_V2B64_ZERO_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR64:$r, GPR64:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_v4i8_zero
           GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_2D_V4B8_ZERO_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_2d_v4i16_zero
           GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_2D_V4B16_ZERO_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_2d_v4i32_zero
           GPR64:$s, GPR32:$x, GPR32:$y,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a),
          (SUST_B_2D_V4B32_ZERO_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a)>;



def : Pat<(int_rvgpu_sust_b_2d_array_i8_zero
          GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR16:$r),
          (SUST_B_2D_ARRAY_B8_ZERO_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_array_i16_zero
          GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR16:$r),
          (SUST_B_2D_ARRAY_B16_ZERO_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_array_i32_zero
          GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR32:$r),
          (SUST_B_2D_ARRAY_B32_ZERO_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR32:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_array_i64_zero
          GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR64:$r),
          (SUST_B_2D_ARRAY_B64_ZERO_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR64:$r)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v2i8_zero
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g),
          (SUST_B_2D_ARRAY_V2B8_ZERO_R GPR64:$s, GPR32:$l,
           GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v2i16_zero
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g),
          (SUST_B_2D_ARRAY_V2B16_ZERO_R GPR64:$s, GPR32:$l,
           GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v2i32_zero
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR32:$r,
           GPR32:$g),
          (SUST_B_2D_ARRAY_V2B32_ZERO_R GPR64:$s, GPR32:$l,
           GPR32:$x, GPR32:$y, GPR32:$r, GPR32:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v2i64_zero
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR64:$r,
           GPR64:$g),
          (SUST_B_2D_ARRAY_V2B64_ZERO_R GPR64:$s, GPR32:$l,
           GPR32:$x, GPR32:$y, GPR64:$r, GPR64:$g)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v4i8_zero
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_2D_ARRAY_V4B8_ZERO_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v4i16_zero
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_2D_ARRAY_V4B16_ZERO_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_2d_array_v4i32_zero
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a),
          (SUST_B_2D_ARRAY_V4B32_ZERO_R GPR64:$s, GPR32:$l,
           GPR32:$x, GPR32:$y,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a)>;



def : Pat<(int_rvgpu_sust_b_3d_i8_zero
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r),
          (SUST_B_3D_B8_ZERO_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_3d_i16_zero
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r),
          (SUST_B_3D_B16_ZERO_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_b_3d_i32_zero
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r),
          (SUST_B_3D_B32_ZERO_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r)>;

def : Pat<(int_rvgpu_sust_b_3d_i64_zero
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR64:$r),
          (SUST_B_3D_B64_ZERO_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR64:$r)>;

def : Pat<(int_rvgpu_sust_b_3d_v2i8_zero
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g),
          (SUST_B_3D_V2B8_ZERO_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_3d_v2i16_zero
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g),
          (SUST_B_3D_V2B16_ZERO_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_b_3d_v2i32_zero
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r, GPR32:$g),
          (SUST_B_3D_V2B32_ZERO_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r, GPR32:$g)>;

def : Pat<(int_rvgpu_sust_b_3d_v2i64_zero
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR64:$r, GPR64:$g),
          (SUST_B_3D_V2B64_ZERO_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR64:$r, GPR64:$g)>;

def : Pat<(int_rvgpu_sust_b_3d_v4i8_zero
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_3D_V4B8_ZERO_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_3d_v4i16_zero
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_B_3D_V4B16_ZERO_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_b_3d_v4i32_zero
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a),
          (SUST_B_3D_V4B32_ZERO_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a)>;




def : Pat<(int_rvgpu_sust_p_1d_i8_trap
           GPR64:$s, GPR32:$x, GPR16:$r),
          (SUST_P_1D_B8_TRAP_R GPR64:$s, GPR32:$x, GPR16:$r)>;

def : Pat<(int_rvgpu_sust_p_1d_i16_trap
           GPR64:$s, GPR32:$x, GPR16:$r),
          (SUST_P_1D_B16_TRAP_R GPR64:$s, GPR32:$x, GPR16:$r)>;

def : Pat<(int_rvgpu_sust_p_1d_i32_trap
           GPR64:$s, GPR32:$x, GPR32:$r),
          (SUST_P_1D_B32_TRAP_R GPR64:$s, GPR32:$x, GPR32:$r)>;

def : Pat<(int_rvgpu_sust_p_1d_v2i8_trap
           GPR64:$s, GPR32:$x, GPR16:$r, GPR16:$g),
          (SUST_P_1D_V2B8_TRAP_R GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_p_1d_v2i16_trap
           GPR64:$s, GPR32:$x, GPR16:$r, GPR16:$g),
          (SUST_P_1D_V2B16_TRAP_R GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_p_1d_v2i32_trap
           GPR64:$s, GPR32:$x, GPR32:$r, GPR32:$g),
          (SUST_P_1D_V2B32_TRAP_R GPR64:$s, GPR32:$x,
           GPR32:$r, GPR32:$g)>;

def : Pat<(int_rvgpu_sust_p_1d_v4i8_trap
           GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_P_1D_V4B8_TRAP_R GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_p_1d_v4i16_trap
           GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_P_1D_V4B16_TRAP_R GPR64:$s, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_p_1d_v4i32_trap
           GPR64:$s, GPR32:$x,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a),
          (SUST_P_1D_V4B32_TRAP_R GPR64:$s, GPR32:$x,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a)>;



def : Pat<(int_rvgpu_sust_p_1d_array_i8_trap
           GPR64:$s, GPR32:$l, GPR32:$x, GPR16:$r),
          (SUST_P_1D_ARRAY_B8_TRAP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_p_1d_array_i16_trap
           GPR64:$s, GPR32:$l, GPR32:$x, GPR16:$r),
          (SUST_P_1D_ARRAY_B16_TRAP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_p_1d_array_i32_trap
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$r),
          (SUST_P_1D_ARRAY_B32_TRAP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR32:$r)>;

def : Pat<(int_rvgpu_sust_p_1d_array_v2i8_trap
          GPR64:$s, GPR32:$l, GPR32:$x, GPR16:$r, GPR16:$g),
          (SUST_P_1D_ARRAY_V2B8_TRAP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_p_1d_array_v2i16_trap
          GPR64:$s, GPR32:$l, GPR32:$x, GPR16:$r, GPR16:$g),
          (SUST_P_1D_ARRAY_V2B16_TRAP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_p_1d_array_v2i32_trap
          GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$r, GPR32:$g),
          (SUST_P_1D_ARRAY_V2B32_TRAP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR32:$r, GPR32:$g)>;

def : Pat<(int_rvgpu_sust_p_1d_array_v4i8_trap
           GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_P_1D_ARRAY_V4B8_TRAP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_p_1d_array_v4i16_trap
           GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_P_1D_ARRAY_V4B16_TRAP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_p_1d_array_v4i32_trap
           GPR64:$s, GPR32:$l, GPR32:$x,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a),
          (SUST_P_1D_ARRAY_V4B32_TRAP_R GPR64:$s, GPR32:$l, GPR32:$x,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a)>;



def : Pat<(int_rvgpu_sust_p_2d_i8_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR16:$r),
          (SUST_P_2D_B8_TRAP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_p_2d_i16_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR16:$r),
          (SUST_P_2D_B16_TRAP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_p_2d_i32_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$r),
          (SUST_P_2D_B32_TRAP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR32:$r)>;

def : Pat<(int_rvgpu_sust_p_2d_v2i8_trap
          GPR64:$s, GPR32:$x, GPR32:$y, GPR16:$r, GPR16:$g),
          (SUST_P_2D_V2B8_TRAP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_p_2d_v2i16_trap
          GPR64:$s, GPR32:$x, GPR32:$y, GPR16:$r, GPR16:$g),
          (SUST_P_2D_V2B16_TRAP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_p_2d_v2i32_trap
          GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$r, GPR32:$g),
          (SUST_P_2D_V2B32_TRAP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR32:$r, GPR32:$g)>;

def : Pat<(int_rvgpu_sust_p_2d_v4i8_trap
           GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_P_2D_V4B8_TRAP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_p_2d_v4i16_trap
           GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_P_2D_V4B16_TRAP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_p_2d_v4i32_trap
           GPR64:$s, GPR32:$x, GPR32:$y,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a),
          (SUST_P_2D_V4B32_TRAP_R GPR64:$s, GPR32:$x, GPR32:$y,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a)>;



def : Pat<(int_rvgpu_sust_p_2d_array_i8_trap
          GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR16:$r),
          (SUST_P_2D_ARRAY_B8_TRAP_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_p_2d_array_i16_trap
          GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR16:$r),
          (SUST_P_2D_ARRAY_B16_TRAP_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_p_2d_array_i32_trap
          GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR32:$r),
          (SUST_P_2D_ARRAY_B32_TRAP_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR32:$r)>;

def : Pat<(int_rvgpu_sust_p_2d_array_v2i8_trap
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g),
          (SUST_P_2D_ARRAY_V2B8_TRAP_R GPR64:$s, GPR32:$l,
           GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_p_2d_array_v2i16_trap
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g),
          (SUST_P_2D_ARRAY_V2B16_TRAP_R GPR64:$s, GPR32:$l,
           GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_p_2d_array_v2i32_trap
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y, GPR32:$r,
           GPR32:$g),
          (SUST_P_2D_ARRAY_V2B32_TRAP_R GPR64:$s, GPR32:$l,
           GPR32:$x, GPR32:$y, GPR32:$r, GPR32:$g)>;

def : Pat<(int_rvgpu_sust_p_2d_array_v4i8_trap
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_P_2D_ARRAY_V4B8_TRAP_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_p_2d_array_v4i16_trap
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_P_2D_ARRAY_V4B16_TRAP_R GPR64:$s,
           GPR32:$l, GPR32:$x, GPR32:$y,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_p_2d_array_v4i32_trap
           GPR64:$s, GPR32:$l, GPR32:$x, GPR32:$y,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a),
          (SUST_P_2D_ARRAY_V4B32_TRAP_R GPR64:$s, GPR32:$l,
           GPR32:$x, GPR32:$y,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a)>;



def : Pat<(int_rvgpu_sust_p_3d_i8_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r),
          (SUST_P_3D_B8_TRAP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_p_3d_i16_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r),
          (SUST_P_3D_B16_TRAP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r)>;

def : Pat<(int_rvgpu_sust_p_3d_i32_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r),
          (SUST_P_3D_B32_TRAP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r)>;

def : Pat<(int_rvgpu_sust_p_3d_v2i8_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g),
          (SUST_P_3D_V2B8_TRAP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_p_3d_v2i16_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g),
          (SUST_P_3D_V2B16_TRAP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g)>;

def : Pat<(int_rvgpu_sust_p_3d_v2i32_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r, GPR32:$g),
          (SUST_P_3D_V2B32_TRAP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r, GPR32:$g)>;

def : Pat<(int_rvgpu_sust_p_3d_v4i8_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_P_3D_V4B8_TRAP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_p_3d_v4i16_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a),
          (SUST_P_3D_V4B16_TRAP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR16:$r, GPR16:$g, GPR16:$b, GPR16:$a)>;

def : Pat<(int_rvgpu_sust_p_3d_v4i32_trap
           GPR64:$s, GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a),
          (SUST_P_3D_V4B32_TRAP_R GPR64:$s,
           GPR32:$x, GPR32:$y, GPR32:$z,
           GPR32:$r, GPR32:$g, GPR32:$b, GPR32:$a)>;

//-----------------------------------
// Read Special Registers
//-----------------------------------

class PTX_READ_SREG_R64<string regname, Intrinsic intop, list<Predicate> Preds=[]>
  : RVGPUInst<(outs GPR64:$d), (ins),
              !strconcat("mov.u64 \t$d, ", regname, ""),
              [(set GPR64:$d, (intop))]>,
    Requires<Preds>;

class PTX_READ_SREG_R32<string regname, Intrinsic intop, list<Predicate> Preds=[]>
  : RVGPUInst<(outs GPR32:$d), (ins),
              !strconcat("mov.u32 \t$d, ", regname, ""),
              [(set GPR32:$d, (intop))]>,
    Requires<Preds>;

multiclass PTX_READ_SREG_R32V4<string regname, list<Predicate> Preds=[]> {
   foreach suffix = ["x", "y", "z", "w"] in {
      defvar reg = regname # "." # suffix;
      defvar intr = !cast<Intrinsic>("int_rvgpu_read_ptx_sreg_" # regname # "_" # suffix);
      def "_"#suffix :  PTX_READ_SREG_R32<reg, intr, Preds>;
   }
}

// TODO Add read vector-version of special registers

defm INT_PTX_SREG_TID   : PTX_READ_SREG_R32V4<"tid">;
defm INT_PTX_SREG_NTID  : PTX_READ_SREG_R32V4<"ntid">;
defm INT_PTX_SREG_CTAID : PTX_READ_SREG_R32V4<"ctaid">;
defm INT_PTX_SREG_NCTAID: PTX_READ_SREG_R32V4<"nctaid">;

defm INT_PTX_SREG_CLUSTERID :
       PTX_READ_SREG_R32V4<"clusterid", [hasSM<90>, hasPTX<78>]>;
defm INT_PTX_SREG_NCLUSTERID :
       PTX_READ_SREG_R32V4<"nclusterid", [hasSM<90>, hasPTX<78>]>;
defm INT_PTX_SREG_CLUSTER_CTAID :
       PTX_READ_SREG_R32V4<"cluster_ctaid", [hasSM<90>, hasPTX<78>]>;
defm INT_PTX_SREG_CLUSTER_NCTAID:
       PTX_READ_SREG_R32V4<"cluster_nctaid", [hasSM<90>, hasPTX<78>]>;

def  INT_PTX_SREG_CLUSTER_CTARANK :
       PTX_READ_SREG_R32<"cluster_ctarank",
                         int_rvgpu_read_ptx_sreg_cluster_ctarank,
                         [hasSM<90>, hasPTX<78>]>;
def  INT_PTX_SREG_CLUSTER_NCTARANK:
       PTX_READ_SREG_R32<"cluster_nctarank",
                         int_rvgpu_read_ptx_sreg_cluster_nctarank,
                         [hasSM<90>, hasPTX<78>]>;


def INT_PTX_SREG_LANEID :
    PTX_READ_SREG_R32<"laneid", int_rvgpu_read_ptx_sreg_laneid>;
def INT_PTX_SREG_WARPID :
    PTX_READ_SREG_R32<"warpid", int_rvgpu_read_ptx_sreg_warpid>;
def INT_PTX_SREG_NWARPID :
    PTX_READ_SREG_R32<"nwarpid", int_rvgpu_read_ptx_sreg_nwarpid>;
def INT_PTX_SREG_SMID :
    PTX_READ_SREG_R32<"smid", int_rvgpu_read_ptx_sreg_smid>;
def INT_PTX_SREG_NSMID :
    PTX_READ_SREG_R32<"nsmid", int_rvgpu_read_ptx_sreg_nsmid>;
def INT_PTX_SREG_GRIDID :
    PTX_READ_SREG_R32<"gridid", int_rvgpu_read_ptx_sreg_gridid>;

def INT_PTX_SREG_LANEMASK_EQ :
    PTX_READ_SREG_R32<"lanemask_eq", int_rvgpu_read_ptx_sreg_lanemask_eq>;
def INT_PTX_SREG_LANEMASK_LE :
    PTX_READ_SREG_R32<"lanemask_le", int_rvgpu_read_ptx_sreg_lanemask_le>;
def INT_PTX_SREG_LANEMASK_LT :
    PTX_READ_SREG_R32<"lanemask_lt", int_rvgpu_read_ptx_sreg_lanemask_lt>;
def INT_PTX_SREG_LANEMASK_GE :
    PTX_READ_SREG_R32<"lanemask_ge", int_rvgpu_read_ptx_sreg_lanemask_ge>;
def INT_PTX_SREG_LANEMASK_GT :
    PTX_READ_SREG_R32<"lanemask_gt", int_rvgpu_read_ptx_sreg_lanemask_gt>;

def INT_PTX_SREG_CLOCK :
    PTX_READ_SREG_R32<"clock", int_rvgpu_read_ptx_sreg_clock>;
def INT_PTX_SREG_CLOCK64 :
    PTX_READ_SREG_R64<"clock64", int_rvgpu_read_ptx_sreg_clock64>;

def INT_PTX_SREG_PM0 : PTX_READ_SREG_R32<"pm0", int_rvgpu_read_ptx_sreg_pm0>;
def INT_PTX_SREG_PM1 : PTX_READ_SREG_R32<"pm1", int_rvgpu_read_ptx_sreg_pm1>;
def INT_PTX_SREG_PM2 : PTX_READ_SREG_R32<"pm2", int_rvgpu_read_ptx_sreg_pm2>;
def INT_PTX_SREG_PM3 : PTX_READ_SREG_R32<"pm3", int_rvgpu_read_ptx_sreg_pm3>;

// TODO: It would be nice to use PTX_READ_SREG here, but it doesn't
// handle the constant.
def INT_PTX_SREG_WARPSIZE :
    RVGPUInst<(outs GPR32:$dst), (ins), "mov.u32 \t$dst, WARP_SZ",
              [(set GPR32:$dst, (int_rvgpu_read_ptx_sreg_warpsize))]>;

// Helper class that represents a 'fragment' of an RVGPU *MMA instruction.
// In addition to target-independent fields provided by WMMA_REGS, it adds
// the fields commonly used to implement specific PTX instruction -- register
// types and names, constraints, parts of assembly, etc.
class WMMA_REGINFO<WMMA_REGS r, string op>
      : WMMA_REGS<r.geom, r.frag, r.ptx_elt_type> {
  // RVGPU register types used to carry fragment data.
  RVGPURegClass regclass = !cond(
    !eq(ptx_elt_type, "f16") : GPR32,
    !eq(ptx_elt_type, "f32") : GPR32,
    !eq(ptx_elt_type, "f64") : GPR64,
    !eq(ptx_elt_type, "bf16") : GPR32,
    !eq(ptx_elt_type, "tf32") : GPR32,
    !eq(ptx_elt_type, "s32") : GPR32,
    !eq(ptx_elt_type, "b16") : GPR32,
    !eq(ptx_elt_type, "s8") : GPR32,
    !eq(ptx_elt_type, "u8") : GPR32,
    !eq(ptx_elt_type, "s4") : GPR32,
    !eq(ptx_elt_type, "u4") : GPR32,
    !eq(ptx_elt_type, "b1") : GPR32);

  // Instruction input/output arguments for the fragment.
  list<RVGPURegClass> ptx_regs = !listsplat(regclass, !size(regs));

  // List of register names for the fragment -- ["ra0", "ra1",...]
  list<string> reg_names = RegSeq<!size(ptx_regs), "r"#frag>.ret;

  // Generates "{{$r0, $r1,.... $rN-1}}" for use in asm string construction.
  string regstring = "{{$" # !interleave(reg_names, ", $") # "}}";

  // Predicates for particular fragment variant. Technically those are
  // per-instruction predicates, but currently all fragments that can be used in
  // a given instruction are subject to the same constraints, so an instruction
  // can use predicates from any of its fragments. If/when this is no
  // longer the case, we can concat all per-fragment predicates to enforce that
  // all fragments of the instruction are viable.
  list<Predicate> Predicates = !cond(
    // fp16 -> fp16/fp32 @ m16n16k16
    !and(!eq(geom, "m16n16k16"),
         !or(!eq(ptx_elt_type, "f16"),
             !eq(ptx_elt_type, "f32"))) : [hasSM<70>, hasPTX<60>],

    !and(!eq(geom,"m8n8k4"),
         !eq(ptx_elt_type, "f64")) : [hasSM<80>, hasPTX<70>],

    // fp16 -> fp16/fp32 @ m8n32k16/m32n8k16
    !and(!or(!eq(geom, "m8n32k16"),
             !eq(geom, "m32n8k16")),
         !or(!eq(ptx_elt_type, "f16"),
             !eq(ptx_elt_type, "f32"))) : [hasSM<70>, hasPTX<61>],

    // u8/s8 -> s32 @ m16n16k16/m8n32k16/m32n8k16
    !and(!or(!eq(geom,"m16n16k16"),
             !eq(geom,"m8n32k16"),
             !eq(geom,"m32n8k16")),
         !or(!eq(ptx_elt_type, "u8"),
             !eq(ptx_elt_type, "s8"),
             !eq(ptx_elt_type, "s32"))) : [hasSM<72>, hasPTX<63>],

    !and(!or(!eq(geom,"m16n16k16"),
             !eq(geom,"m8n32k16"),
             !eq(geom,"m32n8k16")),
         !eq(ptx_elt_type, "bf16")) : [hasSM<80>, hasPTX<70>],

    !and(!eq(geom,"m16n16k8"),
         !eq(ptx_elt_type, "tf32")) : [hasSM<80>, hasPTX<70>],

    !and(!eq(geom,"m16n16k8"),
         !eq(ptx_elt_type, "f32")) : [hasSM<80>, hasPTX<70>],

    // b1 -> s32 @ m8n8k128(b1)
    !and(!ne(op,"mma"),
         !eq(geom,"m8n8k128")) : [hasSM<75>, hasPTX<63>],

    // u4/s4 -> s32 @ m8n8k32 (u4/s4)
    !and(!ne(op,"mma"),
         !eq(geom,"m8n8k32")) : [hasSM<75>, hasPTX<63>],

    !or(!eq(geom,"m16n8k8"),
        !eq(geom,"m8n8k16")) : [hasSM<75>, hasPTX<65>],

    !and(!ne(ptx_elt_type,"f64"),
         !eq(geom, "m8n8k4")) : [hasSM<70>, hasPTX<64>],

    // mma m8n8k32 requires higher PTX version
    !and(!eq(op,"mma"),
         !eq(geom,"m8n8k32")) : [hasSM<75>, hasPTX<65>],

    !and(!eq(ptx_elt_type,"f64"),
         !eq(geom, "m8n8k4")) : [hasSM<80>, hasPTX<70>],

    !and(!eq(op,"mma"),
         !or(!eq(geom, "m16n8k16"),
             !eq(geom, "m16n8k4"),
             !eq(geom, "m16n8k32"),
             !eq(geom, "m16n8k64"),
             !eq(geom, "m8n8k128"),
             !eq(geom, "m16n8k128"),
             !eq(geom, "m16n8k256"))) : [hasSM<80>, hasPTX<70>],

    !and(!eq(op,"ldmatrix"),
         !eq(ptx_elt_type,"b16"),
         !eq(geom, "m8n8")) : [hasSM<75>, hasPTX<65>]);

  // template DAGs for instruction inputs/output.
  dag Outs = !dag(outs, ptx_regs, reg_names);
  dag Ins = !dag(ins, ptx_regs, reg_names);
}

// Convert dag of arguments into a dag to match given intrinsic.
class BuildPatternI<Intrinsic Intr, dag Ins> {
  // Build a dag pattern that matches the intrinsic call.
  dag ret = !foreach(tmp, Ins,
                          !subst(imem, ADDRvar,
                          !subst(MEMri64, ADDRri64,
                          !subst(MEMri, ADDRri,
                          !subst(ins, Intr, tmp)))));
}

// Same as above, but uses PatFrag instead of an Intrinsic.
class BuildPatternPF<PatFrag Intr, dag Ins> {
  // Build a dag pattern that matches the intrinsic call.
  dag ret = !foreach(tmp, Ins,
                          !subst(imem, ADDRvar,
                          !subst(MEMri64, ADDRri64,
                          !subst(MEMri, ADDRri,
                          !subst(ins, Intr, tmp)))));
}

// Common WMMA-related fields used for building patterns for all MMA instructions.
class WMMA_INSTR<string _Intr, list<dag> _Args>
  : RVGPUInst<(outs), (ins), "?", []> {
  Intrinsic Intr = !cast<Intrinsic>(_Intr);
  // Concatenate all arguments into a single dag.
  dag Args = !foldl((ins), _Args, a, b, !con(a,b));
  // Pre-build the pattern to match (intrinsic arg0, arg1, ...).
  dag IntrinsicPattern = BuildPatternI<!cast<Intrinsic>(Intr), Args>.ret;
}

//
// wmma.load.[a|b|c].sync.[row|col].m16n16k16[|.global|.shared].[f16|f32]
//

class WMMA_LOAD<WMMA_REGINFO Frag, string Layout, string Space, bit WithStride,
                DAGOperand SrcOp>
  : WMMA_INSTR<WMMA_NAME_LDST<"load", Frag, Layout, WithStride>.record,
                              [!con((ins SrcOp:$src),
                                    !if(WithStride, (ins GPR32:$ldm), (ins)))]>,
    Requires<Frag.Predicates> {
  // Load/store intrinsics are overloaded on pointer's address space.
  // To match the right intrinsic, we need to build AS-constrained PatFrag.
  // Operands is a dag equivalent in shape to Args, but using (ops node:$name, .....).
  dag PFOperands = !if(WithStride, (ops node:$src, node:$ldm), (ops node:$src));
  dag PFOperandsIntr = !if(WithStride, (Intr node:$src, node:$ldm), (Intr node:$src));
  // Build PatFrag that only matches particular address space.
  PatFrag IntrFrag = PatFrag<PFOperands,
                             PFOperandsIntr,
                             !cond(!eq(Space, ".shared"): AS_match.shared,
                                   !eq(Space, ".global"): AS_match.global,
                                   true: AS_match.generic)>;
  // Build AS-constrained pattern.
  let IntrinsicPattern = BuildPatternPF<IntrFrag, Args>.ret;

  let OutOperandList = Frag.Outs;
  let InOperandList = !con(Args, (ins MmaCode:$ptx));
  let AsmString = "wmma.load."
                  # Frag.frag
                  # ".sync"
                  # "${ptx:aligned}"
                  # "." # Layout
                  # "." # Frag.geom
                  # Space
                  # "." # Frag.ptx_elt_type # " \t"
                  # Frag.regstring
                  # ", [$src]"
                  # !if(WithStride, ", $ldm", "")
                  # "";
}

//
// wmma.store.d.sync.[row|col].m16n16k16[|.global|.shared].[f16|f32]
//
class WMMA_STORE_D<WMMA_REGINFO Frag, string Layout, string Space,
                   bit WithStride, DAGOperand DstOp>
  : WMMA_INSTR<WMMA_NAME_LDST<"store", Frag, Layout, WithStride>.record,
               [!con((ins DstOp:$dst),
                     Frag.Ins,
                     !if(WithStride, (ins GPR32:$ldm), (ins)))]>,
    Requires<Frag.Predicates> {

  // Load/store intrinsics are overloaded on pointer's address space.
  // To match the right intrinsic, we need to build AS-constrained PatFrag.
  // Operands is a dag equivalent in shape to Args, but using (ops node:$name, .....).
  dag PFOperands = !con((ops node:$dst),
                        !dag(ops, !listsplat(node, !size(Frag.regs)), Frag.reg_names),
                        !if(WithStride, (ops node:$ldm), (ops)));
  // Build PatFrag that only matches particular address space.
  PatFrag IntrFrag = PatFrag<PFOperands,
                             !foreach(tmp, PFOperands, !subst(ops, Intr, tmp)),
                             !cond(!eq(Space, ".shared"): AS_match.shared,
                                   !eq(Space, ".global"): AS_match.global,
                                   true: AS_match.generic)>;
  // Build AS-constrained pattern.
  let IntrinsicPattern = BuildPatternPF<IntrFrag, Args>.ret;
  let isCodeGenOnly = 1;
  let InOperandList  = !con(Args, (ins MmaCode:$ptx));
  let OutOperandList = (outs);
  let AsmString = "wmma.store.d.sync"
                  # "${ptx:aligned}"
                  # "." # Layout
                  # "." # Frag.geom
                  # Space
                  # "." # Frag.ptx_elt_type
                  # " \t[$dst],"
                  # Frag.regstring
                  # !if(WithStride, ", $ldm", "")
                  # "";
}

// Create all load/store variants
defset list<WMMA_INSTR> MMA_LDSTs  = {
  foreach layout = ["row", "col"] in {
    foreach stride = [false, true] in {
      foreach space = [".global", ".shared", ""] in {
        foreach addr = [imem, GPR32, GPR64, MEMri, MEMri64] in {
          foreach frag = NVVM_MMA_OPS.all_ld_ops in
            if NVVM_WMMA_LDST_SUPPORTED<frag, layout>.ret then
              let isCodeGenOnly = 1 in {
                def : WMMA_LOAD<WMMA_REGINFO<frag, "load">, layout, space, stride, addr>;
              }
          foreach frag = NVVM_MMA_OPS.all_st_ops in
            if NVVM_WMMA_LDST_SUPPORTED<frag, layout>.ret then
              def : WMMA_STORE_D<WMMA_REGINFO<frag, "store">, layout, space, stride, addr>;
        } // addr
      } // space
    } // stride
  } // layout
} // defset

// B1 instruction variants need extra constraints.
class MMA_OP_PREDICATES<WMMA_REGINFO FragA, string b1op> {
  string Op = b1op;
  WMMA_REGINFO Frag = FragA;
  list<Predicate> ret = !listconcat(
    FragA.Predicates,
    !if(!eq(b1op, ".and.popc"), [hasSM<80>,hasPTX<71>],[])
  );
}
// WMMA.MMA
class WMMA_MMA<WMMA_REGINFO FragA, WMMA_REGINFO FragB,
               WMMA_REGINFO FragC, WMMA_REGINFO FragD,
               string ALayout, string BLayout, int Satfinite, string rnd, string b1op>
  : WMMA_INSTR<WMMA_NAME<ALayout, BLayout, Satfinite, rnd, b1op, FragA, FragB, FragC, FragD>.record,
                         [FragA.Ins, FragB.Ins, FragC.Ins]>,
    // Requires does not seem to have effect on Instruction w/o Patterns.
    // We set it here anyways and propagate to the Pat<> we construct below.
    Requires<MMA_OP_PREDICATES<FragA, b1op>.ret> {
  let OutOperandList = FragD.Outs;
  let InOperandList  = !con(Args, (ins MmaCode:$ptx));
  string TypeList = !cond(
    !eq(FragA.ptx_elt_type, "f16") : "." # FragD.ptx_elt_type
                                     # "." # FragC.ptx_elt_type,
    1: "." # FragD.ptx_elt_type
       # "." # FragA.ptx_elt_type
       # "." # FragB.ptx_elt_type
       # "." # FragC.ptx_elt_type,
  );
  let AsmString = "wmma.mma"
                  # b1op
                  # ".sync"
                  # "${ptx:aligned}"
                  # "." # ALayout
                  # "." # BLayout
                  # "." # FragA.geom
                  # !if(!ne(rnd, ""), !strconcat(".", rnd), "")
                  # TypeList
                  # !if(Satfinite, ".satfinite", "") # "\n\t\t"
                  # FragD.regstring # ",\n\t\t"
                  # FragA.regstring # ",\n\t\t"
                  # FragB.regstring # ",\n\t\t"
                  # FragC.regstring # "";
}

defset list<WMMA_INSTR> WMMAs  = {
  foreach layout_a = ["row", "col"] in {
    foreach layout_b = ["row", "col"] in {
      foreach satf = [0, 1] in {
        foreach rnd = ["", "rn", "rz", "rm", "rp"] in {
          foreach op = NVVM_MMA_OPS.all_wmma_ops in {
            foreach b1op = NVVM_MMA_B1OPS<op>.ret in {
              if NVVM_WMMA_SUPPORTED<op, layout_a, layout_b, satf, rnd>.ret then {
                  let isCodeGenOnly = 1 in {
                    def : WMMA_MMA<WMMA_REGINFO<op[0], "wmma.mma">,
                              WMMA_REGINFO<op[1], "wmma.mma">,
                              WMMA_REGINFO<op[2], "wmma.mma">,
                              WMMA_REGINFO<op[3], "wmma.mma">,
                              layout_a, layout_b, satf, rnd, b1op>;
                  }
              }
            } // b1op
          } // op
        } // rnd
      } // satf
    } // layout_b
  } // layout_a
} // defset

// MMA
class MMA<WMMA_REGINFO FragA, WMMA_REGINFO FragB,
               WMMA_REGINFO FragC, WMMA_REGINFO FragD,
               string ALayout, string BLayout, int Satfinite, string b1op>
  : WMMA_INSTR<MMA_NAME<ALayout, BLayout, Satfinite, b1op, FragA, FragB, FragC, FragD>.record,
                        [FragA.Ins, FragB.Ins, FragC.Ins]>,
    // Requires does not seem to have effect on Instruction w/o Patterns.
    // We set it here anyways and propagate to the Pat<> we construct below.
  Requires<MMA_OP_PREDICATES<FragA, b1op>.ret> {
  let isCodeGenOnly = 1;
  let OutOperandList = FragD.Outs;
  let InOperandList  = !con(Args, (ins MmaCode:$ptx));
  string TypeList = "." # FragD.ptx_elt_type
                    # "." # FragA.ptx_elt_type
                    # "." # FragB.ptx_elt_type
                    # "." # FragC.ptx_elt_type;
  let AsmString = "mma.sync.aligned."
                  # FragA.geom
                  # "." # ALayout
                  # "." # BLayout
                  # !if(Satfinite, ".satfinite", "")
                  # TypeList
                  # b1op # "\n\t\t"
                  # FragD.regstring # ",\n\t\t"
                  # FragA.regstring # ",\n\t\t"
                  # FragB.regstring # ",\n\t\t"
                  # FragC.regstring # "";
}

defset list<WMMA_INSTR> MMAs  = {
  foreach layout_a = ["row", "col"] in {
    foreach layout_b = ["row", "col"] in {
      foreach satf = [0, 1] in {
        foreach op = NVVM_MMA_OPS.all_mma_ops in {
          foreach b1op = NVVM_MMA_B1OPS<op>.ret in {
            if NVVM_MMA_SUPPORTED<op, layout_a, layout_b, satf>.ret then {
              def : MMA<WMMA_REGINFO<op[0], "mma">,
                        WMMA_REGINFO<op[1], "mma">,
                        WMMA_REGINFO<op[2], "mma">,
                        WMMA_REGINFO<op[3], "mma">,
                        layout_a, layout_b, satf, b1op>;
            }
          } // b1op
        } // op
      } // satf
    } // layout_b
  } // layout_a
} // defset

//
// ldmatrix.sync.aligned.m8n8[|.trans][|.shared].b16
//
class LDMATRIX<WMMA_REGINFO Frag, bit Transposed, string Space,
               DAGOperand SrcOp>
  : WMMA_INSTR<LDMATRIX_NAME<Frag, Transposed>.record, [(ins SrcOp:$src)]>,
    Requires<Frag.Predicates> {
  // Build PatFrag that only matches particular address space.
  PatFrag IntrFrag = PatFrag<(ops node:$src), (Intr node:$src),
                             !cond(!eq(Space, ".shared"): AS_match.shared,
                                   true: AS_match.generic)>;
  // Build AS-constrained pattern.
  let IntrinsicPattern = BuildPatternPF<IntrFrag, Args>.ret;

  let OutOperandList = Frag.Outs;
  let InOperandList = !con(Args, (ins MmaCode:$ptx));
  let AsmString = "ldmatrix.sync.aligned."
                  # Frag.geom
                  # "." # Frag.frag
                  # !if(Transposed, ".trans", "")
                  # Space
                  # "." # Frag.ptx_elt_type
                  # " " # Frag.regstring # ", [$src]";
}

// Create all ldmatrix variants
defset list<WMMA_INSTR> LDMATRIXs  = {
  foreach transposed = [false, true] in {
    foreach space = [".shared", ""] in {
      foreach addr = [imem, GPR32, GPR64, MEMri, MEMri64] in {
        foreach frag = NVVM_MMA_OPS.all_ldmatrix_ops in
          if NVVM_LDMATRIX_SUPPORTED<frag>.ret then
            let isCodeGenOnly = 1 in {
              def : LDMATRIX<WMMA_REGINFO<frag, "ldmatrix">, transposed, space,
                            addr>;
            }
      } // addr
    } // space
  } // transposed
} // defset

// Constructing non-flat DAGs is still a pain. I can't !subst a dag node with a
// dag, so the ptx.version must be appended *after* foreach replaces 'ins' with
// the instruction record.
class MMA_PAT<WMMA_INSTR wi>
      : Pat<wi.IntrinsicPattern,
            !con(!foreach(tmp, wi.Args, !subst(ins, wi, tmp)),
                 (wi ptx.version))>,
        Requires<wi.Predicates>;

// Build intrinsic->instruction patterns for all MMA instructions.
foreach mma = !listconcat(MMAs, WMMAs, MMA_LDSTs, LDMATRIXs) in
  def : MMA_PAT<mma>;

multiclass MAPA<string suffix, Intrinsic Intr> {
  def _32: RVGPUInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b),
              "mapa" # suffix # ".u32\t$d, $a, $b",
              [(set GPR32:$d, (Intr GPR32:$a, GPR32:$b))]>,
    Requires<[hasSM<90>, hasPTX<78>]>;
  def _32i: RVGPUInst<(outs GPR32:$d), (ins GPR32:$a, i32imm:$b),
              "mapa" # suffix # ".u32\t$d, $a, $b",
              [(set GPR32:$d, (Intr GPR32:$a, imm:$b))]>,
    Requires<[hasSM<90>, hasPTX<78>]>;
  def _64: RVGPUInst<(outs GPR64:$d), (ins GPR64:$a, GPR32:$b),
              "mapa" # suffix # ".u64\t$d, $a, $b",
              [(set GPR64:$d, (Intr GPR64:$a, GPR32:$b))]>,
    Requires<[hasSM<90>, hasPTX<78>]>;
  def _64i: RVGPUInst<(outs GPR64:$d), (ins GPR64:$a, i32imm:$b),
              "mapa" # suffix # ".u64\t$d, $a, $b",
              [(set GPR64:$d, (Intr GPR64:$a, imm:$b))]>,
    Requires<[hasSM<90>, hasPTX<78>]>;
}

defm mapa  : MAPA<"", int_rvgpu_mapa>;
defm mapa_shared_cluster  : MAPA<".shared::cluster", int_rvgpu_mapa_shared_cluster>;


multiclass GETCTARANK<string suffix, Intrinsic Intr> {
  def _32: RVGPUInst<(outs GPR32:$d), (ins GPR32:$a),
              "getctarank" # suffix # ".u32\t$d, $a",
              [(set GPR32:$d, (Intr GPR32:$a))]>,
    Requires<[hasSM<90>, hasPTX<78>]>;
  def _64: RVGPUInst<(outs GPR32:$d), (ins GPR64:$a),
              "getctarank" # suffix # ".u64\t$d, $a",
              [(set GPR32:$d, (Intr GPR64:$a))]>,
    Requires<[hasSM<90>, hasPTX<78>]>;
}

defm getctarank  : GETCTARANK<"", int_rvgpu_getctarank>;
defm getctarank_shared_cluster  : GETCTARANK<".shared::cluster", int_rvgpu_getctarank_shared_cluster>;

def is_explicit_cluster: RVGPUInst<(outs Int1Regs:$d), (ins),
              "mov.pred\t$d, %is_explicit_cluster",
              [(set Int1Regs:$d, (int_rvgpu_is_explicit_cluster))]>,
    Requires<[hasSM<90>, hasPTX<78>]>;
