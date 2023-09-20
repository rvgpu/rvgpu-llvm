let TargetPrefix = "riscv" in {
    class ReadPreloadRegisterIntrinsic
       : Intrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrSpeculatable]>;
    multiclass ReadPreloadRegisterIntrinsic_xyz {
       def _x : ReadPreloadRegisterIntrinsic;
 
       def _y : ReadPreloadRegisterIntrinsic;
       def _z : ReadPreloadRegisterIntrinsic;
    }
 
    defm int_riscv_rvgpu_read_tid : ReadPreloadRegisterIntrinsic_xyz;

}

