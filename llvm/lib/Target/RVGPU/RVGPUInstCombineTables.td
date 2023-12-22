include "llvm/TableGen/SearchableTable.td"
include "llvm/IR/Intrinsics.td"

def RVGPUImageDMaskIntrinsicTable : GenericTable {
  let FilterClass = "RVGPUImageDMaskIntrinsic";
  let Fields = ["Intr"];

  let PrimaryKey = ["Intr"];
  let PrimaryKeyName = "getRVGPUImageDMaskIntrinsic";
  let PrimaryKeyEarlyOut = 1;
}
