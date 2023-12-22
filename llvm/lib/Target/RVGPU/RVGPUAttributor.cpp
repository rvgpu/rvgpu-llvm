//===- RVGPUAttributor.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass uses Attributor framework to deduce RVGPU attributes.
//
//===----------------------------------------------------------------------===//

#include "RVGPU.h"
#include "RVSubtarget.h"
#include "Utils/RVGPUBaseInfo.h"
#include "llvm/Analysis/CycleAnalysis.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IntrinsicsRVGPU.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/Attributor.h"

#define DEBUG_TYPE "rvgpu-attributor"

namespace llvm {
void initializeCycleInfoWrapperPassPass(PassRegistry &);
}

using namespace llvm;

static cl::opt<unsigned> KernargPreloadCount(
    "rvgpu-kernarg-preload-count",
    cl::desc("How many kernel arguments to preload onto SGPRs"), cl::init(0));

#define RVGPU_ATTRIBUTE(Name, Str) Name##_POS,

enum ImplicitArgumentPositions {
  #include "RVGPUAttributes.def"
  LAST_ARG_POS
};

#define RVGPU_ATTRIBUTE(Name, Str) Name = 1 << Name##_POS,

enum ImplicitArgumentMask {
  NOT_IMPLICIT_INPUT = 0,
  #include "RVGPUAttributes.def"
  ALL_ARGUMENT_MASK = (1 << LAST_ARG_POS) - 1
};

#define RVGPU_ATTRIBUTE(Name, Str) {Name, Str},
static constexpr std::pair<ImplicitArgumentMask,
                           StringLiteral> ImplicitAttrs[] = {
 #include "RVGPUAttributes.def"
};

// We do not need to note the x workitem or workgroup id because they are always
// initialized.
//
// TODO: We should not add the attributes if the known compile time workgroup
// size is 1 for y/z.
static ImplicitArgumentMask
intrinsicToAttrMask(Intrinsic::ID ID, bool &NonKernelOnly, bool &NeedsImplicit,
                    bool HasApertureRegs, bool SupportsGetDoorBellID,
                    unsigned CodeObjectVersion) {
  switch (ID) {
  case Intrinsic::rvgpu_workitem_id_x:
    NonKernelOnly = true;
    return WORKITEM_ID_X;
  case Intrinsic::rvgpu_workgroup_id_x:
    NonKernelOnly = true;
    return WORKGROUP_ID_X;
  case Intrinsic::rvgpu_workitem_id_y:
    return WORKITEM_ID_Y;
  case Intrinsic::rvgpu_workitem_id_z:
    return WORKITEM_ID_Z;
  case Intrinsic::rvgpu_workgroup_id_y:
    return WORKGROUP_ID_Y;
  case Intrinsic::rvgpu_workgroup_id_z:
    return WORKGROUP_ID_Z;
  case Intrinsic::rvgpu_lds_kernel_id:
    return LDS_KERNEL_ID;
  case Intrinsic::rvgpu_dispatch_ptr:
    return DISPATCH_PTR;
  case Intrinsic::rvgpu_dispatch_id:
    return DISPATCH_ID;
  case Intrinsic::rvgpu_implicitarg_ptr:
    return IMPLICIT_ARG_PTR;
  // Need queue_ptr anyway. But under V5, we also need implicitarg_ptr to access
  // queue_ptr.
  case Intrinsic::rvgpu_queue_ptr:
    NeedsImplicit = (CodeObjectVersion >= RVGPU::RVHSA_COV5);
    return QUEUE_PTR;
  case Intrinsic::rvgpu_is_shared:
  case Intrinsic::rvgpu_is_private:
    if (HasApertureRegs)
      return NOT_IMPLICIT_INPUT;
    // Under V5, we need implicitarg_ptr + offsets to access private_base or
    // shared_base. For pre-V5, however, need to access them through queue_ptr +
    // offsets.
    return CodeObjectVersion >= RVGPU::RVHSA_COV5 ? IMPLICIT_ARG_PTR :
                                                      QUEUE_PTR;
  case Intrinsic::trap:
    if (SupportsGetDoorBellID) // GetDoorbellID support implemented since V4.
      return CodeObjectVersion >= RVGPU::RVHSA_COV4 ? NOT_IMPLICIT_INPUT :
                                                        QUEUE_PTR;
    NeedsImplicit = (CodeObjectVersion >= RVGPU::RVHSA_COV5);
    return QUEUE_PTR;
  default:
    return NOT_IMPLICIT_INPUT;
  }
}

static bool castRequiresQueuePtr(unsigned SrcAS) {
  return SrcAS == RVGPUAS::LOCAL_ADDRESS || SrcAS == RVGPUAS::PRIVATE_ADDRESS;
}

static bool isDSAddress(const Constant *C) {
  const GlobalValue *GV = dyn_cast<GlobalValue>(C);
  if (!GV)
    return false;
  unsigned AS = GV->getAddressSpace();
  return AS == RVGPUAS::LOCAL_ADDRESS || AS == RVGPUAS::REGION_ADDRESS;
}

/// Returns true if the function requires the implicit argument be passed
/// regardless of the function contents.
static bool funcRequiresHostcallPtr(const Function &F) {
  // Sanitizers require the hostcall buffer passed in the implicit arguments.
  return F.hasFnAttribute(Attribute::SanitizeAddress) ||
         F.hasFnAttribute(Attribute::SanitizeThread) ||
         F.hasFnAttribute(Attribute::SanitizeMemory) ||
         F.hasFnAttribute(Attribute::SanitizeHWAddress) ||
         F.hasFnAttribute(Attribute::SanitizeMemTag);
}

namespace {
class RVGPUInformationCache : public InformationCache {
public:
  RVGPUInformationCache(const Module &M, AnalysisGetter &AG,
                         BumpPtrAllocator &Allocator,
                         SetVector<Function *> *CGSCC, TargetMachine &TM)
      : InformationCache(M, AG, Allocator, CGSCC), TM(TM),
        CodeObjectVersion(RVGPU::getCodeObjectVersion(M)) {}

  TargetMachine &TM;

  enum ConstantStatus { DS_GLOBAL = 1 << 0, ADDR_SPACE_CAST = 1 << 1 };

  /// Check if the subtarget has aperture regs.
  bool hasApertureRegs(Function &F) {
    const RVSubtarget &ST = TM.getSubtarget<RVSubtarget>(F);
    return ST.hasApertureRegs();
  }

  /// Check if the subtarget supports GetDoorbellID.
  bool supportsGetDoorbellID(Function &F) {
    const RVSubtarget &ST = TM.getSubtarget<RVSubtarget>(F);
    return ST.supportsGetDoorbellID();
  }

  std::pair<unsigned, unsigned> getFlatWorkGroupSizes(const Function &F) {
    const RVSubtarget &ST = TM.getSubtarget<RVSubtarget>(F);
    return ST.getFlatWorkGroupSizes(F);
  }

  std::pair<unsigned, unsigned>
  getMaximumFlatWorkGroupRange(const Function &F) {
    const RVSubtarget &ST = TM.getSubtarget<RVSubtarget>(F);
    return {ST.getMinFlatWorkGroupSize(), ST.getMaxFlatWorkGroupSize()};
  }

  /// Get code object version.
  unsigned getCodeObjectVersion() const {
    return CodeObjectVersion;
  }

  /// Get the effective value of "rvgpu-waves-per-eu" for the function,
  /// accounting for the interaction with the passed value to use for
  /// "rvgpu-flat-work-group-size".
  std::pair<unsigned, unsigned>
  getWavesPerEU(const Function &F,
                std::pair<unsigned, unsigned> FlatWorkGroupSize) {
    const RVSubtarget &ST = TM.getSubtarget<RVSubtarget>(F);
    return ST.getWavesPerEU(F, FlatWorkGroupSize);
  }

  std::pair<unsigned, unsigned>
  getEffectiveWavesPerEU(const Function &F,
                         std::pair<unsigned, unsigned> WavesPerEU,
                         std::pair<unsigned, unsigned> FlatWorkGroupSize) {
    const RVSubtarget &ST = TM.getSubtarget<RVSubtarget>(F);
    return ST.getEffectiveWavesPerEU(WavesPerEU, FlatWorkGroupSize);
  }

  unsigned getMaxWavesPerEU(const Function &F) {
    const RVSubtarget &ST = TM.getSubtarget<RVSubtarget>(F);
    return ST.getMaxWavesPerEU();
  }

private:
  /// Check if the ConstantExpr \p CE requires the queue pointer.
  static bool visitConstExpr(const ConstantExpr *CE) {
    if (CE->getOpcode() == Instruction::AddrSpaceCast) {
      unsigned SrcAS = CE->getOperand(0)->getType()->getPointerAddressSpace();
      return castRequiresQueuePtr(SrcAS);
    }
    return false;
  }

  /// Get the constant access bitmap for \p C.
  uint8_t getConstantAccess(const Constant *C,
                            SmallPtrSetImpl<const Constant *> &Visited) {
    auto It = ConstantStatus.find(C);
    if (It != ConstantStatus.end())
      return It->second;

    uint8_t Result = 0;
    if (isDSAddress(C))
      Result = DS_GLOBAL;

    if (const auto *CE = dyn_cast<ConstantExpr>(C))
      if (visitConstExpr(CE))
        Result |= ADDR_SPACE_CAST;

    for (const Use &U : C->operands()) {
      const auto *OpC = dyn_cast<Constant>(U);
      if (!OpC || !Visited.insert(OpC).second)
        continue;

      Result |= getConstantAccess(OpC, Visited);
    }
    return Result;
  }

public:
  /// Returns true if \p Fn needs the queue pointer because of \p C.
  bool needsQueuePtr(const Constant *C, Function &Fn) {
    bool IsNonEntryFunc = !RVGPU::isEntryFunctionCC(Fn.getCallingConv());
    bool HasAperture = hasApertureRegs(Fn);

    // No need to explore the constants.
    if (!IsNonEntryFunc && HasAperture)
      return false;

    SmallPtrSet<const Constant *, 8> Visited;
    uint8_t Access = getConstantAccess(C, Visited);

    // We need to trap on DS globals in non-entry functions.
    if (IsNonEntryFunc && (Access & DS_GLOBAL))
      return true;

    return !HasAperture && (Access & ADDR_SPACE_CAST);
  }

private:
  /// Used to determine if the Constant needs the queue pointer.
  DenseMap<const Constant *, uint8_t> ConstantStatus;
  const unsigned CodeObjectVersion;
};

struct AARVAttributes
    : public StateWrapper<BitIntegerState<uint32_t, ALL_ARGUMENT_MASK, 0>,
                          AbstractAttribute> {
  using Base = StateWrapper<BitIntegerState<uint32_t, ALL_ARGUMENT_MASK, 0>,
                            AbstractAttribute>;

  AARVAttributes(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

  /// Create an abstract attribute view for the position \p IRP.
  static AARVAttributes &createForPosition(const IRPosition &IRP,
                                            Attributor &A);

  /// See AbstractAttribute::getName().
  const std::string getName() const override { return "AARVAttributes"; }

  /// See AbstractAttribute::getIdAddr().
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AARVAttributes.
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};
const char AARVAttributes::ID = 0;

struct AAUniformWorkGroupSize
    : public StateWrapper<BooleanState, AbstractAttribute> {
  using Base = StateWrapper<BooleanState, AbstractAttribute>;
  AAUniformWorkGroupSize(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

  /// Create an abstract attribute view for the position \p IRP.
  static AAUniformWorkGroupSize &createForPosition(const IRPosition &IRP,
                                                   Attributor &A);

  /// See AbstractAttribute::getName().
  const std::string getName() const override {
    return "AAUniformWorkGroupSize";
  }

  /// See AbstractAttribute::getIdAddr().
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AARVAttributes.
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};
const char AAUniformWorkGroupSize::ID = 0;

struct AAUniformWorkGroupSizeFunction : public AAUniformWorkGroupSize {
  AAUniformWorkGroupSizeFunction(const IRPosition &IRP, Attributor &A)
      : AAUniformWorkGroupSize(IRP, A) {}

  void initialize(Attributor &A) override {
    Function *F = getAssociatedFunction();
    CallingConv::ID CC = F->getCallingConv();

    if (CC != CallingConv::RVGPU_KERNEL)
      return;

    bool InitialValue = false;
    if (F->hasFnAttribute("uniform-work-group-size"))
      InitialValue = F->getFnAttribute("uniform-work-group-size")
                         .getValueAsString()
                         .equals("true");

    if (InitialValue)
      indicateOptimisticFixpoint();
    else
      indicatePessimisticFixpoint();
  }

  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Change = ChangeStatus::UNCHANGED;

    auto CheckCallSite = [&](AbstractCallSite CS) {
      Function *Caller = CS.getInstruction()->getFunction();
      LLVM_DEBUG(dbgs() << "[AAUniformWorkGroupSize] Call " << Caller->getName()
                        << "->" << getAssociatedFunction()->getName() << "\n");

      const auto *CallerInfo = A.getAAFor<AAUniformWorkGroupSize>(
          *this, IRPosition::function(*Caller), DepClassTy::REQUIRED);
      if (!CallerInfo)
        return false;

      Change = Change | clampStateAndIndicateChange(this->getState(),
                                                    CallerInfo->getState());

      return true;
    };

    bool AllCallSitesKnown = true;
    if (!A.checkForAllCallSites(CheckCallSite, *this, true, AllCallSitesKnown))
      return indicatePessimisticFixpoint();

    return Change;
  }

  ChangeStatus manifest(Attributor &A) override {
    SmallVector<Attribute, 8> AttrList;
    LLVMContext &Ctx = getAssociatedFunction()->getContext();

    AttrList.push_back(Attribute::get(Ctx, "uniform-work-group-size",
                                      getAssumed() ? "true" : "false"));
    return A.manifestAttrs(getIRPosition(), AttrList,
                           /* ForceReplace */ true);
  }

  bool isValidState() const override {
    // This state is always valid, even when the state is false.
    return true;
  }

  const std::string getAsStr(Attributor *) const override {
    return "RVWorkGroupSize[" + std::to_string(getAssumed()) + "]";
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {}
};

AAUniformWorkGroupSize &
AAUniformWorkGroupSize::createForPosition(const IRPosition &IRP,
                                          Attributor &A) {
  if (IRP.getPositionKind() == IRPosition::IRP_FUNCTION)
    return *new (A.Allocator) AAUniformWorkGroupSizeFunction(IRP, A);
  llvm_unreachable(
      "AAUniformWorkGroupSize is only valid for function position");
}

struct AARVAttributesFunction : public AARVAttributes {
  AARVAttributesFunction(const IRPosition &IRP, Attributor &A)
      : AARVAttributes(IRP, A) {}

  void initialize(Attributor &A) override {
    Function *F = getAssociatedFunction();

    // If the function requires the implicit arg pointer due to sanitizers,
    // assume it's needed even if explicitly marked as not requiring it.
    const bool NeedsHostcall = funcRequiresHostcallPtr(*F);
    if (NeedsHostcall) {
      removeAssumedBits(IMPLICIT_ARG_PTR);
      removeAssumedBits(HOSTCALL_PTR);
    }

    for (auto Attr : ImplicitAttrs) {
      if (NeedsHostcall &&
          (Attr.first == IMPLICIT_ARG_PTR || Attr.first == HOSTCALL_PTR))
        continue;

      if (F->hasFnAttribute(Attr.second))
        addKnownBits(Attr.first);
    }

    if (F->isDeclaration())
      return;

    // Ignore functions with graphics calling conventions, these are currently
    // not allowed to have kernel arguments.
    if (RVGPU::isGraphics(F->getCallingConv())) {
      indicatePessimisticFixpoint();
      return;
    }
  }

  ChangeStatus updateImpl(Attributor &A) override {
    Function *F = getAssociatedFunction();
    // The current assumed state used to determine a change.
    auto OrigAssumed = getAssumed();

    // Check for Intrinsics and propagate attributes.
    const AACallEdges *AAEdges = A.getAAFor<AACallEdges>(
        *this, this->getIRPosition(), DepClassTy::REQUIRED);
    if (!AAEdges || AAEdges->hasNonAsmUnknownCallee())
      return indicatePessimisticFixpoint();

    bool IsNonEntryFunc = !RVGPU::isEntryFunctionCC(F->getCallingConv());

    bool NeedsImplicit = false;
    auto &InfoCache = static_cast<RVGPUInformationCache &>(A.getInfoCache());
    bool HasApertureRegs = InfoCache.hasApertureRegs(*F);
    bool SupportsGetDoorbellID = InfoCache.supportsGetDoorbellID(*F);
    unsigned COV = InfoCache.getCodeObjectVersion();

    for (Function *Callee : AAEdges->getOptimisticEdges()) {
      Intrinsic::ID IID = Callee->getIntrinsicID();
      if (IID == Intrinsic::not_intrinsic) {
        const AARVAttributes *AARV = A.getAAFor<AARVAttributes>(
            *this, IRPosition::function(*Callee), DepClassTy::REQUIRED);
        if (!AARV)
          return indicatePessimisticFixpoint();
        *this &= *AARV;
        continue;
      }

      bool NonKernelOnly = false;
      ImplicitArgumentMask AttrMask =
          intrinsicToAttrMask(IID, NonKernelOnly, NeedsImplicit,
                              HasApertureRegs, SupportsGetDoorbellID, COV);
      if (AttrMask != NOT_IMPLICIT_INPUT) {
        if ((IsNonEntryFunc || !NonKernelOnly))
          removeAssumedBits(AttrMask);
      }
    }

    // Need implicitarg_ptr to acess queue_ptr, private_base, and shared_base.
    if (NeedsImplicit)
      removeAssumedBits(IMPLICIT_ARG_PTR);

    if (isAssumed(QUEUE_PTR) && checkForQueuePtr(A)) {
      // Under V5, we need implicitarg_ptr + offsets to access private_base or
      // shared_base. We do not actually need queue_ptr.
      if (COV >= 5)
        removeAssumedBits(IMPLICIT_ARG_PTR);
      else
        removeAssumedBits(QUEUE_PTR);
    }

    if (funcRetrievesMultigridSyncArg(A, COV)) {
      assert(!isAssumed(IMPLICIT_ARG_PTR) &&
             "multigrid_sync_arg needs implicitarg_ptr");
      removeAssumedBits(MULTIGRID_SYNC_ARG);
    }

    if (funcRetrievesHostcallPtr(A, COV)) {
      assert(!isAssumed(IMPLICIT_ARG_PTR) && "hostcall needs implicitarg_ptr");
      removeAssumedBits(HOSTCALL_PTR);
    }

    if (funcRetrievesHeapPtr(A, COV)) {
      assert(!isAssumed(IMPLICIT_ARG_PTR) && "heap_ptr needs implicitarg_ptr");
      removeAssumedBits(HEAP_PTR);
    }

    if (isAssumed(QUEUE_PTR) && funcRetrievesQueuePtr(A, COV)) {
      assert(!isAssumed(IMPLICIT_ARG_PTR) && "queue_ptr needs implicitarg_ptr");
      removeAssumedBits(QUEUE_PTR);
    }

    if (isAssumed(LDS_KERNEL_ID) && funcRetrievesLDSKernelId(A)) {
      removeAssumedBits(LDS_KERNEL_ID);
    }

    if (isAssumed(DEFAULT_QUEUE) && funcRetrievesDefaultQueue(A, COV))
      removeAssumedBits(DEFAULT_QUEUE);

    if (isAssumed(COMPLETION_ACTION) && funcRetrievesCompletionAction(A, COV))
      removeAssumedBits(COMPLETION_ACTION);

    return getAssumed() != OrigAssumed ? ChangeStatus::CHANGED
                                       : ChangeStatus::UNCHANGED;
  }

  ChangeStatus manifest(Attributor &A) override {
    SmallVector<Attribute, 8> AttrList;
    LLVMContext &Ctx = getAssociatedFunction()->getContext();

    for (auto Attr : ImplicitAttrs) {
      if (isKnown(Attr.first))
        AttrList.push_back(Attribute::get(Ctx, Attr.second));
    }

    return A.manifestAttrs(getIRPosition(), AttrList,
                           /* ForceReplace */ true);
  }

  const std::string getAsStr(Attributor *) const override {
    std::string Str;
    raw_string_ostream OS(Str);
    OS << "RVInfo[";
    for (auto Attr : ImplicitAttrs)
      if (isAssumed(Attr.first))
        OS << ' ' << Attr.second;
    OS << " ]";
    return OS.str();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {}

private:
  bool checkForQueuePtr(Attributor &A) {
    Function *F = getAssociatedFunction();
    bool IsNonEntryFunc = !RVGPU::isEntryFunctionCC(F->getCallingConv());

    auto &InfoCache = static_cast<RVGPUInformationCache &>(A.getInfoCache());

    bool NeedsQueuePtr = false;

    auto CheckAddrSpaceCasts = [&](Instruction &I) {
      unsigned SrcAS = static_cast<AddrSpaceCastInst &>(I).getSrcAddressSpace();
      if (castRequiresQueuePtr(SrcAS)) {
        NeedsQueuePtr = true;
        return false;
      }
      return true;
    };

    bool HasApertureRegs = InfoCache.hasApertureRegs(*F);

    // `checkForAllInstructions` is much more cheaper than going through all
    // instructions, try it first.

    // The queue pointer is not needed if aperture regs is present.
    if (!HasApertureRegs) {
      bool UsedAssumedInformation = false;
      A.checkForAllInstructions(CheckAddrSpaceCasts, *this,
                                {Instruction::AddrSpaceCast},
                                UsedAssumedInformation);
    }

    // If we found  that we need the queue pointer, nothing else to do.
    if (NeedsQueuePtr)
      return true;

    if (!IsNonEntryFunc && HasApertureRegs)
      return false;

    for (BasicBlock &BB : *F) {
      for (Instruction &I : BB) {
        for (const Use &U : I.operands()) {
          if (const auto *C = dyn_cast<Constant>(U)) {
            if (InfoCache.needsQueuePtr(C, *F))
              return true;
          }
        }
      }
    }

    return false;
  }

  bool funcRetrievesMultigridSyncArg(Attributor &A, unsigned COV) {
    auto Pos = llvm::RVGPU::getMultigridSyncArgImplicitArgPosition(COV);
    AA::RangeTy Range(Pos, 8);
    return funcRetrievesImplicitKernelArg(A, Range);
  }

  bool funcRetrievesHostcallPtr(Attributor &A, unsigned COV) {
    auto Pos = llvm::RVGPU::getHostcallImplicitArgPosition(COV);
    AA::RangeTy Range(Pos, 8);
    return funcRetrievesImplicitKernelArg(A, Range);
  }

  bool funcRetrievesDefaultQueue(Attributor &A, unsigned COV) {
    auto Pos = llvm::RVGPU::getDefaultQueueImplicitArgPosition(COV);
    AA::RangeTy Range(Pos, 8);
    return funcRetrievesImplicitKernelArg(A, Range);
  }

  bool funcRetrievesCompletionAction(Attributor &A, unsigned COV) {
    auto Pos = llvm::RVGPU::getCompletionActionImplicitArgPosition(COV);
    AA::RangeTy Range(Pos, 8);
    return funcRetrievesImplicitKernelArg(A, Range);
  }

  bool funcRetrievesHeapPtr(Attributor &A, unsigned COV) {
    if (COV < 5)
      return false;
    AA::RangeTy Range(RVGPU::ImplicitArg::HEAP_PTR_OFFSET, 8);
    return funcRetrievesImplicitKernelArg(A, Range);
  }

  bool funcRetrievesQueuePtr(Attributor &A, unsigned COV) {
    if (COV < 5)
      return false;
    AA::RangeTy Range(RVGPU::ImplicitArg::QUEUE_PTR_OFFSET, 8);
    return funcRetrievesImplicitKernelArg(A, Range);
  }

  bool funcRetrievesImplicitKernelArg(Attributor &A, AA::RangeTy Range) {
    // Check if this is a call to the implicitarg_ptr builtin and it
    // is used to retrieve the hostcall pointer. The implicit arg for
    // hostcall is not used only if every use of the implicitarg_ptr
    // is a load that clearly does not retrieve any byte of the
    // hostcall pointer. We check this by tracing all the uses of the
    // initial call to the implicitarg_ptr intrinsic.
    auto DoesNotLeadToKernelArgLoc = [&](Instruction &I) {
      auto &Call = cast<CallBase>(I);
      if (Call.getIntrinsicID() != Intrinsic::rvgpu_implicitarg_ptr)
        return true;

      const auto *PointerInfoAA = A.getAAFor<AAPointerInfo>(
          *this, IRPosition::callsite_returned(Call), DepClassTy::REQUIRED);
      if (!PointerInfoAA)
        return false;

      return PointerInfoAA->forallInterferingAccesses(
          Range, [](const AAPointerInfo::Access &Acc, bool IsExact) {
            return Acc.getRemoteInst()->isDroppable();
          });
    };

    bool UsedAssumedInformation = false;
    return !A.checkForAllCallLikeInstructions(DoesNotLeadToKernelArgLoc, *this,
                                              UsedAssumedInformation);
  }

  bool funcRetrievesLDSKernelId(Attributor &A) {
    auto DoesNotRetrieve = [&](Instruction &I) {
      auto &Call = cast<CallBase>(I);
      return Call.getIntrinsicID() != Intrinsic::rvgpu_lds_kernel_id;
    };
    bool UsedAssumedInformation = false;
    return !A.checkForAllCallLikeInstructions(DoesNotRetrieve, *this,
                                              UsedAssumedInformation);
  }
};

AARVAttributes &AARVAttributes::createForPosition(const IRPosition &IRP,
                                                    Attributor &A) {
  if (IRP.getPositionKind() == IRPosition::IRP_FUNCTION)
    return *new (A.Allocator) AARVAttributesFunction(IRP, A);
  llvm_unreachable("AARVAttributes is only valid for function position");
}

/// Base class to derive different size ranges.
struct AARVSizeRangeAttribute
    : public StateWrapper<IntegerRangeState, AbstractAttribute, uint32_t> {
  using Base = StateWrapper<IntegerRangeState, AbstractAttribute, uint32_t>;

  StringRef AttrName;

  AARVSizeRangeAttribute(const IRPosition &IRP, Attributor &A,
                          StringRef AttrName)
      : Base(IRP, 32), AttrName(AttrName) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {}

  template <class AttributeImpl>
  ChangeStatus updateImplImpl(Attributor &A) {
    ChangeStatus Change = ChangeStatus::UNCHANGED;

    auto CheckCallSite = [&](AbstractCallSite CS) {
      Function *Caller = CS.getInstruction()->getFunction();
      LLVM_DEBUG(dbgs() << '[' << getName() << "] Call " << Caller->getName()
                        << "->" << getAssociatedFunction()->getName() << '\n');

      const auto *CallerInfo = A.getAAFor<AttributeImpl>(
          *this, IRPosition::function(*Caller), DepClassTy::REQUIRED);
      if (!CallerInfo)
        return false;

      Change |=
          clampStateAndIndicateChange(this->getState(), CallerInfo->getState());

      return true;
    };

    bool AllCallSitesKnown = true;
    if (!A.checkForAllCallSites(CheckCallSite, *this, true, AllCallSitesKnown))
      return indicatePessimisticFixpoint();

    return Change;
  }

  ChangeStatus emitAttributeIfNotDefault(Attributor &A, unsigned Min,
                                         unsigned Max) {
    // Don't add the attribute if it's the implied default.
    if (getAssumed().getLower() == Min && getAssumed().getUpper() - 1 == Max)
      return ChangeStatus::UNCHANGED;

    Function *F = getAssociatedFunction();
    LLVMContext &Ctx = F->getContext();
    SmallString<10> Buffer;
    raw_svector_ostream OS(Buffer);
    OS << getAssumed().getLower() << ',' << getAssumed().getUpper() - 1;
    return A.manifestAttrs(getIRPosition(),
                           {Attribute::get(Ctx, AttrName, OS.str())},
                           /* ForceReplace */ true);
  }

  const std::string getAsStr(Attributor *) const override {
    std::string Str;
    raw_string_ostream OS(Str);
    OS << getName() << '[';
    OS << getAssumed().getLower() << ',' << getAssumed().getUpper() - 1;
    OS << ']';
    return OS.str();
  }
};

/// Propagate rvgpu-flat-work-group-size attribute.
struct AARVFlatWorkGroupSize : public AARVSizeRangeAttribute {
  AARVFlatWorkGroupSize(const IRPosition &IRP, Attributor &A)
      : AARVSizeRangeAttribute(IRP, A, "rvgpu-flat-work-group-size") {}

  void initialize(Attributor &A) override {
    Function *F = getAssociatedFunction();
    auto &InfoCache = static_cast<RVGPUInformationCache &>(A.getInfoCache());
    unsigned MinGroupSize, MaxGroupSize;
    std::tie(MinGroupSize, MaxGroupSize) = InfoCache.getFlatWorkGroupSizes(*F);
    intersectKnown(
        ConstantRange(APInt(32, MinGroupSize), APInt(32, MaxGroupSize + 1)));

    if (RVGPU::isEntryFunctionCC(F->getCallingConv()))
      indicatePessimisticFixpoint();
  }

  ChangeStatus updateImpl(Attributor &A) override {
    return updateImplImpl<AARVFlatWorkGroupSize>(A);
  }

  /// Create an abstract attribute view for the position \p IRP.
  static AARVFlatWorkGroupSize &createForPosition(const IRPosition &IRP,
                                                   Attributor &A);

  ChangeStatus manifest(Attributor &A) override {
    Function *F = getAssociatedFunction();
    auto &InfoCache = static_cast<RVGPUInformationCache &>(A.getInfoCache());
    unsigned Min, Max;
    std::tie(Min, Max) = InfoCache.getMaximumFlatWorkGroupRange(*F);
    return emitAttributeIfNotDefault(A, Min, Max);
  }

  /// See AbstractAttribute::getName()
  const std::string getName() const override {
    return "AARVFlatWorkGroupSize";
  }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AARVFlatWorkGroupSize
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

const char AARVFlatWorkGroupSize::ID = 0;

AARVFlatWorkGroupSize &
AARVFlatWorkGroupSize::createForPosition(const IRPosition &IRP,
                                          Attributor &A) {
  if (IRP.getPositionKind() == IRPosition::IRP_FUNCTION)
    return *new (A.Allocator) AARVFlatWorkGroupSize(IRP, A);
  llvm_unreachable(
      "AARVFlatWorkGroupSize is only valid for function position");
}

/// Propagate rvgpu-waves-per-eu attribute.
struct AARVWavesPerEU : public AARVSizeRangeAttribute {
  AARVWavesPerEU(const IRPosition &IRP, Attributor &A)
      : AARVSizeRangeAttribute(IRP, A, "rvgpu-waves-per-eu") {}

  bool isValidState() const override {
    return !Assumed.isEmptySet() && IntegerRangeState::isValidState();
  }

  void initialize(Attributor &A) override {
    Function *F = getAssociatedFunction();
    auto &InfoCache = static_cast<RVGPUInformationCache &>(A.getInfoCache());

    if (const auto *AssumedGroupSize = A.getAAFor<AARVFlatWorkGroupSize>(
            *this, IRPosition::function(*F), DepClassTy::REQUIRED)) {

      unsigned Min, Max;
      std::tie(Min, Max) = InfoCache.getWavesPerEU(
          *F, {AssumedGroupSize->getAssumed().getLower().getZExtValue(),
               AssumedGroupSize->getAssumed().getUpper().getZExtValue() - 1});

      ConstantRange Range(APInt(32, Min), APInt(32, Max + 1));
      intersectKnown(Range);
    }

    if (RVGPU::isEntryFunctionCC(F->getCallingConv()))
      indicatePessimisticFixpoint();
  }

  ChangeStatus updateImpl(Attributor &A) override {
    auto &InfoCache = static_cast<RVGPUInformationCache &>(A.getInfoCache());
    ChangeStatus Change = ChangeStatus::UNCHANGED;

    auto CheckCallSite = [&](AbstractCallSite CS) {
      Function *Caller = CS.getInstruction()->getFunction();
      Function *Func = getAssociatedFunction();
      LLVM_DEBUG(dbgs() << '[' << getName() << "] Call " << Caller->getName()
                        << "->" << Func->getName() << '\n');

      const auto *CallerInfo = A.getAAFor<AARVWavesPerEU>(
          *this, IRPosition::function(*Caller), DepClassTy::REQUIRED);
      const auto *AssumedGroupSize = A.getAAFor<AARVFlatWorkGroupSize>(
          *this, IRPosition::function(*Func), DepClassTy::REQUIRED);
      if (!CallerInfo || !AssumedGroupSize)
        return false;

      unsigned Min, Max;
      std::tie(Min, Max) = InfoCache.getEffectiveWavesPerEU(
          *Caller,
          {CallerInfo->getAssumed().getLower().getZExtValue(),
           CallerInfo->getAssumed().getUpper().getZExtValue() - 1},
          {AssumedGroupSize->getAssumed().getLower().getZExtValue(),
           AssumedGroupSize->getAssumed().getUpper().getZExtValue() - 1});
      ConstantRange CallerRange(APInt(32, Min), APInt(32, Max + 1));
      IntegerRangeState CallerRangeState(CallerRange);
      Change |= clampStateAndIndicateChange(this->getState(), CallerRangeState);

      return true;
    };

    bool AllCallSitesKnown = true;
    if (!A.checkForAllCallSites(CheckCallSite, *this, true, AllCallSitesKnown))
      return indicatePessimisticFixpoint();

    return Change;
  }

  /// Create an abstract attribute view for the position \p IRP.
  static AARVWavesPerEU &createForPosition(const IRPosition &IRP,
                                            Attributor &A);

  ChangeStatus manifest(Attributor &A) override {
    Function *F = getAssociatedFunction();
    auto &InfoCache = static_cast<RVGPUInformationCache &>(A.getInfoCache());
    unsigned Max = InfoCache.getMaxWavesPerEU(*F);
    return emitAttributeIfNotDefault(A, 1, Max);
  }

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AARVWavesPerEU"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AARVWavesPerEU
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

const char AARVWavesPerEU::ID = 0;

AARVWavesPerEU &AARVWavesPerEU::createForPosition(const IRPosition &IRP,
                                                    Attributor &A) {
  if (IRP.getPositionKind() == IRPosition::IRP_FUNCTION)
    return *new (A.Allocator) AARVWavesPerEU(IRP, A);
  llvm_unreachable("AARVWavesPerEU is only valid for function position");
}

static void addPreloadKernArgHint(Function &F, TargetMachine &TM) {
  const RVSubtarget &ST = TM.getSubtarget<RVSubtarget>(F);
  for (unsigned I = 0;
       I < F.arg_size() &&
       I < std::min(KernargPreloadCount.getValue(), ST.getMaxNumUserSGPRs());
       ++I) {
    Argument &Arg = *F.getArg(I);
    // Check for incompatible attributes.
    if (Arg.hasByRefAttr() || Arg.hasNestAttr())
      break;

    Arg.addAttr(Attribute::InReg);
  }
}

static bool runImpl(Module &M, AnalysisGetter &AG, TargetMachine &TM) {
  SetVector<Function *> Functions;
  for (Function &F : M) {
    if (!F.isIntrinsic())
      Functions.insert(&F);
  }

  CallGraphUpdater CGUpdater;
  BumpPtrAllocator Allocator;
  RVGPUInformationCache InfoCache(M, AG, Allocator, nullptr, TM);
  DenseSet<const char *> Allowed(
      {&AARVAttributes::ID, &AAUniformWorkGroupSize::ID,
       &AAPotentialValues::ID, &AARVFlatWorkGroupSize::ID,
       &AARVWavesPerEU::ID, &AACallEdges::ID, &AAPointerInfo::ID,
       &AAPotentialConstantValues::ID, &AAUnderlyingObjects::ID});

  AttributorConfig AC(CGUpdater);
  AC.Allowed = &Allowed;
  AC.IsModulePass = true;
  AC.DefaultInitializeLiveInternals = false;
  AC.IPOAmendableCB = [](const Function &F) {
    return F.getCallingConv() == CallingConv::RVGPU_KERNEL;
  };

  Attributor A(Functions, InfoCache, AC);

  for (Function &F : M) {
    if (!F.isIntrinsic()) {
      A.getOrCreateAAFor<AARVAttributes>(IRPosition::function(F));
      A.getOrCreateAAFor<AAUniformWorkGroupSize>(IRPosition::function(F));
      CallingConv::ID CC = F.getCallingConv();
      if (!RVGPU::isEntryFunctionCC(CC)) {
        A.getOrCreateAAFor<AARVFlatWorkGroupSize>(IRPosition::function(F));
        A.getOrCreateAAFor<AARVWavesPerEU>(IRPosition::function(F));
      } else if (CC == CallingConv::RVGPU_KERNEL) {
        addPreloadKernArgHint(F, TM);
      }
    }
  }

  ChangeStatus Change = A.run();
  return Change == ChangeStatus::CHANGED;
}

class RVGPUAttributorLegacy : public ModulePass {
public:
  RVGPUAttributorLegacy() : ModulePass(ID) {}

  /// doInitialization - Virtual method overridden by subclasses to do
  /// any necessary initialization before any pass is run.
  bool doInitialization(Module &) override {
    auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
    if (!TPC)
      report_fatal_error("TargetMachine is required");

    TM = &TPC->getTM<TargetMachine>();
    return false;
  }

  bool runOnModule(Module &M) override {
    AnalysisGetter AG(this);
    return runImpl(M, AG, *TM);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<CycleInfoWrapperPass>();
  }

  StringRef getPassName() const override { return "RVGPU Attributor"; }
  TargetMachine *TM;
  static char ID;
};
} // namespace

PreservedAnalyses llvm::RVGPUAttributorPass::run(Module &M,
                                                  ModuleAnalysisManager &AM) {

  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  AnalysisGetter AG(FAM);

  // TODO: Probably preserves CFG
  return runImpl(M, AG, TM) ? PreservedAnalyses::none()
                            : PreservedAnalyses::all();
}

char RVGPUAttributorLegacy::ID = 0;

Pass *llvm::createRVGPUAttributorLegacyPass() {
  return new RVGPUAttributorLegacy();
}
INITIALIZE_PASS_BEGIN(RVGPUAttributorLegacy, DEBUG_TYPE, "RVGPU Attributor",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(CycleInfoWrapperPass);
INITIALIZE_PASS_END(RVGPUAttributorLegacy, DEBUG_TYPE, "RVGPU Attributor",
                    false, false)
