/*
 * 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
 */
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
#ifdef USE_MACA
#include "TritonMETAXGPUTransforms/MACACommon.h"
#include "TritonMETAXGPUTransforms/Passes.h"
#endif

using llvm::MapVector;
using namespace mlir;
namespace ttg = triton::gpu;
namespace tt = mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

#define int_attr(num) builder.getI64IntegerAttr(num)

namespace {

// Pass named attrs (e.g., tt.contiguity) from Triton to Triton
void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
#ifdef USE_MACA
  NamedAttrList attrs = op->getAttrs();
#else
  NamedAttrList attrs = op->getDiscardableAttrs();
#endif
  // Collect the attributes to propagate: the ones in dictAttrs and not yet on
  // the operation.
  SmallVector<NamedAttribute> toPropagate;
  for (const NamedAttribute attr : dictAttrs.getValue()) {
    if (!attrs.get(attr.getName()))
      toPropagate.push_back(attr);
  }
  // If we found any, let's set them here as a single step.
  if (toPropagate.size()) {
    attrs.append(toPropagate);
#ifdef USE_MACA
    op->setAttrs(attrs);
#else
    op->setDiscardableAttrs(attrs);
#endif
  }
}

#ifdef USE_MACA
void addNamedAttrsForCpAsync(Operation *op, DictionaryAttr dictAttrs) {

  NamedAttrList attrs = op->getAttrs();
  // Collect the attributes to propagate: the ones in dictAttrs and not yet on
  // the operation.
  SmallVector<NamedAttribute> toPropagate;
  for (NamedAttribute attr : dictAttrs.getValue()) {
    if (!attrs.get("tt.contiguity") && attr.getName() == "tt.contiguity") {
      Attribute val;
      val = attr.getValue();
      auto denseValue = dyn_cast<DenseElementsAttr>(val);
      Type denseValueType = denseValue.getElementType();

      auto valsInt = denseValue.getValues<int>();
      std::vector<APInt> vecShape;
      std::vector<int64_t> vecSize;
      vecShape.push_back(APInt(32, 1));
      vecSize.push_back(3);
      for (int i = 0; i < valsInt.size(); i++) {
        vecShape.push_back(APInt(32, valsInt[i]));
      }
      ArrayRef<APInt> vecShapeRef(vecShape);
      ArrayRef<int64_t> vecSizeRef(vecSize);

      auto shapeType = RankedTensorType::get(vecSizeRef, denseValueType);
      auto continAttr = DenseElementsAttr::get(shapeType, vecShapeRef);

      attr.setValue(continAttr);
      toPropagate.push_back(attr);
    }
  }
  // If we found any, let's set them here as a single step.
  if (toPropagate.size()) {
    attrs.append(toPropagate);
    op->setAttrs(attrs);
  }
}
#endif

class LoopPipeliner {
  /// Cache of ForOp and YieldOp related to this pipeliner.
  scf::ForOp forOp;
  scf::YieldOp yieldOp;

  /// Loads to be pipelined
  SetVector<Value> validLoads;
  /// The value that each load will be mapped to (after layout conversion)
  DenseMap<Value, Value> loadsMapping;
  /// load => buffer
  DenseMap<Value, Value> loadsBuffer;
  /// load => buffer type (with shared layout after swizzling)
  DenseMap<Value, tt::MemDescType> loadsBufferType;
  /// load => buffer at stage N
  DenseMap<Value, SmallVector<Value>> loadStageBuffer;
  /// load => after extractextractSlices
  DenseMap<Value, Value> loadsExtract;
#ifdef USE_MACA
  /// load => extrace at stage 0
  DenseMap<Value, Value> loadsExtractStage0;
  /// load => dotoperands
  /// TODO(MACA): only single dot support!
  DenseMap<Value, Value> dotOperandStage;
  DenseMap<Value, Value> nextDotOperands;
  // vector store cvt & trans
  SmallVector<Value> cvtAndTransVector;
  // vector store dots
  SmallVector<Value> dotVector;
  // load => if generate mask
  DenseMap<Value, bool> loadGenMask;
  // enable copy async saddr optimization.
  bool enableSaddrOpt = false;
  // if collect all ops can be pipelined.
  bool isFullStage = false;
  // addptrs to be pipelined
  SetVector<Value> validAddPtrs;
  // load => dep addptrs
  DenseMap<Value, Value> loadPtrMapping;
  // dep addptrs => orig ptr
  DenseMap<Value, Value> rootPtrMapping;
  // dep addptrs => newForLoop argIdx
  DenseMap<Value, int64_t> addPtrIdxMapping;
  // dep addptrs => offsets
  DenseMap<Value, SmallVector<Value>> ptrOffsetsMapping;
  // the chain of addptrs generate the ptr of valid ptrs.
  DenseMap<tt::AddPtrOp, SmallVector<tt::AddPtrOp>> validPtrDepMapping;

  /// deps op of dot operands in loop
  SetVector<Operation *> dotsDeps;

  /// deps op of dot operands out of loop
  SetVector<Operation *> dotsDepsOutOfLoop;

  /// dot deps => local load
  DenseMap<Value, Value> elemsMapping;

  /// cvt => buffer at stage N
  DenseMap<Value, Value> cvtStageBuffer;

  /// cvt => dep op
  DenseMap<Value, Value> cvtMapping;
#endif

  /// Iterator values
  Value pipelineIterIdx;
  Value loopIterIdx;
  Value nextIV;

  /// Yield values
#ifdef USE_MACA
  DenseMap<Value, SmallVector<Value>> nextAddPtrArgs;
#endif
  SmallVector<Value> nextBuffers;
  SmallVector<Value> extractSlices;
  SmallVector<Value> yieldValues;

  /// The number of stages in the pipeline.
  /// Stages in the range of [0, numStages-1) are in the prologue.
  /// numStages-1 is appended after the loop body.
  int numStages;

  /// Arg indicies
  size_t bufferIdx, loadIdx, depArgsBeginIdx, ivIndex;
  DenseMap<BlockArgument, size_t> depArgsIdx;

  /// value (in loop) => value at stage N
  DenseMap<Value, SmallVector<Value>> valueMapping;
  /// loop iter arg => value
  DenseMap<BlockArgument, Value> depArgsMapping;
  /// forOp value => newForOp value
  IRMapping mapping;
  /// forOp value => prefetch value
  IRMapping nextMapping;

  /// Dependency ops by program order
  SmallVector<Operation *> orderedDeps;

  /// arg => source operand defined stages
  DenseMap<BlockArgument, DenseSet<int>> immediateArgStages;

  /// block arguments that loads depend on
  SetVector<BlockArgument> depArgs;

  /// operation => source operand defined stages
  DenseMap<Operation *, DenseSet<int>> immediateOpStages;

  /// operations that loads depend on
  SetVector<Operation *> depOps;

  /// Collect all pipelinable ops
  LogicalResult collectOps(SetVector<Operation *> &ops);

  /// Collect values that `v` depends on and are defined inside the loop
  void collectValueDep(Value v, int stage, SetVector<Value> &opDeps);

  /// Collect all op dependencies
  void collectDeps(SetVector<Operation *> &ops,
                   MapVector<Operation *, SetVector<Value>> &opDeps);

  /// Check if none of the ops has valid uses
  LogicalResult checkOpUses(SetVector<Operation *> &ops);

  /// Check if ops have dependencies that are not pipelinable
  void checkOpDeps(SetVector<Operation *> &ops);

  void createBufferTypes();

  void createOrderedDeps();

  /// Return the stage at which `v` is defined prior to `stage`
  int getValueDefStage(Value v, int stage);

  /// Map `origin` to `newValue` at `stage`
  void setValueMapping(Value origin, Value newValue, int stage);

  /// Map `origin` to `newValue` at `stage` according to the association between
  /// yieldOp and forOp
  void setValueMappingYield(Value origin, Value newValue, int stage);

  /// Map `origin` to `newValue` at the next stage according to the association
  /// between yieldOp and forOp
  void setValueMappingYield(scf::ForOp newForOp, Value origin, Value newValue);

  /// Return the value mapped to `origin` at `stage`, if it exists.
  Value lookupOrDefault(Value origin, int stage);

  /// Get the load mask for `loadOp`, given the mapped mask `mappedMask` (if
  /// exists) and the current iteration's `loopCond`.
  Value getLoadMask(triton::LoadOp loadOp, Value mappedMask, Value loopCond,
                    OpBuilder &builder);

  /// Return an empty buffer of size <numStages, ...>
  ttg::LocalAllocOp allocateEmptyBuffer(triton::LoadOp loadOp,
                                        OpBuilder &builder);

  /// Collect all args of the new loop
  SmallVector<Value> collectNewLoopArgs();

  /// Clone the forOp and return the new forOp
  scf::ForOp cloneForOp(ArrayRef<Value> newLoopArgs, OpBuilder &builder);

  /// Prefetch the next iteration for `newForOp`
  void prefetchNextIteration(scf::ForOp newForOp, OpBuilder &builder);

  /// Assemble `newForOp`'s yield op
  void finalizeYield(scf::ForOp newForOp, OpBuilder &builder);

public:
  LoopPipeliner(scf::ForOp forOp, int numStages, bool isFullStage)
      : forOp(forOp), numStages(numStages), isFullStage(isFullStage) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  /// Collect loads to pipeline. Return success if we can pipeline this loop
  LogicalResult initialize();

  /// Emit pipelined loads (before loop body)
  void emitPrologue();

  /// emit pipelined loads (after loop body)
  void emitEpilogue();

  /// create the new ForOp (add new args & insert prefetched ops)
  scf::ForOp createNewForOp();

  friend struct PipelinePass;
};

/// Collect loads to pipeline. Return success if we can pipeline this loop
LogicalResult LoopPipeliner::collectOps(SetVector<Operation *> &ops) {
  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  mlir::triton::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
#ifdef USE_MACA
  int dotCnt = 0;
#endif

  // We cannot use forOp.walk(...) here because we only want to visit the
  // operations in the loop body block. Nested blocks are handled separately.
  for (Operation &op : forOp)
    if (auto loadOp = dyn_cast<triton::LoadOp>(&op)) {
#ifndef USE_MACA
      auto ptr = loadOp.getPtr();
      unsigned vec = axisInfoAnalysis.getPtrContiguity(ptr);

      if (auto mask = loadOp.getMask())
        vec = std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));

      auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
      if (!tensorTy || tensorTy.getRank() < 2)
        continue;
      auto ty =
          cast<triton::PointerType>(tensorTy.getElementType()).getPointeeType();
      unsigned width = vec * ty.getIntOrFloatBitWidth();
      // We do not pipeline all loads for the following reasons:
      // 1. cp.async's cp-size can only be 4, 8 and 16.
      // 2. It's likely that pipling small loads won't offer much performance
      //    improvement and may even hurt performance by increasing register
      //    pressure.
      if (width >= 32)
        ops.insert(loadOp);
    }
#else
      ops.insert(loadOp);
    } else if (auto dotOp = dyn_cast<triton::DotOp>(&op)) {
      dotCnt++;
    } else if (auto insertAsyncOp =
                   dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(&op)) {
      // check if not pipelined by previous TN/TT pass
      return failure();
    }

  if (dotCnt > 1)
    return failure();
#endif

  if (ops.empty())
    return failure();
  else
    return success();
}

void LoopPipeliner::collectValueDep(Value v, int stage,
                                    SetVector<Value> &deps) {
  // Loop-invariant value, skip
  if (v.getParentRegion() != forOp.getRegion())
    return;

  // Since we only need to peel the loop numStages-1 times, don't worry
  // about depends that are too far away
  if (stage < 0)
    return;

  if (auto arg = dyn_cast<BlockArgument>(v)) {
    if (arg.getArgNumber() > 0) {
      deps.insert(v);
      collectValueDep(yieldOp->getOperand(arg.getArgNumber() - 1), stage - 1,
                      deps);
    }
  } else { // value
    deps.insert(v);
    for (Value op : v.getDefiningOp()->getOperands())
      collectValueDep(op, stage, deps);
  }
}

void LoopPipeliner::collectDeps(
    SetVector<Operation *> &ops,
    MapVector<Operation *, SetVector<Value>> &valueDeps) {
  for (auto op : ops) {
    for (Value v : op->getOperands()) {
      SetVector<Value> deps;
      collectValueDep(v, numStages - 1, deps);
      valueDeps[op] = deps;
    }
  }
}

LogicalResult LoopPipeliner::checkOpUses(SetVector<Operation *> &ops) {
  DenseSet<Operation *> invalidOps;
  // Collect all ops' dependencies
  MapVector<Operation *, SetVector<Value>> opDeps;
  collectDeps(ops, opDeps);

  for (Operation *op : ops) {
    if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      // Don't pipeline valid loads that depend on other valid loads
      // (Because if a valid load depends on another valid load, this load needs
      // to wait on the other load in the prologue, which is against the point
      // of the pipeline pass)
      bool isCandidate = true;
      for (Operation *other : ops)
        if (isa<triton::LoadOp>(other))
          if (opDeps[op].contains(other->getResult(0))) {
            isCandidate = false;
            break;
          }
      // We only pipeline loads that have one covert_layout (to dot_op) use
      // TODO: lift this constraint in the future
      // TODO: Or if two convert_layout can be merged to one
      if (isCandidate && loadOp.getResult().hasOneUse()) {
        isCandidate = false;
        Operation *use = *loadOp.getResult().getUsers().begin();

        // Advance to the first conversion as long as the use resides in shared
        // memory and it has a single use itself
        while (use) {
          if (use->getNumResults() != 1 || !use->getResult(0).hasOneUse())
            break;
          auto tensorType =
              dyn_cast<RankedTensorType>(use->getResult(0).getType());
          if (tensorType &&
              !mlir::isa<ttg::SharedEncodingAttr>(tensorType.getEncoding()))
            break;
          use = *use->getResult(0).getUsers().begin();
        }

        if (auto convertLayout = llvm::dyn_cast<ttg::ConvertLayoutOp>(use)) {
          if (auto tensorType = dyn_cast<RankedTensorType>(
                  convertLayout.getResult().getType())) {
            if (auto dotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(
                    tensorType.getEncoding())) {
              isCandidate = true;
              loadsMapping[loadOp] = convertLayout;
            }
          }
        } else if (auto convertLayout = llvm::dyn_cast<ttg::LocalLoadOp>(use)) {
          if (auto tensorType = dyn_cast<RankedTensorType>(
                  convertLayout.getResult().getType())) {
            if (auto dotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(
                    tensorType.getEncoding())) {
              isCandidate = true;
              loadsMapping[loadOp] = convertLayout;
            }
          }
        }
      } else
        isCandidate = false;

      if (!isCandidate)
        invalidOps.insert(loadOp);
      else
        validLoads.insert(loadOp);
    }
  }

  for (Operation *op : invalidOps)
    ops.remove(op);

  if (ops.empty())
    return failure();
  else
    return success();
}

void LoopPipeliner::checkOpDeps(SetVector<Operation *> &ops) {
  SetVector<BlockArgument> nonImmediateDepArgs;
  SetVector<Operation *> nonImmediateOps;
  for (Operation *op : ops) {
    for (Value v : op->getOperands()) {
      SetVector<Value> deps;
      collectValueDep(v, numStages - 1, deps);
      int defStage = getValueDefStage(v, numStages - 1);
      assert(defStage >= 0 &&
             "newLoopArgs has null args without a define op. Consider either "
             "rewrite the loop to reduce cross iteration dependencies or "
             "increase the num_stages value.");
      for (auto dep : deps) {
        auto immediate = isa<BlockArgument>(deps.front());
        if (auto arg = dyn_cast<BlockArgument>(dep)) {
          depArgs.insert(arg);
          if (immediate)
            immediateArgStages[arg].insert(defStage);
          else
            nonImmediateDepArgs.insert(arg);
        } else {
#ifdef USE_MACA
          if (auto load = dyn_cast<triton::LoadOp>(op)) {
            if (auto addptr = dyn_cast<triton::AddPtrOp>(dep.getDefiningOp())) {
              validAddPtrs.insert(addptr);
              loadPtrMapping[load] = addptr;
            }
          }
#endif
          depOps.insert(dep.getDefiningOp());
          if (immediate)
            immediateOpStages[dep.getDefiningOp()].insert(defStage);
          else
            nonImmediateOps.insert(dep.getDefiningOp());
        }
      }
    }
  }

#ifdef USE_MACA
  // saddr optimization only if each load mapped only one addptrs.
  if (validLoads.size() == validAddPtrs.size()) {
    enableSaddrOpt = true;
  }
#endif

  // XXX: We could remove the following constraints if we can rematerialize in
  // the loop.
  // Check if immediateDepArgs and nonImmediateDepArgs are disjoint.
  for (auto &[arg, stages] : immediateArgStages) {
    assert(stages.size() == 1 &&
           "Triton doesn't support an argument provides values for "
           "immediate operands of loads from multiple stages. Consider "
           "removing post load instructions dependency on this argument.");
    assert(!(nonImmediateDepArgs.contains(arg) &&
             stages.contains(numStages - 2)) &&
           "Loop-carried arguments provide values for both immediate and "
           "non-immediate operands of loads. Please consider removing "
           "pre/post load instructions dependency on this argument.");
  }

  // Check if immediateOps and nonImmediateOps are disjoint.
  for (auto &[op, stages] : immediateOpStages) {
    assert(stages.size() == 1 &&
           "Triton doesn't support an operation provides values for "
           "immediate operands of loads from multiple stages. Consider "
           "removing post load instructions dependency on this argument.");
    assert(!(nonImmediateOps.contains(op) && stages.contains(numStages - 2)) &&
           "Operations provide values for both immediate and "
           "non-immediate operands of loads.  Please consider "
           "removing pre/post load instructions dependency on this "
           "operation.");
  }
}

// helpers
void LoopPipeliner::setValueMapping(Value origin, Value newValue, int stage) {
  if (valueMapping.find(origin) == valueMapping.end())
    valueMapping[origin] = SmallVector<Value>(numStages);
  valueMapping[origin][stage] = newValue;
}

void LoopPipeliner::setValueMappingYield(Value origin, Value newValue,
                                         int stage) {
  for (OpOperand &operand : origin.getUses()) {
    if (operand.getOwner() == yieldOp) {
      auto yieldIdx = operand.getOperandNumber();
      auto value = forOp.getRegionIterArgs()[yieldIdx];
      setValueMapping(value, newValue, stage);
    }
  }
}

void LoopPipeliner::setValueMappingYield(scf::ForOp newForOp, Value origin,
                                         Value newValue) {
  for (OpOperand &operand : origin.getUses()) {
    if (operand.getOwner() == yieldOp) {
      auto yieldIdx = operand.getOperandNumber();
      auto depYieldIdx = depArgsIdx[forOp.getRegionIterArgs()[yieldIdx]];
      auto originArg = forOp.getRegionIterArgs()[yieldIdx];
      nextMapping.map(originArg, newValue);
      auto newArg = newForOp.getRegionIterArgs()[depYieldIdx];
#ifdef USE_MACA
      if (depArgsMapping.find(newArg) == depArgsMapping.end())
#else
      if (!depArgsMapping.contains(newArg))
#endif
        depArgsMapping[newArg] = newValue;
    }
  }
}

Value LoopPipeliner::lookupOrDefault(Value origin, int stage) {
  if (valueMapping.find(origin) == valueMapping.end())
    return origin;
  return valueMapping[origin][stage];
}

void LoopPipeliner::createBufferTypes() {
  for (auto loadCvt : loadsMapping) {
    auto loadOp = loadCvt.first;
    Value cvt = loadCvt.second;
    auto enc = cast<RankedTensorType>(cvt.getType()).getEncoding();
    auto dotOpEnc = cast<ttg::DotOperandEncodingAttr>(enc);
    auto ty = cast<RankedTensorType>(loadOp.getType());
    SmallVector<int64_t> bufferShape(ty.getShape().begin(),
                                     ty.getShape().end());
#ifdef USE_MACA
    unsigned bitWidth = ty.getElementType().getIntOrFloatBitWidth();
    bufferShape.insert(bufferShape.begin(), numStages - 1);
    //
    // match pattern:
    //   %0 load ptr
    //   %1 lalloc %0->shared
    //   %2 trans %1->shared1
    //   %3 lload %2->dot
    // we get tensor type of %1, so it is necessary to create shared encoding
    // with trans flag.
    Operation *cvt_ = cvt.getDefiningOp();
    Value cvtSrc;
    if (auto convertLayout = llvm::dyn_cast<ttg::LocalLoadOp>(cvt_))
      cvtSrc = convertLayout.getSrc();
    if (auto convertLayout = llvm::dyn_cast<ttg::ConvertLayoutOp>(cvt_))
      cvtSrc = convertLayout.getSrc();
    Operation *trans = cvtSrc.getDefiningOp();
    if (auto transOp = llvm::dyn_cast<triton::TransOp>(trans)) {
      auto sharedEnc = ttg::SharedEncodingAttr::get(
          ty.getContext(), dotOpEnc, ty.getShape(),
          ttg::getOrder(ty.getEncoding()), ttg::getCTALayout(ty.getEncoding()),
          ty.getElementType(), true);
      loadsBufferType[loadOp] = tt::MemDescType::get(
          bufferShape, ty.getElementType(), sharedEnc, /*mutable*/ true);
    } else {
      auto sharedEnc = ttg::SharedEncodingAttr::get(
          ty.getContext(), dotOpEnc, ty.getShape(),
          ttg::getOrder(ty.getEncoding()), ttg::getCTALayout(ty.getEncoding()),
          ty.getElementType());
      loadsBufferType[loadOp] = tt::MemDescType::get(
          bufferShape, ty.getElementType(), sharedEnc, /*mutable*/ true);
    }
#else
    unsigned bitWidth = dotOpEnc.getMMAv2kWidth()
                            ? 32 / dotOpEnc.getMMAv2kWidth()
                            : ty.getElementType().getIntOrFloatBitWidth();
    bufferShape.insert(bufferShape.begin(), numStages);
    auto sharedEnc =
        ttg::SharedEncodingAttr::get(ty.getContext(), dotOpEnc, ty.getShape(),
                                     ttg::getOrder(ty.getEncoding()), bitWidth);
    loadsBufferType[loadOp] =
        RankedTensorType::get(bufferShape, ty.getElementType(), sharedEnc);
#endif
  }
}

void LoopPipeliner::createOrderedDeps() {
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (depOps.contains(&op))
      orderedDeps.push_back(&op);
    else if (op.getNumResults() > 0 && validLoads.contains(op.getResult(0)))
      orderedDeps.push_back(&op);
  }
  assert(depOps.size() + validLoads.size() == orderedDeps.size() &&
         "depOps contains invalid values");
}

int LoopPipeliner::getValueDefStage(Value v, int stage) {
  if (stage < 0)
    return -1;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    if (arg.getArgNumber() > 0)
      return getValueDefStage(yieldOp->getOperand(arg.getArgNumber() - 1),
                              stage - 1);
    llvm_unreachable("Loop induction variable should not be a dependency");
  } else
    return stage;
}

ttg::LocalAllocOp LoopPipeliner::allocateEmptyBuffer(triton::LoadOp loadOp,
                                                     OpBuilder &builder) {
  // Allocate a buffer for each pipelined tensor
  // shape: e.g. (numStages==4), <32x64xbf16> -> <4x32x64xbf16>
  Value convertLayout = loadsMapping[loadOp];
  if (auto tensorType = dyn_cast<RankedTensorType>(convertLayout.getType()))
    return builder.create<ttg::LocalAllocOp>(convertLayout.getLoc(),
                                             loadsBufferType[loadOp], Value());
  llvm_unreachable("Async copy's return should be of RankedTensorType");
}

LogicalResult LoopPipeliner::initialize() {
  // All ops that maybe pipelined
  SetVector<Operation *> ops;

#ifdef USE_MACA
  // number of numStages load in prologue
  numStages += 1;

  if (isFullStage) {
    if (collectValidOp(forOp, yieldOp, numStages, loadsMapping, elemsMapping,
                       validLoads, dotsDeps, dotsDepsOutOfLoop, ops, true)
            .failed())
      return failure();
  } else {
    if (collectOps(ops).failed())
      return failure();

    if (checkOpUses(ops)
            .failed()) // check if two cvt layout can be merged to one
      return failure();
  }
  // TODO(MACA): support single cpasync
  if (validLoads.size() < 2)
    return failure();
#endif

  checkOpDeps(ops);

  createBufferTypes(); // create one buffer, define sharedEncoding here (cpy to
                       // checkOpUses

  createOrderedDeps();

#ifdef USE_MACA
  if (checkMaskDeps(forOp, ops, loadGenMask, loadsBufferType).failed())
    return failure();

  collectAddPtrDeps(forOp, enableSaddrOpt, orderedDeps, validAddPtrs,
                    validPtrDepMapping);
#endif

  return success();
}

Value LoopPipeliner::getLoadMask(triton::LoadOp loadOp, Value mappedMask,
                                 Value loopCond, OpBuilder &builder) {
  Type maskType = triton::getI1SameShape(loadOp.getType());
  Value mask = loadOp.getMask();
  Value newMask;
  if (mask) {
    Value cond = loopCond;
    if (isa<RankedTensorType>(maskType)) {
      cond = builder.create<triton::SplatOp>(mask.getLoc(), maskType, loopCond);
    }
    newMask = builder.create<arith::AndIOp>(mask.getLoc(), mappedMask, cond);
  } else {
    if (isa<RankedTensorType>(maskType)) {
      newMask = builder.create<triton::SplatOp>(loopCond.getLoc(), maskType,
                                                loopCond);
    } else {
      newMask = loopCond;
    }
  }
  return newMask;
}

void LoopPipeliner::emitPrologue() {
  OpBuilder builder(forOp);
  // Get init operands for loop carried values
  for (BlockArgument &arg : forOp.getRegionIterArgs()) {
    OpOperand *operand = forOp.getTiedLoopInit(arg);
    setValueMapping(arg, operand->get(), 0);
  }

  // Emit prologue from [0, numStage-1)
  Value iv = forOp.getLowerBound();
  pipelineIterIdx = builder.create<arith::ConstantIntOp>(iv.getLoc(), 0, 32);
  for (int stage = 0; stage < numStages - 1; ++stage) {
    // Special handling for induction variable as the increment is implicit
    if (stage != 0)
      iv = builder.create<arith::AddIOp>(iv.getLoc(), iv, forOp.getStep());
    setValueMapping(forOp.getInductionVar(), iv, stage);

    // Special handling for loop condition as there is no condition in ForOp
    Value loopCond = builder.create<arith::CmpIOp>(
        iv.getLoc(), arith::CmpIPredicate::slt, iv, forOp.getUpperBound());
    for (Operation *op : orderedDeps) {
      Operation *newOp = nullptr;
#ifdef USE_MACA
      auto addptr = llvm::dyn_cast<triton::AddPtrOp>(op);
#endif
      if (validLoads.contains(op->getResult(0))) {
        auto load = cast<triton::LoadOp>(op);
        // Allocate empty buffer
        if (stage == 0) {
          loadsBuffer[load] = allocateEmptyBuffer(load, builder);
          // loadStageBuffer[load] = {loadsBuffer[load]};
        }
        // load => copy async
        if (auto loadOp = llvm::dyn_cast<triton::LoadOp>(op)) {
          Value newMask =
              getLoadMask(loadOp, lookupOrDefault(loadOp.getMask(), stage),
                          loopCond, builder);
#ifdef USE_MACA
          auto srcTy = cast<RankedTensorType>(loadOp.getPtr().getType());
          auto srcEncoding =
              cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
          tt::MemDescType allocTy =
              cast<tt::MemDescType>(loadsBuffer[loadOp].getType());
          auto bufferShape = allocTy.getShape();
          auto resEncoding =
              cast<ttg::SharedEncodingAttr>(allocTy.getEncoding());
          if (loadGenMask[loadOp]) {
            // generate swizzle mask
            newMask = genSwiSubMask(newMask, loadOp, srcEncoding, resEncoding,
                                    builder, {bufferShape[1], bufferShape[2]},
                                    {0, 0}, numStages);
          }
          Value zero =
              builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);

          tt::MemDescType subviewTy = tt::MemDescType::get(
              allocTy.getShape().drop_front(), allocTy.getElementType(),
              allocTy.getEncoding(), /*mutableMemory=*/true);
          SmallVector<Value> copyOffsets(allocTy.getRank(), zero);
          copyOffsets[0] = pipelineIterIdx;
          Value subViewBuffer = builder.create<ttg::MemDescSubviewOp>(
              op->getLoc(), subviewTy, loadsBuffer[load], copyOffsets,
              /*isConstantOffs*/ false);
          newOp = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
              op->getLoc(), lookupOrDefault(loadOp.getPtr(), stage),
              subViewBuffer, newMask, lookupOrDefault(loadOp.getOther(), stage),
              loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile(),
              /*intrinsic*/ true);
          addNamedAttrsForCpAsync(newOp, op->getAttrDictionary());
#else
          newOp = builder.create<ttg::InsertSliceAsyncOp>(
              op->getLoc(), loadsBuffer[loadOp].getType(),
              lookupOrDefault(loadOp.getPtr(), stage),
              loadStageBuffer[loadOp][stage], pipelineIterIdx, newMask,
              lookupOrDefault(loadOp.getOther(), stage), loadOp.getCache(),
              loadOp.getEvict(), loadOp.getIsVolatile(), /*axis*/ 0);
          builder.create<ttg::AsyncCommitGroupOp>(op->getLoc());
#endif
          if (stage == 0) {
            loadStageBuffer[loadOp] = {subViewBuffer};
          } else {
            loadStageBuffer[loadOp].push_back(subViewBuffer);
          }
        } else
          llvm_unreachable("This should be LoadOp");
#ifdef USE_MACA
      } else if (addptr &&
                 validPtrDepMapping.find(addptr) != validPtrDepMapping.end()) {
        newOp = builder.clone(*op);
        auto it = valueMapping.find(op->getOperand(1));
        Value loopOffset = newOp->getOperand(1);
        if (it != valueMapping.end()) {
          Value v = it->second[stage];
          assert(v && "Value not found in valueMapping");
          loopOffset = v;
        }
        // initialize rootPtrMapping and ptrOffsetsMapping when stage == 0;
        if (stage == 0) {
          auto addptrs = validPtrDepMapping[addptr];
          Value rootPtr, prevOffset;
          for (int i = 0; i < addptrs.size(); ++i) {
            rootPtr = addptrs[i].getPtr();
            auto accOffset = addptrs[i].getOffset();
            if (prevOffset) {
              prevOffset = addInt32OrInt64(builder, addptr.getLoc(), prevOffset,
                                           accOffset, newOp);
            } else {
              prevOffset = accOffset;
            }
          }
          rootPtrMapping[addptr] = rootPtr;
          ptrOffsetsMapping[addptr].push_back(prevOffset);
        }
        loopOffset = addInt32OrInt64(builder, addptr.getLoc(),
                                     ptrOffsetsMapping[addptr][stage],
                                     loopOffset, newOp);
        if (stage != numStages - 2) {
          ptrOffsetsMapping[addptr].push_back(loopOffset);
        }
        newOp->setOperand(0, rootPtrMapping[addptr]);
        newOp->setOperand(1, loopOffset);
#endif
      } else {
        if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
          Value newMask =
              getLoadMask(loadOp, lookupOrDefault(loadOp.getMask(), stage),
                          loopCond, builder);
          newOp = builder.create<triton::LoadOp>(
              loadOp.getLoc(), loadOp.getResult().getType(),
              lookupOrDefault(loadOp.getPtr(), stage), newMask,
              lookupOrDefault(loadOp.getOther(), stage),
              loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
              loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
#ifdef USE_MACA
          addNamedAttrs(newOp, op->getAttrDictionary());
#else
          addNamedAttrs(newOp, op->getDiscardableAttrDictionary());
#endif
        } else
          newOp = builder.clone(*op);
        // Update loop-carried uses
        for (unsigned opIdx = 0; opIdx < op->getNumOperands(); ++opIdx) {
          auto it = valueMapping.find(op->getOperand(opIdx));
          if (it != valueMapping.end()) {
            Value v = it->second[stage];
            assert(v && "Value not found in valueMapping");
            newOp->setOperand(opIdx, v);
          } // else, op at opIdx is a loop-invariant value
        }
      }

      for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults())) {
        Value originResult = op->getResult(dstIdx);
        if (validLoads.contains(originResult))
          break;
        setValueMapping(originResult, newOp->getResult(dstIdx), stage);
        // Update mapping for loop-carried values (args)
        setValueMappingYield(op->getResult(dstIdx), newOp->getResult(dstIdx),
                             stage + 1);
      }
    } // for (Operation *op : orderedDeps)

    // Update pipeline index
    pipelineIterIdx = builder.create<arith::AddIOp>(
        iv.getLoc(), pipelineIterIdx,
        builder.create<arith::ConstantIntOp>(iv.getLoc(), 1, 32));
    // Some values have not been used by any ops in the loop body
    for (BlockArgument arg : forOp.getRegionIterArgs())
      setValueMappingYield(arg, valueMapping[arg][stage], stage + 1);
  } // for (int stage = 0; stage < numStages-1; ++stage)

  // async.wait & extract_slice & cvt
  // eg.num_stages == 2
  // memdesc_subview SA[0], SB[0] -> SA0, SB0
  // async_copy_global_to_local A0, B0 -> SA0, SB0
  // memdesc_subview SA[1], SB[1] -> SA1, SB1
  // async_copy_global_to_local A1, B1 -> SA1, SB1
  // async_wait(1)
  // local_load SA0, SB0 -> RA_DOT_0, RB_DOT_0
  // for ...
#ifdef USE_MACA
  loopIterIdx = builder.create<arith::ConstantIntOp>(iv.getLoc(), 0, 32);
  for (Value loadOp : validLoads) {
    // extract_slice
    // stage==0
    loadsExtractStage0[loadOp] = loadStageBuffer[loadOp][0];
    // stage==1
    loadsExtract[loadOp] = loadStageBuffer[loadOp][1];
  }
  // Bump up loopIterIdx, this is used for getting the correct slice for the
  // `next` iteration
  // add 2 stages
  loopIterIdx = builder.create<arith::AddIOp>(
      loopIterIdx.getLoc(), loopIterIdx,
      builder.create<arith::ConstantIntOp>(loopIterIdx.getLoc(), 2, 32));
  // async.wait
  int gvmNumberPerOp = 0;
  for (Value loadOp : validLoads) {
    auto blockType = dyn_cast<RankedTensorType>(loadOp.getType());
    auto blockEnc =
        dyn_cast<triton::gpu::BlockedEncodingAttr>(blockType.getEncoding());
    gvmNumberPerOp += getGVMNumberPerOp(blockEnc, blockType);
  }
  //  builder.create<ttg::AsyncWaitOp>(validLoads.front().getLoc(),
  //                                                         gvmNumberPerOp *
  //                                                         (numStages - 2));
  builder.create<ttg::GVMArriveOp>(validLoads.front().getLoc(),
                                   gvmNumberPerOp * (numStages - 2));
  Operation *asyncWait =
      builder.create<ttg::BarrierSharedOp>(validLoads[0].getLoc());
  // cvt
  // stage==0
  for (Value loadOp : validLoads) {
    // TODO: support multiple cvt for flashattn bwd
    // if pattern 1
    //   %0 = load %ptr
    //   %1 = cvt(blocked->dot) %0
    // then
    //   memdesc_subview -> %0
    //   %0 = async_copy_global_to_local %ptr
    //   %1 = local_load(shared->dot) %0 (new)

    // if pattern 2
    //   %0 = load %ptr
    //   %1 = local_alloc %0
    //   %2 = trans %1
    //   %3 = local_load %2
    // then
    //   memdesc_subview -> %0
    //   %0 = async_copy_global_to_local %ptr
    //   %1 = trans %0
    //   %2 = local_load %1

    Value cvt = loadsMapping[loadOp];
    Value src = loadsExtractStage0[loadOp];
    Value cvtSrc = cvt.getDefiningOp()->getOperand(0);
    auto srcTy = dyn_cast<tt::MemDescType>(src.getType());
    assert(srcTy && "unsupported pattern in PipelineAsyncBase!");
    auto transOp = dyn_cast<triton::TransOp>(cvtSrc.getDefiningOp());
    if (transOp) {
      auto transSrc = transOp.getSrc();
      auto transTy = dyn_cast<triton::MemDescType>(transSrc.getType());
      assert(transTy && "unsupported pattern in PipelineAsyncBase!");
      src = builder.create<triton::TransOp>(transOp.getLoc(), src,
                                            transOp.getOrder());
    }
    auto dstTy = dyn_cast<RankedTensorType>(cvt.getType());
    assert(dstTy && "unsupported pattern in PipelineAsyncBase!");
    auto dotEnc = dyn_cast<ttg::DotOperandEncodingAttr>(dstTy.getEncoding());
    assert(dotEnc && "unsupported pattern in PipelineAsyncBase!");
    auto newDstTy = RankedTensorType::get(
        dstTy.getShape(), srcTy.getElementType(), dstTy.getEncoding());
    Value lload = builder.create<triton::gpu::LocalLoadOp>(
        cvt.getDefiningOp()->getLoc(), newDstTy, src, nullptr, true);
    if (dotEnc.getOpIdx() == 0) {
      dotOperandStage[loadOp] = lload; // a
    } else {
      dotOperandStage[loadOp] = lload; // b
    }
  }
#else
  builder.create<ttg::AsyncWaitOp>(validLoads.front().getLoc(),
                                   validLoads.size() * (numStages - 2));
  loopIterIdx = builder.create<arith::ConstantIntOp>(iv.getLoc(), 0, 32);
  for (Value loadOp : validLoads) {
    auto bufferType = loadStageBuffer[loadOp][numStages - 1]
                          .getType()
                          .cast<RankedTensorType>();
    auto bufferShape = bufferType.getShape();
    auto sliceType = loadsMapping[loadOp].getType().cast<RankedTensorType>();
    sliceType = RankedTensorType::get({bufferShape[1], bufferShape[2]},
                                      sliceType.getElementType(),
                                      loadsBufferType[loadOp].getEncoding());
    Value extractSlice = builder.create<ttg::ExtractSliceOp>(
        loadOp.getLoc(), sliceType, loadStageBuffer[loadOp][numStages - 1],
        SmallVector<OpFoldResult>{int_attr(0), int_attr(0), int_attr(0)},
        SmallVector<OpFoldResult>{int_attr(1),
                                  int_attr(sliceType.getShape()[0]),
                                  int_attr(sliceType.getShape()[1])},
        SmallVector<OpFoldResult>{int_attr(1), int_attr(1), int_attr(1)});
    loadsExtract[loadOp] = extractSlice;
  }
  // Bump up loopIterIdx, this is used for getting the correct slice for the
  // `next` iteration
  loopIterIdx = builder.create<arith::AddIOp>(
      loopIterIdx.getLoc(), loopIterIdx,
      builder.create<arith::ConstantIntOp>(loopIterIdx.getLoc(), 1, 32));
#endif
}

void LoopPipeliner::emitEpilogue() {
  // If there's any outstanding async copies, we need to wait for them.
  OpBuilder builder(forOp);
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointAfter(forOp);
  builder.create<ttg::GVMArriveOp>(forOp.getLoc(), 0);
  Operation *asyncWait = builder.create<ttg::BarrierSharedOp>(forOp.getLoc());
}

SmallVector<Value> LoopPipeliner::collectNewLoopArgs() {
  // Order of new args:
  //   (original args)
  //   (insertSliceAsync buffer at stage numStages - 1) for each load
  //   (extracted tensor) for each load
  //   (dotOp reg) for each dot (?)
  //   (depArgs at stage numStages - 1)
  //   (depArgs at stage numStages - 2)
  //   ...
  //   (iv at stage numStages - 2)
  //   (pipeline iteration index)
  //   (loop iteration index)

  // We need this to update operands for yield
  // original block arg => new arg's idx
  SmallVector<Value> newLoopArgs;
  for (auto v : forOp.getInits()) {
    newLoopArgs.push_back(v);
  }

  bufferIdx = newLoopArgs.size();
  for (auto loadOp : validLoads) {
    newLoopArgs.push_back(loadsBuffer[loadOp]);
  }

  loadIdx = newLoopArgs.size();
  for (auto loadOp : validLoads)
    newLoopArgs.push_back(loadsExtract[loadOp]);

#ifdef USE_MACA
  // change to multidot
  for (auto loadOp : validLoads)
    newLoopArgs.push_back(dotOperandStage[loadOp]);

  for (auto addptr : validAddPtrs) {
    int64_t idx = newLoopArgs.size();
    if (rootPtrMapping.find(addptr) != rootPtrMapping.end()) {
      int mapsize = ptrOffsetsMapping[addptr].size();
      addPtrIdxMapping[addptr] = idx;
      newLoopArgs.push_back(rootPtrMapping[addptr]);
      newLoopArgs.push_back(ptrOffsetsMapping[addptr][mapsize - 1]);
    }
  }
#endif

  depArgsBeginIdx = newLoopArgs.size();
  for (auto depArg : depArgs) {
    depArgsIdx[depArg] = newLoopArgs.size();
    if (immediateArgStages[depArg].contains(numStages - 2))
      // Peel off post load ops in numStage-1
      newLoopArgs.push_back(valueMapping[depArg][numStages - 2]);
    else
      newLoopArgs.push_back(valueMapping[depArg][numStages - 1]);
  }

  ivIndex = newLoopArgs.size();
  newLoopArgs.push_back(valueMapping[forOp.getInductionVar()][numStages - 2]);
  newLoopArgs.push_back(pipelineIterIdx);
  newLoopArgs.push_back(loopIterIdx);
  return newLoopArgs;
}

scf::ForOp LoopPipeliner::cloneForOp(ArrayRef<Value> newLoopArgs,
                                     OpBuilder &builder) {
  // Clone the original ForOp
  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newLoopArgs);

  // Set mapping on body of the new ForOp
  builder.setInsertionPointToStart(newForOp.getBody());
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  // Clone the loop body, replace original args with args of the new ForOp.
  // We want to find cvt ops that match the following pattern:
  // %0 = load %ptr
  // %1 (dotOperand) = cvt %0
  for (Operation &op : forOp.getBody()->without_terminator()) {
#ifdef USE_MACA
    if (dotsDeps.contains(&op))
      continue;
    if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
      auto result = op.getResult(0);
      auto a = newForOp.getRegionIterArgs()[loadIdx + validLoads.size()];
      auto b = newForOp.getRegionIterArgs()[loadIdx + validLoads.size() + 1];
      nextMapping.map(dotOp.getA(), a);
      nextMapping.map(dotOp.getB(), b);
      Value c = dotOp.getC();
      // dotOprand C may come from forLoopArg/forLoopBody/beforeLoop
      if (isa<BlockArgument>(c) || c.getParentBlock() == forOp.getBody()) {
        c = mapping.lookup(c);
      }

      if (a && b && c) {
        Operation *newDot = builder.create<triton::DotOp>(
            op.getLoc(), a, b, c, dotOp.getInputPrecision(),
            dotOp.getMaxNumImpreciseAcc());
        mapping.map(result, newDot->getResult(0));
        nextMapping.map(result, newDot->getResult(0));
        dotVector.push_back(newDot->getResult(0));
        continue;
      } else {
        assert(false && "Invalid Encoding for PipelineAsyncBase!");
      }
    } else if (auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
      // if pattern 1
      //   %0 = load %ptr
      //   %1 = cvt(blocked->dot) %0
      auto cvtResult = op.getResult(0);
      auto cvtDstTy = cast<RankedTensorType>(cvtResult.getType());
      if (isa<ttg::DotOperandEncodingAttr>(cvtDstTy.getEncoding())) {
        Operation *prev_op = cvtOp.getSrc().getDefiningOp();
        if (isa<triton::LoadOp>(prev_op)) {
          auto it =
              std::find(validLoads.begin(), validLoads.end(), op.getOperand(0));
          if (it != validLoads.end()) {
            // We replace the use new load use with a convert layout
            auto loadArgIdx = std::distance(validLoads.begin(), it);
            auto dotEnc =
                dyn_cast<ttg::DotOperandEncodingAttr>(cvtDstTy.getEncoding());
            auto cvt = builder.create<ttg::LocalLoadOp>(
                cvtResult.getLoc(), cvtDstTy,
                newForOp.getRegionIterArgs()[loadIdx + loadArgIdx], nullptr,
                /*intrinsic*/ true);
            mapping.map(cvtResult, cvt.getResult());
            nextMapping.map(cvtResult, cvt.getResult());
            if (dotEnc.getOpIdx() == 0) {
              nextDotOperands[cvtOp.getSrc()] = cvt.getResult();
            } else {
              nextDotOperands[cvtOp.getSrc()] = cvt.getResult();
            }
            cvtAndTransVector.push_back(cvt.getResult());
            continue;
          }
        }
      }
    } else if (auto cvtOp = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
      // if pattern 2
      //   %0 = load %ptr
      //   %1 = local_alloc %0
      //   %2 = trans %1
      //   %3 = local_load %2
      Operation *prev_op = cvtOp.getSrc().getDefiningOp();
      auto transOp = dyn_cast<triton::TransOp>(prev_op);
      if (transOp) {
        Operation *local_alloc = transOp.getSrc().getDefiningOp();
        // TODO(MACA): only one dot
        auto laOp1 = dyn_cast<triton::gpu::LocalAllocOp>(local_alloc);
        if (laOp1) {
          auto it =
              std::find(validLoads.begin(), validLoads.end(), laOp1.getSrc());
          if (it != validLoads.end()) {
            auto cvtResult = op.getResult(0);
            auto cvtDstTy = cast<RankedTensorType>(cvtResult.getType());
            auto dotEnc =
                dyn_cast<ttg::DotOperandEncodingAttr>(cvtDstTy.getEncoding());
            auto loadArgIdx = std::distance(validLoads.begin(), it);
            Value newTrans = builder.create<triton::TransOp>(
                transOp.getLoc(),
                newForOp.getRegionIterArgs()[loadIdx + loadArgIdx],
                transOp.getOrder());
            auto cvt = builder.create<ttg::LocalLoadOp>(
                transOp.getLoc(), cvtDstTy, newTrans, nullptr,
                /*intrinsic*/ true);
            mapping.map(cvtResult, cvt.getResult());
            nextMapping.map(cvtResult, cvt.getResult());
            if (dotEnc.getOpIdx() == 0) {
              nextDotOperands[laOp1.getSrc()] = cvt.getResult();
            } else {
              nextDotOperands[laOp1.getSrc()] = cvt.getResult();
            }
            cvtAndTransVector.push_back(newTrans);
            cvtAndTransVector.push_back(cvt.getResult());
            continue;
          }
        }
      }
    }
#else
    if (auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
      auto result = op.getResult(0);
      auto cvtDstTy = result.getType().cast<RankedTensorType>();
      if (cvtDstTy.getEncoding().isa<ttg::DotOperandEncodingAttr>()) {
        auto it =
            std::find(validLoads.begin(), validLoads.end(), op.getOperand(0));
        if (it != validLoads.end()) {
          // We replace the use new load use with a convert layout
          auto loadArgIdx = std::distance(validLoads.begin(), it);
          auto cvt = builder.create<ttg::ConvertLayoutOp>(
              result.getLoc(), cvtDstTy,
              newForOp.getRegionIterArgs()[loadIdx + loadArgIdx]);
          mapping.map(result, cvt.getResult());
          continue;
        }
      }
    }
#endif
    cloneWithInferType(builder, &op, mapping);
  }

  return newForOp;
}

void LoopPipeliner::prefetchNextIteration(scf::ForOp newForOp,
                                          OpBuilder &builder) {
#ifdef USE_MACA
  builder.setInsertionPointToStart(newForOp.getBody());
  size_t initArgIdx = 0;
  for (auto v : forOp.getRegionIterArgs()) {
    BlockArgument nextArg = newForOp.getRegionIterArgs()[initArgIdx];
    nextMapping.map(v, nextArg);
    ++initArgIdx;
  }
#endif
  // Map the dep args of the next iteration to the dep args of the current
  size_t argIdx = 0;
  for (auto depArg : depArgs) {
    BlockArgument nextArg =
        newForOp.getRegionIterArgs()[argIdx + depArgsBeginIdx];
    nextMapping.map(depArg, nextArg);
    ++argIdx;
  }

  // Special handling for iv & loop condition
  Value curIV = newForOp.getRegionIterArgs()[ivIndex];
  nextIV = builder.create<arith::AddIOp>(newForOp.getInductionVar().getLoc(),
                                         curIV, newForOp.getStep());
  Value nextLoopCond =
      builder.create<arith::CmpIOp>(nextIV.getLoc(), arith::CmpIPredicate::slt,
                                    nextIV, newForOp.getUpperBound());

  pipelineIterIdx = newForOp.getRegionIterArgs()[ivIndex + 1];
#ifdef USE_MACA
  Value insertSliceIndex = builder.create<arith::RemSIOp>(
      nextIV.getLoc(), pipelineIterIdx,
      builder.create<arith::ConstantIntOp>(nextIV.getLoc(), numStages - 1, 32));
  loopIterIdx = newForOp.getRegionIterArgs()[ivIndex + 2];
  Value extractSliceIndex = builder.create<arith::RemSIOp>(
      nextIV.getLoc(), loopIterIdx,
      builder.create<arith::ConstantIntOp>(nextIV.getLoc(), numStages - 1, 32));
#else
  Value insertSliceIndex = builder.create<arith::RemSIOp>(
      nextIV.getLoc(), pipelineIterIdx,
      builder.create<arith::ConstantIntOp>(nextIV.getLoc(), numStages, 32));
  loopIterIdx = newForOp.getRegionIterArgs()[ivIndex + 2];
  Value extractSliceIndex = builder.create<arith::RemSIOp>(
      nextIV.getLoc(), loopIterIdx,
      builder.create<arith::ConstantIntOp>(nextIV.getLoc(), numStages, 32));
#endif
  // Prefetch load deps
  // If a load-dependent instruction that uses a block argument, we
  // shouldn't update the new mapping of the block argument in the current
  // iteration.
  // For example.
  // %a = add %arg0, %c
  // %b = add %arg0, %d
  //
  // Update %arg0 will cause the value of %b to be incorrect.
  // We do need to use the next iteration value of %arg0 because it could be a
  // immediate arg of a load op.
  // load %arg0
  // %a = add %arg0, %c
  // yield %a
  //
  // We reroder instructions so %a and yield are actually before load. load
  // %arg0 should use the updated %arg0.
  IRMapping curMapping = nextMapping;
  for (Operation *op : orderedDeps)
    if (!validLoads.contains(op->getResult(0))) {
      if (immediateOpStages[op].contains(numStages - 2))
        // A post load op that provides values for numStage - 2
        curMapping.map(forOp.getInductionVar(), curIV);
      else
        curMapping.map(forOp.getInductionVar(), nextIV);
      Operation *nextOp;
      if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
        auto newMask =
            getLoadMask(loadOp, curMapping.lookupOrDefault(loadOp.getMask()),
                        nextLoopCond, builder);
        nextOp = builder.create<triton::LoadOp>(
            loadOp.getLoc(), loadOp.getResult().getType(),
            curMapping.lookupOrDefault(loadOp.getPtr()), newMask,
            curMapping.lookupOrDefault(loadOp.getOther()),
            loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
            loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
#ifdef USE_MACA
        addNamedAttrs(nextOp, op->getAttrDictionary());
#else
        addNamedAttrs(nextOp, op->getDiscardableAttrDictionary());
#endif
        curMapping.map(loadOp.getResult(), nextOp->getResult(0));
        nextMapping.map(loadOp.getResult(), nextOp->getResult(0));
      } else {
        nextOp = builder.clone(*op, curMapping);
        for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults()))
          nextMapping.map(op->getResult(dstIdx), nextOp->getResult(dstIdx));
      }

      for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults()))
        setValueMappingYield(newForOp, op->getResult(dstIdx),
                             nextOp->getResult(dstIdx));
    }

  // loads -> async loads
  for (Operation *op : orderedDeps) {
    Operation *nextOp = nullptr;
    // Update loading mask
    if (validLoads.contains(op->getResult(0))) {
      auto loadOp = llvm::cast<triton::LoadOp>(op);
      auto mask = loadOp.getMask();
      auto newMask =
          getLoadMask(loadOp, nextMapping.lookupOrDefault(loadOp.getMask()),
                      nextLoopCond, builder);
      if (mask) {
        // If mask is defined outside the loop, don't update the map more than
        // once
        if (!(forOp.isDefinedOutsideOfLoop(mask) && nextMapping.contains(mask)))
          nextMapping.map(loadOp.getMask(), newMask);
        newMask = nextMapping.lookupOrDefault(mask);
      }

#ifdef USE_MACA
      auto srcTy = cast<RankedTensorType>(loadOp.getPtr().getType());
      auto srcEncoding = cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
      tt::MemDescType allocTy =
          cast<tt::MemDescType>(loadsBuffer[loadOp].getType());
      auto bufferShape = allocTy.getShape();
      auto resEncoding = cast<ttg::SharedEncodingAttr>(allocTy.getEncoding());
      if (loadGenMask[loadOp]) {
        // generate swizzle mask
        newMask =
            genSwiSubMask(newMask, loadOp, srcEncoding, resEncoding, builder,
                          {bufferShape[1], bufferShape[2]}, {0, 0}, numStages);
      }

      // get root ptr and accumulate offset for saddr optimization.
      auto mappedAddPtr = loadPtrMapping[loadOp];
      Value ptr, offset;
      if (addPtrIdxMapping.find(mappedAddPtr) != addPtrIdxMapping.end()) {
        ptr = newForOp.getRegionIterArgs()[addPtrIdxMapping[mappedAddPtr]];
        offset =
            newForOp.getRegionIterArgs()[addPtrIdxMapping[mappedAddPtr] + 1];
        auto loopOffset = nextMapping.lookupOrDefault(
            mappedAddPtr.getDefiningOp()->getOperand(1));
        offset = addInt32OrInt64(builder, op->getLoc(), offset, loopOffset);
        nextAddPtrArgs[mappedAddPtr].push_back(ptr);
        nextAddPtrArgs[mappedAddPtr].push_back(offset);
        auto ptrType = cast<RankedTensorType>(ptr.getType());
        ptr = builder.create<triton::AddPtrOp>(op->getLoc(), ptrType, ptr,
                                               offset);

      } else {
        ptr = nextMapping.lookupOrDefault(loadOp.getPtr());
      }

      Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
      tt::MemDescType subviewTy = tt::MemDescType::get(
          allocTy.getShape().drop_front(), allocTy.getElementType(),
          allocTy.getEncoding(), /*mutableMemory=*/true);
      SmallVector<Value> insertOffsets(allocTy.getRank(), zero);
      insertOffsets[0] = insertSliceIndex;
      Value subViewBuffer = builder.create<ttg::MemDescSubviewOp>(
          op->getLoc(), subviewTy,
          newForOp.getRegionIterArgs()[bufferIdx + nextBuffers.size()],
          insertOffsets, /*isConstantOffs*/ false);
      Value insertAsyncOp = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
          op->getLoc(), ptr, subViewBuffer, newMask,
          nextMapping.lookupOrDefault(loadOp.getOther()), loadOp.getCache(),
          loadOp.getEvict(), loadOp.getIsVolatile(), /*intrinsic*/ true);
      addNamedAttrsForCpAsync(insertAsyncOp.getDefiningOp(),
                              op->getAttrDictionary());
#else
      Value insertAsyncOp = builder.create<ttg::InsertSliceAsyncOp>(
          op->getLoc(), loadsBuffer[loadOp].getType(),
          nextMapping.lookupOrDefault(loadOp.getPtr()),
          newForOp.getRegionIterArgs()[bufferIdx + nextBuffers.size()],
          insertSliceIndex, newMask,
          nextMapping.lookupOrDefault(loadOp.getOther()), loadOp.getCache(),
          loadOp.getEvict(), loadOp.getIsVolatile(), /*axis*/ 0);
      builder.create<ttg::AsyncCommitGroupOp>(op->getLoc());
#endif
      // Extract slice
      SmallVector<Value> extractOffsets(allocTy.getRank(), zero);
      extractOffsets[0] = extractSliceIndex;
      nextOp = builder.create<ttg::MemDescSubviewOp>(
          op->getLoc(), subviewTy,
          newForOp.getRegionIterArgs()[bufferIdx + nextBuffers.size()],
          extractOffsets, /*isConstantOffs*/ false);
      nextBuffers.push_back(insertAsyncOp);
      extractSlices.push_back(nextOp->getResult(0));
      int sliceIdx = extractSlices.size() - 1;
      nextMapping.map(op->getResult(0),
                      newForOp.getRegionIterArgs()[loadIdx + sliceIdx]);

      // Update mapping of results
      for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults()))
        // If this is a loop-carried value, update the mapping for yield
        setValueMappingYield(newForOp, op->getResult(dstIdx),
                             nextOp->getResult(dstIdx));
    }
  }

  // Some values have not been used by any ops in the loop body
  for (BlockArgument arg : forOp.getRegionIterArgs())
    setValueMappingYield(newForOp, arg,
                         newForOp.getRegionIterArgs()[depArgsIdx[arg]]);

  // async.wait & extract_slice
#ifdef USE_MACA
  int gvmNumberPerOp = 0;
  for (Value loadOp : validLoads) {
    auto blockType = dyn_cast<RankedTensorType>(loadOp.getType());
    auto blockEnc =
        dyn_cast<triton::gpu::BlockedEncodingAttr>(blockType.getEncoding());
    gvmNumberPerOp += getGVMNumberPerOp(blockEnc, blockType);
  }
  // Operation *asyncWait =
  // builder.create<ttg::AsyncWaitOp>(validLoads.front().getLoc(),
  //                                                         gvmNumberPerOp *
  //                                                         (numStages - 3));
  builder.create<ttg::GVMArriveOp>(validLoads.front().getLoc(),
                                   gvmNumberPerOp * (numStages - 3));
  Operation *asyncWait =
      builder.create<ttg::BarrierSharedOp>(validLoads[0].getLoc());
  // generate dot && dot's deps
  int loadidx = loadIdx + validLoads.size();
  auto genOutOfLoopDeps =
      genDotDeps(builder, newForOp, validLoads, nextMapping, mapping,
                 loadsMapping, elemsMapping, dotsDepsOutOfLoop, nextDotOperands,
                 cvtStageBuffer, cvtMapping, Value(), true, loadidx);
  auto genDeps =
      genDotDeps(builder, newForOp, validLoads, nextMapping, mapping,
                 loadsMapping, elemsMapping, dotsDeps, nextDotOperands,
                 cvtStageBuffer, cvtMapping, Value(), true, loadidx);

  for (auto it = genOutOfLoopDeps.begin(); it != genOutOfLoopDeps.end(); ++it) {
    (*it)->moveBefore(newForOp);
  }
  // builder.create<ttg::IGLPOp>(
  //     validLoads[0].getLoc(), 2, -1, 1, -1, -1, 1, 1);

  for (auto it = nextBuffers.rbegin(); it != nextBuffers.rend(); ++it) {
    // move insert_slice_async after asyncWait
    it->getDefiningOp()->moveAfter(asyncWait);
  }

  if (dotVector.size()) {
    Operation *anchor = dotVector.back().getDefiningOp();
    for (auto it = extractSlices.rbegin(); it != extractSlices.rend(); ++it) {
      // move extract_slice after anchor
      it->getDefiningOp()->moveAfter(anchor);
    }
    for (auto it = cvtAndTransVector.rbegin(); it != cvtAndTransVector.rend();
         ++it) {
      // move cvt and trans after anchor
      it->getDefiningOp()->moveAfter(anchor);
    }
  }

  for (auto it = genDeps.rbegin(); it != genDeps.rend(); ++it) {
    (*it)->moveAfter(asyncWait);
  }
  // for (auto it = dotVector.rbegin(); it != dotVector.rend(); ++it) {
  //   // move dot after anchor
  //   it->getDefiningOp()->moveAfter(anchor);
  // }
  // anchor->erase();
#else
  Operation *asyncWait = builder.create<ttg::AsyncWaitOp>(
      validLoads[0].getLoc(), validLoads.size() * (numStages - 2));
  for (auto it = extractSlices.rbegin(); it != extractSlices.rend(); ++it) {
    // move extract_slice after asyncWait
    it->getDefiningOp()->moveAfter(asyncWait);
  }
#endif
  // Bump iteration count
  pipelineIterIdx = builder.create<arith::AddIOp>(
      nextIV.getLoc(), pipelineIterIdx,
      builder.create<arith::ConstantIntOp>(nextIV.getLoc(), 1, 32));
  loopIterIdx = builder.create<arith::AddIOp>(
      nextIV.getLoc(), loopIterIdx,
      builder.create<arith::ConstantIntOp>(nextIV.getLoc(), 1, 32));
}

void LoopPipeliner::finalizeYield(scf::ForOp newForOp, OpBuilder &builder) {
  SmallVector<Value> yieldValues;
  for (Value v : yieldOp->getOperands())
    yieldValues.push_back(mapping.lookup(v));
  for (auto loadOp : validLoads) {
    yieldValues.push_back(loadsBuffer[loadOp]);
  }
  for (Value nextSlice : extractSlices)
    yieldValues.push_back(nextSlice);

#ifdef USE_MACA
  for (auto loadOp : validLoads)
    yieldValues.push_back(nextDotOperands[loadOp]);

  for (Value addptr : validAddPtrs) {
    for (Value arg : nextAddPtrArgs[addptr]) {
      yieldValues.push_back(arg);
    }
  }
#endif

  for (size_t i = depArgsBeginIdx; i < ivIndex; ++i) {
    auto arg = newForOp.getRegionIterArgs()[i];
    assert(depArgsMapping.count(arg) && "Missing loop-carried value");
    yieldValues.push_back(depArgsMapping[arg]);
  }
  yieldValues.push_back(nextIV);
  yieldValues.push_back(pipelineIterIdx);
  yieldValues.push_back(loopIterIdx);

  builder.setInsertionPointToEnd(newForOp.getBody());
  builder.create<scf::YieldOp>(yieldOp->getLoc(), yieldValues);
}

scf::ForOp LoopPipeliner::createNewForOp() {
  OpBuilder builder(forOp);
  auto newLoopArgs = collectNewLoopArgs();
  auto newForOp = cloneForOp(newLoopArgs, builder);
  prefetchNextIteration(newForOp, builder);
  finalizeYield(newForOp, builder);

  return newForOp;
}

#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"

// ref: mlir/lib/Dialect/SCF/Transforms/LoopPipelining.cpp
struct TritonMETAXGPUPipelineAsyncBasePass
    : public TritonMETAXGPUPipelineAsyncNTBase<
          TritonMETAXGPUPipelineAsyncBasePass> {
  TritonMETAXGPUPipelineAsyncBasePass() = default;
  TritonMETAXGPUPipelineAsyncBasePass(int numStages, bool isFullStage) {
    this->numStages = numStages;
    this->isFullStage = isFullStage;
  }

  void runOnOperation() override {
    int numStages = this->numStages;
    bool isFullStage = this->isFullStage;
    if (numStages <= 1)
      return;

    // Pre-processing
    // we make sure element-wise ops are done *after* the conversion
    // to dot operands
    // we can achieve this with simple recursive pattern matching
    // MLIRContext *context = &getContext();
    // mlir::RewritePatternSet patterns(context);
    // patterns.add<MoveOpAfterLayoutConversion>(context);
    // auto didPreprocess =
    //     applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));

    // Do the pipelining
    getOperation()->walk([&](scf::ForOp forOp) -> void {
      LoopPipeliner pipeliner(forOp, numStages, isFullStage);

      if (pipeliner.initialize().failed()) {
        return;
      }

      pipeliner.emitPrologue();
      scf::ForOp newForOp = pipeliner.createNewForOp();
      pipeliner.emitEpilogue();

      // Replace the original loop
      for (unsigned i = 0; i < forOp->getNumResults(); ++i) {
        forOp->getResult(i).replaceAllUsesWith(newForOp->getResult(i));
      }
      forOp->erase();
    });
  }
};
} // anonymous namespace

std::unique_ptr<Pass>
mlir::createTritonMETAXGPUPipelineAsyncBasePass(int numStages,
                                                bool isFullStage) {
  return std::make_unique<TritonMETAXGPUPipelineAsyncBasePass>(numStages,
                                                               isFullStage);
}
