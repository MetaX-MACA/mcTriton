/*
 * 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
 */

#include "TritonMETAXGPUTransforms/MACACommon.h"
#include "TritonMETAXGPUTransforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

using namespace mlir;

namespace {

using mlir::triton::gpu::MACAMmaEncodingAttr;
namespace ttg = triton::gpu;
namespace tt = triton;

class Prefetcher {
  /// cache the ForOp we are working on
  scf::ForOp forOp;
  // TODO: add a hook to infer prefetchWidth
  int prefetchWidth = 32;
  // get intrinsic flag from src
  bool isIntrinsicA = false;
  bool isIntrinsicB = false;

  /// dots to be prefetched
  SetVector<triton::DotOp> dots;
  /// dot => dot operand
  DenseMap<Value, Value> dot2aMem;
  DenseMap<Value, Value> dot2bMem;
  DenseMap<Value, SmallVector<Value>> dot2aVals;
  DenseMap<Value, SmallVector<Value>> dot2bVals;
  /// operand => defining
  DenseMap<Value, Value> operand2headPrefetch;

  Value generatePrefetch(Value v, unsigned opIdx, Attribute dotEncoding,
                         OpBuilder &builder,
                         std::optional<int64_t> offsetK = std::nullopt,
                         std::optional<int64_t> shapeK = std::nullopt);

  // clone all ops between dotOp and cvtOp(shared->dotoperand)
  void cloneElementwiseOps(Value &bRem, const SmallVector<Value> &vals,
                           OpBuilder &builder);

public:
  Prefetcher() = delete;

  Prefetcher(scf::ForOp forOp) : forOp(forOp) {}

  LogicalResult initialize();

  // generate prefetchs
  void run();

  // remove all arrive & barrier
  void removeSyncInLoop();
};

void Prefetcher::cloneElementwiseOps(Value &ret, const SmallVector<Value> &vals,
                                     OpBuilder &builder) {
  IRMapping mapping;
  mapping.map(vals[0], ret);
  for (int i = 1; i < vals.size(); i++) {
    Value v = vals[i];
    Value curr = builder.clone(*v.getDefiningOp(), mapping)->getResult(0);
    auto retType = RankedTensorType::get(
        cast<tt::MemDescType>(ret.getType()).getShape(),
        cast<RankedTensorType>(curr.getType()).getElementType(),
        cast<RankedTensorType>(curr.getType()).getEncoding());
    curr.setType(retType);
    mapping.map(v, curr);
  }
  if (vals.size() > 1)
    ret = mapping.lookup(vals.back());
}

Value Prefetcher::generatePrefetch(Value v, unsigned opIdx,
                                   Attribute dotEncoding, OpBuilder &builder,
                                   std::optional<int64_t> offsetK,
                                   std::optional<int64_t> shapeK) {
  // opIdx: 0 => a, 1 => b
  auto type = cast<tt::MemDescType>(v.getType());
  SmallVector<int64_t> shape{type.getShape().begin(), type.getShape().end()};
  SmallVector<int64_t> offset{0, 0};
  Type elementType = type.getElementType();

  auto intAttr = [&](int64_t val) { return builder.getI64IntegerAttr(val); };

  // k => (prefetchWidth, k - prefetchWidth)
  int64_t kIdx = opIdx == 0 ? 1 : 0;

  if (shapeK)
    shape[kIdx] = *shapeK;
  if (offsetK)
    offset[kIdx] = *offsetK;
  bool isIntrinsic = opIdx == 0 ? isIntrinsicA : isIntrinsicB;

  SmallVector<Value> offsetsVal;
  for (int64_t off : offset)
    offsetsVal.push_back(
        builder.create<arith::ConstantIntOp>(v.getLoc(), off, 32));

  Value prefetchSlice = builder.create<triton::gpu::MemDescSubviewOp>(
      v.getLoc(), tt::MemDescType::get(shape, elementType, type.getEncoding()),
      v, offsetsVal);
  return prefetchSlice;
}

void Prefetcher::removeSyncInLoop() {
  Block *loop = forOp.getBody();
  SmallVector<Operation *> erase_ops;
  for (Operation &op : *loop) {
    if (auto arriveOp = dyn_cast<triton::gpu::GVMArriveOp>(op)) {
      erase_ops.push_back(&op);
    }
    if (auto barrierOp = dyn_cast<gpu::BarrierOp>(op)) {
      erase_ops.push_back(&op);
    }
    if (auto barrierOp = dyn_cast<triton::gpu::BarrierSharedOp>(op)) {
      erase_ops.push_back(&op);
    }
    if (auto barrierOp = dyn_cast<triton::gpu::SchedBoundOp>(op)) {
      erase_ops.push_back(&op);
    }
  }
  for (auto op : erase_ops) {
    op->erase();
  }
}

LogicalResult Prefetcher::initialize() {
  Block *loop = forOp.getBody();

  SmallVector<triton::DotOp> dotsInFor;
  for (Operation &op : *loop)
    if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
      auto tensorType = dyn_cast<RankedTensorType>(dotOp.getResult().getType());
      if (auto mmaLayout = dyn_cast<mlir::triton::gpu::MACAMmaEncodingAttr>(
              tensorType.getEncoding())) {
        if (mmaLayout.getVersionMajor() >= 2)
          dotsInFor.push_back(dotOp);
      }
    }

  if (dotsInFor.empty())
    return failure();

  // TODO: segfault (original for still has uses)
  // when used in flash attention that has 2 dots in the loop
  if (dotsInFor.size() > 1)
    return failure();

  // returns source of cvt
  auto getPrefetchSrc = [&](Value v, int opIdx) -> SmallVector<Value> {
    // walk back to conversion
    Operation *op = v.getDefiningOp();
    bool foundConvertFromShared = false;
    SmallVector<Value> rets;
    rets.push_back(op->getResult(0));
    while (op) {
      if (op->getNumOperands() != 1)
        break;
      if (!op->getResult(0).hasOneUse())
        break;
      rets.push_back(op->getOperand(0));
      if (auto cvt = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
        auto cvtTy = op->getOperand(0).getType();
        if (auto memTy = dyn_cast<tt::MemDescType>(cvtTy)) {
          if (mlir::isa<ttg::SharedEncodingAttr>(memTy.getEncoding())) {
            foundConvertFromShared = true;
            // if (opIdx == 0) {
            //   isIntrinsicA = cvt.getIntrinsic();
            // } else {
            //   isIntrinsicB = cvt.getIntrinsic();
            // }
            break;
          }
        }
      }
      op = op->getOperand(0).getDefiningOp();
    }
    std::reverse(rets.begin(), rets.end());

    if (foundConvertFromShared)
      return rets;
    return {};
  };

  for (triton::DotOp dot : dotsInFor) {
    auto aType = cast<RankedTensorType>(dot.getA().getType());
    auto bType = cast<RankedTensorType>(dot.getB().getType());
    auto aEnc = cast<ttg::DotOperandEncodingAttr>(aType.getEncoding());
    auto bEnc = cast<ttg::DotOperandEncodingAttr>(bType.getEncoding());
    int kSize = aType.getShape()[1];
    // TODO(MACA): currently only support prefetch 2 slice.
    prefetchWidth = kSize / 2;
    auto mmaLayout = dyn_cast<ttg::MACAMmaEncodingAttr>(aEnc.getParent());
    if (!mmaLayout)
      return failure();

    auto elemsPerThread = mmaLayout.getElementsMNK();
    auto aVals = getPrefetchSrc(dot.getA(), 0);
    auto bVals = getPrefetchSrc(dot.getB(), 1);
    auto minPrefetchSize = elemsPerThread[2] * 4; // 4 threads along K.

    if (prefetchWidth < minPrefetchSize)
      return failure();

    if (aVals.size() && bVals.size()) {
      dots.insert(dot);
      Value aSmem = aVals.front();
      Value bSmem = bVals.front();
      dot2aMem[dot] = aSmem;
      dot2bMem[dot] = bSmem;
      dot2aVals[dot] = aVals;
      dot2bVals[dot] = bVals;
    } else {
      return failure();
    }
  }

  return success();
}

void Prefetcher::run() {
  OpBuilder builder(forOp);
  for (triton::DotOp dot : dots) {
    builder.setInsertionPoint(dot);
    Attribute dotEncoding = cast<RankedTensorType>(dot.getType()).getEncoding();
    // create first dot
    //
    // 1st dot
    // dot <- insertion point
    //
    Operation *firstDot = builder.clone(*dot);
    // create barrier
    //
    // barrier
    // 1st dot <- insertion point
    // dot
    //
    builder.setInsertionPoint(firstDot);
    // builder.create<gpu::BarrierOp>(firstDot->getLoc());
    builder.create<triton::gpu::BarrierSharedOp>(firstDot->getLoc());
    // first prefetch
    //
    // barrier
    // extract slice a
    // cvt a
    // extract slice b
    // cvt b
    // 1st dot <- insertion point
    // dot
    //
    int64_t kShape = prefetchWidth;
    int64_t kOff = 0;
    Value aRem =
        generatePrefetch(dot2aMem[dot], 0, dotEncoding, builder, kOff, kShape);
    cloneElementwiseOps(aRem, dot2aVals[dot], builder);
    Value bRem =
        generatePrefetch(dot2bMem[dot], 1, dotEncoding, builder, kOff, kShape);
    cloneElementwiseOps(bRem, dot2bVals[dot], builder);

    firstDot->setOperand(0, aRem);
    firstDot->setOperand(1, bRem);
    firstDot->setOperand(2, dot->getOperand(2));

    // create second dot
    //
    // barrier
    // extract slice a
    // cvt a
    // extract slice b
    // cvt b
    // 1st dot
    // 2nd dot
    // dot <- insertion point
    //
    kOff = prefetchWidth;
    builder.setInsertionPoint(dot);
    Operation *secondDot = builder.clone(*dot);
    // second prefetch and arrive gvm.
    //
    // barrier
    // extract slice a
    // cvt a
    // extract slice b
    // cvt b
    // 1st dot
    // extract slice a
    // cvt a
    // extract slice b
    // cvt b
    // arrive
    // 2nd dot <- insertion point
    // dot
    //
    builder.setInsertionPoint(secondDot);
    aRem =
        generatePrefetch(dot2aMem[dot], 0, dotEncoding, builder, kOff, kShape);
    cloneElementwiseOps(aRem, dot2aVals[dot], builder);
    bRem =
        generatePrefetch(dot2bMem[dot], 1, dotEncoding, builder, kOff, kShape);
    cloneElementwiseOps(bRem, dot2bVals[dot], builder);
    secondDot->setOperand(0, aRem);
    secondDot->setOperand(1, bRem);
    secondDot->setOperand(2, firstDot->getResult(0));
    builder.create<triton::gpu::SchedBoundOp>(secondDot->getLoc());

    // erase old dot
    dot.replaceAllUsesWith(secondDot->getResult(0));
    dot->erase();
  }
}

#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"

struct TritonMETAXGPUPrefetchMACAPass
    : public TritonMETAXGPUPrefetchMACABase<TritonMETAXGPUPrefetchMACAPass> {
  TritonMETAXGPUPrefetchMACAPass() = default;
  void runOnOperation() override {
    getOperation()->walk([&](scf::ForOp forOp) {
      Prefetcher prefetcher(forOp);

      if (prefetcher.initialize().failed())
        return;

      prefetcher.removeSyncInLoop();
      prefetcher.run();
    });
  }
};

} // anonymous namespace

std::unique_ptr<Pass> mlir::createTritonMETAXGPUPrefetchMACAPass() {
  return std::make_unique<TritonMETAXGPUPrefetchMACAPass>();
}
