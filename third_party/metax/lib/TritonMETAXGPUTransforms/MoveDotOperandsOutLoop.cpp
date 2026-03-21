
#include "TritonMETAXGPUTransforms/Passes.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <memory>
using namespace mlir;
using triton::DotOp;
using triton::LoadOp;
using triton::MemDescType;
using triton::TransOp;
using triton::gpu::ConvertLayoutOp;
using triton::gpu::DotOperandEncodingAttr;
using triton::gpu::MACAMmaEncodingAttr;
using triton::gpu::SharedEncodingAttr;
#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"

class FixupLoop : public mlir::RewritePattern {

public:
  explicit FixupLoop(mlir::MLIRContext *context)
      : mlir::RewritePattern(scf::ForOp::getOperationName(), 2, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto forOp = cast<scf::ForOp>(op);

    // Rewrite init argument
    SmallVector<Value, 4> newInitArgs = forOp.getInitArgs();
    bool shouldRematerialize = false;
    for (size_t i = 0; i < newInitArgs.size(); i++) {
      if (newInitArgs[i].getType() != forOp.getRegionIterArgs()[i].getType() ||
          newInitArgs[i].getType() != forOp.getResultTypes()[i]) {
        shouldRematerialize = true;
        break;
      }
    }
    if (!shouldRematerialize)
      return failure();

    scf::ForOp newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInitArgs);
    newForOp->moveBefore(forOp);
    rewriter.setInsertionPointToStart(newForOp.getBody());
    IRMapping mapping;
    for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
      mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

    for (Operation &op : forOp.getBody()->getOperations()) {
      rewriter.clone(op, mapping);
    }
    rewriter.replaceOp(forOp, newForOp.getResults());
    return success();
  }
};

LogicalResult fixupLoops(ModuleOp mod) {
  auto *ctx = mod.getContext();
  mlir::RewritePatternSet patterns(ctx);
  patterns.add<FixupLoop>(ctx);
  if (applyPatternsAndFoldGreedily(mod, std::move(patterns)).failed())
    return failure();
  return success();
}

class TritonMETAXGPUMoveDotOperandsOutLoopPass
    : public TritonMETAXGPUMoveDotOperandsOutLoopBase<
          TritonMETAXGPUMoveDotOperandsOutLoopPass> {
public:
  TritonMETAXGPUMoveDotOperandsOutLoopPass() = default;
  static bool isValueInForOpDomain(Value value, scf::ForOp &forOp) {
    if (!value) {
      return true;
    }
    Operation *defOp = value.getDefiningOp();
    if (defOp && forOp.getRegion().isAncestor(defOp->getParentRegion())) {
      return true;
    }
    return false;
  }
  static bool isOpInForOpDomain(Operation *op, scf::ForOp &forOp) {
    if (!op || !op->getParentRegion()) {
      return true;
    }
    if (forOp.getRegion().isAncestor(op->getParentRegion())) {
      return true;
    }
    return false;
  }
  bool findIntersection(const SetVector<Operation *> &S1,
                        const SetVector<Operation *> &S2) {
    for (auto op : S1) {
      if (S2.count(op)) {
        return true;
      }
    }
    return false;
  }
  static bool canMoveOutForLoop(scf::ForOp &forOp, triton::DotOp &dotOp) {
    // just support dot operand a just now
    auto a = dotOp.getA();
    // we only support the following pattern
    // TODO: support more flexible check pattern
    // %3
    // scf.for
    //   ...
    //   %10 = triton_gpu.convert_layout %3 blocked -> dot_op
    //   %11 = tt.dot %10 ...
    //   ...
    Operation *aDefOp = a.getDefiningOp();
    auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(aDefOp);
    if (!cvtOp)
      return false;
    auto cvtOpSrc = cvtOp.getSrc();
    if (cvtOpSrc && !isValueInForOpDomain(cvtOpSrc, forOp)) {
      return true;
    }
    return false;
  }
  static bool checkLoadOp(SetVector<Operation *> &slice) {
    // just now we assume loadop is at the top of the slice if exist loadop
    // and if exit loadop we assume only one
    int loadOpCount = 0;
    for (auto op : slice) {
      auto loadOp = dyn_cast<triton::LoadOp>(op);
      if (loadOp) {
        loadOpCount++;
      }
    }
    if (loadOpCount > 1)
      return false;
    if (loadOpCount == 0)
      return true;
    return isa<triton::LoadOp>(slice.front());
  }

  void getSlice(Operation *defOp, SetVector<Operation *> &slice,
                scf::ForOp &forOp) {
    if (!defOp)
      return;
    // collect cvtop, transop, loadop, truncfop
    // we assume the last op outside the forop is loadop or trcnfop
    // NOTICE: we only collect op whith only one operand, if add op with more
    // then one operand
    //         we maybe extend the getSlice func
    auto backwardFilter = [&](Operation *op) {
      auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op);
      auto transOp = dyn_cast<triton::TransOp>(op);
      auto loadOp = dyn_cast<triton::LoadOp>(op);
      auto truncfOp = dyn_cast<mlir::arith::TruncFOp>(op);
      auto localLoadOp = dyn_cast<triton::gpu::LocalLoadOp>(op);
      auto localAllocOp = dyn_cast<triton::gpu::LocalAllocOp>(op);
      if (cvtOp != nullptr || transOp != nullptr || loadOp != nullptr ||
          truncfOp != nullptr || localLoadOp != nullptr ||
          localAllocOp != nullptr)
        return true;
      return false;
    };
    BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    opt.filter = backwardFilter;
    SetVector<Operation *> backwardSlice;
    getBackwardSlice(defOp, &backwardSlice, opt);
    // check can move out for loop
    // if the collected op with more than one operand, we should check all the
    // operand
    if (backwardSlice.empty())
      return;
    backwardSlice = mlir::topologicalSort(backwardSlice);
    auto topOp = backwardSlice.front();
    if (!topOp)
      return;
    if (checkLoadOp(backwardSlice) && !isOpInForOpDomain(topOp, forOp)) {
      slice.insert(backwardSlice.begin(), backwardSlice.end());
      slice.insert(defOp);
    }
    return;
  }

  bool collectLegalOperations(triton::DotOp dotOp, scf::ForOp &forOp,
                              SetVector<Operation *> &aSlices,
                              SetVector<Operation *> &bSlices) {
    // a
    auto a = dotOp.getA();
    Operation *aDefOp = a.getDefiningOp();
    getSlice(aDefOp, aSlices, forOp);
    // b
    auto b = dotOp.getB();
    Operation *bDefOp = b.getDefiningOp();
    getSlice(bDefOp, bSlices, forOp);
    // if aSlices interset with bSlices we do nothing
    if (findIntersection(aSlices, bSlices)) {
      aSlices.clear();
      bSlices.clear();
      return false;
    }
    return !aSlices.empty() || !bSlices.empty();
  }

  bool needBarrier(Operation *op) {
    if (isa<triton::gpu::LocalLoadOp>(op) ||
        isa<triton::gpu::ConvertLayoutOp>(op)) {
      // local_load shared -> dotop or cvt shared -> dotop
      Value src = nullptr;
      Value res = nullptr;
      if (auto localLoadOp = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
        src = localLoadOp.getSrc();
        res = localLoadOp.getResult();
      } else if (auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
        src = cvtOp.getSrc();
        res = cvtOp.getResult();
      } else {
        return false;
      }
      if (!src || !res)
        return false;
      auto tensorTypeSrc = dyn_cast<triton::MemDescType>(src.getType());
      auto tensorTypeRes = dyn_cast<RankedTensorType>(res.getType());
      if (!tensorTypeSrc || !tensorTypeRes)
        return false;
      auto sharedEnc = dyn_cast<triton::gpu::SharedEncodingAttr>(
          tensorTypeSrc.getEncoding());
      if (!sharedEnc)
        return false;
      auto dotOpEnc = dyn_cast<triton::gpu::DotOperandEncodingAttr>(
          tensorTypeRes.getEncoding());
      if (!dotOpEnc)
        return false;
      return true;
    }
    return false;
  }

  void moveOpOutLoop(MLIRContext *context, SetVector<Operation *> &slice,
                     IRMapping &mapping) {
    if (slice.empty())
      return;
    OpBuilder builder(context);
    OpBuilder::InsertionGuard g(builder);
    for (auto op : slice) {
      if (auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
        auto src = cvtOp.getSrc();
        Operation *srcDefOp = src.getDefiningOp();
        builder.setInsertionPointAfter(srcDefOp);
      } else if (auto localLoadOp = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
        auto src = localLoadOp.getSrc();
        Operation *srcDefOp = src.getDefiningOp();
        builder.setInsertionPointAfter(srcDefOp);
      } else if (auto localAllocOp = dyn_cast<triton::gpu::LocalAllocOp>(op)) {
        auto src = localAllocOp.getSrc();
        Operation *srcDefOp = src.getDefiningOp();
        builder.setInsertionPointAfter(srcDefOp);
      } else if (auto transOp = dyn_cast<triton::TransOp>(op)) {
        auto src = transOp.getSrc();
        Operation *srcDefOp = src.getDefiningOp();
        builder.setInsertionPointAfter(srcDefOp);
      } else if (auto truncfOp = dyn_cast<mlir::arith::TruncFOp>(op)) {
        // maybe get the following pattern
        // %3 = load
        // scf.for ...
        //    %7 = truncf %3
        //    %9 = convert %7
        //    %10 = dot %9 ...
        auto src = truncfOp.getIn();
        Operation *srcDefOp = src.getDefiningOp();
        builder.setInsertionPointAfter(srcDefOp);
      } else if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
        // do nothing, maybe need later, we assume loadop allocate the top of
        // the graph if exist
        continue;
      } else {
        assert(false && "Get unexpected op when move op out of loop!");
      }
      // the newOp result type is the same as oldOp
      auto *newOp = cloneWithInferType(builder, op, mapping);
      for (auto [result, newResult] :
           llvm::zip(op->getResults(), newOp->getResults())) {
        mapping.map(result, newResult);
        result.replaceAllUsesWith(newResult);
      }
      if (needBarrier(op)) {
        // we need barrierShared after lds wait for all threads finish the lds
        // operation
        Operation *barrier =
            builder.create<triton::gpu::BarrierSharedOp>(newOp->getLoc());
      }
    }
    for (Operation *op : llvm::reverse(slice)) {
      // if loadop we do nothing
      if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
        continue;
      }
      op->erase();
    }
  }
  void moveDotOperandOutLoop(triton::DotOp &dotOp, MLIRContext *context) {
    auto parentOp = dotOp->getParentOp();
    if (!parentOp)
      return;
    auto forOp = dyn_cast<scf::ForOp>(parentOp);
    SetVector<Operation *> aSlices;
    SetVector<Operation *> bSlices;
    if (forOp && collectLegalOperations(dotOp, forOp, aSlices, bSlices)) {
      // check dot operand can move out for loop
      IRMapping mapping;
      moveOpOutLoop(context, aSlices, mapping);
      moveOpOutLoop(context, bSlices, mapping);
      aSlices.clear();
      bSlices.clear();
    }
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    m->walk(
        [&](triton::DotOp dotOp) { moveDotOperandOutLoop(dotOp, context); });
    // the op out of loop maybe used in the forop args, replaceAllUsesWith only
    // update the initArgs in forOp call fixupLoops update the args in forOp if
    // necessary
    if (fixupLoops(m).failed())
      signalPassFailure();
  }
};
std::unique_ptr<Pass> mlir::createTritonMETAXGPUMoveDotOperandsOutLoopPass() {
  return std::make_unique<TritonMETAXGPUMoveDotOperandsOutLoopPass>();
}
