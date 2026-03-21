#include "TritonMETAXGPUTransforms/MACACommon.h"
#include "TritonMETAXGPUTransforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include <memory>

using namespace mlir;
using triton::DotOp;
using triton::MemDescType;
using triton::gpu::BlockedEncodingAttr;
using triton::gpu::LocalAllocOp;
using triton::gpu::MACAMmaEncodingAttr;
using triton::gpu::SharedEncodingAttr;
#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"

// check if two cvtLayoutOps coming from same src are equal and can be merged
// if can be merged, rewrite the shared encoding and change the forward Ops
// encodings
class MergeSharedLayout : public mlir::RewritePattern {

public:
  MergeSharedLayout(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::LocalAllocOp::getOperationName(), 1,
                             context) {}
  // Before Pass:
  //     %0 = load
  //     %1 = local_alloc %0 blocked -> shared1
  //     %2 = local_alloc %0 blocked -> shared2
  // Suppose sharedLayout2 can be merged to sharedLayout1
  // After Pass:
  //     %0 = load
  //     %1 = local_alloc %0 blocked -> shared1
  //     %2 = local_alloc %0 blocked -> shared1

  // If there is trans after local_alloc:
  // Before Pass:
  //     %0 = load
  //     %1 = local_alloc %0 blocked (order[1,0]) -> shared1 (order[1,0])
  //     %2 = local_alloc %0 blocked (order[1,0]) -> shared2 (order[1,0])
  //     %3 = trans %2 shared2 (order[1,0]) -> shared3 (order[0,1])
  // After Pass:
  //     %0 = load
  //     %1 = local_alloc %0 blocked (order[1,0]) -> shared1 (order[1,0])
  //     %2 = local_alloc %0 blocked (order[1,0]) -> shared1 (order[1,0])
  //     %3 = trans %2 shared1 (order[1,0]) -> shared4 (order[0,1])
  // sharedLayout1 and 4 only differ in order, same vec/perphase/maxphase
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto curCvtOp = dyn_cast<triton::gpu::LocalAllocOp>(op);
    if (!curCvtOp)
      return failure();
    if (!curCvtOp.getSrc())
      return failure();
    auto curSrcEncoding =
        cast<RankedTensorType>(curCvtOp.getSrc().getType()).getEncoding();
    auto curDstEncoding =
        cast<MemDescType>(curCvtOp.getResult().getType()).getEncoding();

    // Only deal with Blocked to Shared
    if (!isa<triton::gpu::BlockedEncodingAttr>(curSrcEncoding) ||
        !isa<triton::gpu::SharedEncodingAttr>(curDstEncoding))
      return failure();

    bool mergeMatch = false;
    LocalAllocOp newOp;

    auto curSharedEncoding =
        cast<triton::gpu::SharedEncodingAttr>(curDstEncoding);
    ArrayRef<int64_t> curShape =
        cast<MemDescType>(curCvtOp.getResult().getType()).getShape();
    Value curSrc = curCvtOp.getSrc();
    // traverse cvtLayoutOps from same blocked src
    for (auto *user : curSrc.getUsers()) {
      auto targetCvtOp = dyn_cast<triton::gpu::LocalAllocOp>(user);
      if (!targetCvtOp || targetCvtOp == curCvtOp)
        continue;

      auto tarType = cast<MemDescType>(targetCvtOp.getResult().getType());
      auto tarSharedEncoding =
          cast<triton::gpu::SharedEncodingAttr>(tarType.getEncoding());
      ArrayRef<int64_t> tarShape = tarType.getShape();

      // check if current cvtOp's sharedLayout can merge to target CvtOp's
      // sharedLayout
      mergeMatch = isNeedMerge(curSharedEncoding, tarSharedEncoding, curShape);
      mergeMatch = mergeMatch && (tarShape == curShape);
      if (mergeMatch) {
        newOp = targetCvtOp;
        break;
      }
    }

    if (!mergeMatch)
      return failure();
    // TODO: support cvtOp having more than one user?
    if (!curCvtOp.getResult().hasOneUse())
      return failure();
    auto newType = cast<MemDescType>(newOp.getResult().getType());
    auto transOp =
        dyn_cast<triton::TransOp>(*curCvtOp.getResult().getUsers().begin());
    // change dstSharedEncoding to tarSharedEncoding
    // if there is transOp after curCvtOp, also replace the transOp
    if (transOp) {
      // make sure only one trans after cvtOp
      for (auto *opAfterTransOp : transOp.getResult().getUsers()) {
        if (dyn_cast<triton::TransOp>(opAfterTransOp))
          return failure();
      }
      auto resOrder = transOp.getOrder();
      auto tmpValue = rewriter.create<triton::gpu::LocalAllocOp>(
          curCvtOp.getLoc(), newType, curSrc);
      rewriter.replaceOpWithNewOp<triton::TransOp>(transOp, tmpValue, resOrder);
    } else {
      rewriter.replaceOpWithNewOp<triton::gpu::LocalAllocOp>(curCvtOp, newType,
                                                             curSrc);
    }
    return success();
  }
};

class TritonMETAXGPUMergeEqualSharedLayoutPass
    : public TritonMETAXGPUMergeEqualSharedLayoutBase<
          TritonMETAXGPUMergeEqualSharedLayoutPass> {
public:
  TritonMETAXGPUMergeEqualSharedLayoutPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<::MergeSharedLayout>(context);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonMETAXGPUMergeEqualSharedLayoutPass() {
  return std::make_unique<TritonMETAXGPUMergeEqualSharedLayoutPass>();
}