/*
 * 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
 */
#include "TritonMETAXGPUTransforms/MACACommon.h"
#include "TritonMETAXGPUTransforms/Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cmath>
#include <memory>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {
using triton::DotOp;
using triton::gpu::BlockedEncodingAttr;
using triton::gpu::ConvertLayoutOp;
using triton::gpu::DotOperandEncodingAttr;
using triton::gpu::MACAMmaEncodingAttr;
using triton::gpu::SharedEncodingAttr;
using triton::gpu::SliceEncodingAttr;

int computeCapabilityToMMAVersion(int computeCapability) {
  if (computeCapability < 70) {
    return 0;
  } else if (computeCapability < 80) {
    return 1;
  } else if (computeCapability < 90) {
    return 2;
  } else if (computeCapability < 100) {
    // FIXME: temporarily add this to pass unis tests
    return 3;
  } else {
    assert(false && "computeCapability > 100 not supported");
    return 3;
  }
}

SmallVector<int64_t, 2> mmaVersionToShapePerWarp(int version) {
  if (version == 1)
    return {16, 16};
  else if (version == 2)
    return {16, 8};
  else if (version == 3)
    return {16, 16};
  else {
    assert(false && "version not supported");
    return {0, 0};
  }
}

SmallVector<unsigned, 2> warpsPerTileV2(triton::DotOp dotOp,
                                        const ArrayRef<int64_t> shape,
                                        int numWarps) {
  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  auto slices = mlir::getSlice(dotOp, {filter});
  for (Operation *op : slices)
    if (isa<triton::DotOp>(op) && (op != dotOp))
      return {(unsigned)numWarps, 1};

  SmallVector<unsigned, 2> ret = {1, 1};
  SmallVector<int64_t, 2> shapePerWarp = {16, 8};
  bool changed = false;
  do {
    changed = false;
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (shape[0] / shapePerWarp[0] / ret[0] >=
        shape[1] / (shapePerWarp[1] * 2) / ret[1]) {
      if (ret[0] < shape[0] / shapePerWarp[0]) {
        ret[0] *= 2;
      } else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);
  return ret;
}

SmallVector<unsigned, 2> warpsPerTileMACA(triton::DotOp dotOp,
                                          const ArrayRef<int64_t> shape,
                                          int numWarps) {
  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  auto slices = mlir::getSlice(dotOp, {filter});
  for (Operation *op : slices)
    if (isa<triton::DotOp>(op) && (op != dotOp))
      return {(unsigned)numWarps, 1};

  SmallVector<unsigned, 2> ret = {1, 1};
  SmallVector<int64_t, 2> shapePerWarp = {16, 16};
  bool changed = false;
  do {
    changed = false;
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (shape[0] / shapePerWarp[0] / ret[0] >=
        shape[1] / (shapePerWarp[1] * 2) / ret[1]) {
      if (ret[0] < shape[0] / shapePerWarp[0]) {
        ret[0] *= 2;
      } else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);
  return ret;
}

class BlockedToMMA : public mlir::RewritePattern {
  int computeCapability;
  mutable int mmaV1Counter{}; // used to generate ID for MMAv1 encoding
  int dotCnt;
  int numStages;
  int disablePrefetch;
  int storeCoalesce;

public:
  BlockedToMMA(mlir::MLIRContext *context, int computeCapability, int dotCnt,
               int numStages, bool disablePrefetch, bool storeCoalesce)
      : mlir::RewritePattern(triton::DotOp::getOperationName(), 2, context),
        computeCapability(computeCapability), dotCnt(dotCnt),
        numStages(numStages), disablePrefetch(disablePrefetch),
        storeCoalesce(storeCoalesce) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (computeCapability < 70)
      return failure();
    auto dotOp = cast<triton::DotOp>(op);
    // TODO: Check data-types and SM compatibility
    auto oldRetType = cast<RankedTensorType>(dotOp.getResult().getType());
    if (!oldRetType.getEncoding() ||
        isa<triton::gpu::MACAMmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    // for FMA, should retain the blocked layout.
    int versionMajor = computeCapabilityToMMAVersion(computeCapability);
    if (!supportMMA(dotOp, versionMajor, computeCapability % 10))
      return failure();

    // get MMA encoding for the given number of warps
    auto retShape = oldRetType.getShape();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);

    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());

    // enable opt maca layout or not
    auto aTensorTy = cast<RankedTensorType>(a.getType());
    auto bTensorTy = cast<RankedTensorType>(b.getType());
    int m = aTensorTy.getShape()[0];
    int k = aTensorTy.getShape()[1];
    int n = bTensorTy.getShape()[1];
    auto elementTy = aTensorTy.getElementType();
    bool enableTf32 = dotOp.getInputPrecision() == tt::InputPrecision::TF32;

    auto parentOp = dotOp->getParentOp();
    // enable MACA's optimized MMA layout, currently must satisfy:
    // a) only one dot op in the compiled kernel
    // b) num_stage == 2
    // c) dot op contained in a for loop
    // d) not enable TRITON_DISABLE_MACA_OPT_MMA = 1
    bool enableOptMMA =
        this->dotCnt == 1 && (4 >= this->numStages) ||
        (this->numStages >= 2) && isa<scf::ForOp>(parentOp) &&
            std::getenv("TRITON_DISABLE_MACA_OPT_MMA") == nullptr;

    triton::gpu::MACAMmaEncodingAttr mmaEnc;
    SmallVector<unsigned> aorder, border;
    auto elemATy = oldAType.getElementType();
    auto elemBTy = oldBType.getElementType();
    bool enableALdsTrans, enableBLdsTrans;
    aorder = triton::gpu::getOrder(oldAType.getEncoding());
    border = triton::gpu::getOrder(oldBType.getEncoding());
    if (versionMajor == 2) {
      auto elemsPerThread =
          getDefaultElemsPerThread(elementTy, enableTf32, computeCapability);
      auto warpsPerTile = warpsPerTileMACA(dotOp, retShape, numWarps);
      int versionMajor_ = versionMajor;
      int versionMinor_ = computeCapability % 10;
      if (enableOptMMA) {
        Operation *aOp = a.getDefiningOp();
        Operation *bOp = b.getDefiningOp();
        auto aCvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(aOp);
        auto bCvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(bOp);
        if (aCvtOp && bCvtOp) {
          // when load row major and trans to col major
          //
          // Before AccelerateMatmul Pass:
          //
          //   cvt_pre block(order=(1,0)) -> shared(order=(0,1))
          //   trans shared(order=(0,1)) -> shared1(order=(1,0))
          //   cvt shared1(order=(1,0)) -> dot(order=(1,0))
          //
          // however order=(1,0) is not final order, but (0, 1)
          //
          // After OptimizeDotOperand Pass:
          //
          //   cvt_pre block(order=(1,0)) -> shared(order=(1,0))
          //   trans shared(order=(1,0)) -> shared1(order=(0,1))
          //   cvt shared1(order=(0,1)) -> dot(order=(0,1))
          //
          //
          // So in AccelerateMatmul Pass we reverse layout order
          // of cvt_pre input.
          aorder = getOrder(aCvtOp);
          border = getOrder(bCvtOp);
          if (!(aorder.empty() || border.empty())) {
            llvm::SmallVector<int, 4> tile({m, n, k, numWarps});
            int version = -1;
            bool isOpt =
                updateLayout(elemsPerThread, warpsPerTile, tile, version,
                             numWarps, enableTf32, aTensorTy.getElementType(),
                             aorder, border, this->disablePrefetch, false,
                             this->storeCoalesce, computeCapability);
            if (isOpt) {
              mod->setAttr(
                  "use.opt.maca.mma",
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(mod.getContext(), 32), 1));
            }
          }
        }
      }
      enableALdsTrans = getIfLdsTrans(elemsPerThread, versionMajor_,
                                      versionMinor_, aorder, true, elemATy);
      enableBLdsTrans = getIfLdsTrans(elemsPerThread, versionMajor_,
                                      versionMinor_, border, false, elemBTy);
      auto elemStride = triton::gpu::getLdsTransVec(elemATy);
      SmallVector<unsigned> elementsStride = {1, 1};
      if (enableALdsTrans)
        elementsStride[0] = elemStride;
      if (enableBLdsTrans)
        elementsStride[1] = elemStride;
      int colMajor = 0;
      // set single dot colMajor=1 for debug
      if (std::getenv("TRITON_SET_COLMAJOR_MANUAL"))
        colMajor = 1;
      mmaEnc = triton::gpu::MACAMmaEncodingAttr::get(
          oldRetType.getContext(), versionMajor_, versionMinor_, warpsPerTile,
          elemsPerThread, colMajor, enableALdsTrans, enableBLdsTrans,
          elementsStride);
    } else {
      llvm_unreachable("Mma layout only supports versionMajor in {2}");
    }
    auto newRetType =
        RankedTensorType::get(retShape, oldRetType.getElementType(), mmaEnc);

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = rewriter.create<triton::gpu::ConvertLayoutOp>(
        oldAcc.getLoc(), newRetType, oldAcc);

    auto oldAOrder =
        cast<triton::gpu::BlockedEncodingAttr>(
            cast<triton::gpu::DotOperandEncodingAttr>(oldAType.getEncoding())
                .getParent())
            .getOrder();
    auto oldBOrder =
        cast<triton::gpu::BlockedEncodingAttr>(
            cast<triton::gpu::DotOperandEncodingAttr>(oldBType.getEncoding())
                .getParent())
            .getOrder();

    auto newAEncoding = triton::gpu::DotOperandEncodingAttr::get(
        oldAType.getContext(), 0, newRetType.getEncoding(),
        oldAType.getElementType());
    auto newBEncoding = triton::gpu::DotOperandEncodingAttr::get(
        oldBType.getContext(), 1, newRetType.getEncoding(),
        oldBType.getElementType());

    auto newAType = RankedTensorType::get(
        oldAType.getShape(), oldAType.getElementType(), newAEncoding);
    auto newBType = RankedTensorType::get(
        oldBType.getShape(), oldBType.getElementType(), newBEncoding);

    a = rewriter.create<triton::gpu::ConvertLayoutOp>(a.getLoc(), newAType, a);
    b = rewriter.create<triton::gpu::ConvertLayoutOp>(b.getLoc(), newBType, b);
    auto newDot = rewriter.create<triton::DotOp>(
        dotOp.getLoc(), newRetType, a, b, newAcc, dotOp.getInputPrecision());

    rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(
        op, oldRetType, newDot.getResult());
    return success();
  }
};

class BlockedToMMAChainDot : public mlir::RewritePattern {
  int computeCapability;
  mutable int mmaV1Counter{}; // used to generate ID for MMAv1 encoding
  int dotCnt;
  int numStages;
  llvm::MapVector<triton::DotOp, llvm::SetVector<triton::DotOp>> chainDotMap;
  int disablePrefetch;

public:
  BlockedToMMAChainDot(
      mlir::MLIRContext *context, int computeCapability, int dotCnt,
      int numStages,
      llvm::MapVector<triton::DotOp, llvm::SetVector<triton::DotOp>>
          chainDotMap,
      bool disablePrefetch)
      : mlir::RewritePattern(triton::DotOp::getOperationName(), 2, context),
        computeCapability(computeCapability), dotCnt(dotCnt),
        numStages(numStages), chainDotMap(chainDotMap),
        disablePrefetch(disablePrefetch) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (computeCapability < 70)
      return failure();
    auto dotOp = cast<triton::DotOp>(op);
    // TODO: Check data-types and SM compatibility
    auto oldRetType = cast<RankedTensorType>(dotOp.getResult().getType());
    if (!oldRetType.getEncoding() ||
        isa<triton::gpu::MACAMmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    // for FMA, should retain the blocked layout.
    int versionMajor = computeCapabilityToMMAVersion(computeCapability);
    if (!supportMMA(dotOp, versionMajor, computeCapability % 10))
      return failure();

    // get MMA encoding for the given number of warps
    auto retShape = oldRetType.getShape();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);

    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());

    // enable opt maca layout or not
    auto aTensorTy = cast<RankedTensorType>(a.getType());
    auto bTensorTy = cast<RankedTensorType>(b.getType());
    int m = aTensorTy.getShape()[0];
    int k = aTensorTy.getShape()[1];
    int n = bTensorTy.getShape()[1];
    auto elementTy = aTensorTy.getElementType();
    bool enableTf32 = dotOp.getInputPrecision() == tt::InputPrecision::TF32;
    auto parentOp = dotOp->getParentOp();
    auto checkDotIn = [&](triton::DotOp &dotOp) {
      for (auto d : chainDotMap) {
        if (d.first == dotOp || d.second.count(dotOp) > 0)
          return true;
      }
      return false;
    };
    // numStages == 2 and more than two dot can get the right result?
    bool chainDot =
        (!chainDotMap.empty() && checkDotIn(dotOp) &&
         std::getenv("TRITON_DISABLE_MACA_CHAIN_DOT_OPT") == nullptr) &&
        (aTensorTy.getElementType().isF16() ||
         aTensorTy.getElementType().isBF16());
    // maybe add check dotcnt == 2 and numStages == 2, dotcnt == 5 and numStages
    // == 1?
    bool enableOptMMA = isa<scf::ForOp>(parentOp) &&
                        std::getenv("TRITON_DISABLE_MACA_OPT_MMA") == nullptr;

    triton::gpu::MACAMmaEncodingAttr mmaEnc;
    auto elemATy = oldAType.getElementType();
    auto elemBTy = oldBType.getElementType();
    bool enableALdsTrans, enableBLdsTrans;
    SmallVector<unsigned> aorder, border;
    aorder = triton::gpu::getOrder(oldAType.getEncoding());
    border = triton::gpu::getOrder(oldBType.getEncoding());
    if (versionMajor == 2) {
      auto elemsPerThread = getDefaultElemsPerThread(elementTy, enableTf32);
      auto warpsPerTile = warpsPerTileMACA(dotOp, retShape, numWarps);
      int versionMajor_ = versionMajor;
      int versionMinor_ = computeCapability % 10;
      bool isOpt = false;
      if (enableOptMMA) {
        Operation *aOp = a.getDefiningOp();
        Operation *bOp = b.getDefiningOp();
        auto aCvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(aOp);
        auto bCvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(bOp);
        if (aCvtOp && bCvtOp) {
          aorder = getOrder(aCvtOp);
          border = getOrder(bCvtOp);
          if (!(aorder.empty() || border.empty())) {
            llvm::SmallVector<int, 4> tile({m, n, k, numWarps});
            int version = -1;
            isOpt = updateLayout(
                elemsPerThread, warpsPerTile, tile, version, numWarps,
                enableTf32, aTensorTy.getElementType(), aorder, border,
                this->disablePrefetch, chainDot, false, computeCapability);
            if (isOpt) {
              mod->setAttr(
                  "use.opt.maca.mma",
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(mod.getContext(), 32), 1));
            }
          }
        }
      }
      // chaindot and match the table
      // TODO(MACA): only support A/C matrix chain dot now, B matrix chain dot
      // to be supported.
      enableALdsTrans = getIfLdsTrans(elemsPerThread, versionMajor_,
                                      versionMinor_, aorder, true, elemATy);
      enableBLdsTrans = getIfLdsTrans(elemsPerThread, versionMajor_,
                                      versionMinor_, border, false, elemBTy);
      auto elemStride = triton::gpu::getLdsTransVec(elemATy);
      SmallVector<unsigned> elementsStride = {1, 1};
      if (enableALdsTrans)
        elementsStride[0] = elemStride;
      if (enableBLdsTrans)
        elementsStride[1] = elemStride;
      if (enableOptMMA && chainDot && isOpt && versionMajor_ >= 2) {
        mmaEnc = triton::gpu::MACAMmaEncodingAttr::get(
            oldRetType.getContext(), versionMajor_, versionMinor_, warpsPerTile,
            elemsPerThread, 1, enableALdsTrans, enableBLdsTrans,
            elementsStride);
      } else {
        mmaEnc = triton::gpu::MACAMmaEncodingAttr::get(
            oldRetType.getContext(), versionMajor_, versionMinor_, warpsPerTile,
            elemsPerThread, 0, enableALdsTrans, enableBLdsTrans,
            elementsStride);
      }
    } else {
      llvm_unreachable("Mma layout only supports versionMajor in {1, 2}");
    }
    auto newRetType =
        RankedTensorType::get(retShape, oldRetType.getElementType(), mmaEnc);

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = rewriter.create<triton::gpu::ConvertLayoutOp>(
        oldAcc.getLoc(), newRetType, oldAcc);
    auto oldAOrder =
        cast<triton::gpu::BlockedEncodingAttr>(
            cast<triton::gpu::DotOperandEncodingAttr>(oldAType.getEncoding())
                .getParent())
            .getOrder();
    auto oldBOrder =
        cast<triton::gpu::BlockedEncodingAttr>(
            cast<triton::gpu::DotOperandEncodingAttr>(oldBType.getEncoding())
                .getParent())
            .getOrder();

    auto newAEncoding = triton::gpu::DotOperandEncodingAttr::get(
        oldAType.getContext(), 0, newRetType.getEncoding(),
        oldAType.getElementType());
    auto newBEncoding = triton::gpu::DotOperandEncodingAttr::get(
        oldBType.getContext(), 1, newRetType.getEncoding(),
        oldBType.getElementType());

    auto newAType = RankedTensorType::get(
        oldAType.getShape(), oldAType.getElementType(), newAEncoding);
    auto newBType = RankedTensorType::get(
        oldBType.getShape(), oldBType.getElementType(), newBEncoding);

    a = rewriter.create<triton::gpu::ConvertLayoutOp>(a.getLoc(), newAType, a);
    b = rewriter.create<triton::gpu::ConvertLayoutOp>(b.getLoc(), newBType, b);
    auto newDot = rewriter.create<triton::DotOp>(
        dotOp.getLoc(), newRetType, a, b, newAcc, dotOp.getInputPrecision());

    rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(
        op, oldRetType, newDot.getResult());
    return success();
  }
};

class BlockedToMMAMultiDot : public mlir::RewritePattern {
  int computeCapability;
  mutable int mmaV1Counter{}; // used to generate ID for MMAv1 encoding
  DenseMap<triton::DotOp, bool> multiDotMapFlag;
  llvm::MapVector<triton::DotOp, llvm::SmallVector<unsigned, 3>>
      multiDotElemsMap;
  llvm::MapVector<triton::DotOp, llvm::SmallVector<unsigned, 2>>
      multiDotWarpsMap;

public:
  BlockedToMMAMultiDot(
      mlir::MLIRContext *context, int computeCapability,
      DenseMap<triton::DotOp, bool> multiDotMapFlag,
      llvm::MapVector<triton::DotOp, llvm::SmallVector<unsigned, 3>>
          multiDotElemsMap,
      llvm::MapVector<triton::DotOp, llvm::SmallVector<unsigned, 2>>
          multiDotWarpsMap)
      : mlir::RewritePattern(triton::DotOp::getOperationName(), 2, context),
        computeCapability(computeCapability), multiDotMapFlag(multiDotMapFlag),
        multiDotElemsMap(multiDotElemsMap), multiDotWarpsMap(multiDotWarpsMap) {
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (computeCapability < 70)
      return failure();
    auto dotOp = cast<triton::DotOp>(op);
    // TODO: Check data-types and SM compatibility
    auto oldRetType = cast<RankedTensorType>(dotOp.getResult().getType());
    if (!oldRetType.getEncoding() ||
        isa<triton::gpu::MACAMmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    // for FMA, should retain the blocked layout.
    int versionMajor = computeCapabilityToMMAVersion(computeCapability);
    if (!supportMMA(dotOp, versionMajor, computeCapability % 10))
      return failure();

    // get MMA encoding for the given number of warps
    auto retShape = oldRetType.getShape();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);

    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());

    // enable opt maca layout or not
    auto aTensorTy = cast<RankedTensorType>(a.getType());
    auto bTensorTy = cast<RankedTensorType>(b.getType());
    int m = aTensorTy.getShape()[0];
    int k = aTensorTy.getShape()[1];
    int n = bTensorTy.getShape()[1];
    auto elementTy = aTensorTy.getElementType();
    bool enableTf32 = dotOp.getInputPrecision() == tt::InputPrecision::TF32;

    triton::gpu::MACAMmaEncodingAttr mmaEnc;
    if (versionMajor == 2) {
      auto elemsPerThread = getDefaultElemsPerThread(elementTy, enableTf32);
      auto warpsPerTile = warpsPerTileMACA(dotOp, retShape, numWarps);
      int versionMajor_ = versionMajor;
      int versionMinor_ = computeCapability % 10;
      if (multiDotMapFlag.lookup(dotOp)) {
        elemsPerThread = multiDotElemsMap.lookup(dotOp);
        warpsPerTile = multiDotWarpsMap.lookup(dotOp);
        mod->setAttr("use.opt.maca.mma",
                     mlir::IntegerAttr::get(
                         mlir::IntegerType::get(mod.getContext(), 32), 1));
      }
      auto elemATy = oldAType.getElementType();
      auto elemBTy = oldBType.getElementType();
      Operation *aOp = a.getDefiningOp();
      Operation *bOp = b.getDefiningOp();
      auto aCvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(aOp);
      auto bCvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(bOp);
      SmallVector<unsigned> aorder, border;
      if (aCvtOp && bCvtOp) {
        aorder = getOrder(aCvtOp);
        border = getOrder(bCvtOp);
      }
      bool enableALdsTrans = getIfLdsTrans(
          elemsPerThread, versionMajor_, versionMinor_, aorder, true, elemATy);
      bool enableBLdsTrans = getIfLdsTrans(
          elemsPerThread, versionMajor_, versionMinor_, border, false, elemBTy);
      auto elemStride = triton::gpu::getLdsTransVec(elemATy);
      SmallVector<unsigned> elementsStride = {1, 1};
      if (enableALdsTrans)
        elementsStride[0] = elemStride;
      if (enableBLdsTrans)
        elementsStride[1] = elemStride;
      mmaEnc = triton::gpu::MACAMmaEncodingAttr::get(
          oldRetType.getContext(), versionMajor_, versionMinor_, warpsPerTile,
          elemsPerThread, 0, enableALdsTrans, enableBLdsTrans, elementsStride);
    } else {
      llvm_unreachable("Mma layout only supports versionMajor in {1, 2}");
    }
    auto newRetType =
        RankedTensorType::get(retShape, oldRetType.getElementType(), mmaEnc);

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = rewriter.create<triton::gpu::ConvertLayoutOp>(
        oldAcc.getLoc(), newRetType, oldAcc);
    auto oldAOrder =
        cast<triton::gpu::BlockedEncodingAttr>(
            cast<triton::gpu::DotOperandEncodingAttr>(oldAType.getEncoding())
                .getParent())
            .getOrder();
    auto oldBOrder =
        cast<triton::gpu::BlockedEncodingAttr>(
            cast<triton::gpu::DotOperandEncodingAttr>(oldBType.getEncoding())
                .getParent())
            .getOrder();

    auto newAEncoding = triton::gpu::DotOperandEncodingAttr::get(
        oldAType.getContext(), 0, newRetType.getEncoding(),
        oldAType.getElementType());
    auto newBEncoding = triton::gpu::DotOperandEncodingAttr::get(
        oldBType.getContext(), 1, newRetType.getEncoding(),
        oldBType.getElementType());

    auto newAType = RankedTensorType::get(
        oldAType.getShape(), oldAType.getElementType(), newAEncoding);
    auto newBType = RankedTensorType::get(
        oldBType.getShape(), oldBType.getElementType(), newBEncoding);

    a = rewriter.create<triton::gpu::ConvertLayoutOp>(a.getLoc(), newAType, a);
    b = rewriter.create<triton::gpu::ConvertLayoutOp>(b.getLoc(), newBType, b);
    auto newDot = rewriter.create<triton::DotOp>(
        dotOp.getLoc(), newRetType, a, b, newAcc, dotOp.getInputPrecision());

    rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(
        op, oldRetType, newDot.getResult());
    return success();
  }
};
} // namespace

#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"

class TritonMETAXGPUAccelerateMatmulPass
    : public TritonMETAXGPUAccelerateMatmulBase<
          TritonMETAXGPUAccelerateMatmulPass> {
public:
  TritonMETAXGPUAccelerateMatmulPass() = default;
  TritonMETAXGPUAccelerateMatmulPass(int numStages, bool disablePrefetch,
                                     bool storeCoalesce,
                                     int computeCapability = 80) {
    this->computeCapability = computeCapability;
    this->numStages = numStages;
    this->disablePrefetch = disablePrefetch;
    this->storeCoalesce = storeCoalesce;
  }

  SetVector<Operation *> findIntersection(const SetVector<Operation *> &S1,
                                          const SetVector<Operation *> &S2) {
    SetVector<Operation *> resultSet;
    for (auto op : S1) {
      if (S2.count(op)) {
        resultSet.insert(op);
      }
    }
    return resultSet;
  }

  void
  collectSlice(llvm::MapVector<triton::DotOp, llvm::SetVector<triton::DotOp>>
                   &chainDotMap,
               triton::DotOp &dotOp, Value operand) {
    SetVector<Operation *> backwardSlice;
    // get all the backward op releated with dotop
    BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    getBackwardSlice(operand, &backwardSlice, opt);
    // get all the other dot releated with the dotop, maybe exist multi chain
    // dot
    llvm::SetVector<triton::DotOp> backwardSliceDot;
    for (auto op : backwardSlice) {
      if (auto dot = dyn_cast<triton::DotOp>(op)) {
        backwardSliceDot.insert(dot);
      }
    }
    // only one dot
    if (backwardSliceDot.empty())
      return;
    auto checkLegalFastRoute = [&](SetVector<Operation *> &slice) {
      if (slice.empty())
        return true;
      for (auto op : slice) {
        auto transOp = dyn_cast<triton::TransOp>(op);
        if (transOp)
          return false;
      }
      return true;
    };

    for (auto dot : backwardSliceDot) {
      ForwardSliceOptions options;
      options.filter = [&](Operation *op) {
        auto dot = dyn_cast<triton::DotOp>(op);
        if (dot && dot == dotOp) {
          return false;
        }
        return true;
      };
      SetVector<Operation *> forwardSlice;
      getForwardSlice(dot.getResult(), &forwardSlice, options);
      // from dotOp backward get the backwardSlice, form dot which included in
      // backwardSlice, forward get the forwardSlice, and interset get the path
      // form dot to dotOp
      auto intersection = findIntersection(forwardSlice, backwardSlice);
      // no transop on the chain path
      if (checkLegalFastRoute(intersection)) {
        chainDotMap[dotOp].insert(dot);
      }
    }
  }

  void
  collectLegalDot(llvm::MapVector<triton::DotOp, llvm::SetVector<triton::DotOp>>
                      &chainDotMap,
                  triton::DotOp &dotOp) {
    auto checkDotIn = [&](triton::DotOp &dotOp) {
      for (auto d : chainDotMap) {
        if (d.first == dotOp || d.second.count(dotOp) > 0)
          return true;
      }
      return false;
    };
    if (checkDotIn(dotOp))
      return;
    collectSlice(chainDotMap, dotOp, dotOp.getA()); // A
    // TODO: support B
    collectSlice(chainDotMap, dotOp, dotOp.getC()); // C
  }

  void getMajorMinorValue(ModuleOp &mod, triton::DotOp &dotOp, int &version,
                          bool disablePrefetch) {
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    auto oldRetType = cast<RankedTensorType>(dotOp.getResult().getType());
    if (!oldRetType.getEncoding() ||
        isa<triton::gpu::MACAMmaEncodingAttr>(oldRetType.getEncoding())) {
      return;
    }
    auto retShape = oldRetType.getShape();
    auto a = dotOp.getA();
    auto b = dotOp.getB();
    auto aTensorTy = cast<RankedTensorType>(a.getType());
    auto bTensorTy = cast<RankedTensorType>(b.getType());
    auto elementTy = aTensorTy.getElementType();
    bool enableTf32 = dotOp.getInputPrecision() == tt::InputPrecision::TF32;
    int m = aTensorTy.getShape()[0];
    int k = aTensorTy.getShape()[1];
    int n = bTensorTy.getShape()[1];
    Operation *aOp = a.getDefiningOp();
    Operation *bOp = b.getDefiningOp();
    auto aCvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(aOp);
    auto bCvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(bOp);
    auto elemsPerThread = getDefaultElemsPerThread(elementTy, enableTf32);
    auto warpsPerTile = warpsPerTileMACA(dotOp, retShape, numWarps);
    bool chainDot =
        (std::getenv("TRITON_DISABLE_MACA_CHAIN_DOT_OPT") == nullptr) &&
        (aTensorTy.getElementType().isF16() ||
         aTensorTy.getElementType().isBF16());
    if (aCvtOp && bCvtOp) {
      auto aorder = getOrder(aCvtOp);
      auto border = getOrder(bCvtOp);
      if (!(aorder.empty() || border.empty())) {
        llvm::SmallVector<int, 4> tile({m, n, k, numWarps});
        updateLayout(elemsPerThread, warpsPerTile, tile, version, numWarps,
                     enableTf32, aTensorTy.getElementType(), aorder, border,
                     disablePrefetch, chainDot);
      }
    }
  }

  bool checkIfMatchMma(llvm::SetVector<triton::DotOp> &component,
                       llvm::SmallVector<unsigned, 3> &elemsPerThread,
                       llvm::SmallVector<unsigned, 2> &warpsPerTile) {
    for (auto dotOp : component) {
      auto a = dotOp.getA();
      auto b = dotOp.getB();
      auto aTensorTy = cast<RankedTensorType>(a.getType());
      auto bTensorTy = cast<RankedTensorType>(b.getType());
      int m = aTensorTy.getShape()[0];
      int k = aTensorTy.getShape()[1];
      int n = bTensorTy.getShape()[1];
      // TODO: C600 maybe need modify the check
      // this is for multidot not chaindot, so the thereasPerWarp always be {16,
      // 4}
      SmallVector<unsigned> threadsPerWarp = {16, 4};
      bool match =
          (threadsPerWarp[0] * warpsPerTile[0] * elemsPerThread[0] <= m) &&
          (threadsPerWarp[1] * warpsPerTile[1] * elemsPerThread[1] <= n) &&
          (elemsPerThread[2] <= k);
      if (!match)
        return false;
    }
    return true;
  }

  void
  processMultiDot(ModuleOp &mod, bool disablePrefetch, int computeCapability,
                  llvm::MapVector<triton::DotOp, llvm::SetVector<triton::DotOp>>
                      &multiDotMap,
                  DenseMap<triton::DotOp, bool> &multiDotMapFlag,
                  llvm::MapVector<triton::DotOp, llvm::SmallVector<unsigned, 3>>
                      &multiDotElemsMap,
                  llvm::MapVector<triton::DotOp, llvm::SmallVector<unsigned, 2>>
                      &multiDotWarpsMap) {
    // By constructing a connected component, we can find the connected
    // component where each dot is located. Since the connected component is
    // fixed, we can ensure that every dot in the connected component has the
    // same mmaAttr. We will prioritize searching in the keys of the
    // multiDotMap. If the found mmaAttr satisfies all the dots, we will use
    // this mmaAttr. Otherwise, we will continue to search until we find it all.
    // Otherwise, we will use the default mmaAttr
    DenseMap<triton::DotOp, llvm::SetVector<triton::DotOp>> reverseMap;
    for (const auto &kv : multiDotMap) {
      auto op = kv.first;
      for (auto dep : kv.second) {
        reverseMap[dep].insert(op);
      }
    }
    DenseSet<triton::DotOp> visited;
    std::vector<llvm::SetVector<triton::DotOp>> components;
    // BFS find all the components
    for (const auto &kv : multiDotMap) {
      auto op = kv.first;
      if (!visited.count(op)) {
        llvm::SetVector<triton::DotOp> component;
        std::deque<triton::DotOp> queue;
        queue.push_back(op);
        visited.insert(op);
        while (!queue.empty()) {
          auto curOp = queue.front();
          queue.pop_front();
          component.insert(curOp);
          if (multiDotMap.count(curOp)) {
            for (auto dep : multiDotMap[curOp]) {
              if (!visited.count(dep)) {
                visited.insert(dep);
                queue.push_back(dep);
              }
            }
          }
          if (reverseMap.count(curOp)) {
            for (auto parent : reverseMap.lookup(curOp)) {
              if (!visited.count(parent)) {
                visited.insert(parent);
                queue.push_back(parent);
              }
            }
          }
        }
        components.push_back(component);
      }
    }

    for (auto component : components) {
      bool findMma = false;
      llvm::SmallVector<unsigned, 3> elemsPerThread;
      llvm::SmallVector<unsigned, 2> warpsPerTile;
      llvm::SmallVector<triton::DotOp> orderedOps;
      // search the mmaAttr in the keys of multiDotMap first
      for (const auto &kv : multiDotMap) {
        auto op = kv.first;
        if (component.count(op)) {
          orderedOps.push_back(op);
        }
      }
      for (auto op : orderedOps) {
        if (checkConfig(mod, op, elemsPerThread, warpsPerTile, disablePrefetch,
                        computeCapability, false)) {
          if (checkIfMatchMma(component, elemsPerThread, warpsPerTile)) {
            findMma = true;
            break;
          }
        }
      }
      // search the mmaAttr in other dots in multiDotMap
      if (!findMma) {
        for (auto op : component) {
          auto it = std::find(orderedOps.begin(), orderedOps.end(), op);
          if (it == orderedOps.end()) {
            if (checkConfig(mod, op, elemsPerThread, warpsPerTile,
                            disablePrefetch, computeCapability, false)) {
              if (checkIfMatchMma(component, elemsPerThread, warpsPerTile)) {
                findMma = true;
                break;
              }
            }
          }
        }
      }

      for (auto dot : component) {
        if (findMma) {
          multiDotMapFlag[dot] = true;
          multiDotElemsMap[dot] = elemsPerThread;
          multiDotWarpsMap[dot] = warpsPerTile;
        } else {
          multiDotMapFlag[dot] = false;
        }
      }
    }
  }

  void checkChainDotConfig(
      ModuleOp &mod,
      llvm::MapVector<triton::DotOp, llvm::SetVector<triton::DotOp>>
          &chainDotMap,
      bool disablePrefetch) {
    for (auto [dot, chain] : chainDotMap) {
      bool match = true;
      int version = 0;
      getMajorMinorValue(mod, dot, version, disablePrefetch);
      if (chain.empty() || version == 0) {
        match = false;
      }
      if (match) {
        for (auto childDot : chain) {
          int childVersion = 0;
          getMajorMinorValue(mod, childDot, childVersion, disablePrefetch);
          if (childVersion != version) {
            match = false;
            break;
          }
        }
      }
      if (!match) {
        chainDotMap.erase(dot);
      }
    }
  }

  /// conjunction: one op's result is directed to another op's operand.
  /// for example:
  ///        dot1
  ///          \
  ///           \
  ///           add
  ///             \
  ///              \
  ///              dot2
  /// we can say there is conjunction between dot1 and dot2.
  /// then we use backwardSlice from dot2 and forwardSlice from dot1
  /// to get intersection in order to collect any other operators
  /// in this conjunction. This function will collect and check all the
  /// conjunction between dotOp and all other dot operators in the ModuleOp.
  void collectConjunction(
      ModuleOp &mod, triton::DotOp &dotOp,
      llvm::MapVector<triton::DotOp, llvm::SetVector<triton::DotOp>>
          &chainDotMap,
      llvm::SetVector<triton::DotOp> &invalidDots, Value operand,
      bool disablePrefetch, int computeCapability) {
    SetVector<Operation *> backwardSlice;
    // get all the backward op releated with dotop
    BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    // get all the other dot releated with the dotop, maybe exist multi chain
    // dot
    llvm::SetVector<triton::DotOp> backwardSliceDot;
    opt.filter = [&](Operation *op) {
      if (auto dot = dyn_cast<triton::DotOp>(op)) {
        backwardSliceDot.insert(dot);
        return false;
      } else {
        return true;
      }
    };
    getBackwardSlice(operand, &backwardSlice, opt);
    // only one dot
    if (backwardSliceDot.empty())
      return;
    auto checkLegalFastRoute = [&](SetVector<Operation *> &slice) {
      if (slice.empty())
        return true;
      for (auto op : slice) {
        // TODO(MACA): support TransOp in conjunction.
        if (isa<triton::TransOp>(op))
          return false;
        ;
      }
      return true;
    };

    for (auto dot : backwardSliceDot) {
      ForwardSliceOptions options;
      options.filter = [&](Operation *op) { return !isa<triton::DotOp>(op); };
      SetVector<Operation *> forwardSlice;
      getForwardSlice(dot.getResult(), &forwardSlice, options);
      // from dotOp backward get the backwardSlice, form dot which included in
      // backwardSlice, forward get the forwardSlice, and interset get the path
      // form dot to dotOp
      auto intersection = findIntersection(forwardSlice, backwardSlice);
      bool isValid = checkIfMatch(mod, dotOp, dot, operand, disablePrefetch,
                                  computeCapability);
      // no transop on the chain path
      if (checkLegalFastRoute(intersection) && isValid &&
          !invalidDots.contains(dot)) {
        if (!chainDotMap[dotOp].contains(dot))
          chainDotMap[dotOp].insert(dot);
      } else {
        if (!invalidDots.contains(dot))
          chainDotMap[dotOp].insert(dot);
      }
    }
  }

  /// intersection: two ops' result will finally point to one common op.
  /// for example:
  ///        dot1    dot2
  ///          \     /
  ///           \   /
  ///            add
  /// we can see there is an intersection(add) between dot1 and dot2.
  /// we use forwardSlice from both dot1 and dot2 then find the intersection
  /// of 2 slices in order to collect the intersection operators.
  /// This function will collect and check intersections between dotOp and
  /// any other dot operators.
  void collectIntersection(
      ModuleOp &mod, triton::DotOp &dotOp,
      llvm::MapVector<triton::DotOp, llvm::SetVector<triton::DotOp>>
          &chainDotMap,
      llvm::SetVector<triton::DotOp> &invalidDots, Value operand,
      bool disablePrefetch, int computeCapability) {
    // check intersection
    assert(operand == dotOp.getResult());
    ForwardSliceOptions rootOpt;
    SetVector<Operation *> rootForwardSlice;
    rootOpt.filter = [&](Operation *op) {
      return !(isa<triton::DotOp>(op) || isa<scf::YieldOp>(op) ||
               isa<triton::ReturnOp>(op));
    };
    getForwardSlice(operand, &rootForwardSlice, rootOpt);
    mod.walk([&](triton::DotOp dot) {
      SetVector<Operation *> forwardSlice;
      if (dot != dotOp) {
        getForwardSlice(dot.getResult(), &forwardSlice, rootOpt);
      }
      auto intersection = findIntersection(rootForwardSlice, forwardSlice);
      if (!intersection.empty()) {
        auto checkLegalFastRoute = [&](SetVector<Operation *> &slice) {
          if (slice.empty())
            return true;
          for (auto op : slice) {
            if (isa<triton::TransOp>(op))
              return false;
            ;
          }
          return true;
        };
        BackwardSliceOptions backwardopt;
        backwardopt.omitBlockArguments = true;
        for (auto op : intersection) {
          // get all the other dot releated with the dotop, maybe exist multi
          // chain dot
          SetVector<Operation *> backwardSlice;
          backwardopt.filter = [&](Operation *op) {
            if (rootForwardSlice.count(op) || forwardSlice.count(op)) {
              return true;
            } else {
              return false;
            }
          };
          getBackwardSlice(op, &backwardSlice, backwardopt);
          if (!checkLegalFastRoute(backwardSlice))
            return;
        }
        bool isValid = checkIfMatch(mod, dotOp, dot, dotOp.getResult(),
                                    disablePrefetch, computeCapability);
        if (isValid && !invalidDots.contains(dot)) {
          // only intersection but not chain-dot
          if (!chainDotMap[dotOp].contains(dot))
            chainDotMap[dotOp].insert(dot);
        } else {
          if (!invalidDots.contains(dot))
            chainDotMap[dotOp].insert(dot);
        }
      }
    });
  }

  /// collect all edges from one specific dot to all other dot operators and
  /// check if there is a common layout for all edges.
  /// Edges have two types : conjunction and intersection.
  /// collect conjunction of specific dot's operands between all other dot
  /// operators, operands including A, B, C matrix. collect intersection of
  /// specific dot's result between all other dot operators
  void
  collectDotEdges(llvm::MapVector<triton::DotOp, llvm::SetVector<triton::DotOp>>
                      &chainDotMap,
                  triton::DotOp &dotOp, ModuleOp &mod, bool disablePrefetch,
                  int computeCapability) {
    llvm::SetVector<triton::DotOp> invalidDots;
    collectConjunction(mod, dotOp, chainDotMap, invalidDots, dotOp.getA(),
                       disablePrefetch, computeCapability); // A
    collectConjunction(mod, dotOp, chainDotMap, invalidDots, dotOp.getB(),
                       disablePrefetch, computeCapability); // B
    collectConjunction(mod, dotOp, chainDotMap, invalidDots, dotOp.getC(),
                       disablePrefetch, computeCapability); // C
    collectIntersection(mod, dotOp, chainDotMap, invalidDots, dotOp.getResult(),
                        disablePrefetch, computeCapability); // D
    if (chainDotMap[dotOp].empty())
      chainDotMap.erase(dotOp);
  }

  /// check if the tile of tensor match the optimization config.
  bool checkConfig(ModuleOp &mod, triton::DotOp &dot,
                   SmallVector<unsigned, 3> &elemsPerThread,
                   SmallVector<unsigned, 2> &warpsPerTile, bool disablePrefetch,
                   int computeCapability, bool isChainDot = true) {
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    auto oldRetType = cast<RankedTensorType>(dot.getResult().getType());
    if (!oldRetType.getEncoding() ||
        isa<triton::gpu::MACAMmaEncodingAttr>(oldRetType.getEncoding())) {
      return false;
    }
    auto retShape = oldRetType.getShape();
    auto a = dot.getA();
    auto b = dot.getB();
    auto aTensorTy = cast<RankedTensorType>(a.getType());
    auto bTensorTy = cast<RankedTensorType>(b.getType());
    auto elementTy = aTensorTy.getElementType();
    bool enableTf32 = dot.getInputPrecision() == tt::InputPrecision::TF32;
    int m = aTensorTy.getShape()[0];
    int k = aTensorTy.getShape()[1];
    int n = bTensorTy.getShape()[1];
    Operation *aOp = a.getDefiningOp();
    Operation *bOp = b.getDefiningOp();
    auto aCvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(aOp);
    auto bCvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(bOp);
    elemsPerThread = getDefaultElemsPerThread(elementTy, enableTf32);
    warpsPerTile = warpsPerTileMACA(dot, retShape, numWarps);
    bool chainDot =
        (std::getenv("TRITON_DISABLE_MACA_CHAIN_DOT_OPT") == nullptr) &&
        isChainDot &&
        (aTensorTy.getElementType().isF16() ||
         aTensorTy.getElementType().isBF16());
    if (aCvtOp && bCvtOp) {
      auto aorder = getOrder(aCvtOp);
      auto border = getOrder(bCvtOp);
      if (!(aorder.empty() || border.empty())) {
        llvm::SmallVector<int, 4> tile({m, n, k, numWarps});
        int version = -1;
        return updateLayout(elemsPerThread, warpsPerTile, tile, version,
                            numWarps, enableTf32, aTensorTy.getElementType(),
                            aorder, border, disablePrefetch, chainDot,
                            computeCapability = computeCapability);
      }
    }
    return false;
  }

  /// check if op_1 and op_2 have a common layout,
  /// For conjunction the function will check if the op_2's result has a common
  /// layout with op_1's operands. For intersection the function will check if
  /// the op_2's result has a common layout with op_1's result.
  bool checkIfMatch(ModuleOp &mod, triton::DotOp &op_1, triton::DotOp &op_2,
                    Value operand, bool disablePrefetch,
                    int computeCapability) {
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    SmallVector<unsigned, 3> elemsOp1, elemsOp2;
    SmallVector<unsigned, 2> warpsOp1, warpsOp2;
    // matrix D each thread has 4 element, M axis if rowmajor, N axis if
    // colmajor.
    int elementPerThread = 4;

    if (checkConfig(mod, op_1, elemsOp1, warpsOp1, disablePrefetch,
                    computeCapability) &&
        checkConfig(mod, op_2, elemsOp2, warpsOp2, disablePrefetch,
                    computeCapability)) {
      // op_2<->op_1 (intersection) || op_2->op_1, C (conjunction)
      if (operand == op_1.getResult() || operand == op_1.getC()) {
        for (int i = 0; i < elemsOp2.size(); i++) {
          if (elemsOp1[i] != elemsOp2[i])
            return false;
        }
        for (int i = 0; i < warpsOp2.size(); i++) {
          if (warpsOp1[i] != warpsOp2[i])
            return false;
        }
        return true;
        // op_2->op_1, A (conjunction)
      } else if (operand == op_1.getA()) {
        bool valid = false;
        // TODO(MACA): assume no trans between two dots, the dimension will
        // remain.
        valid = (warpsOp2[1] == 1) && (warpsOp2[0] == warpsOp1[0]);
        // M2 == M1; N2 * 4 == K1;
        valid = valid && (elemsOp2[0] == elemsOp1[0]) &&
                (elemsOp2[1] * elementPerThread == elemsOp2[2]);
        return valid;
      } else {
        // TODO(MACA): op_2->op_1, B to be supported
        return false;
      }
    } else {
      return false;
    }
  }

  void
  mergeChainDot(llvm::MapVector<triton::DotOp, llvm::SetVector<triton::DotOp>>
                    &chainDotMap) {
    // merge chain if one is inclued by another one
    std::vector<triton::DotOp> keysToRemove;
    for (auto d : chainDotMap) {
      auto key1 = d.first;
      auto values1 = d.second;
      bool isSub = false;
      for (auto dd : chainDotMap) {
        auto key2 = dd.first;
        auto values2 = dd.second;
        if (key1 == key2)
          continue;
        bool key1InValues2 = values2.contains(key1);
        if (!key1InValues2)
          continue;
        bool allValuesIncluded = true;
        for (auto op : values1) {
          if (!values2.contains(op)) {
            allValuesIncluded = false;
            break;
          }
        }
        if (allValuesIncluded) {
          isSub = true;
          break;
        }
      }
      if (isSub) {
        keysToRemove.push_back(key1);
      }
    }
    for (auto k : keysToRemove) {
      chainDotMap.erase(k);
    }
  }

  void filterCOnlyChain(
      llvm::MapVector<triton::DotOp, llvm::SetVector<triton::DotOp>>
          &chainDotMap,
      llvm::MapVector<triton::DotOp, llvm::SetVector<triton::DotOp>>
          &multiDotMapOnlyC) {
    std::vector<triton::DotOp> keysToRemove;
    for (auto kv : chainDotMap) {
      auto dot = kv.first;
      DenseSet<triton::DotOp> visited;
      std::deque<triton::DotOp> queue;
      queue.push_back(dot);
      visited.insert(dot);
      int dotCnt = 0;
      while (!queue.empty()) {
        dotCnt++;
        auto curOp = queue.front();
        queue.pop_front();
        // only C
        auto operand = curOp.getC();
        BackwardSliceOptions opt;
        opt.omitBlockArguments = true;
        SetVector<Operation *> backwardSlice;
        llvm::SetVector<triton::DotOp> backwardSliceDot;
        opt.filter = [&](Operation *op) {
          if (auto d = dyn_cast<triton::DotOp>(op)) {
            backwardSliceDot.insert(d);
            return false;
          } else {
            return true;
          }
        };
        getBackwardSlice(operand, &backwardSlice, opt);
        if (!backwardSliceDot.empty()) {
          assert(backwardSliceDot.size() == 1);
          auto head = backwardSliceDot.front();
          if (!visited.count(head)) {
            visited.insert(head);
            queue.push_back(head);
          }
        }
      }
      // all the dot can get access by C
      if (dotCnt == (kv.second.size() + 1)) {
        keysToRemove.push_back(dot);
      }
    }
    for (auto k : keysToRemove) {
      multiDotMapOnlyC[k] = chainDotMap.lookup(k);
      chainDotMap.erase(k);
    }
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    m->setAttr(
        "use.opt.maca.mma",
        mlir::IntegerAttr::get(mlir::IntegerType::get(m.getContext(), 32), 0));

    // modify by maca calculating dotOp count
    int dot_cnt = 0;
    // if there are multi dot in the graph, we will check chain dot and modify
    // mma as follow: a) collect chain dot to the chainDotMap
    //    -- collect chain dot in backward and forward slice, no transop on the
    //    chain path and all the dot
    //       on the chain path own the same tilesize
    // b) modify dot mma attr
    //    -- the dot on the chain path will set the colMajor== 1 (TODO(MACA):
    //    only A/C conjunction. B conjunction to be supported);
    //    -- the other dot will match the table and get the best tile and
    //    colMajor==0
    // chain dot releated dot op
    llvm::MapVector<triton::DotOp, llvm::SetVector<triton::DotOp>> chainDotMap;
    // there two types of chaindot in multidot:
    // 1) there are two dots connected by A or B in the chain
    //    2 dots:
    //    %1 = tt.dot           |    %1 = tt.dot
    //    %5 = tt.dot %1 %2 %3  |    %5 = tt.dot %2 %1 %3
    //    3 dots:
    //    %1 = tt.dot           |    %1 = tt.dot            |   %1 = tt.dot
    //    %5 = tt.dot %2 %3 %1  |    %5 = tt.dot %2 %3 %1   |   %5 = tt.dot
    //    %10 = tt.dot %5 %6 %7 |    %10 = tt.dot %6 %5 %7  |   %10 = tt.dot %1
    //    %7 %5
    //   etc...
    // 2) the two dots connected in the chain are only accessed by C
    //    2 dots:
    //    %1 = tt.dot
    //    %5 = tt.dot %2 %3 %1
    //    3 dots:
    //    %1 = tt.dot
    //    %5 = tt.dot %2 %3 %1
    //    %10 = tt.dot %6 %7 %5
    //   etc...
    // In the first type, there will be a CVT from mma to dotoperand, so a
    // shortcut is needed. But in the second type, there will be no CVT and no
    // shortcut is needed. So in the second type, try to configure mmaEnc in a
    // single dot manner as much as possible to ensure the optimal performance
    // of a single dot, but also ensure that the mmaEnc of the dots in the chain
    // is the same
    llvm::MapVector<triton::DotOp, llvm::SetVector<triton::DotOp>>
        multiDotMapOnlyC;
    if (std::getenv("TRITON_ENABLE_AUTOCHECK_CHAINDOT") != nullptr) {
      m.walk([&](triton::DotOp dotOp) {
        dot_cnt++;
        collectDotEdges(chainDotMap, dotOp, m, disablePrefetch,
                        computeCapability);
      });
      mergeChainDot(chainDotMap);
      filterCOnlyChain(chainDotMap, multiDotMapOnlyC);
    } else {
      m.walk([&](triton::DotOp dotOp) {
        dot_cnt++;
        collectLegalDot(chainDotMap, dotOp);
      });
      mergeChainDot(chainDotMap);
      filterCOnlyChain(chainDotMap, multiDotMapOnlyC);
      // all dots in chainDot should get the same versionMajor_ and
      // versionMinor_ if dots match different versionMajor_ and versionMinor_
      // may cause error or additional cvt
      checkChainDotConfig(m, chainDotMap, disablePrefetch);
    }

    bool chainDot =
        (!chainDotMap.empty() &&
         std::getenv("TRITON_DISABLE_MACA_CHAIN_DOT_OPT") == nullptr);
    bool multiDotOnlyC = !multiDotMapOnlyC.empty();

    mlir::RewritePatternSet patterns(context);
    if (chainDot || multiDotOnlyC) { // chaindot or multidot only connected by C
      if (chainDot) {
        patterns.add<::BlockedToMMAChainDot>(context, computeCapability,
                                             dot_cnt, numStages, chainDotMap,
                                             disablePrefetch);
      }
      if (multiDotOnlyC) {
        DenseMap<triton::DotOp, bool> multiDotMapFlag;
        llvm::MapVector<triton::DotOp, llvm::SmallVector<unsigned, 3>>
            multiDotElemsMap;
        llvm::MapVector<triton::DotOp, llvm::SmallVector<unsigned, 2>>
            multiDotWarpsMap;
        processMultiDot(m, disablePrefetch, computeCapability, multiDotMapOnlyC,
                        multiDotMapFlag, multiDotElemsMap, multiDotWarpsMap);
        patterns.add<::BlockedToMMAMultiDot>(context, computeCapability,
                                             multiDotMapFlag, multiDotElemsMap,
                                             multiDotWarpsMap);
      }
    } else { // others
      // TODO: if dot_cnt > 1 BlockedToMMA will fail, support dot_cnt > 1
      patterns.add<::BlockedToMMA>(context, computeCapability, dot_cnt,
                                   numStages, disablePrefetch, storeCoalesce);
    }
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonMETAXGPUAccelerateMatmulPass(
    int numStages, bool disablePrefetch, bool storeCoalesce,
    int computeCapability) {
  return std::make_unique<TritonMETAXGPUAccelerateMatmulPass>(
      numStages, disablePrefetch, storeCoalesce, computeCapability);
}
