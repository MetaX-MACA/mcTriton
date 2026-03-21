#include "TritonMETAXGPUTransforms/Passes.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <memory>
using namespace mlir;
using namespace triton;
using namespace triton::gpu;
using triton::LoadOp;
using triton::gpu::ConvertLayoutOp;
#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"

class TritonMETAXGPUMergeConvertLayoutPass
    : public TritonMETAXGPUMergeConvertLayoutBase<
          TritonMETAXGPUMergeConvertLayoutPass> {
public:
  TritonMETAXGPUMergeConvertLayoutPass() = default;

  bool collectRecursion(Operation *op, SetVector<Operation *> &path) {
    if (!op)
      return false;
    // all the ops in the slice own BlockedEncodingAttr or SliceEncodingAttr
    // (result type)
    auto ty = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!ty)
      return false;
    auto enc = ty.getEncoding();
    if (!(isa<BlockedEncodingAttr>(enc) || isa<SliceEncodingAttr>(enc)))
      return false;
    if (auto sliceEnc = dyn_cast<SliceEncodingAttr>(enc)) {
      auto sliceParent = sliceEnc.getParent();
      if (!isa<BlockedEncodingAttr>(sliceParent))
        return false;
    }
    if (path.count(op))
      return true;
    path.insert(op);
    // MakeRangeOp or SplatOp or ConstantOp or LoadOp
    if (isa<triton::MakeRangeOp>(op) || isa<triton::SplatOp>(op) ||
        isa<arith::ConstantOp>(op) || isa<triton::LoadOp>(op)) {
      if (auto splatOp = dyn_cast<triton::SplatOp>(op)) {
        // need to ensure that SplatOp does not introduce any additional layout
        // changes, simply, check its operand is from arg or is a number
        auto src = splatOp.getSrc();
        Operation *defOp = src.getDefiningOp();
        // src comes from arg
        if (!defOp)
          return true;
        if (isa<triton::PointerType>(src.getType()) ||
            isa<mlir::IntegerType>(src.getType()) ||
            isa<mlir::FloatType>(src.getType())) {
          return true;
        }
        return false;
      }
      return true;
    }
    bool valid = true;
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp) {
        // operand come from arg
        valid = false;
        continue;
      }
      if (!collectRecursion(defOp, path)) {
        valid = false;
      }
    }
    if (!valid) {
      path.remove(op);
    }
    return valid;
  }

  void setEncoding(MLIRContext *context, SetVector<Operation *> &slice,
                   IRMapping &mapping, BlockedEncodingAttr &newEnc) {
    if (!newEnc || slice.empty())
      return;
    OpBuilder builder(context);
    OpBuilder::InsertionGuard g(builder);
    // all the ops own BlockedEncodingAttr except MakeRangeOp, which is
    // BlockedEncodingAttr or SliceEncodingAttr
    /// TODO: support more patterns
    for (auto op : slice) {
      auto loc = op->getLoc();
      if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
        builder.setInsertionPointAfter(op);
        auto valueAttr = constantOp.getValue();
        if (!valueAttr)
          return;
        auto denseAttr = dyn_cast<DenseIntElementsAttr>(valueAttr);
        if (!denseAttr)
          return;
        auto valueInt = denseAttr.getSplatValue<APInt>();
        auto ty = dyn_cast<RankedTensorType>(constantOp.getType());
        auto enc = ty.getEncoding();
        if (!isa<BlockedEncodingAttr>(enc))
          return;
        auto elemTy = ty.getElementType();
        auto newTy = RankedTensorType::get(ty.getShape(), elemTy, newEnc);
        IntegerAttr constantAttr = IntegerAttr::get(elemTy, valueInt);
        auto newConstant = builder.create<triton::SplatOp>(
            loc, newTy, builder.create<arith::ConstantOp>(loc, constantAttr));
        mapping.map(constantOp.getResult(), newConstant.getResult());
      } else if (auto makerangeOp = dyn_cast<triton::MakeRangeOp>(op)) {
        builder.setInsertionPointAfter(op);
        auto ty = dyn_cast<RankedTensorType>(makerangeOp.getType());
        auto enc = ty.getEncoding();
        auto elemTy = ty.getElementType();
        if (isa<SliceEncodingAttr>(enc)) {
          auto sliceEncoding = dyn_cast<SliceEncodingAttr>(enc);
          auto newSliceAttr =
              SliceEncodingAttr::get(context, sliceEncoding.getDim(), newEnc);
          auto newTy =
              RankedTensorType::get(ty.getShape(), elemTy, newSliceAttr);
          auto newMakerangeOp = builder.create<triton::MakeRangeOp>(
              loc, newTy, makerangeOp.getStart(), makerangeOp.getEnd());
          mapping.map(makerangeOp.getResult(), newMakerangeOp.getResult());
        } else if (isa<BlockedEncodingAttr>(enc)) {
          auto newTy = RankedTensorType::get(ty.getShape(), elemTy, newEnc);
          auto newMakerangeOp = builder.create<triton::MakeRangeOp>(
              loc, newTy, makerangeOp.getStart(), makerangeOp.getEnd());
          mapping.map(makerangeOp.getResult(), newMakerangeOp.getResult());
        } else {
          return;
        }
      } else if (auto splatOp = dyn_cast<triton::SplatOp>(op)) {
        builder.setInsertionPointAfter(op);
        auto ty = dyn_cast<RankedTensorType>(splatOp.getType());
        auto enc = ty.getEncoding();
        if (!isa<BlockedEncodingAttr>(enc))
          return;
        auto elemTy = ty.getElementType();
        auto newTy = RankedTensorType::get(ty.getShape(), elemTy, newEnc);
        auto src = mapping.lookupOrDefault(splatOp.getSrc());
        auto newSplatOp = builder.create<triton::SplatOp>(loc, newTy, src);
        mapping.map(splatOp.getResult(), newSplatOp.getResult());
      } else if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
        builder.setInsertionPointAfter(op);
        auto ty = dyn_cast<RankedTensorType>(loadOp.getType());
        auto elemTy = ty.getElementType();
        auto newTy = RankedTensorType::get(ty.getShape(), elemTy, newEnc);
        auto newCvtOp =
            builder.create<ConvertLayoutOp>(loc, newTy, loadOp.getResult());
        mapping.map(loadOp.getResult(), newCvtOp.getResult());
      } else if (auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
        builder.setInsertionPointAfter(op);
        auto src = mapping.lookupOrDefault(cvtOp.getSrc());
        mapping.map(cvtOp.getResult(), src);
        cvtOp.getResult().replaceAllUsesWith(src);
      } else {
        builder.setInsertionPointAfter(op);
        auto ty = dyn_cast<RankedTensorType>(op->getResult(0).getType());
        if (!ty)
          return;
        auto enc = ty.getEncoding();
        if (!isa<BlockedEncodingAttr>(enc))
          return;
        Operation *cloneOp = cloneWithInferType(builder, op, mapping);
        mapping.map(op->getResult(0), cloneOp->getResult(0));
      }
    }
  }

  bool findIntersection(SetVector<Operation *> &s1, SetVector<Operation *> &s2,
                        SetVector<Operation *> &resultSet) {
    SetVector<Operation *> bwdIntersection;
    for (auto op : s1) {
      if (s2.count(op)) {
        bwdIntersection.insert(op);
      }
    }
    // check own loadOp
    int loadCnt = 0;
    Operation *loadOp = nullptr;
    for (auto op : bwdIntersection) {
      if (isa<LoadOp>(op)) {
        ++loadCnt;
        loadOp = op;
      }
    }
    if (loadCnt != 1)
      return false;
    resultSet.insert(loadOp);
    return true;
  }

  void mergeConvertLayout(
      MLIRContext *context,
      llvm::MapVector<ConvertLayoutOp, SetVector<Operation *>> &cvtMap) {
    // Group together the cvts that have intersection lines with loadOp on the
    // intersection and own the same layout after cvt conversion. If the cvts in
    // a certain group have different layout after conversion, the entire group
    // will be dispersed because this situation may introduce other cvts.
    /// TODO: support multi cvts own different layout after conversion
    llvm::MapVector<uint8_t, SetVector<ConvertLayoutOp>> cvtGroup;
    // all the ops in the group(include cvt and load)
    llvm::MapVector<uint8_t, SetVector<Operation *>> cvtGroupOps;
    // group -> loadOp
    llvm::MapVector<uint8_t, Operation *> groupInsertPoint;
    // init group
    int8_t group = 0;
    auto checkOpIn = [&](ConvertLayoutOp &op) {
      for (auto d : cvtGroup) {
        if (d.second.count(op) > 0)
          return true;
      }
      return false;
    };
    for (auto &[cvt, deps] : cvtMap) {
      auto baseEnc = cvt.getResult().getType().getEncoding();
      if (!checkOpIn(cvt))
        group++;
      for (auto &[cvtKey, depsValue] : cvtMap) {
        if (cvt != cvtKey && !checkOpIn(cvtKey)) {
          auto enc = cvtKey.getResult().getType().getEncoding();
          SetVector<Operation *> loadSet;
          if (findIntersection(deps, depsValue, loadSet) && baseEnc == enc) {
            cvtGroup[group].insert(cvt);
            cvtGroup[group].insert(cvtKey);
            groupInsertPoint[group] = loadSet.front();
            cvtGroupOps[group].insert(deps.begin(), deps.end());
            cvtGroupOps[group].insert(depsValue.begin(), depsValue.end());
            cvtGroupOps[group].insert(cvt);
            cvtGroupOps[group].insert(cvtKey);
            cvtGroupOps[group].insert(loadSet.front());
          }
        }
      }
    }
    // need to ensure ops that getForward from loadOp to cvt should inclueded in
    // the cvtGroupOps, if the loadOp is used by other ops after the cvtOp, may
    // cuase error %1 = tt.load #blocked %5 = ... %1 %10 =
    // triton_gpu.convert_layout %5 #blocked -> #blocked1 %11 = tt.load %10
    // #blocked1
    // ...
    // %20 = triton_gpu.convert_layout %5 #blocked -> #blocked1
    // %21 = tt.load %20 #blocked1
    // ...
    // %30 = ... %5
    // in the graph above, %10 and %20 will be in the same group, but since %5
    // was used by %30, which located after %10 and %20, the ops need to change
    // the layout are collected getBackward from ConvertLayoutOp, so change the
    // layout may bring more additional cvts or error, so it is not supported
    // now
    /// TODO: support more complex situations
    for (auto &[g, s] : cvtGroup) {
      auto groupOps = cvtGroupOps[g];
      SetVector<Operation *> fwdIntersection;
      ForwardSliceOptions options;
      options.filter = [&](Operation *op) {
        if (isa<scf::ForOp>(op) || isa<scf::YieldOp>(op) ||
            isa<ConvertLayoutOp>(op) || isa<LoadOp>(op)) {
          return false;
        }
        return true;
      };
      getForwardSlice(groupInsertPoint[g], &fwdIntersection, options);
      for (auto op : fwdIntersection) {
        if (!groupOps.count(op)) {
          cvtGroup.erase(g);
          break;
        }
      }
    }
    // Now we directly insert the new cvt after the loadOp and change all the
    // ops collected by cvt into the same layout. Do not delete the old layout
    // ops for now, they may be used after the cvt. For example:
    //  %1 = tt.load #blocked
    //  %2 = arith.constant #blocked
    //  %5 = arith.addi %1 %2 #blocked
    //  ...
    //  %10 = triton_gpu.convert_layout %5 #blocked -> #blocked1
    //  %11 = tt.load %10 #blocked1
    //  ...
    //  %20 = triton_gpu.convert_layout %5 #blocked -> #blocked1
    //  %21 = tt.load %20 #blocked1
    //  ...
    //  %30 = ... %2 #blocked
    // the graph above will be modified to
    //  %1 = tt.load #blocked
    //  %2 = triton_gpu.convert_layout %1 #blocked -> #blocked1
    //  %3 = arith.constant #blocked1
    //  %4 = arith.constant #blocked
    //  %5 = arith.addi %2 %3 #blocked1
    //  ...
    //  %11 = tt.load %5 #blocked1
    //  ...
    //  %21 = tt.load %5 #blocked1
    //  ...
    //  %30 = ... %4 #blocked
    /// TODO: support more flexible insertion point to support more complex
    /// graph like follow
    //  %1 = tt.load #blocked
    //  %2 = arith.constant #blocked
    //  %5 = arith.addi %1 %2 #blocked
    //  ...
    //  %10 = triton_gpu.convert_layout %5 #blocked -> #blocked1
    //  %11 = tt.load %10 #blocked1
    //  ...
    //  %20 = triton_gpu.convert_layout %5 #blocked -> #blocked1
    //  %21 = tt.load %20 #blocked1
    //  ...
    //  %30 = triton_gpu.convert_layout %5 #blocked -> #blocked2
    //  %21 = tt.load %31 #blocked2
    for (auto &[g, s] : cvtGroup) {
      if (s.size() > 1) {
        auto cvt = s.front();
        auto newEnc = dyn_cast<BlockedEncodingAttr>(
            cvt.getResult().getType().getEncoding());
        IRMapping mapping;
        setEncoding(context, cvtGroupOps[g], mapping, newEnc);
      }
    }
  }

  void collectDepsRecursion(
      ConvertLayoutOp &cvtOp,
      llvm::MapVector<ConvertLayoutOp, SetVector<Operation *>> &cvtMap) {
    auto cvtOperandDefOp = cvtOp.getOperand().getDefiningOp();
    if (!cvtOperandDefOp)
      return;
    bool collected = collectRecursion(cvtOperandDefOp, cvtMap[cvtOp]);
    // not support chain cvt
    // %1 = triton_gpu.convert_layout ...
    // %5 = ... %1
    // %10 = triton_gpu.convert_layout %5
    // %1 may mark as legal cvt, but %10 marked as illegal cvt
    for (auto op : cvtMap[cvtOp]) {
      if (isa<ConvertLayoutOp>(op)) {
        collected = false;
        break;
      }
    }
    // only one loadOp
    /// TODO: support more pattern
    int loadCnt = 0;
    for (auto op : cvtMap[cvtOp]) {
      if (isa<LoadOp>(op)) {
        ++loadCnt;
      }
      if (loadCnt > 1)
        break;
    }
    if (loadCnt != 1)
      collected = false;
    if (!collected) {
      cvtMap.erase(cvtOp);
      return;
    }
    cvtMap[cvtOp] = mlir::topologicalSort(cvtMap[cvtOp]);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    /// in this pass, we assume that the changes of layout only come from loadOp
    /// TODO: suppport more ops may change the layout
    llvm::MapVector<ConvertLayoutOp, llvm::SetVector<Operation *>> cvtMap;
    m->walk(
        [&](ConvertLayoutOp cvtOp) { collectDepsRecursion(cvtOp, cvtMap); });
    mergeConvertLayout(context, cvtMap);
  }
};
std::unique_ptr<Pass> mlir::createTritonMETAXGPUMergeConvertLayoutPass() {
  return std::make_unique<TritonMETAXGPUMergeConvertLayoutPass>();
}