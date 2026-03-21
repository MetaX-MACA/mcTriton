#include "TritonMETAXGPUTransforms/MACACommon.h"
#include "TritonMETAXGPUTransforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include <memory>

using namespace mlir;
using triton::gpu::ConvertLayoutOp;
using triton::gpu::SharedEncodingAttr;

#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"

class TritonMETAXGPUChangeConvertLayoutAttr
    : public TritonMETAXGPUChangeConvertLayoutAttrBase<
          TritonMETAXGPUChangeConvertLayoutAttr> {
public:
  TritonMETAXGPUChangeConvertLayoutAttr() = default;
  TritonMETAXGPUChangeConvertLayoutAttr(bool changeIntrinsic = false,
                                        bool intrinsicValue = false,
                                        bool changeWithPad = false,
                                        bool withPadValue = true) {
    this->changeIntrinsic = changeIntrinsic;
    this->intrinsicValue = intrinsicValue;
    this->changeWithPad = changeWithPad;
    this->withPadValue = withPadValue;
  }
  void runOnOperation() override {
    if (!changeIntrinsic && !changeWithPad)
      return;
    bool changeIntrinsic = this->changeIntrinsic;
    bool intrinsicValue = this->intrinsicValue;
    bool changeWithPad = this->changeWithPad;
    bool withPadValue = this->withPadValue;

    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    m->walk([&](triton::gpu::ConvertLayoutOp cvtOp) {
      auto srcTy = cvtOp.getSrc().getType();
      auto dstTy = cvtOp.getType();
      auto srcEncoding = srcTy.getEncoding();
      auto dstEncoding = dstTy.getEncoding();
      if (mlir::isa<triton::gpu::SharedEncodingAttr>(srcEncoding) ||
          mlir::isa<triton::gpu::SharedEncodingAttr>(dstEncoding)) {
        // Conversions from/to shared memory do not need scratch memory.
        return;
      }
      if (changeIntrinsic) {
        cvtOp.setIntrinsic(intrinsicValue);
      }
      if (changeWithPad) {
        cvtOp.setWithPad(withPadValue);
      }
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonMETAXGPUChangeConvertLayoutAttrPass(
    bool changeIntrinsic, bool intrinsicValue, bool changeWithPad,
    bool withPadValue) {
  return std::make_unique<TritonMETAXGPUChangeConvertLayoutAttr>(
      changeIntrinsic, intrinsicValue, changeWithPad, withPadValue);
}