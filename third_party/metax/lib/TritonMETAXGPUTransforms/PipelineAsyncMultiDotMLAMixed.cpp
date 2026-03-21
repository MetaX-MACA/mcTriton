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

#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"

struct PipelineAsyncMultiDotMLAMixedPass
    : public TritonMETAXGPUPipelineAsyncMultiDotMLAMixedBase<
          PipelineAsyncMultiDotMLAMixedPass> {
  PipelineAsyncMultiDotMLAMixedPass() = default;
  PipelineAsyncMultiDotMLAMixedPass(int numStages, bool isFullStage,
                                    int pipelineLoadNum, bool isSingleShm,
                                    bool onlyCpAsync) {
    this->numStages = numStages;
    this->isFullStage = isFullStage;
    this->pipelineLoadNum = pipelineLoadNum;
    this->isSingleShm = isSingleShm;
    this->onlyCpAsync = onlyCpAsync;
  }

  void runOnOperation() override {
    // NOT IMPLEMENT
    return;
  }
};

std::unique_ptr<Pass>
mlir::createTritonMETAXGPUPipelineAsyncMultiDotMLAMixedPass(int numStages,
                                                            bool isFullStage,
                                                            int pipelineLoadNum,
                                                            bool isSingleShm,
                                                            bool onlyCpAsync) {
  return std::make_unique<PipelineAsyncMultiDotMLAMixedPass>(
      numStages, isFullStage, pipelineLoadNum, isSingleShm, onlyCpAsync);
}
