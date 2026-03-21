#ifndef TRITON_DIALECT_TRITONMETAXGPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONMETAXGPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<Pass> createTritonMETAXGPUAccelerateMatmulPass(
    int numStages = 2, bool disablePrefetch = false, bool storeCoalesce = false,
    int computeCapability = 80);

std::unique_ptr<Pass> createTritonMETAXGPUPipelineMACAPass(
    int numStages = 2, int pipelineLoadNum = -1, bool isFullStage = false,
    bool isSingleShm = false);

std::unique_ptr<Pass> createTritonMETAXGPUPrefetchMACAPass();

std::unique_ptr<Pass> createTritonMETAXGPUMoveDotOperandsOutLoopPass();

std::unique_ptr<Pass>
createTritonMETAXGPUPipelineAsyncTNPass(int numStages = 2);

std::unique_ptr<Pass>
createTritonMETAXGPUPipelineAsyncTTPass(int numStages = 2);

std::unique_ptr<Pass>
createTritonMETAXGPUPipelineAsyncBasePass(int numStages = 2,
                                          bool isFullStage = false);

std::unique_ptr<Pass> createTritonMETAXGPUMergeEqualSharedLayoutPass();

std::unique_ptr<Pass> createTritonMETAXGPUPipelineAsyncMultiDotMLAMixedPass(
    int numStages = 2, bool isFullStage = false, int pipelineLoadNum = -1,
    bool isSingleShm = true, bool onlyCpAsync = false);

std::unique_ptr<Pass> createTritonMETAXGPUChangeLayoutFromRepNToElemNPass();

std::unique_ptr<Pass> createTritonMETAXGPUChangeLayoutForConstancyLoadPass();

std::unique_ptr<Pass> createTritonMETAXGPUOptimizeCStorePass(int numStages = 2);

std::unique_ptr<Pass> createTritonMETAXGPUChangeLayoutForInt8Pass(
    int numStages = 2, std::string pipeline = std::string());

std::unique_ptr<Pass> createTritonMETAXGPUChangeConvertLayoutAttrPass(
    bool changeIntrinsic = false, bool intrinsicValue = false,
    bool changeWithPad = false, bool withPadValue = true);

std::unique_ptr<Pass> createTritonMETAXGPUMergeConvertLayoutPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "TritonMETAXGPUTransforms/Passes.h.inc"

} // namespace mlir
#endif
