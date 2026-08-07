#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#ifdef USE_MACA
#include "TritonMETAXGPUTransforms/MACACommon.h"
#include "TritonMETAXGPUTransforms/Passes.h"
#endif

using namespace mlir;
namespace ttg = triton::gpu;
namespace tt = triton;

#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"

struct TritonMETAXGPUSplitTensorMapPass
    : public TritonMETAXGPUSplitTensorMapBase<
          TritonMETAXGPUSplitTensorMapPass> {
  TritonMETAXGPUSplitTensorMapPass() = default;
  TritonMETAXGPUSplitTensorMapPass(int innerStageM, int innerStageN) {
    this->innerStageM = innerStageM;
    this->innerStageN = innerStageN;
  }

  static void setLoadAttrs(OpBuilder &builder, tt::LoadOp load,
                           StringRef operandName, int stage) {
    load->setAttr("metax.split_tensor_map.operand",
                  builder.getStringAttr(operandName));
    load->setAttr("metax.split_tensor_map.inner_stage",
                  builder.getI32IntegerAttr(stage));
  }

  static RankedTensorType getSubType(RankedTensorType type,
                                     ArrayRef<int64_t> shape) {
    return RankedTensorType::get(shape, type.getElementType(),
                                 type.getEncoding());
  }

  static Value extractTensor(OpBuilder &builder, Location loc, Value source,
                             RankedTensorType subType,
                             ArrayRef<int64_t> tileIdx) {
    auto sourceType = cast<RankedTensorType>(source.getType());
    auto [ctaIdx, elemIdx] =
        calExtractTensorIdx(sourceType, subType, tileIdx);
    return builder.create<ttg::ExtractTensorOp>(
        loc, subType, source, ctaIdx, elemIdx);
  }

  static tt::LoadOp createSubLoad(OpBuilder &builder, tt::LoadOp load,
                                  ArrayRef<int64_t> shape,
                                  ArrayRef<int64_t> tileIdx,
                                  StringRef operandName, int stage) {
    auto resultType = cast<RankedTensorType>(load.getType());
    auto ptrType = cast<RankedTensorType>(load.getPtr().getType());
    auto subResultType = getSubType(resultType, shape);
    auto subPtrType = getSubType(ptrType, shape);
    Value ptr =
        extractTensor(builder, load.getLoc(), load.getPtr(), subPtrType, tileIdx);

    Value mask;
    if (load.getMask()) {
      auto maskType = cast<RankedTensorType>(load.getMask().getType());
      mask = extractTensor(builder, load.getLoc(), load.getMask(),
                           getSubType(maskType, shape), tileIdx);
    }

    Value other;
    if (load.getOther()) {
      if (auto otherType =
              dyn_cast<RankedTensorType>(load.getOther().getType())) {
        other = extractTensor(builder, load.getLoc(), load.getOther(),
                              getSubType(otherType, shape), tileIdx);
      } else {
        other = load.getOther();
      }
    }

    auto subLoad = builder.create<tt::LoadOp>(
        load.getLoc(), subResultType, ptr, mask, other,
        load.getBoundaryCheckAttr(), load.getPaddingAttr(), load.getCache(),
        load.getEvict(), load.getIsVolatile(),
        load.getPipeline(),
        load.getContiguityInterConstGroup());
    setLoadAttrs(builder, subLoad, operandName, stage);
    return subLoad;
  }

  LogicalResult splitDot(tt::DotOp dot, int configuredStageM,
                         int configuredStageN) {
    auto loop = dot->getParentOfType<scf::ForOp>();
    if (!loop || loop->hasAttr("metax.split_tensor_map"))
      return failure();

    auto aConvert = dot.getA().getDefiningOp<ttg::ConvertLayoutOp>();
    auto bConvert = dot.getB().getDefiningOp<ttg::ConvertLayoutOp>();
    if (!aConvert || !bConvert)
      return failure();
    auto aLoad = aConvert.getSrc().getDefiningOp<tt::LoadOp>();
    auto bLoad = bConvert.getSrc().getDefiningOp<tt::LoadOp>();
    if (!aLoad || !bLoad || aLoad->getParentOfType<scf::ForOp>() != loop ||
        bLoad->getParentOfType<scf::ForOp>() != loop)
      return failure();

    auto aType = cast<RankedTensorType>(dot.getA().getType());
    auto bType = cast<RankedTensorType>(dot.getB().getType());
    auto cType = cast<RankedTensorType>(dot.getC().getType());
    if (aType.getRank() != 2 || bType.getRank() != 2 ||
        cType.getRank() != 2)
      return failure();

    int64_t m = aType.getShape()[0];
    int64_t k = aType.getShape()[1];
    int64_t n = bType.getShape()[1];
    if (bType.getShape()[0] != k || cType.getShape()[0] != m ||
        cType.getShape()[1] != n)
      return failure();

    int stageM = configuredStageM;
    int stageN = configuredStageN;
    if (stageM == 0 || stageN == 0) {
      auto module = dot->getParentOfType<ModuleOp>();
      SmallVector<int, 4> tile = {
          static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
          getNumWarps(module)};
      auto stages =
          matchMNStage(std::make_pair(tile, Layout::TN),
                       aType.getElementType(), /*enableTf32=*/false);
      if (stageM == 0)
        stageM = stages[0];
      if (stageN == 0)
        stageN = stages[1];
    }

    if (stageM <= 0 || stageN <= 0 || m % stageM != 0 ||
        n % stageN != 0)
      return failure();

    OpBuilder builder(dot);
    if (stageM == 1 && stageN == 1) {
      setLoadAttrs(builder, aLoad, "A", 0);
      setLoadAttrs(builder, bLoad, "B", 0);
      loop->setAttr("metax.split_tensor_map", builder.getUnitAttr());
      loop->setAttr("metax.split_tensor_map.stage_m",
                    builder.getI32IntegerAttr(stageM));
      loop->setAttr("metax.split_tensor_map.stage_n",
                    builder.getI32IntegerAttr(stageN));
      return success();
    }

    int64_t subM = m / stageM;
    int64_t subN = n / stageN;
    SmallVector<Value> aValues;
    SmallVector<Value> bValues;

    Attribute aDotEncoding = aType.getEncoding();
    Attribute bDotEncoding = bType.getEncoding();
    for (int i = 0; i < stageM; ++i) {
      auto load =
          createSubLoad(builder, aLoad, {subM, k}, {i, 0}, "A", i);
      auto subDotType =
          RankedTensorType::get({subM, k}, aType.getElementType(),
                                aDotEncoding);
      aValues.push_back(builder.create<ttg::ConvertLayoutOp>(
          dot.getLoc(), subDotType, load.getResult()));
    }
    for (int j = 0; j < stageN; ++j) {
      auto load =
          createSubLoad(builder, bLoad, {k, subN}, {0, j}, "B", j);
      auto subDotType =
          RankedTensorType::get({k, subN}, bType.getElementType(),
                                bDotEncoding);
      bValues.push_back(builder.create<ttg::ConvertLayoutOp>(
          dot.getLoc(), subDotType, load.getResult()));
    }

    Value assembled = dot.getC();
    auto subCType = RankedTensorType::get(
        {subM, subN}, cType.getElementType(), cType.getEncoding());
    for (int i = 0; i < stageM; ++i) {
      for (int j = 0; j < stageN; ++j) {
        SmallVector<int64_t> tileIdx = {i, j};
        Value subC =
            extractTensor(builder, dot.getLoc(), dot.getC(), subCType, tileIdx);
        Value subDot = builder.create<tt::DotOp>(
            dot.getLoc(), subCType, aValues[i], bValues[j], subC,
            dot.getInputPrecision());
        auto [ctaIdx, elemIdx] =
            calExtractTensorIdx(cType, subCType, tileIdx);
        assembled = builder.create<ttg::InsertTensorOp>(
            dot.getLoc(), cType, assembled, subDot, ctaIdx, elemIdx);
      }
    }

    dot.getResult().replaceAllUsesWith(assembled);
    dot.erase();
    if (aConvert->use_empty())
      aConvert.erase();
    if (bConvert->use_empty())
      bConvert.erase();
    if (aLoad->use_empty())
      aLoad.erase();
    if (bLoad->use_empty())
      bLoad.erase();
    loop->setAttr("metax.split_tensor_map", builder.getUnitAttr());
    loop->setAttr("metax.split_tensor_map.stage_m",
                  builder.getI32IntegerAttr(stageM));
    loop->setAttr("metax.split_tensor_map.stage_n",
                  builder.getI32IntegerAttr(stageN));
    return success();
  }

  void runOnOperation() override {
    SmallVector<tt::DotOp> dots;
    getOperation().walk([&](tt::DotOp dot) { dots.push_back(dot); });
    for (tt::DotOp dot : dots)
      (void)splitDot(dot, innerStageM, innerStageN);
  }
};

std::unique_ptr<Pass>
mlir::createTritonMETAXGPUSplitTensorMapPass(int innerStageM,
                                             int innerStageN) {
  return std::make_unique<TritonMETAXGPUSplitTensorMapPass>(innerStageM,
                                                            innerStageN);
}
