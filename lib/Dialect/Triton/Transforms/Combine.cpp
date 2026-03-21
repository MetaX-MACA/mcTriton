#include <memory>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#define GEN_PASS_CLASSES
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace mlir::triton {
namespace {

bool isZero(Value val) {
  if (matchPattern(val, m_Zero()) || matchPattern(val, m_AnyZeroFloat()))
    return true;
  // broadcast(constant_0)
  if (auto bc = val.getDefiningOp<BroadcastOp>()) {
    if (matchPattern(bc.getSrc(), m_Zero()) ||
        matchPattern(bc.getSrc(), m_AnyZeroFloat()))
      return true;
  }
  return false;
}

bool isBroadcastConstantCombinable(Attribute value) {
  if (auto denseValue = dyn_cast<DenseElementsAttr>(value)) {
    return denseValue.isSplat();
  }
  return isa<FloatAttr, IntegerAttr>(value);
}

DenseElementsAttr getConstantValue(Builder &builder, Attribute value,
                                   Value bcast_res) {
  auto resType = cast<ShapedType>(bcast_res.getType());
  DenseElementsAttr res;
  if (auto denseValue = dyn_cast<DenseElementsAttr>(value)) {
    res =
        DenseElementsAttr::get(resType, denseValue.getSplatValue<Attribute>());
  } else {
    res = DenseElementsAttr::get(resType, value);
  }
  return res;
}

bool isAddPtrOffsetCombinable(Value first, Value second) {
  auto GetConstantIntValue = [](Value val) -> std::optional<llvm::APInt> {
    DenseElementsAttr constAttr;
    auto defOp = val.getDefiningOp();
    if (defOp) {
      if (auto splatOp = llvm::dyn_cast<SplatOp>(defOp))
        val = splatOp.getSrc();
      else if (matchPattern(defOp, m_Constant(&constAttr)) &&
               constAttr.isSplat()) {
        auto attr = constAttr.getSplatValue<Attribute>();
        // Check IntegerAttr
        if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
          return intAttr.getValue();
      }
    }

    // Check constant value.
    llvm::APInt intVal;
    if (matchPattern(val, m_ConstantInt(&intVal)))
      return intVal;

    return std::nullopt;
  };

  if (first.getType() == second.getType()) {
    // Whether bitwidth of element type is equal to pointer
    if (getElementTypeOrSelf(first.getType()).getIntOrFloatBitWidth() == 64)
      return true;

    // first + second does not overflow
    auto firstVal = GetConstantIntValue(first);
    auto secondVal = GetConstantIntValue(second);
    if (firstVal && secondVal) {
      bool overflow = false;
      auto resVal = firstVal->sadd_ov(*secondVal, overflow);
      return !overflow;
    }
  }
  return false;
}

#ifdef USE_MACA
bool isFmaCombinable(Operation *op) {
  auto op_mul = llvm::dyn_cast_or_null<mlir::arith::MulFOp>(op);
  if (!op_mul) {
    return false;
  }

  Type result_type = op_mul.getResult().getType();  // tensortype or scalar type
  if (auto ty = llvm::dyn_cast<RankedTensorType>(result_type)){
    result_type = ty.getElementType();
  }

  if (op_mul.getResult().hasOneUse() && result_type && (result_type.isF16() || result_type.isF32() || result_type.isF64())) {
    return true;
  }
  return false;
}
#endif

// TODO(csigg): remove after next LLVM integrate.
using FastMathFlags = arith::FastMathFlags;

#include "TritonCombine.inc"

// select(cond, load(ptrs, splat(cond), ???), other)
//   => load(ptrs, splat(cond), other)
class CombineSelectMaskedLoadPattern : public RewritePattern {
public:
  CombineSelectMaskedLoadPattern(MLIRContext *context)
      : RewritePattern(arith::SelectOp::getOperationName(), 3, context,
                       {LoadOp::getOperationName()}) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto selectOp = llvm::dyn_cast<arith::SelectOp>(op);
    if (!selectOp)
      return failure();

    Value trueValue = selectOp.getTrueValue();
    Value falseValue = selectOp.getFalseValue();
    Value condSelect = selectOp.getCondition();

    auto *loadOpCandidate = trueValue.getDefiningOp();
    auto loadOp = llvm::dyn_cast_or_null<LoadOp>(loadOpCandidate);
    if (!loadOp)
      return failure();

    Value mask = loadOp.getMask();
    if (!mask)
      return failure();

    auto *splatOpCandidate = mask.getDefiningOp();
    auto splatOp = llvm::dyn_cast_or_null<SplatOp>(splatOpCandidate);
    if (!splatOp)
      return failure();

    auto splatCond = splatOp.getSrc();
    if (splatCond != condSelect)
      return failure();

    rewriter.replaceOpWithNewOp<LoadOp>(
        op, loadOp.getPtr(), loadOp.getMask(), /*other=*/falseValue,
        loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
        loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
    return success();
  }
};

// sum(x[:, :, None] * y[None, :, :], 1)
// -> dot(x, y)
class CombineBroadcastMulReducePattern : public RewritePattern {
private:
  static bool isAddF32(const Operation *op) {
    if (auto addf = dyn_cast_or_null<arith::AddFOp>(op))
      return addf.getType().getIntOrFloatBitWidth() <= 32;
    return false;
  }

  static SmallVector<int> getEqualIndices(ArrayRef<int64_t> x,
                                          ArrayRef<int64_t> y) {
    SmallVector<int> res;
    for (int i = 0; i < x.size(); ++i)
      if (x[i] == y[i])
        res.push_back(i);
    return res;
  }

public:
  CombineBroadcastMulReducePattern(MLIRContext *context)
      : RewritePattern(ReduceOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const {
    auto reduceOp = llvm::dyn_cast<ReduceOp>(op);
    if (!reduceOp)
      return failure();
    // only support reduce with simple addition
    Region &combineOp = reduceOp.getCombineOp();
    bool isReduceAdd = combineOp.hasOneBlock() &&
                       combineOp.front().getOperations().size() == 2 &&
                       isAddF32(&*combineOp.front().getOperations().begin());
    if (!isReduceAdd)
      return failure();
    // operand of reduce has to be mul
    auto mulOp = llvm::dyn_cast_or_null<arith::MulFOp>(
        reduceOp.getOperand(0).getDefiningOp());
    if (!mulOp)
      return failure();
    // mul operand has to be broadcast
    auto broadcastLhsOp = llvm::dyn_cast_or_null<BroadcastOp>(
        mulOp.getOperand(0).getDefiningOp());
    if (!broadcastLhsOp)
      return failure();
    auto broadcastRhsOp = llvm::dyn_cast_or_null<BroadcastOp>(
        mulOp.getOperand(1).getDefiningOp());
    if (!broadcastRhsOp)
      return failure();
    // broadcast operand is expand dims
    auto expandLhsOp = llvm::dyn_cast_or_null<ExpandDimsOp>(
        broadcastLhsOp.getSrc().getDefiningOp());
    if (!expandLhsOp)
      return failure();
    auto expandRhsOp = llvm::dyn_cast_or_null<ExpandDimsOp>(
        broadcastRhsOp.getSrc().getDefiningOp());
    if (!expandRhsOp)
      return failure();
    // get not-broadcast dimensions
    int expandLhsAxis = expandLhsOp.getAxis();
    int expandRhsAxis = expandRhsOp.getAxis();
    if (expandLhsAxis != 2 || expandRhsAxis != 0)
      return failure();
    auto broadcastLhsShape =
        cast<ShapedType>(broadcastLhsOp.getType()).getShape();
    auto broadcastRhsShape =
        cast<ShapedType>(broadcastLhsOp.getType()).getShape();
    if (broadcastLhsShape[2] < 16 || broadcastRhsShape[0] < 16)
      return failure();
    Type newAccType = RankedTensorType::get(
        {broadcastLhsShape[0], broadcastRhsShape[2]},
        cast<ShapedType>(broadcastLhsOp.getSrc().getType()).getElementType());
    rewriter.setInsertionPoint(op);
    auto newAcc = rewriter.create<SplatOp>(
        op->getLoc(), newAccType,
        rewriter.create<arith::ConstantOp>(op->getLoc(),
                                           rewriter.getF32FloatAttr(0)));
    rewriter.replaceOpWithNewOp<DotOp>(op, expandLhsOp.getSrc(),
                                       expandRhsOp.getSrc(), newAcc,
                                       InputPrecision::TF32, 0);
    return success();
  }
};

#ifdef USE_MACA
// addf(mulf(a, b), c) => fma(a, b, c)
// addf(c, mulf(a, b)) => fma(a, b, c)
class CombineAddfmulfPattern : public mlir::RewritePattern {
public:
  CombineAddfmulfPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(mlir::arith::AddFOp::getOperationName(), 3,
                             context, {mlir::math::FmaOp::getOperationName()}) {}

  mlir::LogicalResult matchAndRewrite(Operation *op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto AddOp = llvm::dyn_cast<mlir::arith::AddFOp>(op);
    if (!AddOp)
      return mlir::failure();
    int idx = 0;
    for (auto operand : op->getOperands()) {
      auto definingOp = operand.getDefiningOp();
      if (isFmaCombinable(definingOp)) {
        auto op_mul = llvm::dyn_cast<mlir::arith::MulFOp>(definingOp);
        auto mul_inputs = op_mul->getOperands();
        auto FmaOp = rewriter.create<mlir::math::FmaOp>(op->getLoc(), op->getResultTypes()[0], 
                                                        mul_inputs[0], mul_inputs[1], 
                                                        op->getOperands()[idx ^ 1]);
        rewriter.replaceOpWithNewOp<mlir::math::FmaOp>(
          op, FmaOp.getA(), FmaOp.getB(), FmaOp.getC());
        return mlir::success();
      }
      idx += 1;
    }
    return mlir::failure();
  }
};

// subf(mulf(a, b), c) => fma(a, b, mulf(c, splat(constant(-1.0))))
// subf(c, mulf(a, b)) => fma(a, mulf(b, splat(constant(-1.0))), c)
class CombineSubfmulfPattern : public mlir::RewritePattern {
public:
  CombineSubfmulfPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(mlir::arith::SubFOp::getOperationName(), 3,
                             context, {mlir::math::FmaOp::getOperationName()}) {}

  mlir::LogicalResult matchAndRewrite(Operation *op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto sub_op = llvm::dyn_cast<mlir::arith::SubFOp>(op);
    if (!sub_op)
      return mlir::failure();

    if (auto definingOp = op->getOperands()[0].getDefiningOp()){
      if (isFmaCombinable(definingOp)) {
        auto op_mul = llvm::dyn_cast<mlir::arith::MulFOp>(definingOp);
        auto mul_inputs = op_mul->getOperands();
        // RankedTensorType ty = op->getOperands()[1].getType().template dyn_cast<RankedTensorType>();
        RankedTensorType ty = llvm::dyn_cast<RankedTensorType>(op->getOperands()[1].getType());
        if (!ty)
          return mlir::failure();
        Type ty_t = RankedTensorType::get(ty.getShape(), ty.getElementType());
        FloatAttr constantAttr = FloatAttr::get(ty.getElementType(), -1.0);
        auto ConstOp = rewriter.create<triton::SplatOp>(
            op->getLoc(), 
            ty_t,
            rewriter.create<arith::ConstantOp>(op->getLoc(), constantAttr));
        auto MulOp =  rewriter.create<mlir::arith::MulFOp>(op->getLoc(), ty_t,
            op->getOperands()[1], ConstOp);
        auto FmaOp = rewriter.create<mlir::math::FmaOp>(op->getLoc(), ty_t,
                                                        mul_inputs[0], mul_inputs[1], MulOp);
        rewriter.replaceOpWithNewOp<mlir::math::FmaOp>(op, FmaOp.getA(), FmaOp.getB(), FmaOp.getC());
        return mlir::success();
      }
    }

    if (auto definingOp = op->getOperands()[1].getDefiningOp()){
      if (isFmaCombinable(definingOp)) {
        auto op_mul = llvm::dyn_cast<mlir::arith::MulFOp>(definingOp);
        auto mul_inputs = op_mul->getOperands();
        // RankedTensorType ty = mul_inputs[1].getType().template dyn_cast<RankedTensorType>();
        RankedTensorType ty = llvm::dyn_cast<RankedTensorType>(mul_inputs[1].getType());
        if (!ty)
          return mlir::failure();
        Type ty_t = RankedTensorType::get(ty.getShape(), ty.getElementType());
        FloatAttr constantAttr = FloatAttr::get(ty.getElementType(), -1.0);
        auto ConstOp = rewriter.create<triton::SplatOp>(
            op->getLoc(), ty_t,
            rewriter.create<arith::ConstantOp>(op->getLoc(), constantAttr));
        auto MulOp =  rewriter.create<mlir::arith::MulFOp>(op->getLoc(), ty_t,
            mul_inputs[1], ConstOp);
        auto FmaOp = rewriter.create<mlir::math::FmaOp>(op->getLoc(), ty_t,
                                                        mul_inputs[0], MulOp, 
                                                        op->getOperands()[0]);
        rewriter.replaceOpWithNewOp<mlir::math::FmaOp>(
          op, FmaOp.getA(), FmaOp.getB(), FmaOp.getC());
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

bool getConstOpValue(arith::ConstantOp &constOp, int64_t &value) {
  auto valueAttr = constOp.getValue();
  if (!valueAttr)
    return false;
  auto denseAttr = dyn_cast<DenseIntElementsAttr>(valueAttr);
  if (!denseAttr)
    return false;
  auto valueInt = denseAttr.getSplatValue<APInt>();
  value = valueInt.getSExtValue();
  return true;
}

bool checkAddRem(arith::ConstantOp &constOp, Operation* op, int64_t &divValue) {
  auto remsiOp = dyn_cast<arith::RemSIOp>(op);
  if (!remsiOp)
    return false;
  int64_t addValue = 0;
  if (!getConstOpValue(constOp, addValue))
    return false;
  // RemSIOp
  auto remsiRhsOp = remsiOp.getRhs().getDefiningOp();
  if (!remsiRhsOp)
    return false;
  auto remsiRhsConstOp = dyn_cast<arith::ConstantOp>(remsiRhsOp);
  if (!remsiRhsConstOp)
    return false;
  int64_t remsiValue = 0;
  if (!getConstOpValue(remsiRhsConstOp, remsiValue))
    return false;
  if (addValue + remsiValue <= divValue && addValue + remsiValue >= 0 && remsiValue > 0 && divValue > 0)
    return true;
  return false;
}

// y = (const_a + x % const_b) / const_c
// if we can infer that const_a + const_b <= const_c, then we can simplify y to 0
class FoldRemAddDivPattern : public mlir::RewritePattern {
public:
  FoldRemAddDivPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(mlir::arith::DivSIOp::getOperationName(), 1,
                             context) {}

  mlir::LogicalResult matchAndRewrite(Operation *op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto divSIOp = dyn_cast<arith::DivSIOp>(op);
    if (!divSIOp)
      return failure();
    auto ty = dyn_cast<RankedTensorType>(divSIOp.getType());
    if (!ty)
      return failure();
    // TODO(): support more type, only int now
    if (!ty.getElementType().isInteger())
      return failure();
    // TODO(): support shape size > 1
    if (ty.getShape().size() != 1)
      return failure();
    auto divRhsDefOp = divSIOp.getRhs().getDefiningOp();
    if (!divRhsDefOp)
      return failure();
    auto divRhsConstOp = dyn_cast<arith::ConstantOp>(divRhsDefOp);
    if (!divRhsConstOp)
      return failure();
    int64_t divValue = 0;
    if (!getConstOpValue(divRhsConstOp, divValue))
      return failure();
    auto divLhsDefOp = divSIOp.getLhs().getDefiningOp();
    if (!divLhsDefOp)
      return failure();
    // AddIOp
    auto divLhsAddOp = dyn_cast<arith::AddIOp>(divLhsDefOp);
    if (!divLhsAddOp)
      return failure();
    auto addLhsDefOp = divLhsAddOp.getLhs().getDefiningOp();
    if (!addLhsDefOp)
      return failure();
    auto addRhsDefOp = divLhsAddOp.getRhs().getDefiningOp();
    if (!addRhsDefOp)
      return failure();
    // AddIOp lhs or rhs is const
    auto addLhsConstOp = dyn_cast<arith::ConstantOp>(addLhsDefOp);
    auto addRhsConstOp = dyn_cast<arith::ConstantOp>(addRhsDefOp);
    if (addLhsConstOp || addRhsConstOp) {
      if (addLhsConstOp) {
        // const_a + x % const_b
        if (checkAddRem(addLhsConstOp, addRhsDefOp, divValue)) {
          IntegerAttr constantAttr = IntegerAttr::get(ty.getElementType(), 0);
          auto newConstant = rewriter.create<triton::SplatOp>(divSIOp.getLoc(), ty,
            rewriter.create<arith::ConstantOp>(divSIOp.getLoc(), constantAttr));
          divSIOp.replaceAllUsesWith(newConstant.getResult());
          divSIOp.erase();
          return success();
        } else {
          return failure();
        }
      } else if (addRhsConstOp) {
        // x % const_b + const_a
        if (checkAddRem(addRhsConstOp, addLhsDefOp, divValue)) {
          IntegerAttr constantAttr = IntegerAttr::get(ty.getElementType(), 0);
          auto newConstant = rewriter.create<triton::SplatOp>(divSIOp.getLoc(), ty,
            rewriter.create<arith::ConstantOp>(divSIOp.getLoc(), constantAttr));
          divSIOp.replaceAllUsesWith(newConstant.getResult());
          divSIOp.erase();
          return success();
        } else {
          return failure();
        }
      }
    }
    return failure();
  }
};
#endif

class CombineOpsPass : public TritonCombineOpsBase<CombineOpsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp m = getOperation();

    // Dot Add %{
    patterns.add<CombineDotAddIPattern>(context);
    patterns.add<CombineDotAddFPattern>(context);
    patterns.add<CombineDotAddIRevPattern>(context);
    patterns.add<CombineDotAddFRevPattern>(context);
    // %}
    patterns.add<CombineSelectMaskedLoadPattern>(context);
#ifdef USE_MACA
    if(!std::getenv("TRITON_DISABLE_COMBINE_ADD_PTR_PASS")) {
      patterns.add<CombineAddPtrPattern>(context);
    }
    // fma
    if(!std::getenv("TRITON_DISABLE_OP_FUSION_PASS")){
      patterns.add<CombineAddfmulfPattern>(context);
      patterns.add<CombineSubfmulfPattern>(context);
    }
    if (!std::getenv("TRITON_DISABLE_FOLD_REM_ADD_DIV_PASS")) {
      patterns.add<FoldRemAddDivPattern>(context);
    }
#else
    patterns.add<CombineAddPtrPattern>(context);
#endif
    patterns.add<CombineBroadcastConstantPattern>(context);
    patterns.add<CombineBroadcastMulReducePattern>(context);

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createCombineOpsPass() {
  return std::make_unique<CombineOpsPass>();
}

} // namespace mlir::triton
